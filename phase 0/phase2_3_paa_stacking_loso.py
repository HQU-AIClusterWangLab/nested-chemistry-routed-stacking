# -*- coding: utf-8 -*-
"""
Phase 2.3a: PAA-SchNet + Stacking LOSO
Workspace: D:\lunwen\2.1sci\phase 2\

Uses trained PAA-SchNet per fold, then trains a Ridge stacking model
on (pred_energy, mean_force_norm) to correct energy predictions.
Compares with baseline SchNet and PAA-SchNet.
"""
import os, time, copy, random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────
DATASET_DIR   = r'D:\lunwen\2.1sci\phase 0\dataset\processed'
OUTPUT_DIR    = r'D:\lunwen\2.1sci\phase 2\loso_paa_stacking_output'
RANDOM_SEED   = 42
BATCH_SIZE    = 64
EPOCHS        = 150
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
FORCE_WEIGHT  = 0.5
CUTOFF        = 5.0
N_RBF         = 20
HIDDEN_DIM    = 128
N_INTERACTIONS = 3
EARLY_STOP    = 30

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f'DEVICE: {DEVICE}')
PHYS_DIM = 5
EDGE_PHYS_DIM = 5


# ═══════════════════════════════════════════════════════════
# 1. PAA-SchNet Model (same as Phase 2.2a)
# ═══════════════════════════════════════════════════════════

class RBFExpansion(nn.Module):
    def __init__(self, cutoff, n_rbf):
        super().__init__()
        self.cutoff = cutoff
        self.centers = nn.Parameter(torch.linspace(0.0, cutoff, n_rbf), requires_grad=False)
        gamma = 0.5 / ((self.centers[1] - self.centers[0]) ** 2 + 1e-8)
        self.gamma = gamma

    def forward(self, distances):
        d = distances.unsqueeze(-1)
        c = self.centers.unsqueeze(0)
        rbf = torch.exp(-self.gamma * (d - c) ** 2)
        cutoff_val = 0.5 * (1.0 + torch.cos(np.pi * distances / self.cutoff))
        return rbf * cutoff_val.unsqueeze(-1)


class EdgeBiasGate(nn.Module):
    def __init__(self, edge_phys_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_phys_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, edge_diffs):
        return self.net(edge_diffs).squeeze(-1)


class PAAInteractionBlock(nn.Module):
    def __init__(self, hidden_dim, n_rbf, edge_phys_dim):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_gate = EdgeBiasGate(edge_phys_dim)
        self.atom_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_src, edge_dst, rbf, edge_diffs, num_nodes):
        filters = self.filter_net(rbf)
        gate = self.edge_gate(edge_diffs)
        src_features = x[edge_src]
        messages = src_features * filters * gate.unsqueeze(-1)
        aggregated = torch.zeros(num_nodes, x.size(-1), device=x.device)
        aggregated = aggregated.index_add(0, edge_dst, messages)
        x = x + self.out_net(self.atom_net(x) + aggregated)
        return x


class PAASchNet(nn.Module):
    def __init__(self, n_atom_types=100, phys_dim=5, edge_phys_dim=5, hidden_dim=128,
                 n_rbf=20, n_interactions=3, cutoff=5.0):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.phys_proj = nn.Linear(phys_dim, hidden_dim)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.interactions = nn.ModuleList([
            PAAInteractionBlock(hidden_dim, n_rbf, edge_phys_dim) for _ in range(n_interactions)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z, pos, edge_src, edge_dst, phys_feats=None, batch=None):
        N = z.size(0)
        x = self.embedding(z)
        if phys_feats is not None:
            x = x + self.phys_proj(phys_feats)

        vec = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(vec, dim=-1)
        rbf = self.rbf(dist)

        if phys_feats is not None:
            phys_src = phys_feats[edge_src]
            phys_dst = phys_feats[edge_dst]
            edge_diffs = torch.abs(phys_src - phys_dst)
        else:
            edge_diffs = torch.zeros(dist.size(0), PHYS_DIM, device=z.device)

        for interaction in self.interactions:
            x = interaction(x, edge_src, edge_dst, rbf, edge_diffs, N)

        atom_energies = self.readout(x)
        if batch is not None:
            n_mols = batch.max().item() + 1
            energy = torch.zeros(n_mols, 1, device=z.device)
            energy = energy.index_add(0, batch, atom_energies)
        else:
            energy = atom_energies.sum(0, keepdim=True)
        return energy.squeeze(-1), x


# ═══════════════════════════════════════════════════════════
# 2. Data Loading (same as before)
# ═══════════════════════════════════════════════════════════

def load_all_pt_files(dataset_dir):
    samples = []
    for system_id in sorted(os.listdir(dataset_dir)):
        sys_dir = os.path.join(dataset_dir, system_id)
        if not os.path.isdir(sys_dir):
            continue
        for fname in os.listdir(sys_dir):
            if not fname.endswith('.pt'):
                continue
            fpath = os.path.join(sys_dir, fname)
            try:
                data = torch.load(fpath, weights_only=False)
            except:
                continue
            gid = getattr(data, 'group_id', 0) if hasattr(data, 'group_id') else 0
            samples.append({
                'z': data.atomic_numbers.long(),
                'pos': data.pos.float(),
                'y': data.y.float().item(),
                'forces': data.forces.float(),
                'phys_feats': data.x.float(),
                'system_id': system_id,
                'group_id': gid,
                'edge_index': data.edge_index.long(),
            })
    return samples


def collate_batch(batch_samples):
    z_list, pos_list, y_list, f_list, phys_list, edge_list = [], [], [], [], [], []
    batch_idx = []
    n_total = 0
    for i, s in enumerate(batch_samples):
        n = s['z'].size(0)
        z_list.append(s['z'])
        pos_list.append(s['pos'])
        y_list.append(s['y'])
        f_list.append(s['forces'])
        phys_list.append(s['phys_feats'])
        edge_list.append(s['edge_index'] + n_total)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        n_total += n
    return {
        'z': torch.cat(z_list, dim=0),
        'pos': torch.cat(pos_list, dim=0),
        'y': torch.tensor(y_list, dtype=torch.float32),
        'forces': torch.cat(f_list, dim=0),
        'phys_feats': torch.cat(phys_list, dim=0),
        'edge_index': torch.cat(edge_list, dim=1),
        'batch': torch.cat(batch_idx, dim=0),
    }


# ═══════════════════════════════════════════════════════════
# 3. Prediction helpers
# ═══════════════════════════════════════════════════════════

def compute_energy_and_forces(model, batch, device):
    z    = batch['z'].to(device)
    pos  = batch['pos'].to(device).requires_grad_(True)
    phys = batch['phys_feats'].to(device)
    ei   = batch['edge_index'].to(device)
    bch  = batch['batch'].to(device)
    energy, _ = model(z, pos, ei[0], ei[1], phys_feats=phys, batch=bch)
    forces = -torch.autograd.grad(energy.sum(), pos, create_graph=True, retain_graph=True)[0]
    return energy, forces


def get_predictions(model, loader, device):
    """
    Returns dict with per-sample:
        'energy_pred': (N_samples,) tensor
        'energy_true': (N_samples,) tensor
        'force_norm_mean': (N_samples,) tensor (mean |F| per atom)
    """
    model.eval()
    all_energy_pred = []
    all_energy_true = []
    all_force_norm_mean = []

    with torch.no_grad():
        for batch in loader:
            # Need to run with gradient for forces
            z, pos, phys, ei, bch = [batch[k].to(device) for k in ['z', 'pos', 'phys_feats', 'edge_index', 'batch']]
            pos = pos.clone().requires_grad_(True)
            energy_pred, _ = model(z, pos, ei[0], ei[1], phys_feats=phys, batch=bch)
            forces_pred = -torch.autograd.grad(energy_pred.sum(), pos, create_graph=False)[0]

            # Aggregate per sample
            n_mols = bch.max().item() + 1
            for m in range(n_mols):
                mask = (bch == m)
                n_atoms = mask.sum()
                atom_forces = forces_pred[mask]
                force_norm = torch.norm(atom_forces, dim=-1).mean() if n_atoms > 0 else torch.tensor(0.0)

                all_energy_pred.append(energy_pred[m].item())
                energy_true_m = batch['y'][mask.sum()].item()  # This might be wrong: we need the original per-sample y
                # Actually batch['y'] has per-sample energies, so we should index by m
                all_energy_true.append(batch['y'][m].item())
                all_force_norm_mean.append(force_norm.cpu().item())

    # The above per-molecule loop is not fully correct because batch['y'] is permuted.
    # We'll implement a robust version using the batch indices later.
    # For now, we'll use a simpler method.
    pass


# Better implementation of per-sample prediction collection
def predict_and_collect(model, loader, device):
    """
    Returns arrays:
        energy_pred: (N_samples,) float numpy
        energy_true: (N_samples,) float numpy
        force_norm_mean: (N_samples,) float numpy
    """
    model.eval()
    energy_pred_list = []
    energy_true_list = []
    force_norm_mean_list = []

    for batch in loader:
        z = batch['z'].to(device)
        pos = batch['pos'].to(device).requires_grad_(True)
        phys = batch['phys_feats'].to(device)
        ei = batch['edge_index'].to(device)
        bch = batch['batch'].to(device)
        energy_pred, _ = model(z, pos, ei[0], ei[1], phys_feats=phys, batch=bch)
        forces_pred = -torch.autograd.grad(energy_pred.sum(), pos, create_graph=False)[0]

        # Per-molecule aggregation
        n_mols = bch.max().item() + 1
        for m in range(n_mols):
            mask = (bch == m)
            # Get true energy for this molecule (batch['y'] is in same order as molecules in batch)
            true_energy = batch['y'][m].item()
            pred_energy = energy_pred[m].item()

            atom_forces = forces_pred[mask]
            mean_force_norm = torch.norm(atom_forces, dim=-1).mean().item()
            
            energy_true_list.append(true_energy)
            energy_pred_list.append(pred_energy)
            force_norm_mean_list.append(mean_force_norm)

    return (np.array(energy_pred_list), np.array(energy_true_list),
            np.array(force_norm_mean_list))


# ═══════════════════════════════════════════════════════════
# 4. Training PAA model and collecting predictions
# ═══════════════════════════════════════════════════════════

def train_one_fold(model, train_loader, val_loader, device, force_weight, epochs, early_stop, fold_name):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            energy_pred, forces_pred = compute_energy_and_forces(model, batch, device)
            energy_true = batch['y'].to(device)
            forces_true = batch['forces'].to(device)
            loss = F.l1_loss(energy_pred, energy_true) + force_weight * F.l1_loss(forces_pred, forces_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / n_batches

        # Validation
        model.eval()
        val_total_mae_e = 0.0
        val_total_mae_f = 0.0
        val_n_batches = 0
        for batch in val_loader:
            energy_pred, forces_pred = compute_energy_and_forces(model, batch, device)
            energy_true = batch['y'].to(device)
            forces_true = batch['forces'].to(device)
            val_total_mae_e += F.l1_loss(energy_pred, energy_true).item()
            val_total_mae_f += F.l1_loss(forces_pred, forces_true).item()
            val_n_batches += 1
        val_loss = val_total_mae_e / val_n_batches + force_weight * val_total_mae_f / val_n_batches
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        if patience_counter >= early_stop:
            print(f'    [{fold_name}] Early stop @ epoch {epoch}')
            break

    model.load_state_dict(best_state)
    return model


# ═══════════════════════════════════════════════════════════
# 5. Main LOSO + Stacking Loop
# ═══════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('Phase 2.3a: PAA-SchNet + Stacking LOSO')
    print('=' * 60)

    # Load data
    print('\n[1/5] Loading all .pt files ...')
    t0 = time.time()
    all_samples = load_all_pt_files(DATASET_DIR)
    systems = sorted(set(s['system_id'] for s in all_samples))
    print(f'  Loaded {len(all_samples)} samples, {len(systems)} systems: {systems}')
    for sys_id in systems:
        n = len([s for s in all_samples if s['system_id'] == sys_id])
        print(f'    {sys_id}: {n} samples')

    print(f'\n[2/5] Starting LOSO with Stacking ...')
    print('─' * 60)

    from torch.utils.data import DataLoader

    loso_results_paa = {}
    loso_results_stacking = {}

    for fold_idx, test_system in enumerate(systems):
        print(f'\n  Fold {fold_idx + 1}/{len(systems)}: Leave out [{test_system}]')

        # Split
        test_samples  = [s for s in all_samples if s['system_id'] == test_system]
        train_val_samples = [s for s in all_samples if s['system_id'] != test_system]

        random.seed(RANDOM_SEED)
        train_val_groups = defaultdict(list)
        for s in train_val_samples:
            train_val_groups[(s['system_id'], s['group_id'])].append(s)
        group_keys = sorted(train_val_groups.keys())
        random.shuffle(group_keys)
        n_train_groups = int(len(group_keys) * 0.90)
        train_groups_set = set(group_keys[:n_train_groups])

        train_samples = [s for s in train_val_samples if (s['system_id'], s['group_id']) in train_groups_set]
        val_samples   = [s for s in train_val_samples if (s['system_id'], s['group_id']) not in train_groups_set]

        print(f'    Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}')

        # Dataloaders
        train_loader = DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_batch, drop_last=True)
        val_loader   = DataLoader(val_samples,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        test_loader  = DataLoader(test_samples,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

        # Model
        model = PAASchNet(n_atom_types=100, phys_dim=PHYS_DIM, edge_phys_dim=EDGE_PHYS_DIM,
                          hidden_dim=HIDDEN_DIM, n_rbf=N_RBF, n_interactions=N_INTERACTIONS, cutoff=CUTOFF).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'    Params: {n_params:,}')

        # Train PAA
        t_start = time.time()
        model = train_one_fold(model, train_loader, val_loader, DEVICE, FORCE_WEIGHT,
                               EPOCHS, EARLY_STOP, test_system)
        train_time = time.time() - t_start

        # Collect predictions for all sets
        print('    Collecting predictions for stacking...')
        # Need to re-create loaders without drop_last for train to get all samples
        train_loader_full = DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        val_loader_full   = val_loader
        test_loader_full  = test_loader

        train_epred, train_etrue, train_fnorm = predict_and_collect(model, train_loader_full, DEVICE)
        val_epred, val_etrue, val_fnorm = predict_and_collect(model, val_loader_full, DEVICE)
        test_epred, test_etrue, test_fnorm = predict_and_collect(model, test_loader_full, DEVICE)

        # Stacking features: [pred_energy, mean_force_norm]
        X_train = np.stack([train_epred, train_fnorm], axis=1)
        y_train = train_etrue
        X_val   = np.stack([val_epred, val_fnorm], axis=1)
        y_val   = val_etrue
        X_test  = np.stack([test_epred, test_fnorm], axis=1)
        y_test  = test_etrue

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train stacking model (RidgeCV)
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(X_train_scaled, y_train)

        # Evaluate
        y_val_pred = ridge.predict(X_val_scaled)
        y_test_pred = ridge.predict(X_test_scaled)

        # PAA baseline MAE
        paae_mae = np.mean(np.abs(test_epred - y_test))
        # Stacking MAE
        stack_mae = np.mean(np.abs(y_test_pred - y_test))

        # Also compute PAA and stacking MAE_95
        def tail_mae(y_pred, y_true, percentile=5):
            abs_err = np.abs(y_pred - y_true)
            k = max(1, int(len(abs_err) * percentile / 100))
            top_idx = np.argpartition(abs_err, -k)[-k:]
            return np.mean(abs_err[top_idx])

        paae_mae_95 = tail_mae(test_epred, y_test)
        paae_maf = np.mean(np.abs(test_fnorm))  # Approx force MAE (mean norm difference)
        # For stacking we don't have force predictions, so keep PAA force
        stack_mae_95 = tail_mae(y_test_pred, y_test)

        loso_results_paa[test_system] = {
            'mae_e': paae_mae,
            'mae_95_e': paae_mae_95,
            'mae_f': paae_maf,
            'n_test': len(test_samples),
            'train_time': train_time}
        loso_results_stacking[test_system] = {
            'mae_e': stack_mae,
            'mae_95_e': stack_mae_95,
            'mae_f': paae_maf,  # same as PAA force
            'n_test': len(test_samples),
            'train_time': train_time}

        print(f'    PAA MAE(E)={paae_mae:.6f} eV, Stacking MAE(E)={stack_mae:.6f} eV, '
              f'Delta={paae_mae - stack_mae:.6f} eV')

        # Save model
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_paa_stacking_{test_system}.pt'))

    # ── Load baseline SchNet results ──
    baseline_csv = r'D:\lunwen\2.1sci\phase 1\loso_schnet_output\loso_results.csv'
    baseline = {}
    if os.path.exists(baseline_csv):
        with open(baseline_csv) as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    baseline[parts[0]] = {'mae_e': float(parts[1]), 'mae_95_e': float(parts[2]), 'mae_f': float(parts[3])}

    # ── Print comparison ──
    print('\n' + '=' * 78)
    print('SCHNET vs PAA vs PAA+STACKING (ENERGY)')
    print('=' * 78)
    sys_sorted = sorted(loso_results_paa.keys())
    print(f'{"Left-Out":<12} {"SchNet":>8} {"PAA":>8} {"Stack":>8} | {"S_95":>8} {"P_95":>8} {"St_95":>8}')
    print('-' * 78)
    for sys_id in sys_sorted:
        s_e  = f'{baseline[sys_id]["mae_e"]:.3f}' if sys_id in baseline else '-'
        p_e  = f'{loso_results_paa[sys_id]["mae_e"]:.3f}'
        st_e = f'{loso_results_stacking[sys_id]["mae_e"]:.3f}'
        s_95 = f'{baseline[sys_id]["mae_95_e"]:.3f}' if sys_id in baseline else '-'
        p_95 = f'{loso_results_paa[sys_id]["mae_95_e"]:.3f}'
        st_95= f'{loso_results_stacking[sys_id]["mae_95_e"]:.3f}'
        print(f'{sys_id:<12} {s_e:>8} {p_e:>8} {st_e:>8} | {s_95:>8} {p_95:>8} {st_95:>8}')
    print('-' * 78)

    # Summary stats
    all_paa_e = [loso_results_paa[s]['mae_e'] for s in sys_sorted]
    all_stack_e = [loso_results_stacking[s]['mae_e'] for s in sys_sorted]
    print(f'{"Mean":<12} {"":>8} {np.mean(all_paa_e):>8.3f} {np.mean(all_stack_e):>8.3f}')

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'loso_paa_stacking_results.csv')
    with open(csv_path, 'w') as f:
        f.write('System,PAAMAE_E,PAA95_E,StackMAE_E,Stack95_E\n')
        for sys_id in sys_sorted:
            p = loso_results_paa[sys_id]
            st = loso_results_stacking[sys_id]
            f.write(f'{sys_id},{p["mae_e"]:.6f},{p["mae_95_e"]:.6f},{st["mae_e"]:.6f},{st["mae_95_e"]:.6f}\n')
    print(f'  Results CSV saved to: {csv_path}')

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(sys_sorted))
    width = 0.25
    b_schnet = [baseline[s]['mae_e'] if s in baseline else 0 for s in sys_sorted]
    b_paa = [loso_results_paa[s]['mae_e'] for s in sys_sorted]
    b_stack = [loso_results_stacking[s]['mae_e'] for s in sys_sorted]
    axes[0].bar(x - width, b_schnet, width, label='SchNet')
    axes[0].bar(x, b_paa, width, label='PAA')
    axes[0].bar(x + width, b_stack, width, label='PAA+Stacking')
    axes[0].set_xticks(x); axes[0].set_xticklabels(sys_sorted, rotation=45)
    axes[0].set_title('MAE Energy')
    axes[0].legend()

    b_s95 = [baseline[s]['mae_95_e'] if s in baseline else 0 for s in sys_sorted]
    b_p95 = [loso_results_paa[s]['mae_95_e'] for s in sys_sorted]
    b_st95= [loso_results_stacking[s]['mae_95_e'] for s in sys_sorted]
    axes[1].bar(x - width, b_s95, width, label='SchNet')
    axes[1].bar(x, b_p95, width, label='PAA')
    axes[1].bar(x + width, b_st95, width, label='PAA+Stacking')
    axes[1].set_xticks(x); axes[1].set_xticklabels(sys_sorted, rotation=45)
    axes[1].set_title('MAE_95 Energy')
    axes[1].legend()
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, 'loso_stacking_comparison.png')
    plt.savefig(bar_path, dpi=150)
    print(f'  Bar chart saved to: {bar_path}')

    print('\n' + '=' * 60)
    print('Phase 2.3a Complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
