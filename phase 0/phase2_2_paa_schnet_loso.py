# -*- coding: utf-8 -*-
"""
Phase 2.2a: PAA-SchNet LOSO
Workspace: D:\lunwen\2.1sci\phase 2\

Physically-Augmented Attention (PAA) via electronegativity difference as edge bias.
Contrasts with baseline SchNet and SchNet+Phys (Phase 2.1).
"""
import os, time, copy, random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────
DATASET_DIR   = r'D:\lunwen\2.1sci\phase 0\dataset\processed'
OUTPUT_DIR    = r'D:\lunwen\2.1sci\phase 2\loso_paa_schnet_output'
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
PHYS_DIM = 5  # node physical features
EDGE_PHYS_DIM = 5  # differences for edges


# ═══════════════════════════════════════════════════════════
# 1. PAA-SchNet Model
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
    """
    Compute scalar gate g_ij from physical differences on edge.
    Input: (E, EDGE_PHYS_DIM) → MLP → scalar → sigmoid.
    """
    def __init__(self, edge_phys_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_phys_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()  # output in (0,1)
        )

    def forward(self, edge_diffs):
        return self.net(edge_diffs).squeeze(-1)  # (E,)


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
        """
        x: (N, D)
        rbf: (E, n_rbf)
        edge_diffs: (E, EDGE_PHYS_DIM)
        """
        filters = self.filter_net(rbf)  # (E, D)
        gate = self.edge_gate(edge_diffs)  # (E,)

        src_features = x[edge_src]  # (E, D)
        # Apply gate as scalar multiplication on messages
        messages = src_features * filters * gate.unsqueeze(-1)  # (E, D)

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

        # Compute edge physical differences
        if phys_feats is not None:
            phys_src = phys_feats[edge_src]
            phys_dst = phys_feats[edge_dst]
            edge_diffs = torch.abs(phys_src - phys_dst)  # (E, PHYS_DIM)
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
# 2. Data Loading (same as Phase 2.1)
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
# 3. Training & Evaluation
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


def evaluate(model, loader, device, force_weight, return_preds=False):
    model.eval()
    total_mae_e = 0.0
    total_mae_f = 0.0
    total_loss  = 0.0
    n_batches   = 0
    all_y_true  = []
    all_y_pred  = []
    for batch in loader:
        energy_pred, forces_pred = compute_energy_and_forces(model, batch, device)
        energy_true = batch['y'].to(device)
        forces_true = batch['forces'].to(device)
        loss_e = F.l1_loss(energy_pred, energy_true)
        loss_f = F.l1_loss(forces_pred, forces_true)
        loss   = loss_e + force_weight * loss_f
        total_mae_e += loss_e.item()
        total_mae_f += loss_f.item()
        total_loss  += loss.item()
        n_batches   += 1
        if return_preds:
            all_y_true.append(energy_true.cpu())
            all_y_pred.append(energy_pred.detach().cpu())
    result = (total_mae_e / n_batches, total_mae_f / n_batches, total_loss / n_batches)
    if return_preds:
        result += (torch.cat(all_y_true), torch.cat(all_y_pred))
    return result


def train_epoch(model, loader, optimizer, device, force_weight):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for batch in loader:
        optimizer.zero_grad()
        energy_pred, forces_pred = compute_energy_and_forces(model, batch, device)
        energy_true = batch['y'].to(device)
        forces_true = batch['forces'].to(device)
        loss = F.l1_loss(energy_pred, energy_true) + force_weight * F.l1_loss(forces_pred, forces_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches


def train_one_fold(model, train_loader, val_loader, device, force_weight, epochs, early_stop, fold_name):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, force_weight)
        val_mae_e, val_mae_f, val_loss = evaluate(model, val_loader, device, force_weight)
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
    return model, best_val_loss


# ═══════════════════════════════════════════════════════════
# 4. Main LOSO Loop
# ═══════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('Phase 2.2a: PAA-SchNet LOSO')
    print('=' * 60)

    print('\n[1/4] Loading all .pt files ...')
    t0 = time.time()
    all_samples = load_all_pt_files(DATASET_DIR)
    systems = sorted(set(s['system_id'] for s in all_samples))
    print(f'  Loaded {len(all_samples)} samples, {len(systems)} systems: {systems}')
    for sys_id in systems:
        print(f'    {sys_id}: {len([s for s in all_samples if s["system_id"]==sys_id])} samples')

    print(f'\n[2/4] Starting LOSO 7-fold CV (PAA-SchNet) ...')
    print('─' * 60)

    from torch.utils.data import DataLoader

    loso_results = {}

    for fold_idx, test_system in enumerate(systems):
        print(f'\n  Fold {fold_idx + 1}/{len(systems)}: Leave out [{test_system}]')

        test_samples  = [s for s in all_samples if s['system_id'] == test_system]
        train_val_samples = [s for s in all_samples if s['system_id'] != test_system]

        random.seed(RANDOM_SEED)
        train_val_groups = defaultdict(list)
        for s in train_val_samples:
            train_val_groups[(s['system_id'], s['group_id'])].append(s)
        group_keys = sorted(train_val_groups.keys())
        random.shuffle(group_keys)
        n_train = int(len(group_keys) * 0.90)
        train_groups_set = set(group_keys[:n_train])

        train_samples = [s for s in train_val_samples if (s['system_id'], s['group_id']) in train_groups_set]
        val_samples   = [s for s in train_val_samples if (s['system_id'], s['group_id']) not in train_groups_set]

        print(f'    Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}')

        train_loader = DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_batch, drop_last=True)
        val_loader   = DataLoader(val_samples,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        test_loader  = DataLoader(test_samples,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

        model = PAASchNet(n_atom_types=100, phys_dim=PHYS_DIM, edge_phys_dim=EDGE_PHYS_DIM,
                          hidden_dim=HIDDEN_DIM, n_rbf=N_RBF, n_interactions=N_INTERACTIONS, cutoff=CUTOFF).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'    Params: {n_params:,}')

        t_start = time.time()
        model, best_val = train_one_fold(model, train_loader, val_loader, DEVICE, FORCE_WEIGHT,
                                         EPOCHS, EARLY_STOP, test_system)
        train_time = time.time() - t_start

        test_mae_e, test_mae_f, test_loss, y_true, y_pred = evaluate(
            model, test_loader, DEVICE, FORCE_WEIGHT, return_preds=True
        )
        abs_errors = torch.abs(y_true - y_pred)
        sorted_err, _ = torch.sort(abs_errors, descending=True)
        tail_k = max(1, int(len(sorted_err) * 0.05))
        mae_95 = sorted_err[:tail_k].mean().item()

        loso_results[test_system] = {
            'mae_e': test_mae_e, 'mae_f': test_mae_f, 'loss': test_loss,
            'mae_95_e': mae_95, 'n_test': len(test_samples), 'train_time': train_time,
        }

        print(f'    Result: MAE(E)={test_mae_e:.6f} eV, MAE(F)={test_mae_f:.6f} eV/A, '
              f'MAE_95(E)={mae_95:.6f} eV, Time={train_time:.1f}s')

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_paa_loso_{test_system}.pt'))

    # ── Load baseline results for comparison ──
    baseline_schnet_csv = r'D:\lunwen\2.1sci\phase 1\loso_schnet_output\loso_results.csv'
    baseline_phys_csv   = r'D:\lunwen\2.1sci\phase 2\loso_schnet_phys_output\loso_schnet_phys_results.csv'

    def load_csv(csv_path):
        res = {}
        if not os.path.exists(csv_path):
            return res
        with open(csv_path) as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    res[parts[0]] = {'mae_e': float(parts[1]), 'mae_95_e': float(parts[2]), 'mae_f': float(parts[3])}
        return res

    schnet_res = load_csv(baseline_schnet_csv)
    phys_res   = load_csv(baseline_phys_csv)

    # ── Print three-way comparison ──
    print('\n' + '=' * 90)
    print('SCHNET vs SCHNET+PHYS vs PAA-SCHNet LOSO')
    print('=' * 90)
    sys_names_sorted = sorted(loso_results.keys())

    # Headers
    print(f'{"Left-Out":<12} {"SchNet":>8} {"+Phys":>8} {"PAA":>8} | {"S_95":>8} {"P_95":>8} {"PAA_95":>8} | {"S_F":>8} {"P_F":>8} {"PAA_F":>8}')
    print('-' * 90)

    for sys_id in sys_names_sorted:
        r = loso_results[sys_id]
        s_e  = f'{schnet_res[sys_id]["mae_e"]:.3f}' if sys_id in schnet_res else '-'
        p_e  = f'{phys_res[sys_id]["mae_e"]:.3f}'   if sys_id in phys_res   else '-'
        paa_e = f'{r["mae_e"]:.3f}'

        s_95  = f'{schnet_res[sys_id]["mae_95_e"]:.3f}' if sys_id in schnet_res else '-'
        p_95  = f'{phys_res[sys_id]["mae_95_e"]:.3f}'   if sys_id in phys_res   else '-'
        paa_95 = f'{r["mae_95_e"]:.3f}'

        s_f   = f'{schnet_res[sys_id]["mae_f"]:.3f}' if sys_id in schnet_res else '-'
        p_f   = f'{phys_res[sys_id]["mae_f"]:.3f}'   if sys_id in phys_res   else '-'
        paa_f = f'{r["mae_f"]:.3f}'

        print(f'{sys_id:<12} {s_e:>8} {p_e:>8} {paa_e:>8} | {s_95:>8} {p_95:>8} {paa_95:>8} | {s_f:>8} {p_f:>8} {paa_f:>8}')

    print('-' * 90)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'loso_paa_schnet_results.csv')
    with open(csv_path, 'w') as f:
        f.write('Left_Out_System,MAE_E_eV,MAE_95_E_eV,MAE_F_eV_A,N_Test,Time_s\n')
        for sys_id in sys_names_sorted:
            r = loso_results[sys_id]
            f.write(f'{sys_id},{r["mae_e"]:.6f},{r["mae_95_e"]:.6f},{r["mae_f"]:.6f},{r["n_test"]},{r["train_time"]:.1f}\n')
    print(f'  Results CSV saved to: {csv_path}')

    # ── Bar chart (three-way) ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(sys_names_sorted))
    width = 0.25

    for idx, metric in enumerate(['mae_e', 'mae_95_e', 'mae_f']):
        ax = axes[idx]
        b1 = [schnet_res[s][metric] if s in schnet_res else 0 for s in sys_names_sorted]
        b2 = [phys_res[s][metric]   if s in phys_res   else 0 for s in sys_names_sorted]
        b3 = [loso_results[s][metric] for s in sys_names_sorted]
        ax.bar(x - width, b1, width, label='SchNet', color='#1f77b4')
        ax.bar(x,       b2, width, label='SchNet+Phys', color='#ff7f0e')
        ax.bar(x + width, b3, width, label='PAA-SchNet', color='#2ca02c')
        ax.set_xticks(x)
        ax.set_xticklabels(sys_names_sorted, rotation=45)
        title = {'mae_e': 'MAE Energy (eV)', 'mae_95_e': 'MAE_95 Energy (eV)', 'mae_f': 'MAE Forces (eV/A)'}[metric]
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, 'loso_three_model_comparison.png')
    plt.savefig(bar_path, dpi=150)
    print(f'  Comparison bar chart saved to: {bar_path}')

    print('\n' + '=' * 60)
    print('Phase 2.2a Complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
