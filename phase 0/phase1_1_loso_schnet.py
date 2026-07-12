# -*- coding: utf-8 -*-
"""
Phase 1.1: LOSO Baseline -- SchNet (Leave-One-System-Out)
Workspace: D:\lunwen\2.1sci\phase 1\

Leave-One-System-Out: 7-fold, each fold leaves out one system.
Train on 6 systems, test on the held-out system.
Reports per-system MAE(E) and MAE(F) + summary table.
"""
import os, time, copy
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
OUTPUT_DIR    = r'D:\lunwen\2.1sci\phase 1\loso_schnet_output'
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

# Re-set seeds at start (data loading uses random for split)
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f'DEVICE: {DEVICE}')


# ═══════════════════════════════════════════════════════════
# 1. SchNet Model (same as Phase 0.2)
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


class InteractionBlock(nn.Module):
    def __init__(self, hidden_dim, n_rbf):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.atom_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_src, edge_dst, rbf, num_nodes):
        filters = self.filter_net(rbf)
        messages = x[edge_src] * filters
        aggregated = torch.zeros(num_nodes, x.size(-1), device=x.device)
        aggregated = aggregated.index_add(0, edge_dst, messages)
        x = x + self.out_net(self.atom_net(x) + aggregated)
        return x


class SchNet(nn.Module):
    def __init__(self, n_atom_types=100, hidden_dim=128, n_rbf=20, n_interactions=3, cutoff=5.0):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, n_rbf) for _ in range(n_interactions)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z, pos, edge_src, edge_dst, batch=None):
        N = z.size(0)
        x = self.embedding(z)
        vec = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(vec, dim=-1)
        rbf = self.rbf(dist)
        for interaction in self.interactions:
            x = interaction(x, edge_src, edge_dst, rbf, N)
        atom_energies = self.readout(x)
        if batch is not None:
            n_mols = batch.max().item() + 1
            energy = torch.zeros(n_mols, 1, device=z.device)
            energy = energy.index_add(0, batch, atom_energies)
        else:
            energy = atom_energies.sum(0, keepdim=True)
        return energy.squeeze(-1), x


# ═══════════════════════════════════════════════════════════
# 2. Data Loading (per-system)
# ═══════════════════════════════════════════════════════════

def load_all_pt_files(dataset_dir):
    """Load all .pt files with system_id."""
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
            z   = data.atomic_numbers.long()
            pos = data.pos.float()
            y   = data.y.float().item()
            ei  = data.edge_index.long()
            
            # Use cleaned system_id from directory name or from data attribute
            gid = getattr(data, 'group_id', 0) if hasattr(data, 'group_id') else 0
            samples.append({
                'z': z, 'pos': pos, 'y': y, 'forces': data.forces.float(),
                'system_id': system_id, 'group_id': gid,
                'edge_index': ei,
            })
    return samples


def collate_batch(batch_samples):
    z_list, pos_list, y_list, f_list, edge_list = [], [], [], [], []
    batch_idx = []
    n_total = 0
    for i, s in enumerate(batch_samples):
        n = s['z'].size(0)
        z_list.append(s['z'])
        pos_list.append(s['pos'])
        y_list.append(s['y'])
        f_list.append(s['forces'])
        edge_list.append(s['edge_index'] + n_total)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        n_total += n
    return {
        'z': torch.cat(z_list, dim=0),
        'pos': torch.cat(pos_list, dim=0),
        'y': torch.tensor(y_list, dtype=torch.float32),
        'forces': torch.cat(f_list, dim=0),
        'edge_index': torch.cat(edge_list, dim=1),
        'batch': torch.cat(batch_idx, dim=0),
    }


# ═══════════════════════════════════════════════════════════
# 3. Training & Evaluation
# ═══════════════════════════════════════════════════════════

def compute_energy_and_forces(model, batch, device):
    z   = batch['z'].to(device)
    pos = batch['pos'].to(device).requires_grad_(True)
    ei  = batch['edge_index'].to(device)
    bch = batch['batch'].to(device)
    edge_src, edge_dst = ei[0], ei[1]
    energy, _ = model(z, pos, edge_src, edge_dst, bch)
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

    result = (
        total_mae_e / n_batches,
        total_mae_f / n_batches,
        total_loss  / n_batches,
    )
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
    print('Phase 1.1: LOSO Baseline -- SchNet')
    print('=' * 60)

    # ── 4a. Load all data ──
    print('\n[1/4] Loading all .pt files ...')
    t0 = time.time()
    all_samples = load_all_pt_files(DATASET_DIR)
    systems = sorted(set(s['system_id'] for s in all_samples))
    print(f'  Loaded {len(all_samples)} samples, {len(systems)} systems: {systems}')
    print(f'  Time: {time.time() - t0:.1f}s')

    # Print system statistics
    for sys_id in systems:
        sys_samples = [s for s in all_samples if s['system_id'] == sys_id]
        unique_groups = len(set(s['group_id'] for s in sys_samples))
        print(f'    {sys_id}: {len(sys_samples)} samples, {unique_groups} groups')

    # ── 4b. LOSO loop ──
    print(f'\n[2/4] Starting LOSO 7-fold cross-validation ...')
    print('─' * 60)

    from torch.utils.data import DataLoader

    loso_results = {}
    all_predictions = {}  # For later analysis

    for fold_idx, test_system in enumerate(systems):
        print(f'\n  Fold {fold_idx + 1}/{len(systems)}: Leave out [{test_system}]')

        # Split data
        test_samples  = [s for s in all_samples if s['system_id'] == test_system]
        train_val_samples = [s for s in all_samples if s['system_id'] != test_system]

        # Split train/val within training systems (by group, 90/10)
        import random as _random
        _random.seed(RANDOM_SEED)
        train_val_groups = defaultdict(list)
        for s in train_val_samples:
            key = (s['system_id'], s['group_id'])
            train_val_groups[key].append(s)

        group_keys = sorted(train_val_groups.keys())
        _random.shuffle(group_keys)
        n_train_groups = int(len(group_keys) * 0.90)
        train_groups_set = set(group_keys[:n_train_groups])

        train_samples = []
        val_samples   = []
        for s in train_val_samples:
            key = (s['system_id'], s['group_id'])
            if key in train_groups_set:
                train_samples.append(s)
            else:
                val_samples.append(s)

        print(f'    Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}')

        # ── Create dataloaders ──
        train_loader = DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_batch, drop_last=True)
        val_loader   = DataLoader(val_samples,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        test_loader  = DataLoader(test_samples,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

        # ── Build fresh model per fold ──
        model = SchNet(
            n_atom_types=100, hidden_dim=HIDDEN_DIM, n_rbf=N_RBF,
            n_interactions=N_INTERACTIONS, cutoff=CUTOFF
        ).to(DEVICE)

        # ── Train ──
        t_start = time.time()
        model, best_val_loss = train_one_fold(
            model, train_loader, val_loader, DEVICE, FORCE_WEIGHT,
            EPOCHS, EARLY_STOP, test_system
        )
        train_time = time.time() - t_start

        # ── Test on held-out system ──
        test_mae_e, test_mae_f, test_loss, y_true, y_pred = evaluate(
            model, test_loader, DEVICE, FORCE_WEIGHT, return_preds=True
        )

        # Compute tail error (MAE_95 / worst-5%)
        abs_errors = torch.abs(y_true - y_pred)
        sorted_errors, _ = torch.sort(abs_errors, descending=True)
        tail_k = max(1, int(len(sorted_errors) * 0.05))
        mae_95 = sorted_errors[:tail_k].mean().item()

        loso_results[test_system] = {
            'mae_e': test_mae_e,
            'mae_f': test_mae_f,
            'loss': test_loss,
            'mae_95_e': mae_95,
            'n_test': len(test_samples),
            'train_time': train_time,
        }
        all_predictions[test_system] = (y_true, y_pred)

        print(f'    Result: MAE(E)={test_mae_e:.6f} eV, MAE(F)={test_mae_f:.6f} eV/A, '
              f'MAE_95(E)={mae_95:.6f} eV, Time={train_time:.1f}s')

        # Save fold model
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_loso_{test_system}.pt'))

    # ── 4c. Summary table ──
    print('\n' + '=' * 60)
    print('LOSO RESULTS SUMMARY')
    print('=' * 60)
    print(f'{"Left-Out":<12} {"MAE(E) eV":>12} {"MAE_95(E)":>12} {"MAE(F) eV/A":>14} {"Samples":>9}')
    print('-' * 60)
    
    all_mae_e = []
    all_mae_f = []
    all_mae_95 = []
    
    for sys_id in sorted(loso_results.keys()):
        r = loso_results[sys_id]
        all_mae_e.append(r['mae_e'])
        all_mae_f.append(r['mae_f'])
        all_mae_95.append(r['mae_95_e'])
        print(f'{sys_id:<12} {r["mae_e"]:>12.6f} {r["mae_95_e"]:>12.6f} {r["mae_f"]:>14.6f} {r["n_test"]:>9}')

    print('-' * 60)
    print(f'{"Mean":<12} {np.mean(all_mae_e):>12.6f} {np.mean(all_mae_95):>12.6f} {np.mean(all_mae_f):>14.6f}')
    print(f'{"Std":<12} {np.std(all_mae_e):>12.6f} {np.std(all_mae_95):>12.6f} {np.std(all_mae_f):>14.6f}')

    # ── 4d. Bar chart ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sys_names = sorted(loso_results.keys())
    
    colors_e  = ['#d62728' if s == 'LaCu12' else '#1f77b4' for s in sys_names]
    colors_95 = ['#d62728' if s == 'LaCu12' else '#ff7f0e' for s in sys_names]
    colors_f  = ['#d62728' if s == 'LaCu12' else '#2ca02c' for s in sys_names]

    axes[0].bar(sys_names, [loso_results[s]['mae_e'] for s in sys_names], color=colors_e)
    axes[0].set_ylabel('MAE (eV)')
    axes[0].set_title('MAE Energy')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(sys_names, [loso_results[s]['mae_95_e'] for s in sys_names], color=colors_95)
    axes[1].set_ylabel('MAE_95 (eV)')
    axes[1].set_title('MAE_95 Energy (Tail)')
    axes[1].tick_params(axis='x', rotation=45)

    axes[2].bar(sys_names, [loso_results[s]['mae_f'] for s in sys_names], color=colors_f)
    axes[2].set_ylabel('MAE (eV/A)')
    axes[2].set_title('MAE Forces')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, 'loso_results_bar.png')
    plt.savefig(bar_path, dpi=150)
    print(f'\n  Bar chart saved to: {bar_path}')

    # ── 4e. Save results to CSV ──
    csv_path = os.path.join(OUTPUT_DIR, 'loso_results.csv')
    with open(csv_path, 'w') as f:
        f.write('Left-Out System,MAE_E_eV,MAE_95_E_eV,MAE_F_eV_A,N_Test,Time_s\n')
        for sys_id in sorted(loso_results.keys()):
            r = loso_results[sys_id]
            f.write(f'{sys_id},{r["mae_e"]:.6f},{r["mae_95_e"]:.6f},'
                    f'{r["mae_f"]:.6f},{r["n_test"]},{r["train_time"]:.1f}\n')
    print(f'  Results CSV saved to: {csv_path}')

    print('\n' + '=' * 60)
    print('Phase 1.1 Complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
