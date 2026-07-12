# -*- coding: utf-8 -*-
"""
Phase 0.2: SchNet Baseline -- Group-Aware Random Split + E+F Sanity Check
Workspace: D:\lunwen\2.1sci\phase 0\
"""
import os, time, random
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
OUTPUT_DIR    = r'D:\lunwen\2.1sci\phase 0\phase0_2_output'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED   = 42
TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.10
BATCH_SIZE    = 32
EPOCHS        = 120
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
FORCE_WEIGHT  = 0.5
CUTOFF        = 5.0              # Angstrom
N_RBF         = 20               # Number of radial basis functions
HIDDEN_DIM    = 128              # SchNet embedding dimension
N_INTERACTIONS = 3               # Number of interaction blocks
EARLY_STOP    = 30               # Patience for early stopping

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
print(f'DEVICE: {DEVICE}')


# ═══════════════════════════════════════════════════════════
# 1. SchNet Model (Pure PyTorch)
# ═══════════════════════════════════════════════════════════

class RBFExpansion(nn.Module):
    """Radial basis function expansion of interatomic distances."""
    def __init__(self, cutoff, n_rbf):
        super().__init__()
        self.cutoff = cutoff
        self.centers = nn.Parameter(torch.linspace(0.0, cutoff, n_rbf), requires_grad=False)
        gamma = 0.5 / ((self.centers[1] - self.centers[0]) ** 2 + 1e-8)
        self.gamma = gamma
        self.n_rbf = n_rbf

    def forward(self, distances):
        """distances: (E,) → (E, n_rbf)"""
        d = distances.unsqueeze(-1)  # (E, 1)
        c = self.centers.unsqueeze(0)  # (1, n_rbf)
        rbf = torch.exp(-self.gamma * (d - c) ** 2)
        # Apply cutoff envelope: cosine cutoff
        cutoff_val = 0.5 * (1.0 + torch.cos(np.pi * distances / self.cutoff))
        cutoff_val = cutoff_val.unsqueeze(-1)
        return rbf * cutoff_val


class InteractionBlock(nn.Module):
    """Continuous-filter convolution block (SchNet interaction)."""
    def __init__(self, hidden_dim, n_rbf):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.atom_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_src, edge_dst, rbf, num_nodes):
        """
        x: (N, D) node features
        edge_src, edge_dst: (E,) edge indices
        rbf: (E, n_rbf) RBF-expanded distances
        num_nodes: N
        """
        # Compute filter weights from RBF
        filters = self.filter_net(rbf)  # (E, D)
        # Gather source node features
        src_features = x[edge_src]  # (E, D)
        # Element-wise multiplication
        messages = src_features * filters  # (E, D)
        # Aggregate messages to destination nodes
        aggregated = torch.zeros(num_nodes, x.size(-1), device=x.device)
        aggregated = aggregated.index_add(0, edge_dst, messages)
        # Update node features
        x = x + self.out_net(self.atom_net(x) + aggregated)
        return x


class SchNet(nn.Module):
    """SchNet for energy prediction with force computation via autograd."""
    def __init__(self, n_atom_types=100, hidden_dim=128, n_rbf=20, n_interactions=3, cutoff=5.0):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, n_rbf) for _ in range(n_interactions)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z, pos, edge_src, edge_dst, batch=None):
        """
        z: (N,) atomic numbers
        pos: (N, 3) positions (for force computation)
        edge_src, edge_dst: (E,) edge indices
        batch: (N,) batch index (for per-molecule predictions)
        Returns: energy (B,), atom features (N, D)
        """
        N = z.size(0)
        x = self.embedding(z)  # (N, D)

        # Compute distances
        vec = pos[edge_src] - pos[edge_dst]  # (E, 3)
        dist = torch.norm(vec, dim=-1)  # (E,)
        rbf = self.rbf(dist)  # (E, n_rbf)

        # Interaction blocks
        for interaction in self.interactions:
            x = interaction(x, edge_src, edge_dst, rbf, N)

        # Per-atom energy contributions
        atom_energies = self.readout(x)  # (N, 1)

        if batch is not None:
            # Sum per-molecule
            n_mols = batch.max().item() + 1
            energy = torch.zeros(n_mols, 1, device=z.device)
            energy = energy.index_add(0, batch, atom_energies)
        else:
            energy = atom_energies.sum(0, keepdim=True)

        return energy.squeeze(-1), x


# ═══════════════════════════════════════════════════════════
# 2. Data Loading
# ═══════════════════════════════════════════════════════════

def load_all_pt_files(dataset_dir):
    """Load all .pt files, return list of dicts with graph info + split keys."""
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

            sid = getattr(data, 'system_id', system_id) if hasattr(data, 'system_id') else system_id
            gid = getattr(data, 'group_id', -1)      if hasattr(data, 'group_id')       else -1
            tid = getattr(data, 'trajectory_id', -1) if hasattr(data, 'trajectory_id')  else -1
            iid = getattr(data, 'init_structure_id', -1) if hasattr(data, 'init_structure_id') else -1

            z    = data.atomic_numbers.long()
            pos  = data.pos.float()
            y    = data.y.float().item()
            edge_index = data.edge_index.long()

            samples.append({
                'z': z, 'pos': pos, 'y': y, 'forces': data.forces.float(),
                'system_id': sid, 'group_id': gid, 'trajectory_id': tid,
                'edge_index': edge_index,
            })
    return samples


def group_aware_split(samples, train_r=0.80, val_r=0.10, seed=42):
    """
    Split by group_id: same group_id → same fold.
    Returns list of sample dicts annotated with 'fold' key.
    """
    rng = random.Random(seed)
    # Group indices by (system_id, group_id)
    groups = defaultdict(list)
    for idx, s in enumerate(samples):
        key = (s['system_id'], s['group_id'])
        groups[key].append(idx)

    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)
    n_groups = len(group_keys)
    n_train = int(n_groups * train_r)
    n_val   = int(n_groups * val_r)

    fold_map = {}
    for i, gk in enumerate(group_keys):
        if i < n_train:
            fold_map[gk] = 'train'
        elif i < n_train + n_val:
            fold_map[gk] = 'val'
        else:
            fold_map[gk] = 'test'

    for s in samples:
        key = (s['system_id'], s['group_id'])
        s['fold'] = fold_map[key]
    return samples


def collate_batch(batch_samples):
    """Collate list of sample dicts into a single graph batch."""
    z_list, pos_list, y_list, f_list, edge_list = [], [], [], [], []
    batch_idx = []
    n_total = 0

    for i, s in enumerate(batch_samples):
        n = s['z'].size(0)
        z_list.append(s['z'])
        pos_list.append(s['pos'])
        y_list.append(s['y'])
        f_list.append(s['forces'])
        # Shift edge indices
        ei = s['edge_index']
        edge_list.append(ei + n_total)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        n_total += n

    return {
        'z':          torch.cat(z_list, dim=0),
        'pos':        torch.cat(pos_list, dim=0),
        'y':          torch.tensor(y_list, dtype=torch.float32),
        'forces':     torch.cat(f_list, dim=0),
        'edge_index': torch.cat(edge_list, dim=1),
        'batch':      torch.cat(batch_idx, dim=0),
    }


# ═══════════════════════════════════════════════════════════
# 3. Training & Evaluation
# ═══════════════════════════════════════════════════════════

def compute_energy_and_forces(model, batch, device):
    """
    Forward pass + force computation via autograd.
    Returns predicted energy (B,), predicted forces (N, 3).
    """
    z     = batch['z'].to(device)
    pos   = batch['pos'].to(device).requires_grad_(True)
    ei    = batch['edge_index'].to(device)
    bch   = batch['batch'].to(device)

    edge_src, edge_dst = ei[0], ei[1]
    energy, _ = model(z, pos, edge_src, edge_dst, bch)  # (B,)

    # Forces = -grad(E) w.r.t positions
    if pos.grad is not None:
        pos.grad.zero_()
    forces = -torch.autograd.grad(
        energy.sum(), pos, create_graph=True, retain_graph=True
    )[0]

    return energy, forces


def evaluate(model, loader, device, force_weight):
    model.eval()
    total_mae_e = 0.0
    total_mae_f = 0.0
    total_loss  = 0.0
    n_batches   = 0

    # Do NOT use torch.no_grad() here, because force computation needs grad graph
    for batch in loader:
        energy_pred, forces_pred = compute_energy_and_forces(model, batch, device)
        energy_true  = batch['y'].to(device)
        forces_true  = batch['forces'].to(device)

        loss_e = F.l1_loss(energy_pred, energy_true)
        loss_f = F.l1_loss(forces_pred, forces_true)
        loss   = loss_e + force_weight * loss_f

        total_mae_e += loss_e.item()
        total_mae_f += loss_f.item()
        total_loss  += loss.item()
        n_batches   += 1

    return total_mae_e / n_batches, total_mae_f / n_batches, total_loss / n_batches


def train_epoch(model, loader, optimizer, device, force_weight):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for batch in loader:
        optimizer.zero_grad()
        energy_pred, forces_pred = compute_energy_and_forces(model, batch, device)
        energy_true  = batch['y'].to(device)
        forces_true  = batch['forces'].to(device)

        loss_e = F.l1_loss(energy_pred, energy_true)
        loss_f = F.l1_loss(forces_pred, forces_true)
        loss   = loss_e + force_weight * loss_f

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches


# ═══════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('Phase 0.2: SchNet Baseline -- Group-Aware Random Split')
    print('=' * 60)

    # ── 4a. Load data ──
    print('\n[1/5] Loading .pt files ...')
    t0 = time.time()
    samples = load_all_pt_files(DATASET_DIR)
    print(f'  Loaded {len(samples)} samples in {time.time() - t0:.1f}s')

    # ── 4b. Group-aware split ──
    print('\n[2/5] Performing group-aware random split ...')
    samples = group_aware_split(samples, train_r=0.80, val_r=0.10, seed=42)

    train_data = [s for s in samples if s['fold'] == 'train']
    val_data   = [s for s in samples if s['fold'] == 'val']
    test_data  = [s for s in samples if s['fold'] == 'test']
    print(f'  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')

    # Check group leakage
    train_groups = set((s['system_id'], s['group_id']) for s in train_data)
    val_groups   = set((s['system_id'], s['group_id']) for s in val_data)
    test_groups  = set((s['system_id'], s['group_id']) for s in test_data)
    assert train_groups.isdisjoint(val_groups), 'Train/Val group overlap!'
    assert train_groups.isdisjoint(test_groups), 'Train/Test group overlap!'
    assert val_groups.isdisjoint(test_groups), 'Val/Test group overlap!'
    print('  Group leakage check: PASSED')

    # ── 4c. Create dataloaders ──
    print('\n[3/5] Building dataloaders ...')
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_batch)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    print(f'  Train batches: {len(train_loader)}')
    print(f'  Val   batches: {len(val_loader)}')
    print(f'  Test  batches: {len(test_loader)}')

    # ── 4d. Build model ──
    print('\n[4/5] Building SchNet model ...')
    model = SchNet(
        n_atom_types=100,
        hidden_dim=HIDDEN_DIM,
        n_rbf=N_RBF,
        n_interactions=N_INTERACTIONS,
        cutoff=CUTOFF
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # ── 4e. Training loop ──
    print(f'\n[5/5] Training on {DEVICE} ...')
    print('─' * 50)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae_e': [], 'val_mae_f': []}

    for epoch in range(1, EPOCHS + 1):
        t_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, FORCE_WEIGHT)
        val_mae_e, val_mae_f, val_loss = evaluate(model, val_loader, DEVICE, FORCE_WEIGHT)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae_e'].append(val_mae_e)
        history['val_mae_f'].append(val_mae_f)

        elapsed = time.time() - t_start
        print(f'  Epoch {epoch:3d}/{EPOCHS} | '
              f'Train Loss: {train_loss:.6f} | '
              f'Val Loss: {val_loss:.6f} | '
              f'Val MAE(E): {val_mae_e:.6f} eV | '
              f'Val MAE(F): {val_mae_f:.6f} eV/A | '
              f'Time: {elapsed:.1f}s')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOP:
            print(f'  Early stopping at epoch {epoch}')
            break

    # ── 4f. Final test evaluation ──
    print('\n' + '─' * 50)
    print('Final Test Evaluation')
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt')))
    test_mae_e, test_mae_f, test_loss = evaluate(model, test_loader, DEVICE, FORCE_WEIGHT)
    print(f'  Test MAE(E):  {test_mae_e:.6f} eV')
    print(f'  Test MAE(F):  {test_mae_f:.6f} eV/A')
    print(f'  Test Loss:    {test_loss:.6f}')

    # ── 4g. Plot loss curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', color='steelblue')
    axes[0].plot(epochs_range, history['val_loss'],   label='Val Loss',   color='darkorange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (E + lambda*F)')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history['val_mae_e'], label='Val MAE(E) [eV]',  color='forestgreen')
    axes[1].plot(epochs_range, history['val_mae_f'], label='Val MAE(F) [eV/A]', color='crimson')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Validation MAE (Energy & Forces)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    loss_plot_path = os.path.join(OUTPUT_DIR, 'loss_curves.png')
    plt.savefig(loss_plot_path, dpi=150)
    print(f'\n  Loss curves saved to: {loss_plot_path}')

    # ── 4h. Summary ──
    print('\n' + '=' * 60)
    print('Phase 0.2 Complete!')
    print(f'  Best Val Loss:     {best_val_loss:.6f}')
    print(f'  Test MAE(E):       {test_mae_e:.6f} eV')
    print(f'  Test MAE(F):       {test_mae_f:.6f} eV/A')
    print(f'  Model saved:       {os.path.join(OUTPUT_DIR, "best_model.pt")}')
    print(f'  Total epochs run:  {len(history["train_loss"])}')
    print('=' * 60)

    return test_mae_e, test_mae_f


if __name__ == '__main__':
    main()
