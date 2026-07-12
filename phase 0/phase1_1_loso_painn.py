# -*- coding: utf-8 -*-
"""
Phase 1.1: LOSO Baseline -- PaiNN (Equivariant)
Workspace: D:\lunwen\2.1sci\phase 1\

Leave-One-System-Out: 7-fold, each fold leaves out one system.
Contrasts with SchNet LOSO results.
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
OUTPUT_DIR    = r'D:\lunwen\2.1sci\phase 1\loso_painn_output'
RANDOM_SEED   = 42
BATCH_SIZE    = 32              # PaiNN is heavier, smaller batch
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


# ═══════════════════════════════════════════════════════════
# 1. PaiNN Model
# ═══════════════════════════════════════════════════════════

class RBFExpansion(nn.Module):
    """Same as SchNet: exp(-gamma * (d - center)^2) with cosine cutoff."""
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


class PaiNNMessage(nn.Module):
    """
    PaiNN Message Block:
    - Computes messages from scalar + vector features of neighbors
    - Updates scalar features with GatedEquivariantBlock-style residual
    """
    def __init__(self, hidden_dim, n_rbf):
        super().__init__()
        # Scalar message: radial * scalar[j]
        self.scalar_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )
        # Filter network for distance
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, s, v, edge_src, edge_dst, rbf, num_nodes):
        """
        s: (N, D) scalar features
        v: (N, 3, D) vector features
        edge_src, edge_dst: (E,)
        rbf: (E, n_rbf)
        """
        D = s.size(-1)

        # Compute filters from RBF
        filter_out = self.filter_net(rbf)  # (E, 3D)
        W_s = filter_out[:, :D]              # (E, D)
        W_s2 = filter_out[:, D:2*D]          # (E, D)
        W_v = filter_out[:, 2*D:3*D]          # (E, D)

        # Gather source features
        s_src = s[edge_src]       # (E, D)
        v_src = v[edge_src]       # (E, 3, D)

        # Compute direction vectors
        # We need unit vectors: v_dir = r_ij / |r_ij|
        # We can reconstruct from rbf centers or pass separately.
        # Here we'll compute direction in the message using a proxy:
        # The vector message needs the direction unit vectors.
        # We'll compute them in main forward and pass via extra args.
        # For now, keep the structure and compute directions separately.
        # (This simplified version computes v * direction internally.)

        # Scalar update
        ds = s_src * W_s  # (E, D)
        ds = ds.index_add(0, edge_dst, torch.zeros(num_nodes, D, device=s.device))
        # Actually: scatter_add
        ds_agg = torch.zeros(num_nodes, D, device=s.device)
        ds_agg = ds_agg.index_add(0, edge_dst, s_src * W_s)

        # Vector update (frame dependent on direction vectors)
        # v[r_i] dot u_ij -> scalar, then multiply by u_ij and v[r_i]
        # Need unit vectors -- pass in main forward
        dv = torch.zeros(num_nodes, 3, D, device=s.device)
        # For now placeholder; full implementation in forward
        return s + ds_agg, v + dv


class PaiNNInteraction(nn.Module):
    """
    Full PaiNN message block with direction vectors.
    Combines scalar and vector features with proper equivariance.
    """
    def __init__(self, hidden_dim, n_rbf):
        super().__init__()
        D = hidden_dim
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, 3 * D), nn.SiLU(),
            nn.Linear(3 * D, 3 * D),
        )

    def forward(self, s, v, edge_src, edge_dst, rbf, dir_vec, num_nodes):
        """
        s: (N, D) scalar features
        v: (N, 3, D) vector features (equivariant)
        edge_src, edge_dst: (E,)
        dir_vec: (E, 3) unit direction vectors r_ij / |r_ij|
        """
        D = s.size(-1)
        E = edge_src.size(0)

        # Filter from RBF
        filters = self.filter_net(rbf)  # (E, 3D)
        W_ss = filters[:, :D]              # (E, D)  scalar -> scalar
        W_sv = filters[:, D:2*D]            # (E, D)  vector -> scalar via dot with dir
        W_vv = filters[:, 2*D:3*D]          # (E, D)  vector -> vector

        s_src = s[edge_src]       # (E, D)
        v_src = v[edge_src]       # (E, 3, D)

        # Scalar message: s_j * W_ss + (v_j dot dir) * W_sv
        v_dot_dir = torch.einsum('ejd,ej->ed', v_src, dir_vec)  # (E, D)
        msg_s = s_src * W_ss + v_dot_dir * W_sv  # (E, D)

        # Aggregate scalar messages
        ds = torch.zeros(num_nodes, D, device=s.device)
        ds = ds.index_add(0, edge_dst, msg_s)

        # Vector message: s_j * dir_vec * W_sv (equivariant) + v_j * W_vv
        msg_v1 = torch.einsum('ed,ej->ejd', s_src * W_sv, dir_vec)  # (E, 3, D)
        msg_v2 = v_src * W_vv.unsqueeze(1)  # (E, 3, D)
        msg_v = msg_v1 + msg_v2

        # Aggregate vector messages
        dv = torch.zeros(num_nodes, 3, D, device=s.device)
        dv = dv.index_add(0, edge_dst, msg_v)

        return s + ds, v + dv


class PaiNNUpdate(nn.Module):
    """
    PaiNN Update Block:
    - Mixes scalar and vector features at each node.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        D = hidden_dim
        # U block for scalars
        self.U = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 3 * D),
        )
        # V block for vectors
        self.V = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 2 * D),
        )

    def forward(self, s, v):
        """s: (N, D), v: (N, 3, D)"""
        D = s.size(-1)
        N = s.size(0)

        # --- Update s ---
        U_out = self.U(s)  # (N, 3D)
        a_ss = U_out[:, :D]
        a_sv = U_out[:, D:2*D]
        a_vv = U_out[:, 2*D:3*D]

        # v norm
        v_norm = torch.linalg.norm(v, dim=1)  # (N, D)
        ds = a_ss + a_sv * v_norm

        # --- Update v ---
        V_out = self.V(v_norm)  # (N, 2D)
        a_vv_new = V_out[:, :D]
        a_vs = V_out[:, D:]

        # Linear combination of v and s (via outer product with identity?)
        # v_new = v * a_vv_new + outer(s, a_vs) -- but s is scalar
        # Simplified: v_new = v * a_vv_new + s.unsqueeze(1) * a_vs.unsqueeze(1)
        dv = v * a_vv_new.unsqueeze(1) + s.unsqueeze(1) * a_vs.unsqueeze(1)

        return s + ds, v + dv


class PaiNN(nn.Module):
    """PaiNN: Polarizable Atom Interaction Neural Network."""
    def __init__(self, n_atom_types=100, hidden_dim=128, n_rbf=20, n_interactions=3, cutoff=5.0):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.interactions = nn.ModuleList([
            PaiNNInteraction(hidden_dim, n_rbf) for _ in range(n_interactions)
        ])
        self.updates = nn.ModuleList([
            PaiNNUpdate(hidden_dim) for _ in range(n_interactions)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.hidden_dim = hidden_dim

    def forward(self, z, pos, edge_src, edge_dst, batch=None):
        N = z.size(0)
        D = self.hidden_dim

        # Initialize scalar features
        s = self.embedding(z)  # (N, D)
        # Initialize vector features as zeros
        v = torch.zeros(N, 3, D, device=z.device)

        # Compute distances and directions
        vec = pos[edge_src] - pos[edge_dst]  # (E, 3)
        dist = torch.norm(vec, dim=-1)  # (E,)
        rbf = self.rbf(dist)  # (E, n_rbf)
        # Unit direction vectors (avoid division by zero)
        dir_vec = vec / (dist.unsqueeze(-1) + 1e-10)

        # Interaction + Update blocks
        for interaction, update in zip(self.interactions, self.updates):
            s, v = interaction(s, v, edge_src, edge_dst, rbf, dir_vec, N)
            s, v = update(s, v)

        # Readout: scalar features -> per-atom energy
        atom_energies = self.readout(s)  # (N, 1)

        if batch is not None:
            n_mols = batch.max().item() + 1
            energy = torch.zeros(n_mols, 1, device=z.device)
            energy = energy.index_add(0, batch, atom_energies)
        else:
            energy = atom_energies.sum(0, keepdim=True)

        return energy.squeeze(-1), s


# ═══════════════════════════════════════════════════════════
# 2. Data Loading
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
                'system_id': system_id,
                'group_id': gid,
                'edge_index': data.edge_index.long(),
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
    energy, _ = model(z, pos, ei[0], ei[1], bch)
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
    print('Phase 1.1: LOSO Baseline -- PaiNN')
    print('=' * 60)

    print('\n[1/4] Loading all .pt files ...')
    t0 = time.time()
    all_samples = load_all_pt_files(DATASET_DIR)
    systems = sorted(set(s['system_id'] for s in all_samples))
    print(f'  Loaded {len(all_samples)} samples, {len(systems)} systems: {systems}')
    for sys_id in systems:
        sys_samples = [s for s in all_samples if s['system_id'] == sys_id]
        print(f'    {sys_id}: {len(sys_samples)} samples')

    print(f'\n[2/4] Starting LOSO 7-fold CV (PaiNN) ...')
    print('─' * 60)

    from torch.utils.data import DataLoader

    loso_results = {}

    for fold_idx, test_system in enumerate(systems):
        print(f'\n  Fold {fold_idx + 1}/{len(systems)}: Leave out [{test_system}]')

        test_samples  = [s for s in all_samples if s['system_id'] == test_system]
        train_val_samples = [s for s in all_samples if s['system_id'] != test_system]

        # Group-aware 90/10 train/val split
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

        model = PaiNN(n_atom_types=100, hidden_dim=HIDDEN_DIM, n_rbf=N_RBF,
                      n_interactions=N_INTERACTIONS, cutoff=CUTOFF).to(DEVICE)
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

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_painn_loso_{test_system}.pt'))

    # ── Summary ──
    print('\n' + '=' * 60)
    print('PAINN LOSO RESULTS SUMMARY')
    print('=' * 60)
    print(f'{"Left-Out":<12} {"MAE(E) eV":>12} {"MAE_95(E)":>12} {"MAE(F) eV/A":>14} {"Samples":>9}')
    print('-' * 60)
    all_mae_e, all_mae_f, all_mae_95 = [], [], []
    for sys_id in sorted(loso_results.keys()):
        r = loso_results[sys_id]
        all_mae_e.append(r['mae_e'])
        all_mae_f.append(r['mae_f'])
        all_mae_95.append(r['mae_95_e'])
        print(f'{sys_id:<12} {r["mae_e"]:>12.6f} {r["mae_95_e"]:>12.6f} {r["mae_f"]:>14.6f} {r["n_test"]:>9}')
    print('-' * 60)
    print(f'{"Mean":<12} {np.mean(all_mae_e):>12.6f} {np.mean(all_mae_95):>12.6f} {np.mean(all_mae_f):>14.6f}')
    print(f'{"Std":<12} {np.std(all_mae_e):>12.6f} {np.std(all_mae_95):>12.6f} {np.std(all_mae_f):>14.6f}')

    # ── Bar chart ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sys_names = sorted(loso_results.keys())
    colors_e  = ['#d62728' if s == 'LaCu12' else '#1f77b4' for s in sys_names]
    axes[0].bar(sys_names, [loso_results[s]['mae_e'] for s in sys_names], color=colors_e)
    axes[0].set_ylabel('MAE (eV)'); axes[0].set_title('MAE Energy'); axes[0].tick_params(axis='x', rotation=45)
    axes[1].bar(sys_names, [loso_results[s]['mae_95_e'] for s in sys_names], color=['#d62728' if s == 'LaCu12' else '#ff7f0e' for s in sys_names])
    axes[1].set_ylabel('MAE_95 (eV)'); axes[1].set_title('MAE_95 Energy (Tail)'); axes[1].tick_params(axis='x', rotation=45)
    colors_f  = ['#d62728' if s == 'LaCu12' else '#2ca02c' for s in sys_names]
    axes[2].bar(sys_names, [loso_results[s]['mae_f'] for s in sys_names], color=colors_f)
    axes[2].set_ylabel('MAE (eV/A)'); axes[2].set_title('MAE Forces'); axes[2].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, 'loso_painn_results_bar.png')
    plt.savefig(bar_path, dpi=150)
    print(f'\n  Bar chart saved to: {bar_path}')

    csv_path = os.path.join(OUTPUT_DIR, 'loso_painn_results.csv')
    with open(csv_path, 'w') as f:
        f.write('Left_Out_System,MAE_E_eV,MAE_95_E_eV,MAE_F_eV_A,N_Test,Time_s\n')
        for sys_id in sorted(loso_results.keys()):
            r = loso_results[sys_id]
            f.write(f'{sys_id},{r["mae_e"]:.6f},{r["mae_95_e"]:.6f},{r["mae_f"]:.6f},{r["n_test"]},{r["train_time"]:.1f}\n')
    print(f'  Results CSV saved to: {csv_path}')

    print('\n' + '=' * 60)
    print('Phase 1.1 PaiNN LOSO Complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
