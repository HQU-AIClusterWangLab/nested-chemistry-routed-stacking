# -*- coding: utf-8 -*-
"""
Phase 3: Ensemble UQ Evaluation — PAA-SchNet
Workspace: D:\lunwen\2.1sci\phase 3\

Trains 5-seed ensemble of PAA-SchNet per LOSO fold (and IID split).
Evaluates UQ quality on held-out test data via:
  1. Pearson r(var, |error|)
  2. Risk-Coverage Curve
  3. Expected Normalized Calibration Error (ENCE)
Compares LOSO vs IID UQ behavior.
"""
import os, time, random, copy, json
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ── Config ──────────────────────────────────────────────
DATASET_DIR   = r'D:\lunwen\2.1sci\phase 0\dataset\processed'
OUTPUT_DIR    = r'D:\lunwen\2.1sci\phase 3\ensemble_uq_output'
N_SEEDS       = 5
SEEDS         = [42, 123, 456, 789, 1024]
BATCH_SIZE    = 64
EPOCHS        = 120             # slightly reduced for ensemble
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
FORCE_WEIGHT  = 0.5
CUTOFF        = 5.0
N_RBF         = 20
HIDDEN_DIM    = 128
N_INTERACTIONS = 3
EARLY_STOP    = 25
PHYS_DIM      = 5
EDGE_PHYS_DIM = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'DEVICE: {DEVICE}')
print(f'N_SEEDS: {N_SEEDS}, SEEDS: {SEEDS}')


# ═══════════════════════════════════════════════════════════
# 1. PAA-SchNet Model (identical to Phase 2.2a)
# ═══════════════════════════════════════════════════════════

class RBFExpansion(torch.nn.Module):
    def __init__(self, cutoff, n_rbf):
        super().__init__()
        self.cutoff = cutoff
        self.centers = torch.nn.Parameter(torch.linspace(0.0, cutoff, n_rbf), requires_grad=False)
        self.gamma = 0.5 / ((self.centers[1] - self.centers[0]) ** 2 + 1e-8)

    def forward(self, distances):
        d = distances.unsqueeze(-1)
        c = self.centers.unsqueeze(0)
        rbf = torch.exp(-self.gamma * (d - c) ** 2)
        cutoff_val = 0.5 * (1.0 + torch.cos(np.pi * distances / self.cutoff))
        return rbf * cutoff_val.unsqueeze(-1)


class EdgeBiasGate(torch.nn.Module):
    def __init__(self, edge_phys_dim, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(edge_phys_dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 1), torch.nn.Sigmoid()
        )
    def forward(self, edge_diffs):
        return self.net(edge_diffs).squeeze(-1)


class PAAInteractionBlock(torch.nn.Module):
    def __init__(self, hidden_dim, n_rbf, edge_phys_dim):
        super().__init__()
        self.filter_net = torch.nn.Sequential(
            torch.nn.Linear(n_rbf, hidden_dim), torch.nn.SiLU(), torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_gate = EdgeBiasGate(edge_phys_dim)
        self.atom_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(), torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(), torch.nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_src, edge_dst, rbf, edge_diffs, num_nodes):
        filters = self.filter_net(rbf)
        gate = self.edge_gate(edge_diffs)
        messages = x[edge_src] * filters * gate.unsqueeze(-1)
        aggregated = torch.zeros(num_nodes, x.size(-1), device=x.device)
        aggregated = aggregated.index_add(0, edge_dst, messages)
        x = x + self.out_net(self.atom_net(x) + aggregated)
        return x


class PAASchNet(torch.nn.Module):
    def __init__(self, n_atom_types=100, phys_dim=PHYS_DIM, edge_phys_dim=EDGE_PHYS_DIM,
                 hidden_dim=128, n_rbf=20, n_interactions=3, cutoff=5.0):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.phys_proj = torch.nn.Linear(phys_dim, hidden_dim)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.interactions = torch.nn.ModuleList([
            PAAInteractionBlock(hidden_dim, n_rbf, edge_phys_dim) for _ in range(n_interactions)
        ])
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2), torch.nn.SiLU(), torch.nn.Linear(hidden_dim // 2, 1)
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
            edge_diffs = torch.abs(phys_feats[edge_src] - phys_feats[edge_dst])
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
                'z':        data.atomic_numbers.long(),
                'pos':      data.pos.float(),
                'y':        data.y.float().item(),
                'forces':   data.forces.float(),
                'phys_feats': data.x.float(),
                'system_id':  system_id,
                'group_id':   gid,
                'edge_index': data.edge_index.long(),
            })
    return samples


def collate_batch(batch_samples):
    z_list, pos_list, y_list, f_list, phys_list, edge_list = [], [], [], [], [], []
    batch_idx, n_total = [], 0
    for i, s in enumerate(batch_samples):
        n = s['z'].size(0)
        z_list.append(s['z']); pos_list.append(s['pos']); y_list.append(s['y'])
        f_list.append(s['forces']); phys_list.append(s['phys_feats'])
        edge_list.append(s['edge_index'] + n_total)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        n_total += n
    return {
        'z': torch.cat(z_list), 'pos': torch.cat(pos_list),
        'y': torch.tensor(y_list, dtype=torch.float32),
        'forces': torch.cat(f_list), 'phys_feats': torch.cat(phys_list),
        'edge_index': torch.cat(edge_list, dim=1), 'batch': torch.cat(batch_idx),
    }


# ═══════════════════════════════════════════════════════════
# 3. Training
# ═══════════════════════════════════════════════════════════

def compute_energy_and_forces(model, batch, device):
    z = batch['z'].to(device)
    pos = batch['pos'].to(device).requires_grad_(True)
    phys = batch['phys_feats'].to(device)
    ei = batch['edge_index'].to(device)
    bch = batch['batch'].to(device)
    energy, _ = model(z, pos, ei[0], ei[1], phys_feats=phys, batch=bch)
    forces = -torch.autograd.grad(energy.sum(), pos, create_graph=True, retain_graph=True)[0]
    return energy, forces


def train_one_model(model, train_loader, val_loader, device, force_weight, epochs, early_stop, verbose=False):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ep, fp = compute_energy_and_forces(model, batch, device)
            loss = F.l1_loss(ep, batch['y'].to(device)) + force_weight * F.l1_loss(fp, batch['forces'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        for batch in val_loader:
            ep, fp = compute_energy_and_forces(model, batch, device)
            val_loss += (F.l1_loss(ep, batch['y'].to(device)) + force_weight * F.l1_loss(fp, batch['forces'].to(device))).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= early_stop:
            break

    model.load_state_dict(best_state)
    return model


def predict_single_model(model, loader, device):
    """Return per-sample energy predictions."""
    model.eval()
    all_preds = []
    all_true = []
    for batch in loader:
        z = batch['z'].to(device)
        pos = batch['pos'].to(device).requires_grad_(True)
        phys = batch['phys_feats'].to(device)
        ei = batch['edge_index'].to(device)
        bch = batch['batch'].to(device)
        ep, _ = model(z, pos, ei[0], ei[1], phys_feats=phys, batch=bch)
        n_mols = int(bch.max().item()) + 1
        for m in range(n_mols):
            all_preds.append(ep[m].item())
            all_true.append(batch['y'][m].item())
    return np.array(all_preds), np.array(all_true)


# ═══════════════════════════════════════════════════════════
# 4. Ensemble prediction + UQ collection
# ═══════════════════════════════════════════════════════════

def train_ensemble_and_predict(train_samples, val_samples, test_samples, seed_list,
                                device, fold_name, output_dir):
    """
    Train an ensemble of PAASchNet with given seeds.
    Collect predictions from each seed on test set.
    Returns:
        all_preds:  (N_seeds, N_test) array
        y_true:     (N_test,) array
    """
    from torch.utils.data import DataLoader

    all_preds = []
    y_true = None

    for s_idx, seed in enumerate(seed_list):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_loader = DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_batch, drop_last=True)
        val_loader   = DataLoader(val_samples,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        test_loader  = DataLoader(test_samples,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

        model = PAASchNet(n_atom_types=100, phys_dim=PHYS_DIM, edge_phys_dim=EDGE_PHYS_DIM,
                          hidden_dim=HIDDEN_DIM, n_rbf=N_RBF, n_interactions=N_INTERACTIONS,
                          cutoff=CUTOFF).to(device)

        print(f'    Seed {seed} ({s_idx+1}/{len(seed_list)}) ...', end=' ', flush=True)
        t0 = time.time()
        model = train_one_model(model, train_loader, val_loader, device, FORCE_WEIGHT,
                                EPOCHS, EARLY_STOP, verbose=False)
        preds, true = predict_single_model(model, test_loader, device)
        all_preds.append(preds)
        if y_true is None:
            y_true = true
        print(f'done ({time.time()-t0:.0f}s)')

        # Save model
        torch.save(model.state_dict(),
                   os.path.join(output_dir, f'ensemble_{fold_name}_seed{seed}.pt'))

    all_preds = np.array(all_preds)  # (N_seeds, N_test)
    return all_preds, y_true


# ═══════════════════════════════════════════════════════════
# 5. UQ Metrics
# ═══════════════════════════════════════════════════════════

def compute_uq_metrics(ensemble_preds, y_true, n_bins=15):
    """
    ensemble_preds: (N_seeds, N_samples)
    y_true: (N_samples,)

    Returns dict with:
        mean_pred:     (N,) ensemble mean
        variance:      (N,) ensemble variance
        abs_error:     (N,) |mean_pred - y_true|
        pearson_r:     float, correlation(var, abs_error)
        pearson_p:     float, p-value
        rank_corr:     float, Spearman rank correlation
        ence:          float, Expected Normalized Calibration Error
        rmse_per_bin:  list
        rmv_per_bin:   list
        risk_coverage: dict with 'coverage' and 'risk' arrays
    """
    mean_pred = ensemble_preds.mean(axis=0)
    variance  = ensemble_preds.var(axis=0)  # population variance
    abs_error = np.abs(mean_pred - y_true)

    # Pearson correlation: var vs abs_error
    r, p = pearsonr(variance, abs_error)
    
    # Spearman rank correlation (more robust to outliers)
    from scipy.stats import spearmanr
    sr, sp = spearmanr(variance, abs_error)

    # ── ENCE ──
    # Bin samples by predicted std (sqrt(variance))
    std = np.sqrt(variance + 1e-10)
    bin_edges = np.percentile(std, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1e-6; bin_edges[-1] += 1e-6  # ensure inclusivity
    
    rmse_per_bin = []
    rmv_per_bin  = []
    bin_weights   = []
    
    for j in range(n_bins):
        mask = (std >= bin_edges[j]) & (std < bin_edges[j+1])
        n_in_bin = mask.sum()
        if n_in_bin < 2:
            continue
        bin_err = abs_error[mask]
        bin_var = variance[mask]
        rmse = np.sqrt(np.mean(bin_err ** 2))
        rmv  = np.sqrt(np.mean(bin_var))
        rmse_per_bin.append(rmse)
        rmv_per_bin.append(rmv)
        bin_weights.append(n_in_bin)
    
    rmse_arr = np.array(rmse_per_bin)
    rmv_arr  = np.array(rmv_per_bin)
    weights  = np.array(bin_weights, dtype=float)
    weights /= weights.sum()
    
    ence = np.sum(weights * np.abs(rmse_arr - rmv_arr) / (rmv_arr + 1e-10))

    # ── Risk-Coverage Curve ──
    # Sort by variance descending, compute cumulative MAE
    sort_idx = np.argsort(-variance)  # highest var first
    sorted_ae = abs_error[sort_idx]
    cumulative_mae = np.cumsum(sorted_ae) / (np.arange(len(sorted_ae)) + 1)
    coverage = np.linspace(0, 1, len(sorted_ae))

    return {
        'mean_pred': mean_pred,
        'variance': variance,
        'abs_error': abs_error,
        'pearson_r': r,
        'pearson_p': p,
        'spearman_r': sr,
        'spearman_p': sp,
        'ence': ence,
        'rmse_per_bin': rmse_arr.tolist(),
        'rmv_per_bin': rmv_arr.tolist(),
        'env_bins': bin_edges.tolist(),
        'risk_coverage': {
            'coverage': coverage.tolist(),
            'risk': cumulative_mae.tolist(),
        },
        'variance_mean': float(variance.mean()),
        'variance_std': float(variance.std()),
    }


# ═══════════════════════════════════════════════════════════
# 6. Plotting
# ═══════════════════════════════════════════════════════════

def plot_uq_diagnostics(metrics, fold_name, out_dir):
    """Generate 3-panel UQ diagnostic plot for one fold."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Variance vs Abs Error scatter
    ax = axes[0]
    ax.scatter(metrics['variance'], metrics['abs_error'], alpha=0.4, s=8, c='steelblue', edgecolors='none')
    ax.set_xlabel('Ensemble Variance')
    ax.set_ylabel('Absolute Error (eV)')
    ax.set_title(f'{fold_name}: r={metrics["pearson_r"]:.3f} (p={metrics["pearson_p"]:.4f})')
    # Add trend line
    if len(metrics['variance']) > 1:
        z = np.polyfit(metrics['variance'], metrics['abs_error'], 1)
        x_line = np.linspace(metrics['variance'].min(), metrics['variance'].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=1.5)
    ax.grid(alpha=0.3)
    
    # Panel 2: Calibration (RMSE vs RMV per bin)
    ax = axes[1]
    rmse = np.array(metrics['rmse_per_bin'])
    rmv  = np.array(metrics['rmv_per_bin'])
    ax.plot(rmv, rmse, 'o-', color='darkorange', markersize=8, label='Binned')
    max_val = max(rmv.max(), rmse.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Ideal')
    ax.set_xlabel('RMV (Root Mean Variance)')
    ax.set_ylabel('RMSE (Root Mean Squared Error)')
    ax.set_title(f'Calibration (ENCE={metrics["ence"]:.3f})')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 3: Risk-Coverage Curve
    ax = axes[2]
    rc = metrics['risk_coverage']
    coverage = np.array(rc['coverage'])
    risk = np.array(rc['risk'])
    ax.plot(coverage, risk, color='forestgreen', linewidth=2)
    ax.fill_between(coverage, risk, alpha=0.15, color='forestgreen')
    ax.set_xlabel('Fraction of Samples Covered (lowest var first)')
    ax.set_ylabel('Cumulative Mean Risk (MAE, eV)')
    ax.set_title('Risk-Coverage Curve')
    ax.grid(alpha=0.3)
    # Invert: samples covered from low var
    ax.invert_xaxis()
    
    plt.tight_layout()
    path = os.path.join(out_dir, f'uq_diagnostics_{fold_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ═══════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('Phase 3: Ensemble UQ Evaluation — PAA-SchNet')
    print('=' * 60)

    # Load all data
    print('\n[1/4] Loading all .pt files ...')
    t0 = time.time()
    all_samples = load_all_pt_files(DATASET_DIR)
    systems = sorted(set(s['system_id'] for s in all_samples))
    print(f'  Loaded {len(all_samples)} samples, {len(systems)} systems: {systems}')

    # ============================================================
    # Part A: IID Ensemble
    # ============================================================
    print('\n' + '=' * 60)
    print('Part A: IID Ensemble (Random Split)')
    print('=' * 60)
    
    random.seed(42)
    np.random.seed(42)
    indices = list(range(len(all_samples)))
    random.shuffle(indices)
    n_train = int(len(all_samples) * 0.80)
    n_val   = int(len(all_samples) * 0.10)
    iid_train = [all_samples[i] for i in indices[:n_train]]
    iid_val   = [all_samples[i] for i in indices[n_train:n_train+n_val]]
    iid_test  = [all_samples[i] for i in indices[n_train+n_val:]]
    print(f'  Train: {len(iid_train)}, Val: {len(iid_val)}, Test: {len(iid_test)}')
    
    print('  Training IID ensemble ...')
    iid_preds, iid_true = train_ensemble_and_predict(
        iid_train, iid_val, iid_test, SEEDS, DEVICE, 'iid', OUTPUT_DIR)
    
    print('  Computing IID UQ metrics ...')
    iid_metrics = compute_uq_metrics(iid_preds, iid_true)
    print(f'    Pearson r = {iid_metrics["pearson_r"]:.4f} (p={iid_metrics["pearson_p"]:.4f})')
    print(f'    Spearman r = {iid_metrics["spearman_r"]:.4f}')
    print(f'    ENCE = {iid_metrics["ence"]:.4f}')
    print(f'    Var mean = {iid_metrics["variance_mean"]:.6f}')
    plot_uq_diagnostics(iid_metrics, 'IID', OUTPUT_DIR)

    # ============================================================
    # Part B: LOSO Ensemble (per fold)
    # ============================================================
    print('\n' + '=' * 60)
    print('Part B: LOSO Ensemble (per fold)')
    print('=' * 60)

    from torch.utils.data import DataLoader

    loso_metrics_all = {}  # system -> metrics dict
    loso_aggregated_var = []
    loso_aggregated_ae  = []

    for fold_idx, test_system in enumerate(systems):
        print(f'\n  Fold {fold_idx + 1}/{len(systems)}: Leave out [{test_system}]')

        test_samples  = [s for s in all_samples if s['system_id'] == test_system]
        train_val_samples = [s for s in all_samples if s['system_id'] != test_system]

        random.seed(42)
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

        # Train ensemble
        fold_preds, fold_true = train_ensemble_and_predict(
            train_samples, val_samples, test_samples, SEEDS, DEVICE, test_system, OUTPUT_DIR)

        # Compute UQ metrics
        metrics = compute_uq_metrics(fold_preds, fold_true)
        loso_metrics_all[test_system] = metrics
        loso_aggregated_var.append(metrics['variance'])
        loso_aggregated_ae.append(metrics['abs_error'])

        print(f'    Pearson r = {metrics["pearson_r"]:.4f} (p={metrics["pearson_p"]:.4f})')
        print(f'    Spearman r = {metrics["spearman_r"]:.4f}')
        print(f'    ENCE = {metrics["ence"]:.4f}')
        print(f'    Var mean = {metrics["variance_mean"]:.6f}')
        plot_uq_diagnostics(metrics, test_system, OUTPUT_DIR)

    # ============================================================
    # Part C: Global Comparison (IID vs LOSO)
    # ============================================================
    print('\n' + '=' * 60)
    print('Part C: UQ Comparison — IID vs LOSO')
    print('=' * 60)

    # Aggregate all LOSO predictions
    all_loso_var = np.concatenate(loso_aggregated_var)
    all_loso_ae  = np.concatenate(loso_aggregated_ae)

    # Combined LOSO metrics
    loso_global_r, loso_global_p = pearsonr(all_loso_var, all_loso_ae)
    from scipy.stats import spearmanr
    loso_global_sr, loso_global_sp = spearmanr(all_loso_var, all_loso_ae)

    print(f'\n  {"":>20} {"IID":>12} {"LOSO":>12}')
    print(f'  {"-"*44}')
    print(f'  {"Pearson r":>20} {iid_metrics["pearson_r"]:>12.4f} {loso_global_r:>12.4f}')
    print(f'  {"Spearman r":>20} {iid_metrics["spearman_r"]:>12.4f} {loso_global_sr:>12.4f}')
    print(f'  {"ENCE":>20} {iid_metrics["ence"]:>12.4f} {"per-fold avg":>12}')
    print(f'  {"Var Mean":>20} {iid_metrics["variance_mean"]:>12.6f} {all_loso_var.mean():>12.6f}')
    print(f'  {"Var Std":>20} {iid_metrics["variance_std"]:>12.6f} {all_loso_var.std():>12.6f}')

    # ── Per-system summary table ──
    print(f'\n  {"System":<12} {"Pearson r":>10} {"p-value":>10} {"Spearman r":>12} {"ENCE":>8} {"Var Mean":>12}')
    print(f'  {"-"*64}')
    for sys_id in sorted(loso_metrics_all.keys()):
        m = loso_metrics_all[sys_id]
        print(f'  {sys_id:<12} {m["pearson_r"]:>10.4f} {m["pearson_p"]:>10.4f} '
              f'{m["spearman_r"]:>12.4f} {m["ence"]:>8.4f} {m["variance_mean"]:>12.6f}')

    # ── Save all metrics to JSON ──
    results_json = {
        'iid': {
            'pearson_r': float(iid_metrics['pearson_r']),
            'pearson_p': float(iid_metrics['pearson_p']),
            'spearman_r': float(iid_metrics['spearman_r']),
            'ence': float(iid_metrics['ence']),
            'variance_mean': float(iid_metrics['variance_mean']),
            'variance_std': float(iid_metrics['variance_std']),
        },
        'loso_global': {
            'pearson_r': float(loso_global_r),
            'pearson_p': float(loso_global_p),
            'spearman_r': float(loso_global_sr),
            'variance_mean': float(all_loso_var.mean()),
            'variance_std': float(all_loso_var.std()),
        },
        'loso_per_system': {}
    }
    for sys_id in sorted(loso_metrics_all.keys()):
        m = loso_metrics_all[sys_id]
        results_json['loso_per_system'][sys_id] = {
            'pearson_r': float(m['pearson_r']),
            'pearson_p': float(m['pearson_p']),
            'spearman_r': float(m['spearman_r']),
            'ence': float(m['ence']),
            'variance_mean': float(m['variance_mean']),
            'variance_std': float(m['variance_std']),
        }

    json_path = os.path.join(OUTPUT_DIR, 'uq_metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f'\n  Metrics JSON saved to: {json_path}')

    # ── IID vs LOSO comparison bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: IID vs LOSO scatter overlay
    ax = axes[0]
    # Downsample for visualization
    n_iid = min(2000, len(iid_metrics['variance']))
    idx_iid = np.random.choice(len(iid_metrics['variance']), n_iid, replace=False)
    ax.scatter(iid_metrics['variance'][idx_iid], iid_metrics['abs_error'][idx_iid],
               alpha=0.5, s=10, label=f'IID (r={iid_metrics["pearson_r"]:.3f})', c='steelblue')
    n_loso = min(2000, len(all_loso_var))
    idx_loso = np.random.choice(len(all_loso_var), n_loso, replace=False)
    ax.scatter(all_loso_var[idx_loso], all_loso_ae[idx_loso],
               alpha=0.5, s=10, label=f'LOSO (r={loso_global_r:.3f})', c='darkorange')
    ax.set_xlabel('Ensemble Variance')
    ax.set_ylabel('Absolute Error (eV)')
    ax.set_title('IID vs LOSO: Variance-Error Scatter')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Risk-Coverage curves for IID and LOSO
    ax = axes[1]
    for label, m, color in [('IID', iid_metrics, 'steelblue'), ('LOSO', {'risk_coverage': {'coverage': np.linspace(0,1,len(all_loso_var)).tolist(), 'risk': (np.cumsum(all_loso_ae[np.argsort(-all_loso_var)]) / (np.arange(len(all_loso_ae))+1)).tolist()}}, 'darkorange')]:
        rc = m['risk_coverage']
        ax.plot(np.array(rc['coverage']), np.array(rc['risk']), color=color, linewidth=2, label=label)
    ax.set_xlabel('Fraction of Samples Covered (lowest var first)')
    ax.set_ylabel('Cumulative Mean MAE (eV)')
    ax.set_title('Risk-Coverage Curve: IID vs LOSO')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    cmp_path = os.path.join(OUTPUT_DIR, 'uq_iid_vs_loso_comparison.png')
    plt.savefig(cmp_path, dpi=150)
    plt.close()
    print(f'  Comparison plot saved to: {cmp_path}')

    print('\n' + '=' * 60)
    print('Phase 3 Complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
