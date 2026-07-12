# -*- coding: utf-8 -*-
"""
Phase 4.2: UQ-Gated Stacking — PAA-SchNet + PAA-PaiNN ensemble
Workspace: D:\lunwen\2.1sci\phase 4\

LOSO 7-fold. Each fold:
  - 3 PAA-SchNet (seeds 42, 123, 456)
  - 3 PAA-PaiNN (seeds 42, 123, 456)
  - K=5 group OOF predictions on training set
  - Gating network (MLP) trained on OOF predictions
  - Evaluate: best single / simple ensemble mean / gated stacking

Contrasts:
  - Stacking value mainly in tail (MAE_95), not mean
"""
import os, sys, time, random, copy, json
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
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

# ── Config ──────────────────────────────────────────────
DATASET_DIR   = os.environ.get('PAA_DATASET_DIR', r'D:\lunwen\2.1sci\phase 0\dataset\processed')
DEFAULT_OUTPUT_ROOT = r'D:\lunwen\2.1sci\phase 4\uq_gated_stacking_output'
OUTPUT_DIR_ENV = os.environ.get('PAA_OUTPUT_DIR')
ALLOW_OVERWRITE = os.environ.get('PAA_ALLOW_OVERWRITE', '0').lower() in ('1', 'true', 'yes')

if OUTPUT_DIR_ENV:
    OUTPUT_DIR = OUTPUT_DIR_ENV
elif os.path.exists(DEFAULT_OUTPUT_ROOT) and os.listdir(DEFAULT_OUTPUT_ROOT) and not ALLOW_OVERWRITE:
    run_stamp = time.strftime('%Y%m%d_%H%M%S')
    OUTPUT_DIR = f'{DEFAULT_OUTPUT_ROOT}_{run_stamp}'
else:
    OUTPUT_DIR = DEFAULT_OUTPUT_ROOT
PAA_SCHNET_DIR = r'D:\lunwen\2.1sci\phase 2\loso_paa_schnet_output'
PHASE4_1_DIR  = r'D:\lunwen\2.1sci\phase 4\oof_stacking_output'
PHASE4_1_DIR  = r'D:\lunwen\2.1sci\phase 4\oof_stacking_output'
RANDOM_SEED   = 42
BATCH_SIZE_SCHNET = 64
BATCH_SIZE_PAINN  = 32
EPOCHS        = 150
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
FORCE_WEIGHT  = 0.5
CUTOFF        = 5.0
N_RBF         = 20
HIDDEN_DIM    = 128
N_INTERACTIONS = 3
EARLY_STOP    = 30
PHYS_DIM      = 5
EDGE_PHYS_DIM = 5

# Gate network config
GATE_HIDDEN   = 64
GATE_EPOCHS   = 200
GATE_LR       = 1e-3
GATE_PATIENCE = 30

# OOF
K_OOF         = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)
if os.path.abspath(OUTPUT_DIR) != os.path.abspath(DEFAULT_OUTPUT_ROOT):
    print(f'Output protection active. Writing this run to: {OUTPUT_DIR}')
elif ALLOW_OVERWRITE:
    print(f'Overwrite explicitly allowed for output dir: {OUTPUT_DIR}')

SEEDS_SCHNET = [42, 123, 456]
SEEDS_PAINN  = [42, 123, 456]
N_MODELS = len(SEEDS_SCHNET) + len(SEEDS_PAINN)  # 6
N_BASE_MODELS = N_MODELS
DISABLE_BIJ = os.environ.get('PAA_DISABLE_BIJ', '0').lower() in ('1', 'true', 'yes')

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f'DEVICE: {DEVICE}')
print(f'N_MODELS: {N_MODELS} (3 PAA-SchNet + 3 PAA-PaiNN)')
print(f'K_OOF: {K_OOF}')
print(f'DATASET_DIR: {DATASET_DIR}')
print(f'OUTPUT_DIR: {OUTPUT_DIR}')
print(f'PAA_DISABLE_BIJ: {DISABLE_BIJ}')


# ═══════════════════════════════════════════════════════════
# 1. PAA-SchNet Model (unchanged from Phase 2.2)
# ═══════════════════════════════════════════════════════════

class RBFExpansion(nn.Module):
    def __init__(self, cutoff, n_rbf):
        super().__init__()
        self.cutoff = cutoff
        self.centers = nn.Parameter(torch.linspace(0.0, cutoff, n_rbf), requires_grad=False)
        self.gamma = 0.5 / ((self.centers[1] - self.centers[0]) ** 2 + 1e-8)

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
            nn.Linear(edge_phys_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
    def forward(self, edge_diffs):
        if DISABLE_BIJ:
            return torch.ones(edge_diffs.size(0), device=edge_diffs.device, dtype=edge_diffs.dtype)
        return self.net(edge_diffs).squeeze(-1)


class SchNetPAAInteraction(nn.Module):
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
        messages = x[edge_src] * filters * gate.unsqueeze(-1)
        aggregated = torch.zeros(num_nodes, x.size(-1), device=x.device)
        aggregated = aggregated.index_add(0, edge_dst, messages)
        x = x + self.out_net(self.atom_net(x) + aggregated)
        return x


class PAASchNet(nn.Module):
    def __init__(self, n_atom_types=100, phys_dim=PHYS_DIM, edge_phys_dim=EDGE_PHYS_DIM,
                 hidden_dim=128, n_rbf=20, n_interactions=3, cutoff=5.0):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.phys_proj = nn.Linear(phys_dim, hidden_dim)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.interactions = nn.ModuleList([
            SchNetPAAInteraction(hidden_dim, n_rbf, edge_phys_dim) for _ in range(n_interactions)
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
# 2. PAA-PaiNN Model (new: phys + gate on PaiNN)
# ═══════════════════════════════════════════════════════════

class PaiNNInteraction(nn.Module):
    """PaiNN message block with edge gate on scalar and vector messages."""
    def __init__(self, hidden_dim, n_rbf, edge_phys_dim):
        super().__init__()
        D = hidden_dim
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, 3 * D), nn.SiLU(), nn.Linear(3 * D, 3 * D),
        )
        self.edge_gate = EdgeBiasGate(edge_phys_dim)

    def forward(self, s, v, edge_src, edge_dst, rbf, dir_vec, edge_diffs, num_nodes):
        D = s.size(-1)
        filters = self.filter_net(rbf)  # (E, 3D)
        W_ss = filters[:, :D]
        W_sv = filters[:, D:2*D]
        W_vv = filters[:, 2*D:3*D]

        gate = self.edge_gate(edge_diffs)  # (E,)

        s_src = s[edge_src]
        v_src = v[edge_src]

        v_dot_dir = torch.einsum('ejd,ej->ed', v_src, dir_vec)
        msg_s = s_src * W_ss + v_dot_dir * W_sv
        msg_s = msg_s * gate.unsqueeze(-1)  # gate scalar message

        ds = torch.zeros(num_nodes, D, device=s.device)
        ds = ds.index_add(0, edge_dst, msg_s)

        msg_v1 = torch.einsum('ed,ej->ejd', s_src * W_sv, dir_vec)
        msg_v2 = v_src * W_vv.unsqueeze(1)
        msg_v = (msg_v1 + msg_v2) * gate.unsqueeze(-1).unsqueeze(-1)  # gate vector message

        dv = torch.zeros(num_nodes, 3, D, device=s.device)
        dv = dv.index_add(0, edge_dst, msg_v)

        return s + ds, v + dv


class PaiNNUpdate(nn.Module):
    """PaiNN update block (unchanged, no edge info)."""
    def __init__(self, hidden_dim):
        super().__init__()
        D = hidden_dim
        self.U = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 3 * D))
        self.V = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 2 * D))

    def forward(self, s, v):
        D = s.size(-1)
        U_out = self.U(s)
        a_ss = U_out[:, :D]
        a_sv = U_out[:, D:2*D]
        a_vv = U_out[:, 2*D:3*D]
        v_norm = torch.linalg.norm(v, dim=1)
        ds = a_ss + a_sv * v_norm

        V_out = self.V(v_norm)
        a_vv_new = V_out[:, :D]
        a_vs = V_out[:, D:]
        dv = v * a_vv_new.unsqueeze(1) + s.unsqueeze(1) * a_vs.unsqueeze(1)

        return s + ds, v + dv


class PAAPaiNN(nn.Module):
    def __init__(self, n_atom_types=100, phys_dim=PHYS_DIM, edge_phys_dim=EDGE_PHYS_DIM,
                 hidden_dim=128, n_rbf=20, n_interactions=3, cutoff=5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.phys_proj = nn.Linear(phys_dim, hidden_dim)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.interactions = nn.ModuleList([
            PaiNNInteraction(hidden_dim, n_rbf, edge_phys_dim) for _ in range(n_interactions)
        ])
        self.updates = nn.ModuleList([
            PaiNNUpdate(hidden_dim) for _ in range(n_interactions)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z, pos, edge_src, edge_dst, phys_feats=None, batch=None):
        N = z.size(0)
        D = self.hidden_dim
        s = self.embedding(z)
        if phys_feats is not None:
            s = s + self.phys_proj(phys_feats)
        v = torch.zeros(N, 3, D, device=z.device)

        vec = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(vec, dim=-1)
        rbf = self.rbf(dist)
        dir_vec = vec / (dist.unsqueeze(-1) + 1e-10)

        if phys_feats is not None:
            edge_diffs = torch.abs(phys_feats[edge_src] - phys_feats[edge_dst])
        else:
            edge_diffs = torch.zeros(dist.size(0), PHYS_DIM, device=z.device)

        for interaction, update in zip(self.interactions, self.updates):
            s, v = interaction(s, v, edge_src, edge_dst, rbf, dir_vec, edge_diffs, N)
            s, v = update(s, v)

        atom_energies = self.readout(s)
        if batch is not None:
            n_mols = batch.max().item() + 1
            energy = torch.zeros(n_mols, 1, device=z.device)
            energy = energy.index_add(0, batch, atom_energies)
        else:
            energy = atom_energies.sum(0, keepdim=True)
        return energy.squeeze(-1), s


# ═══════════════════════════════════════════════════════════
# 3. Data Loading (unchanged)
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
                'sample_id': os.path.splitext(fname)[0].replace('_sample', ''),
                'pt_path': fpath,
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
# 4. Training helpers
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


def train_one_model(model, train_loader, val_loader, device, force_weight, epochs, early_stop):
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


def predict_energy(model, loader, device):
    """Return per-sample energy predictions (numpy arrays)."""
    model.eval()
    all_preds, all_true = [], []
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
# 5. OOF prediction generation
# ═══════════════════════════════════════════════════════════

def generate_oof_predictions(model_class, model_kwargs, train_samples, val_samples,
                              seeds, batch_size, device, force_weight, epochs, early_stop, K):
    """
    For a single model class, generate OOF predictions for training set.
    train_samples + val_samples together form the training pool for OOF.
    Only train_samples get OOF predictions (val is for early stopping per fold).
    Returns:
        oof_preds: (N_train, len(seeds)) numpy array
        test_models: list of trained models (one per seed, trained on ALL training data)
    """
    from torch.utils.data import DataLoader

    # Group-aware K-fold split
    # Build groups from train_samples
    groups = defaultdict(list)
    for s in train_samples:
        groups[(s['system_id'], s['group_id'])].append(s)
    group_keys = sorted(groups.keys())
    random.shuffle(group_keys)

    n_groups = len(group_keys)
    fold_size = max(1, n_groups // K)
    oof_preds_list = [[] for _ in range(len(seeds))]

    sample_to_idx = {id(s): i for i, s in enumerate(train_samples)}
    for seed_idx, seed in enumerate(seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f'      Seed {seed} ({seed_idx+1}/{len(seeds)}) OOF ...', end=' ', flush=True)
        t0 = time.time()

        # For each OOF fold
        fold_predictions = [None] * len(train_samples)  # placeholder

        for fold_k in range(K):
            # Split groups
            val_fold_keys = set(group_keys[fold_k * fold_size:(fold_k + 1) * fold_size])
            train_fold_keys = [k for k in group_keys if k not in val_fold_keys]

            fold_train = [s for s in train_samples if (s['system_id'], s['group_id']) in set(train_fold_keys)]
            fold_val   = [s for s in train_samples if (s['system_id'], s['group_id']) in val_fold_keys]

            if len(fold_val) == 0:
                continue

            train_loader = DataLoader(fold_train, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_batch, drop_last=True)
            val_loader_oof = DataLoader(fold_val, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

            model = model_class(**model_kwargs).to(device)
            model = train_one_model(model, train_loader, val_loader_oof, device, force_weight, epochs, early_stop)

            preds, _ = predict_energy(model, val_loader_oof, device)

            # Map predictions back to original indices
            for i, s in enumerate(fold_val):
                orig_idx = sample_to_idx[id(s)]
                fold_predictions[orig_idx] = preds[i]

        # Fill any remaining None values with the mean prediction of this seed
        valid_vals = [v for v in fold_predictions if v is not None]
        fill_val = np.mean(valid_vals) if valid_vals else 0.0
        fold_predictions_filled = [v if v is not None else fill_val for v in fold_predictions]
        oof_preds_arr = np.array(fold_predictions_filled, dtype=np.float32)
        oof_preds_list[seed_idx] = oof_preds_arr

        print(f'done ({time.time()-t0:.0f}s)')

    oof_preds = np.stack(oof_preds_list, axis=1)  # (N_train, N_seeds)
    return oof_preds


def train_final_models(model_class, model_kwargs, train_samples, val_samples,
                        seeds, batch_size, device, force_weight, epochs, early_stop):
    """Train final models on ALL training data (no OOF) for test prediction."""
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_batch, drop_last=True)
    val_loader   = DataLoader(val_samples, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    final_models = []
    sample_to_idx = {id(s): i for i, s in enumerate(train_samples)}
    for seed_idx, seed in enumerate(seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f'      Final model seed {seed} ({seed_idx+1}/{len(seeds)}) ...', end=' ', flush=True)
        t0 = time.time()
        model = model_class(**model_kwargs).to(device)
        model = train_one_model(model, train_loader, val_loader, device, force_weight, epochs, early_stop)
        final_models.append(model)
        print(f'done ({time.time()-t0:.0f}s)')

    return final_models


# ═══════════════════════════════════════════════════════════
# 6. Gating Network
# ═══════════════════════════════════════════════════════════

class GatingNetwork(nn.Module):
    """
    Input: (N_models,) base model predictions
    Output: (N_models,) softmax weights
    Final prediction: sum(w_i * E_i)
    """
    def __init__(self, n_models, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_models, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, n_models),
            nn.Softmax(dim=-1)
        )

    def forward(self, base_preds):
        """
        base_preds: (N_samples, n_models)
        Returns:
            weights: (N_samples, n_models)
            final_pred: (N_samples,)
        """
        weights = self.net(base_preds)  # (N, n_models)
        final_pred = (weights * base_preds).sum(dim=-1)  # (N,)
        return weights, final_pred


# ═══════════════════════════════════════════════════════════
# 7. UQ-Gated Network (7-dim input: 6 preds + 1 variance)
# ═══════════════════════════════════════════════════════════

class UQGatingNetwork(nn.Module):
    """Input: (N, n_models+1) = 6 base preds + 1 variance feature."""
    def __init__(self, n_models=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_models + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, n_models),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, base_preds, uq_var):
        if uq_var.dim() == 1:
            uq_var = uq_var.unsqueeze(-1)
        x = torch.cat([base_preds, uq_var], dim=-1)
        weights = self.net(x)
        final_pred = (weights * base_preds).sum(dim=-1)
        return weights, final_pred


# ═══════════════════════════════════════════════════════════
# 8. Main LOSO Loop
# ═══════════════════════════════════════════════════════════

def main():
    global PHYS_DIM, EDGE_PHYS_DIM
    print('=' * 60)
    print('Phase 4.2: UQ-Gated Stacking')
    print('=' * 60)

    print('\n[1/5] Loading all .pt files ...')
    t0 = time.time()
    all_samples = load_all_pt_files(DATASET_DIR)
    if all_samples:
        inferred_phys_dim = int(all_samples[0]['phys_feats'].shape[1])
        if inferred_phys_dim != PHYS_DIM:
            print(f'  Inferred PHYS_DIM={inferred_phys_dim} from dataset; overriding config PHYS_DIM={PHYS_DIM}')
            PHYS_DIM = inferred_phys_dim
            EDGE_PHYS_DIM = inferred_phys_dim
    systems = sorted(set(s['system_id'] for s in all_samples))
    print(f'  Loaded {len(all_samples)} samples, {len(systems)} systems: {systems}')
    for sys_id in systems:
        print(f'    {sys_id}: {len([s for s in all_samples if s["system_id"]==sys_id])} samples')

    from torch.utils.data import DataLoader

    results = {}  # system -> {best_single, simple_mean, gated_stacking}
    uq_diagnostics = []

    for fold_idx, test_system in enumerate(systems):
        print(f'\n[2/5] Fold {fold_idx + 1}/{len(systems)}: Leave out [{test_system}]')
        print('=' * 60)

        # ── Split ──
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

        print(f'  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}')

        # ── 3a. Generate OOF predictions for SchNet ──
        print(f'\n  [3a] PAA-SchNet OOF predictions (K={K_OOF}) ...')
        t_oof = time.time()
        schnet_kwargs = dict(n_atom_types=100, phys_dim=PHYS_DIM, edge_phys_dim=EDGE_PHYS_DIM,
                             hidden_dim=HIDDEN_DIM, n_rbf=N_RBF, n_interactions=N_INTERACTIONS, cutoff=CUTOFF)
        oof_schnet = generate_oof_predictions(
            PAASchNet, schnet_kwargs, train_samples, val_samples,
            SEEDS_SCHNET, BATCH_SIZE_SCHNET, DEVICE, FORCE_WEIGHT, EPOCHS, EARLY_STOP, K_OOF
        )
        print(f'    OOF SchNet shape: {oof_schnet.shape}, time: {(time.time()-t_oof)/60:.1f} min')

        # ── 3b. Generate OOF predictions for PaiNN ──
        print(f'\n  [3b] PAA-PaiNN OOF predictions (K={K_OOF}) ...')
        t_oof = time.time()
        painn_kwargs = dict(n_atom_types=100, phys_dim=PHYS_DIM, edge_phys_dim=EDGE_PHYS_DIM,
                            hidden_dim=HIDDEN_DIM, n_rbf=N_RBF, n_interactions=N_INTERACTIONS, cutoff=CUTOFF)
        oof_painn = generate_oof_predictions(
            PAAPaiNN, painn_kwargs, train_samples, val_samples,
            SEEDS_PAINN, BATCH_SIZE_PAINN, DEVICE, FORCE_WEIGHT, EPOCHS, EARLY_STOP, K_OOF
        )
        print(f'    OOF PaiNN shape: {oof_painn.shape}, time: {(time.time()-t_oof)/60:.1f} min')

        # ── Combine OOF predictions ──
        oof_all = np.concatenate([oof_schnet, oof_painn], axis=1)  # (N_train, 6)
        train_y_true = np.array([s['y'] for s in train_samples])

        # ── Compute UQ feature: variance → winsorization → isotonic calibration ──
        oof_ensemble_mean = oof_all.mean(axis=1)
        oof_errors = np.abs(oof_ensemble_mean - train_y_true)
        oof_variance = np.var(oof_all, axis=1)
        print(f'\n  OOF var stats: mean={oof_variance.mean():.4f}, max={oof_variance.max():.4f}')
        # Winsorization: clip to 95th percentile
        var_p95 = np.percentile(oof_variance, 95)
        var_winsor = np.clip(oof_variance, 0, var_p95)
        print(f'  Winsorized var max: {var_p95:.4f}')
        # Isotonic regression: calibrate var → monotonic with error
        iso = IsotonicRegression(out_of_bounds='clip')
        var_calibrated = iso.fit_transform(var_winsor, oof_errors)
        # Spearman rank correlation
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(var_calibrated, oof_errors)
        print(f'  Calibrated var-err Spearman: {spearman:.4f}')
        # StandardScaler on calibrated variance for Gate input
        var_scaler = StandardScaler()
        oof_var_scaled = var_scaler.fit_transform(var_calibrated.reshape(-1, 1)).ravel()

        # ── 3c. Train gating network (no UQ) on OOF predictions ──
        X_gate = torch.tensor(oof_all, dtype=torch.float32).to(DEVICE)
        y_gate = torch.tensor(train_y_true, dtype=torch.float32).to(DEVICE)
        # Gate train/val split (80/20)
        n_gate = len(X_gate)
        idx_gate = list(range(n_gate))
        random.shuffle(idx_gate)
        n_gate_train = int(n_gate * 0.80)
        X_gate_train = X_gate[idx_gate[:n_gate_train]]
        y_gate_train = y_gate[idx_gate[:n_gate_train]]
        X_gate_val   = X_gate[idx_gate[n_gate_train:]]
        y_gate_val   = y_gate[idx_gate[n_gate_train:]]
        UQ_gate_train = torch.tensor(oof_var_scaled[idx_gate[:n_gate_train]], dtype=torch.float32).to(DEVICE)
        UQ_gate_val   = torch.tensor(oof_var_scaled[idx_gate[n_gate_train:]], dtype=torch.float32).to(DEVICE)

        print('\n  [3c] Training Gate (no UQ) ... ', end='', flush=True)
        gate_no_uq = GatingNetwork(N_MODELS, hidden=GATE_HIDDEN).to(DEVICE)
        opt_no_uq = Adam(gate_no_uq.parameters(), lr=GATE_LR, weight_decay=1e-5)
        sched_no_uq = ReduceLROnPlateau(opt_no_uq, mode='min', factor=0.5, patience=15, min_lr=1e-6)
        best_loss_no_uq = float('inf'); best_state_no_uq = None; pat_no_uq = 0
        for ep in range(1, GATE_EPOCHS + 1):
            gate_no_uq.train(); opt_no_uq.zero_grad()
            _, pred = gate_no_uq(X_gate_train)
            loss = F.l1_loss(pred, y_gate_train)
            loss.backward(); opt_no_uq.step()
            gate_no_uq.eval()
            with torch.no_grad():
                _, pred_val = gate_no_uq(X_gate_val)
                val_loss = F.l1_loss(pred_val, y_gate_val)
            sched_no_uq.step(val_loss)
            if val_loss.item() < best_loss_no_uq:
                best_loss_no_uq = val_loss.item(); best_state_no_uq = copy.deepcopy(gate_no_uq.state_dict()); pat_no_uq = 0
            else:
                pat_no_uq += 1
            if pat_no_uq >= GATE_PATIENCE: break
        gate_no_uq.load_state_dict(best_state_no_uq)
        print(f' done, best val loss={best_loss_no_uq:.4f}')

        print('  [3c] Training Gate (+UQ) ... ', end='', flush=True)
        gate_uq = UQGatingNetwork(n_models=N_BASE_MODELS, hidden=GATE_HIDDEN).to(DEVICE)
        opt_uq = Adam(gate_uq.parameters(), lr=GATE_LR, weight_decay=1e-5)
        sched_uq = ReduceLROnPlateau(opt_uq, mode='min', factor=0.5, patience=15, min_lr=1e-6)
        best_loss_uq = float('inf'); best_state_uq = None; pat_uq = 0
        for ep in range(1, GATE_EPOCHS + 1):
            gate_uq.train(); opt_uq.zero_grad()
            _, pred = gate_uq(X_gate_train, UQ_gate_train)
            loss = F.l1_loss(pred, y_gate_train)
            loss.backward(); opt_uq.step()
            gate_uq.eval()
            with torch.no_grad():
                _, pred_val = gate_uq(X_gate_val, UQ_gate_val)
                val_loss = F.l1_loss(pred_val, y_gate_val)
            sched_uq.step(val_loss)
            if val_loss.item() < best_loss_uq:
                best_loss_uq = val_loss.item(); best_state_uq = copy.deepcopy(gate_uq.state_dict()); pat_uq = 0
            else:
                pat_uq += 1
            if pat_uq >= GATE_PATIENCE: break
        gate_uq.load_state_dict(best_state_uq)
        print(f' done, best val loss={best_loss_uq:.4f}')

        # ── 3d. Train final models (on all training data) ──
        print(f'\n  [3d] Training final models on all training data ...')
        final_models = []

        print('    PAA-SchNet final models ...')
        schnet_final = train_final_models(
            PAASchNet, schnet_kwargs, train_samples, val_samples,
            SEEDS_SCHNET, BATCH_SIZE_SCHNET, DEVICE, FORCE_WEIGHT, EPOCHS, EARLY_STOP
        )
        final_models.extend(schnet_final)

        print('    PAA-PaiNN final models ...')
        painn_final = train_final_models(
            PAAPaiNN, painn_kwargs, train_samples, val_samples,
            SEEDS_PAINN, BATCH_SIZE_PAINN, DEVICE, FORCE_WEIGHT, EPOCHS, EARLY_STOP
        )
        final_models.extend(painn_final)

        # ── 3e. Predict on test set ──
        print(f'\n  [3e] Predicting on test set ({len(test_samples)} samples) ...')
        test_loader = DataLoader(test_samples, batch_size=BATCH_SIZE_SCHNET, shuffle=False, collate_fn=collate_batch)

        test_preds_all = []
        for m_idx, model in enumerate(final_models):
            preds, true = predict_energy(model, test_loader, DEVICE)
            test_preds_all.append(preds)
        test_preds_all = np.stack(test_preds_all, axis=1)  # (N_test, 6)
        test_y_true = true

               # ── Evaluate ──
        # Best single model
        best_single_mae = float('inf')
        best_single_mae_95 = 0.0
        best_single_idx = 0
        for m_idx in range(N_MODELS):
            mae = np.mean(np.abs(test_preds_all[:, m_idx] - test_y_true))
            ae = np.abs(test_preds_all[:, m_idx] - test_y_true)
            k = max(1, int(len(ae) * 0.05))
            mae_95 = np.mean(np.sort(ae)[-k:])
            if mae < best_single_mae:
                best_single_mae = mae
                best_single_mae_95 = mae_95
                best_single_idx = m_idx
        
        # Simple ensemble mean
        ensemble_mean = test_preds_all.mean(axis=1)
        ensemble_mae = np.mean(np.abs(ensemble_mean - test_y_true))
        ae_ens = np.abs(ensemble_mean - test_y_true)
        k = max(1, int(len(ae_ens) * 0.05))
        ensemble_mae_95 = np.mean(np.sort(ae_ens)[-k:])
        
        # Gated stacking (without UQ)
        gate_no_uq.eval()
        with torch.no_grad():
            X_test_gate = torch.tensor(test_preds_all, dtype=torch.float32).to(DEVICE)
            _, y_gated_no_uq = gate_no_uq(X_test_gate)
            y_gated_no_uq = y_gated_no_uq.cpu().numpy()
        gated_no_uq_mae = np.mean(np.abs(y_gated_no_uq - test_y_true))
        ae_gated_no_uq = np.abs(y_gated_no_uq - test_y_true)
        k = max(1, int(len(ae_gated_no_uq) * 0.05))
        gated_no_uq_mae_95 = np.mean(np.sort(ae_gated_no_uq)[-k:])
        
        # Gated stacking (with UQ)
        test_variance = np.var(test_preds_all, axis=1)
        test_var_winsor = np.clip(test_variance, 0, var_p95)
        test_var_calib = iso.transform(test_var_winsor)
        test_var_scaled = var_scaler.transform(test_var_calib.reshape(-1, 1)).ravel()
        X_test_uq = torch.tensor(test_var_scaled, dtype=torch.float32).to(DEVICE)
        
        gate_uq.eval()
        with torch.no_grad():
            _, y_gated_uq = gate_uq(X_test_gate, X_test_uq)
            y_gated_uq = y_gated_uq.cpu().numpy()
        gated_uq_mae = np.mean(np.abs(y_gated_uq - test_y_true))
        ae_gated_uq = np.abs(y_gated_uq - test_y_true)
        k = max(1, int(len(ae_gated_uq) * 0.05))
        gated_uq_mae_95 = np.mean(np.sort(ae_gated_uq)[-k:])

        from scipy.stats import spearmanr
        raw_spearman, _ = spearmanr(test_variance, ae_ens)
        calibrated_spearman, _ = spearmanr(test_var_calib, ae_ens)
        uq_top_k = max(1, int(len(test_variance) * 0.05))
        high_uq_idx = np.argsort(-test_variance)[:uq_top_k]
        high_uq_ae = ae_ens[high_uq_idx]
        high_uq_enrichment = float(high_uq_ae.mean() / (ae_ens.mean() + 1e-12))

        uq_diagnostics.append({
            'system': test_system,
            'raw_var_spearman_vs_ensemble_error': float(raw_spearman),
            'calibrated_var_spearman_vs_ensemble_error': float(calibrated_spearman),
            'top5pct_high_uq_abs_error_mean': float(high_uq_ae.mean()),
            'ensemble_abs_error_mean': float(ae_ens.mean()),
            'top5pct_high_uq_error_enrichment': high_uq_enrichment,
            'variance_mean': float(test_variance.mean()),
            'variance_max': float(test_variance.max()),
            'winsor_p95_from_oof': float(var_p95),
        })
        
        results[test_system] = {
            'best_single_mae': best_single_mae,
            'best_single_mae_95': best_single_mae_95,
            'ensemble_mae': ensemble_mae,
            'ensemble_mae_95': ensemble_mae_95,
            'gated_no_uq_mae': gated_no_uq_mae,
            'gated_no_uq_mae_95': gated_no_uq_mae_95,
            'gated_uq_mae': gated_uq_mae,
            'gated_uq_mae_95': gated_uq_mae_95
        }

        print(f'    Best Single:   MAE={best_single_mae:.4f}, MAE_95={best_single_mae_95:.4f}')
        print(f'    Ensemble Mean: MAE={ensemble_mae:.4f}, MAE_95={ensemble_mae_95:.4f}')
        print(f'    Gated No-UQ:   MAE={gated_no_uq_mae:.4f}, MAE_95={gated_no_uq_mae_95:.4f}')
        print(f'    Gated +UQ:     MAE={gated_uq_mae:.4f}, MAE_95={gated_uq_mae_95:.4f}')
        print(f'    UQ diagnostic: raw rho={raw_spearman:.4f}, calibrated rho={calibrated_spearman:.4f}, '
              f'top-5% enrichment={high_uq_enrichment:.2f}x')

        pred_csv = os.path.join(OUTPUT_DIR, f'phase4_2_fold_predictions_{test_system}.csv')
        model_names = [f'schnet_{s}' for s in SEEDS_SCHNET] + [f'painn_{s}' for s in SEEDS_PAINN]
        with open(pred_csv, 'w') as f:
            cols = ['sample_id', 'system_id', 'group_id', 'y_true']
            cols += [f'pred_{name}' for name in model_names]
            cols += [
                'pred_best_single', 'pred_ensemble_mean', 'pred_gate_no_uq', 'pred_gate_uq',
                'uq_raw_variance', 'uq_winsor_variance', 'uq_calibrated', 'uq_scaled',
                'abs_error_best_single', 'abs_error_ensemble', 'abs_error_gate_no_uq',
                'abs_error_gate_uq'
            ]
            f.write(','.join(cols) + '\n')
            for row_idx, s in enumerate(test_samples):
                row = [
                    s.get('sample_id', f'{test_system}_{row_idx}'),
                    s['system_id'],
                    str(s['group_id']),
                    f'{test_y_true[row_idx]:.10f}',
                ]
                row += [f'{test_preds_all[row_idx, j]:.10f}' for j in range(N_MODELS)]
                row += [
                    f'{test_preds_all[row_idx, best_single_idx]:.10f}',
                    f'{ensemble_mean[row_idx]:.10f}',
                    f'{y_gated_no_uq[row_idx]:.10f}',
                    f'{y_gated_uq[row_idx]:.10f}',
                    f'{test_variance[row_idx]:.10f}',
                    f'{test_var_winsor[row_idx]:.10f}',
                    f'{test_var_calib[row_idx]:.10f}',
                    f'{test_var_scaled[row_idx]:.10f}',
                    f'{abs(test_preds_all[row_idx, best_single_idx] - test_y_true[row_idx]):.10f}',
                    f'{ae_ens[row_idx]:.10f}',
                    f'{ae_gated_no_uq[row_idx]:.10f}',
                    f'{ae_gated_uq[row_idx]:.10f}',
                ]
                f.write(','.join(row) + '\n')
        print(f'    Fold predictions saved to: {pred_csv}')

        # Save fold artifacts
        torch.save(gate_uq.state_dict(), os.path.join(OUTPUT_DIR, f'gate_{test_system}.pt'))
        for i, model in enumerate(final_models):
            tag = f'schnet_{SEEDS_SCHNET[i]}' if i < 3 else f'painn_{SEEDS_PAINN[i-3]}'
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{test_system}_{tag}.pt'))

    # ═══════════════════════════════════════════════════════
    # 8. Summary — UQ vs no-UQ comparison
    # ═══════════════════════════════════════════════════════
    print('\n' + '=' * 78)
    print('PHASE 4.2 RESULTS: UQ-GATED vs NO-UQ GATE')
    print('=' * 78)
    sys_sorted = sorted(results.keys())

    # Load SchNet baseline for comparison
    bl_csv = r'D:\lunwen\2.1sci\phase 1\loso_schnet_output\loso_results.csv'
    bl = {}
    if os.path.exists(bl_csv):
        with open(bl_csv) as f:
            next(f)
            for line in f:
                p = line.strip().split(',')
                if len(p) >= 4:
                    bl[p[0]] = {'mae_e': float(p[1]), 'mae_95_e': float(p[2])}

    # Gate(noUQ) vs Gate(+UQ) comparison
    header = (f'{"System":<12} {"noUQ_MAE":>10} {"+UQ_MAE":>10} {"ΔMAE":>8} | '
              f'{"noUQ_95":>10} {"+UQ_95":>10} {"Δ95":>8}')
    print(header)
    print('-' * 78)

    all_noUQ = []; all_UQ = []
    all_noUQ_95 = []; all_UQ_95 = []

    for sys_id in sys_sorted:
        r = results[sys_id]
        nq = r['gated_no_uq_mae']
        uq = r['gated_uq_mae']
        nq95 = r['gated_no_uq_mae_95']
        uq95 = r['gated_uq_mae_95']
        dmae = nq - uq
        d95 = nq95 - uq95
        marker = '  WIN' if dmae > 0 else ''
        print(f'{sys_id:<12} {nq:>10.4f} {uq:>10.4f} {dmae:>+8.4f}{marker} | '
              f'{nq95:>10.4f} {uq95:>10.4f} {d95:>+8.4f}')
        all_noUQ.append(nq); all_UQ.append(uq)
        all_noUQ_95.append(nq95); all_UQ_95.append(uq95)

    print('-' * 78)
    print(f'{"Mean":<12} {np.mean(all_noUQ):>10.4f} {np.mean(all_UQ):>10.4f} '
          f'{np.mean(all_noUQ)-np.mean(all_UQ):>+8.4f} | '
          f'{np.mean(all_noUQ_95):>10.4f} {np.mean(all_UQ_95):>10.4f} '
          f'{np.mean(all_noUQ_95)-np.mean(all_UQ_95):>+8.4f}')

    # Delta vs SchNet baseline
    if bl:
        print(f'\n  Delta vs SchNet (Gate +UQ):')
        for sys_id in sys_sorted:
            if sys_id in bl:
                d_e = (bl[sys_id]['mae_e'] - results[sys_id]['gated_uq_mae']) / bl[sys_id]['mae_e'] * 100
                d_95 = (bl[sys_id]['mae_95_e'] - results[sys_id]['gated_uq_mae_95']) / bl[sys_id]['mae_95_e'] * 100
                print(f'    {sys_id:<12} ΔMAE(E): {d_e:+.1f}%, ΔMAE_95: {d_95:+.1f}%')

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'phase4_2_results.csv')
    with open(csv_path, 'w') as f:
        f.write('System,BestSingle_MAE,BestSingle_MAE95,Ensemble_MAE,Ensemble_MAE95,'
                'GateNoUQ_MAE,GateNoUQ_MAE95,GateUQ_MAE,GateUQ_MAE95\n')
        for sys_id in sys_sorted:
            r = results[sys_id]
            f.write(f'{sys_id},{r["best_single_mae"]:.6f},{r["best_single_mae_95"]:.6f},'
                    f'{r["ensemble_mae"]:.6f},{r["ensemble_mae_95"]:.6f},'
                    f'{r["gated_no_uq_mae"]:.6f},{r["gated_no_uq_mae_95"]:.6f},'
                    f'{r["gated_uq_mae"]:.6f},{r["gated_uq_mae_95"]:.6f}\n')
    print(f'\n  CSV saved to: {csv_path}')

    uq_csv_path = os.path.join(OUTPUT_DIR, 'phase4_2_uq_diagnostics.csv')
    with open(uq_csv_path, 'w') as f:
        cols = [
            'system', 'raw_var_spearman_vs_ensemble_error',
            'calibrated_var_spearman_vs_ensemble_error',
            'top5pct_high_uq_abs_error_mean', 'ensemble_abs_error_mean',
            'top5pct_high_uq_error_enrichment', 'variance_mean', 'variance_max',
            'winsor_p95_from_oof'
        ]
        f.write(','.join(cols) + '\n')
        for row in uq_diagnostics:
            f.write(','.join(str(row[c]) for c in cols) + '\n')
    print(f'  UQ diagnostics saved to: {uq_csv_path}')

    # Bar chart: noUQ vs +UQ
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(sys_sorted))
    width = 0.30

    for ax_idx, (key1, key2, title) in enumerate([
        ('gated_no_uq_mae', 'gated_uq_mae', 'Gate MAE: No-UQ vs +UQ'),
        ('gated_no_uq_mae_95', 'gated_uq_mae_95', 'Gate MAE_95: No-UQ vs +UQ')
    ]):
        ax = axes[ax_idx]
        b1 = [results[s][key1] for s in sys_sorted]
        b2 = [results[s][key2] for s in sys_sorted]
        b0 = [results[s]['best_single_mae'] for s in sys_sorted]
        ax.bar(x - width, b0, width, label='Best Single', color='#1f77b4')
        ax.bar(x, b1, width, label='Gate (no UQ)', color='#ff7f0e')
        ax.bar(x + width, b2, width, label='Gate (+UQ)', color='#2ca02c')
        ax.set_xticks(x); ax.set_xticklabels(sys_sorted, rotation=45)
        ax.set_title(title)
        if ax_idx == 0:
            ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, 'phase4_2_uq_vs_nouq.png')
    plt.savefig(bar_path, dpi=150)
    print(f'  Bar chart saved to: {bar_path}')

    print('\n' + '=' * 60)
    print('Phase 4.2 Complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
