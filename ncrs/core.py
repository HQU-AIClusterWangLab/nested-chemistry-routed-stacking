# -*- coding: utf-8 -*-
"""Core models and strict-LOSO utilities for NCRS.

The final experts are SchNet-static-phys, PAA-SchNet-coord, and
PaiNN-coord-bond. Mulliken quantities are deliberately excluded from model
inputs: they belong to post-DFT interpretation, not to inference.
"""
import copy
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


RANDOM_SEED = 42
SEEDS = [42, 123, 456]
BATCH_SIZE_SCHNET = 64
BATCH_SIZE_PAINN = 32
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-5
FORCE_WEIGHT = 0.5
CUTOFF = 5.0
N_RBF = 20
HIDDEN_DIM = 128
N_INTERACTIONS = 3
EARLY_STOP = 30
K_OOF = 5
GATE_MAX_ALPHA = 0.5


def seed_all(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        cutoff_val = 0.5 * (1.0 + torch.cos(math.pi * distances / self.cutoff))
        cutoff_val = torch.where(distances <= self.cutoff, cutoff_val, torch.zeros_like(cutoff_val))
        return rbf * cutoff_val.unsqueeze(-1)


class SchNetInteraction(nn.Module):
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
        return x + self.out_net(self.atom_net(x) + aggregated)


class SchNetPhys(nn.Module):
    def __init__(self, n_atom_types=100, phys_dim=5, hidden_dim=128, n_rbf=20,
                 n_interactions=3, cutoff=5.0):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.phys_proj = nn.Linear(phys_dim, hidden_dim)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim, n_rbf) for _ in range(n_interactions)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z, pos, edge_src, edge_dst, phys_feats=None, batch=None):
        n_nodes = z.size(0)
        x = self.embedding(z) + self.phys_proj(phys_feats)
        vec = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(vec, dim=-1)
        rbf = self.rbf(dist)
        for interaction in self.interactions:
            x = interaction(x, edge_src, edge_dst, rbf, n_nodes)
        atom_energies = self.readout(x)
        return pool_atom_energies(atom_energies, batch)


class EdgeBiasGate(nn.Module):
    def __init__(self, edge_bias_dim, hidden=64, max_alpha=GATE_MAX_ALPHA):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_bias_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.max_alpha = max_alpha

    def forward(self, edge_bias):
        gate = torch.sigmoid(self.net(edge_bias)).squeeze(-1)
        alpha = self.max_alpha * torch.sigmoid(self.raw_alpha)
        scale = 1.0 + alpha * (2.0 * gate - 1.0)
        return scale


class PAASchNetInteraction(nn.Module):
    def __init__(self, hidden_dim, n_rbf, edge_bias_dim):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_gate = EdgeBiasGate(edge_bias_dim)
        self.atom_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_src, edge_dst, rbf, edge_bias, num_nodes):
        filters = self.filter_net(rbf)
        scale = self.edge_gate(edge_bias)
        messages = x[edge_src] * filters * scale.unsqueeze(-1)
        aggregated = torch.zeros(num_nodes, x.size(-1), device=x.device)
        aggregated = aggregated.index_add(0, edge_dst, messages)
        return x + self.out_net(self.atom_net(x) + aggregated)


class PAASchNet(nn.Module):
    def __init__(self, n_atom_types=100, phys_dim=6, hidden_dim=128, n_rbf=20,
                 n_interactions=3, cutoff=5.0):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.phys_proj = nn.Linear(phys_dim, hidden_dim)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        edge_bias_dim = n_rbf + phys_dim
        self.interactions = nn.ModuleList([
            PAASchNetInteraction(hidden_dim, n_rbf, edge_bias_dim)
            for _ in range(n_interactions)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z, pos, edge_src, edge_dst, phys_feats=None, batch=None):
        n_nodes = z.size(0)
        x = self.embedding(z) + self.phys_proj(phys_feats)
        vec = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(vec, dim=-1)
        rbf = self.rbf(dist)
        edge_diffs = torch.abs(phys_feats[edge_src] - phys_feats[edge_dst])
        edge_bias = torch.cat([rbf, edge_diffs], dim=-1)
        for interaction in self.interactions:
            x = interaction(x, edge_src, edge_dst, rbf, edge_bias, n_nodes)
        atom_energies = self.readout(x)
        return pool_atom_energies(atom_energies, batch)


class PaiNNInteraction(nn.Module):
    def __init__(self, hidden_dim, n_rbf):
        super().__init__()
        d = hidden_dim
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, 3 * d), nn.SiLU(), nn.Linear(3 * d, 3 * d)
        )

    def forward(self, s, v, edge_src, edge_dst, rbf, dir_vec, num_nodes):
        d = s.size(-1)
        filters = self.filter_net(rbf)
        w_ss = filters[:, :d]
        w_sv = filters[:, d:2 * d]
        w_vv = filters[:, 2 * d:3 * d]

        s_src = s[edge_src]
        v_src = v[edge_src]
        v_dot_dir = torch.einsum("ejd,ej->ed", v_src, dir_vec)
        msg_s = s_src * w_ss + v_dot_dir * w_sv
        ds = torch.zeros(num_nodes, d, device=s.device)
        ds = ds.index_add(0, edge_dst, msg_s)

        msg_v1 = torch.einsum("ed,ej->ejd", s_src * w_sv, dir_vec)
        msg_v2 = v_src * w_vv.unsqueeze(1)
        dv = torch.zeros(num_nodes, 3, d, device=s.device)
        dv = dv.index_add(0, edge_dst, msg_v1 + msg_v2)
        return s + ds, v + dv


class PaiNNUpdate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        d = hidden_dim
        self.u = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3 * d))
        self.v = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2 * d))

    def forward(self, s, v):
        d = s.size(-1)
        u_out = self.u(s)
        a_ss = u_out[:, :d]
        a_sv = u_out[:, d:2 * d]
        v_norm = torch.linalg.norm(v, dim=1)
        ds = a_ss + a_sv * v_norm

        v_out = self.v(v_norm)
        a_vv = v_out[:, :d]
        a_vs = v_out[:, d:]
        dv = v * a_vv.unsqueeze(1) + s.unsqueeze(1) * a_vs.unsqueeze(1)
        return s + ds, v + dv


class PaiNNPhys(nn.Module):
    def __init__(self, n_atom_types=100, phys_dim=8, hidden_dim=128, n_rbf=20,
                 n_interactions=3, cutoff=5.0):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.phys_proj = nn.Linear(phys_dim, hidden_dim)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.interactions = nn.ModuleList([
            PaiNNInteraction(hidden_dim, n_rbf) for _ in range(n_interactions)
        ])
        self.updates = nn.ModuleList([PaiNNUpdate(hidden_dim) for _ in range(n_interactions)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.hidden_dim = hidden_dim

    def forward(self, z, pos, edge_src, edge_dst, phys_feats=None, batch=None):
        n_nodes = z.size(0)
        s = self.embedding(z) + self.phys_proj(phys_feats)
        v = torch.zeros(n_nodes, 3, self.hidden_dim, device=z.device)
        vec = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(vec, dim=-1)
        rbf = self.rbf(dist)
        dir_vec = vec / (dist.unsqueeze(-1) + 1e-10)
        for interaction, update in zip(self.interactions, self.updates):
            s, v = interaction(s, v, edge_src, edge_dst, rbf, dir_vec, n_nodes)
            s, v = update(s, v)
        atom_energies = self.readout(s)
        return pool_atom_energies(atom_energies, batch)


def pool_atom_energies(atom_energies, batch):
    if batch is not None:
        n_mols = batch.max().item() + 1
        energy = torch.zeros(n_mols, 1, device=atom_energies.device)
        energy = energy.index_add(0, batch, atom_energies)
    else:
        energy = atom_energies.sum(0, keepdim=True)
    return energy.squeeze(-1), atom_energies


def branch_configs(dataset_roots, seeds=SEEDS):
    """Build the final expert specification from explicit private-data roots.

    ``dataset_roots`` must provide one root for each expert. This prevents a
    public checkout from silently depending on a private project directory.
    """
    required = {"schnet_static_phys", "paa_schnet_coord", "painn_coord_bond"}
    missing = required.difference(dataset_roots)
    if missing:
        raise ValueError(f"Missing dataset roots for: {sorted(missing)}")
    return [
        {
            "key": "schnet_static_phys",
            "label": "SchNet-static-phys",
            "dataset": Path(dataset_roots["schnet_static_phys"]),
            "model_class": SchNetPhys,
            "batch_size": BATCH_SIZE_SCHNET,
            "seeds": list(seeds),
        },
        {
            "key": "paa_schnet_coord",
            "label": "PAA-SchNet-coord",
            "dataset": Path(dataset_roots["paa_schnet_coord"]),
            "model_class": PAASchNet,
            "batch_size": BATCH_SIZE_SCHNET,
            "seeds": list(seeds),
        },
        {
            "key": "painn_coord_bond",
            "label": "PaiNN-coord-bond",
            "dataset": Path(dataset_roots["painn_coord_bond"]),
            "model_class": PaiNNPhys,
            "batch_size": BATCH_SIZE_PAINN,
            "seeds": list(seeds),
        },
    ]


def prediction_columns(branches):
    cols = []
    for branch in branches:
        for seed in branch["seeds"]:
            cols.append(f"{branch['key']}_seed{seed}")
    return cols


def load_all_pt_files(dataset_dir):
    samples = []
    dataset_dir = Path(dataset_dir)
    for system_id in sorted(p.name for p in dataset_dir.iterdir() if p.is_dir()):
        for fpath in sorted((dataset_dir / system_id).glob("*.pt")):
            try:
                data = torch.load(fpath, weights_only=False)
            except Exception as exc:
                print(f"Skip unreadable {fpath}: {exc}")
                continue
            if not hasattr(data, "group_id"):
                raise ValueError(f"Missing required group_id in {fpath}")
            gid = getattr(data, "group_id")
            sample_id = fpath.stem
            samples.append({
                "key": f"{system_id}/{sample_id}",
                "sample_id": sample_id,
                "path": str(fpath),
                "z": data.atomic_numbers.long(),
                "pos": data.pos.float(),
                "y": data.y.float().item(),
                "forces": data.forces.float(),
                "phys_feats": data.x.float(),
                "system_id": system_id,
                "group_id": gid,
                "edge_index": data.edge_index.long(),
            })
    return samples


def collate_batch(batch_samples):
    z_list, pos_list, y_list, f_list, phys_list, edge_list = [], [], [], [], [], []
    batch_idx, keys, sample_ids, system_ids = [], [], [], []
    n_total = 0
    for i, sample in enumerate(batch_samples):
        n = sample["z"].size(0)
        z_list.append(sample["z"])
        pos_list.append(sample["pos"])
        y_list.append(sample["y"])
        f_list.append(sample["forces"])
        phys_list.append(sample["phys_feats"])
        edge_list.append(sample["edge_index"] + n_total)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        keys.append(sample["key"])
        sample_ids.append(sample["sample_id"])
        system_ids.append(sample["system_id"])
        n_total += n
    return {
        "z": torch.cat(z_list, dim=0),
        "pos": torch.cat(pos_list, dim=0),
        "y": torch.tensor(y_list, dtype=torch.float32),
        "forces": torch.cat(f_list, dim=0),
        "phys_feats": torch.cat(phys_list, dim=0),
        "edge_index": torch.cat(edge_list, dim=1),
        "batch": torch.cat(batch_idx, dim=0),
        "keys": keys,
        "sample_ids": sample_ids,
        "system_ids": system_ids,
    }


def compute_energy_and_forces(model, batch, device):
    z = batch["z"].to(device)
    pos = batch["pos"].to(device).requires_grad_(True)
    phys = batch["phys_feats"].to(device)
    edge_index = batch["edge_index"].to(device)
    batch_idx = batch["batch"].to(device)
    energy, _ = model(z, pos, edge_index[0], edge_index[1], phys_feats=phys, batch=batch_idx)
    forces = -torch.autograd.grad(energy.sum(), pos, create_graph=True, retain_graph=True)[0]
    return energy, forces


def train_one_model(model, train_loader, val_loader, device, epochs=EPOCHS, early_stop=EARLY_STOP):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    best_state, best_val, patience = None, float("inf"), 0
    for _ in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            ep, fp = compute_energy_and_forces(model, batch, device)
            loss = (
                F.l1_loss(ep, batch["y"].to(device))
                + FORCE_WEIGHT * F.l1_loss(fp, batch["forces"].to(device))
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        for batch in val_loader:
            ep, fp = compute_energy_and_forces(model, batch, device)
            val_loss += (
                F.l1_loss(ep, batch["y"].to(device))
                + FORCE_WEIGHT * F.l1_loss(fp, batch["forces"].to(device))
            ).item()
        val_loss /= max(1, len(val_loader))
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= early_stop:
            break
    if best_state is None:
        raise RuntimeError("No best model state captured.")
    model.load_state_dict(best_state)
    return model


def predict_energy(model, loader, device):
    model.eval()
    preds, true, keys, sample_ids, systems = [], [], [], [], []
    for batch in loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        phys = batch["phys_feats"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_idx = batch["batch"].to(device)
        ep, _ = model(z, pos, edge_index[0], edge_index[1], phys_feats=phys, batch=batch_idx)
        preds.extend(ep.detach().cpu().numpy().tolist())
        true.extend(batch["y"].numpy().tolist())
        keys.extend(batch["keys"])
        sample_ids.extend(batch["sample_ids"])
        systems.extend(batch["system_ids"])
    return {
        "pred": np.array(preds, dtype=np.float64),
        "true": np.array(true, dtype=np.float64),
        "keys": keys,
        "sample_ids": sample_ids,
        "systems": systems,
    }


def group_split(samples, train_fraction=0.9, seed=RANDOM_SEED):
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample["system_id"], sample["group_id"])].append(sample)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_train = max(1, int(len(keys) * train_fraction))
    train_keys = set(keys[:n_train])
    train = [s for s in samples if (s["system_id"], s["group_id"]) in train_keys]
    val = [s for s in samples if (s["system_id"], s["group_id"]) not in train_keys]
    if not val:
        val = train[-max(1, len(train) // 10):]
        train = train[:-len(val)]
    return train, val


def group_folds(samples, k=K_OOF, seed=RANDOM_SEED):
    groups = sorted({(s["system_id"], s["group_id"]) for s in samples})
    rng = random.Random(seed)
    rng.shuffle(groups)
    folds = [set() for _ in range(k)]
    for idx, group in enumerate(groups):
        folds[idx % k].add(group)
    return folds


def build_model(branch, phys_dim, seed, device):
    seed_all(seed)
    model = branch["model_class"](
        n_atom_types=100,
        phys_dim=phys_dim,
        hidden_dim=HIDDEN_DIM,
        n_rbf=N_RBF,
        n_interactions=N_INTERACTIONS,
        cutoff=CUTOFF,
    )
    return model.to(device)


def generate_oof_predictions(branch, train_pool_samples, device, fold_name):
    from torch.utils.data import DataLoader

    phys_dim = int(train_pool_samples[0]["phys_feats"].shape[1])
    oof = {sample["key"]: {} for sample in train_pool_samples}
    folds = group_folds(train_pool_samples, K_OOF, RANDOM_SEED)
    for seed in branch["seeds"]:
        print(f"      {branch['label']} seed={seed} OOF", flush=True)
        for fold_idx, val_groups in enumerate(folds, start=1):
            fold_val = [s for s in train_pool_samples if (s["system_id"], s["group_id"]) in val_groups]
            fold_train_pool = [s for s in train_pool_samples if (s["system_id"], s["group_id"]) not in val_groups]
            fold_train, fold_es_val = group_split(fold_train_pool, 0.9, seed + fold_idx)
            train_loader = DataLoader(
                fold_train, batch_size=branch["batch_size"], shuffle=True,
                collate_fn=collate_batch, drop_last=True,
            )
            val_loader = DataLoader(
                fold_es_val, batch_size=branch["batch_size"], shuffle=False,
                collate_fn=collate_batch,
            )
            pred_loader = DataLoader(
                fold_val, batch_size=branch["batch_size"], shuffle=False,
                collate_fn=collate_batch,
            )
            model = build_model(branch, phys_dim, seed, device)
            model = train_one_model(model, train_loader, val_loader, device)
            pred = predict_energy(model, pred_loader, device)
            col = f"{branch['key']}_seed{seed}"
            for key, value in zip(pred["keys"], pred["pred"]):
                oof[key][col] = float(value)
            print(f"        fold {fold_idx}/{K_OOF} done for {fold_name}", flush=True)
    return oof


def train_final_and_predict(branch, train_pool_samples, test_samples, device, output_dir, test_system):
    from torch.utils.data import DataLoader

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phys_dim = int(train_pool_samples[0]["phys_feats"].shape[1])
    train_samples, val_samples = group_split(train_pool_samples, 0.9, RANDOM_SEED)
    train_loader = DataLoader(
        train_samples, batch_size=branch["batch_size"], shuffle=True,
        collate_fn=collate_batch, drop_last=True,
    )
    val_loader = DataLoader(
        val_samples, batch_size=branch["batch_size"], shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_samples, batch_size=branch["batch_size"], shuffle=False,
        collate_fn=collate_batch,
    )
    preds = {}
    meta = None
    for seed in branch["seeds"]:
        print(f"      final {branch['label']} seed={seed}", flush=True)
        model = build_model(branch, phys_dim, seed, device)
        model = train_one_model(model, train_loader, val_loader, device)
        torch.save(model.state_dict(), output_dir / f"model_{branch['key']}_{test_system}_seed{seed}.pt")
        pred = predict_energy(model, test_loader, device)
        col = f"{branch['key']}_seed{seed}"
        preds[col] = pred["pred"]
        if meta is None:
            meta = pred
    return meta, preds


def load_branch_samples(branches):
    loaded = {}
    for branch in branches:
        dataset = Path(branch["dataset"])
        if not dataset.exists():
            raise FileNotFoundError(f"Missing dataset for {branch['label']}: {dataset}")
        samples = load_all_pt_files(dataset)
        if not samples:
            raise RuntimeError(f"No samples loaded for {branch['label']} from {dataset}")
        loaded[branch["key"]] = samples
    return loaded
