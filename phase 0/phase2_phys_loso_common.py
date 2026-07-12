# -*- coding: utf-8 -*-
"""
Shared Phase 2 LOSO code for static/dynamic physical-feature baselines.

This module keeps SchNet+Phys and PaiNN+Phys experiments aligned with the
Phase 1.1 LOSO protocol:
  - leave one system out
  - split remaining systems into train/val by (system_id, group_id), 90/10
  - train energy+force model
  - report MAE(E), MAE_95(E), MAE(F)
"""
import csv
import os
import random
import time
import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_SEED = 42
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
        cutoff_val = 0.5 * (1.0 + torch.cos(np.pi * distances / self.cutoff))
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
        x = self.embedding(z)
        x = x + self.phys_proj(phys_feats)
        vec = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(vec, dim=-1)
        rbf = self.rbf(dist)
        for interaction in self.interactions:
            x = interaction(x, edge_src, edge_dst, rbf, n_nodes)
        atom_energies = self.readout(x)
        if batch is not None:
            n_mols = batch.max().item() + 1
            energy = torch.zeros(n_mols, 1, device=z.device)
            energy = energy.index_add(0, batch, atom_energies)
        else:
            energy = atom_energies.sum(0, keepdim=True)
        return energy.squeeze(-1), x


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
    def __init__(self, n_atom_types=100, phys_dim=5, hidden_dim=128, n_rbf=20,
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
        d = self.hidden_dim
        s = self.embedding(z) + self.phys_proj(phys_feats)
        v = torch.zeros(n_nodes, 3, d, device=z.device)

        vec = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(vec, dim=-1)
        rbf = self.rbf(dist)
        dir_vec = vec / (dist.unsqueeze(-1) + 1e-10)

        for interaction, update in zip(self.interactions, self.updates):
            s, v = interaction(s, v, edge_src, edge_dst, rbf, dir_vec, n_nodes)
            s, v = update(s, v)

        atom_energies = self.readout(s)
        if batch is not None:
            n_mols = batch.max().item() + 1
            energy = torch.zeros(n_mols, 1, device=z.device)
            energy = energy.index_add(0, batch, atom_energies)
        else:
            energy = atom_energies.sum(0, keepdim=True)
        return energy.squeeze(-1), s


def load_all_pt_files(dataset_dir):
    samples = []
    for system_id in sorted(os.listdir(dataset_dir)):
        sys_dir = os.path.join(dataset_dir, system_id)
        if not os.path.isdir(sys_dir):
            continue
        for fname in sorted(os.listdir(sys_dir)):
            if not fname.endswith(".pt"):
                continue
            fpath = os.path.join(sys_dir, fname)
            try:
                data = torch.load(fpath, weights_only=False)
            except Exception as exc:
                print(f"Skip unreadable {fpath}: {exc}")
                continue
            gid = getattr(data, "group_id", 0) if hasattr(data, "group_id") else 0
            samples.append({
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
    batch_idx, n_total = [], 0
    for i, sample in enumerate(batch_samples):
        n = sample["z"].size(0)
        z_list.append(sample["z"])
        pos_list.append(sample["pos"])
        y_list.append(sample["y"])
        f_list.append(sample["forces"])
        phys_list.append(sample["phys_feats"])
        edge_list.append(sample["edge_index"] + n_total)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        n_total += n
    return {
        "z": torch.cat(z_list, dim=0),
        "pos": torch.cat(pos_list, dim=0),
        "y": torch.tensor(y_list, dtype=torch.float32),
        "forces": torch.cat(f_list, dim=0),
        "phys_feats": torch.cat(phys_list, dim=0),
        "edge_index": torch.cat(edge_list, dim=1),
        "batch": torch.cat(batch_idx, dim=0),
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


def evaluate(model, loader, device, force_weight, return_preds=False):
    model.eval()
    total_mae_e, total_mae_f, total_loss, n_batches = 0.0, 0.0, 0.0, 0
    all_y_true, all_y_pred = [], []
    for batch in loader:
        energy_pred, forces_pred = compute_energy_and_forces(model, batch, device)
        energy_true = batch["y"].to(device)
        forces_true = batch["forces"].to(device)
        loss_e = F.l1_loss(energy_pred, energy_true)
        loss_f = F.l1_loss(forces_pred, forces_true)
        loss = loss_e + force_weight * loss_f
        total_mae_e += loss_e.item()
        total_mae_f += loss_f.item()
        total_loss += loss.item()
        n_batches += 1
        if return_preds:
            all_y_true.append(energy_true.cpu())
            all_y_pred.append(energy_pred.detach().cpu())
    result = (total_mae_e / n_batches, total_mae_f / n_batches, total_loss / n_batches)
    if return_preds:
        result += (torch.cat(all_y_true), torch.cat(all_y_pred))
    return result


def train_epoch(model, loader, optimizer, device, force_weight):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        energy_pred, forces_pred = compute_energy_and_forces(model, batch, device)
        loss = (
            F.l1_loss(energy_pred, batch["y"].to(device))
            + force_weight * F.l1_loss(forces_pred, batch["forces"].to(device))
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def train_one_fold(model, train_loader, val_loader, device, force_weight, epochs, early_stop, fold_name):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    best_val_loss, best_state, patience = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, device, force_weight)
        _, _, val_loss = evaluate(model, val_loader, device, force_weight)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= early_stop:
            print(f"    [{fold_name}] Early stop @ epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return model, best_val_loss


def read_result_csv(path):
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sys_id = row.get("Left-Out System") or row.get("Left_Out_System") or row.get("System")
            if not sys_id:
                continue
            rows[sys_id] = {
                "mae_e": float(row["MAE_E_eV"]),
                "mae_95_e": float(row["MAE_95_E_eV"]),
                "mae_f": float(row["MAE_F_eV_A"]),
            }
    return rows


def write_results_and_bar(output_dir, csv_name, bar_name, results, title):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Left_Out_System", "MAE_E_eV", "MAE_95_E_eV", "MAE_F_eV_A", "N_Test", "Time_s"],
        )
        writer.writeheader()
        for sys_id in sorted(results):
            r = results[sys_id]
            writer.writerow({
                "Left_Out_System": sys_id,
                "MAE_E_eV": f"{r['mae_e']:.6f}",
                "MAE_95_E_eV": f"{r['mae_95_e']:.6f}",
                "MAE_F_eV_A": f"{r['mae_f']:.6f}",
                "N_Test": r["n_test"],
                "Time_s": f"{r['train_time']:.1f}",
            })

    sys_names = sorted(results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_e = ["#d62728" if s == "LaCu12" else "#1f77b4" for s in sys_names]
    colors_95 = ["#d62728" if s == "LaCu12" else "#ff7f0e" for s in sys_names]
    colors_f = ["#d62728" if s == "LaCu12" else "#2ca02c" for s in sys_names]
    axes[0].bar(sys_names, [results[s]["mae_e"] for s in sys_names], color=colors_e)
    axes[0].set_ylabel("MAE (eV)")
    axes[0].set_title(f"{title}: MAE Energy")
    axes[1].bar(sys_names, [results[s]["mae_95_e"] for s in sys_names], color=colors_95)
    axes[1].set_ylabel("MAE_95 (eV)")
    axes[1].set_title(f"{title}: MAE_95 Energy")
    axes[2].bar(sys_names, [results[s]["mae_f"] for s in sys_names], color=colors_f)
    axes[2].set_ylabel("MAE (eV/A)")
    axes[2].set_title(f"{title}: MAE Forces")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / bar_name, dpi=150)
    plt.close(fig)
    return csv_path


def write_comparison(output_dir, model_label, dynamic_results, baseline_csv, static_csv=None):
    output_dir = Path(output_dir)
    baseline = read_result_csv(baseline_csv)
    static = read_result_csv(static_csv) if static_csv else {}
    path = output_dir / f"{model_label.lower()}_phys_comparison.csv"
    fieldnames = [
        "System",
        "NoPhys_MAE_E", "StaticPhys_MAE_E", "DynamicPhys_MAE_E",
        "Dynamic_vs_NoPhys_pct", "Dynamic_vs_StaticPhys_pct",
        "NoPhys_MAE95_E", "StaticPhys_MAE95_E", "DynamicPhys_MAE95_E",
        "Dynamic95_vs_NoPhys_pct", "Dynamic95_vs_StaticPhys_pct",
        "NoPhys_MAE_F", "StaticPhys_MAE_F", "DynamicPhys_MAE_F",
        "DynamicF_vs_NoPhys_pct", "DynamicF_vs_StaticPhys_pct",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sys_id in sorted(dynamic_results):
            d = dynamic_results[sys_id]
            b = baseline.get(sys_id)
            s = static.get(sys_id)

            def value(row, key):
                return "" if row is None else f"{row[key]:.6f}"

            def gain(old, new, key):
                if old is None or old[key] == 0:
                    return ""
                return f"{(old[key] - new[key]) / old[key] * 100:.2f}"

            writer.writerow({
                "System": sys_id,
                "NoPhys_MAE_E": value(b, "mae_e"),
                "StaticPhys_MAE_E": value(s, "mae_e"),
                "DynamicPhys_MAE_E": f"{d['mae_e']:.6f}",
                "Dynamic_vs_NoPhys_pct": gain(b, d, "mae_e"),
                "Dynamic_vs_StaticPhys_pct": gain(s, d, "mae_e"),
                "NoPhys_MAE95_E": value(b, "mae_95_e"),
                "StaticPhys_MAE95_E": value(s, "mae_95_e"),
                "DynamicPhys_MAE95_E": f"{d['mae_95_e']:.6f}",
                "Dynamic95_vs_NoPhys_pct": gain(b, d, "mae_95_e"),
                "Dynamic95_vs_StaticPhys_pct": gain(s, d, "mae_95_e"),
                "NoPhys_MAE_F": value(b, "mae_f"),
                "StaticPhys_MAE_F": value(s, "mae_f"),
                "DynamicPhys_MAE_F": f"{d['mae_f']:.6f}",
                "DynamicF_vs_NoPhys_pct": gain(b, d, "mae_f"),
                "DynamicF_vs_StaticPhys_pct": gain(s, d, "mae_f"),
            })
    return path


def run_loso_experiment(
    model_name,
    model_class,
    dataset_dir,
    output_dir,
    result_csv_name,
    bar_name,
    model_prefix,
    batch_size,
    baseline_csv,
    static_csv=None,
):
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"DEVICE: {device}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")

    all_samples = load_all_pt_files(dataset_dir)
    if not all_samples:
        raise RuntimeError(f"No samples loaded from {dataset_dir}")
    phys_dim = int(all_samples[0]["phys_feats"].shape[1])
    systems = sorted({s["system_id"] for s in all_samples})
    print(f"Loaded {len(all_samples)} samples, systems={systems}, PHYS_DIM={phys_dim}")

    from torch.utils.data import DataLoader

    loso_results = {}
    for fold_idx, test_system in enumerate(systems, start=1):
        print(f"\nFold {fold_idx}/{len(systems)}: Leave out [{test_system}]")
        test_samples = [s for s in all_samples if s["system_id"] == test_system]
        train_val_samples = [s for s in all_samples if s["system_id"] != test_system]

        random.seed(RANDOM_SEED)
        train_val_groups = defaultdict(list)
        for sample in train_val_samples:
            train_val_groups[(sample["system_id"], sample["group_id"])].append(sample)
        group_keys = sorted(train_val_groups.keys())
        random.shuffle(group_keys)
        n_train = int(len(group_keys) * 0.90)
        train_groups = set(group_keys[:n_train])
        train_samples = [s for s in train_val_samples if (s["system_id"], s["group_id"]) in train_groups]
        val_samples = [s for s in train_val_samples if (s["system_id"], s["group_id"]) not in train_groups]
        print(f"  Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

        train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_batch, drop_last=True)
        val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_batch)
        test_loader = DataLoader(test_samples, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_batch)

        model = model_class(
            n_atom_types=100,
            phys_dim=phys_dim,
            hidden_dim=HIDDEN_DIM,
            n_rbf=N_RBF,
            n_interactions=N_INTERACTIONS,
            cutoff=CUTOFF,
        ).to(device)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        t0 = time.time()
        model, _ = train_one_fold(
            model, train_loader, val_loader, device, FORCE_WEIGHT, EPOCHS, EARLY_STOP, test_system
        )
        train_time = time.time() - t0
        mae_e, mae_f, _, y_true, y_pred = evaluate(model, test_loader, device, FORCE_WEIGHT, return_preds=True)
        abs_errors = torch.abs(y_true - y_pred)
        tail_k = max(1, int(len(abs_errors) * 0.05))
        mae_95 = torch.sort(abs_errors, descending=True)[0][:tail_k].mean().item()

        loso_results[test_system] = {
            "mae_e": mae_e,
            "mae_f": mae_f,
            "mae_95_e": mae_95,
            "n_test": len(test_samples),
            "train_time": train_time,
        }
        print(f"  Result: MAE(E)={mae_e:.6f}, MAE_95(E)={mae_95:.6f}, MAE(F)={mae_f:.6f}")
        torch.save(model.state_dict(), output_dir / f"{model_prefix}_{test_system}.pt")

    csv_path = write_results_and_bar(output_dir, result_csv_name, bar_name, loso_results, model_name)
    comparison_path = write_comparison(output_dir, model_name, loso_results, baseline_csv, static_csv)
    print(f"\nResults CSV saved to: {csv_path}")
    print(f"Comparison CSV saved to: {comparison_path}")
    print("Complete.")
