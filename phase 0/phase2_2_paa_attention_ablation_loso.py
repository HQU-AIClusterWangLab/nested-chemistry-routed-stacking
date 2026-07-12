# -*- coding: utf-8 -*-
"""
Phase 2.2: PAA attention/bias ablations for SchNet and PaiNN.

This script keeps the LOSO protocol used in Phase 1.1/2.1, but adds a
physically augmented edge gate:

    Bij = [RBF(d_ij), abs(Pi_i - Pi_j)]
    message_ij = message_ij * (1 + alpha * (2 * sigmoid(MLP(Bij)) - 1))

Default runs exclude Mulliken features. Mulliken charge is reserved for
post-hoc explanation/oracle checks, not deployable prediction.
"""
import argparse
import csv
import math
import random
import subprocess
import sys
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


ROOT = Path(r"D:\lunwen\2.1sci")
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "phase 2" / "phase2_2_paa_attention_ablation_output"

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
GATE_MAX_ALPHA = 0.5


DATASETS = {
    "static": ROOT / "phase 0" / "dataset" / "processed",
    "coord": ROOT / "phase 0" / "dataset" / "processed_dynamic_coord",
    "coord_bond": ROOT / "phase 0" / "dataset" / "processed_dynamic_coord_bond",
}

RUNS = {
    "schnet_static": {
        "model_key": "schnet",
        "variant": "static",
        "label": "PAA-SchNet-static",
        "batch_size": BATCH_SIZE_SCHNET,
        "baseline_csv": ROOT / "phase 1" / "loso_schnet_output" / "loso_results.csv",
        "phys_csv": ROOT / "phase 2" / "loso_schnet_phys_output" / "loso_schnet_phys_results.csv",
    },
    "schnet_coord": {
        "model_key": "schnet",
        "variant": "coord",
        "label": "PAA-SchNet-coord",
        "batch_size": BATCH_SIZE_SCHNET,
        "baseline_csv": ROOT / "phase 1" / "loso_schnet_output" / "loso_results.csv",
        "phys_csv": ROOT / "phase 2" / "loso_schnet_phys_output" / "loso_schnet_phys_results.csv",
    },
    "painn_static": {
        "model_key": "painn",
        "variant": "static",
        "label": "PAA-PaiNN-static",
        "batch_size": BATCH_SIZE_PAINN,
        "baseline_csv": ROOT / "phase 1" / "loso_painn_output" / "loso_painn_results.csv",
        "phys_csv": ROOT / "phase 2" / "loso_painn_phys_output" / "loso_painn_phys_results.csv",
    },
    "painn_coord": {
        "model_key": "painn",
        "variant": "coord",
        "label": "PAA-PaiNN-coord",
        "batch_size": BATCH_SIZE_PAINN,
        "baseline_csv": ROOT / "phase 1" / "loso_painn_output" / "loso_painn_results.csv",
        "phys_csv": ROOT / "phase 2" / "loso_painn_phys_output" / "loso_painn_phys_results.csv",
    },
    "painn_coord_bond": {
        "model_key": "painn",
        "variant": "coord_bond",
        "label": "PAA-PaiNN-coord_bond",
        "batch_size": BATCH_SIZE_PAINN,
        "baseline_csv": ROOT / "phase 1" / "loso_painn_output" / "loso_painn_results.csv",
        "phys_csv": ROOT / "phase 2" / "loso_painn_phys_output" / "loso_painn_phys_results.csv",
    },
}


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
        return gate, scale, alpha


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
        gate, scale, alpha = self.edge_gate(edge_bias)
        messages = x[edge_src] * filters * scale.unsqueeze(-1)
        aggregated = torch.zeros(num_nodes, x.size(-1), device=x.device)
        aggregated = aggregated.index_add(0, edge_dst, messages)
        x = x + self.out_net(self.atom_net(x) + aggregated)
        return x, gate, scale, alpha


class PAASchNet(nn.Module):
    def __init__(self, n_atom_types=100, phys_dim=5, hidden_dim=128, n_rbf=20,
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

    def forward(self, z, pos, edge_src, edge_dst, phys_feats=None, batch=None, return_attention=False):
        num_nodes = z.size(0)
        x = self.embedding(z) + self.phys_proj(phys_feats)
        vec = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(vec, dim=-1)
        rbf = self.rbf(dist)
        edge_diffs = torch.abs(phys_feats[edge_src] - phys_feats[edge_dst])
        edge_bias = torch.cat([rbf, edge_diffs], dim=-1)

        gates, scales, alphas = [], [], []
        for interaction in self.interactions:
            x, gate, scale, alpha = interaction(x, edge_src, edge_dst, rbf, edge_bias, num_nodes)
            gates.append(gate)
            scales.append(scale)
            alphas.append(alpha)

        atom_energies = self.readout(x)
        if batch is not None:
            n_mols = batch.max().item() + 1
            energy = torch.zeros(n_mols, 1, device=z.device)
            energy = energy.index_add(0, batch, atom_energies)
        else:
            energy = atom_energies.sum(0, keepdim=True)
        energy = energy.squeeze(-1)
        if not return_attention:
            return energy, x
        aux = {
            "dist": dist,
            "edge_diffs": edge_diffs,
            "gates": gates,
            "scales": scales,
            "alphas": alphas,
        }
        return energy, x, aux


class PAAPaiNNInteraction(nn.Module):
    def __init__(self, hidden_dim, n_rbf, edge_bias_dim):
        super().__init__()
        d = hidden_dim
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, 3 * d), nn.SiLU(), nn.Linear(3 * d, 3 * d)
        )
        self.edge_gate = EdgeBiasGate(edge_bias_dim)

    def forward(self, s, v, edge_src, edge_dst, rbf, dir_vec, edge_bias, num_nodes):
        d = s.size(-1)
        filters = self.filter_net(rbf)
        gate, scale, alpha = self.edge_gate(edge_bias)
        w_ss = filters[:, :d]
        w_sv = filters[:, d:2 * d]
        w_vv = filters[:, 2 * d:3 * d]

        s_src = s[edge_src]
        v_src = v[edge_src]
        v_dot_dir = torch.einsum("ejd,ej->ed", v_src, dir_vec)
        msg_s = (s_src * w_ss + v_dot_dir * w_sv) * scale.unsqueeze(-1)
        ds = torch.zeros(num_nodes, d, device=s.device)
        ds = ds.index_add(0, edge_dst, msg_s)

        msg_v1 = torch.einsum("ed,ej->ejd", s_src * w_sv, dir_vec)
        msg_v2 = v_src * w_vv.unsqueeze(1)
        msg_v = (msg_v1 + msg_v2) * scale.view(-1, 1, 1)
        dv = torch.zeros(num_nodes, 3, d, device=s.device)
        dv = dv.index_add(0, edge_dst, msg_v)
        return s + ds, v + dv, gate, scale, alpha


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


class PAAPaiNN(nn.Module):
    def __init__(self, n_atom_types=100, phys_dim=5, hidden_dim=128, n_rbf=20,
                 n_interactions=3, cutoff=5.0):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim, padding_idx=0)
        self.phys_proj = nn.Linear(phys_dim, hidden_dim)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        edge_bias_dim = n_rbf + phys_dim
        self.interactions = nn.ModuleList([
            PAAPaiNNInteraction(hidden_dim, n_rbf, edge_bias_dim)
            for _ in range(n_interactions)
        ])
        self.updates = nn.ModuleList([PaiNNUpdate(hidden_dim) for _ in range(n_interactions)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.hidden_dim = hidden_dim

    def forward(self, z, pos, edge_src, edge_dst, phys_feats=None, batch=None, return_attention=False):
        num_nodes = z.size(0)
        s = self.embedding(z) + self.phys_proj(phys_feats)
        v = torch.zeros(num_nodes, 3, self.hidden_dim, device=z.device)

        vec = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(vec, dim=-1)
        rbf = self.rbf(dist)
        dir_vec = vec / (dist.unsqueeze(-1) + 1e-10)
        edge_diffs = torch.abs(phys_feats[edge_src] - phys_feats[edge_dst])
        edge_bias = torch.cat([rbf, edge_diffs], dim=-1)

        gates, scales, alphas = [], [], []
        for interaction, update in zip(self.interactions, self.updates):
            s, v, gate, scale, alpha = interaction(
                s, v, edge_src, edge_dst, rbf, dir_vec, edge_bias, num_nodes
            )
            s, v = update(s, v)
            gates.append(gate)
            scales.append(scale)
            alphas.append(alpha)

        atom_energies = self.readout(s)
        if batch is not None:
            n_mols = batch.max().item() + 1
            energy = torch.zeros(n_mols, 1, device=z.device)
            energy = energy.index_add(0, batch, atom_energies)
        else:
            energy = atom_energies.sum(0, keepdim=True)
        energy = energy.squeeze(-1)
        if not return_attention:
            return energy, s
        aux = {
            "dist": dist,
            "edge_diffs": edge_diffs,
            "gates": gates,
            "scales": scales,
            "alphas": alphas,
        }
        return energy, s, aux


def load_all_pt_files(dataset_dir):
    samples = []
    dataset_dir = Path(dataset_dir)
    for system_id in sorted(p.name for p in dataset_dir.iterdir() if p.is_dir()):
        sys_dir = dataset_dir / system_id
        for fpath in sorted(sys_dir.glob("*.pt")):
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
                "sample_id": fpath.stem,
                "path": str(fpath),
            })
    return samples


def collate_batch(batch_samples):
    z_list, pos_list, y_list, f_list, phys_list, edge_list = [], [], [], [], [], []
    batch_idx, node_sample_idx, node_local_idx = [], [], []
    sample_ids, system_ids, group_ids, paths = [], [], [], []
    n_total = 0
    for sample_idx, sample in enumerate(batch_samples):
        n = sample["z"].size(0)
        z_list.append(sample["z"])
        pos_list.append(sample["pos"])
        y_list.append(sample["y"])
        f_list.append(sample["forces"])
        phys_list.append(sample["phys_feats"])
        edge_list.append(sample["edge_index"] + n_total)
        batch_idx.append(torch.full((n,), sample_idx, dtype=torch.long))
        node_sample_idx.append(torch.full((n,), sample_idx, dtype=torch.long))
        node_local_idx.append(torch.arange(n, dtype=torch.long))
        sample_ids.append(sample["sample_id"])
        system_ids.append(sample["system_id"])
        group_ids.append(sample["group_id"])
        paths.append(sample["path"])
        n_total += n
    return {
        "z": torch.cat(z_list, dim=0),
        "pos": torch.cat(pos_list, dim=0),
        "y": torch.tensor(y_list, dtype=torch.float32),
        "forces": torch.cat(f_list, dim=0),
        "phys_feats": torch.cat(phys_list, dim=0),
        "edge_index": torch.cat(edge_list, dim=1),
        "batch": torch.cat(batch_idx, dim=0),
        "node_sample_idx": torch.cat(node_sample_idx, dim=0),
        "node_local_idx": torch.cat(node_local_idx, dim=0),
        "sample_ids": sample_ids,
        "system_ids": system_ids,
        "group_ids": group_ids,
        "paths": paths,
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
    all_y_true, all_y_pred, all_sample_ids = [], [], []
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
            all_y_true.append(energy_true.detach().cpu())
            all_y_pred.append(energy_pred.detach().cpu())
            all_sample_ids.extend(batch["sample_ids"])
    result = (total_mae_e / n_batches, total_mae_f / n_batches, total_loss / n_batches)
    if return_preds:
        result += (torch.cat(all_y_true), torch.cat(all_y_pred), all_sample_ids)
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


def train_one_fold(model, train_loader, val_loader, device, fold_name):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    best_val_loss, best_state, patience = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, device, FORCE_WEIGHT)
        _, _, val_loss = evaluate(model, val_loader, device, FORCE_WEIGHT)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOP:
            print(f"    [{fold_name}] Early stop @ epoch {epoch}")
            break
    if best_state is None:
        raise RuntimeError(f"No best state captured for fold {fold_name}")
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


def percent_gain(old, new):
    if old is None or old == 0:
        return ""
    return f"{(old - new) / old * 100:.2f}"


def export_attention_edges(model, loader, device, output_csv, max_rows):
    model.eval()
    rows_written = 0
    output_csv = Path(output_csv)
    with open(output_csv, "w", newline="") as f:
        fieldnames = [
            "system_id", "sample_id", "atom_i", "atom_j", "Z_i", "Z_j", "distance",
            "phys_diff_mean", "phys_diff_max", "coord_diff", "bond_mean_diff", "bond_std_diff",
            "gate_layer_0", "gate_layer_1", "gate_layer_2", "gate_mean",
            "scale_layer_0", "scale_layer_1", "scale_layer_2", "scale_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for batch in loader:
                z = batch["z"].to(device)
                pos = batch["pos"].to(device)
                phys = batch["phys_feats"].to(device)
                edge_index = batch["edge_index"].to(device)
                batch_idx = batch["batch"].to(device)
                _, _, aux = model(
                    z, pos, edge_index[0], edge_index[1],
                    phys_feats=phys, batch=batch_idx, return_attention=True,
                )
                edge_src = edge_index[0].detach().cpu()
                edge_dst = edge_index[1].detach().cpu()
                node_sample_idx = batch["node_sample_idx"]
                node_local_idx = batch["node_local_idx"]
                z_cpu = batch["z"].cpu()
                dist = aux["dist"].detach().cpu()
                edge_diffs = aux["edge_diffs"].detach().cpu()
                gates = [g.detach().cpu() for g in aux["gates"]]
                scales = [s.detach().cpu() for s in aux["scales"]]

                for edge_pos in range(edge_src.numel()):
                    if rows_written >= max_rows:
                        return output_csv
                    src = int(edge_src[edge_pos])
                    dst = int(edge_dst[edge_pos])
                    sample_idx = int(node_sample_idx[src])
                    diffs = edge_diffs[edge_pos]
                    gate_vals = [float(g[edge_pos]) for g in gates]
                    scale_vals = [float(s[edge_pos]) for s in scales]
                    writer.writerow({
                        "system_id": batch["system_ids"][sample_idx],
                        "sample_id": batch["sample_ids"][sample_idx],
                        "atom_i": int(node_local_idx[src]),
                        "atom_j": int(node_local_idx[dst]),
                        "Z_i": int(z_cpu[src]),
                        "Z_j": int(z_cpu[dst]),
                        "distance": f"{float(dist[edge_pos]):.6f}",
                        "phys_diff_mean": f"{float(diffs.mean()):.6f}",
                        "phys_diff_max": f"{float(diffs.max()):.6f}",
                        "coord_diff": f"{float(diffs[5]):.6f}" if diffs.numel() > 5 else "",
                        "bond_mean_diff": f"{float(diffs[6]):.6f}" if diffs.numel() > 6 else "",
                        "bond_std_diff": f"{float(diffs[7]):.6f}" if diffs.numel() > 7 else "",
                        "gate_layer_0": f"{gate_vals[0]:.6f}" if len(gate_vals) > 0 else "",
                        "gate_layer_1": f"{gate_vals[1]:.6f}" if len(gate_vals) > 1 else "",
                        "gate_layer_2": f"{gate_vals[2]:.6f}" if len(gate_vals) > 2 else "",
                        "gate_mean": f"{float(np.mean(gate_vals)):.6f}",
                        "scale_layer_0": f"{scale_vals[0]:.6f}" if len(scale_vals) > 0 else "",
                        "scale_layer_1": f"{scale_vals[1]:.6f}" if len(scale_vals) > 1 else "",
                        "scale_layer_2": f"{scale_vals[2]:.6f}" if len(scale_vals) > 2 else "",
                        "scale_mean": f"{float(np.mean(scale_vals)):.6f}",
                    })
                    rows_written += 1
    return output_csv


def write_run_outputs(run_dir, run_key, label, results, predictions, baseline_csv, phys_csv):
    run_dir.mkdir(parents=True, exist_ok=True)
    result_csv = run_dir / f"loso_{run_key}_results.csv"
    with open(result_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Left_Out_System", "MAE_E_eV", "MAE_95_E_eV", "MAE_F_eV_A",
                "N_Test", "Time_s", "Gate_Alpha_Mean",
            ],
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
                "Gate_Alpha_Mean": f"{r['gate_alpha_mean']:.6f}",
            })

    pred_csv = run_dir / f"loso_{run_key}_predictions.csv"
    with open(pred_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Left_Out_System", "sample_id", "y_true", "y_pred", "abs_error"],
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)

    baseline = read_result_csv(baseline_csv)
    phys = read_result_csv(phys_csv)
    comparison_csv = run_dir / f"loso_{run_key}_comparison.csv"
    with open(comparison_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "System",
                "NoPhys_MAE_E", "Phys_MAE_E", "PAA_MAE_E",
                "PAA_vs_NoPhys_pct", "PAA_vs_Phys_pct",
                "NoPhys_MAE95_E", "Phys_MAE95_E", "PAA_MAE95_E",
                "PAA95_vs_NoPhys_pct", "PAA95_vs_Phys_pct",
                "NoPhys_MAE_F", "Phys_MAE_F", "PAA_MAE_F",
                "PAAF_vs_NoPhys_pct", "PAAF_vs_Phys_pct",
            ],
        )
        writer.writeheader()
        for sys_id in sorted(results):
            r = results[sys_id]
            b = baseline.get(sys_id)
            p = phys.get(sys_id)
            writer.writerow({
                "System": sys_id,
                "NoPhys_MAE_E": f"{b['mae_e']:.6f}" if b else "",
                "Phys_MAE_E": f"{p['mae_e']:.6f}" if p else "",
                "PAA_MAE_E": f"{r['mae_e']:.6f}",
                "PAA_vs_NoPhys_pct": percent_gain(b["mae_e"], r["mae_e"]) if b else "",
                "PAA_vs_Phys_pct": percent_gain(p["mae_e"], r["mae_e"]) if p else "",
                "NoPhys_MAE95_E": f"{b['mae_95_e']:.6f}" if b else "",
                "Phys_MAE95_E": f"{p['mae_95_e']:.6f}" if p else "",
                "PAA_MAE95_E": f"{r['mae_95_e']:.6f}",
                "PAA95_vs_NoPhys_pct": percent_gain(b["mae_95_e"], r["mae_95_e"]) if b else "",
                "PAA95_vs_Phys_pct": percent_gain(p["mae_95_e"], r["mae_95_e"]) if p else "",
                "NoPhys_MAE_F": f"{b['mae_f']:.6f}" if b else "",
                "Phys_MAE_F": f"{p['mae_f']:.6f}" if p else "",
                "PAA_MAE_F": f"{r['mae_f']:.6f}",
                "PAAF_vs_NoPhys_pct": percent_gain(b["mae_f"], r["mae_f"]) if b else "",
                "PAAF_vs_Phys_pct": percent_gain(p["mae_f"], r["mae_f"]) if p else "",
            })

    systems = sorted(results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, title, color in [
        (axes[0], "mae_e", "MAE Energy (eV)", "#1f77b4"),
        (axes[1], "mae_95_e", "MAE_95 Energy (eV)", "#ff7f0e"),
        (axes[2], "mae_f", "MAE Force (eV/A)", "#2ca02c"),
    ]:
        colors = ["#d62728" if s == "LaCu12" else color for s in systems]
        ax.bar(systems, [results[s][metric] for s in systems], color=colors)
        ax.set_title(f"{label}: {title}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    bar_png = run_dir / f"loso_{run_key}_bar.png"
    plt.savefig(bar_png, dpi=150)
    plt.close(fig)

    return result_csv, pred_csv, comparison_csv, bar_png


def resolve_dataset_dir(path):
    path = Path(path)
    if path.exists():
        return path
    candidates = sorted(
        path.parent.glob(f"{path.name}_20*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else path


def build_dynamic_datasets():
    phase5_script = ROOT / "phase 0" / "phase5_dynamic_phys_features.py"
    command = [sys.executable, str(phase5_script), "--variant", "all", "--allow-overwrite"]
    print("Missing dynamic datasets. Building them with:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


def run_one(run_key, cfg, max_attention_rows_per_fold):
    from torch.utils.data import DataLoader

    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = resolve_dataset_dir(DATASETS[cfg["variant"]])
    if not dataset_dir.exists() and cfg["variant"] != "static":
        build_dynamic_datasets()
        dataset_dir = resolve_dataset_dir(DATASETS[cfg["variant"]])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found for {run_key}: {dataset_dir}")

    run_dir = OUTPUT_ROOT / f"paa_{run_key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"Running {cfg['label']}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output : {run_dir}")
    print(f"Device : {device}")

    all_samples = load_all_pt_files(dataset_dir)
    if not all_samples:
        raise RuntimeError(f"No .pt samples loaded from {dataset_dir}")
    phys_dim = int(all_samples[0]["phys_feats"].shape[1])
    systems = sorted({s["system_id"] for s in all_samples})
    print(f"Loaded {len(all_samples)} samples, systems={systems}, phys_dim={phys_dim}")

    model_class = PAASchNet if cfg["model_key"] == "schnet" else PAAPaiNN
    results = {}
    predictions = []
    for fold_idx, test_system in enumerate(systems, start=1):
        print(f"\n  Fold {fold_idx}/{len(systems)}: Leave out [{test_system}]")
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
        print(f"    Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

        train_loader = DataLoader(
            train_samples, batch_size=cfg["batch_size"], shuffle=True,
            collate_fn=collate_batch, drop_last=True,
        )
        val_loader = DataLoader(
            val_samples, batch_size=cfg["batch_size"], shuffle=False,
            collate_fn=collate_batch,
        )
        test_loader = DataLoader(
            test_samples, batch_size=cfg["batch_size"], shuffle=False,
            collate_fn=collate_batch,
        )

        model = model_class(
            n_atom_types=100,
            phys_dim=phys_dim,
            hidden_dim=HIDDEN_DIM,
            n_rbf=N_RBF,
            n_interactions=N_INTERACTIONS,
            cutoff=CUTOFF,
        ).to(device)
        print(f"    Params={sum(p.numel() for p in model.parameters()):,}")

        t0 = time.time()
        model, _ = train_one_fold(model, train_loader, val_loader, device, test_system)
        train_time = time.time() - t0

        mae_e, mae_f, _, y_true, y_pred, sample_ids = evaluate(
            model, test_loader, device, FORCE_WEIGHT, return_preds=True
        )
        abs_errors = torch.abs(y_true - y_pred)
        tail_k = max(1, int(len(abs_errors) * 0.05))
        mae_95 = torch.sort(abs_errors, descending=True)[0][:tail_k].mean().item()
        alpha_vals = [
            float(module.edge_gate.max_alpha * torch.sigmoid(module.edge_gate.raw_alpha).detach().cpu())
            for module in getattr(model, "interactions")
        ]
        results[test_system] = {
            "mae_e": mae_e,
            "mae_f": mae_f,
            "mae_95_e": mae_95,
            "n_test": len(test_samples),
            "train_time": train_time,
            "gate_alpha_mean": float(np.mean(alpha_vals)),
        }
        for sid, yt, yp, ae in zip(sample_ids, y_true.tolist(), y_pred.tolist(), abs_errors.tolist()):
            predictions.append({
                "Left_Out_System": test_system,
                "sample_id": sid,
                "y_true": f"{yt:.8f}",
                "y_pred": f"{yp:.8f}",
                "abs_error": f"{ae:.8f}",
            })
        print(
            f"    Result: MAE(E)={mae_e:.6f}, MAE_95(E)={mae_95:.6f}, "
            f"MAE(F)={mae_f:.6f}, alpha_mean={np.mean(alpha_vals):.4f}"
        )

        torch.save(model.state_dict(), run_dir / f"model_{run_key}_loso_{test_system}.pt")
        attention_csv = run_dir / f"attention_edges_{run_key}_{test_system}.csv"
        export_attention_edges(model, test_loader, device, attention_csv, max_attention_rows_per_fold)
        print(f"    Attention export: {attention_csv}")

    result_csv, pred_csv, comparison_csv, bar_png = write_run_outputs(
        run_dir, run_key, cfg["label"], results, predictions, cfg["baseline_csv"], cfg["phys_csv"]
    )
    return {
        "run_key": run_key,
        "label": cfg["label"],
        "variant": cfg["variant"],
        "model": cfg["model_key"],
        "mean_mae_e": float(np.mean([v["mae_e"] for v in results.values()])),
        "mean_mae95_e": float(np.mean([v["mae_95_e"] for v in results.values()])),
        "mean_mae_f": float(np.mean([v["mae_f"] for v in results.values()])),
        "result_csv": str(result_csv),
        "prediction_csv": str(pred_csv),
        "comparison_csv": str(comparison_csv),
        "bar_png": str(bar_png),
    }


def write_summary(finished):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_csv = OUTPUT_ROOT / "phase2_2_paa_attention_ablation_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_key", "model", "variant", "label",
                "mean_mae_e", "mean_mae95_e", "mean_mae_f",
                "result_csv", "prediction_csv", "comparison_csv", "bar_png",
            ],
        )
        writer.writeheader()
        for row in finished:
            writer.writerow({
                **{k: row[k] for k in ["run_key", "model", "variant", "label"]},
                "mean_mae_e": f"{row['mean_mae_e']:.6f}",
                "mean_mae95_e": f"{row['mean_mae95_e']:.6f}",
                "mean_mae_f": f"{row['mean_mae_f']:.6f}",
                "result_csv": row["result_csv"],
                "prediction_csv": row["prediction_csv"],
                "comparison_csv": row["comparison_csv"],
                "bar_png": row["bar_png"],
            })
    print(f"\nSummary saved to: {summary_csv}")
    return summary_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        default="schnet_static,schnet_coord,painn_static,painn_coord,painn_coord_bond",
        help=f"Comma-separated run keys. Available: {','.join(RUNS)}",
    )
    parser.add_argument(
        "--max-attention-rows-per-fold",
        type=int,
        default=200000,
        help="Maximum exported test-set edge attention rows per LOSO fold.",
    )
    args = parser.parse_args()

    selected = [x.strip() for x in args.runs.split(",") if x.strip()]
    bad = [x for x in selected if x not in RUNS]
    if bad:
        raise ValueError(f"Unknown run keys: {bad}. Available: {sorted(RUNS)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    finished = []
    for run_key in selected:
        finished.append(run_one(run_key, RUNS[run_key], args.max_attention_rows_per_fold))
    write_summary(finished)


if __name__ == "__main__":
    main()
