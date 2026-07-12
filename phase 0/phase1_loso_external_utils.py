# -*- coding: utf-8 -*-
"""
Shared utilities for Phase 1 external baseline projects.

The MACE and NequIP scripts use the same LOSO split as phase1_1_loso_schnet.py:
leave one system out, then split the remaining systems into train/val by
(system_id, group_id) with a 90/10 group-aware split.
"""
import csv
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "This script must be run in the same environment as your Phase 1 training scripts. "
            "The current Python cannot import torch."
        ) from exc
    return torch


def require_ase():
    try:
        from ase import Atoms
        from ase.io import read, write
    except ImportError as exc:
        raise RuntimeError("ASE is required to export/evaluate external-model datasets.") from exc
    return Atoms, read, write


def load_all_pt_files(dataset_dir):
    torch = require_torch()
    dataset_dir = Path(dataset_dir)
    samples = []
    for system_id in sorted(os.listdir(dataset_dir)):
        sys_dir = dataset_dir / system_id
        if not sys_dir.is_dir():
            continue
        for fname in sorted(os.listdir(sys_dir)):
            if not fname.endswith(".pt"):
                continue
            fpath = sys_dir / fname
            try:
                data = torch.load(fpath, weights_only=False)
            except Exception as exc:
                print(f"Skip unreadable file {fpath}: {exc}")
                continue
            gid = getattr(data, "group_id", 0) if hasattr(data, "group_id") else 0
            sample_id = Path(fname).stem
            if sample_id.endswith("_sample"):
                sample_id = sample_id[:-7]
            samples.append({
                "sample_id": sample_id,
                "pt_path": str(fpath),
                "z": data.atomic_numbers.long().cpu().numpy(),
                "pos": data.pos.float().cpu().numpy(),
                "y": float(data.y.float().item()),
                "forces": data.forces.float().cpu().numpy(),
                "system_id": system_id,
                "group_id": int(gid),
            })
    return samples


def loso_split(all_samples, test_system, random_seed=42):
    test_samples = [s for s in all_samples if s["system_id"] == test_system]
    train_val_samples = [s for s in all_samples if s["system_id"] != test_system]

    rng = random.Random(random_seed)
    train_val_groups = defaultdict(list)
    for sample in train_val_samples:
        train_val_groups[(sample["system_id"], sample["group_id"])].append(sample)
    group_keys = sorted(train_val_groups.keys())
    rng.shuffle(group_keys)
    n_train_groups = int(len(group_keys) * 0.90)
    train_group_set = set(group_keys[:n_train_groups])

    train_samples = [
        s for s in train_val_samples
        if (s["system_id"], s["group_id"]) in train_group_set
    ]
    val_samples = [
        s for s in train_val_samples
        if (s["system_id"], s["group_id"]) not in train_group_set
    ]
    return train_samples, val_samples, test_samples


def samples_to_atoms(samples):
    Atoms, _, _ = require_ase()
    atoms_list = []
    for sample in samples:
        atoms = Atoms(numbers=sample["z"], positions=sample["pos"])
        atoms.info["energy"] = sample["y"]
        atoms.info["REF_energy"] = sample["y"]
        atoms.info["sample_id"] = sample["sample_id"]
        atoms.info["system_id"] = sample["system_id"]
        atoms.info["group_id"] = sample["group_id"]
        atoms.arrays["forces"] = sample["forces"]
        atoms.arrays["REF_forces"] = sample["forces"]
        atoms_list.append(atoms)
    return atoms_list


def write_extxyz(samples, path):
    _, _, write = require_ase()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), samples_to_atoms(samples), format="extxyz")
    return path


def write_split_files(train_samples, val_samples, test_samples, fold_dir):
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": write_extxyz(train_samples, fold_dir / "train.xyz"),
        "val": write_extxyz(val_samples, fold_dir / "val.xyz"),
        "test": write_extxyz(test_samples, fold_dir / "test.xyz"),
    }
    with open(fold_dir / "split_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "n_samples"])
        writer.writeheader()
        writer.writerows([
            {"split": "train", "n_samples": len(train_samples)},
            {"split": "val", "n_samples": len(val_samples)},
            {"split": "test", "n_samples": len(test_samples)},
        ])
    return paths


def run_command(command, cwd, log_path):
    cwd = Path(cwd)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(command) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        log.write(f"\nRETURN_CODE={rc}\n")
        log.write(f"ELAPSED_S={time.time() - started:.1f}\n")
    if rc != 0:
        raise RuntimeError(f"Command failed with return code {rc}. See {log_path}")
    return time.time() - started


def mae95(abs_errors):
    abs_errors = np.asarray(abs_errors, dtype=float)
    k = max(1, int(len(abs_errors) * 0.05))
    return float(np.sort(abs_errors)[-k:].mean())


def write_loso_results(output_dir, csv_name, bar_name, results, title):
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
            row = results[sys_id]
            writer.writerow({
                "Left_Out_System": sys_id,
                "MAE_E_eV": f"{row['mae_e']:.6f}",
                "MAE_95_E_eV": f"{row['mae_95_e']:.6f}",
                "MAE_F_eV_A": f"{row['mae_f']:.6f}",
                "N_Test": row["n_test"],
                "Time_s": f"{row['train_time']:.1f}",
            })
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
        axes[1].set_title(f"{title}: Tail Error")
        axes[2].bar(sys_names, [results[s]["mae_f"] for s in sys_names], color=colors_f)
        axes[2].set_ylabel("MAE (eV/A)")
        axes[2].set_title(f"{title}: MAE Forces")
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(output_dir / bar_name, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Could not create bar chart: {exc}")
    return csv_path


def write_prediction_csv(path, sample_ids, y_true, y_pred, f_mae_per_sample=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "y_true", "y_pred", "abs_error", "force_mae"],
        )
        writer.writeheader()
        for idx, sid in enumerate(sample_ids):
            abs_error = abs(float(y_pred[idx]) - float(y_true[idx]))
            force_mae = "" if f_mae_per_sample is None else f"{float(f_mae_per_sample[idx]):.10f}"
            writer.writerow({
                "sample_id": sid,
                "y_true": f"{float(y_true[idx]):.10f}",
                "y_pred": f"{float(y_pred[idx]):.10f}",
                "abs_error": f"{abs_error:.10f}",
                "force_mae": force_mae,
            })


def get_device():
    try:
        torch = require_torch()
        return "cuda" if torch.cuda.is_available() else "cpu"
    except RuntimeError:
        return "cpu"
