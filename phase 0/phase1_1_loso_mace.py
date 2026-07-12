# -*- coding: utf-8 -*-
"""
Phase 1.1: LOSO Baseline -- MACE

Same LOSO logic as SchNet/PaiNN Phase 1.1. This script uses the official MACE
CLI/package when available. It exports each fold to extxyz and stores all
artifacts under phase 1/loso_mace_output.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase1_loso_external_utils import (  # noqa: E402
    load_all_pt_files,
    loso_split,
    mae95,
    run_command,
    samples_to_atoms,
    write_loso_results,
    write_prediction_csv,
    write_split_files,
)


DATASET_DIR = Path(r"D:\lunwen\2.1sci\phase 0\dataset\processed")
OUTPUT_DIR = Path(r"D:\lunwen\2.1sci\phase 1\loso_mace_output")
RANDOM_SEED = 42
EPOCHS = 150
BATCH_SIZE = 64
LR = 1e-3
CUTOFF = 5.0
HIDDEN_DIM = 128
MAX_L = 1


def require_mace():
    exe = shutil.which("mace_run_train")
    if exe is None:
        raise RuntimeError(
            "Could not find mace_run_train. Install MACE in your training environment, "
            "then rerun this script."
        )
    try:
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise RuntimeError(
            "Could not import mace.calculators.MACECalculator. "
            "Install/activate MACE before running this baseline."
        ) from exc
    return exe, MACECalculator


def find_mace_model(fold_dir):
    candidates = sorted(Path(fold_dir).glob("*.model"))
    candidates += sorted(Path(fold_dir).glob("*.pt"))
    if not candidates:
        candidates = sorted(Path(fold_dir).rglob("*.model"))
        candidates += sorted(Path(fold_dir).rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No trained MACE model found in {fold_dir}")
    return candidates[-1]


def evaluate_mace(model_path, test_samples, output_csv):
    _, MACECalculator = require_mace()
    calc = MACECalculator(model_paths=str(model_path), device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")
    atoms_list = samples_to_atoms(test_samples)
    y_true, y_pred, force_mae, sample_ids = [], [], [], []
    for sample, atoms in zip(test_samples, atoms_list):
        atoms.calc = calc
        pred_e = atoms.get_potential_energy()
        pred_f = atoms.get_forces()
        true_f = sample["forces"]
        y_true.append(sample["y"])
        y_pred.append(pred_e)
        force_mae.append(np.mean(np.abs(pred_f - true_f)))
        sample_ids.append(sample["sample_id"])
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    force_mae = np.asarray(force_mae, dtype=float)
    abs_e = np.abs(y_pred - y_true)
    write_prediction_csv(output_csv, sample_ids, y_true, y_pred, force_mae)
    return float(abs_e.mean()), mae95(abs_e), float(force_mae.mean())


def build_train_command(exe, fold_dir, split_paths, args):
    # MACE CLI option names follow current public MACE examples. Keep the generated
    # command in train.log so any local version mismatch is easy to adjust.
    return [
        exe,
        "--name", f"mace_loso_{fold_dir.name}",
        "--train_file", str(split_paths["train"]),
        "--valid_file", str(split_paths["val"]),
        "--test_file", str(split_paths["test"]),
        "--E0s", "average",
        "--model", "MACE",
        "--num_channels", str(args.hidden_dim),
        "--max_L", str(args.max_l),
        "--r_max", str(args.cutoff),
        "--batch_size", str(args.batch_size),
        "--max_num_epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--seed", str(args.random_seed),
        "--energy_key", "energy",
        "--forces_key", "forces",
        "--default_dtype", "float32",
        "--device", args.device,
        "--save_cpu",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--cutoff", type=float, default=CUTOFF)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--max-l", type=int, default=MAX_L)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--prepare-only", action="store_true", help="Only export LOSO extxyz splits")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exe, _ = require_mace()
    print(f"Using MACE executable: {exe}")
    print(f"Output directory: {output_dir}")

    all_samples = load_all_pt_files(args.dataset_dir)
    systems = sorted({s["system_id"] for s in all_samples})
    print(f"Loaded {len(all_samples)} samples across {len(systems)} systems: {systems}")

    results = {}
    for fold_idx, test_system in enumerate(systems, start=1):
        print(f"\nFold {fold_idx}/{len(systems)}: leave out {test_system}")
        fold_dir = output_dir / f"fold_loso_{test_system}"
        train_samples, val_samples, test_samples = loso_split(all_samples, test_system, args.random_seed)
        split_paths = write_split_files(train_samples, val_samples, test_samples, fold_dir)
        print(f"  Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
        if args.prepare_only:
            continue

        command = build_train_command(exe, fold_dir, split_paths, args)
        train_time = run_command(command, cwd=fold_dir, log_path=fold_dir / "mace_train.log")
        model_path = find_mace_model(fold_dir)
        mae_e, mae_95_e, mae_f = evaluate_mace(
            model_path, test_samples, fold_dir / f"mace_predictions_{test_system}.csv"
        )
        results[test_system] = {
            "mae_e": mae_e,
            "mae_95_e": mae_95_e,
            "mae_f": mae_f,
            "n_test": len(test_samples),
            "train_time": train_time,
        }
        print(f"  Result: MAE(E)={mae_e:.6f}, MAE_95(E)={mae_95_e:.6f}, MAE(F)={mae_f:.6f}")

    if not args.prepare_only:
        csv_path = write_loso_results(
            output_dir,
            "loso_mace_results.csv",
            "loso_mace_results_bar.png",
            results,
            "MACE LOSO",
        )
        print(f"\nResults saved to: {csv_path}")
    else:
        print("\nPrepare-only complete. LOSO extxyz splits were written.")


if __name__ == "__main__":
    main()
