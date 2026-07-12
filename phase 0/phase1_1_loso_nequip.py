# -*- coding: utf-8 -*-
"""
Phase 1.1: LOSO Baseline -- NequIP

Same LOSO logic as SchNet/PaiNN Phase 1.1. This script prepares official
NequIP-style YAML configs for each held-out system and stores all artifacts
under phase 1/loso_nequip_output.
"""
import argparse
import os
import shutil
import sys
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
OUTPUT_DIR = Path(r"D:\lunwen\2.1sci\phase 1\loso_nequip_output")
RANDOM_SEED = 42
EPOCHS = 150
BATCH_SIZE = 32
LR = 1e-3
CUTOFF = 5.0
HIDDEN_DIM = 128
L_MAX = 1


def require_nequip():
    exe = shutil.which("nequip-train")
    if exe is None:
        raise RuntimeError(
            "Could not find nequip-train. Install NequIP in your training environment, "
            "then rerun this script."
        )
    try:
        from nequip.ase import NequIPCalculator
    except ImportError as exc:
        raise RuntimeError(
            "Could not import nequip.ase.NequIPCalculator. "
            "Install/activate NequIP before running this baseline."
        ) from exc
    return exe, NequIPCalculator


def write_nequip_config(path, fold_dir, split_paths, args):
    text = f"""# Auto-generated LOSO NequIP config for {fold_dir.name}
root: {str(fold_dir).replace(os.sep, '/')}
run_name: nequip_{fold_dir.name}
seed: {args.random_seed}
dataset_seed: {args.random_seed}
append: false
default_dtype: float32
allow_tf32: true

dataset: ase
dataset_file_name: {str(split_paths['train']).replace(os.sep, '/')}
validation_dataset: ase
validation_dataset_file_name: {str(split_paths['val']).replace(os.sep, '/')}
ase_args:
  format: extxyz
key_mapping:
  energy: total_energy
  forces: forces

r_max: {args.cutoff}
num_layers: 3
l_max: {args.l_max}
parity: true
num_features: {args.hidden_dim}
nonlinearity_type: gate

batch_size: {args.batch_size}
max_epochs: {args.epochs}
learning_rate: {args.lr}
optimizer_name: Adam
optimizer_amsgrad: true
lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 10
lr_scheduler_factor: 0.5
early_stopping_patiences:
  validation_loss: 30
early_stopping_lower_bounds:
  LR: 1.0e-6

loss_coeffs:
  total_energy:
    - 1.0
    - PerAtomMSELoss
  forces:
    - 0.5
    - MSELoss
metrics_components:
  - - total_energy
    - mae
  - - forces
    - mae
"""
    path.write_text(text, encoding="utf-8")
    return path


def find_nequip_model(fold_dir):
    fold_dir = Path(fold_dir)
    candidates = sorted(fold_dir.rglob("deployed_model.pth"))
    candidates += sorted(fold_dir.rglob("best_model.pth"))
    candidates += sorted(fold_dir.rglob("*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No NequIP .pth model found in {fold_dir}. "
            "If your NequIP version requires deployment, deploy the best checkpoint and rerun evaluation."
        )
    return candidates[-1]


def evaluate_nequip(model_path, test_samples, output_csv):
    _, NequIPCalculator = require_nequip()
    calc = NequIPCalculator.from_deployed_model(model_path=str(model_path), device="cuda")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--cutoff", type=float, default=CUTOFF)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--l-max", type=int, default=L_MAX)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--prepare-only", action="store_true", help="Only export LOSO extxyz splits/configs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exe, _ = require_nequip()
    print(f"Using NequIP executable: {exe}")
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
        config_path = write_nequip_config(fold_dir / "nequip_loso_config.yaml", fold_dir, split_paths, args)
        print(f"  Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
        print(f"  Config: {config_path}")
        if args.prepare_only:
            continue

        train_time = run_command([exe, str(config_path)], cwd=fold_dir, log_path=fold_dir / "nequip_train.log")
        model_path = find_nequip_model(fold_dir)
        mae_e, mae_95_e, mae_f = evaluate_nequip(
            model_path, test_samples, fold_dir / f"nequip_predictions_{test_system}.csv"
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
            "loso_nequip_results.csv",
            "loso_nequip_results_bar.png",
            results,
            "NequIP LOSO",
        )
        print(f"\nResults saved to: {csv_path}")
    else:
        print("\nPrepare-only complete. LOSO extxyz splits/configs were written.")


if __name__ == "__main__":
    main()
