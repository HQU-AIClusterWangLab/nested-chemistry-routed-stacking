# -*- coding: utf-8 -*-
"""
Phase 2.1e: dynamic physical-feature ablations for SchNet/PaiNN.

Runs recommended feature variants:
  - static + coordination
  - static + coordination + bond mean/std
  - static + coordination + bond mean/std/min/max
  - static + coordination + bond mean/std + Mulliken charge (oracle/leakage-check only)

Outputs are stored under phase 2/dynamic_phys_ablation_output.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase2_phys_loso_common import (  # noqa: E402
    BATCH_SIZE_PAINN,
    BATCH_SIZE_SCHNET,
    PaiNNPhys,
    SchNetPhys,
    run_loso_experiment,
)


ROOT = Path(r"D:\lunwen\2.1sci")
OUTPUT_ROOT = ROOT / "phase 2" / "dynamic_phys_ablation_output"

DATASETS = {
    "coord": ROOT / "phase 0" / "dataset" / "processed_dynamic_coord",
    "coord_bond": ROOT / "phase 0" / "dataset" / "processed_dynamic_coord_bond",
    "full_dynamic": ROOT / "phase 0" / "dataset" / "processed_dynamic",
    "mulliken_oracle": ROOT / "phase 0" / "dataset" / "processed_dynamic_mulliken_oracle",
}

MODELS = {
    "schnet": {
        "label": "SchNet",
        "class": SchNetPhys,
        "batch_size": BATCH_SIZE_SCHNET,
        "baseline_csv": ROOT / "phase 1" / "loso_schnet_output" / "loso_results.csv",
        "static_csv": ROOT / "phase 2" / "loso_schnet_phys_output" / "loso_schnet_phys_results.csv",
    },
    "painn": {
        "label": "PaiNN",
        "class": PaiNNPhys,
        "batch_size": BATCH_SIZE_PAINN,
        "baseline_csv": ROOT / "phase 1" / "loso_painn_output" / "loso_painn_results.csv",
        "static_csv": ROOT / "phase 2" / "loso_painn_phys_output" / "loso_painn_phys_results.csv",
    },
}


def read_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "system": row["Left_Out_System"],
                "mae_e": float(row["MAE_E_eV"]),
                "mae_95": float(row["MAE_95_E_eV"]),
                "mae_f": float(row["MAE_F_eV_A"]),
            })
    return rows


def resolve_dataset_dir(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists():
        return dataset_dir
    candidates = sorted(
        dataset_dir.parent.glob(f"{dataset_dir.name}_20*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"Dataset {dataset_dir} not found; using latest timestamped directory {candidates[0]}")
        return candidates[0]
    return dataset_dir


def missing_dataset_message(missing):
    lines = [
        "Missing required ablation dataset(s):",
        *[f"  - {path}" for path in missing],
        "",
        "Build them first with:",
        '  python "phase 0\\phase5_dynamic_phys_features.py" --variant all',
        "",
        "Expected output directories:",
        "  phase 0\\dataset\\processed_dynamic_coord",
        "  phase 0\\dataset\\processed_dynamic_coord_bond",
        "  phase 0\\dataset\\processed_dynamic",
        "  phase 0\\dataset\\processed_dynamic_mulliken_oracle",
    ]
    return "\n".join(lines)


def build_ablation_datasets():
    phase5_script = ROOT / "phase 0" / "phase5_dynamic_phys_features.py"
    command = [sys.executable, str(phase5_script), "--variant", "all", "--allow-overwrite"]
    print("\nAblation datasets are missing. Building them with current Python:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


def write_summary(output_root, finished_runs):
    summary_path = output_root / "dynamic_phys_ablation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "variant", "mean_mae_e", "mean_mae95_e", "mean_mae_f",
                "result_csv", "note",
            ],
        )
        writer.writeheader()
        for item in finished_runs:
            rows = read_results(item["result_csv"])
            writer.writerow({
                "model": item["model"],
                "variant": item["variant"],
                "mean_mae_e": f"{sum(r['mae_e'] for r in rows) / len(rows):.6f}",
                "mean_mae95_e": f"{sum(r['mae_95'] for r in rows) / len(rows):.6f}",
                "mean_mae_f": f"{sum(r['mae_f'] for r in rows) / len(rows):.6f}",
                "result_csv": str(item["result_csv"]),
                "note": "oracle/leakage-check only" if item["variant"] == "mulliken_oracle" else "",
            })
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="schnet,painn", help="Comma-separated: schnet,painn")
    parser.add_argument(
        "--variants",
        default="coord,coord_bond,full_dynamic,mulliken_oracle",
        help="Comma-separated feature variants",
    )
    args = parser.parse_args()

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    selected_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    required = []
    for variant in selected_variants:
        if variant not in DATASETS:
            raise ValueError(f"Unknown variant '{variant}'. Use one of {sorted(DATASETS)}")
        required.append(resolve_dataset_dir(DATASETS[variant]))
    missing = [p for p in required if not p.exists()]
    if missing:
        build_ablation_datasets()
    missing = [resolve_dataset_dir(DATASETS[v]) for v in selected_variants if not resolve_dataset_dir(DATASETS[v]).exists()]
    if missing:
        raise FileNotFoundError(missing_dataset_message(missing))

    finished = []
    for model_key in selected_models:
        if model_key not in MODELS:
            raise ValueError(f"Unknown model '{model_key}'. Use one of {sorted(MODELS)}")
        cfg = MODELS[model_key]
        for variant in selected_variants:
            if variant not in DATASETS:
                raise ValueError(f"Unknown variant '{variant}'. Use one of {sorted(DATASETS)}")
            dataset_dir = resolve_dataset_dir(DATASETS[variant])
            if not dataset_dir.exists():
                missing = [resolve_dataset_dir(DATASETS[v]) for v in selected_variants]
                missing = [p for p in missing if not p.exists()]
                raise FileNotFoundError(missing_dataset_message(missing))
            run_dir = OUTPUT_ROOT / f"{model_key}_{variant}"
            result_csv_name = f"loso_{model_key}_{variant}_results.csv"
            print(f"\n=== Running {cfg['label']} variant={variant} ===")
            run_loso_experiment(
                model_name=f"{cfg['label']}_{variant}",
                model_class=cfg["class"],
                dataset_dir=str(dataset_dir),
                output_dir=str(run_dir),
                result_csv_name=result_csv_name,
                bar_name=f"loso_{model_key}_{variant}_bar.png",
                model_prefix=f"model_{model_key}_{variant}_loso",
                batch_size=cfg["batch_size"],
                baseline_csv=str(cfg["baseline_csv"]),
                static_csv=str(cfg["static_csv"]) if cfg["static_csv"].exists() else None,
            )
            finished.append({
                "model": model_key,
                "variant": variant,
                "result_csv": run_dir / result_csv_name,
            })

    summary_path = write_summary(OUTPUT_ROOT, finished)
    print(f"\nAblation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
