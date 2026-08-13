"""End-to-end strict LOSO training and nested routing for NCRS."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .core import (
    RANDOM_SEED,
    branch_configs,
    generate_oof_predictions,
    load_branch_samples,
    prediction_columns,
    train_final_and_predict,
)
from .fusion import fit_policy_path
from .router import candidate_rules, select_rule
from .stacking import fit_gate_path


def _validate_branch_alignment(samples_by_branch: dict[str, list[dict]]) -> None:
    """Ensure all expert views refer to exactly the same labelled structures."""
    reference_key = next(iter(samples_by_branch))
    reference = {sample["key"]: sample for sample in samples_by_branch[reference_key]}
    for branch, samples in samples_by_branch.items():
        observed = {sample["key"]: sample for sample in samples}
        if set(observed) != set(reference):
            missing = sorted(set(reference).difference(observed))[:3]
            extra = sorted(set(observed).difference(reference))[:3]
            raise ValueError(f"Sample-key mismatch for {branch}; missing={missing}, extra={extra}")
        for key, sample in observed.items():
            if sample["system_id"] != reference[key]["system_id"] or not np.isclose(sample["y"], reference[key]["y"]):
                raise ValueError(f"Metadata or energy mismatch at {key} for {branch}")


def _rows_from_predictions(samples: list[dict], prediction_by_key: dict[str, dict], columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(samples, key=lambda sample: sample["key"])
    y = np.asarray([sample["y"] for sample in ordered], dtype=np.float64)
    predictions = np.asarray([[prediction_by_key[sample["key"]][column] for column in columns] for sample in ordered], dtype=np.float64)
    return y, predictions


def _merge_oof_predictions(oof_by_branch: dict[str, dict[str, dict[str, float]]], columns: list[str]) -> dict[str, dict[str, float]]:
    """Flatten branch-keyed OOF predictions into one row per structure."""
    merged: dict[str, dict[str, float]] = {}
    for branch_rows in oof_by_branch.values():
        for key, values in branch_rows.items():
            merged.setdefault(key, {}).update(values)
    for key, values in merged.items():
        missing = set(columns).difference(values)
        if missing:
            raise ValueError(f"Missing OOF predictions for {key}: {sorted(missing)}")
    return merged


def _write_predictions(path: Path, samples: list[dict], prediction_by_key: dict[str, dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "system_id", "sample_id", "y_true", *columns])
        writer.writeheader()
        for sample in sorted(samples, key=lambda item: item["key"]):
            writer.writerow({
                "key": sample["key"], "system_id": sample["system_id"], "sample_id": sample["sample_id"],
                "y_true": f"{sample['y']:.10f}", **{column: f"{prediction_by_key[sample['key']][column]:.10f}" for column in columns},
            })


def _merge_test_predictions(reference_meta: dict, by_branch: dict[str, dict], columns: list[str]) -> dict[str, dict]:
    merged = {key: {} for key in reference_meta["keys"]}
    for branch_values in by_branch.values():
        if branch_values["meta"]["keys"] != reference_meta["keys"]:
            raise ValueError("Expert test predictions are not in the same sample order.")
        for column, values in branch_values["predictions"].items():
            for key, value in zip(reference_meta["keys"], values):
                merged[key][column] = float(value)
    for key, values in merged.items():
        missing = set(columns).difference(values)
        if missing:
            raise ValueError(f"Missing final predictions for {key}: {sorted(missing)}")
    return merged


def run_strict_loso(
    dataset_roots: dict[str, str | Path],
    output_dir: str | Path,
    seeds: tuple[int, int, int] = (42, 123, 456),
    device: str | None = None,
    router_alpha: float = 0.25,
) -> dict:
    """Train the complete final NCRS workflow on already processed private data.

    Outputs include fold-level OOF/test prediction tables, per-path metrics, and
    a nested-router summary. The outer held-out labels are never used while
    selecting the rule for that outer system.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    branches = branch_configs(dataset_roots, seeds=seeds)
    columns = prediction_columns(branches)
    samples_by_branch = load_branch_samples(branches)
    _validate_branch_alignment(samples_by_branch)
    reference_samples = samples_by_branch[branches[0]["key"]]
    systems = sorted({sample["system_id"] for sample in reference_samples})
    if len(systems) < 3:
        raise ValueError("Nested LOSO routing requires at least three distinct systems.")
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if run_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    fold_results: dict[str, dict] = {}
    for outer_system in systems:
        fold_dir = out / f"fold_{outer_system}"
        fold_dir.mkdir(exist_ok=True)
        print(f"\nStrict LOSO outer system: {outer_system}", flush=True)
        oof_by_branch: dict[str, dict] = {}
        final_by_branch: dict[str, dict] = {}
        reference_train, reference_test = None, None
        reference_meta = None
        for branch in branches:
            branch_samples = samples_by_branch[branch["key"]]
            train_pool = [sample for sample in branch_samples if sample["system_id"] != outer_system]
            test_pool = [sample for sample in branch_samples if sample["system_id"] == outer_system]
            if not train_pool or not test_pool:
                raise ValueError(f"Invalid fold {outer_system} for {branch['label']}")
            if reference_train is None:
                reference_train, reference_test = train_pool, test_pool
            print(f"  {branch['label']}: OOF training", flush=True)
            oof_by_branch[branch["key"]] = generate_oof_predictions(branch, train_pool, run_device, outer_system)
            print(f"  {branch['label']}: final training", flush=True)
            meta, predictions = train_final_and_predict(branch, train_pool, test_pool, run_device, fold_dir, outer_system)
            if reference_meta is None:
                reference_meta = meta
            final_by_branch[branch["key"]] = {"meta": meta, "predictions": predictions}

        test_by_key = _merge_test_predictions(reference_meta, final_by_branch, columns)
        oof_by_key = _merge_oof_predictions(oof_by_branch, columns)
        _write_predictions(fold_dir / "oof_predictions.csv", reference_train, oof_by_key, columns)
        _write_predictions(fold_dir / "test_predictions.csv", reference_test, test_by_key, columns)
        y_oof, predictions_oof = _rows_from_predictions(reference_train, oof_by_key, columns)
        y_test, predictions_test = _rows_from_predictions(reference_test, test_by_key, columns)
        gate = fit_gate_path(y_oof, predictions_oof, y_test, predictions_test, columns, run_device)
        policy = fit_policy_path(outer_system, y_oof, predictions_oof, y_test, predictions_test, columns)
        fold_results[outer_system] = {
            "gate": {"mae": gate["mae"], "mae95": gate["mae95"]},
            "policy": {"mae": policy["mae"], "mae95": policy["mae95"]},
            "context": policy["context"],
            "ood_score": policy["ood_score"],
            "policy_strategy": policy["selected_strategy"],
            "n_test": len(y_test),
        }
        np.savez_compressed(
            fold_dir / "fusion_predictions.npz",
            y_true=y_test, gate_prediction=gate["prediction"], policy_prediction=policy["prediction"], gate_weights=gate["weights"],
        )

    final_rows = []
    rules = candidate_rules()
    for outer_system in systems:
        selected_rule, inner_rows = select_rule(outer_system, fold_results, alpha=router_alpha)
        outer = fold_results[outer_system]
        choice = rules[selected_rule](outer_system, outer["context"], outer["ood_score"])
        chosen = outer[choice]
        final_rows.append({
            "system": outer_system,
            "selected_rule": selected_rule,
            "selected_path": choice,
            "mae": chosen["mae"],
            "mae95": chosen["mae95"],
            "context": outer["context"],
            "ood_score": outer["ood_score"],
            "n_test": outer["n_test"],
            "inner_rule_score": float(np.mean([row["mae"] + router_alpha * row["mae95"] for row in inner_rows])),
        })

    summary = {
        "protocol": "strict LOSO with inner-system nested routing",
        "seeds": list(seeds),
        "router_alpha": router_alpha,
        "mean_ncrs_mae": float(np.mean([row["mae"] for row in final_rows])),
        "mean_ncrs_mae95": float(np.mean([row["mae95"] for row in final_rows])),
        "mean_gate_mae": float(np.mean([fold_results[system]["gate"]["mae"] for system in systems])),
        "mean_policy_mae": float(np.mean([fold_results[system]["policy"]["mae"] for system in systems])),
        "folds": final_rows,
    }
    with (out / "ncrs_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (out / "ncrs_fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    return summary
