# -*- coding: utf-8 -*-
"""
Phase 5: build dynamic physical features for Pi/Bij experiments.

Default model input:
  old static x (5 dims) + coordination number + local bond mean/std/min/max.

Explanation-only fields:
  Mulliken charges parsed from Gaussian logs. They are saved separately as
  data.mulliken_charge and are not appended to data.x by default.
"""
import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_ROOT = Path(r"D:\lunwen\2.1sci")
DEFAULT_INPUT = DEFAULT_ROOT / "phase 0" / "dataset" / "processed"
DEFAULT_LOGS = DEFAULT_ROOT / "phase 0" / "dataset"
DEFAULT_OUTPUT = DEFAULT_ROOT / "phase 0" / "dataset" / "processed_dynamic"


FEATURE_VARIANTS = {
    "coord": {
        "suffix": "processed_dynamic_coord",
        "dynamic_indices": [0],
        "names": ["coordination_number"],
        "append_mulliken": False,
    },
    "coord_bond": {
        "suffix": "processed_dynamic_coord_bond",
        "dynamic_indices": [0, 1, 2],
        "names": ["coordination_number", "local_bond_length_mean", "local_bond_length_std"],
        "append_mulliken": False,
    },
    "full": {
        "suffix": "processed_dynamic",
        "dynamic_indices": [0, 1, 2, 3, 4],
        "names": [
            "coordination_number",
            "local_bond_length_mean",
            "local_bond_length_std",
            "local_bond_length_min",
            "local_bond_length_max",
        ],
        "append_mulliken": False,
    },
    "mulliken_oracle": {
        "suffix": "processed_dynamic_mulliken_oracle",
        "dynamic_indices": [0, 1, 2],
        "names": ["coordination_number", "local_bond_length_mean", "local_bond_length_std"],
        "append_mulliken": True,
    },
}


FEATURE_NAMES_DYNAMIC = {
    0: "normalized_atomic_number",
    1: "electronegativity",
    2: "covalent_radius",
    3: "valence_electrons",
    4: "first_ionization_energy",
    5: "coordination_number",
    6: "local_bond_length_mean",
    7: "local_bond_length_std",
    8: "local_bond_length_min",
    9: "local_bond_length_max",
}


def sample_id_from_pt(path):
    name = Path(path).stem
    return name[:-7] if name.endswith("_sample") else name


def parse_mulliken_charges(log_path, n_atoms):
    """Return the first Mulliken charge block as a float array, or None."""
    if not log_path.exists():
        return None
    lines = log_path.read_text(errors="ignore").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if "Mulliken charges:" in line:
            start = idx + 1
            break
    if start is None:
        return None

    charges = []
    pattern = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,3})\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
    for line in lines[start:]:
        if "Sum of Mulliken charges" in line:
            break
        match = pattern.match(line)
        if match:
            charges.append(float(match.group(3)))
            if len(charges) == n_atoms:
                break
    if len(charges) != n_atoms:
        return None
    return np.asarray(charges, dtype=np.float32)


def compute_local_geometry(pos, edge_index):
    """Compute coordination and local bond length statistics from graph edges."""
    n_atoms = pos.shape[0]
    if edge_index.numel() == 0:
        zeros = torch.zeros(n_atoms, 1, dtype=pos.dtype)
        return torch.cat([zeros, zeros, zeros, zeros, zeros], dim=1)

    src = edge_index[0].long()
    dst = edge_index[1].long()
    distances = torch.linalg.norm(pos[src] - pos[dst], dim=1)

    coord = torch.zeros(n_atoms, dtype=pos.dtype)
    bond_sum = torch.zeros(n_atoms, dtype=pos.dtype)
    bond_sq_sum = torch.zeros(n_atoms, dtype=pos.dtype)
    bond_min = torch.full((n_atoms,), float("inf"), dtype=pos.dtype)
    bond_max = torch.zeros(n_atoms, dtype=pos.dtype)

    for atom_idx, dist in zip(dst.tolist(), distances):
        coord[atom_idx] += 1.0
        bond_sum[atom_idx] += dist
        bond_sq_sum[atom_idx] += dist * dist
        bond_min[atom_idx] = torch.minimum(bond_min[atom_idx], dist)
        bond_max[atom_idx] = torch.maximum(bond_max[atom_idx], dist)

    safe_coord = torch.clamp(coord, min=1.0)
    mean = bond_sum / safe_coord
    var = torch.clamp(bond_sq_sum / safe_coord - mean * mean, min=0.0)
    std = torch.sqrt(var)
    bond_min = torch.where(torch.isfinite(bond_min), bond_min, torch.zeros_like(bond_min))

    return torch.stack([coord, mean, std, bond_min, bond_max], dim=1)


def normalize_dynamic_features(dynamic_x, stats=None):
    if stats is None:
        mean = dynamic_x.mean(dim=0, keepdim=True)
        std = dynamic_x.std(dim=0, keepdim=True)
        stats = {"mean": mean, "std": torch.clamp(std, min=1e-8)}
    return (dynamic_x - stats["mean"]) / stats["std"], stats


def build_one_sample(
    pt_path,
    input_root,
    log_root,
    output_root,
    append_mulliken=False,
    dynamic_indices=None,
    dynamic_names=None,
):
    data = torch.load(pt_path, weights_only=False)
    old_x = data.x.float()
    geom_x = compute_local_geometry(data.pos.float(), data.edge_index.long())
    if dynamic_indices is None:
        dynamic_indices = [0, 1, 2, 3, 4]
    selected_geom_x = geom_x[:, dynamic_indices]

    x_new = torch.cat([old_x, selected_geom_x], dim=1)
    if append_mulliken:
        log_path = log_root / f"{sample_id_from_pt(pt_path)}.log"
        charges = parse_mulliken_charges(log_path, old_x.shape[0])
        if charges is not None:
            x_new = torch.cat([x_new, torch.tensor(charges).view(-1, 1)], dim=1)

    data.x_static = old_x
    data.dynamic_phys = geom_x
    data.dynamic_phys_selected = selected_geom_x
    data.x = x_new
    feature_names = {i: FEATURE_NAMES_DYNAMIC[i] for i in range(5)}
    if dynamic_names is None:
        dynamic_names = [FEATURE_NAMES_DYNAMIC[i + 5] for i in dynamic_indices]
    for offset, name in enumerate(dynamic_names, start=5):
        feature_names[offset] = name
    data.feature_names = feature_names
    if append_mulliken:
        data.feature_names[len(data.feature_names)] = "mulliken_charge_model_input"

    log_path = log_root / f"{sample_id_from_pt(pt_path)}.log"
    charges = parse_mulliken_charges(log_path, old_x.shape[0])
    if charges is not None:
        data.mulliken_charge = torch.tensor(charges, dtype=torch.float32)
        data.mulliken_source = str(log_path)

    rel = Path(pt_path).relative_to(input_root)
    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    return out_path, charges is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input processed .pt root")
    parser.add_argument("--logs", default=str(DEFAULT_LOGS), help="Gaussian log root")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output processed_dynamic root")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory",
    )
    parser.add_argument(
        "--append-mulliken",
        action="store_true",
        help="Append Mulliken charge to data.x for explicit leakage-check ablation only",
    )
    parser.add_argument(
        "--variant",
        choices=["full", "coord", "coord_bond", "mulliken_oracle", "all"],
        default="full",
        help="Feature variant to build. 'all' writes every recommended ablation dataset.",
    )
    args = parser.parse_args()

    input_root = Path(args.input)
    log_root = Path(args.logs)
    pt_files = sorted(input_root.glob("*/*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under {input_root}")

    if args.variant == "all":
        variants = ["coord", "coord_bond", "full", "mulliken_oracle"]
    else:
        variants = [args.variant]

    base_output = Path(args.output)
    for variant in variants:
        spec = FEATURE_VARIANTS[variant]
        if args.output == str(DEFAULT_OUTPUT):
            output_root = DEFAULT_ROOT / "phase 0" / "dataset" / spec["suffix"]
        else:
            output_root = base_output
            if args.variant == "all":
                output_root = base_output.with_name(spec["suffix"])

        if output_root.exists() and any(output_root.iterdir()) and not args.allow_overwrite:
            stamped = output_root.with_name(f"{output_root.name}_{time.strftime('%Y%m%d_%H%M%S')}")
            print(f"Output directory is not empty: {output_root}")
            print(f"Writing this run to timestamped directory instead: {stamped}")
            output_root = stamped

        converted = 0
        mulliken_found = 0
        append_mulliken = args.append_mulliken or spec["append_mulliken"]
        print(f"\nBuilding variant '{variant}' -> {output_root}")
        for pt_path in pt_files:
            _, has_mulliken = build_one_sample(
                pt_path,
                input_root,
                log_root,
                output_root,
                append_mulliken=append_mulliken,
                dynamic_indices=spec["dynamic_indices"],
                dynamic_names=spec["names"],
            )
            converted += 1
            mulliken_found += int(has_mulliken)
            if converted % 500 == 0:
                print(f"Converted {converted}/{len(pt_files)}")

        print(f"Done. Converted {converted} samples to {output_root}")
        print(f"Mulliken charges parsed for {mulliken_found}/{converted} samples")
        if append_mulliken:
            print("Mulliken charge WAS appended to data.x. Use this only as oracle/leakage-check ablation.")
        else:
            print("Mulliken charge excluded from data.x and saved only as explanation metadata.")


if __name__ == "__main__":
    main()
