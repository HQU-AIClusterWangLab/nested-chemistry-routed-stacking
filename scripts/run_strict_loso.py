"""Command-line entry point for the complete NCRS strict-LOSO workflow."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete NCRS strict LOSO training on external processed data.")
    parser.add_argument("--schnet-data", required=True, help="Root of SchNet-static-phys .pt files.")
    parser.add_argument("--paa-data", required=True, help="Root of PAA-SchNet-coord .pt files.")
    parser.add_argument("--painn-data", required=True, help="Root of PaiNN-coord-bond .pt files.")
    parser.add_argument("--output", required=True, help="External directory for all predictions, checkpoints, and metrics.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu. Defaults to cuda when available.")
    parser.add_argument("--seeds", nargs=3, type=int, default=[42, 123, 456], metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--router-alpha", type=float, default=0.25, help="Tail-risk weight in MAE + alpha * MAE95.")
    args = parser.parse_args()
    from ncrs.pipeline import run_strict_loso
    summary = run_strict_loso(
        dataset_roots={
            "schnet_static_phys": args.schnet_data,
            "paa_schnet_coord": args.paa_data,
            "painn_coord_bond": args.painn_data,
        },
        output_dir=args.output,
        seeds=tuple(args.seeds),
        device=args.device,
        router_alpha=args.router_alpha,
    )
    print("\nNCRS strict LOSO complete")
    print(f"Mean MAE: {summary['mean_ncrs_mae']:.6f}")
    print(f"Mean MAE95: {summary['mean_ncrs_mae95']:.6f}")


if __name__ == "__main__":
    main()
