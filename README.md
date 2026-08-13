# Nested Chemistry-Routed Stacking

This repository contains the final training and evaluation code for **Nested Chemistry-Routed Stacking (NCRS)**, a leakage-safe strict leave-one-system-out (LOSO) workflow for potential-energy ranking across chemically heterogeneous doped clusters.

NCRS is deliberately released as code only. It does not contain the research database, processed structures, DFT inputs or outputs, checkpoints, prediction tables, figure scripts, or historical experiments.

## What is included

The public workflow is exactly:

```text
processed private data
  -> strict LOSO expert training (three seeds per expert)
  -> OOF predictions for each outer training pool
  -> learned gated stacking and a predeclared chemistry/context policy
  -> inner-system rule selection using MAE + 0.25 * MAE95
  -> final NCRS prediction on the held-out system
```

The three final expert branches are:

1. `SchNet-static-phys`
2. `PAA-SchNet-coord`
3. `PaiNN-coord-bond`

The learned gate is the manuscript architecture `15 -> 96 -> 48 -> 9`. NCRS does not train these paths serially: for each outer fold, it chooses between the learned gate and the rule-based chemistry path using results from the remaining inner systems only.

## Installation

Create an isolated environment with a PyTorch build appropriate for the target GPU, then install the small set of NCRS dependencies:

```powershell
conda create -n ncrs python=3.10
conda activate ncrs
pip install -r requirements.txt
```

## Data Interface

NCRS consumes three already processed private `.pt` dataset roots. It does not provide a dataset-building pipeline. The required folder layout and tensor fields are documented in [examples/dataset_schema.md](examples/dataset_schema.md).

The three roots must be aligned one-to-one by `system-id/sample-id`; the runner stops if keys, system labels, or energy labels disagree. Keep all private inputs and outputs outside this repository.

## Run

```powershell
python scripts/run_strict_loso.py `
  --schnet-data D:\private_data\processed `
  --paa-data D:\private_data\processed_dynamic_coord `
  --painn-data D:\private_data\processed_dynamic_coord_bond `
  --output D:\private_results\ncrs_loso_run `
  --device cuda
```

The runner writes fold-level OOF and held-out prediction tables, fusion-path predictions, trained fold checkpoints, `ncrs_fold_metrics.csv`, and `ncrs_summary.json` only to the explicit external output directory.

## Reproducibility Boundary

The public code documents and implements the final NCRS method. Numerical reproduction of the manuscript requires an independently authorized dataset with the same preprocessing, labels, graph construction, and split metadata. This repository grants no rights to the private database or derived computational outputs.

## License

MIT License, code only.
