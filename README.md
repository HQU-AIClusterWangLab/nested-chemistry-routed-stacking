# Nested Chemistry-Routed Stacking

Research code for strict leave-one-system-out (LOSO) potential-energy ranking in heterogeneous doped clusters. The code implements the progression from baseline atomistic models through physics-aware attention, uncertainty diagnostics, nested chemistry-routed stacking, replay screening, and finite-budget DFT candidate selection.

## Scope

The repository contains code only. It deliberately excludes the private cluster database, geometries, DFT outputs, trained checkpoints, intermediate predictions, and manuscript assets. Those materials are central to the underlying data-driven project and are not distributed through this repository.

Consequently, this repository is an implementation and workflow release, not a self-contained benchmark release. The public scripts can be inspected and adapted, but the reported numerical results cannot be reproduced without an independently authorized dataset and the required electronic-structure software.

## Layout

- `phase 0/`: dataset preparation, strict-LOSO baselines, physics-aware branches, UQ diagnostics, and stacking/routing experiments.
- `phase_6/`: deterministic prediction preparation, replay screening, DFT candidate selection, and chemistry-validation packaging.
- `picture/`: scripts used to construct manuscript and supporting-information figures from saved numerical and post-DFT inputs.
- `orca/`: Windows-oriented ORCA and Multiwfn helper scripts for the post-DFT workflow.
- `data/`: documentation only. No scientific data are committed.

The final paper-facing path is:

`strict LOSO -> Nested Chemistry-Routed Stacking -> finite-budget DFT validation`

The central final scripts are `phase 0/phase4_3_nested_router_validation.py`, `phase_6/phase6_0_prepare_nested_router_predictions.py`, `phase_6/phase6_1_replay_screening_final.py`, and `phase_6/phase6_2_select_dft_candidates_final.py`.

## Environment

Create an environment appropriate for the selected branch of the workflow:

```powershell
conda create -n ncrs python=3.10
conda activate ncrs
pip install -r requirements.txt
```

The MACE and NequIP baselines additionally require their framework-specific installations and a compatible PyTorch/PyG build. The DFT and real-space analysis scripts require separately licensed or installed tools, including Gaussian or ORCA, Multiwfn, and related post-processing utilities.

## Private-data Interface

Before executing a data-dependent script, point its input variables to an authorized local dataset and choose an output directory outside the repository. Historical scripts retain some Windows path configuration from the original research environment; these settings must be replaced locally and never committed with data paths or data files.

See [data/README.md](data/README.md) for the required data boundary and expected categories of inputs.

## Data Policy

No raw or processed database, structure collection, DFT log, cube file, checkpoint, prediction table, or generated figure is tracked. The `.gitignore` is intentionally conservative and blocks these categories even if they are later placed inside the repository.

## License

Released under the MIT License. The license applies to code only and grants no access rights to the private database or computational outputs.
