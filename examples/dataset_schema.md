# Processed dataset contract

NCRS intentionally accepts already processed PyTorch `.pt` objects only. This release does not include raw-data conversion, structure generation, electronic-structure calculations, or any private dataset.

Each expert root must use the layout:

```text
<expert-root>/
  <system-id>/
    <sample-id>.pt
```

The three expert roots must contain identical `<system-id>/<sample-id>` keys and identical scalar energy labels. Each serialized object must expose:

| Field | Type | Meaning |
| --- | --- | --- |
| `atomic_numbers` | integer tensor `[n_atoms]` | Atomic numbers, indexed below 100. |
| `pos` | float tensor `[n_atoms, 3]` | Cartesian coordinates in the unit used by the original model. |
| `y` | scalar float tensor | Target potential energy. |
| `forces` | float tensor `[n_atoms, 3]` | Force labels aligned with `pos`. |
| `x` | float tensor `[n_atoms, n_features]` | Expert-specific input features. |
| `edge_index` | integer tensor `[2, n_edges]` | Directed graph connectivity. |
| `group_id` | hashable scalar | Required split unit within a system. Samples sharing a value stay together in inner train/validation and OOF splits. |

The expected feature views are: five static elemental descriptors for `SchNet-static-phys`; those descriptors plus coordination for `PAA-SchNet-coord`; and those descriptors plus coordination, local bond-length mean, and local bond-length standard deviation for `PaiNN-coord-bond`.

No claim of manuscript-number reproduction is made without the authorized private dataset and the exact original preprocessing convention.
