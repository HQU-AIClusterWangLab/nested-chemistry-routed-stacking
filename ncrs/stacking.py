"""Learned gated-stacking path used by NCRS."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.preprocessing import StandardScaler

from .metrics import metric_pair


class GatingNetwork(nn.Module):
    """The manuscript MLP: 15 -> 96 -> 48 -> 9."""

    def __init__(self, input_dim: int, n_models: int) -> None:
        super().__init__()
        if input_dim != 15 or n_models != 9:
            raise ValueError("Final NCRS gate requires 15 inputs and 9 seed predictions.")
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 9),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.net(features), dim=-1)
        return (weights * features[:, :9]).sum(dim=-1), weights


def branch_slices(prediction_columns: list[str]) -> dict[str, list[int]]:
    return {
        "schnet_static_phys": [i for i, name in enumerate(prediction_columns) if name.startswith("schnet_static_phys_")],
        "paa_schnet_coord": [i for i, name in enumerate(prediction_columns) if name.startswith("paa_schnet_coord_")],
        "painn_coord_bond": [i for i, name in enumerate(prediction_columns) if name.startswith("painn_coord_bond_")],
    }


def gate_features(predictions: np.ndarray, prediction_columns: list[str]) -> np.ndarray:
    """Nine seed predictions plus three branch means and three branch variances."""
    groups = branch_slices(prediction_columns)
    if predictions.shape[1] != 9 or any(len(indices) != 3 for indices in groups.values()):
        raise ValueError("NCRS requires three seeds for each of three expert branches.")
    means = [predictions[:, indices].mean(axis=1) for indices in groups.values()]
    variances = [predictions[:, indices].var(axis=1) for indices in groups.values()]
    return np.concatenate([predictions, np.stack(means, axis=1), np.stack(variances, axis=1)], axis=1)


def train_gate(
    features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    seed: int = 42,
    epochs: int = 400,
    patience: int = 50,
) -> tuple[GatingNetwork, StandardScaler, float]:
    """Fit the gate using OOF predictions only; return the early-stopped model."""
    if len(targets) < 5:
        raise ValueError("At least five OOF samples are required for gate training.")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(targets))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    train_indices, val_indices = indices[:split], indices[split:]
    if len(val_indices) == 0:
        val_indices = train_indices[-1:]
        train_indices = train_indices[:-1]

    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(features), dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    model = GatingNetwork(features.shape[1], 9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long, device=device)
    val_index_tensor = torch.tensor(val_indices, dtype=torch.long, device=device)
    best_state, best_loss, stalled = None, float("inf"), 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction, _ = model(x[train_index_tensor])
        loss = functional.l1_loss(prediction, y[train_index_tensor])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction, _ = model(x[val_index_tensor])
            validation_loss = functional.l1_loss(validation_prediction, y[val_index_tensor]).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
        if stalled >= patience:
            break

    if best_state is None:
        raise RuntimeError("Gate did not retain a valid early-stopping state.")
    model.load_state_dict(best_state)
    return model, scaler, float(best_loss)


def predict_gate(model: GatingNetwork, scaler: StandardScaler, features: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(scaler.transform(features), dtype=torch.float32, device=device)
        prediction, weights = model(x)
    return prediction.cpu().numpy(), weights.cpu().numpy()


def fit_gate_path(
    y_oof: np.ndarray,
    predictions_oof: np.ndarray,
    y_test: np.ndarray,
    predictions_test: np.ndarray,
    prediction_columns: list[str],
    device: torch.device,
) -> dict:
    model, scaler, validation_loss = train_gate(gate_features(predictions_oof, prediction_columns), y_oof, device)
    predicted, weights = predict_gate(model, scaler, gate_features(predictions_test, prediction_columns), device)
    mae, tail_mae = metric_pair(y_test, predicted)
    return {
        "prediction": predicted,
        "weights": weights,
        "mae": mae,
        "mae95": tail_mae,
        "gate_validation_loss": validation_loss,
    }
