"""Train ML behavior model: audio features → mouth-group controls.

Fills **missing** transition data (gaps between measured samples, and live
speech with no video clock). Measured track remains authority when present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aiface.behavior.schema import (
    CONTROL_DIM,
    CONTROL_NAMES,
    FEATURE_DIM,
    meta_path,
    model_path,
)


def _load_dataset(dataset: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    data = np.load(dataset, allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    noise = float(np.asarray(data["noise_floor"]).reshape(-1)[0])
    peak = float(np.asarray(data["peak_hint"]).reshape(-1)[0])
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] < CONTROL_DIM:
        pad = np.zeros((len(y), CONTROL_DIM), dtype=np.float64)
        pad[:, : y.shape[1]] = y
        # Derive lip/cavity from openness when only live-vector dims exist.
        if y.shape[1] >= 1:
            pad[:, 3] = -pad[:, 0]
            pad[:, 4] = pad[:, 0]
            pad[:, 7] = pad[:, 0]
            pad[:, 6] = pad[:, 0] * 0.65
        if y.shape[1] >= 3:
            pad[:, 5] = pad[:, 2]
        y = pad
    y = y[:, :CONTROL_DIM]
    if x.shape[1] != FEATURE_DIM:
        raise ValueError(f"bad X shape {x.shape}")
    return x, y, noise, peak


def fit_behavior_model(
    dataset: Path,
    *,
    world_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 17,
) -> dict[str, Any]:
    """Fit MLP on measured transition dataset → behavior_model.joblib."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    x, y, noise, peak = _load_dataset(Path(dataset))
    if len(y) < 8:
        raise RuntimeError(f"behavior dataset too small: {len(y)}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    n_val = max(1, int(round(len(y) * val_fraction)))
    x_val, y_val = x[:n_val], y[:n_val]
    x_train, y_train = x[n_val:], y[n_val:]
    if len(y_train) < 4:
        x_train, y_train = x, y
        x_val, y_val = x, y

    # Oversample open mouths so gaps during speech get better fill.
    open_idx = np.flatnonzero(y_train[:, 0] >= 0.12)
    if open_idx.size:
        extra = rng.choice(
            open_idx, size=min(len(y_train), open_idx.size * 3), replace=True
        )
        x_train = np.concatenate([x_train, x_train[extra]], axis=0)
        y_train = np.concatenate([y_train, y_train[extra]], axis=0)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=1400,
                    random_state=seed,
                    early_stopping=len(y_train) >= 20,
                    validation_fraction=0.15 if len(y_train) >= 20 else 0.1,
                    alpha=1e-3,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    train_pred = np.clip(pipeline.predict(x_train), -1.0, 1.0)
    val_pred = np.clip(pipeline.predict(x_val), -1.0, 1.0)
    # Clamp signed lip dy; keep [0,1] channels non-negative.
    train_pred[:, 0:3] = np.clip(train_pred[:, 0:3], 0.0, 1.0)
    train_pred[:, 5:8] = np.clip(train_pred[:, 5:8], 0.0, 1.0)
    val_pred[:, 0:3] = np.clip(val_pred[:, 0:3], 0.0, 1.0)
    val_pred[:, 5:8] = np.clip(val_pred[:, 5:8], 0.0, 1.0)

    baseline = np.mean(y_train, axis=0)
    baseline_mae = mae(
        np.repeat(baseline.reshape(1, -1), len(y_val), axis=0), y_val
    )
    train_mae = mae(train_pred, y_train)
    val_mae = mae(val_pred, y_val)

    out = model_path(world_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "noise_floor": noise,
            "peak_hint": peak,
            "feature_dim": FEATURE_DIM,
            "control_dim": CONTROL_DIM,
            "controls": list(CONTROL_NAMES),
            "version": "behavior-1.0",
            "role": "fill_missing_transitions",
        },
        out,
    )
    meta = {
        "model": out.name,
        "version": "behavior-1.0",
        "role": "fill_missing_transitions",
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_mae": train_mae,
        "val_mae": val_mae,
        "baseline_mean_mae": baseline_mae,
        "beats_baseline": bool(val_mae < baseline_mae),
        "controls": list(CONTROL_NAMES),
        "dataset": str(dataset),
        "authority": "measured_track > ml_fill > viseme_table",
    }
    meta_path(world_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"behavior-train: wrote {out}")
    return meta


__all__ = ["fit_behavior_model"]
