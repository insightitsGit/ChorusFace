"""Train audio features → live control vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aiface.live_vector.schema import (
    CONTROL_DIM,
    CONTROL_NAMES,
    FEATURE_DIM,
    HISTORY,
    META_NAME,
    MODEL_NAME,
    meta_path,
    model_path,
)


def _load(dataset: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    data = np.load(dataset, allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    noise = float(np.asarray(data["noise_floor"]).reshape(-1)[0])
    peak = float(np.asarray(data["peak_hint"]).reshape(-1)[0])
    if y.ndim == 1:
        y = np.stack([y, y, np.clip(y * 0.35, 0.0, 1.0)], axis=1)
    y = y[:, :CONTROL_DIM]
    if x.shape[1] != FEATURE_DIM:
        raise ValueError(f"bad X shape {x.shape}")
    return x, y, noise, peak


def fit_live_vector_model(
    dataset: Path,
    *,
    world_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 17,
) -> dict[str, Any]:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    x, y, noise, peak = _load(dataset)
    if len(y) < 8:
        raise RuntimeError(f"dataset too small: {len(y)}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    n_val = max(1, int(round(len(y) * val_fraction)))
    x_val, y_val = x[:n_val], y[:n_val]
    x_train, y_train = x[n_val:], y[n_val:]
    if len(y_train) < 4:
        x_train, y_train = x, y
        x_val, y_val = x, y

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
                    hidden_layer_sizes=(48, 24),
                    activation="relu",
                    max_iter=1200,
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

    train_pred = np.clip(pipeline.predict(x_train), 0.0, 1.0)
    val_pred = np.clip(pipeline.predict(x_val), 0.0, 1.0)
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
            "history": HISTORY,
            "control_dim": CONTROL_DIM,
            "controls": list(CONTROL_NAMES),
            "version": "live-vector-1.0",
        },
        out,
    )
    meta = {
        "model": MODEL_NAME,
        "version": "live-vector-1.0",
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_mae": train_mae,
        "val_mae": val_mae,
        "baseline_mean_mae": baseline_mae,
        "beats_baseline": bool(val_mae < baseline_mae),
        "controls": list(CONTROL_NAMES),
        "dataset": str(dataset),
    }
    meta_path(world_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"train: wrote {out}")
    return meta
