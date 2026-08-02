"""Train L1–L5 TickFeed ML layers independently (abstract packets)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aiface.tickfeed.calibration import load_calibration_script, beat_at_time
from aiface.tickfeed.schema import TICK_RATE_HZ, VISEME_TABLE

CODE_DIM = 64
AUDIO_FEAT = 8
ML_DIR = "ml"


def _ml_root(world: Path) -> Path:
    root = world if world.is_dir() else world.parent
    out = root / ML_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_timeline(world: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = world if world.is_dir() else world.parent
    path = root / "face_cell_timeline.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run prepare_tickfeed.py first")
    data = np.load(path)
    ticks = np.asarray(data["ticks"], dtype=np.int32)
    vel = np.asarray(data["velocity"], dtype=np.float32)
    box = np.asarray(data["face_box"], dtype=np.int32)
    return ticks, vel, box


def _patch_stats(vel: np.ndarray) -> np.ndarray:
    """Cheap 8-D motion summary from a face patch (teacher side features)."""
    vx = vel[..., 0]
    vy = vel[..., 1]
    return np.asarray(
        [
            float(np.mean(np.abs(vx))),
            float(np.mean(np.abs(vy))),
            float(np.mean(vx)),
            float(np.mean(vy)),
            float(np.percentile(np.abs(vx), 90)),
            float(np.percentile(np.abs(vy), 90)),
            float(np.std(vx)),
            float(np.std(vy)),
        ],
        dtype=np.float64,
    )


def _audio_proxy(open_amt: float, smile_amt: float, t: float) -> np.ndarray:
    """Deterministic 8-D audio-like features when WAV align unavailable."""
    return np.asarray(
        [
            open_amt,
            smile_amt,
            abs(np.sin(t * 9.0)) * open_amt,
            abs(np.cos(t * 7.0)) * smile_amt,
            open_amt * smile_amt,
            float(open_amt > 0.2),
            float(smile_amt > 0.3),
            min(1.0, open_amt + smile_amt),
        ],
        dtype=np.float64,
    )


def _look_from_beat(beat_id: str, open_s: float, smile_s: float) -> np.ndarray:
    surprise = 0.7 if beat_id == "SURPRISE" else 0.0
    angry = 0.6 if beat_id == "ANGRY" else 0.0
    return np.asarray(
        [
            max(smile_s, 0.85 if beat_id == "SMILE" else 0.0),
            max(open_s, 0.85 if beat_id == "OPEN" else 0.0),
            surprise,
            angry * 0.5,
        ],
        dtype=np.float64,
    )


def _viseme_from_open(open_amt: float, beat_id: str) -> int:
    if beat_id == "SAY_HI":
        return VISEME_TABLE.index("EE")
    if beat_id == "OPEN":
        return VISEME_TABLE.index("AA")
    if open_amt < 0.05:
        return VISEME_TABLE.index("REST")
    if open_amt < 0.25:
        return VISEME_TABLE.index("FF")
    if open_amt < 0.45:
        return VISEME_TABLE.index("EH")
    if open_amt < 0.7:
        return VISEME_TABLE.index("OH")
    return VISEME_TABLE.index("AA")


def build_training_tables(world: Path | str) -> dict[str, np.ndarray]:
    """Build X/y tables for all layers from face_cell_timeline + calibration."""
    world = Path(world)
    ticks, vel, _box = _load_timeline(world)
    script = load_calibration_script(world)
    # Recover open/smile proxies from patch stats for labeling
    n = len(ticks)
    x_audio = np.zeros((n, AUDIO_FEAT), dtype=np.float64)
    y_viseme = np.zeros((n,), dtype=np.int64)
    y_look = np.zeros((n, 4), dtype=np.float64)
    patches = []
    for i, tick in enumerate(ticks):
        t = float(tick) / float(TICK_RATE_HZ)
        beat = beat_at_time(script, t)
        bid = str(beat.get("id") or "REST")
        stats = _patch_stats(vel[i])
        open_amt = float(np.clip(stats[1] * 3.0, 0.0, 1.0))
        smile_amt = float(np.clip(stats[0] * 3.0, 0.0, 1.0))
        if bid == "SMILE":
            smile_amt = max(smile_amt, 0.75)
        if bid == "OPEN":
            open_amt = max(open_amt, 0.75)
        x_audio[i] = _audio_proxy(open_amt, smile_amt, t)
        y_viseme[i] = _viseme_from_open(open_amt, bid)
        y_look[i] = _look_from_beat(bid, open_amt, smile_amt)
        patches.append(vel[i].reshape(-1))
    y_patch = np.stack(patches, axis=0).astype(np.float32)
    return {
        "x_audio": x_audio,
        "y_viseme": y_viseme,
        "y_look": y_look,
        "y_patch": y_patch,
        "ticks": ticks.astype(np.int32),
    }


def fit_all_layers(world: Path | str, *, seed: int = 17) -> dict[str, Any]:
    """Train L1–L5 into world/ml/ and return metrics."""
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    world = Path(world)
    root = _ml_root(world)
    tables = build_training_tables(world)
    x_audio = tables["x_audio"]
    y_viseme = tables["y_viseme"]
    y_look = tables["y_look"]
    y_patch = tables["y_patch"]
    rng = np.random.default_rng(seed)

    meta: dict[str, Any] = {"layers": {}, "code_dim": CODE_DIM, "n": int(len(y_patch))}

    # L4 codec first (teacher for L3)
    n_comp = min(CODE_DIM, max(4, len(y_patch) - 1), y_patch.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    codes = pca.fit_transform(y_patch)
    recon = pca.inverse_transform(codes)
    l4_mae = float(np.mean(np.abs(recon - y_patch)))
    joblib.dump({"pca": pca, "patch_dim": int(y_patch.shape[1])}, root / "l4_tick_codec.joblib")
    meta["layers"]["l4"] = {"mae": l4_mae, "n_components": n_comp}

    # L1 speech clock
    l1 = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=400,
                    random_state=seed,
                ),
            ),
        ]
    )
    l1.fit(x_audio, y_viseme)
    l1_acc = float((l1.predict(x_audio) == y_viseme).mean())
    joblib.dump(l1, root / "l1_speech_clock.joblib")
    meta["layers"]["l1"] = {"train_acc": l1_acc}

    # L2 look drive
    l2 = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    random_state=seed,
                ),
            ),
        ]
    )
    # features: audio + viseme id
    x_l2 = np.concatenate(
        [x_audio, y_viseme.reshape(-1, 1).astype(np.float64)], axis=1
    )
    l2.fit(x_l2, y_look)
    l2_mae = float(np.mean(np.abs(l2.predict(x_l2) - y_look)))
    joblib.dump(l2, root / "l2_look_drive.joblib")
    meta["layers"]["l2"] = {"mae": l2_mae}

    # L3 face motion → code
    x_l3 = np.concatenate([x_l2, y_look], axis=1)
    l3 = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    max_iter=600,
                    random_state=seed,
                ),
            ),
        ]
    )
    l3.fit(x_l3, codes)
    pred_codes = l3.predict(x_l3)
    l3_mae = float(np.mean(np.abs(pred_codes - codes)))
    joblib.dump(l3, root / "l3_face_motion.joblib")
    meta["layers"]["l3"] = {"code_mae": l3_mae}

    # L5 gap prior: recover code from punched holes in features
    x_l5 = x_l3.copy()
    mask = rng.random(x_l5.shape) < 0.25
    x_l5[mask] = 0.0
    l5 = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500,
                    random_state=seed,
                ),
            ),
        ]
    )
    l5.fit(x_l5, codes)
    l5_mae = float(np.mean(np.abs(l5.predict(x_l5) - codes)))
    joblib.dump(l5, root / "l5_gap_prior.joblib")
    meta["layers"]["l5"] = {"code_mae": l5_mae}

    meta_path = root / "tickfeed_ml.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"TickFeed ML: trained L1–L5 → {root} ({meta})")
    return meta


def fit_layer(world: Path | str, layer: str, *, seed: int = 17) -> dict[str, Any]:
    """Retrain a single layer (l1…l5) via abstract packet tables.

    L3/L5 need an existing L4 codec on disk (or train l4 first).
    """
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    layer = layer.lower().strip()
    if layer in {"all", "l1-l5", "*"}:
        return fit_all_layers(world, seed=seed)
    allowed = {"l1", "l2", "l3", "l4", "l5"}
    if layer not in allowed:
        raise ValueError(f"unknown layer {layer}; expected one of {sorted(allowed)}")

    world = Path(world)
    root = _ml_root(world)
    tables = build_training_tables(world)
    x_audio = tables["x_audio"]
    y_viseme = tables["y_viseme"]
    y_look = tables["y_look"]
    y_patch = tables["y_patch"]
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"layer": layer, "n": int(len(y_patch))}

    if layer == "l4":
        n_comp = min(CODE_DIM, max(4, len(y_patch) - 1), y_patch.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        codes = pca.fit_transform(y_patch)
        recon = pca.inverse_transform(codes)
        mae = float(np.mean(np.abs(recon - y_patch)))
        joblib.dump(
            {"pca": pca, "patch_dim": int(y_patch.shape[1])},
            root / "l4_tick_codec.joblib",
        )
        out["metrics"] = {"mae": mae, "n_components": n_comp}
        return out

    l4_path = root / "l4_tick_codec.joblib"
    if not l4_path.is_file():
        raise FileNotFoundError(f"missing {l4_path}; train --layer l4 first")
    l4 = joblib.load(l4_path)
    codes = l4["pca"].transform(y_patch)
    x_l2 = np.concatenate(
        [x_audio, y_viseme.reshape(-1, 1).astype(np.float64)], axis=1
    )
    x_l3 = np.concatenate([x_l2, y_look], axis=1)

    if layer == "l1":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=400,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(x_audio, y_viseme)
        acc = float((model.predict(x_audio) == y_viseme).mean())
        joblib.dump(model, root / "l1_speech_clock.joblib")
        out["metrics"] = {"train_acc": acc}
    elif layer == "l2":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        max_iter=500,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(x_l2, y_look)
        mae = float(np.mean(np.abs(model.predict(x_l2) - y_look)))
        joblib.dump(model, root / "l2_look_drive.joblib")
        out["metrics"] = {"mae": mae}
    elif layer == "l3":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        max_iter=600,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(x_l3, codes)
        mae = float(np.mean(np.abs(model.predict(x_l3) - codes)))
        joblib.dump(model, root / "l3_face_motion.joblib")
        out["metrics"] = {"code_mae": mae}
    else:  # l5
        x_l5 = x_l3.copy()
        mask = rng.random(x_l5.shape) < 0.25
        x_l5[mask] = 0.0
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        max_iter=500,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(x_l5, codes)
        mae = float(np.mean(np.abs(model.predict(x_l5) - codes)))
        joblib.dump(model, root / "l5_gap_prior.joblib")
        out["metrics"] = {"code_mae": mae}

    print(f"TickFeed ML: retrained {layer} → {root} ({out.get('metrics')})")
    return out


__all__ = ["CODE_DIM", "build_training_tables", "fit_all_layers", "fit_layer"]
