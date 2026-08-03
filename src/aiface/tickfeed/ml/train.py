"""Train L1–L5 TickFeed ML layers independently (abstract packets)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from aiface.tickfeed.audio_feat import AUDIO_FEAT, load_audio_feat
from aiface.tickfeed.calibration import load_calibration_script, beat_at_time
from aiface.tickfeed.schema import TICK_RATE_HZ, VISEME_TABLE

CODE_DIM = 64
ML_DIR = "ml"
# Upgrade PCA → nonlinear AE when holdout reconstruction MAE exceeds this
# (NWR-scaled patches ~±1.5). Force with AIFACE_TICKFEED_L4_AE=1.
L4_PCA_HOLDOUT_MAE_MAX = float(os.environ.get("AIFACE_TICKFEED_L4_PCA_MAE", "0.12"))


def _force_l4_ae() -> bool:
    return os.environ.get("AIFACE_TICKFEED_L4_AE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "ae",
    }


def l4_encode_codes(codec: dict[str, Any], patches: np.ndarray) -> np.ndarray:
    """Encode flat patches with an L4 codec dict (pca or ae)."""
    x = np.asarray(patches, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    kind = str(codec.get("kind") or "pca")
    if kind == "ae":
        return np.asarray(codec["encoder"].predict(x), dtype=np.float64)
    pca = codec["pca"]
    return np.asarray(pca.transform(x), dtype=np.float64)


def l4_decode_codes(codec: dict[str, Any], codes: np.ndarray) -> np.ndarray:
    """Decode compact codes with an L4 codec dict (pca or ae)."""
    c = np.asarray(codes, dtype=np.float64)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    kind = str(codec.get("kind") or "pca")
    n = int(
        codec.get("n_components")
        or getattr(codec.get("pca"), "n_components_", c.shape[1])
    )
    if c.shape[1] < n:
        c = np.pad(c, ((0, 0), (0, n - c.shape[1])))
    c = c[:, :n]
    if kind == "ae":
        return np.asarray(codec["decoder"].predict(c), dtype=np.float32)
    return np.asarray(codec["pca"].inverse_transform(c), dtype=np.float32)


def fit_l4_codec(
    y_patch: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int = 17,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Fit L4 PCA, upgrade to AE when reconstruction is insufficient."""
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y_patch = np.asarray(y_patch, dtype=np.float64)
    n_comp = min(CODE_DIM, max(4, len(train_idx) - 1), y_patch.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(y_patch[train_idx])
    codes_pca = pca.transform(y_patch)
    recon_pca = pca.inverse_transform(codes_pca)
    pca_mae = float(np.mean(np.abs(recon_pca[train_idx] - y_patch[train_idx])))
    pca_hold = float(np.mean(np.abs(recon_pca[test_idx] - y_patch[test_idx])))
    explained = float(np.sum(pca.explained_variance_ratio_))

    use_ae = _force_l4_ae() or pca_hold > L4_PCA_HOLDOUT_MAE_MAX or explained < 0.90
    metrics: dict[str, Any] = {
        "mae": pca_mae,
        "holdout_mae": pca_hold,
        "n_components": n_comp,
        "explained_variance": explained,
        "kind": "pca",
        "pca_holdout_mae": pca_hold,
    }
    if not use_ae:
        codec = {
            "kind": "pca",
            "pca": pca,
            "patch_dim": int(y_patch.shape[1]),
            "n_components": n_comp,
        }
        return codec, np.asarray(codes_pca, dtype=np.float64), metrics

    encoder = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(256, 128),
                    max_iter=700,
                    random_state=seed,
                ),
            ),
        ]
    )
    decoder = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 256),
                    max_iter=700,
                    random_state=seed,
                ),
            ),
        ]
    )
    # Nonlinear map into the PCA latent, then nonlinear reconstruct patches.
    encoder.fit(y_patch[train_idx], codes_pca[train_idx])
    codes_ae = np.asarray(encoder.predict(y_patch), dtype=np.float64)
    decoder.fit(codes_ae[train_idx], y_patch[train_idx])
    recon_ae = np.asarray(decoder.predict(codes_ae), dtype=np.float64)
    ae_mae = float(np.mean(np.abs(recon_ae[train_idx] - y_patch[train_idx])))
    ae_hold = float(np.mean(np.abs(recon_ae[test_idx] - y_patch[test_idx])))
    metrics.update(
        {
            "mae": ae_mae,
            "holdout_mae": ae_hold,
            "kind": "ae",
            "ae_holdout_mae": ae_hold,
            "upgraded_from_pca": True,
        }
    )
    # Keep AE only when it wins (or was forced).
    if ae_hold <= pca_hold or _force_l4_ae():
        codec = {
            "kind": "ae",
            "encoder": encoder,
            "decoder": decoder,
            "pca": pca,
            "patch_dim": int(y_patch.shape[1]),
            "n_components": n_comp,
        }
        return codec, codes_ae, metrics

    metrics["kind"] = "pca"
    metrics["mae"] = pca_mae
    metrics["holdout_mae"] = pca_hold
    metrics["ae_rejected_holdout_mae"] = ae_hold
    codec = {
        "kind": "pca",
        "pca": pca,
        "patch_dim": int(y_patch.shape[1]),
        "n_components": n_comp,
    }
    return codec, np.asarray(codes_pca, dtype=np.float64), metrics


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
    """Legacy fallback only when WAV features are missing — never claim as audio."""
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


def build_training_tables(world: Path | str) -> dict[str, np.ndarray | str]:
    """Build X/y tables from FaceCellTimeline + speech_align + look_drive + audio."""
    world = Path(world)
    ticks, vel, _box = _load_timeline(world)
    script = load_calibration_script(world)
    tdir = world if world.is_dir() else world.parent
    speech_path = tdir / "face_cell_timeline" / "speech_align.json"
    look_path = tdir / "face_cell_timeline" / "look_drive.json"
    speech_by_tick: dict[int, dict] = {}
    look_by_tick: dict[int, dict] = {}
    if speech_path.is_file():
        for row in json.loads(speech_path.read_text(encoding="utf-8")).get("ticks") or []:
            speech_by_tick[int(row["tick"])] = row
    if look_path.is_file():
        for row in json.loads(look_path.read_text(encoding="utf-8")).get("ticks") or []:
            look_by_tick[int(row["tick"])] = row

    audio_loaded = load_audio_feat(tdir)
    audio_source = "proxy_fallback"
    audio_table: np.ndarray | None = None
    if audio_loaded is not None:
        audio_table, audio_source = audio_loaded
        if audio_source == "zeros":
            audio_source = "proxy_fallback"
            audio_table = None

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
        if tick in look_by_tick:
            lk = look_by_tick[int(tick)]
            open_amt = max(open_amt, float(lk.get("open") or 0.0))
            smile_amt = max(smile_amt, float(lk.get("smile") or 0.0))
            y_look[i] = np.asarray(
                [
                    float(lk.get("smile") or 0.0),
                    float(lk.get("open") or 0.0),
                    float(lk.get("surprise") or 0.0),
                    float(lk.get("brow") or 0.0),
                ],
                dtype=np.float64,
            )
        else:
            if bid == "SMILE":
                smile_amt = max(smile_amt, 0.75)
            if bid == "OPEN":
                open_amt = max(open_amt, 0.75)
            y_look[i] = _look_from_beat(bid, open_amt, smile_amt)
        if (
            audio_table is not None
            and audio_source == "wav_rms"
            and int(tick) < len(audio_table)
        ):
            x_audio[i] = np.asarray(audio_table[int(tick)], dtype=np.float64)
        elif (
            audio_table is not None
            and audio_source == "wav_rms"
            and i < len(audio_table)
        ):
            x_audio[i] = np.asarray(audio_table[i], dtype=np.float64)
        else:
            x_audio[i] = _audio_proxy(open_amt, smile_amt, t)
        if tick in speech_by_tick:
            y_viseme[i] = int(speech_by_tick[int(tick)].get("viseme_id") or 0)
        else:
            y_viseme[i] = _viseme_from_open(open_amt, bid)
        patches.append(vel[i].reshape(-1))
    y_patch = np.stack(patches, axis=0).astype(np.float32)
    if audio_table is None or audio_source != "wav_rms":
        audio_source = "proxy_fallback"
    return {
        "x_audio": x_audio,
        "y_viseme": y_viseme,
        "y_look": y_look,
        "y_patch": y_patch,
        "ticks": ticks.astype(np.int32),
        "audio_feat_source": audio_source,
    }


def _holdout_split(n: int, *, seed: int, frac: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(int(n))
    if n < 12:
        return idx, idx
    split = max(4, int(n * frac))
    split = min(split, n - max(2, n // 10))
    return idx[:split], idx[split:]


def fit_all_layers(world: Path | str, *, seed: int = 17) -> dict[str, Any]:
    """Train L1–L5 into world/ml/ and return metrics (train + holdout)."""
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    world = Path(world)
    root = _ml_root(world)
    tables = build_training_tables(world)
    x_audio = np.asarray(tables["x_audio"], dtype=np.float64)
    y_viseme = np.asarray(tables["y_viseme"], dtype=np.int64)
    y_look = np.asarray(tables["y_look"], dtype=np.float64)
    y_patch = np.asarray(tables["y_patch"], dtype=np.float32)
    audio_source = str(tables["audio_feat_source"])
    rng = np.random.default_rng(seed)
    train_idx, test_idx = _holdout_split(len(y_patch), seed=seed)

    meta: dict[str, Any] = {
        "layers": {},
        "code_dim": CODE_DIM,
        "n": int(len(y_patch)),
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(test_idx)),
        "audio_feat_source": audio_source,
    }

    # L4 codec first (teacher for L3/L5) — PCA, AE when insufficient.
    codec, codes, l4_metrics = fit_l4_codec(
        y_patch, train_idx, test_idx, seed=seed
    )
    joblib.dump(codec, root / "l4_tick_codec.joblib")
    meta["layers"]["l4"] = l4_metrics

    # L1 speech clock — real WAV features when available
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
    l1.fit(x_audio[train_idx], y_viseme[train_idx])
    l1_acc = float((l1.predict(x_audio[train_idx]) == y_viseme[train_idx]).mean())
    l1_hold = float((l1.predict(x_audio[test_idx]) == y_viseme[test_idx]).mean())
    joblib.dump(l1, root / "l1_speech_clock.joblib")
    meta["layers"]["l1"] = {
        "train_acc": l1_acc,
        "holdout_acc": l1_hold,
        "feat_source": audio_source,
    }

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
    x_l2 = np.concatenate(
        [x_audio, y_viseme.reshape(-1, 1).astype(np.float64)], axis=1
    )
    l2.fit(x_l2[train_idx], y_look[train_idx])
    l2_mae = float(np.mean(np.abs(l2.predict(x_l2[train_idx]) - y_look[train_idx])))
    l2_hold = float(np.mean(np.abs(l2.predict(x_l2[test_idx]) - y_look[test_idx])))
    joblib.dump(l2, root / "l2_look_drive.joblib")
    meta["layers"]["l2"] = {"mae": l2_mae, "holdout_mae": l2_hold}

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
    l3.fit(x_l3[train_idx], codes[train_idx])
    pred_tr = l3.predict(x_l3[train_idx])
    pred_te = l3.predict(x_l3[test_idx])
    l3_mae = float(np.mean(np.abs(pred_tr - codes[train_idx])))
    l3_hold = float(np.mean(np.abs(pred_te - codes[test_idx])))
    joblib.dump(l3, root / "l3_face_motion.joblib")
    meta["layers"]["l3"] = {"code_mae": l3_mae, "holdout_code_mae": l3_hold}

    # L5 gap prior: recover full code from punched holes in the *patch*
    hole_patches = y_patch.copy()
    cell_mask = rng.random(hole_patches.shape) < 0.25
    hole_patches[cell_mask] = 0.0
    codes_holes = l4_encode_codes(codec, hole_patches)
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
    l5.fit(codes_holes[train_idx], codes[train_idx])
    l5_mae = float(
        np.mean(np.abs(l5.predict(codes_holes[train_idx]) - codes[train_idx]))
    )
    l5_hold = float(
        np.mean(np.abs(l5.predict(codes_holes[test_idx]) - codes[test_idx]))
    )
    joblib.dump(l5, root / "l5_gap_prior.joblib")
    meta["layers"]["l5"] = {
        "code_mae": l5_mae,
        "holdout_code_mae": l5_hold,
        "task": "patch_hole_code_recover",
    }

    meta_path = root / "tickfeed_ml.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"TickFeed ML: trained L1-L5 -> {root} ({meta})")
    return meta


def fit_layer(world: Path | str, layer: str, *, seed: int = 17) -> dict[str, Any]:
    """Retrain a single layer (l1…l5). L3/L5 need L4 on disk."""
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
    x_audio = np.asarray(tables["x_audio"], dtype=np.float64)
    y_viseme = np.asarray(tables["y_viseme"], dtype=np.int64)
    y_look = np.asarray(tables["y_look"], dtype=np.float64)
    y_patch = np.asarray(tables["y_patch"], dtype=np.float32)
    audio_source = str(tables["audio_feat_source"])
    rng = np.random.default_rng(seed)
    train_idx, test_idx = _holdout_split(len(y_patch), seed=seed)
    out: dict[str, Any] = {
        "layer": layer,
        "n": int(len(y_patch)),
        "audio_feat_source": audio_source,
    }

    if layer == "l4":
        codec, _codes, metrics = fit_l4_codec(
            y_patch, train_idx, test_idx, seed=seed
        )
        out["metrics"] = metrics
        joblib.dump(codec, root / "l4_tick_codec.joblib")
        print(f"TickFeed ML: retrained {layer} -> {root} ({out.get('metrics')})")
        return out

    l4_path = root / "l4_tick_codec.joblib"
    if not l4_path.is_file():
        raise FileNotFoundError(f"missing {l4_path}; train --layer l4 first")
    l4 = joblib.load(l4_path)
    codes = l4_encode_codes(l4, y_patch)
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
        model.fit(x_audio[train_idx], y_viseme[train_idx])
        joblib.dump(model, root / "l1_speech_clock.joblib")
        out["metrics"] = {
            "train_acc": float(
                (model.predict(x_audio[train_idx]) == y_viseme[train_idx]).mean()
            ),
            "holdout_acc": float(
                (model.predict(x_audio[test_idx]) == y_viseme[test_idx]).mean()
            ),
            "feat_source": audio_source,
        }
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
        model.fit(x_l2[train_idx], y_look[train_idx])
        joblib.dump(model, root / "l2_look_drive.joblib")
        out["metrics"] = {
            "mae": float(
                np.mean(np.abs(model.predict(x_l2[train_idx]) - y_look[train_idx]))
            ),
            "holdout_mae": float(
                np.mean(np.abs(model.predict(x_l2[test_idx]) - y_look[test_idx]))
            ),
        }
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
        model.fit(x_l3[train_idx], codes[train_idx])
        joblib.dump(model, root / "l3_face_motion.joblib")
        out["metrics"] = {
            "code_mae": float(
                np.mean(np.abs(model.predict(x_l3[train_idx]) - codes[train_idx]))
            ),
            "holdout_code_mae": float(
                np.mean(np.abs(model.predict(x_l3[test_idx]) - codes[test_idx]))
            ),
        }
    else:  # l5 — patch-hole code recovery
        hole_patches = y_patch.copy()
        cell_mask = rng.random(hole_patches.shape) < 0.25
        hole_patches[cell_mask] = 0.0
        codes_holes = l4_encode_codes(l4, hole_patches)
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
        model.fit(codes_holes[train_idx], codes[train_idx])
        joblib.dump(model, root / "l5_gap_prior.joblib")
        out["metrics"] = {
            "code_mae": float(
                np.mean(
                    np.abs(model.predict(codes_holes[train_idx]) - codes[train_idx])
                )
            ),
            "holdout_code_mae": float(
                np.mean(
                    np.abs(model.predict(codes_holes[test_idx]) - codes[test_idx])
                )
            ),
            "task": "patch_hole_code_recover",
        }

    print(f"TickFeed ML: retrained {layer} -> {root} ({out.get('metrics')})")
    return out


__all__ = [
    "AUDIO_FEAT",
    "CODE_DIM",
    "L4_PCA_HOLDOUT_MAE_MAX",
    "build_training_tables",
    "fit_all_layers",
    "fit_l4_codec",
    "fit_layer",
    "l4_decode_codes",
    "l4_encode_codes",
]
