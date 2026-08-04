#!/usr/bin/env python3
"""Prove behavior model is real: artifacts, train weights, load, predict, drive."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORLD = ROOT / "output" / "worlds" / "avatar"
VIDEO = (
    ROOT
    / "assets"
    / "avatar_video_inputs"
    / "Generate_a_single_continuous_.mp4"
)


def main() -> int:
    files = [
        "cell_transition_track.npz",
        "cell_transition_track.json",
        "behavior_dataset.npz",
        "behavior_model.joblib",
        "behavior_model.meta.json",
    ]
    print("=== ARTIFACTS ON DISK ===")
    for name in files:
        path = WORLD / name
        if not path.is_file():
            print(f"MISSING {name}")
            return 1
        print(f"OK      {name:32} {path.stat().st_size} bytes")

    meta = json.loads((WORLD / "behavior_model.meta.json").read_text(encoding="utf-8"))
    print("\n=== TRAIN META ===")
    for key in (
        "version",
        "role",
        "n_train",
        "n_val",
        "train_mae",
        "val_mae",
        "baseline_mean_mae",
        "beats_baseline",
        "controls",
    ):
        print(f"  {key}: {meta.get(key)}")
    if not meta.get("beats_baseline"):
        print("WARN: model did not beat baseline (still a real fit)")

    import joblib

    payload = joblib.load(WORLD / "behavior_model.joblib")
    if not isinstance(payload, dict) or "pipeline" not in payload:
        print("FAIL: joblib payload is not a trained pipeline dict")
        return 1
    pipe = payload["pipeline"]
    mlp = pipe.named_steps["mlp"]
    print("\n=== JOBLIB PAYLOAD ===")
    print(f"  pipeline type: {type(pipe).__name__}")
    print(f"  steps: {[name for name, _ in pipe.steps]}")
    print(f"  mlp layers: {mlp.hidden_layer_sizes}")
    print(f"  mlp n_iter_: {mlp.n_iter_}")
    print(f"  coef shapes: {[c.shape for c in mlp.coefs_]}")
    if not mlp.coefs_ or mlp.n_iter_ is None:
        print("FAIL: MLP has no fitted coefficients")
        return 1

    data = np.load(WORLD / "behavior_dataset.npz", allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    print("\n=== DATASET ===")
    print(f"  X {x.shape} y {y.shape}")
    print(f"  y open range [{y[:, 0].min():.3f}, {y[:, 0].max():.3f}]")
    if len(x) < 8:
        print("FAIL: dataset too small to be a real train")
        return 1

    pred = np.asarray(pipe.predict(x[:8]), dtype=np.float64)
    print("\n=== RAW MODEL PREDICTS (first 8) open/jaw/width/upper ===")
    print(np.round(pred[:, :4], 4))
    if float(np.std(pred)) < 1e-4:
        print("FAIL: predictions are constant — looks fake")
        return 1

    from chorusface.behavior.driver import BehaviorDriver

    driver = BehaviorDriver.try_load(WORLD)
    if not driver.using_ml or driver._pipeline is None:
        print("FAIL: BehaviorDriver did not load ML pipeline")
        return 1
    if not driver.has_track:
        print("FAIL: BehaviorDriver has no measured track")
        return 1

    driver.clear_history()
    for _ in range(5):
        driver.push_rms(0.0)
    low = driver.resolve(phoneme="REST", video_t=None)

    driver.clear_history()
    for _ in range(5):
        driver.push_rms(0.35)
    high = driver.resolve(phoneme="AH", video_t=None)

    measured = driver.resolve(phoneme="AH", video_t=2.0)
    print("\n=== DRIVER RESOLVE ===")
    print(
        f"  low      source={low.source} open={low.openness_n:.3f} "
        f"width={low.width_n:.3f}"
    )
    print(
        f"  high     source={high.source} open={high.openness_n:.3f} "
        f"width={high.width_n:.3f}"
    )
    print(
        f"  measured source={measured.source} open={measured.openness_n:.3f} "
        f"delta_open={measured.delta_open:.3f}"
    )
    if high.source != "ml_fill":
        print(f"FAIL: expected ml_fill for live speech, got {high.source}")
        return 1
    if not measured.source.startswith("measured"):
        print(f"FAIL: expected measured@t, got {measured.source}")
        return 1
    if not (
        high.openness_n > low.openness_n
        or high.teeth_reveal > low.teeth_reveal
        or high.cavity_n > low.cavity_n
    ):
        print("FAIL: ML output ignores audio energy (looks unused)")
        return 1

    # Prove app wiring calls BehaviorDriver (not a dead field).
    import inspect
    from chorusface import app as app_mod

    src = inspect.getsource(app_mod.AvatarFaceApp._update_open_close_ml)
    if "self._behavior.resolve" not in src:
        print("FAIL: app does not call BehaviorDriver.resolve")
        return 1
    if "apply_behavior_flow" not in src:
        print("FAIL: app does not apply behavior flow to MouthCellPlan")
        return 1
    print("\n=== APP WIRING ===")
    print("  _update_open_close_ml calls behavior.resolve + apply_behavior_flow")

    if not VIDEO.is_file():
        print(f"FAIL: training video missing: {VIDEO}")
        return 1

    from chorusface.behavior.pipeline import train_behavior_from_video

    landmarker = WORLD / "face_landmarker.task"
    print("\n=== RETRAIN FRESH INTO TEMP DIR ===")
    with TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        meta2 = train_behavior_from_video(
            VIDEO,
            world_dir=tmp,
            sample_fps=8.0,
            landmarker_model=landmarker if landmarker.is_file() else None,
            seed=99,
        )
        model_path = tmp / "behavior_model.joblib"
        track_path = tmp / "cell_transition_track.npz"
        if not model_path.is_file() or not track_path.is_file():
            print("FAIL: retrain did not write model/track")
            return 1
        fresh = BehaviorDriver.try_load(tmp)
        if not fresh.using_ml or not fresh.has_track:
            print("FAIL: fresh retrain not loadable")
            return 1
        for _ in range(5):
            fresh.push_rms(0.3)
        state = fresh.resolve(phoneme="AH", video_t=None)
        print(
            f"  samples={meta2['n_track_samples']} val_mae={meta2['val_mae']:.4f} "
            f"beats={meta2['beats_baseline']}"
        )
        print(f"  resolve source={state.source} open={state.openness_n:.3f}")
        if state.source != "ml_fill":
            print("FAIL: fresh model not used for fill")
            return 1
        # Coefficients exist and are finite.
        coefs = joblib.load(model_path)["pipeline"].named_steps["mlp"].coefs_
        flat = np.concatenate([c.reshape(-1) for c in coefs])
        if not np.isfinite(flat).all() or float(np.std(flat)) < 1e-8:
            print("FAIL: fresh model coefficients look empty")
            return 1

    print(
        "\nVERIFY PASS: model trained, persisted, loaded, predicts, "
        "driver uses ml_fill/measured, app wired"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
