"""Measured transitions + ML fill for missing avatar behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chorusface.behavior.driver import BehaviorDriver
from chorusface.behavior.schema import (
    CONTROL_DIM,
    CONTROL_NAMES,
    FEATURE_DIM,
    landmarks_to_controls,
)
from chorusface.behavior.track import TransitionTrack, save_transition_track
from chorusface.behavior.train import fit_behavior_model
from chorusface.mouth_cell_plan import MouthCellPlan, viseme_flow
from chorusface.cell_cluster import CellCluster


def _fake_track(n: int = 40, fps: float = 12.0) -> TransitionTrack:
    times = np.arange(n, dtype=np.float64) / fps
    controls = np.zeros((n, CONTROL_DIM), dtype=np.float64)
    features = np.zeros((n, FEATURE_DIM), dtype=np.float64)
    for i, t in enumerate(times):
        # Smooth open pulse mid-take — simulates second 1 → 2 → 3 motion.
        open_n = float(np.clip(np.sin(t * 2.2) * 0.5 + 0.45, 0.0, 1.0))
        width_n = float(np.clip(0.35 + 0.4 * np.sin(t * 1.1 + 0.3), 0.0, 1.0))
        controls[i] = landmarks_to_controls(
            openness_n=open_n, width_n=width_n, teeth_n=open_n * 0.7
        )
        features[i, 0] = open_n
        features[i, 1] = width_n
        features[i, 2:] = open_n * 0.1
    # Punch a gap (drop samples) so ML must fill.
    keep = np.ones(n, dtype=bool)
    keep[15:22] = False
    times_k = times[keep]
    controls_k = controls[keep]
    features_k = features[keep]
    deltas = np.zeros_like(controls_k)
    if len(controls_k) > 1:
        deltas[1:] = controls_k[1:] - controls_k[:-1]
    return TransitionTrack(
        times=times_k,
        controls=controls_k,
        deltas=deltas,
        features=features_k,
        video="synthetic.mp4",
        sample_fps=fps,
        noise_floor=0.01,
        peak_hint=0.5,
    )


def test_landmarks_to_controls_shape() -> None:
    vec = landmarks_to_controls(openness_n=0.8, width_n=0.6, teeth_n=0.5)
    assert len(vec) == CONTROL_DIM
    assert vec[0] == 0.8
    assert vec[3] == -0.8  # upper rises
    assert vec[4] == 0.8  # lower drops


def test_track_sample_and_gap(tmp_path: Path) -> None:
    track = _fake_track()
    save_transition_track(track, tmp_path)
    assert (tmp_path / "cell_transition_track.npz").is_file()
    assert (tmp_path / "cell_transition_track.json").is_file()

    # Dense region — measured lerp.
    state = track.sample_at(0.5)
    assert state is not None
    assert state.source in ("measured", "measured_lerp")
    assert state.openness_n >= 0.0

    # Gap region around dropped samples (~1.25–1.75s at 12fps).
    assert track.gap_at(1.4)


def test_behavior_driver_measured_then_ml(tmp_path: Path) -> None:
    track = _fake_track()
    save_transition_track(track, tmp_path)
    meta = fit_behavior_model(
        tmp_path / "behavior_dataset.npz", world_dir=tmp_path, seed=3
    )
    assert meta["beats_baseline"] or meta["val_mae"] < 0.5

    driver = BehaviorDriver.try_load(tmp_path)
    assert driver.has_track
    assert driver.using_ml

    measured = driver.resolve(phoneme="AH", video_t=0.4)
    assert measured.source.startswith("measured")

    # Live speech — no video clock → ML fill (with table blend).
    for _ in range(5):
        driver.push_rms(0.25)
    filled = driver.resolve(phoneme="AH", video_t=None)
    assert filled.source == "ml_fill"
    assert filled.openness_n >= 0.0


def test_apply_behavior_flow_on_plan() -> None:
    cells = np.asarray(
        [[10 + i, 20 + (i % 3)] for i in range(40)], dtype=np.int32
    )
    cluster = CellCluster(region_id=0, name="mouth_unlocked", cells=cells)
    from chorusface.cell_cluster import CellClusterIndex

    index = CellClusterIndex(width=64, height=64, clusters=[cluster])
    for x, y in cells.tolist():
        index._membership[(int(x), int(y))] = "mouth_unlocked"
    plan = MouthCellPlan(index)
    plan.sync_from_timeline("AH", active_until=10.0, now=0.0)
    table = viseme_flow("AH")
    plan.apply_behavior_flow(0.5, 0.7, 0.1, source="ml_fill")
    assert abs(plan._open - 0.5) < 1e-6
    assert abs(plan._width - 0.7) < 1e-6
    assert table[0] != 0.5  # behavior overrode table AH open
