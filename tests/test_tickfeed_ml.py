"""TickFeed L1–L5 stack + calibration + cosmetics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aiface.tickfeed.calibration import (
    beat_at_time,
    calibration_script_payload,
    write_calibration_script,
)
from aiface.tickfeed.cosmetics import load_cosmetic_prefs, write_cosmetic_prefs
from aiface.tickfeed.driver import TickFeedDriver
from aiface.tickfeed.ml.runtime import TickFeedMLStack
from aiface.tickfeed.ml.train import fit_all_layers
from aiface.tickfeed.package import FaceBox
from aiface.tickfeed.schema import PackageKind, BeatId


def test_calibration_script_beats() -> None:
    script = calibration_script_payload()
    assert script["duration_s"] == 8.0
    assert beat_at_time(script, 1.5)["id"] == "SMILE"
    assert beat_at_time(script, 3.2)["beat_id"] == int(BeatId.SAY_HI)


def test_cosmetics_roundtrip(tmp_path: Path) -> None:
    write_cosmetic_prefs(tmp_path)
    prefs = load_cosmetic_prefs(tmp_path)
    assert prefs.skin_tint_rgb == (1.0, 1.0, 1.0)
    prefs.eye_tint_rgb = (0.2, 0.4, 0.9)
    write_cosmetic_prefs(tmp_path, prefs)
    back = load_cosmetic_prefs(tmp_path)
    assert abs(back.eye_tint_rgb[2] - 0.9) < 1e-6


def test_ml_train_load_and_driver(tmp_path: Path) -> None:
    face = FaceBox(0, 0, 16, 12)
    n = 48
    vel = np.zeros((n, face.h, face.w, 2), dtype=np.float32)
    for t in range(n):
        vel[t, :, :, 1] = 0.01 * np.sin(t / 5.0)
    conf = np.full((n, face.n_cells), 200, dtype=np.uint8)
    np.savez_compressed(
        tmp_path / "face_cell_timeline.npz",
        ticks=np.arange(n, dtype=np.int32),
        velocity=vel,
        conf=conf,
        face_box=np.asarray([0, 0, 16, 12], dtype=np.int32),
        tick_rate=np.asarray([60.0]),
    )
    write_calibration_script(tmp_path)
    meta = fit_all_layers(tmp_path)
    assert "l3" in meta["layers"]
    stack = TickFeedMLStack.try_load(tmp_path)
    assert stack is not None
    speech, look, flat, code = stack.resolve(
        tick=10, open_amt=0.5, smile_amt=0.2
    )
    assert flat.shape[0] == face.n_cells * 2
    assert len(code) >= 4
    recon = stack.decode_code(stack.encode_patch(vel[10]))
    assert recon.shape == flat.shape

    drv = TickFeedDriver.try_load_timeline(tmp_path, face, mouth_uv=(8.0, 8.0))
    assert drv.ml is not None
    k0 = drv.push_drives(tick=0, open_amt=0.1, smile_amt=0.0)
    assert k0.kind == PackageKind.KEYFRAME
    assert k0.conf is not None
    # Beyond timeline → ML path
    live = drv.push_drives(tick=n + 5, open_amt=0.6, smile_amt=0.3)
    assert live.kind in {PackageKind.KEYFRAME, PackageKind.DELTA}
    assert speech.viseme_id >= 0
    assert 0.0 <= look.smile <= 1.0
