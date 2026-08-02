"""TickFeedDriver + synth — live package path."""

from __future__ import annotations

import numpy as np

from aiface.tickfeed.driver import TickFeedDriver
from aiface.tickfeed.force_align import force_align_speech
from aiface.tickfeed.package import FaceBox, decode, encode
from aiface.tickfeed.schema import PackageKind
from aiface.tickfeed.synth import synthesize_velocity


def test_synth_and_driver_key_then_delta() -> None:
    face = FaceBox(10, 20, 32, 24)
    drv = TickFeedDriver.create(face, mouth_uv=(26.0, 35.0))
    k0 = drv.push_drives(tick=0, open_amt=0.0, smile_amt=0.0, phoneme="REST")
    assert k0.kind == PackageKind.KEYFRAME
    d1 = drv.push_drives(tick=1, open_amt=0.8, smile_amt=0.1, phoneme="AH")
    assert d1.kind == PackageKind.DELTA
    blob = encode(d1)
    back = decode(blob)
    assert back.face.w == 32
    vel = synthesize_velocity(face, open_amt=0.8, smile_amt=0.1)
    assert vel.shape == (24, 32, 2)
    assert float(np.abs(vel).max()) > 0.0


def test_look_drive_sole_label_authority() -> None:
    """Measured look_drive replaces caller amounts (no max-merge)."""
    face = FaceBox(10, 20, 16, 12)
    drv = TickFeedDriver.create(face, mouth_uv=(18.0, 28.0))
    drv.look_by_tick[5] = {
        "smile": 0.2,
        "open": 0.1,
        "surprise": 0.0,
        "emotion_id": 1,
    }
    pkg = drv.push_drives(
        tick=5, open_amt=0.95, smile_amt=0.99, surprise_amt=0.8, phoneme="AH"
    )
    assert pkg.labels is not None
    assert abs(pkg.labels.smile_amt - 0.2) < 1e-3
    assert abs(pkg.labels.open_amt - 0.1) < 1e-3
    assert abs(pkg.labels.surprise_amt - 0.0) < 1e-3


def test_force_align_falls_back_without_video(tmp_path) -> None:
    from aiface.tickfeed.calibration import write_calibration_script

    write_calibration_script(tmp_path)
    payload = force_align_speech(tmp_path, video=tmp_path / "missing.mp4", n_ticks=60)
    assert payload["method"] == "script_force_align"
    assert payload["n_ticks"] == 60
