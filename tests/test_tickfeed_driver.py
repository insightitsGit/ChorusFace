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


def test_live_speech_beats_look_drive() -> None:
    face = FaceBox(10, 20, 16, 12)
    drv = TickFeedDriver.create(face, mouth_uv=(18.0, 28.0))
    drv.look_by_tick[0] = {"smile": 0.0, "open": 0.0, "surprise": 0.0, "emotion_id": 0}
    drv.timeline_length = 10
    pkg = drv.push_drives(
        tick=0,
        open_amt=0.72,
        smile_amt=0.0,
        phoneme="EE",
        live_speech=True,
    )
    assert pkg.labels is not None
    assert abs(pkg.labels.open_amt - 0.72) < 1e-3


def test_timeline_teacher_tick_loops() -> None:
    face = FaceBox(10, 20, 8, 8)
    drv = TickFeedDriver.create(face, mouth_uv=(14.0, 24.0))
    drv.timeline_length = 10
    drv.loop_timeline = True
    assert drv._teacher_tick(0) == 0
    assert drv._teacher_tick(10) == 0
    assert drv._teacher_tick(25) == 5


def test_ring_producer_lead_absorbs_jitter() -> None:
    """B3: push tick+depth, pop current — early pops damp until lead fills."""
    from aiface.tickfeed.ring import FaceVelocityState, LockstepPlayer
    from aiface.tickfeed.schema import RING_DEPTH

    face = FaceBox(10, 20, 8, 8)
    player = LockstepPlayer(state=FaceVelocityState.zeros(face))
    # Master at 0 with empty ring → damp
    assert player.step() == "damp"
    # Schedule ahead
    values = np.zeros((face.h, face.w, 2), dtype=np.float32)
    values[:] = (0.5, 0.0)
    from aiface.tickfeed.package import build_keyframe

    lead = int(RING_DEPTH)
    player.submit(build_keyframe(lead, face, values))
    # ticks 1..lead-1 still miss
    for _ in range(lead - 1):
        assert player.step() == "damp"
    assert player.step() == "keyframe"


def test_package_bytes_spool_lane_b(tmp_path) -> None:
    from aiface.tickfeed.chorus_transport import TickFeedTransport
    from aiface.tickfeed.package import build_keyframe, encode

    face = FaceBox(2, 2, 4, 4)
    values = np.zeros((4, 4, 2), dtype=np.float32)
    blob = encode(build_keyframe(3, face, values))
    transport = TickFeedTransport(
        world=tmp_path, use_chorus=False, spool_packages=True
    )
    path = transport.push_package_bytes(3, blob)
    assert path is not None and path.is_file()
    assert transport.pull_latest_package_bytes() == blob
