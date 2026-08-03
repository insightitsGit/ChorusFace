"""TickFeedDriver + synth — live package path."""

from __future__ import annotations

import numpy as np

from aiface.tickfeed.driver import TickFeedDriver
from aiface.tickfeed.force_align import force_align_speech
from aiface.tickfeed.package import FaceBox, decode, encode
from aiface.tickfeed.schema import PackageKind
from aiface.tickfeed.synth import synthesize_velocity


def test_synth_and_driver_key_then_delta(monkeypatch) -> None:
    monkeypatch.setenv("AIFACE_TICKFEED_ABSOLUTE", "0")
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


def test_absolute_key_every_tick_by_default() -> None:
    """Default fidelity mode: each 16.7 ms tick is KEY assign, not Δ residue."""
    face = FaceBox(10, 20, 8, 8)
    drv = TickFeedDriver.create(face, mouth_uv=(14.0, 24.0))
    k0 = drv.push_drives(tick=0, open_amt=0.5, smile_amt=0.0, phoneme="AH")
    k1 = drv.push_drives(tick=1, open_amt=0.6, smile_amt=0.0, phoneme="AH")
    assert k0.kind == PackageKind.KEYFRAME
    assert k1.kind == PackageKind.KEYFRAME


def test_synth_open_moves_both_lips() -> None:
    """Live open must raise upper lip and drop lower — not jaw-only slide."""
    face = FaceBox(10, 20, 32, 24)
    vel = synthesize_velocity(face, open_amt=0.9, smile_amt=0.0, mouth_uv=(26.0, 35.0))
    # mouth_uv → local (16, 15); upper rows y<15, lower y>15
    upper = vel[:15, :, 1]
    lower = vel[15:, :, 1]
    assert float(upper.max()) > 0.05
    assert float(lower.min()) < -0.08


def test_brow_amt_roundtrips_in_labels() -> None:
    from aiface.tickfeed.package import TickLabels

    labels = TickLabels(brow_amt=0.7, surprise_amt=0.8, emotion_id=2)
    back = TickLabels.unpack(labels.pack())
    assert abs(back.brow_amt - 0.7) < 1e-5
    assert abs(back.surprise_amt - 0.8) < 1e-5


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


def test_hearing_live_keeps_field_still() -> None:
    """Typing/hearing overlay owns LOOK but must not invent mouth FIELD motion."""
    face = FaceBox(10, 20, 8, 8)
    drv = TickFeedDriver.create(face, mouth_uv=(14.0, 24.0))
    pkg = drv.push_drives(
        tick=0,
        open_amt=0.0,
        smile_amt=0.0,
        surprise_amt=0.15,
        phoneme="REST",
        emotion="THINKING",
        live_speech=True,
    )
    assert pkg.labels is not None
    assert abs(pkg.labels.surprise_amt - 0.15) < 1e-3
    assert pkg.values is not None
    assert float(np.abs(pkg.values).max()) < 1e-6


def test_timeline_teacher_tick_loops() -> None:
    face = FaceBox(10, 20, 8, 8)
    drv = TickFeedDriver.create(face, mouth_uv=(14.0, 24.0))
    drv.timeline_length = 10
    drv.loop_timeline = True
    assert drv._teacher_tick(0) == 0
    assert drv._teacher_tick(10) == 0
    assert drv._teacher_tick(25) == 5


def test_entering_still_emits_zero_keyframe() -> None:
    """Motion → still must KEY-clear GPU velocity (not sparse/EMPTY residue)."""
    face = FaceBox(10, 20, 8, 8)
    drv = TickFeedDriver.create(face, mouth_uv=(14.0, 24.0))
    moving = np.zeros((face.h, face.w, 2), dtype=np.float32)
    moving[..., 1] = -0.35
    drv.timeline[0] = moving
    drv.timeline_conf[0] = np.full(face.n_cells, 220, dtype=np.uint8)
    drv.look_by_tick[0] = {
        "smile": 0.0,
        "open": 0.5,
        "surprise": 0.0,
        "emotion_id": 0,
    }
    drv.timeline_length = 2
    k0 = drv.push_drives(tick=0, open_amt=0.5, smile_amt=0.0, phoneme="AH")
    assert k0.kind == PackageKind.KEYFRAME
    assert float(np.abs(k0.values).max()) > 0.1

    drv.timeline[1] = np.zeros((face.h, face.w, 2), dtype=np.float32)
    drv.timeline_conf[1] = np.full(face.n_cells, 255, dtype=np.uint8)
    drv.look_by_tick[1] = {
        "smile": 0.0,
        "open": 0.0,
        "surprise": 0.0,
        "emotion_id": 0,
    }
    k1 = drv.push_drives(tick=1, open_amt=0.0, smile_amt=0.0, phoneme="REST")
    assert k1.kind == PackageKind.KEYFRAME
    assert k1.values is not None
    assert float(np.abs(k1.values).max()) < 1e-6


def test_rest_labels_zero_field() -> None:
    face = FaceBox(10, 20, 8, 8)
    drv = TickFeedDriver.create(face, mouth_uv=(14.0, 24.0))
    drv.look_by_tick[0] = {
        "smile": 0.0,
        "open": 0.0,
        "surprise": 0.0,
        "emotion_id": 0,
    }
    drv.timeline[0] = np.ones((face.h, face.w, 2), dtype=np.float32)
    drv.timeline_conf[0] = np.full(face.n_cells, 200, dtype=np.uint8)
    drv.timeline_length = 1
    pkg = drv.push_drives(tick=0, open_amt=0.0, smile_amt=0.0)
    assert pkg.values is not None
    assert float(np.abs(pkg.values).max()) < 1e-6


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


def test_local_ring_same_tick_has_no_misses() -> None:
    """Local-ring Side B feed: produce tick T then pop T — no warm-up MISS."""
    face = FaceBox(10, 20, 8, 8)
    drv = TickFeedDriver.create(face, mouth_uv=(14.0, 24.0))
    for master in range(1, 8):
        drv.push_drives(tick=master, open_amt=0.4, smile_amt=0.0, phoneme="AH")
        pkg = drv.pop_for_master(master)
        assert pkg is not None
        assert int(pkg.tick) == master
        assert pkg.kind == PackageKind.KEYFRAME


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


def test_lane_b_magics_and_crc_survive_float32() -> None:
    """Magics exact in f32; CRC round-trips via uint16 halves (not one u32 float)."""
    import zlib

    from aiface.tickfeed.chorus_transport import (
        TPK_CHUNK_MAGIC,
        TPK_REF_MAGIC,
        TickFeedTransport,
        assert_f32_exact_int,
        crc32_from_halves,
        crc32_to_halves,
        parse_lane_b_header,
        reassemble_lane_b_chunks,
    )
    from aiface.tickfeed.package import build_keyframe, encode

    assert_f32_exact_int(TPK_CHUNK_MAGIC)
    assert_f32_exact_int(TPK_REF_MAGIC)
    # Prove the old date-style magic would have failed:
    assert float(np.float32(20260803.0)) != 20260803.0

    crc = 0xDEADBEEF
    lo, hi = crc32_to_halves(crc)
    assert crc32_from_halves(float(np.float32(lo)), float(np.float32(hi))) == crc
    # Single-float packing loses low bits — must not be used:
    assert int(np.float32(float(crc))) != crc

    face = FaceBox(2, 2, 4, 4)
    values = np.zeros((4, 4, 2), dtype=np.float32)
    values[0, 0] = (0.25, -0.5)
    blob = encode(build_keyframe(9, face, values))
    transport = TickFeedTransport(world=".", use_chorus=False)
    frames = transport._frame_chunks(blob, 9)
    # Simulate wire: cast every frame through float32 storage
    wire = [np.asarray(f, dtype=np.float32).copy() for f in frames]
    header = parse_lane_b_header(wire[0])
    assert header["kind"] == "chunk"
    assert header["magic"] == float(np.float32(TPK_CHUNK_MAGIC))
    assert header["crc32"] == (zlib.crc32(blob) & 0xFFFFFFFF)
    assert reassemble_lane_b_chunks(wire) == blob

    ref = np.zeros(64, dtype=np.float32)
    ref[0] = np.float32(TPK_REF_MAGIC)
    ref[1] = np.float32(9)
    ref[2] = np.float32(len(blob))
    lo, hi = crc32_to_halves(zlib.crc32(blob) & 0xFFFFFFFF)
    ref[3] = np.float32(lo)
    ref[4] = np.float32(hi)
    rh = parse_lane_b_header(ref)
    assert rh["kind"] == "ref"
    assert rh["crc32"] == (zlib.crc32(blob) & 0xFFFFFFFF)


def test_angry_beat_keeps_measured_field() -> None:
    """Still-face gate must not erase ANGRY (brow-only LOOK amounts)."""
    from aiface.tickfeed.schema import EmotionId

    face = FaceBox(10, 20, 8, 8)
    drv = TickFeedDriver.create(face, mouth_uv=(14.0, 24.0))
    drv.look_by_tick[0] = {
        "smile": 0.0,
        "open": 0.0,
        "surprise": 0.0,
        "brow": 0.7,
        "emotion_id": int(EmotionId.ANGRY),
    }
    measured = np.zeros((face.h, face.w, 2), dtype=np.float32)
    measured[:, :, 1] = -0.4
    drv.timeline[0] = measured
    drv.timeline_conf[0] = np.full(face.n_cells, 220, dtype=np.uint8)
    drv.timeline_length = 1
    pkg = drv.push_drives(tick=0, open_amt=0.0, smile_amt=0.0)
    assert pkg.values is not None
    assert float(np.abs(pkg.values).max()) > 0.1


def test_wire_loop_code_feeds_ring(tmp_path) -> None:
    """Master ring is empty until ingest_from_wire expands the pushed c_t."""
    from aiface.tickfeed.chorus_transport import TickFeedTransport

    face = FaceBox(0, 0, 8, 8)

    class _FakeML:
        def encode_patch(self, patch):
            flat = np.asarray(patch, dtype=np.float32).reshape(-1)
            return [float(flat.max())] + [0.0] * 63

        def decode_code(self, code):
            v = float(np.asarray(code, dtype=np.float32).reshape(-1)[0])
            return np.full(face.h * face.w * 2, v, dtype=np.float32)

    drv = TickFeedDriver.create(face, mouth_uv=(4.0, 5.0))
    # Measured timeline avoids ml.resolve; FakeML only encodes/decodes c_t.
    patch = synthesize_velocity(face, open_amt=0.9, smile_amt=0.0)
    drv.timeline[0] = patch
    drv.timeline_conf[0] = np.full(face.n_cells, 220, dtype=np.uint8)
    drv.timeline_length = 1
    drv.ml = _FakeML()  # type: ignore[assignment]
    drv.wire_loop = True
    drv.wire_loop_source = "code"
    drv.transport = TickFeedTransport(
        world=tmp_path,
        use_chorus=False,
        spool_codes=False,
        spool_packages=False,
    )
    drv.push_drives(tick=0, open_amt=0.9, smile_amt=0.0, phoneme="AH")
    assert len(drv.player.ring) == 0
    assert drv.transport.pull_latest_code() is not None
    assert drv.ingest_from_wire(0) is not None
    pkg = drv.pop_for_master(0)
    assert pkg is not None
    assert pkg.kind == PackageKind.KEYFRAME


def test_wire_loop_package_feeds_ring(tmp_path) -> None:
    from aiface.tickfeed.chorus_transport import TickFeedTransport

    face = FaceBox(0, 0, 8, 8)
    drv = TickFeedDriver.create(face, mouth_uv=(4.0, 5.0))
    drv.wire_loop = True
    drv.wire_loop_source = "package"
    drv.transport = TickFeedTransport(
        world=tmp_path,
        use_chorus=False,
        spool_packages=True,
        spool_keys_only=False,
        spool_codes=False,
    )
    drv.push_drives(tick=0, open_amt=0.85, smile_amt=0.1, phoneme="AH")
    assert len(drv.player.ring) == 0
    assert drv.ingest_from_wire(0) is not None
    pkg = drv.pop_for_master(0)
    assert pkg is not None
    assert float(np.abs(pkg.values).max()) > 0.0 if pkg.values is not None else True


def test_spool_trim_throttled(tmp_path) -> None:
    from aiface.tickfeed.chorus_transport import TickFeedTransport

    transport = TickFeedTransport(
        world=tmp_path,
        use_chorus=False,
        spool_codes=True,
        spool_packages=False,
        spool_keep=5,
        trim_every=3,
    )
    spool = tmp_path / "tickfeed_chorus_spool"
    for i in range(2):
        transport.push_code(i, np.zeros(64, dtype=np.float32))
    assert len(list(spool.glob("tick_*.f32"))) == 2
    transport.push_code(2, np.zeros(64, dtype=np.float32))  # 3rd write → trim
    assert len(list(spool.glob("tick_*.f32"))) == 3  # 3 <= keep 5, nothing deleted
    for i in range(3, 9):
        transport.push_code(i, np.zeros(64, dtype=np.float32))
    # Trims every 3 writes; after overflow, keep at most spool_keep.
    assert len(list(spool.glob("tick_*.f32"))) <= 5
