"""Close remaining TickFeed research items: CHORUS recv, Whisper teacher, DIS, L4 AE."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from chorusface.tickfeed.calibration import write_calibration_script
from chorusface.tickfeed.chorus_transport import (
    TickFeedTransport,
    reassemble_lane_b_chunks,
)
from chorusface.tickfeed.force_align import force_align_speech
from chorusface.tickfeed.ml.train import fit_all_layers, fit_l4_codec
from chorusface.tickfeed.package import FaceBox, build_keyframe, encode
from chorusface.tts import WordSpan


def test_pull_recv_code_and_package(tmp_path: Path) -> None:
    transport = TickFeedTransport(world=tmp_path, use_chorus=False, dim=64)
    recv = transport.recv_spool
    recv.mkdir(parents=True, exist_ok=True)

    code = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    (recv / "vec_00000001.f32").write_bytes(np.ascontiguousarray(code).tobytes())
    got = transport.pull_recv_code()
    assert got is not None
    assert np.allclose(got, code, atol=1e-6)

    face = FaceBox(0, 0, 8, 6)
    blob = encode(
        build_keyframe(
            3,
            face,
            np.zeros((face.h, face.w, 2), dtype=np.float32),
        )
    )
    frames = transport._frame_chunks(blob, tick=3)
    for i, frame in enumerate(frames):
        (recv / f"vec_{i + 10:08d}.f32").write_bytes(
            np.ascontiguousarray(frame.astype("<f4")).tobytes()
        )
    rebuilt = transport.pull_recv_package_bytes()
    assert rebuilt == blob
    assert reassemble_lane_b_chunks(frames) == blob


def test_force_align_whisper_words_teacher(tmp_path: Path) -> None:
    write_calibration_script(tmp_path)
    video = tmp_path / "calibration_take.mp4"
    video.write_bytes(b"fake")

    spans = [
        WordSpan("hi", 3.05, 3.25),
        WordSpan("there", 3.30, 3.70),
    ]

    with (
        patch("chorusface.tickfeed.force_align._extract_wav", return_value=True),
        patch(
            "chorusface.tickfeed.force_align._rms_at_60hz",
            return_value=np.ones(480, dtype=np.float32),
        ),
        patch(
            "chorusface.tickfeed.force_align._whisper_word_spans",
            return_value=spans,
        ),
    ):
        payload = force_align_speech(tmp_path, video=video, n_ticks=480)

    assert payload["method"] == "whisper_words_force_align"
    stamped = [t for t in payload["ticks"] if t.get("teacher") == "whisper_words"]
    assert stamped
    assert any(t.get("word") == "hi" for t in stamped)


def test_dis_dense_flow_rest_relative() -> None:
    import cv2

    from chorusface.tickfeed.collect import _dense_rest_flow

    rest = np.zeros((48, 48), dtype=np.uint8)
    rest[20:28, 10:38] = 180
    frame = np.zeros((48, 48), dtype=np.uint8)
    frame[24:32, 10:38] = 180  # downward lip motion
    flow = _dense_rest_flow(rest, frame, cv2)
    assert flow.shape == (48, 48, 2)
    # Mean-removed later in collect; here raw should show downward dy > 0 in band.
    band = flow[20:32, 10:38, 1]
    assert float(np.mean(band)) > 0.2


def test_l4_ae_forced_codec(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 64
    dim = 96
    # Nonlinear target so PCA alone is weak relative to AE path.
    base = rng.normal(size=(n, 8)).astype(np.float64)
    y = np.concatenate(
        [np.sin(base), np.cos(base), base**2, np.tanh(base)], axis=1
    ).astype(np.float64)[:, :dim]
    idx = np.arange(n)
    train, test = idx[:48], idx[48:]
    with patch.dict("os.environ", {"CHORUSFACE_TICKFEED_L4_AE": "1"}):
        codec, codes, metrics = fit_l4_codec(y, train, test, seed=3)
    assert metrics["kind"] == "ae"
    assert codec["kind"] == "ae"
    assert codes.shape[0] == n
    assert "encoder" in codec and "decoder" in codec

    # End-to-end train with forced AE on a tiny world.
    face = FaceBox(0, 0, 8, 6)
    vel = np.zeros((40, face.h, face.w, 2), dtype=np.float32)
    for t in range(40):
        vel[t, :, :, 1] = 0.05 * np.sin(t / 3.0)
    conf = np.full((40, face.n_cells), 200, dtype=np.uint8)
    from chorusface.tickfeed.timeline_io import write_face_cell_timeline

    write_calibration_script(tmp_path)
    write_face_cell_timeline(
        tmp_path, face=face, velocity=vel, conf=conf, video_name="t"
    )
    with patch.dict("os.environ", {"CHORUSFACE_TICKFEED_L4_AE": "1"}):
        meta = fit_all_layers(tmp_path)
    assert meta["layers"]["l4"]["kind"] == "ae"
