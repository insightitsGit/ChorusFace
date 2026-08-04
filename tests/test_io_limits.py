"""Ingress caps and pinned joblib load (QA CR-001…007, 009)."""

from __future__ import annotations

import hashlib
import io
import zlib
from pathlib import Path

import numpy as np
import pytest

from chorusface.io_limits import (
    MAX_FACE_SIDE,
    MAX_TTS_RESPONSE_BYTES,
    MAX_VOICE_BUFFER_SECONDS,
    MAX_ZLIB_OUTPUT_BYTES,
    is_weak_bridge_token,
    read_capped,
    safe_joblib_load,
    token_fingerprint,
    validate_face_box,
    write_model_digest,
    zlib_decompress_capped,
)
from chorusface.stream import StreamConfig, StreamError, VoiceStream
from chorusface.tickfeed.package import FaceBox, build_keyframe, decode, encode


def test_weak_token_and_fingerprint() -> None:
    assert is_weak_bridge_token("tickfeed-lab")
    assert not is_weak_bridge_token("x" * 32)
    assert len(token_fingerprint("abc")) == 12


def test_validate_face_box_rejects_oom_dims() -> None:
    validate_face_box(158, 199)
    with pytest.raises(ValueError, match="max side"):
        validate_face_box(MAX_FACE_SIDE + 1, 16)
    with pytest.raises(ValueError, match="invalid face"):
        validate_face_box(0, 16)


def test_decode_rejects_huge_face_header() -> None:
    import struct

    face = FaceBox(0, 0, 64, 64)
    values = np.zeros((64, 64, 2), dtype=np.float32)
    pkg = build_keyframe(1, face, values)
    patched = bytearray(encode(pkg))
    # Header layout ``<IHHIf4H …``: face_w/h are the 3rd/4th H after time → offset 20.
    struct.pack_into("<HH", patched, 20, 4000, 4000)
    # Zero CRC so size validation runs before integrity check (crc at offset 36).
    struct.pack_into("<I", patched, 36, 0)
    with pytest.raises(ValueError, match="max side|exceeds"):
        decode(bytes(patched))


def test_zlib_decompress_capped_blocks_zip_bomb() -> None:
    huge = b"A" * (MAX_ZLIB_OUTPUT_BYTES + 10_000)
    compressed = zlib.compress(huge, level=9)
    with pytest.raises(ValueError, match="exceeds|outside"):
        zlib_decompress_capped(
            compressed,
            max_output=MAX_ZLIB_OUTPUT_BYTES,
            expect_nbytes=len(huge),
        )
    ok = b"hello-tickfeed"
    out = zlib_decompress_capped(
        zlib.compress(ok),
        max_output=MAX_ZLIB_OUTPUT_BYTES,
        expect_nbytes=len(ok),
    )
    assert out == ok


def test_read_capped() -> None:
    data = b"x" * 100
    assert read_capped(io.BytesIO(data), 100) == data
    with pytest.raises(ValueError, match="exceeds"):
        read_capped(io.BytesIO(b"x" * (MAX_TTS_RESPONSE_BYTES + 1)), 64)


def test_voice_stream_feed_cap() -> None:
    stream = VoiceStream(StreamConfig(sample_rate=8_000))
    max_samples = int(MAX_VOICE_BUFFER_SECONDS * 8_000)
    stream.feed(np.zeros(max_samples - 1000, dtype=np.float32))
    with pytest.raises(StreamError, match="voice buffer exceeds"):
        stream.feed(np.zeros(2000, dtype=np.float32))


def test_safe_joblib_load_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    joblib = pytest.importorskip("joblib")
    model = tmp_path / "toy.joblib"
    joblib.dump({"ok": True}, model)
    digest = write_model_digest(model)
    assert hashlib.sha256(model.read_bytes()).hexdigest() == digest
    payload = safe_joblib_load(model, world_root=tmp_path)
    assert payload["ok"] is True
    # Tamper after pin.
    model.write_bytes(model.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="digest mismatch"):
        safe_joblib_load(model, world_root=tmp_path)
    # Unpinned + require digest.
    model2 = tmp_path / "toy2.joblib"
    joblib.dump({"ok": 2}, model2)
    monkeypatch.setenv("CHORUSFACE_REQUIRE_MODEL_DIGEST", "1")
    monkeypatch.delenv("CHORUSFACE_ALLOW_UNPINNED_MODELS", raising=False)
    with pytest.raises(ValueError, match="unpinned model"):
        safe_joblib_load(model2, world_root=tmp_path)
