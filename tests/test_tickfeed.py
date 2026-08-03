"""TickFeed KEY/DELTA codec, ring damp, GPU pack — design contract tests."""

from __future__ import annotations

import numpy as np

from aiface.tickfeed.package import (
    FaceBox,
    TickLabels,
    apply_to_state,
    build_delta,
    build_hello,
    build_keyframe,
    decode,
    encode,
    negotiate_hello,
)
from aiface.tickfeed.ring import FaceVelocityState, LockstepPlayer
from aiface.tickfeed.schema import BeatId
from aiface.tickfeed.gpu_pack import (
    dense_uints_from_package,
    ingest_encoding,
    pack_half2_uints,
    sparse_buffers_from_package,
)
from aiface.tickfeed.schema import DeltaEncoding, PackageKind, ValueDtype


def _face() -> FaceBox:
    return FaceBox(x=2, y=3, w=8, h=4)


def test_keyframe_roundtrip_f16() -> None:
    face = _face()
    values = np.zeros((face.h, face.w, 2), dtype=np.float32)
    values[1, 2] = (0.5, -0.25)
    labels = TickLabels(
        beat_id=int(BeatId.SMILE),
        smile_amt=0.8,
        word="hi",
    )
    pkg = build_keyframe(0, face, values, labels=labels, value_dtype=ValueDtype.F16)
    blob = encode(pkg)
    assert len(blob) >= 64 + 48
    back = decode(blob)
    assert back.kind == PackageKind.KEYFRAME
    assert back.tick == 0
    assert back.face.as_tuple() == face.as_tuple()
    assert back.labels is not None
    assert back.labels.beat_id == int(BeatId.SMILE)
    assert back.labels.word == "hi"
    assert abs(back.labels.smile_amt - 0.8) < 1e-3
    np.testing.assert_allclose(back.values[1, 2], (0.5, -0.25), atol=1e-3)


def test_sparse_delta_roundtrip_and_apply() -> None:
    face = _face()
    prev = np.zeros((face.h, face.w, 2), dtype=np.float32)
    curr = prev.copy()
    curr[0, 0] = (0.1, 0.0)
    curr[2, 3] = (0.0, -0.2)
    pkg = build_delta(1, face, prev, curr, value_dtype=ValueDtype.F16)
    assert pkg.delta_encoding == DeltaEncoding.SPARSE_DELTA
    assert pkg.sparse_idx is not None
    assert pkg.sparse_idx.size == 2

    blob = encode(pkg)
    back = decode(blob)
    state = apply_to_state(prev, back)
    np.testing.assert_allclose(state[0, 0], (0.1, 0.0), atol=1e-3)
    np.testing.assert_allclose(state[2, 3], (0.0, -0.2), atol=1e-3)


def test_empty_delta() -> None:
    face = _face()
    z = np.zeros((face.h, face.w, 2), dtype=np.float32)
    pkg = build_delta(2, face, z, z)
    assert pkg.delta_encoding == DeltaEncoding.EMPTY
    back = decode(encode(pkg))
    assert back.delta_encoding == DeltaEncoding.EMPTY
    out = apply_to_state(z, back)
    np.testing.assert_array_equal(out, z)


def test_dense_delta_when_many_change() -> None:
    face = _face()
    prev = np.zeros((face.h, face.w, 2), dtype=np.float32)
    curr = np.ones((face.h, face.w, 2), dtype=np.float32) * 0.05
    pkg = build_delta(3, face, prev, curr)
    assert pkg.delta_encoding == DeltaEncoding.DENSE_DELTA
    state = apply_to_state(prev, decode(encode(pkg)))
    np.testing.assert_allclose(state, curr, atol=1e-3)


def test_lockstep_ring_damp_on_miss() -> None:
    face = _face()
    player = LockstepPlayer(state=FaceVelocityState.zeros(face))
    values = np.zeros((face.h, face.w, 2), dtype=np.float32)
    values[:, :] = (1.0, 0.5)
    player.submit(build_keyframe(0, face, values))
    assert player.step() == "keyframe"
    np.testing.assert_allclose(player.state.velocity[0, 0], (1.0, 0.5), atol=1e-3)
    # tick 1 missing → damp
    assert player.step() == "damp"
    assert float(player.state.velocity[0, 0, 0]) < 1.0


def test_hello_negotiate_roundtrip() -> None:
    face = _face()
    hello = build_hello(face, world_id="avatar")
    blob = encode(hello)
    back = decode(blob)
    assert back.kind == PackageKind.HELLO
    assert back.hello is not None
    assert back.hello.world_id == "avatar"
    assert back.hello.is_ack is False
    ack = negotiate_hello(back)
    assert ack.hello is not None and ack.hello.ok is True
    assert ack.hello.apply_mode == "velocity_write"
    ack2 = decode(encode(ack))
    assert ack2.hello is not None and ack2.hello.is_ack is True


def test_sparse_delta_with_conf() -> None:
    face = _face()
    prev = np.zeros((face.h, face.w, 2), dtype=np.float32)
    curr = prev.copy()
    curr[0, 0] = (0.1, 0.0)
    conf = np.full(face.n_cells, 200, dtype=np.uint8)
    conf[0] = 255
    pkg = build_delta(1, face, prev, curr, conf=conf, value_dtype=ValueDtype.F16)
    assert pkg.flags & 2  # FLAG_HAS_CONF
    back = decode(encode(pkg))
    assert back.conf is not None
    assert int(back.conf[0]) == 255


def test_crc_covers_header_prefix_and_body() -> None:
    """Handshake: crc32(header[0..35] + body), not payload alone."""
    import zlib

    face = _face()
    values = np.zeros((face.h, face.w, 2), dtype=np.float32)
    values[0, 0] = (0.1, -0.2)
    pkg = build_keyframe(0, face, values, value_dtype=ValueDtype.F16)
    blob = encode(pkg)
    payload_len = int.from_bytes(blob[32:36], "little")
    payload = blob[64 : 64 + payload_len]
    crc = int.from_bytes(blob[36:40], "little")
    assert crc == (zlib.crc32(blob[:36] + payload) & 0xFFFFFFFF)
    assert crc != (zlib.crc32(payload) & 0xFFFFFFFF) or len(payload) == 0
    decode(blob)  # must accept


def test_gpu_pack_dense_and_sparse() -> None:
    face = _face()
    values = np.random.randn(face.h, face.w, 2).astype(np.float32) * 0.1
    key = build_keyframe(0, face, values, value_dtype=ValueDtype.F16)
    packed = dense_uints_from_package(key)
    assert packed.shape == (face.n_cells,)
    assert ingest_encoding(key) == 0

    prev = np.zeros((face.h, face.w, 2), dtype=np.float32)
    curr = prev.copy()
    curr[0, 0] = (0.2, 0.1)
    delta = build_delta(1, face, prev, curr, value_dtype=ValueDtype.F16)
    assert delta.delta_encoding == DeltaEncoding.SPARSE_DELTA
    idx, vel = sparse_buffers_from_package(delta)
    assert idx.size == vel.size == 1
    assert ingest_encoding(delta) == 2

    # half2 roundtrip rough check
    u = pack_half2_uints(np.array([[0.5, -0.25]], dtype=np.float32))
    raw = np.frombuffer(u.tobytes(), dtype="<f2").astype(np.float32)
    np.testing.assert_allclose(raw, (0.5, -0.25), atol=1e-3)
