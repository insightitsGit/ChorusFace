"""W expand + TickPackage emit tests."""

from __future__ import annotations

import numpy as np

from chorusface.tickfeed.package import FaceBox, decode
from chorusface.tickfeed.schema import PackageKind
from chorusface.vowel.expand import author_w_from_catalog, expand_controls, save_wexpand, load_wexpand
from chorusface.vowel.pipeline import compose_utterance
from chorusface.vowel.tick_emit import EmitConfig, emit_encoded_bytes, emit_tick_packages


def test_author_w_synthetic_and_expand():
    W, cells = author_w_from_catalog({})
    assert W.shape[0] == 9
    assert W.shape[1] == len(cells) > 100
    c = np.zeros(9, dtype=np.float32)
    c[8] = 0.9
    c[5] = 0.8
    vx, vy = expand_controls(c, W, cells)
    assert vx.shape == vy.shape == (len(cells),)
    assert float(np.max(np.abs(vy))) > 0.0


def test_wexpand_roundtrip(tmp_path):
    W, cells = author_w_from_catalog({})
    path = tmp_path / "x.wexpand"
    save_wexpand(path, W, cells, decoder_ver=3)
    W2, cells2, ver = load_wexpand(path)
    assert ver == 3
    assert cells2 == cells
    np.testing.assert_allclose(W2, W)


def test_emit_tick_packages_key_delta():
    result = compose_utterance(
        {
            "utterance_id": "u_emit",
            "text": "See you",
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 2.0}],
        }
    )
    W, cells = author_w_from_catalog({})
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    face = FaceBox(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
    cfg = EmitConfig(face=face, W=W, cells=cells)
    pkgs = emit_tick_packages(result.chunk, cfg)
    assert len(pkgs) == result.chunk.n_ticks
    assert pkgs[0].kind == PackageKind.KEYFRAME
    assert pkgs[0].labels is not None
    raws = emit_encoded_bytes(result.chunk, cfg)
    assert len(raws) == len(pkgs)
    back = decode(raws[0])
    assert back.kind == PackageKind.KEYFRAME


def test_f8_size_based_key_crossover():
    """When Δ encode size ≥ KEY size, emit KEY (not mean|Δ| heuristic)."""
    from chorusface.tickfeed.package import build_delta, build_keyframe, encode
    from chorusface.vowel.pulsechunk import PulseChunk
    from chorusface.vowel.schema import TICK_HZ
    from chorusface.vowel.tick_emit import controls_to_velocity_grid

    W, cells = author_w_from_catalog({})
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    face = FaceBox(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
    cfg = EmitConfig(face=face, W=W, cells=cells)
    n = 24
    controls = np.zeros((n, 9), dtype=np.float32)
    # Alternate extreme oral/eye rows so many Δ payloads bloat past KEY.
    for t in range(n):
        if t % 2 == 0:
            controls[t] = np.array(
                [0.9, 0.1, 0.8, 0.7, 0.9, 0.95, 0.2, 0.9, 0.95], dtype=np.float32
            )
        else:
            controls[t] = np.array(
                [0.05, 0.9, 0.05, 0.05, 0.05, 0.05, 0.9, 0.05, 0.05], dtype=np.float32
            )
    chunk = PulseChunk(
        utterance_id="f8_size",
        n_ticks=n,
        primary_emotion=0,
        word_slices=[],
        controls=controls,
        tick_hz=TICK_HZ,
        key_ticks=[0, 12],
    )
    pkgs = emit_tick_packages(chunk, cfg)
    assert pkgs[0].kind == PackageKind.KEYFRAME
    assert pkgs[12].kind == PackageKind.KEYFRAME
    prev = controls_to_velocity_grid(controls[0], cfg)
    size_keys = 0
    for t in range(1, n):
        if t == 12:
            prev = controls_to_velocity_grid(controls[t], cfg)
            continue
        grid = controls_to_velocity_grid(controls[t], cfg)
        key_pkg = build_keyframe(t, face, grid)
        delta_pkg = build_delta(t, face, prev, grid)
        if len(encode(delta_pkg)) >= len(encode(key_pkg)):
            assert pkgs[t].kind == PackageKind.KEYFRAME
            size_keys += 1
        else:
            assert pkgs[t].kind == PackageKind.DELTA
        prev = grid
    assert size_keys >= 1


def test_push_to_transport_roundtrip():
    from chorusface.vowel.runtime import VowelRuntime

    result = compose_utterance(
        {
            "utterance_id": "u_tpk",
            "text": "Hello",
            "emotion_track": [{"emotion": "NEUTRAL", "start_s": 0.0, "end_s": 2.0}],
        }
    )
    rt = VowelRuntime()
    W, cells = author_w_from_catalog({})
    rt.W, rt.cells = W, cells
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    rt.face = FaceBox(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)

    class _FakeTransport:
        def __init__(self) -> None:
            self.items: list[tuple[int, bytes, int | None]] = []

        def push_package_bytes(
            self, tick: int, raw: bytes, *, kind: int | None = None
        ) -> None:
            self.items.append((int(tick), bytes(raw), kind))

    transport = _FakeTransport()
    n = rt.push_to_transport(result.chunk, transport)
    assert n == result.chunk.n_ticks == len(transport.items)
    assert transport.items[0][1][:4]  # non-empty TPK bytes
    back = decode(transport.items[0][1])
    assert back.kind == PackageKind.KEYFRAME
