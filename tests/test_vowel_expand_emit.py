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
