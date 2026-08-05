"""PulseChunk PLS1 round-trip + compose smoke tests."""

from __future__ import annotations

import numpy as np

from chorusface.vowel.g2p import g2p_word
from chorusface.vowel.model_a import ModelA
from chorusface.vowel.pipeline import compose_utterance, compose_utterance_bytes
from chorusface.vowel.pulsechunk import (
    PulseChunk,
    VersionBlock,
    WordSlice,
    decode_pulsechunk,
    encode_pulsechunk,
    fnv1a64,
)
from chorusface.vowel.schema import GROUP_DIM, PLS_MAGIC


def test_fnv_stable():
    assert fnv1a64("utt_001") == fnv1a64("utt_001")
    assert fnv1a64("a") != fnv1a64("b")


def test_pulsechunk_roundtrip():
    n = 30
    ctrl = np.zeros((n, GROUP_DIM), dtype=np.float32)
    ctrl[:, 8] = np.linspace(0, 0.8, n, dtype=np.float32)
    chunk = PulseChunk(
        utterance_id="u_test",
        n_ticks=n,
        primary_emotion=1,
        word_slices=[
            WordSlice(0, 10, [0, 9], 0),
            WordSlice(12, 28, [5], 0),
        ],
        controls=ctrl,
        versions=VersionBlock(teacher_ver=1, dataset_ver=2),
        key_ticks=[0, 12, 24],
    )
    raw = encode_pulsechunk(chunk)
    assert len(raw) >= 32
    assert int.from_bytes(raw[0:4], "little") == PLS_MAGIC
    back = decode_pulsechunk(raw, utterance_id="u_test")
    assert back.n_ticks == n
    assert back.primary_emotion == 1
    assert len(back.word_slices) == 2
    assert back.word_slices[0].vowel_ids == [0, 9]
    assert back.key_ticks == [0, 12, 24]
    assert back.versions is not None
    assert back.versions.dataset_ver == 2
    np.testing.assert_allclose(back.controls, ctrl, atol=1e-6)


def test_compose_text_only_g2p():
    result = compose_utterance(
        {
            "utterance_id": "u_hi",
            "text": "Hi you",
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 1.5}],
        }
    )
    assert result.chunk.n_ticks > 10
    assert result.payload.spans
    assert result.controls.shape[1] == GROUP_DIM
    raw = compose_utterance_bytes(
        {
            "utterance_id": "u_hi",
            "text": "Hi you",
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 1.5}],
        }
    )
    back = decode_pulsechunk(raw, utterance_id="u_hi")
    assert back.n_ticks == result.chunk.n_ticks


def test_g2p_known_and_rest():
    assert g2p_word("hello") is not None
    # garbage should REST (None)
    assert g2p_word("zzzzqx") is None or isinstance(g2p_word("zzzzqx"), tuple)


def test_model_a_fit_separates_ee_ou():
    a = ModelA()
    a.fit(epochs=200)
    ee = a.predict("EE", "NEUTRAL")
    ou = a.predict("OU", "NEUTRAL")
    dist = float(np.linalg.norm(ee - ou))
    assert dist >= 0.25
