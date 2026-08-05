"""Blink overlay tests."""

from __future__ import annotations

import numpy as np

from chorusface.vowel.blinks import apply_blinks, blink_envelope, plan_blink_starts
from chorusface.vowel.pipeline import compose_utterance


def test_blink_envelope_peaks_at_one():
    env = blink_envelope()
    assert env.max() == 1.0
    assert env[0] > 0
    assert env[-1] < 1.0


def test_plan_blinks_on_long_clip():
    starts = plan_blink_starts(300, interval_s=3.0, seed=1)
    assert len(starts) >= 1
    assert starts[0] >= 1


def test_apply_blinks_closes_eyes():
    ctrl = np.zeros((180, 9), dtype=np.float64)
    out = apply_blinks(ctrl, interval_s=2.5, seed=0, enabled=True)
    assert float(out[:, 0].max()) >= 0.99
    # mouth untouched
    np.testing.assert_allclose(out[:, 4:], 0.0)


def test_compose_includes_blink_by_default():
    result = compose_utterance(
        {
            "utterance_id": "b1",
            "text": "Hello how are you today please",
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 4.0}],
        }
    )
    assert float(result.controls[:, 0].max()) >= 0.9


def test_compose_can_disable_blinks():
    result = compose_utterance(
        {
            "utterance_id": "b0",
            "text": "Hello how are you today please",
            "blinks": False,
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 4.0}],
        },
        blinks=False,
    )
    # emotion bias may raise aperture a bit, but not a full blink shut
    assert float(result.controls[:, 0].max()) < 0.6
