"""Bilabial onset pin — PP/MM readable before the vowel."""

from __future__ import annotations

from chorusface.speech import TOKEN_WORD, PhonemeSpan, SpokenToken
from chorusface.tts import bias_bilabial_onsets, _subdivide


def test_subdivide_pins_leading_pp() -> None:
    token = SpokenToken(text="pa", kind=TOKEN_WORD, visemes=("PP", "AH"))
    spans = _subdivide(token, 0.0, 0.20)
    assert spans[0].phoneme == "PP"
    assert spans[0].end - spans[0].start >= 0.04
    assert spans[0].start == 0.0
    assert spans[-1].end == 0.20


def test_bias_borrows_from_following_vowel() -> None:
    spans = [
        PhonemeSpan("PP", 0.00, 0.02),
        PhonemeSpan("EE", 0.02, 0.20),
    ]
    out = bias_bilabial_onsets(spans, pin=0.045)
    assert out[0].end - out[0].start == 0.045
    assert out[1].start == 0.045
    assert out[1].end == 0.20
