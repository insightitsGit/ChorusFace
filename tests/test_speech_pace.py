"""Speech pace stretches audio and viseme spans together."""

from __future__ import annotations

import numpy as np

from aiface.audio import AudioClip, time_stretch
from aiface.speech import PhonemeSpan
from aiface.tts import PreparedSpeech, WordSpan, apply_speech_pace


def test_time_stretch_lengthens_clip() -> None:
    clip = AudioClip(
        samples=np.linspace(-0.2, 0.2, 1000, dtype=np.float32),
        sample_rate=1000,
    )
    slow = time_stretch(clip, 1.12)
    assert abs(slow.duration - 1.12) < 1e-3
    assert slow.sample_rate == 1000


def test_apply_speech_pace_keeps_span_ratio() -> None:
    clip = AudioClip(
        samples=np.zeros(1000, dtype=np.float32),
        sample_rate=1000,
    )
    speech = PreparedSpeech(
        text="ah",
        clip=clip,
        spans=(
            PhonemeSpan("AH", 0.0, 0.40),
            PhonemeSpan("EE", 0.40, 0.80),
            PhonemeSpan("REST", 0.80, 1.0),
        ),
        words=(WordSpan("ah", 0.0, 0.8),),
        voice="test",
        alignment="linear",
    )
    paced = apply_speech_pace(speech, 1.25, min_hold=0.0)
    assert abs(paced.duration - 1.25) < 1e-3
    assert abs(paced.spans[0].start - 0.0) < 1e-6
    assert abs(paced.spans[0].end - 0.50) < 1e-3
    assert abs(paced.spans[1].start - 0.50) < 1e-3
    assert abs(paced.words[0].end - 1.0) < 1e-3


def test_apply_speech_pace_min_hold() -> None:
    clip = AudioClip(
        samples=np.zeros(1000, dtype=np.float32),
        sample_rate=1000,
    )
    speech = PreparedSpeech(
        text="t",
        clip=clip,
        spans=(PhonemeSpan("DD", 0.10, 0.14), PhonemeSpan("REST", 0.14, 1.0)),
    )
    paced = apply_speech_pace(speech, 1.0, min_hold=0.12)
    assert paced.spans[0].duration + 1e-9 >= 0.12
    assert paced.spans[0].end <= paced.spans[1].start + 1e-9
