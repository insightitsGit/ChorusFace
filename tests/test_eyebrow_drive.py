"""Eyebrow drive must not stay parked at zero under HAPPY / surprise."""

from __future__ import annotations

from chorusface.biomechanics.emotion import EmotionSystem
from chorusface.biomechanics.face import BiomechanicalFace


def test_happy_emotion_drives_frontalis_axes() -> None:
    emotion = EmotionSystem()
    emotion.from_label("HAPPY")
    assert emotion.state.arousal >= 0.5
    assert emotion.state.curiosity >= 0.25
    assert emotion.state.valence >= 0.8


def test_happy_face_reports_nonzero_brow_raise() -> None:
    face = BiomechanicalFace.from_file(seed=3)
    face.emotion.from_label("HAPPY")
    render = None
    for tick in range(12):
        render, _field = face.step(1.0 / 60.0, tick=tick)
    assert render is not None
    assert render.brow_raise > 0.08
    assert render.eye_widen > 0.05


def test_surprised_emotion_raises_brows_hard() -> None:
    face = BiomechanicalFace.from_file(seed=3)
    face.emotion.from_label("SURPRISED")
    render = None
    for tick in range(12):
        render, _field = face.step(1.0 / 60.0, tick=tick)
    assert render is not None
    assert render.brow_raise > 0.35
    assert render.eye_widen > 0.35
