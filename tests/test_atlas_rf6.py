"""RF6 — AA beat time-hint + thicker vowel oral matte (keep open.png)."""

from __future__ import annotations

from types import SimpleNamespace

from chorusface.plates import score_frame_for_viseme


def _frame(*, mouth_open: float, smile: float, t: float, teeth: float = 0.5) -> object:
    return SimpleNamespace(
        time_seconds=t,
        metrics=SimpleNamespace(
            mouth_open=mouth_open,
            smile_width=smile,
            teeth=teeth,
            sharpness=100.0,
        ),
    )


def test_rf6_aa_prefers_open_beat_window() -> None:
    # Same landmarks; OPEN beat (1.4–2.3) should score better than smile beat.
    in_open = _frame(mouth_open=0.35, smile=0.2, t=1.8)
    in_smile = _frame(mouth_open=0.35, smile=0.2, t=1.0)
    kwargs = dict(open_lo=0.05, open_hi=0.45, smile_lo=0.1, smile_hi=0.5)
    assert score_frame_for_viseme(in_open, "AA", **kwargs) < score_frame_for_viseme(
        in_smile, "AA", **kwargs
    )


def test_rf6_capture_thicker_vowel_matte() -> None:
    from pathlib import Path

    text = Path("src/chorusface/capture.py").read_text(encoding="utf-8")
    assert "RF6: vowels need a taller oral disk" in text
    assert "matte_open = max(matte_open, 0.38)" in text
    plates = Path("src/chorusface/plates.py").read_text(encoding="utf-8")
    assert "_VISEME_TIME_HINTS" in plates
