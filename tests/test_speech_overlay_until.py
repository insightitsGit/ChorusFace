"""Absolute audio-clock overlay release (TickFeed word sync)."""

from __future__ import annotations

from aiface.speech import speech_overlay_until


def test_overlay_until_uses_scheduled_span_end() -> None:
    until = speech_overlay_until(
        now=1.05,
        due_at=1.00,
        duration=0.12,
        next_due_at=None,
    )
    assert until == 1.12


def test_overlay_until_never_overruns_next_due() -> None:
    until = speech_overlay_until(
        now=1.00,
        due_at=1.00,
        duration=0.20,
        next_due_at=1.08,
    )
    assert until == 1.08


def test_overlay_until_late_fire_still_one_frame() -> None:
    until = speech_overlay_until(
        now=1.20,
        due_at=1.00,
        duration=0.10,
        next_due_at=None,
        frame=1.0 / 60.0,
    )
    assert until == 1.20 + (1.0 / 60.0)


def test_overlay_until_closure_before_vowel_lands() -> None:
    # PP ends before next vowel due — must release for bilabial lock.
    until = speech_overlay_until(
        now=2.00,
        due_at=2.00,
        duration=0.05,
        next_due_at=2.05,
    )
    assert until == 2.05
