"""Viseme-clock mouth layer appear / disappear."""

from __future__ import annotations

from aiface.mouth_timeline import (
    MAX_BRIDGE_GAP_S,
    MIN_SPEECH_DWELL_S,
    MouthLayerTimeline,
)


def test_open_appears_and_disappears_with_span() -> None:
    tl = MouthLayerTimeline()
    tl.fire("AH", now=1.0, duration=0.20, emotion="NEUTRAL")
    on = tl.tick(1.05, hard_snap=True)
    assert on.phoneme == "AH"
    assert on.open_amount == 1.0
    assert on.atlas_viseme == "AH"
    assert on.jaw_target > 0.5

    off = tl.tick(1.25, hard_snap=True)  # past until=1.20
    assert off.phoneme == "REST"
    assert off.open_amount == 0.0
    assert off.smile_amount == 0.0


def test_pp_clears_open_immediately() -> None:
    tl = MouthLayerTimeline()
    tl.fire("OH", now=0.0, duration=0.30)
    assert tl.tick(0.10, hard_snap=True).open_amount == 1.0
    tl.fire("PP", now=0.12, duration=0.15)
    cmd = tl.tick(0.13, hard_snap=True)
    assert cmd.phoneme == "PP"
    assert cmd.open_amount == 0.0
    assert cmd.jaw_target == 0.0
    assert cmd.smile_amount == 0.0


def test_rest_does_not_park_smile_with_zero_floor() -> None:
    tl = MouthLayerTimeline()
    tl.fire("REST", now=0.0, duration=1.0, emotion="HAPPY")
    cmd = tl.tick(0.1, width_n=0.9, smile_happy_floor=0.0, hard_snap=True)
    assert cmd.smile_amount == 0.0


def test_next_viseme_overwrites_span() -> None:
    tl = MouthLayerTimeline()
    tl.fire("AH", now=0.0, duration=1.0)
    tl.fire("EE", now=0.05, duration=0.10)
    cmd = tl.tick(0.06, hard_snap=True)
    assert cmd.phoneme == "EE"
    assert cmd.atlas_viseme == "EE"


def test_min_speech_dwell_extends_short_span() -> None:
    tl = MouthLayerTimeline()
    tl.fire("NN", now=1.0, duration=0.02)  # ~1 tick from energy align
    assert tl.active_until >= 1.0 + MIN_SPEECH_DWELL_S - 1e-6
    mid = tl.tick(1.05, hard_snap=True)
    assert mid.phoneme == "NN"
    assert mid.open_amount == 1.0


def test_hold_timing_slider_extends_dwell() -> None:
    tl = MouthLayerTimeline()
    tl.set_hold_timing(min_dwell_s=0.30, max_bridge_s=0.35)
    tl.fire("AH", now=0.0, duration=0.02)
    assert tl.active_until >= 0.30 - 1e-6
    assert tl.tick(0.20, hard_snap=True).open_amount == 1.0


def test_bridge_fills_gap_to_next_due() -> None:
    tl = MouthLayerTimeline()
    # Span ends at 1.05; next speech at 1.12 (70ms gap) — should bridge.
    tl.fire("AH", now=1.0, duration=0.05, next_due_at=1.12)
    assert tl.active_until == 1.12
    cmd = tl.tick(1.08, hard_snap=True)
    assert cmd.phoneme == "AH"
    assert cmd.open_amount == 1.0
    assert cmd.source == "timeline-bridge"


def test_tick_bridges_imminent_upcoming() -> None:
    tl = MouthLayerTimeline()
    tl.fire("AH", now=0.0, duration=0.05)  # until >= 0.08 from min dwell
    # Force expiry, then bridge via upcoming peek.
    tl._until = 0.04  # noqa: SLF001 — test expires early
    cmd = tl.tick(
        0.05,
        hard_snap=True,
        upcoming_due_at=0.05 + MAX_BRIDGE_GAP_S * 0.5,
        upcoming_phoneme="OH",
    )
    assert cmd.phoneme == "AH"
    assert cmd.open_amount == 1.0


def test_no_bridge_across_large_gap() -> None:
    tl = MouthLayerTimeline()
    tl.fire("AH", now=1.0, duration=0.05, next_due_at=1.0 + MIN_SPEECH_DWELL_S + 0.5)
    # Large gap: only min dwell, not all the way to next.
    assert tl.active_until == 1.0 + MIN_SPEECH_DWELL_S
    off = tl.tick(1.0 + MIN_SPEECH_DWELL_S + 0.01, hard_snap=True)
    assert off.phoneme == "REST"
