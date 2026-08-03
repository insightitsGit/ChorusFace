"""Blink envelope: slow close, visible full shut, slower open."""

from __future__ import annotations

from aiface.biomechanics.eyes import (
    BLINK_CLOSE_S,
    BLINK_HOLD_S,
    BLINK_OPEN_S,
    BLINK_TOTAL_S,
    EyeSystem,
    _lid_close_amount,
)


def test_blink_total_is_readable() -> None:
    # Was ~0.24s with an instant snap-shut; need a clear close/hold/open.
    assert BLINK_TOTAL_S >= 0.35
    assert BLINK_OPEN_S > BLINK_CLOSE_S
    assert BLINK_HOLD_S >= 0.05


def test_lid_envelope_reaches_full_close() -> None:
    # Mid-hold should be fully closed.
    mid_hold_remaining = BLINK_OPEN_S + BLINK_HOLD_S * 0.5
    assert _lid_close_amount(mid_hold_remaining) == 1.0
    # Start of blink (full remaining) is still open.
    assert _lid_close_amount(BLINK_TOTAL_S) < 0.05
    # End of blink is open again.
    assert _lid_close_amount(0.0) == 0.0


def test_request_blink_runs_full_duration() -> None:
    eyes = EyeSystem(seed=7)
    eyes.request_blink()
    dt = 1.0 / 60.0
    saw_closed = False
    steps = 0
    max_steps = int(BLINK_TOTAL_S / dt) + 5
    while eyes.state.blink_phase > 0.0 and steps < max_steps:
        eyes.step(dt, arousal=0.2, emit_impulses=False)
        if min(eyes.state.lid_left, eyes.state.lid_right) < 0.05:
            saw_closed = True
        steps += 1
    assert saw_closed
    assert eyes.state.blink_phase == 0.0
    assert eyes.state.lid_left == 1.0
    # Must have taken most of the envelope (not a 3-frame flicker).
    assert steps * dt >= BLINK_TOTAL_S * 0.9
