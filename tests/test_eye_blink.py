"""Blink envelope: slow close, visible full shut, slower open."""

from __future__ import annotations

from aiface.biomechanics.eyes import (
    BLINK_CLOSE_S,
    BLINK_HOLD_S,
    BLINK_MAX_STEP_S,
    BLINK_OPEN_S,
    BLINK_STATE_CLOSED,
    BLINK_STATE_CLOSING,
    BLINK_STATE_OPEN,
    BLINK_STATE_OPENING,
    BLINK_TOTAL_S,
    EyeSystem,
    _lid_close_amount,
    blink_state_from_remaining,
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


def test_blink_state_machine() -> None:
    assert blink_state_from_remaining(0.0) == BLINK_STATE_OPEN
    assert blink_state_from_remaining(BLINK_TOTAL_S) == BLINK_STATE_CLOSING
    assert (
        blink_state_from_remaining(BLINK_OPEN_S + BLINK_HOLD_S * 0.5)
        == BLINK_STATE_CLOSED
    )
    assert blink_state_from_remaining(BLINK_OPEN_S * 0.5) == BLINK_STATE_OPENING


def test_large_dt_cannot_skip_closed_hold() -> None:
    eyes = EyeSystem(seed=3)
    eyes.request_blink()
    # One huge hitch — phase must only advance by BLINK_MAX_STEP_S.
    eyes.step(0.5, arousal=0.1, emit_impulses=False)
    assert eyes.state.blink_phase >= BLINK_TOTAL_S - BLINK_MAX_STEP_S - 1e-6
    assert eyes.state.blink_state == BLINK_STATE_CLOSING


def test_request_blink_runs_full_duration() -> None:
    eyes = EyeSystem(seed=7)
    eyes.request_blink()
    dt = 1.0 / 60.0
    saw_closed = False
    saw_states: set[str] = set()
    steps = 0
    max_steps = int(BLINK_TOTAL_S / dt) + 5
    while eyes.state.blink_phase > 0.0 and steps < max_steps:
        eyes.step(dt, arousal=0.2, emit_impulses=False)
        saw_states.add(eyes.state.blink_state)
        if min(eyes.state.lid_left, eyes.state.lid_right) < 0.05:
            saw_closed = True
        steps += 1
    assert saw_closed
    assert eyes.state.blink_phase == 0.0
    assert eyes.state.lid_left == 1.0
    assert eyes.state.blink_state == BLINK_STATE_OPEN
    assert BLINK_STATE_CLOSING in saw_states
    assert BLINK_STATE_CLOSED in saw_states
    # Must have taken most of the envelope (not a 3-frame flicker).
    assert steps * dt >= BLINK_TOTAL_S * 0.9
