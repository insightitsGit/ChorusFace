"""P3 mouth motion phases — LOOK/FIELD policy only."""

from __future__ import annotations

from aiface.mouth_motion import (
    PHASE_ANTICIPATING,
    PHASE_CLOSING,
    PHASE_HOLDING,
    PHASE_OPENING,
    PHASE_RELAXING,
    PHASE_REST,
    MouthMotionState,
)


def test_motion_phases_opening_hold_closing() -> None:
    m = MouthMotionState()
    t = 0.0
    assert m.step(target_open=0.0, now=t) == PHASE_REST
    t += 1.0 / 60.0
    m.step(target_open=0.2, now=t)
    t += 1.0 / 60.0
    phase = m.step(target_open=0.7, now=t)
    assert phase in {PHASE_OPENING, PHASE_HOLDING, PHASE_ANTICIPATING}
    # Hold open.
    for _ in range(8):
        t += 1.0 / 60.0
        m.step(target_open=0.75, now=t)
    assert m.phase == PHASE_HOLDING
    assert m.transition_state() == "OPEN"
    # Close.
    t += 1.0 / 60.0
    m.step(target_open=0.4, now=t)
    t += 1.0 / 60.0
    m.step(target_open=0.05, now=t)
    assert m.phase in {PHASE_CLOSING, PHASE_RELAXING, PHASE_REST}
    assert m.transition_state() in {"CLOSING", "REST", "OPENING"}
