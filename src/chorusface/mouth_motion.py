"""Temporal mouth phase controller (P3) — LOOK/FIELD policy only.

Does **not** replace open.png dual-owner or atlas plates. Phases drive
transition naming / FIELD mute hints already consumed by mouth_owner.
"""

from __future__ import annotations

from dataclasses import dataclass

PHASE_REST = "REST"
PHASE_ANTICIPATING = "ANTICIPATING"
PHASE_OPENING = "OPENING"
PHASE_HOLDING = "HOLDING"
PHASE_CLOSING = "CLOSING"
PHASE_RELAXING = "RELAXING"

# Map rich phases → legacy transition strings used by look_field_gain_scale.
_LEGACY = {
    PHASE_REST: "REST",
    PHASE_ANTICIPATING: "OPENING",
    PHASE_OPENING: "OPENING",
    PHASE_HOLDING: "OPEN",
    PHASE_CLOSING: "CLOSING",
    PHASE_RELAXING: "CLOSING",
}


@dataclass(slots=True)
class MouthMotionState:
    phase: str = PHASE_REST
    open_amt: float = 0.0
    velocity: float = 0.0
    _prev_open: float = 0.0
    _prev_t: float = 0.0

    def step(
        self,
        *,
        target_open: float,
        now: float,
        plate_committed: float | None = None,
    ) -> str:
        open_n = max(0.0, min(1.0, float(target_open)))
        if plate_committed is not None:
            open_n = max(open_n, max(0.0, min(1.0, float(plate_committed))))
        dt = 1.0 / 60.0
        if self._prev_t > 0.0:
            dt = max(1e-4, min(0.10, float(now) - self._prev_t))
        raw_vel = (open_n - self._prev_open) / dt
        self.velocity = 0.65 * self.velocity + 0.35 * raw_vel
        self._prev_open = open_n
        self._prev_t = float(now)
        self.open_amt = open_n
        vel = self.velocity

        if open_n < 0.06 and abs(vel) < 0.6:
            self.phase = PHASE_REST
        elif vel > 0.35 and open_n < 0.12:
            self.phase = PHASE_ANTICIPATING
        elif vel > 0.9 or (0.08 <= open_n < 0.55 and vel > 0.2):
            self.phase = PHASE_OPENING
        elif vel < -0.9 or (open_n > 0.12 and vel < -0.2):
            self.phase = PHASE_CLOSING
        elif open_n >= 0.45 and abs(vel) <= 0.9:
            self.phase = PHASE_HOLDING
        elif open_n < 0.20 and vel < 0.0:
            self.phase = PHASE_RELAXING
        elif open_n >= 0.08:
            self.phase = PHASE_OPENING if vel >= 0.0 else PHASE_CLOSING
        else:
            self.phase = PHASE_REST
        return self.phase

    def transition_state(self) -> str:
        """Legacy REST/OPENING/OPEN/CLOSING for FIELD gain + plate commit."""
        return _LEGACY.get(self.phase, "REST")


__all__ = [
    "PHASE_ANTICIPATING",
    "PHASE_CLOSING",
    "PHASE_HOLDING",
    "PHASE_OPENING",
    "PHASE_RELAXING",
    "PHASE_REST",
    "MouthMotionState",
]
