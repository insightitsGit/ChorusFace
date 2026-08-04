"""Deterministic eye simulation: gaze, blinks, microsaccades, pupil."""

from __future__ import annotations

from dataclasses import dataclass, field

from chorusface.biomechanics.muscles import MuscleImpulse

# Readable human-ish blink (seconds). Close is brisk; open is slower;
# a short fully-closed hold so the lids actually read as shut on camera.
BLINK_CLOSE_S: float = 0.12
BLINK_HOLD_S: float = 0.08
BLINK_OPEN_S: float = 0.22
BLINK_TOTAL_S: float = BLINK_CLOSE_S + BLINK_HOLD_S + BLINK_OPEN_S

# Cap per-step dt so a low-FPS hitch cannot skim the closed hold
# (same idea as absolute speech overlay until for mouths).
BLINK_MAX_STEP_S: float = 1.0 / 20.0

BLINK_STATE_OPEN: str = "OPEN"
BLINK_STATE_CLOSING: str = "CLOSING"
BLINK_STATE_CLOSED: str = "CLOSED"
BLINK_STATE_OPENING: str = "OPENING"


@dataclass(slots=True)
class EyeState:
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    pupil: float = 0.45
    lid_left: float = 1.0
    lid_right: float = 1.0
    blink_timer: float = 0.0
    # Seconds remaining in the active blink (0 = idle). Was a 0..1 phase that
    # decayed too fast and snapped shut on the first frame.
    blink_phase: float = 0.0
    blink_state: str = BLINK_STATE_OPEN
    # When > 0, hold the current blink phase (QA / forced closed still).
    blink_pause_s: float = 0.0
    focus_delay: float = 0.18
    lid_tension: float = 0.0


def _lid_close_amount(remaining: float) -> float:
    """Map blink remaining-time → lid closure 0..1 (1 = fully closed)."""
    if remaining <= 0.0:
        return 0.0
    elapsed = BLINK_TOTAL_S - min(remaining, BLINK_TOTAL_S)
    if elapsed < BLINK_CLOSE_S:
        # Ease-in close so the motion is visible, not a one-frame snap.
        t = elapsed / max(BLINK_CLOSE_S, 1e-4)
        return t * t * (3.0 - 2.0 * t)
    if elapsed < BLINK_CLOSE_S + BLINK_HOLD_S:
        return 1.0
    open_t = (elapsed - BLINK_CLOSE_S - BLINK_HOLD_S) / max(BLINK_OPEN_S, 1e-4)
    open_t = max(0.0, min(1.0, open_t))
    # Ease-out open (slower release near full open).
    u = open_t * open_t * (3.0 - 2.0 * open_t)
    return 1.0 - u


def blink_state_from_remaining(remaining: float) -> str:
    """Single-owner blink state (mouth OPENING/OPEN/CLOSING analogue)."""
    if remaining <= 0.0:
        return BLINK_STATE_OPEN
    elapsed = BLINK_TOTAL_S - min(remaining, BLINK_TOTAL_S)
    if elapsed < BLINK_CLOSE_S:
        return BLINK_STATE_CLOSING
    if elapsed < BLINK_CLOSE_S + BLINK_HOLD_S:
        return BLINK_STATE_CLOSED
    return BLINK_STATE_OPENING


@dataclass(slots=True)
class EyeSystem:
    """Eyes influence nearby facial muscles; they never write pixels."""

    state: EyeState = field(default_factory=EyeState)
    seed: int = 1
    _rng: int = 1

    def __post_init__(self) -> None:
        self._rng = int(self.seed) & 0x7FFFFFFF or 1

    def _next_unit(self) -> float:
        # Deterministic LCG, same family used by idle/breathing systems.
        self._rng = (1103515245 * self._rng + 12345) & 0x7FFFFFFF
        return self._rng / 0x7FFFFFFF

    def look_at(self, x: float, y: float) -> None:
        self.state.target_x = max(-1.0, min(1.0, x))
        self.state.target_y = max(-1.0, min(1.0, y))

    def request_blink(self) -> None:
        """Blink on the next step regardless of the scheduled interval.

        Zeroing ``blink_timer`` from outside does not work: :meth:`set_arousal`
        runs first in :meth:`step` and re-arms an expired timer, so a caller
        that wants a blink now has to start the phase itself.
        """
        if self.state.blink_phase <= 0.0:
            self.state.blink_phase = BLINK_TOTAL_S
            self.state.blink_state = BLINK_STATE_CLOSING
            self.state.blink_timer = 2.8 + self._next_unit() * 2.4

    def set_arousal(self, arousal: float) -> None:
        # Higher arousal → larger pupils and more frequent blinks.
        self.state.pupil = max(0.15, min(0.95, 0.35 + 0.4 * max(0.0, arousal)))
        interval = 3.4 - 1.2 * max(0.0, arousal)
        if self.state.blink_timer <= 0.0:
            self.state.blink_timer = interval + self._next_unit() * 1.5

    def closure_amount(self) -> float:
        """0 = open, 1 = fully closed (max of both lids)."""
        return max(
            0.0,
            min(
                1.0,
                1.0 - min(float(self.state.lid_left), float(self.state.lid_right)),
            ),
        )

    def step(
        self,
        dt: float,
        *,
        arousal: float = 0.0,
        emit_impulses: bool = True,
    ) -> list[MuscleImpulse]:
        self.set_arousal(arousal)
        delay = max(self.state.focus_delay, 1e-3)
        amount = 1.0 - pow(0.5, dt / delay)
        self.state.gaze_x += (self.state.target_x - self.state.gaze_x) * amount
        self.state.gaze_y += (self.state.target_y - self.state.gaze_y) * amount

        # Microsaccades: tiny deterministic offsets.
        saccade = 0.012 + 0.01 * abs(arousal)
        self.state.gaze_x += (self._next_unit() * 2.0 - 1.0) * saccade * dt * 60.0
        self.state.gaze_y += (self._next_unit() * 2.0 - 1.0) * saccade * dt * 60.0
        self.state.gaze_x = max(-1.0, min(1.0, self.state.gaze_x))
        self.state.gaze_y = max(-1.0, min(1.0, self.state.gaze_y))

        impulses: list[MuscleImpulse] = []
        self.state.blink_timer -= dt
        # Cap dt so a hitch cannot erase the closed hold in one step.
        step = min(max(float(dt), 0.0), BLINK_MAX_STEP_S)
        if self.state.blink_pause_s > 0.0 and self.state.blink_phase > 0.0:
            self.state.blink_pause_s = max(0.0, self.state.blink_pause_s - max(float(dt), 0.0))
            close = _lid_close_amount(self.state.blink_phase)
            self.state.blink_state = blink_state_from_remaining(self.state.blink_phase)
            self.state.lid_left = 1.0 - close
            self.state.lid_right = 1.0 - min(1.0, close * 0.94)
            self.state.lid_tension = close
            return impulses
        if self.state.blink_phase > 0.0:
            self.state.blink_phase = max(0.0, self.state.blink_phase - step)
            close = _lid_close_amount(self.state.blink_phase)
            self.state.blink_state = blink_state_from_remaining(self.state.blink_phase)
            # Mild asymmetry: left lid leads slightly.
            self.state.lid_left = 1.0 - close
            self.state.lid_right = 1.0 - min(1.0, close * 0.94)
            self.state.lid_tension = close
            # When TickFeed owns FIELD, skip blink muscle warps — lid overlay
            # alone. Muscle+field stacking reads as double motion on lids.
            if emit_impulses and close > 0.2:
                impulses.append(
                    MuscleImpulse(
                        tick=0,
                        muscle="OrbicularisOculi",
                        strength=close,
                        duration=0.05,
                        falloff=1.0,
                        priority=3,
                        source="Blink",
                    )
                )
            if self.state.blink_phase <= 0.0:
                self.state.blink_state = BLINK_STATE_OPEN
                self.state.lid_left = 1.0
                self.state.lid_right = 1.0
                self.state.lid_tension = 0.0
        elif self.state.blink_timer <= 0.0:
            self.state.blink_phase = BLINK_TOTAL_S
            self.state.blink_state = BLINK_STATE_CLOSING
            self.state.blink_timer = 2.8 + self._next_unit() * 2.4
        else:
            self.state.lid_left = 1.0
            self.state.lid_right = 1.0
            self.state.lid_tension = 0.0
            self.state.blink_state = BLINK_STATE_OPEN

        return impulses


__all__ = [
    "BLINK_CLOSE_S",
    "BLINK_HOLD_S",
    "BLINK_MAX_STEP_S",
    "BLINK_OPEN_S",
    "BLINK_STATE_CLOSED",
    "BLINK_STATE_CLOSING",
    "BLINK_STATE_OPEN",
    "BLINK_STATE_OPENING",
    "BLINK_TOTAL_S",
    "EyeState",
    "EyeSystem",
    "blink_state_from_remaining",
    "_lid_close_amount",
]
