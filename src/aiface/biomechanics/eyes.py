"""Deterministic eye simulation: gaze, blinks, microsaccades, pupil."""

from __future__ import annotations

from dataclasses import dataclass, field

from aiface.biomechanics.muscles import MuscleImpulse


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
    blink_phase: float = 0.0
    focus_delay: float = 0.18
    lid_tension: float = 0.0


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
            self.state.blink_phase = 1.0
            self.state.blink_timer = 2.8 + self._next_unit() * 2.4

    def set_arousal(self, arousal: float) -> None:
        # Higher arousal → larger pupils and more frequent blinks.
        self.state.pupil = max(0.15, min(0.95, 0.35 + 0.4 * max(0.0, arousal)))
        interval = 3.4 - 1.2 * max(0.0, arousal)
        if self.state.blink_timer <= 0.0:
            self.state.blink_timer = interval + self._next_unit() * 1.5

    def step(self, dt: float, *, arousal: float = 0.0) -> list[MuscleImpulse]:
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
        if self.state.blink_phase > 0.0:
            # Slower close/open so blinks read as lids, not a flicker bar.
            self.state.blink_phase = max(0.0, self.state.blink_phase - dt * 4.2)
            close = min(1.0, self.state.blink_phase * 1.65)
            # Mild asymmetry: left lid leads by a few milliseconds of envelope.
            self.state.lid_left = 1.0 - close
            self.state.lid_right = 1.0 - min(1.0, close * 0.92)
            self.state.lid_tension = close
            if close > 0.2:
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
        elif self.state.blink_timer <= 0.0:
            self.state.blink_phase = 1.0
            self.state.blink_timer = 2.8 + self._next_unit() * 2.4
        else:
            self.state.lid_left = 1.0
            self.state.lid_right = 1.0
            self.state.lid_tension = 0.0

        return impulses


__all__ = ["EyeState", "EyeSystem"]
