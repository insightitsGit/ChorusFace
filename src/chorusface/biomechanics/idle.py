"""Deterministic idle micro-behaviours with no obvious loop."""

from __future__ import annotations

from dataclasses import dataclass

from chorusface.biomechanics.eyes import EyeSystem
from chorusface.biomechanics.muscles import MuscleImpulse


@dataclass(slots=True)
class IdleSystem:
    seed: int = 17
    _rng: int = 17
    _next_event: float = 0.6
    _clock: float = 0.0
    eyes: EyeSystem | None = None

    def __post_init__(self) -> None:
        self._rng = int(self.seed) & 0x7FFFFFFF or 1
        if self.eyes is None:
            self.eyes = EyeSystem(seed=self.seed + 91)

    def _unit(self) -> float:
        self._rng = (1664525 * self._rng + 1013904223) & 0xFFFFFFFF
        return (self._rng & 0x7FFFFFFF) / 0x7FFFFFFF

    def step(self, dt: float, *, speaking: bool) -> list[MuscleImpulse]:
        self._clock += dt
        impulses: list[MuscleImpulse] = []
        if speaking:
            self._next_event = max(self._next_event, self._clock + 0.8)
            return impulses
        if self._clock < self._next_event:
            return impulses

        pick = self._unit()
        if pick < 0.35:
            # Tiny gaze drift.
            assert self.eyes is not None
            self.eyes.look_at(self._unit() * 0.35 - 0.175, self._unit() * 0.2 - 0.1)
        elif pick < 0.55:
            impulses.append(
                MuscleImpulse(
                    tick=0,
                    muscle="Corrugator",
                    strength=0.08 + self._unit() * 0.12,
                    duration=0.25 + self._unit() * 0.2,
                    falloff=1.4,
                    priority=1,
                    source="Idle",
                )
            )
        elif pick < 0.75:
            impulses.append(
                MuscleImpulse(
                    tick=0,
                    muscle="OrbicularisOris",
                    strength=0.05 + self._unit() * 0.08,
                    duration=0.18,
                    falloff=1.2,
                    priority=1,
                    source="Idle",
                )
            )
        elif pick < 0.9:
            # Jaw settling — the mandible drops a hair and recovers.
            impulses.append(
                MuscleImpulse(
                    tick=0,
                    muscle="JawOpener",
                    strength=0.04 + self._unit() * 0.06,
                    duration=0.2,
                    falloff=1.5,
                    priority=1,
                    source="Idle",
                )
            )
        else:
            assert self.eyes is not None
            self.eyes.request_blink()

        # Irregular interval: 0.4s .. 2.8s, no fixed period.
        self._next_event = self._clock + 0.4 + self._unit() * 2.4
        return impulses


__all__ = ["IdleSystem"]
