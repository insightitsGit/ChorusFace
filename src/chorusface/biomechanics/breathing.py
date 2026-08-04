"""Slow deterministic breathing oscillations that continue while silent."""

from __future__ import annotations

import math
from dataclasses import dataclass

from chorusface.biomechanics.muscles import MuscleImpulse


@dataclass(slots=True)
class BreathingSystem:
    phase: float = 0.0
    rate_hz: float = 0.22
    depth: float = 0.18

    def step(self, dt: float) -> list[MuscleImpulse]:
        self.phase = (self.phase + dt * self.rate_hz) % 1.0
        wave = math.sin(self.phase * math.tau)
        inhale = max(0.0, wave)
        exhale = max(0.0, -wave)
        strength = self.depth * (0.55 + 0.45 * abs(wave))
        impulses = [
            MuscleImpulse(
                tick=0,
                muscle="Platysma",
                strength=strength * (0.35 + 0.65 * inhale),
                duration=0.08,
                falloff=1.0,
                priority=1,
                source="Breathing",
            ),
            MuscleImpulse(
                tick=0,
                # Exhaling parts the jaw slightly; the closers stay quiet.
                muscle="JawOpener",
                strength=strength * 0.12 * exhale,
                duration=0.08,
                falloff=1.0,
                priority=1,
                source="Breathing",
            ),
            MuscleImpulse(
                tick=0,
                muscle="Buccinator",
                strength=strength * 0.18 * inhale,
                duration=0.08,
                falloff=1.0,
                priority=1,
                source="Breathing",
            ),
        ]
        return impulses


__all__ = ["BreathingSystem"]
