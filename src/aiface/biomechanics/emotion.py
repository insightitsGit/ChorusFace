"""Continuous emotion state that emits muscle impulses — never animation clips."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from aiface.biomechanics.muscles import MuscleImpulse


EMOTION_AXES: tuple[str, ...] = (
    "valence",
    "arousal",
    "confidence",
    "curiosity",
    "surprise",
    "stress",
    "relaxation",
    "embarrassment",
    "thinking",
)

# Axis → (muscle group, strength weight) while the axis reads positive.
EMOTION_MUSCLE_MAP: dict[str, tuple[tuple[str, float], ...]] = {
    "valence": (
        ("ZygomaticusMajor", 0.75),
        ("ZygomaticusMinor", 0.40),
        ("Risorius", 0.45),
        ("LevatorAnguliOris", 0.35),
        ("OrbicularisOris", 0.15),
    ),
    "arousal": (
        ("Frontalis", 0.35),
        ("LevatorPalpebrae", 0.30),
        ("LevatorLabii", 0.25),
        ("Masseter", 0.15),
    ),
    "confidence": (
        ("ZygomaticusMajor", 0.25),
        ("Mentalis", -0.15),
        ("Corrugator", -0.20),
    ),
    "curiosity": (
        ("Frontalis", 0.40),
        ("LevatorPalpebrae", 0.30),
        ("OrbicularisOculi", -0.10),
    ),
    "surprise": (
        ("Frontalis", 1.0),
        ("LevatorPalpebrae", 0.95),
        ("JawOpener", 0.35),
        ("OrbicularisOris", 0.25),
    ),
    "stress": (
        ("Corrugator", 0.70),
        ("Masseter", 0.35),
        ("DepressorAnguliOris", 0.30),
        ("Mentalis", 0.25),
        ("Platysma", 0.20),
    ),
    "relaxation": (
        ("OrbicularisOris", -0.20),
        ("Masseter", -0.25),
        ("Corrugator", -0.30),
        ("Platysma", -0.15),
    ),
    "embarrassment": (
        ("ZygomaticusMajor", 0.20),
        ("Risorius", 0.15),
        ("OrbicularisOculi", 0.35),
        ("Mentalis", 0.20),
    ),
    "thinking": (
        ("Corrugator", 0.45),
        ("Procerus", 0.15),
        ("OrbicularisOris", 0.15),
        ("Mentalis", 0.20),
    ),
}

# The mirror lobe, used while an axis reads negative. Without it a sad or
# unconfident face has nothing to contract — every weight above clamps to zero
# — and sadness renders identically to neutral.
EMOTION_MUSCLE_MAP_NEGATIVE: dict[str, tuple[tuple[str, float], ...]] = {
    "valence": (
        ("DepressorAnguliOris", 0.70),
        ("Corrugator", 0.35),
        ("DepressorLabii", 0.25),
        ("Mentalis", 0.20),
    ),
    "arousal": (
        ("OrbicularisOculi", 0.25),
        ("DepressorAnguliOris", 0.20),
    ),
    "confidence": (
        ("Mentalis", 0.35),
        ("DepressorAnguliOris", 0.30),
        ("Corrugator", 0.25),
    ),
    "relaxation": (
        ("Corrugator", 0.30),
        ("Masseter", 0.30),
        ("Platysma", 0.25),
    ),
}


@dataclass(slots=True)
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0
    confidence: float = 0.0
    curiosity: float = 0.0
    surprise: float = 0.0
    stress: float = 0.0
    relaxation: float = 0.0
    embarrassment: float = 0.0
    thinking: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in EMOTION_AXES}

    def clamp(self) -> None:
        for name in EMOTION_AXES:
            setattr(self, name, max(-1.0, min(1.0, float(getattr(self, name)))))

    def blend_toward(self, target: Mapping[str, float], amount: float) -> None:
        amount = max(0.0, min(1.0, amount))
        for name in EMOTION_AXES:
            current = float(getattr(self, name))
            goal = float(target.get(name, current))
            setattr(self, name, current + (goal - current) * amount)
        self.clamp()

    def dominant(self) -> tuple[str, float]:
        items = sorted(self.as_dict().items(), key=lambda item: (-abs(item[1]), item[0]))
        return items[0]


@dataclass(slots=True)
class EmotionSystem:
    """Maps continuous emotion axes into blended muscle impulses."""

    state: EmotionState = field(default_factory=EmotionState)
    settle_rate: float = 0.35

    def apply_patch(self, patch: Mapping[str, float], *, amount: float = 1.0) -> None:
        self.state.blend_toward(patch, amount)

    def from_label(self, label: str) -> None:
        """Compatibility bridge from the existing discrete emotion tags."""
        presets = {
            "NEUTRAL": {},
            "HAPPY": {
                "valence": 0.85,
                "arousal": 0.55,
                "curiosity": 0.30,
                "confidence": 0.4,
            },
            "SAD": {"valence": -0.7, "arousal": -0.2, "relaxation": -0.1},
            "SURPRISED": {
                "surprise": 1.0,
                "arousal": 0.85,
                "curiosity": 0.45,
            },
            "SURPRISE": {
                "surprise": 1.0,
                "arousal": 0.85,
                "curiosity": 0.45,
            },
            "ANGRY": {"valence": -0.55, "stress": 0.8, "arousal": 0.55},
            "THINKING": {"thinking": 0.75, "curiosity": 0.35, "arousal": -0.1},
        }
        self.apply_patch(presets.get(label.upper(), {}), amount=0.95)

    def step(self, dt: float) -> None:
        # Soft return toward neutral so emotions do not permanently latch.
        decay = max(0.0, 1.0 - self.settle_rate * dt)
        for name in EMOTION_AXES:
            setattr(self.state, name, float(getattr(self.state, name)) * decay)

    def impulses(self, tick: int, *, duration: float = 0.12) -> list[MuscleImpulse]:
        impulses: list[MuscleImpulse] = []
        for axis in EMOTION_AXES:
            value = float(getattr(self.state, axis))
            if abs(value) < 0.04:
                continue
            if value >= 0.0:
                mappings = EMOTION_MUSCLE_MAP.get(axis, ())
            else:
                mappings = EMOTION_MUSCLE_MAP_NEGATIVE.get(axis, ())
            for muscle, weight in mappings:
                strength = abs(value) * weight
                if abs(strength) < 0.03:
                    continue
                impulses.append(
                    MuscleImpulse(
                        tick=tick,
                        muscle=muscle,
                        strength=max(0.0, strength),
                        duration=duration,
                        falloff=1.2,
                        priority=2,
                        source="Emotion",
                    )
                )
        return impulses


__all__ = [
    "EMOTION_AXES",
    "EMOTION_MUSCLE_MAP",
    "EMOTION_MUSCLE_MAP_NEGATIVE",
    "EmotionState",
    "EmotionSystem",
]
