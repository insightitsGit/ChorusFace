"""User-selectable mouth motion speed (60 Hz easing / hold presets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class MouthSpeedPreset:
    """Rates applied each 60 Hz frame and to viseme holds."""

    key: str
    label: str
    # Attack ease (open toward a speech shape). Higher = snappier.
    plate_ease: float
    mouth_ease: float
    # Release scale: fall back to REST this much slower than attack.
    release_scale: float
    # Minimum viseme dwell seconds (open vowels get an extra boost).
    hold_seconds: float
    # Jaw spring-mass tuning.
    jaw_mass: float
    jaw_damping: float
    jaw_elasticity: float


MOUTH_SPEED_PRESETS: Final[tuple[MouthSpeedPreset, ...]] = (
    MouthSpeedPreset(
        key="slow",
        label="Slow",
        plate_ease=3.8,
        mouth_ease=4.0,
        release_scale=0.24,
        hold_seconds=0.50,
        jaw_mass=1.45,
        jaw_damping=6.8,
        jaw_elasticity=16.0,
    ),
    MouthSpeedPreset(
        key="normal",
        label="Normal",
        # Snappier attack so PP vs AA reads; hold long enough to see the shape.
        plate_ease=5.8,
        mouth_ease=6.0,
        release_scale=0.34,
        hold_seconds=0.42,
        jaw_mass=1.20,
        jaw_damping=6.0,
        jaw_elasticity=24.0,
    ),
    MouthSpeedPreset(
        key="fast",
        label="Fast",
        plate_ease=9.5,
        mouth_ease=11.5,
        release_scale=0.48,
        hold_seconds=0.20,
        jaw_mass=0.88,
        jaw_damping=4.8,
        jaw_elasticity=36.0,
    ),
)

#: Normal is the readable default; Slow is available when plates need more dwell.
DEFAULT_MOUTH_SPEED: Final = "normal"


def preset_by_key(key: str) -> MouthSpeedPreset:
    needle = (key or DEFAULT_MOUTH_SPEED).strip().lower()
    for preset in MOUTH_SPEED_PRESETS:
        if preset.key == needle:
            return preset
    return MOUTH_SPEED_PRESETS[1]


def next_preset_key(key: str) -> str:
    keys = [preset.key for preset in MOUTH_SPEED_PRESETS]
    try:
        index = keys.index(preset_by_key(key).key)
    except ValueError:
        index = 1
    return keys[(index + 1) % len(keys)]


__all__ = [
    "DEFAULT_MOUTH_SPEED",
    "MOUTH_SPEED_PRESETS",
    "MouthSpeedPreset",
    "next_preset_key",
    "preset_by_key",
]
