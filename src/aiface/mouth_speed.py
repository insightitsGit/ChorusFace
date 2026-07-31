"""User-selectable mouth motion speed + realtime hold scale (0..1)."""

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
    # 0..1 slider position that drives GPU layer dwell / bridge.
    hold_scale: float


# Realtime "Mouth hold" slider → layer timeline seconds.
HOLD_SCALE_DEFAULT: Final = 0.45
_DWELL_LO: Final = 0.04
_DWELL_HI: Final = 0.42
_BRIDGE_LO: Final = 0.08
_BRIDGE_HI: Final = 0.42
_MUSCLE_LO: Final = 0.16
_MUSCLE_HI: Final = 0.58


def clamp_hold_scale(scale: float) -> float:
    return max(0.0, min(1.0, float(scale)))


def hold_scale_to_params(scale: float) -> tuple[float, float, float]:
    """Map slider 0..1 → (layer_dwell_s, bridge_gap_s, muscle_hold_s)."""
    s = clamp_hold_scale(scale)
    dwell = _DWELL_LO + s * (_DWELL_HI - _DWELL_LO)
    bridge = _BRIDGE_LO + s * (_BRIDGE_HI - _BRIDGE_LO)
    muscle = _MUSCLE_LO + s * (_MUSCLE_HI - _MUSCLE_LO)
    return dwell, bridge, muscle


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
        hold_scale=0.78,
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
        hold_scale=HOLD_SCALE_DEFAULT,
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
        hold_scale=0.18,
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
    "HOLD_SCALE_DEFAULT",
    "MOUTH_SPEED_PRESETS",
    "MouthSpeedPreset",
    "clamp_hold_scale",
    "hold_scale_to_params",
    "next_preset_key",
    "preset_by_key",
]
