"""Frozen VowelDesign Phase-1 constants (F1–F9 wire + control contracts)."""

from __future__ import annotations

from enum import IntEnum
from typing import Final

TICK_HZ: Final = 60
TICK_DT: Final = 1.0 / TICK_HZ
GROUP_DIM: Final = 9

# PulseChunk PLS1
PLS_MAGIC: Final = 0x31534C50  # 'PLS1'
PLS_VERSION: Final = 1
PLS_HEADER_BYTES: Final = 32
PLS_EXT_HEADER_BYTES: Final = 12
WORD_SLICE_BYTES: Final = 12
VOWEL_PAD: Final = 0xFF
MAX_VOWELS_PER_SLICE: Final = 6

FLAG_HAS_EXT_HEADER: Final = 1 << 0
FLAG_IS_SPOOLED: Final = 1 << 1
FLAG_HAS_WORD_SLICES: Final = 1 << 2

INLINE_TICK_LIMIT: Final = 300  # D22

GA16: Final[tuple[str, ...]] = (
    "EE",
    "IH",
    "EY",
    "EH",
    "AE",
    "AA",
    "AO",
    "OH",
    "UH",
    "OU",
    "AH",
    "AX",
    "ER",
    "AY",
    "AW",
    "OY",
)
GA16_INDEX: Final[dict[str, int]] = {t: i for i, t in enumerate(GA16)}

# Morphological teacher split (D6)
PART1_SPREAD_OPEN: Final[tuple[str, ...]] = (
    "EE",
    "IH",
    "EY",
    "EH",
    "AE",
    "AA",
    "AX",
    "ER",
)
PART2_ROUND_DIPH: Final[tuple[str, ...]] = (
    "AO",
    "OH",
    "UH",
    "OU",
    "AH",
    "AY",
    "AW",
    "OY",
)

EMOTIONS: Final[tuple[str, ...]] = (
    "NEUTRAL",
    "HAPPY",
    "SAD",
    "SURPRISED",
    "ANGRY",
    "THINKING",
)
EMOTION_INDEX: Final[dict[str, int]] = {e: i for i, e in enumerate(EMOTIONS)}


class EmotionId(IntEnum):
    NEUTRAL = 0
    HAPPY = 1
    SAD = 2
    SURPRISED = 3
    ANGRY = 4
    THINKING = 5


# F9 ONNX channel names (immutable index order)
CHANNEL_NAMES: Final[tuple[str, ...]] = (
    "eye_aperture",  # 0 [0,1]
    "eye_gaze_or_blink",  # 1 [-1,1]
    "brow_raise",  # 2 [0,1]
    "brow_knit",  # 3 [0,1]
    "mouth_cavity_gap",  # 4 [0,1]
    "lip_spread",  # 5 [-1,1]
    "lip_round",  # 6 [0,1]
    "teeth_visibility",  # 7 [0,1]
    "jaw_drop",  # 8 [0,1]
)

# Attack ticks prior (D15 merged)
ATTACK_TICKS: Final[dict[str, int]] = {
    "ANGRY": 4,
    "SURPRISED": 4,
    "HAPPY": 5,
    "NEUTRAL": 6,
    "THINKING": 6,
    "SAD": 9,
}
RELEASE_TICKS: Final = 6
CROSSFADE_TICKS: Final = 3
MICRO_REST_MAX_TICKS: Final = 3
CONFLICT_BRIDGE_TICKS: Final = 2
COARTIC_BLEND_TICKS: Final = 4

# Lip class for plates / coarticulation
SPREAD_VOWELS: Final[frozenset[str]] = frozenset({"EE", "IH", "EY", "EH", "AE"})
OPEN_VOWELS: Final[frozenset[str]] = frozenset({"AA", "AH", "AE", "AO"})
ROUND_VOWELS: Final[frozenset[str]] = frozenset({"OH", "OU", "UH", "AO"})
CLOSE_VOWELS: Final[frozenset[str]] = frozenset({"AX", "IH", "UH"})

# Diphthong end targets (start is the tag itself for priors; blend to end)
DIPHTHONG_ENDS: Final[dict[str, str]] = {
    "EY": "EE",
    "OH": "OU",
    "AY": "IH",
    "AW": "UH",
    "OY": "IH",
}
