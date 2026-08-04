"""TickPackage v1 constants — full-face KEY/DELTA handshake.

See docs/TickPackageHandshake.md and docs/TickFeedDesign.md.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

TICK_RATE_HZ: Final = 60
TICK_DT: Final = 1.0 / TICK_RATE_HZ

# Header
MAGIC: Final = 0x31504B54  # 'TPK1' little-endian bytes T P K 1
VERSION: Final = 1
HEADER_BYTES: Final = 64
LABELS_BYTES: Final = 48

CHANNEL_MASK_VELOCITY: Final = 0x3  # ch0 vx, ch1 vy
PHASE1_CHANNELS: Final = 2

# Sparse omit epsilon (grid velocity units)
DELTA_EPS: Final = 1e-4

# Ring / damp (bridge B3)
RING_DEPTH: Final = 3
VELOCITY_MISS_DAMP: Final = 0.85
KEY_REFRESH_TICKS: Final = 120  # ~2 s


class PackageKind(IntEnum):
    HELLO = 3
    KEYFRAME = 1
    DELTA = 2


class ValueDtype(IntEnum):
    F32 = 1
    F16 = 2


class DeltaEncoding(IntEnum):
    NONE = 0  # keyframe
    DENSE_DELTA = 1
    SPARSE_DELTA = 2
    EMPTY = 3


class BeatId(IntEnum):
    REST = 0
    SMILE = 1
    OPEN = 2
    SAY_HI = 3
    SURPRISE = 4
    ANGRY = 5
    TALK = 6
    TONGUE_TH = 7
    BLINK = 8
    UNKNOWN = 255


class EmotionId(IntEnum):
    NEUTRAL = 0
    HAPPY = 1
    SURPRISED = 2
    ANGRY = 3
    SAD = 4
    THINKING = 5
    UNKNOWN = 255


# Canonical viseme order for viseme_id (index into this table).
VISEME_TABLE: Final[tuple[str, ...]] = (
    "REST",
    "CLOSED",
    "PP",
    "FF",
    "TH",
    "DD",
    "KK",
    "CH",
    "SS",
    "NN",
    "RR",
    "AA",
    "AH",
    "EH",
    "IH",
    "EE",
    "OH",
    "OU",
)


FLAG_HAS_LABELS: Final = 1 << 0
FLAG_HAS_CONF: Final = 1 << 1
# Phase-1 FIELD is rest→frame displacement (Farneback), not tick-to-tick velocity.
# Encoders set this; masters treat ch0/1 as warp vectors from rest.
FLAG_VS_REST: Final = 1 << 2

# HELLO apply_mode (≤16 chars). Matches collect rest→frame displacement.
APPLY_MODE_DISP_VS_REST: Final = "disp_vs_rest"

# Sparse → dense fallback when too many cells change
SPARSE_DENSE_THRESHOLD: Final = 0.35

__all__ = [
    "APPLY_MODE_DISP_VS_REST",
    "CHANNEL_MASK_VELOCITY",
    "DELTA_EPS",
    "FLAG_HAS_CONF",
    "FLAG_HAS_LABELS",
    "FLAG_VS_REST",
    "HEADER_BYTES",
    "KEY_REFRESH_TICKS",
    "LABELS_BYTES",
    "MAGIC",
    "PHASE1_CHANNELS",
    "RING_DEPTH",
    "SPARSE_DENSE_THRESHOLD",
    "TICK_DT",
    "TICK_RATE_HZ",
    "VERSION",
    "VELOCITY_MISS_DAMP",
    "VISEME_TABLE",
    "BeatId",
    "DeltaEncoding",
    "EmotionId",
    "PackageKind",
    "ValueDtype",
]
