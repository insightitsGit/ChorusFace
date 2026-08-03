"""Abstract packet schemas between TickFeed ML layers.

Phase-1: versioned dataclasses (``schema`` field). Wire/JSON exporters should
include ``schema`` so layers can reject unknown versions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

SPEECH_CLOCK_SCHEMA: Final = "aiface.packet.SpeechClock.v1"
LOOK_DRIVE_SCHEMA: Final = "aiface.packet.LookDrive.v1"
FACE_MOTION_CODE_SCHEMA: Final = "aiface.packet.FaceMotionCode.v1"


@dataclass(slots=True)
class SpeechClock:
    tick: int
    viseme_id: int
    word: str
    conf: float
    audio_feat: list[float]
    schema: str = SPEECH_CLOCK_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LookDrive:
    tick: int
    smile: float
    open: float
    surprise: float
    brow: float
    conf: float
    schema: str = LOOK_DRIVE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FaceMotionCode:
    """Compact face motion (L3/L4), not raw RGB."""

    tick: int
    code: list[float]
    conf: float
    schema: str = FACE_MOTION_CODE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "FACE_MOTION_CODE_SCHEMA",
    "LOOK_DRIVE_SCHEMA",
    "SPEECH_CLOCK_SCHEMA",
    "FaceMotionCode",
    "LookDrive",
    "SpeechClock",
]
