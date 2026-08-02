"""Abstract packet schemas between TickFeed ML layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SpeechClock:
    tick: int
    viseme_id: int
    word: str
    conf: float
    audio_feat: list[float]

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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FaceMotionCode:
    """Compact face motion (L3/L4), not raw RGB."""

    tick: int
    code: list[float]
    conf: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["FaceMotionCode", "LookDrive", "SpeechClock"]
