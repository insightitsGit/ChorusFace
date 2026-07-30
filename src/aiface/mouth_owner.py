"""NWR-first mouth status — Path A ownership SEALS REMOVED.

Master Lock (ch 31) on identity is enforced by the GPU. This module never
blocks jaw, muscle warp, or field velocity on unlocked mouth cells.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

from aiface.speech import canonical_viseme

PLATE_OPEN_FLOOR: Final = 0.04
PLATE_OPEN_FULL: Final = 0.32


@dataclass(frozen=True, slots=True)
class MouthOwnership:
    muscle_warp: bool = True
    jaw: bool = True
    field_velocity: bool = True
    plate_atlas: bool = False
    plate_amount: float = 0.0
    smile_plate: bool = False
    dark_cavity: bool = False
    upper_expr_plate: bool = False
    openness: float = 0.0
    emotion: str = "NEUTRAL"
    phoneme: str = "REST"
    owners: tuple[str, ...] = ("muscle_warp", "jaw", "field_velocity")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["owners"] = list(self.owners)
        payload["policy"] = "nwr-first-no-path-a-seal"
        return payload


def plate_amount_for_openness(
    openness: float,
    *,
    floor: float = PLATE_OPEN_FLOOR,
    full: float = PLATE_OPEN_FULL,
) -> float:
    value = float(openness)
    floor = float(floor)
    if value <= floor:
        return 0.0
    span = max(float(full) - floor, 1e-6)
    return max(0.0, min(1.0, (value - floor) / span))


def resolve_mouth_ownership(
    *,
    openness: float,
    emotion: str = "NEUTRAL",
    phoneme: str = "REST",
    speaking: bool = False,
    surprise_blend: float = 0.0,
) -> MouthOwnership:
    del speaking
    mood = (emotion or "NEUTRAL").strip().upper()
    key = canonical_viseme(phoneme or "REST")
    open_n = max(0.0, min(1.0, float(openness)))
    amount = plate_amount_for_openness(open_n)
    plate_on = amount > 1e-4
    smile_on = mood == "HAPPY"
    upper_on = mood in {"SURPRISED", "SURPRISE"} or float(surprise_blend) > 0.05
    owners = ["muscle_warp", "jaw", "field_velocity"]
    if plate_on:
        owners.append("plate_atlas")
    if smile_on:
        owners.append("smile_plate")
    if upper_on:
        owners.append("upper_expr_plate")
    return MouthOwnership(
        muscle_warp=True,
        jaw=True,
        field_velocity=True,
        plate_atlas=plate_on,
        plate_amount=amount if plate_on else 0.0,
        smile_plate=smile_on,
        dark_cavity=False,
        upper_expr_plate=upper_on,
        openness=open_n,
        emotion=mood,
        phoneme=key,
        owners=tuple(owners),
    )


CLOSED_VISEMES: Final[frozenset[str]] = frozenset()

__all__ = [
    "CLOSED_VISEMES",
    "MouthOwnership",
    "PLATE_OPEN_FLOOR",
    "PLATE_OPEN_FULL",
    "plate_amount_for_openness",
    "resolve_mouth_ownership",
]
