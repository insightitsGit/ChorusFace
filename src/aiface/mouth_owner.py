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
# Lip-tighten / bilabial close — must kill open-plate hold immediately.
CLOSED_VISEMES: Final[frozenset[str]] = frozenset({"PP", "MM", "CLOSED"})


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
    hard_snap: bool = False,
) -> float:
    value = float(openness)
    floor = float(floor)
    if value <= floor:
        return 0.0
    span = max(float(full) - floor, 1e-6)
    linear = max(0.0, min(1.0, (value - floor) / span))
    if hard_snap:
        # Mid amounts are a soft veil over the rest photo — commit 0 or 1.
        return 1.0 if linear >= 0.45 else 0.0
    return linear


def mute_smile_under_open(
    smile: float,
    open_level: float,
    *,
    start: float = 0.12,
    full: float = 0.32,
) -> float:
    """Fade smile.png out as the mouth opens so HAPPY floor cannot veil speech."""
    smile_n = max(0.0, min(1.0, float(smile)))
    open_n = max(0.0, float(open_level))
    if open_n <= start:
        return smile_n
    mute = min(1.0, (open_n - float(start)) / max(float(full) - float(start), 1e-6))
    return smile_n * (1.0 - mute)


def snap_smile_drive(smile: float, *, hard_snap: bool = False) -> float:
    """Commit smile to 0 or 1 under hard snap — mid 0.55 parks are a soft veil."""
    smile_n = max(0.0, min(1.0, float(smile)))
    if not hard_snap:
        return smile_n
    if smile_n <= 0.0:
        return 0.0
    return 1.0 if smile_n >= 0.5 else 0.0


def hold_speech_viseme(
    current: str,
    held: str,
    *,
    open_n: float,
    jaw_n: float,
    open_hold: float = 0.15,
    jaw_hold: float = 0.10,
) -> tuple[str, str]:
    """Keep the last speaking viseme while jaw/open are still elevated.

    Returns ``(viseme_for_plates, new_held)``. Without this, REST snaps the
    atlas to a closed plate over an still-open cavity (ghost smear).

    Tight-lip visemes (PP/MM/CLOSED) never inherit an open hold — that left
    open.png / OH atlas parked over closing lips ("blurry layer won't leave").
    """
    key = canonical_viseme(current or "REST")
    if key in CLOSED_VISEMES:
        return key, key
    if key != "REST":
        return key, key
    if float(open_n) > float(open_hold) or float(jaw_n) > float(jaw_hold):
        held_key = canonical_viseme(held or "REST")
        if held_key not in CLOSED_VISEMES and held_key not in {"REST", ""}:
            return held_key, held_key
    return key, "REST"


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


__all__ = [
    "CLOSED_VISEMES",
    "MouthOwnership",
    "PLATE_OPEN_FLOOR",
    "PLATE_OPEN_FULL",
    "hold_speech_viseme",
    "mute_smile_under_open",
    "plate_amount_for_openness",
    "resolve_mouth_ownership",
    "snap_smile_drive",
]
