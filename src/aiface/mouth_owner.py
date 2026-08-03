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


def look_field_gain_scale(
    *,
    mouth_state: str,
    plate_open: float,
    open_vel: float = 0.0,
    live_speech: bool = False,
) -> float:
    """§14.3 — scale recipe FIELD gain under LOOK plates (single owner).

    Returns a multiplier in ``[0, 1]`` applied to ``field_warp_gain``.
    Live chat/TTS synth FIELD + plates always smear — mute under plates.
    OPENING/CLOSING mid-band owns the oral disk with plates alone.
    """
    plate_o = max(0.0, float(plate_open))
    state = str(mouth_state or "REST").upper()
    vel = abs(float(open_vel))
    # Chat/TTS: synth Gaussian + plates → soft ghost. Plates own oral disk.
    if live_speech and plate_o >= 0.08:
        return 0.0
    if state in {"OPENING", "CLOSING"}:
        if vel > 0.6 or 0.10 <= plate_o <= 0.65:
            return 0.0
        return 0.02
    if plate_o >= 0.45 or state == "OPEN":
        return 0.02
    if plate_o >= 0.12:
        return 0.05
    return 1.0


# Soft double-exposure zone — never park plate amount here (handoff Task 1).
MID_BAND_LO: Final = 0.15
MID_BAND_HI: Final = 0.55
MID_BAND_SPLIT: Final = 0.32


def snap_midband_openness(open_amt: float) -> float:
    """Kill soft mid-band plate amounts (0.15–0.55) — binary commit only.

    Below the smear band → closed (0). Inside or above → full plate (1).
    Transitions never linger at 30–50% veil.
    """
    amount = max(0.0, min(1.0, float(open_amt)))
    if amount <= PLATE_OPEN_FLOOR:
        return 0.0
    if amount < MID_BAND_LO:
        # Tiny crack — still commit closed to avoid soft veil.
        return 0.0
    if amount <= MID_BAND_HI:
        return 0.0 if amount < MID_BAND_SPLIT else 1.0
    return 1.0


def commit_plate_amount(plate_amt: float, mouth_state: str) -> float:
    """Hard-commit LOOK plate amount — mid-band / transitions are never 50/50."""
    amount = max(0.0, min(1.0, float(plate_amt)))
    state = str(mouth_state or "REST").upper()
    if amount <= PLATE_OPEN_FLOOR:
        return 0.0
    # Transitions: full plate ownership immediately (no soft ramp through mid).
    if state in {"OPENING", "CLOSING"}:
        return 1.0
    if MID_BAND_LO <= amount <= MID_BAND_HI:
        return snap_midband_openness(amount)
    if state == "OPEN" or amount >= MID_BAND_HI:
        return 1.0
    return snap_midband_openness(amount)


def viseme_instant_openness(phoneme: str, table: dict[str, float] | None = None) -> float:
    """Viseme → LOOK openness with instant high-energy commit (no linear ramp).

    High-energy vowels jump to full open. Mid consonants snap out of the
    soft 0.15–0.55 band via :func:`snap_midband_openness`.
    """
    from aiface.plates import OPEN_TOOTH_VISEMES, VISEME_OPENNESS

    key = canonical_viseme(phoneme or "REST")
    src = table if table is not None else VISEME_OPENNESS
    raw = float(src.get(key, VISEME_OPENNESS.get(key, 0.0)))
    if key in CLOSED_VISEMES or key in {"REST", "SIL"}:
        return 0.0
    if key in OPEN_TOOTH_VISEMES:
        # Instant full open — never wait for a ramp through mid-band.
        return 1.0
    # Consonant / mid shapes: atlas-primary band but hard amount (not soft veil).
    if raw <= 0.0:
        return 0.0
    # Land just below open.png primary threshold after hard snap → atlas owns
    # at amount 1.0 (see avatar.frag step(0.55, layer_open)).
    return snap_midband_openness(max(raw, MID_BAND_SPLIT))


__all__ = [
    "CLOSED_VISEMES",
    "MID_BAND_HI",
    "MID_BAND_LO",
    "MID_BAND_SPLIT",
    "MouthOwnership",
    "PLATE_OPEN_FLOOR",
    "PLATE_OPEN_FULL",
    "commit_plate_amount",
    "hold_speech_viseme",
    "look_field_gain_scale",
    "mute_smile_under_open",
    "plate_amount_for_openness",
    "resolve_mouth_ownership",
    "snap_midband_openness",
    "snap_smile_drive",
    "viseme_instant_openness",
]
