"""Deterministic LOOK plate lookup (F10 / D10)."""

from __future__ import annotations

from chorusface.vowel.schema import (
    CLOSE_VOWELS,
    OPEN_VOWELS,
    ROUND_VOWELS,
    SPREAD_VOWELS,
)

LipClass = str  # spread | open | round | close

# Placeholder family names — map onto TickFeed plate assets at apply time.
_PLATE_TABLE: dict[tuple[LipClass, str], str] = {}
for emo in ("NEUTRAL", "HAPPY", "SAD", "SURPRISED", "ANGRY", "THINKING"):
    _PLATE_TABLE[("spread", emo)] = f"PLATE_SMILE_WIDE_{emo[:1]}"
    _PLATE_TABLE[("open", emo)] = f"PLATE_OPEN_MID_{emo[:1]}"
    _PLATE_TABLE[("round", emo)] = f"PLATE_ROUND_{emo[:1]}"
    _PLATE_TABLE[("close", emo)] = f"PLATE_REST_{emo[:1]}"

# Map stub names → existing viseme/plate-ish labels used by TickFeed today.
_RUNTIME_ALIAS: dict[str, str] = {
    "PLATE_SMILE_WIDE_N": "SMILE",
    "PLATE_SMILE_WIDE_H": "SMILE",
    "PLATE_SMILE_WIDE_S": "SMILE",
    "PLATE_SMILE_WIDE_A": "SMILE",
    "PLATE_SMILE_WIDE_T": "SMILE",
    "PLATE_OPEN_MID_N": "OPEN",
    "PLATE_OPEN_MID_H": "OPEN",
    "PLATE_OPEN_MID_S": "OPEN",
    "PLATE_OPEN_MID_A": "OPEN",
    "PLATE_OPEN_MID_T": "OPEN",
    "PLATE_OPEN_WIDE_S": "OPEN",
    "PLATE_ROUND_N": "OU",
    "PLATE_ROUND_H": "OU",
    "PLATE_ROUND_S": "OU",
    "PLATE_ROUND_A": "OU",
    "PLATE_ROUND_T": "OU",
    "PLATE_REST_N": "REST",
    "PLATE_REST_H": "REST",
    "PLATE_REST_S": "REST",
    "PLATE_REST_A": "REST",
    "PLATE_REST_T": "REST",
}


def lip_class(tag: str) -> LipClass:
    t = (tag or "AX").upper()
    if t in ROUND_VOWELS and t not in SPREAD_VOWELS:
        return "round"
    if t in SPREAD_VOWELS:
        return "spread"
    if t in OPEN_VOWELS:
        return "open"
    if t in CLOSE_VOWELS or t == "AX":
        return "close"
    if t in ROUND_VOWELS:
        return "round"
    return "open"


def plate_for(tag: str, emotion: str) -> str:
    emo = (emotion or "NEUTRAL").upper()
    if emo not in {"NEUTRAL", "HAPPY", "SAD", "SURPRISED", "ANGRY", "THINKING"}:
        emo = "NEUTRAL"
    stub = _PLATE_TABLE[(lip_class(tag), emo)]
    return _RUNTIME_ALIAS.get(stub, stub)


def plate_from_controls(c: list[float] | tuple[float, ...], emotion: str) -> str:
    """Threshold fallback when only 9D is known (Gemini F10 style)."""
    # C[4] mouth, C[5] spread, C[6] round, C[8] jaw
    spread = float(c[5]) if len(c) > 5 else 0.0
    round_ = float(c[6]) if len(c) > 6 else 0.0
    jaw = float(c[8]) if len(c) > 8 else 0.0
    if jaw < 0.15 and abs(spread) < 0.2 and round_ < 0.2:
        tag = "AX"
    elif round_ > 0.6 or spread < -0.35:
        tag = "OU"
    elif spread > 0.6:
        tag = "EE"
    elif jaw > 0.5:
        tag = "AA"
    else:
        tag = "AH"
    return plate_for(tag, emotion)
