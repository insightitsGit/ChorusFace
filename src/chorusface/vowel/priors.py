"""§5.2 pose priors mapped into frozen 9D control space (F9)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chorusface.vowel.schema import EMOTIONS, GA16, GROUP_DIM

# Tag → (mouth, lip_amount, lip_is_round, teeth, jaws, brow_hint)
# lip_amount is magnitude; lip_is_round selects round vs signed spread.
_PRIOR_6: dict[str, tuple[float, float, bool, float, float, float]] = {
    "EE": (0.2, 0.9, False, 0.8, 0.2, 0.1),
    "IH": (0.3, 0.5, False, 0.5, 0.3, 0.0),
    "EY": (0.4, 0.7, False, 0.6, 0.4, 0.1),
    "EH": (0.5, 0.4, False, 0.5, 0.5, 0.0),
    "AE": (0.7, 0.8, False, 0.8, 0.7, 0.1),
    "AA": (0.9, 0.2, False, 0.4, 0.9, 0.0),
    "AO": (0.8, 0.6, True, 0.2, 0.8, 0.0),
    "OH": (0.6, 0.8, True, 0.1, 0.6, 0.0),
    "UH": (0.3, 0.5, True, 0.1, 0.3, 0.0),
    "OU": (0.2, 1.0, True, 0.0, 0.2, 0.0),
    "AH": (0.6, 0.3, False, 0.3, 0.6, 0.0),
    "AX": (0.2, 0.1, False, 0.1, 0.2, 0.0),
    "ER": (0.3, 0.4, True, 0.2, 0.3, 0.0),
    "AY": (0.8, 0.6, False, 0.7, 0.8, 0.1),
    "AW": (0.8, 0.7, True, 0.5, 0.8, 0.1),
    "OY": (0.7, 0.8, True, 0.5, 0.7, 0.0),
}

# Emotion upper-face / mouth-corner bias applied on top of vowel priors.
_EMOTION_BIAS: dict[str, NDArray[np.float64]] = {
    # eye_aperture, gaze, brow_raise, brow_knit, mouth, spread, round, teeth, jaw
    "NEUTRAL": np.zeros(GROUP_DIM),
    "HAPPY": np.array([0.15, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.05, 0.0]),
    "SAD": np.array([0.1, 0.0, 0.0, 0.35, 0.0, -0.2, 0.0, 0.0, 0.0]),
    "SURPRISED": np.array([-0.2, 0.2, 0.7, 0.0, 0.1, 0.1, 0.0, 0.1, 0.05]),
    "ANGRY": np.array([0.35, 0.0, 0.0, 0.75, 0.0, -0.15, 0.05, 0.0, 0.0]),
    "THINKING": np.array([0.05, 0.0, 0.15, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
}


def prior_9d(tag: str, emotion: str = "NEUTRAL") -> NDArray[np.float64]:
    """Return a 9D group control prior for vowel×emotion."""
    key = (tag or "AX").strip().upper()
    if key not in _PRIOR_6:
        key = "AX"
    mouth, lip_amt, is_round, teeth, jaws, brow = _PRIOR_6[key]
    c = np.zeros(GROUP_DIM, dtype=np.float64)
    c[0] = 0.0  # eye aperture (emotion bias fills)
    c[1] = 0.0
    c[2] = brow
    c[3] = 0.0
    c[4] = mouth
    if is_round:
        c[5] = -0.35 * lip_amt  # signed spread lean toward round
        c[6] = lip_amt
    else:
        c[5] = lip_amt
        c[6] = 0.05 * lip_amt
    c[7] = teeth
    c[8] = jaws
    emo = (emotion or "NEUTRAL").strip().upper()
    if emo not in _EMOTION_BIAS:
        emo = "NEUTRAL"
    c = c + _EMOTION_BIAS[emo]
    return clamp_9d(c)


def clamp_9d(c: NDArray[np.floating]) -> NDArray[np.float64]:
    """Clamp each channel to its F9 domain."""
    out = np.asarray(c, dtype=np.float64).reshape(GROUP_DIM).copy()
    out[0] = float(np.clip(out[0], 0.0, 1.0))
    out[1] = float(np.clip(out[1], -1.0, 1.0))
    out[2] = float(np.clip(out[2], 0.0, 1.0))
    out[3] = float(np.clip(out[3], 0.0, 1.0))
    out[4] = float(np.clip(out[4], 0.0, 1.0))
    out[5] = float(np.clip(out[5], -1.0, 1.0))
    out[6] = float(np.clip(out[6], 0.0, 1.0))
    out[7] = float(np.clip(out[7], 0.0, 1.0))
    out[8] = float(np.clip(out[8], 0.0, 1.0))
    return out


def rest_9d(emotion: str = "NEUTRAL") -> NDArray[np.float64]:
    """Mouth REST under emotion — lower face ~0, upper face keeps emotion."""
    c = np.zeros(GROUP_DIM, dtype=np.float64)
    emo = (emotion or "NEUTRAL").strip().upper()
    bias = _EMOTION_BIAS.get(emo, _EMOTION_BIAS["NEUTRAL"])
    c[0] = float(np.clip(bias[0], 0.0, 1.0))
    c[1] = float(np.clip(bias[1], -1.0, 1.0))
    c[2] = float(np.clip(bias[2], 0.0, 1.0))
    c[3] = float(np.clip(bias[3], 0.0, 1.0))
    # HAPPY may keep tiny corner lift (D3 nuance) → small +spread
    if emo == "HAPPY":
        c[5] = 0.12
    elif emo == "ANGRY":
        c[5] = -0.08
        c[6] = 0.05
    return c


def all_prior_targets() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build Dataset-A-sized prior matrix: X one-hot 22D, Y 9D."""
    xs: list[NDArray[np.float64]] = []
    ys: list[NDArray[np.float64]] = []
    for vi, tag in enumerate(GA16):
        for ei, emo in enumerate(EMOTIONS):
            x = np.zeros(22, dtype=np.float64)
            x[vi] = 1.0
            x[16 + ei] = 1.0
            xs.append(x)
            ys.append(prior_9d(tag, emo))
    return np.stack(xs), np.stack(ys)
