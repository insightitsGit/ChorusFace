"""Synthesize full-face velocity patches from speech/look labels (live TickFeed).

Until Side B dense timelines are loaded, this paints measured-style vx/vy onto
the face box from openness / smile / viseme — not per-cell PaintCommands.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aiface.tickfeed.package import FaceBox, TickLabels, build_keyframe, build_delta
from aiface.tickfeed.schema import BeatId, EmotionId, ValueDtype


def synthesize_velocity(
    face: FaceBox,
    *,
    open_amt: float,
    smile_amt: float,
    surprise_amt: float = 0.0,
    mouth_uv: tuple[float, float] | None = None,
) -> NDArray[np.float32]:
    """Return (H,W,2) velocity for the face patch."""
    h, w = int(face.h), int(face.w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Mouth center in face-local coords (default lower-mid face).
    if mouth_uv is None:
        mx, my = w * 0.50, h * 0.62
    else:
        # mouth_uv in grid space → local
        mx = float(mouth_uv[0]) - float(face.x)
        my = float(mouth_uv[1]) - float(face.y)
    dx = (xx - mx) / max(w * 0.35, 1.0)
    dy = (yy - my) / max(h * 0.28, 1.0)
    r2 = dx * dx + dy * dy
    mouth = np.exp(-r2 * 2.2).astype(np.float32)
    # Lower face weight for jaw/open
    lower = np.clip((yy - my) / max(h * 0.25, 1.0), 0.0, 1.0)
    upper = np.clip((my - yy) / max(h * 0.35, 1.0), 0.0, 1.0)

    open_n = float(np.clip(open_amt, 0.0, 1.0))
    smile_n = float(np.clip(smile_amt, 0.0, 1.0))
    sur_n = float(np.clip(surprise_amt, 0.0, 1.0))

    vx = smile_n * 0.55 * dx * mouth
    # Rest-relative open: upper lip up (+y), lower lip down (−y). A lower-only
    # bias made live speech look like the jaw alone was sliding.
    # Stronger open so live TTS FIELD is readable at recipe field_warp_gain.
    # Peak ~1.2 at open=1 so GPU clamp (±1.5) still has headroom.
    vy = (
        -open_n * 2.10 * mouth * (0.12 + 0.88 * lower)
        + open_n * 1.80 * mouth * upper
        + sur_n * 0.55 * upper * np.exp(-((xx - mx) / max(w * 0.4, 1.0)) ** 2)
    )
    out = np.stack([vx, vy], axis=-1).astype(np.float32)
    return out


def labels_from_drives(
    *,
    phoneme: str,
    smile_amt: float,
    open_amt: float,
    surprise_amt: float = 0.0,
    emotion: str = "NEUTRAL",
    word: str = "",
    brow_amt: float = 0.0,
) -> TickLabels:
    emo = (emotion or "NEUTRAL").strip().upper()
    emotion_id = {
        "NEUTRAL": EmotionId.NEUTRAL,
        "HAPPY": EmotionId.HAPPY,
        "JOY": EmotionId.HAPPY,
        "SURPRISED": EmotionId.SURPRISED,
        "SURPRISE": EmotionId.SURPRISED,
        "ANGRY": EmotionId.ANGRY,
        "SAD": EmotionId.SAD,
        "THINKING": EmotionId.THINKING,
    }.get(emo, EmotionId.NEUTRAL)
    beat = BeatId.REST
    if smile_amt > 0.45 and open_amt < 0.2:
        beat = BeatId.SMILE
    elif open_amt > 0.55:
        beat = BeatId.OPEN
    elif surprise_amt > 0.4:
        beat = BeatId.SURPRISE
    elif emo == "ANGRY":
        beat = BeatId.ANGRY
    elif (phoneme or "REST").upper() not in {"REST", "CLOSED", "SIL"}:
        beat = BeatId.TALK
    return TickLabels(
        beat_id=int(beat),
        emotion_id=int(emotion_id),
        viseme_id=TickLabels.viseme_index(phoneme),
        smile_amt=float(smile_amt),
        open_amt=float(open_amt),
        surprise_amt=float(surprise_amt),
        word=(word or "")[:16],
        brow_amt=float(np.clip(brow_amt, 0.0, 1.0)),
    )


__all__ = ["labels_from_drives", "synthesize_velocity"]
