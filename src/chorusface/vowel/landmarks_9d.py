"""MediaPipe landmarks → F9 9D controls (eyes + brows + mouth).

C[0] ``eye_aperture``: 0 = open, 1 = closed (matches ``blinks.apply_blinks``).
C[1] gaze/blink cue (reserved; blink overlay fills at compose time).
C[2]/C[3] brow raise / knit from eyebrow geometry.
C[4..8] mouth / lips / teeth / jaw from lip geometry.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chorusface.vowel.priors import clamp_9d, prior_9d
from chorusface.vowel.schema import GROUP_DIM

# Face Mesh topology (tasks + classic).
_LEFT_EYE_UPPER = 159
_LEFT_EYE_LOWER = 145
_RIGHT_EYE_UPPER = 386
_RIGHT_EYE_LOWER = 374
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_OUTER = 263
_RIGHT_EYE_INNER = 362
_LEFT_BROW = (70, 63, 105, 66, 107)
_RIGHT_BROW = (300, 293, 334, 296, 336)
_LEFT_BROW_INNER = 107
_RIGHT_BROW_INNER = 336
_LEFT_EYE_CENTER = (33, 133, 159, 145)
_RIGHT_EYE_CENTER = (263, 362, 386, 374)


def _dist(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    return float(np.linalg.norm(a - b))


def ear_from_landmarks(lm: NDArray[np.floating]) -> float:
    """Eye aspect ratio — higher = more open."""
    def ear(u: int, d: int, o: int, i: int) -> float:
        return _dist(lm[u, :2], lm[d, :2]) / max(_dist(lm[o, :2], lm[i, :2]), 1e-6)

    return 0.5 * (
        ear(_LEFT_EYE_UPPER, _LEFT_EYE_LOWER, _LEFT_EYE_OUTER, _LEFT_EYE_INNER)
        + ear(_RIGHT_EYE_UPPER, _RIGHT_EYE_LOWER, _RIGHT_EYE_OUTER, _RIGHT_EYE_INNER)
    )


def brow_metrics(lm: NDArray[np.floating]) -> tuple[float, float]:
    """Return (raise_raw, knit_raw) in face-width units."""
    fw = max(float(lm[:, 0].max() - lm[:, 0].min()), 1e-3)

    def center(idxs: tuple[int, ...]) -> NDArray[np.float64]:
        return np.mean(lm[list(idxs), :2], axis=0)

    left_brow = center(_LEFT_BROW)
    right_brow = center(_RIGHT_BROW)
    left_eye = center(_LEFT_EYE_CENTER)
    right_eye = center(_RIGHT_EYE_CENTER)
    # Image y-down: smaller y = higher on face. Raise = brow above eye.
    raise_l = float(left_eye[1] - left_brow[1]) / fw
    raise_r = float(right_eye[1] - right_brow[1]) / fw
    raise_raw = 0.5 * (raise_l + raise_r)
    knit_raw = _dist(lm[_LEFT_BROW_INNER, :2], lm[_RIGHT_BROW_INNER, :2]) / fw
    return raise_raw, knit_raw


def series_normalize_eyes_brows(
    ears: NDArray[np.floating],
    raises: NDArray[np.floating],
    knits: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Normalize a clip to eye_aperture / brow_raise / brow_knit in [0,1]."""
    ear = np.asarray(ears, dtype=np.float64)
    br = np.asarray(raises, dtype=np.float64)
    bk = np.asarray(knits, dtype=np.float64)
    # Open baseline high EAR; closed floor low EAR → aperture 1=closed.
    open_ref = float(np.percentile(ear, 90))
    closed_ref = float(np.percentile(ear, 5))
    span = max(open_ref - closed_ref, 1e-4)
    lid_open = np.clip((ear - closed_ref) / span, 0.0, 1.0)
    aperture = 1.0 - lid_open
    # Re-center: typical open frames → ~0 so Model A does not learn half-shut rest.
    # Blinks / squints (high percentile) still map toward 1.
    open_floor = float(np.percentile(aperture, 30))
    peak = float(np.percentile(aperture, 99))
    aperture = np.clip(
        (aperture - open_floor) / max(peak - open_floor, 1e-3), 0.0, 1.0
    )
    # Raise: high raw = brows up.
    r0, r1 = float(np.percentile(br, 10)), float(np.percentile(br, 90))
    raise_n = np.clip((br - r0) / max(r1 - r0, 1e-4), 0.0, 1.0)
    # Knit: smaller gap = more knit.
    k0, k1 = float(np.percentile(bk, 10)), float(np.percentile(bk, 90))
    knit_n = np.clip((k1 - bk) / max(k1 - k0, 1e-4), 0.0, 1.0)
    return aperture, raise_n, knit_n


def landmarks_series_to_9d(
    lm60: NDArray[np.floating],
    emotion: str,
) -> NDArray[np.float64]:
    """Full clip (T,478,3) → (T,9) measured controls (mouth + eyes + brows)."""
    lm60 = np.asarray(lm60, dtype=np.float64)
    t = int(lm60.shape[0])
    ears = np.zeros(t, dtype=np.float64)
    raises = np.zeros(t, dtype=np.float64)
    knits = np.zeros(t, dtype=np.float64)
    for i in range(t):
        ears[i] = ear_from_landmarks(lm60[i])
        raises[i], knits[i] = brow_metrics(lm60[i])
    aperture, raise_n, knit_n = series_normalize_eyes_brows(ears, raises, knits)

    left = lm60[:, 61, :2]
    right = lm60[:, 291, :2]
    upper = lm60[:, 13, :2]
    lower = lm60[:, 14, :2]
    fw = np.maximum(lm60[:, :, 0].max(axis=1) - lm60[:, :, 0].min(axis=1), 1e-3)
    width = np.linalg.norm(right - left, axis=1) / fw
    mouth_ap = np.linalg.norm(upper - lower, axis=1) / fw

    out = np.zeros((t, GROUP_DIM), dtype=np.float64)
    for i in range(t):
        base = prior_9d("AX", emotion)
        c = base.copy()
        c[0] = float(aperture[i])
        c[1] = 0.0
        c[2] = float(raise_n[i])
        c[3] = float(knit_n[i])
        spread = float(np.clip((width[i] - 0.25) / 0.25, -1.0, 1.0))
        round_ = float(np.clip((0.35 - width[i]) / 0.2, 0.0, 1.0))
        jaw = float(np.clip(mouth_ap[i] / 0.12, 0.0, 1.0))
        mouth = float(np.clip(mouth_ap[i] / 0.10, 0.0, 1.0))
        teeth = float(np.clip(mouth_ap[i] / 0.08, 0.0, 1.0))
        c[4] = mouth
        c[5] = spread
        c[6] = round_
        c[7] = teeth
        c[8] = jaw
        out[i] = clamp_9d(c)
    # Light temporal smooth (keep blink dips).
    if t >= 3:
        pad = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = 0.25 * pad[:-2] + 0.50 * pad[1:-1] + 0.25 * pad[2:]
        out = np.stack([clamp_9d(row) for row in out])
    return out


def frame_to_9d(lm: NDArray[np.floating], emotion: str, *, clip_stats: dict | None = None) -> NDArray[np.float64]:
    """Single frame → 9D. Prefer ``landmarks_series_to_9d`` for clip-normalized eyes."""
    series = landmarks_series_to_9d(np.asarray(lm, dtype=np.float64)[None, ...], emotion)
    return series[0]


__all__ = [
    "brow_metrics",
    "ear_from_landmarks",
    "frame_to_9d",
    "landmarks_series_to_9d",
    "series_normalize_eyes_brows",
]
