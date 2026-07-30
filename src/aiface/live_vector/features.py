"""Shared audio RMS features for train + runtime (no drift)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from aiface.live_vector.schema import FEATURE_DIM, HISTORY


def rms_history_features(
    rms_history: Sequence[float],
    *,
    noise_floor: float = 0.0,
    voice_threshold: float = 0.14,
    peak_hint: float = 0.0,
) -> npt.NDArray[np.float64]:
    hist = list(rms_history)[-HISTORY:]
    if not hist:
        hist = [0.0]
    while len(hist) < HISTORY:
        hist = [hist[0]] + hist
    arr = np.asarray(hist[-HISTORY:], dtype=np.float64)
    current = float(arr[-1])
    log_rms = float(np.log1p(current * 100.0))
    mean5 = float(arr.mean())
    std5 = float(arr.std())
    d1 = float(arr[-1] - arr[-2])
    d2 = float(arr[-2] - arr[-3])
    max5 = float(arr.max())
    peak = max(float(peak_hint), max5, 1e-6)
    gate = max(voice_threshold * peak, float(noise_floor) * 1.8)
    voiced = 1.0 if current >= gate else 0.0
    feats = np.asarray(
        [current, log_rms, mean5, std5, d1, d2, max5, voiced],
        dtype=np.float64,
    )
    assert feats.shape == (FEATURE_DIM,)
    return feats


def heuristic_vector(
    rms_history: Sequence[float],
    *,
    noise_floor: float = 0.0,
    peak_hint: float = 0.0,
) -> tuple[float, float, float]:
    """Energy fallback when no model weights exist."""
    from aiface.live_vector.schema import plate_gate

    feats = rms_history_features(
        rms_history, noise_floor=noise_floor, peak_hint=peak_hint
    )
    current, _log, mean5, _std, d1, _d2, max5, voiced = feats
    if voiced < 0.5:
        return 0.0, 0.0, 0.0
    peak = max(float(peak_hint), max5, 1e-6)
    level = max(0.0, min(1.0, mean5 / (0.35 * peak + 1e-6)))
    attack = max(0.0, min(0.25, d1 / (peak + 1e-6)))
    open_n = max(
        0.0,
        min(1.0, 0.85 * level + 0.15 * (current / (peak + 1e-6)) + attack),
    )
    return open_n, open_n, max(0.0, open_n * 0.35)


def sample_envelope_rms(envelope: object, time_seconds: float) -> float:
    if envelope is None or getattr(envelope, "frame_count", 0) <= 0:
        return 0.0
    hop = float(envelope.hop)
    index = int(max(0.0, float(time_seconds)) / max(hop, 1e-6))
    index = min(index, int(envelope.frame_count) - 1)
    return float(envelope.values[index])
