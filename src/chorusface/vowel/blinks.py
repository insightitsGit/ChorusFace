"""Natural blink overlays on 9D eye channels (C[0] aperture, C[1] blink cue).

Phase-1: deterministic blink schedule on the composed trajectory. Does not
change Model A targets — blinks are a runtime eyelid event layered on top,
matching TickFeed's "blink as discrete event" guidance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chorusface.vowel.schema import GROUP_DIM, TICK_HZ

# Typical spontaneous blink: ~100–150 ms close+open; interval ~2.5–4.5 s.
BLINK_CLOSE_TICKS = 3  # ~50 ms down
BLINK_HOLD_TICKS = 2  # ~33 ms shut
BLINK_OPEN_TICKS = 4  # ~67 ms up
BLINK_TOTAL_TICKS = BLINK_CLOSE_TICKS + BLINK_HOLD_TICKS + BLINK_OPEN_TICKS
DEFAULT_INTERVAL_S = 3.2
MIN_INTERVAL_TICKS = int(2.0 * TICK_HZ)
FIRST_BLINK_EARLIEST_TICK = int(0.35 * TICK_HZ)


def blink_envelope(n: int = BLINK_TOTAL_TICKS) -> NDArray[np.float64]:
    """Return 0→1→0 eyelid-close amounts for one blink (1 = fully closed)."""
    env = np.zeros(n, dtype=np.float64)
    # close
    for i in range(BLINK_CLOSE_TICKS):
        env[i] = (i + 1) / BLINK_CLOSE_TICKS
    # hold shut
    for i in range(BLINK_HOLD_TICKS):
        env[BLINK_CLOSE_TICKS + i] = 1.0
    # open
    base = BLINK_CLOSE_TICKS + BLINK_HOLD_TICKS
    for i in range(BLINK_OPEN_TICKS):
        env[base + i] = 1.0 - (i + 1) / BLINK_OPEN_TICKS
    return env


def plan_blink_starts(
    n_ticks: int,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    seed: int | None = None,
) -> list[int]:
    """Choose blink start ticks across an utterance."""
    if n_ticks < FIRST_BLINK_EARLIEST_TICK + BLINK_TOTAL_TICKS:
        # short clip: one blink near the middle if long enough
        if n_ticks >= BLINK_TOTAL_TICKS + 4:
            return [max(1, n_ticks // 2 - BLINK_TOTAL_TICKS // 2)]
        return []

    rng = np.random.default_rng(seed)
    interval = max(MIN_INTERVAL_TICKS, int(round(interval_s * TICK_HZ)))
    starts: list[int] = []
    t = FIRST_BLINK_EARLIEST_TICK
    # slight jitter so clips don't look mechanical
    t += int(rng.integers(0, max(1, interval // 4)))
    while t + BLINK_TOTAL_TICKS < n_ticks - 2:
        starts.append(int(t))
        jitter = int(rng.integers(-interval // 6, interval // 6 + 1))
        t += interval + jitter
    # guarantee at least one blink on medium+ utterances
    if not starts and n_ticks > BLINK_TOTAL_TICKS + FIRST_BLINK_EARLIEST_TICK:
        starts.append(FIRST_BLINK_EARLIEST_TICK)
    return starts


def apply_blinks(
    controls: NDArray[np.floating],
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    seed: int | None = 0,
    enabled: bool = True,
) -> NDArray[np.float64]:
    """Overlay blinks onto eye channels. Preserves mouth/lip/jaw channels."""
    out = np.asarray(controls, dtype=np.float64).copy()
    if not enabled or out.ndim != 2 or out.shape[1] < GROUP_DIM:
        return out
    n = out.shape[0]
    env = blink_envelope()
    for start in plan_blink_starts(n, interval_s=interval_s, seed=seed):
        for k, amt in enumerate(env):
            t = start + k
            if t >= n:
                break
            # C[0] eye_aperture: max with blink close (1 = closed)
            out[t, 0] = max(float(out[t, 0]), float(amt))
            # C[1] blink cue pulse (positive = blink event marker)
            out[t, 1] = max(float(out[t, 1]), float(amt) * 0.85)
    # clamp eye domain
    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1], -1.0, 1.0)
    return out
