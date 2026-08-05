"""Model B — residual trajectory generator (D13 / D14 / D15)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chorusface.vowel.priors import clamp_9d, rest_9d
from chorusface.vowel.schema import (
    ATTACK_TICKS,
    COARTIC_BLEND_TICKS,
    CONFLICT_BRIDGE_TICKS,
    DIPHTHONG_ENDS,
    GROUP_DIM,
    RELEASE_TICKS,
    ROUND_VOWELS,
    SPREAD_VOWELS,
)


def smoothstep(tau: float) -> float:
    t = float(np.clip(tau, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def cosine_blend(tau: float) -> float:
    t = float(np.clip(tau, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * t))


class ModelB:
    """Analytic residual path + optional learned velocity corrector.

    Phase-1 default: closed-form attack/hold/release with diphthong smoothstep
    and coarticulation bridges. A tiny residual MLP can be fit later from
    Dataset B without changing the ONNX I/O of Model A.
    """

    def __init__(self) -> None:
        # residual corrector: [C_prev(9)+C_tgt(9)+tau+phase3] → Δ(9)
        rng = np.random.default_rng(1)
        self.W = rng.normal(0, 0.01, size=(22, GROUP_DIM))
        self.b = np.zeros(GROUP_DIM)
        self.use_learned = False

    def attack_ticks(self, emotion: str) -> int:
        return int(ATTACK_TICKS.get((emotion or "NEUTRAL").upper(), 6))

    def target_with_diphthong(
        self,
        start: NDArray[np.floating],
        end: NDArray[np.floating] | None,
        tau: float,
    ) -> NDArray[np.float64]:
        if end is None:
            return clamp_9d(start)
        a = smoothstep(tau)
        return clamp_9d((1.0 - a) * np.asarray(start) + a * np.asarray(end))

    def generate_segment(
        self,
        c0: NDArray[np.floating],
        c_target: NDArray[np.floating],
        n_ticks: int,
        emotion: str,
        *,
        c_end: NDArray[np.floating] | None = None,
        release: bool = False,
    ) -> NDArray[np.float64]:
        """Generate absolute 9D trajectory for one vowel span."""
        n = max(1, int(n_ticks))
        atk = min(self.attack_ticks(emotion), max(1, n // 2))
        out = np.zeros((n, GROUP_DIM), dtype=np.float64)
        c0 = clamp_9d(c0)
        hold_target = clamp_9d(c_target)
        for t in range(n):
            if t < atk:
                tau = t / max(1, atk)
                blend = cosine_blend(tau)
                tgt = self.target_with_diphthong(
                    hold_target, c_end, tau if c_end is not None else 0.0
                )
                out[t] = clamp_9d((1.0 - blend) * c0 + blend * tgt)
            else:
                # hold — advance diphthong through hold as well
                if c_end is not None:
                    tau = t / max(1, n - 1)
                    out[t] = self.target_with_diphthong(hold_target, c_end, tau)
                else:
                    out[t] = hold_target
        if release and n > RELEASE_TICKS:
            rest = rest_9d(emotion)
            for k in range(RELEASE_TICKS):
                t = n - RELEASE_TICKS + k
                blend = cosine_blend((k + 1) / RELEASE_TICKS)
                out[t] = clamp_9d((1.0 - blend) * out[t] + blend * rest)
        if self.use_learned:
            for t in range(1, n):
                feat = np.concatenate(
                    [
                        out[t - 1],
                        hold_target,
                        [t / max(1, n - 1)],
                        [1.0 if t < atk else 0.0, 1.0 if atk <= t < n - RELEASE_TICKS else 0.0, 1.0 if t >= n - RELEASE_TICKS else 0.0],
                    ]
                )
                # pad/trim to 22
                if feat.shape[0] < 22:
                    feat = np.pad(feat, (0, 22 - feat.shape[0]))
                else:
                    feat = feat[:22]
                out[t] = clamp_9d(out[t] + feat @ self.W + self.b)
        return out

    @staticmethod
    def needs_conflict_bridge(tag_a: str, tag_b: str) -> bool:
        a = (tag_a or "").upper()
        b = (tag_b or "").upper()
        a_spread = a in SPREAD_VOWELS
        b_spread = b in SPREAD_VOWELS
        a_round = a in ROUND_VOWELS
        b_round = b in ROUND_VOWELS
        return (a_spread and b_round) or (a_round and b_spread)

    def bridge(
        self, c_from: NDArray[np.floating], emotion: str
    ) -> NDArray[np.float64]:
        rest = rest_9d(emotion)
        n = CONFLICT_BRIDGE_TICKS
        out = np.zeros((n, GROUP_DIM), dtype=np.float64)
        for t in range(n):
            blend = cosine_blend((t + 1) / n)
            # partial ease toward rest (~50%) then next segment continues
            mid = 0.5 * np.asarray(c_from) + 0.5 * rest
            out[t] = clamp_9d((1.0 - blend) * c_from + blend * mid)
        return out

    def crossfade(
        self, a: NDArray[np.floating], b: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """3-tick WordSlice boundary crossfade in 9D (D28)."""
        weights = [(0.75, 0.25), (0.5, 0.5), (0.25, 0.75)]
        out = np.zeros((len(weights), GROUP_DIM), dtype=np.float64)
        for i, (wa, wb) in enumerate(weights):
            out[i] = clamp_9d(wa * np.asarray(a) + wb * np.asarray(b))
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, W=self.W, b=self.b, use_learned=np.array([int(self.use_learned)])
        )

    @classmethod
    def load(cls, path: str | Path) -> ModelB:
        data = np.load(path)
        m = cls()
        m.W = data["W"]
        m.b = data["b"]
        m.use_learned = bool(int(data["use_learned"][0]))
        return m


def diphthong_end_tag(tag: str) -> str | None:
    return DIPHTHONG_ENDS.get((tag or "").upper())
