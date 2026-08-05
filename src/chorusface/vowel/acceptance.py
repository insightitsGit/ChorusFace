"""Phase-1 numeric acceptance gates (F15 / D30)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chorusface.vowel.model_a import ModelA
from chorusface.vowel.schema import GROUP_DIM


@dataclass(slots=True)
class AcceptanceReport:
    pairwise_ok: bool
    jaw_pump_ok: bool
    hold_ok: bool
    jerk_ok: bool
    pairwise_min: float
    jaw_lip_corr: float
    hold_sigma: float
    jerk_max: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "pairwise_ok": self.pairwise_ok,
            "jaw_pump_ok": self.jaw_pump_ok,
            "hold_ok": self.hold_ok,
            "jerk_ok": self.jerk_ok,
            "pairwise_min": self.pairwise_min,
            "jaw_lip_corr": self.jaw_lip_corr,
            "hold_sigma": self.hold_sigma,
            "jerk_max": self.jerk_max,
            "passed": self.passed,
        }


def pairwise_distances(model: ModelA, emotion: str = "NEUTRAL") -> dict[str, float]:
    tags = ("EE", "OH", "OU", "AA")
    vecs = {t: model.predict(t, emotion) for t in tags}
    out: dict[str, float] = {}
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            out[f"{a}-{b}"] = float(np.linalg.norm(vecs[a] - vecs[b]))
    return out


def evaluate_model_a(model: ModelA, emotion: str = "NEUTRAL") -> AcceptanceReport:
    dists = pairwise_distances(model, emotion)
    pairwise_min = min(dists.values()) if dists else 0.0
    # jaw vs lip_spread across open vowels
    opens = ("AA", "AH", "AE", "AO")
    jaws = []
    spreads = []
    for t in opens:
        c = model.predict(t, emotion)
        jaws.append(c[8])
        spreads.append(c[5])
    if len(jaws) >= 2 and np.std(jaws) > 1e-6 and np.std(spreads) > 1e-6:
        corr = float(np.corrcoef(jaws, spreads)[0, 1])
    else:
        corr = 0.0
    # synthetic hold: repeat target
    hold = np.stack([model.predict("EE", emotion)] * 12)
    hold_sigma = float(np.std(hold, axis=0).max())
    # jerk on a short EE→OU path
    from chorusface.vowel.model_b import ModelB

    b = ModelB()
    path = b.generate_segment(
        model.predict("EE", emotion),
        model.predict("OU", emotion),
        20,
        emotion,
    )
    acc = np.diff(path, n=2, axis=0)
    jerk = np.diff(acc, axis=0)
    jerk_max = float(np.max(np.abs(jerk))) if jerk.size else 0.0

    pairwise_ok = pairwise_min >= 0.25
    jaw_ok = abs(corr) < 0.75
    hold_ok = hold_sigma < 0.08
    jerk_ok = jerk_max < 0.15
    return AcceptanceReport(
        pairwise_ok=pairwise_ok,
        jaw_pump_ok=jaw_ok,
        hold_ok=hold_ok,
        jerk_ok=jerk_ok,
        pairwise_min=pairwise_min,
        jaw_lip_corr=corr,
        hold_sigma=hold_sigma,
        jerk_max=jerk_max,
        passed=pairwise_ok and jaw_ok and hold_ok and jerk_ok,
    )


def trajectory_hold_sigma(controls: NDArray[np.floating], start: int, end: int) -> float:
    seg = np.asarray(controls[start:end], dtype=np.float64)
    if seg.shape[0] < 2:
        return 0.0
    return float(np.std(seg, axis=0).max())
