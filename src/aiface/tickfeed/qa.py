"""Side B QA — beat windows vs motion peaks on face_cell_timeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aiface.tickfeed.calibration import load_calibration_script
from aiface.tickfeed.schema import TICK_RATE_HZ


def qa_beat_motion(world: Path | str) -> dict[str, Any]:
    """Check that SMILE/OPEN/TALK windows show elevated face motion."""
    root = Path(world)
    root = root if root.is_dir() else root.parent
    npz = root / "face_cell_timeline.npz"
    if not npz.is_file():
        return {"ok": False, "error": f"missing {npz.name}"}
    data = np.load(npz)
    ticks = np.asarray(data["ticks"], dtype=np.int32)
    vel = np.asarray(data["velocity"], dtype=np.float32)
    script = load_calibration_script(root)
    energy = np.mean(np.abs(vel).reshape(len(ticks), -1), axis=1)
    by_beat: dict[str, list[float]] = {}
    for i, tick in enumerate(ticks):
        t = float(tick) / float(TICK_RATE_HZ)
        bid = "REST"
        for beat in script.get("beats") or []:
            if float(beat["t0"]) <= t < float(beat["t1"]):
                bid = str(beat["id"])
                break
        by_beat.setdefault(bid, []).append(float(energy[i]))
    means = {k: float(np.mean(v)) if v else 0.0 for k, v in by_beat.items()}
    rest = means.get("REST", 0.0) + 1e-8
    checks = {
        "smile_gt_rest": means.get("SMILE", 0.0) > rest * 1.05,
        "open_gt_rest": means.get("OPEN", 0.0) > rest * 1.05,
        "talk_gt_rest": means.get("TALK", 0.0) > rest * 1.02,
    }
    ok = all(checks.values()) if len(means) >= 3 else False
    return {
        "ok": ok,
        "means": means,
        "checks": checks,
        "n_ticks": int(len(ticks)),
    }


__all__ = ["qa_beat_motion"]
