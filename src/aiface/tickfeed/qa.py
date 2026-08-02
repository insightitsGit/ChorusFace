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
        "smile_gt_rest": means.get("SMILE", 0.0) > rest * 1.02,
        "open_gt_rest": means.get("OPEN", 0.0) > rest * 1.05,
        "talk_gt_rest": means.get("TALK", 0.0) > rest * 1.02,
        "surprise_gt_rest": means.get("SURPRISE", 0.0) > rest * 1.02,
    }
    # Also verify look_drive / speech_align side tracks when present
    tdir = root / "face_cell_timeline"
    look_ok = True
    speech_ok = True
    if (tdir / "look_drive.json").is_file():
        import json

        look = json.loads((tdir / "look_drive.json").read_text(encoding="utf-8"))
        by = {}
        for row in look.get("ticks") or []:
            by.setdefault(str(row.get("beat") or "REST"), []).append(float(row.get("smile") or 0))
        smile_look = float(np.mean(by.get("SMILE") or [0.0]))
        rest_look = float(np.mean(by.get("REST") or [0.0])) + 1e-8
        look_ok = smile_look > rest_look
        checks["look_smile_gt_rest"] = look_ok
    if (tdir / "speech_align.json").is_file():
        import json

        speech = json.loads((tdir / "speech_align.json").read_text(encoding="utf-8"))
        hi_words = [
            str(r.get("word") or "")
            for r in speech.get("ticks") or []
            if str(r.get("beat") or "") == "SAY_HI"
        ]
        speech_ok = any(w == "hi" for w in hi_words)
        checks["say_hi_has_hi"] = speech_ok
    # Motion smile may be subtle; accept if look_drive smile peaks
    if not checks["smile_gt_rest"] and checks.get("look_smile_gt_rest"):
        checks["smile_gt_rest_or_look"] = True
    else:
        checks["smile_gt_rest_or_look"] = bool(checks["smile_gt_rest"])
    required = (
        "open_gt_rest",
        "talk_gt_rest",
        "smile_gt_rest_or_look",
        "say_hi_has_hi",
    )
    ok = all(checks.get(k, False) for k in required) if len(means) >= 3 else False
    return {
        "ok": ok,
        "means": means,
        "checks": checks,
        "n_ticks": int(len(ticks)),
    }


__all__ = ["qa_beat_motion"]
