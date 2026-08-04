"""8s calibration script artifact (Side B beat contract)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from aiface.tickfeed.schema import BeatId, TICK_RATE_HZ

SCRIPT_NAME: Final = "calibration_script.json"
SCRIPT_VERSION: Final = "aiface.calibration_script.v3"
DURATION_S: Final = 8.0

# Dense teacher kit packed into 8.0s — REST/OPEN/TH + deliberate BLINK for lids.
DEFAULT_BEATS: Final[tuple[dict[str, Any], ...]] = (
    {
        "id": "REST",
        "beat_id": int(BeatId.REST),
        "t0": 0.0,
        "t1": 0.7,
        "speech": "",
        "notes": "true_neutral_flat_lips",
    },
    {
        "id": "SMILE",
        "beat_id": int(BeatId.SMILE),
        "t0": 0.7,
        "t1": 1.4,
        "speech": "",
        "notes": "closed_lip_max_corners",
    },
    {
        "id": "OPEN",
        "beat_id": int(BeatId.OPEN),
        "t0": 1.4,
        "t1": 2.3,
        "speech": "ah",
        "notes": "wide_ah_upper_and_lower_teeth",
    },
    {
        "id": "SAY_HI",
        "beat_id": int(BeatId.SAY_HI),
        "t0": 2.3,
        "t1": 3.0,
        "speech": "hi",
    },
    {
        "id": "TONGUE_TH",
        "beat_id": int(BeatId.TONGUE_TH),
        "t0": 3.0,
        "t1": 3.7,
        "speech": "think",
        "notes": "tongue_tip_between_teeth",
    },
    {
        "id": "SURPRISE",
        "beat_id": int(BeatId.SURPRISE),
        "t0": 3.7,
        "t1": 4.4,
        "speech": "oh",
    },
    {
        "id": "ANGRY",
        "beat_id": int(BeatId.ANGRY),
        "t0": 4.4,
        "t1": 5.1,
        "speech": "",
    },
    {
        "id": "BLINK",
        "beat_id": int(BeatId.BLINK),
        "t0": 5.1,
        "t1": 5.7,
        "speech": "",
        "notes": "full_bilateral_lid_close_hold",
    },
    {
        "id": "TALK",
        "beat_id": int(BeatId.TALK),
        "t0": 5.7,
        "t1": 7.5,
        "speech": "Hello there. How are you today?",
    },
    {
        "id": "REST",
        "beat_id": int(BeatId.REST),
        "t0": 7.5,
        "t1": 8.0,
        "speech": "",
        "notes": "true_neutral_flat_lips",
    },
)


def calibration_script_payload() -> dict[str, Any]:
    return {
        "schema": SCRIPT_VERSION,
        "duration_s": DURATION_S,
        "tick_rate": TICK_RATE_HZ,
        "talk_line": "Hello there. How are you today?",
        "dense_kit": True,
        "blink_kit": True,
        "beats": [dict(b) for b in DEFAULT_BEATS],
        "prompt_doc": "docs/AvatarCalibrationPrompt.md",
    }


def write_calibration_script(world: Path | str) -> Path:
    root = Path(world)
    root = root if root.is_dir() else root.parent
    path = root / SCRIPT_NAME
    path.write_text(
        json.dumps(calibration_script_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_calibration_script(world: Path | str) -> dict[str, Any]:
    root = Path(world)
    root = root if root.is_dir() else root.parent
    path = root / SCRIPT_NAME
    if not path.is_file():
        write_calibration_script(root)
    return json.loads(path.read_text(encoding="utf-8"))


def beat_at_time(script: dict[str, Any], t: float) -> dict[str, Any]:
    for beat in script.get("beats") or []:
        if float(beat["t0"]) <= t < float(beat["t1"]):
            return beat
    beats = script.get("beats") or []
    return beats[-1] if beats else {"id": "REST", "beat_id": 0}


def validate_calibration_take(
    world: Path | str,
    video: Path | str | None = None,
) -> dict[str, Any]:
    """Accept/reject upload against scaffolding lock (face + duration)."""
    root = Path(world)
    root = root if root.is_dir() else root.parent
    script = load_calibration_script(root)
    report: dict[str, Any] = {
        "ok": False,
        "script_duration_s": float(script.get("duration_s") or DURATION_S),
        "checks": {},
        "video": None,
    }
    vid = Path(video) if video else None
    if vid is None:
        search_roots = (root, Path("assets/avatar_video_inputs"))
        for base in search_roots:
            for name in (
                "calibration_take.mp4",
                "Generate_a_single_continuous_.mp4",
                "source.mp4",
                "avatar.mp4",
                "blonde_woman_8s.mp4",
                "male_8s.mp4",
            ):
                cand = base / name
                if cand.is_file():
                    vid = cand
                    break
                nested = base / "calibration_takes" / name
                if nested.is_file():
                    vid = nested
                    break
            if vid is not None:
                break
    if vid is None or not vid.is_file():
        report["checks"]["video_present"] = False
        report["reason"] = "no calibration video found"
        return report
    report["video"] = str(vid)
    report["checks"]["video_present"] = True
    try:
        import cv2

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            report["checks"]["video_readable"] = False
            report["reason"] = "cannot open video"
            return report
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        dur = (n / fps) if fps > 1e-3 else 0.0
        report["checks"]["video_readable"] = True
        report["checks"]["duration_ok"] = 6.5 <= dur <= 20.0
        report["checks"]["resolution_ok"] = w >= 256 and h >= 256
        report["duration_s"] = dur
        report["size"] = [w, h]
    except Exception as exc:  # noqa: BLE001
        report["checks"]["video_readable"] = False
        report["reason"] = str(exc)
        return report
    report["checks"]["script_present"] = (root / SCRIPT_NAME).is_file()
    beats = script.get("beats") or []
    report["checks"]["has_tongue_th_beat"] = any(
        str(b.get("id") or "") == "TONGUE_TH" for b in beats
    )
    report["checks"]["has_blink_beat"] = any(
        str(b.get("id") or "") == "BLINK" for b in beats
    )
    report["ok"] = all(bool(v) for v in report["checks"].values())
    if not report["ok"]:
        report["reason"] = "failed scaffolding lock checks"
    return report


__all__ = [
    "DURATION_S",
    "SCRIPT_NAME",
    "SCRIPT_VERSION",
    "beat_at_time",
    "calibration_script_payload",
    "load_calibration_script",
    "validate_calibration_take",
    "write_calibration_script",
]
