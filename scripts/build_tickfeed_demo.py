#!/usr/bin/env python3
"""Build a clean TickFeed demo world from the Gemini 8s calibration take.

Deletes nothing itself — expects an empty ``output/worlds/tickfeed`` (or will
overwrite digest artifacts there). Pipeline:

1. amin_train digest (identity .bds + LOOK plates from video)
2. Side B FaceCellTimeline + speech/look + L1–L5
3. Hard identity check before printing the play command
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amin_loop.pipeline import run_all_steps  # noqa: E402
from chorusface.tickfeed.calibration import (
    validate_calibration_take,
    write_calibration_script,
)
from chorusface.tickfeed.collect import prepare_face_timeline  # noqa: E402
from chorusface.tickfeed.cosmetics import write_cosmetic_prefs  # noqa: E402
from chorusface.tickfeed.ml.train import fit_all_layers  # noqa: E402
from chorusface.tickfeed.qa import qa_beat_motion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT
        / "assets"
        / "avatar_video_inputs"
        / "calibration_takes"
        / "blonde_woman_8s.mp4",
    )
    parser.add_argument(
        "--world-dir",
        type=Path,
        default=ROOT / "output" / "worlds" / "tickfeed",
    )
    parser.add_argument("--clean", action="store_true", help="Wipe world dir first")
    args = parser.parse_args()

    video = Path(args.video).resolve()
    world = Path(args.world_dir).resolve()
    if not video.is_file():
        print(f"FAIL: missing video {video}", file=sys.stderr)
        return 2
    if args.clean and world.exists():
        shutil.rmtree(world)
    world.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video, world / "calibration_take.mp4")

    print("=== 1/3 digest identity + LOOK plates ===")
    digest_report = run_all_steps(
        video,
        world_dir=world,
        digest_fps=6.0,
        vector_fps=12.0,
        skip_digest=False,
        seed=17,
    )
    (world / "amin_loop_report.json").write_text(
        json.dumps(digest_report, indent=2) + "\n", encoding="utf-8"
    )

    source = world / "source_face.png"
    if not source.is_file():
        print("FAIL: digest did not write source_face.png", file=sys.stderr)
        return 3

    print("=== 2/3 Side B TickFeed timeline + L1–L5 ===")
    write_calibration_script(world)
    write_cosmetic_prefs(world)
    vreport = validate_calibration_take(world, world / "calibration_take.mp4")
    print("validate:", vreport)
    if not vreport.get("ok"):
        print("FAIL: calibration take validation", file=sys.stderr)
        return 4
    prepare_face_timeline(world, world / "calibration_take.mp4")
    meta = fit_all_layers(world)
    qa = qa_beat_motion(world)
    print("layers:", meta.get("layers"))
    print("qa:", qa)

    print("=== 3/3 identity hard check ===")
    bds = world / "avatar_face.bds"
    ml = world / "ml" / "l3_face_motion.joblib"
    timeline = world / "face_cell_timeline.npz"
    checks = {
        "source_face": source.is_file(),
        "bds": bds.is_file(),
        "smile_plate": (world / "smile.png").is_file(),
        "open_plate": (world / "open.png").is_file(),
        "timeline": timeline.is_file(),
        "ml_l3": ml.is_file(),
        "qa_ok": bool(qa.get("ok")),
        "world_name_is_tickfeed": world.name == "tickfeed",
    }
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        print("FAIL: tickfeed demo world incomplete", file=sys.stderr)
        return 5

    play = (
        f'python -m chorusface --demo --tts --gpu-log --world "{bds}" '
        f'--face-image "{source}"'
    )
    print()
    print("READY TickFeed demo world:", world)
    print("Identity:", source)
    print("Play:", play)
    (world / "PLAY.txt").write_text(play + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
