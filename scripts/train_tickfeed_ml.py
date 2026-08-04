#!/usr/bin/env python3
"""Train TickFeed L1–L5 into world/ml/ from face_cell_timeline.npz."""

from __future__ import annotations

import argparse
from pathlib import Path

from chorusface.tickfeed.calibration import (
    validate_calibration_take,
    write_calibration_script,
)
from chorusface.tickfeed.collect import prepare_face_timeline
from chorusface.tickfeed.cosmetics import write_cosmetic_prefs
from chorusface.tickfeed.ml.train import fit_layer
from chorusface.tickfeed.qa import qa_beat_motion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-dir", type=Path, default=Path("output/worlds/avatar")
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run Side B prepare_face_timeline before train",
    )
    parser.add_argument(
        "--layer",
        default="all",
        help="Retrain layer: all|l1|l2|l3|l4|l5 (default all)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate calibration take against scaffolding lock",
    )
    args = parser.parse_args()
    write_calibration_script(args.world_dir)
    write_cosmetic_prefs(args.world_dir)
    if args.validate:
        report = validate_calibration_take(args.world_dir, args.video)
        print("validate:", report)
        if not report.get("ok"):
            raise SystemExit(2)
    if args.prepare or not (args.world_dir / "face_cell_timeline.npz").is_file():
        prepare_face_timeline(args.world_dir, args.video)
    meta = fit_layer(args.world_dir, args.layer)
    print("layers:", meta.get("layers"))
    print("qa:", qa_beat_motion(args.world_dir))


if __name__ == "__main__":
    main()
