#!/usr/bin/env python3
"""Rebuild TickFeed LOOK plate atlas (+ optional FIELD timeline) without full digest.

Uses the calibration video already in the world (or --video). Improves PP/TH/FF
selection via updated ``select_viseme_atlas_frames`` scoring.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-dir",
        type=Path,
        default=ROOT / "output" / "worlds" / "tickfeed",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Override calibration video (default: world/calibration_take.mp4)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=12.0,
        help="Frame sample rate for plate landmark match",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Also rebuild face_cell_timeline.npz with denser Farneback",
    )
    args = parser.parse_args()

    world = Path(args.world_dir).resolve()
    bds = world / "avatar_face.bds"
    video = Path(args.video).resolve() if args.video else world / "calibration_take.mp4"
    if not bds.is_file():
        print(f"FAIL: missing {bds}", file=sys.stderr)
        return 2
    if not video.is_file():
        print(f"FAIL: missing video {video}", file=sys.stderr)
        return 2

    from chorusface.capture import (
        MIN_SHARPNESS_SOFT,
        RejectReport,
        _write_plate_atlas,
        iter_video_frames,
        resample_frames_hires,
        select_expression_frames,
    )
    from chorusface.plates import select_viseme_atlas_frames

    print(f"Sampling frames from {video.name} @ {args.sample_fps} fps …")
    report = RejectReport()
    frames = iter_video_frames(
        video,
        sample_fps=float(args.sample_fps),
        min_sharpness=MIN_SHARPNESS_SOFT,
        report=report,
    )
    if len(frames) < 8:
        print(f"FAIL: only {len(frames)} usable frames", file=sys.stderr)
        return 3

    selection = select_expression_frames(frames)
    atlas_frames, mapping = select_viseme_atlas_frames(frames, max_plates=12)
    print(
        f"Selected {len(atlas_frames)} unique plates; "
        f"PP→{mapping.get('PP')} TH→{mapping.get('TH')} "
        f"FF→{mapping.get('FF')} CLOSED→{mapping.get('CLOSED')}"
    )
    display_targets = [
        selection.rest,
        selection.smile,
        selection.open,
        *([selection.surprise] if selection.surprise is not None else []),
        *atlas_frames,
    ]
    hires = resample_frames_hires(video, display_targets)
    paths = _write_plate_atlas(
        bds,
        frames,
        source_label=video.name,
        hires=hires,
        reference=selection.rest,
    )
    # Prefer flattest closed talk frame as identity if clearly less smiley.
    rest = selection.rest
    flat = min(frames, key=lambda f: (f.metrics.smile_width, f.metrics.mouth_open))
    if float(flat.metrics.smile_width) + 0.04 < float(rest.metrics.smile_width):
        import cv2

        src = world / "source_face.png"
        bak = world / "source_face.smile_bak.png"
        if src.is_file() and not bak.is_file():
            shutil.copy2(src, bak)
        hi = hires.get(int(flat.index), flat)
        cv2.imwrite(str(src), hi.image_bgr)
        print(
            f"Identity rest updated toward flatter frame "
            f"smile={flat.metrics.smile_width:.3f} (was {rest.metrics.smile_width:.3f})"
        )

    from chorusface.plates import load_plate_atlas

    # Meta path is sibling of the .bds (with_name), not the world directory.
    atlas = load_plate_atlas(bds)
    if atlas is None:
        print("FAIL: could not reload plate_atlas.json", file=sys.stderr)
        return 4
    print(
        f"Wrote {len(paths)} plates; atlas load OK "
        f"({len(atlas.plates)} entries, map={len(atlas.viseme_to_plate)})"
    )
    meta = bds.with_name("plate_atlas.json")
    print(f"  meta: {meta}")

    if args.timeline:
        print("Rebuilding face_cell_timeline (denser Farneback) …")
        from chorusface.tickfeed.collect import prepare_face_timeline

        prepare_face_timeline(world, video=video)
        print("  timeline OK")

    summary = {
        "plates": len(paths),
        "viseme_to_plate": dict(
            sorted((k, int(v)) for k, v in atlas.viseme_to_plate.items())
        ),
        "closed_openness": [
            float(p.openness)
            for p in atlas.plates
            if p.viseme in {"CLOSED", "PP", "REST"}
        ],
    }
    (world / "plate_rebuild_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("DONE", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
