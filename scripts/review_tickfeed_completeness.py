#!/usr/bin/env python3
"""Print TickFeed design completeness against output/worlds/tickfeed."""

from __future__ import annotations

import json
from pathlib import Path

W = Path("output/worlds/tickfeed")


def ok(p: Path) -> str:
    return "YES" if p.is_file() or p.is_dir() else "NO "


def main() -> None:
    print("World:", W.resolve())
    print("exists:", W.is_dir())
    files = [
        "avatar_face.bds",
        "source_face.png",
        "smile.png",
        "open.png",
        "surprise.png",
        "calibration_script.json",
        "calibration_take.mp4",
        "face_cell_timeline.npz",
        "cosmetic_prefs.json",
        "ml/l1_speech_clock.joblib",
        "ml/l2_look_drive.joblib",
        "ml/l3_face_motion.joblib",
        "ml/l4_tick_codec.joblib",
        "ml/l5_gap_prior.joblib",
        "ml/tickfeed_ml.meta.json",
        "face_cell_timeline/meta.json",
        "face_cell_timeline/rest_ref.npz",
        "face_cell_timeline/speech_align.json",
        "face_cell_timeline/look_drive.json",
        "face_cell_timeline/qa_report.json",
    ]
    print("\nARTIFACTS")
    for rel in files:
        p = W / rel
        mark = "YES" if p.is_file() else "NO "
        size = p.stat().st_size if p.is_file() else 0
        print(f"  {mark}  {rel:45s}  {size}")
    chunks = list((W / "face_cell_timeline").glob("ticks_*.npz")) if (W / "face_cell_timeline").is_dir() else []
    print(f"  {'YES' if chunks else 'NO '}  ticks_*.npz chunks                         n={len(chunks)}")
    meta = W / "ml" / "tickfeed_ml.meta.json"
    if meta.is_file():
        print("\nML_META")
        print(meta.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
