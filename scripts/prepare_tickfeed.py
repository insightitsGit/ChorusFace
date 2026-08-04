#!/usr/bin/env python3
"""Prepare face_cell_timeline.npz (Side B) for TickFeed playback."""

from __future__ import annotations

import argparse
from pathlib import Path

from chorusface.tickfeed.collect import prepare_face_timeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-dir",
        type=Path,
        default=Path("output/worlds/avatar"),
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--sample-fps", type=float, default=12.0)
    args = parser.parse_args()
    prepare_face_timeline(
        args.world_dir, args.video, sample_fps=float(args.sample_fps)
    )


if __name__ == "__main__":
    main()
