#!/usr/bin/env python3
"""From-scratch: train avatar live vectors from a capture video.

    python scripts/train_avatar_from_video.py \\
      --video assets/avatar_video_inputs/Generate_a_single_continuous_.mp4 \\
      --world-dir output/worlds/avatar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chorusface.live_vector.pipeline import train_avatar_from_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT
        / "assets"
        / "avatar_video_inputs"
        / "Generate_a_single_continuous_.mp4",
    )
    parser.add_argument(
        "--world-dir",
        type=Path,
        default=ROOT / "output" / "worlds" / "avatar",
    )
    parser.add_argument("--sample-fps", type=float, default=12.0)
    parser.add_argument("--landmarker-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    meta = train_avatar_from_video(
        args.video,
        world_dir=args.world_dir,
        sample_fps=float(args.sample_fps),
        landmarker_model=args.landmarker_model,
        seed=int(args.seed),
    )
    print("---")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
