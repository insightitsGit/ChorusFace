#!/usr/bin/env python3
"""Run all AminIntheLoop steps on a capture video.

    python scripts/amin_train.py \\
      --video assets/avatar_video_inputs/Generate_a_single_continuous_.mp4 \\
      --world-dir output/worlds/avatar

Then play (no Path A seals):

    aiface --demo --tts --world output/worlds/avatar/avatar_face.bds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amin_loop.pipeline import run_all_steps  # noqa: E402


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
    parser.add_argument("--digest-fps", type=float, default=6.0)
    parser.add_argument("--vector-fps", type=float, default=12.0)
    parser.add_argument(
        "--skip-digest",
        action="store_true",
        help="Reuse existing avatar_face.bds; only remap + retrain vectors",
    )
    parser.add_argument(
        "--behavior-only",
        action="store_true",
        help=(
            "Only retrain cell_transition_track + behavior_model from --video "
            "(fast path when user uploads a new take for the same face)"
        ),
    )
    parser.add_argument("--landmarker-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if bool(args.behavior_only):
        from aiface.avatar_profile import write_avatar_profile
        from aiface.behavior.pipeline import train_behavior_from_video

        world_dir = Path(args.world_dir).resolve()
        meta = train_behavior_from_video(
            Path(args.video),
            world_dir=world_dir,
            sample_fps=float(args.vector_fps),
            landmarker_model=args.landmarker_model,
            seed=int(args.seed),
        )
        write_avatar_profile(world_dir, avatar_id=world_dir.name)
        report = {
            "schema": "amin_loop.behavior_only.v1",
            "video": str(Path(args.video).resolve()),
            "world_dir": str(world_dir),
            "steps": {"behavior": meta},
            "note": "Retrainable: new upload → replace behavior_model in world dir",
        }
    else:
        report = run_all_steps(
            args.video,
            world_dir=args.world_dir,
            digest_fps=float(args.digest_fps),
            vector_fps=float(args.vector_fps),
            skip_digest=bool(args.skip_digest),
            landmarker_model=args.landmarker_model,
            seed=int(args.seed),
        )
    print("---")
    print(json.dumps(report, indent=2))
    print()
    print(
        "Play: aiface --demo --tts --world "
        f"{(Path(args.world_dir) / 'avatar_face.bds').as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
