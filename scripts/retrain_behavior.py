#!/usr/bin/env python3
"""Retrain avatar behavior model from a (new) upload video.

Replaces ``cell_transition_track.*`` + ``behavior_model.joblib`` in the world
dir. Does **not** re-digest the identity ``.bds`` unless you also run
``amin_train.py``.

Usage:
    # First train / retrain on the current take
    python scripts/retrain_behavior.py

    # User uploaded a new video for the same avatar folder
    python scripts/retrain_behavior.py --video path/to/new_take.mp4

    # New avatar world
    python scripts/retrain_behavior.py --video NEW.mp4 --world-dir output/worlds/friend_b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aiface.avatar_profile import write_avatar_profile  # noqa: E402
from aiface.behavior.pipeline import train_behavior_from_video  # noqa: E402
from aiface.behavior.schema import (  # noqa: E402
    BEHAVIOR_META,
    BEHAVIOR_MODEL,
    TRACK_JSON,
    TRACK_NPZ,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT
        / "assets"
        / "avatar_video_inputs"
        / "Generate_a_single_continuous_.mp4",
        help="Upload video used to measure transitions + train ML fill",
    )
    parser.add_argument(
        "--world-dir",
        type=Path,
        default=ROOT / "output" / "worlds" / "avatar",
        help="Avatar world directory (model is written here, replacing prior)",
    )
    parser.add_argument("--sample-fps", type=float, default=12.0)
    parser.add_argument("--landmarker-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    video = Path(args.video).resolve()
    world_dir = Path(args.world_dir).resolve()
    if not video.is_file():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 1
    world_dir.mkdir(parents=True, exist_ok=True)

    # Retrain always overwrites prior model artifacts in this world.
    for name in (TRACK_NPZ, TRACK_JSON, BEHAVIOR_MODEL, BEHAVIOR_META):
        stale = world_dir / name
        if stale.is_file():
            print(f"retrain: replacing {stale.name}")

    print(f"retrain: video={video}")
    print(f"retrain: world={world_dir}")
    meta = train_behavior_from_video(
        video,
        world_dir=world_dir,
        sample_fps=float(args.sample_fps),
        landmarker_model=args.landmarker_model,
        seed=int(args.seed),
    )
    profile_path = write_avatar_profile(world_dir, avatar_id=world_dir.name)
    meta["avatar_profile"] = str(profile_path)
    meta["retrain"] = True
    meta["active_model"] = str(world_dir / BEHAVIOR_MODEL)

    report_path = world_dir / "behavior_retrain_report.json"
    report_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("---")
    print(json.dumps(meta, indent=2))
    print()
    print(f"Active model: {world_dir / BEHAVIOR_MODEL}")
    print(
        "Play: aiface --demo --tts --gpu-log --world "
        f"{(world_dir / 'avatar_face.bds').as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
