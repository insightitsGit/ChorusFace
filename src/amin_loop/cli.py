"""Console entry: ``amin-train``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from amin_loop.pipeline import run_all_steps


def main(argv: list[str] | None = None) -> int:
    root = Path.cwd()
    parser = argparse.ArgumentParser(
        prog="amin-train",
        description="Run all AminIntheLoop steps (digest → maps → recipe → vectors)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=root
        / "assets"
        / "avatar_video_inputs"
        / "Generate_a_single_continuous_.mp4",
    )
    parser.add_argument(
        "--world-dir",
        type=Path,
        default=root / "output" / "worlds" / "avatar",
    )
    parser.add_argument("--digest-fps", type=float, default=6.0)
    parser.add_argument("--vector-fps", type=float, default=12.0)
    parser.add_argument(
        "--skip-digest",
        action="store_true",
        help="Reuse existing avatar_face.bds",
    )
    parser.add_argument("--landmarker-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
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
