#!/usr/bin/env python3
"""Launch ONLY the TickFeed demo world — refuse if identity path is wrong."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "output" / "worlds" / "tickfeed"
BDS = WORLD / "avatar_face.bds"
FACE = WORLD / "source_face.png"


def main() -> int:
    missing = [p for p in (BDS, FACE, WORLD / "ml" / "l3_face_motion.joblib") if not p.is_file()]
    if missing:
        print("FAIL: TickFeed demo not built. Missing:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print(
            "Run: python scripts/build_tickfeed_demo.py --clean",
            file=sys.stderr,
        )
        return 2
    if "tickfeed" not in str(FACE.resolve()).replace("\\", "/"):
        print(f"FAIL: refusing non-tickfeed identity {FACE}", file=sys.stderr)
        return 3
    print(f"Launch TickFeed demo")
    print(f"  world: {BDS}")
    print(f"  face:  {FACE}")
    cmd = [
        sys.executable,
        "-m",
        "aiface",
        "--demo",
        "--tts",
        "--gpu-log",
        "--world",
        str(BDS),
        "--face-image",
        str(FACE),
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
