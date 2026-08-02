#!/usr/bin/env python3
"""Launch ONLY the TickFeed demo world — refuse if identity path is wrong.

Optionally starts local CHORUS Fabric (control plane + target) so transport
mode is live fabric rather than spool fallback.
"""

from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "output" / "worlds" / "tickfeed"
BDS = WORLD / "avatar_face.bds"
FACE = WORLD / "source_face.png"


def _start_chorus() -> list[subprocess.Popen]:
    env = os.environ.copy()
    env["CHORUS_DIM"] = "64"
    env["CONTROL_PLANE_PORT"] = "50051"
    env["CONTROL_PLANE_HOST"] = "localhost"
    env["CHORUS_TARGET_HOST"] = "localhost"
    env["CHORUS_TARGET_PORT"] = "50053"
    env["TARGET_PORT"] = "50053"
    env["PYTHONUNBUFFERED"] = "1"
    procs: list[subprocess.Popen] = []
    cp = subprocess.Popen(
        [sys.executable, "-m", "chorus_fabric.servers", "control_plane"],
        cwd=str(ROOT),
        env=env,
    )
    procs.append(cp)
    time.sleep(0.8)
    tgt = subprocess.Popen(
        [sys.executable, "-m", "chorus_fabric.servers", "target"],
        cwd=str(ROOT),
        env=env,
    )
    procs.append(tgt)
    time.sleep(0.6)

    def _stop() -> None:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        time.sleep(0.2)
        for p in procs:
            if p.poll() is None:
                p.kill()

    atexit.register(_stop)
    print(
        f"CHORUS local: cp=localhost:50051 target=localhost:50053 "
        f"pids={[p.pid for p in procs]}"
    )
    return procs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-chorus",
        action="store_true",
        help="Skip starting local CHORUS (spool fallback OK)",
    )
    args = parser.parse_args()

    missing = [
        p
        for p in (BDS, FACE, WORLD / "ml" / "l3_face_motion.joblib")
        if not p.is_file()
    ]
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

    if not args.no_chorus:
        try:
            _start_chorus()
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: CHORUS start failed ({exc}); continuing with spool")

    print("Launch TickFeed demo")
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
