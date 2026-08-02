#!/usr/bin/env python3
"""Start local CHORUS Fabric control plane + target pod for TickFeed.

Uses CHORUS_DIM=64 to match TickFeed L4 code size.
Ports: control plane 50051, target 50053 (direct send, no relay).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIM = "64"
CP_PORT = "50051"
TARGET_PORT = "50053"


def main() -> int:
    env = os.environ.copy()
    env["CHORUS_DIM"] = DIM
    env["CONTROL_PLANE_PORT"] = CP_PORT
    env["CONTROL_PLANE_HOST"] = "localhost"
    env["CHORUS_TARGET_HOST"] = "localhost"
    env["CHORUS_TARGET_PORT"] = TARGET_PORT
    env["TARGET_PORT"] = TARGET_PORT
    env["PYTHONUNBUFFERED"] = "1"

    cp = subprocess.Popen(
        [sys.executable, "-m", "chorus_fabric.servers", "control_plane"],
        cwd=str(ROOT),
        env=env,
    )
    time.sleep(0.8)
    target = subprocess.Popen(
        [sys.executable, "-m", "chorus_fabric.servers", "target"],
        cwd=str(ROOT),
        env=env,
    )
    time.sleep(0.6)
    print(
        f"CHORUS local up: control_plane=localhost:{CP_PORT} "
        f"target=localhost:{TARGET_PORT} dim={DIM}"
    )
    print(f"pids: control_plane={cp.pid} target={target.pid}")
    print("Ctrl+C to stop both.")
    try:
        while True:
            if cp.poll() is not None or target.poll() is not None:
                print("A CHORUS process exited; shutting down.", file=sys.stderr)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (cp, target):
            if proc.poll() is None:
                proc.terminate()
        time.sleep(0.3)
        for proc in (cp, target):
            if proc.poll() is None:
                proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
