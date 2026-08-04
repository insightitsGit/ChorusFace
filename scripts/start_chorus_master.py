#!/usr/bin/env python3
"""Start CHORUS control plane + ChorusFace master target (vector consume spool)."""

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
RECV = ROOT / "output" / "worlds" / "tickfeed" / "tickfeed_chorus_spool" / "recv"


def main() -> int:
    env = os.environ.copy()
    env["CHORUS_DIM"] = DIM
    env["CONTROL_PLANE_PORT"] = CP_PORT
    env["CONTROL_PLANE_HOST"] = "localhost"
    env["CHORUS_TARGET_HOST"] = "localhost"
    env["CHORUS_TARGET_PORT"] = TARGET_PORT
    env["TARGET_PORT"] = TARGET_PORT
    env["CHORUSFACE_CHORUS_RECV_SPOOL"] = str(RECV)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    RECV.mkdir(parents=True, exist_ok=True)

    cp = subprocess.Popen(
        [sys.executable, "-m", "chorus_fabric.servers", "control_plane"],
        cwd=str(ROOT),
        env=env,
    )
    time.sleep(0.8)
    target = subprocess.Popen(
        [sys.executable, "-m", "chorusface.tickfeed.chorus_master"],
        cwd=str(ROOT),
        env=env,
    )
    time.sleep(0.6)
    print(
        f"CHORUS master up: cp=localhost:{CP_PORT} "
        f"target=localhost:{TARGET_PORT} dim={DIM}"
    )
    print(f"recv spool: {RECV}")
    print(f"pids: control_plane={cp.pid} master_target={target.pid}")
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
