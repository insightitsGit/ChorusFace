#!/usr/bin/env python3
"""Launch ONLY the TickFeed demo world — refuse if identity/path is wrong.

Starts local CHORUS Fabric by default so lane A (c_t) + lane B (TickPackage)
use live fabric. Preflight checks the new fidelity path (provenance, align,
ML, LOOK authority prerequisites).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

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


def _preflight() -> list[str]:
    """Return human-readable FAIL lines (empty = OK)."""
    import numpy as np

    fails: list[str] = []
    required = [
        BDS,
        FACE,
        WORLD / "calibration_script.json",
        WORLD / "face_cell_timeline.npz",
        WORLD / "face_cell_timeline" / "meta.json",
        WORLD / "face_cell_timeline" / "speech_align.json",
        WORLD / "face_cell_timeline" / "look_drive.json",
        WORLD / "cosmetic_prefs.json",
        WORLD / "ml" / "l1_speech_clock.joblib",
        WORLD / "ml" / "l2_look_drive.joblib",
        WORLD / "ml" / "l3_face_motion.joblib",
        WORLD / "ml" / "l4_tick_codec.joblib",
        WORLD / "ml" / "l5_gap_prior.joblib",
    ]
    for path in required:
        if not path.is_file():
            fails.append(f"missing {path.relative_to(ROOT)}")

    if "tickfeed" not in str(FACE.resolve()).replace("\\", "/"):
        fails.append(f"identity path not tickfeed: {FACE}")

    meta_path = WORLD / "face_cell_timeline" / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not str(meta.get("schema", "")).startswith("aiface.face_cell_timeline"):
            fails.append(f"bad timeline schema {meta.get('schema')!r}")
        if "source_codes" not in meta and meta.get("schema") != "aiface.face_cell_timeline.v2":
            # v2 required for provenance-aware demo
            fails.append("timeline meta missing provenance (need v2 / source_codes)")
        print(
            f"  timeline: schema={meta.get('schema')} "
            f"measured={meta.get('n_measured')} synth={meta.get('n_synth')} "
            f"n={meta.get('n_ticks')}"
        )

    speech_path = WORLD / "face_cell_timeline" / "speech_align.json"
    if speech_path.is_file():
        speech = json.loads(speech_path.read_text(encoding="utf-8"))
        method = str(speech.get("method") or "")
        print(f"  speech_align: method={method}")
        if method not in {"audio_energy_force_align", "script_force_align"}:
            fails.append(f"unexpected speech_align method {method!r}")

    npz = WORLD / "face_cell_timeline.npz"
    if npz.is_file():
        data = np.load(npz)
        if "source" not in data.files:
            fails.append("face_cell_timeline.npz missing source[] provenance")
        else:
            src = data["source"]
            measured = int((src == 0).sum())
            print(f"  provenance: source=0 (measured) count={measured}/{len(src)}")

    print("  LOOK path: TickFeed labels (MouthLayerTimeline disabled when enabled)")
    print("  ring: local-ring produce=master (same tick); wire-loop uses RING_DEPTH lead")
    print("  transport: CHORUS lane A (c_t) + lane B (TickPackage frames/TPK_REF)")
    print("  wire-loop: opt-in (--wire-loop); default local-ring for FPS")

    catalog_path = WORLD / "expression_catalog.json"
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        rest_open = float((catalog.get("roles") or {}).get("rest", {}).get("mouth_open", 1.0))
        print(f"  identity rest mouth_open={rest_open:.3f}")
        if rest_open > 0.18:
            fails.append(
                f"identity rest mouth_open={rest_open:.3f} too high "
                "(source_face would look open at REST)"
            )
    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-chorus",
        action="store_true",
        help="Skip starting local CHORUS (spool fallback OK)",
    )
    parser.add_argument(
        "--wire-loop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Master consumes c_t from transport (proves bandwidth path). "
            "Off by default — CHORUS per-tick spam tanks FPS on lab GPUs"
        ),
    )
    parser.add_argument(
        "--wire-loop-source",
        choices=("code", "package"),
        default="code",
        help="Wire-loop feed: lane-A c_t (default) or lane-B package bytes",
    )
    parser.add_argument(
        "--speech-pace",
        type=float,
        default=float(os.environ.get("AIFACE_SPEECH_PACE", "0") or 0),
        help=(
            "Slow audio+visemes for clearer mouth (1.12=+12%%). "
            "0 = recipe default"
        ),
    )
    args = parser.parse_args()

    print("TickFeed demo preflight")
    print(f"  world: {WORLD}")
    fails = _preflight()
    if fails:
        print("FAIL: demo preflight", file=sys.stderr)
        for line in fails:
            print(f"  {line}", file=sys.stderr)
        print(
            "Run: python scripts/build_tickfeed_demo.py --clean",
            file=sys.stderr,
        )
        return 2

    if not args.no_chorus:
        try:
            _start_chorus()
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: CHORUS start failed ({exc}); continuing with spool")

    print("Launch TickFeed demo (new fidelity path)")
    print(f"  world: {BDS}")
    print(f"  face:  {FACE}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("CHORUS_DIM", "64")
    env.setdefault("AIFACE_CHORUS_CONTROL", "localhost:50051")
    env.setdefault("AIFACE_CHORUS_TARGET", "localhost:50053")
    cmd = [
        sys.executable,
        "-m",
        "aiface",
        "--demo",
        "--tts",
        "--gpu-log",
        "--bridge",
        "--bridge-direct-speak",
        "--tickfeed-debug",
        "--bridge-token",
        os.environ.get("AIFACE_BRIDGE_TOKEN", "tickfeed-lab"),
        "--bridge-port",
        os.environ.get("AIFACE_BRIDGE_PORT", "8766"),
        "--world",
        str(BDS),
        "--face-image",
        str(FACE),
        "--wire-loop-source",
        str(args.wire_loop_source),
    ]
    cmd.append("--wire-loop" if args.wire_loop else "--no-wire-loop")
    if float(args.speech_pace) > 0.0:
        cmd.extend(["--speech-pace", str(float(args.speech_pace))])
    print(
        f"  master={'wire-loop/' + args.wire_loop_source if args.wire_loop else 'local-ring'}"
    )
    if float(args.speech_pace) > 0.0:
        print(f"  speech_pace: {float(args.speech_pace):.3f}")
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
