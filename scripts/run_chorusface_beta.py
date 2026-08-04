#!/usr/bin/env python3
"""Launch the ChorusFace product beta (host-driven chat face).

Owns the TickFeed calibrated world + FaceBridge. The host owns the LLM **and**
TTS by default — push audio via ``/voice/*`` (see docs/ProductBeta.md,
docs/VoiceSync.md). Local ``--tts`` is lab-only when no host voice is available.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_WORLD = ROOT / "output" / "worlds" / "tickfeed"


def _world_paths(world: Path) -> tuple[Path, Path, Path]:
    world = world.resolve()
    return world, world / "avatar_face.bds", world / "source_face.png"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _preflight(world: Path) -> list[str]:
    """Return human-readable FAIL lines (empty = OK)."""
    fails: list[str] = []
    _, bds, face = _world_paths(world)
    required = [
        bds,
        face,
        world / "calibration_script.json",
        world / "face_cell_timeline.npz",
        world / "face_cell_timeline" / "meta.json",
        world / "face_cell_timeline" / "speech_align.json",
        world / "face_cell_timeline" / "look_drive.json",
        world / "cosmetic_prefs.json",
        world / "ml" / "l1_speech_clock.joblib",
        world / "ml" / "l2_look_drive.joblib",
        world / "ml" / "l3_face_motion.joblib",
        world / "ml" / "l4_tick_codec.joblib",
        world / "ml" / "l5_gap_prior.joblib",
        world / "plate_atlas.json",
    ]
    for path in required:
        if not path.is_file():
            try:
                rel = path.relative_to(ROOT)
            except ValueError:
                rel = path
            fails.append(f"missing {rel}")

    meta_path = world / "face_cell_timeline" / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not str(meta.get("schema", "")).startswith("chorusface.face_cell_timeline"):
            fails.append(f"bad timeline schema {meta.get('schema')!r}")
        print(
            f"  timeline: schema={meta.get('schema')} "
            f"measured={meta.get('n_measured')} n={meta.get('n_ticks')}"
        )
    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world",
        type=Path,
        default=Path(os.environ.get("CHORUSFACE_WORLD", str(DEFAULT_WORLD))),
        help="Calibrated world dir (default: CHORUSFACE_WORLD or output/worlds/tickfeed)",
    )
    parser.add_argument(
        "--fidelity-hud",
        action="store_true",
        help="Show fidelity HUD overlay",
    )
    parser.add_argument(
        "--allow-remote-bind",
        action="store_true",
        help="Bind FaceBridge off loopback (LAN kiosk; pair with CHORUSFACE_BRIDGE_HOST)",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help=(
            "Lab only: enable local ChorusFace TTS. Product default is host-owned "
            "voice via /voice/* (also CHORUSFACE_TTS=1)."
        ),
    )
    args = parser.parse_args()

    world, bds, face = _world_paths(Path(args.world))
    print("ChorusFace product beta")
    print(f"  world: {world}")
    fails = _preflight(world)
    if fails:
        print("FAIL: beta preflight", file=sys.stderr)
        for line in fails:
            print(f"  {line}", file=sys.stderr)
        print(
            "Build the TickFeed world first: python scripts/build_tickfeed_demo.py --clean",
            file=sys.stderr,
        )
        return 2

    host = os.environ.get("CHORUSFACE_BRIDGE_HOST", "127.0.0.1")
    port = os.environ.get("CHORUSFACE_BRIDGE_PORT", "8766")
    token = os.environ.get("CHORUSFACE_BRIDGE_TOKEN", "chorusface-beta")
    cors = os.environ.get("CHORUSFACE_BRIDGE_CORS", "*")
    use_local_tts = bool(args.tts) or _env_flag("CHORUSFACE_TTS")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["CHORUSFACE_PRODUCT_BETA"] = "1"
    env["CHORUSFACE_BRIDGE_TOKEN"] = token
    env["CHORUSFACE_BRIDGE_PORT"] = str(port)
    env["CHORUSFACE_BRIDGE_HOST"] = host
    env["CHORUSFACE_BRIDGE_CORS"] = cors

    cmd = [
        sys.executable,
        "-m",
        "chorusface",
        "--product-beta",
        "--bridge",
        "--bridge-direct-speak",
        "--bridge-token",
        token,
        "--bridge-host",
        host,
        "--bridge-port",
        str(port),
        "--bridge-cors",
        cors,
        "--world",
        str(bds),
        "--face-image",
        str(face),
        "--no-wire-loop",
    ]
    if use_local_tts:
        cmd.append("--tts")
    if args.allow_remote_bind or host not in {"127.0.0.1", "localhost", "::1"}:
        cmd.append("--allow-remote-bind")
    if args.fidelity_hud:
        cmd.append("--fidelity-hud")

    url = f"http://{host}:{port}"
    print(f"  FaceBridge: {url}")
    print(f"  Authorization: Bearer {token}")
    print("  Voice (default): host TTS → POST /voice/expect|/pcm|/end  or /voice/timeline")
    print(f'  Mouth cue only: POST {url}/speak  {{"text":"..."}}  (no ChorusFace audio)')
    if use_local_tts:
        print("  Local TTS: ON (lab) — ChorusFace will synthesize audio")
    else:
        print("  Local TTS: OFF (product default — host owns the voice)")
    print("  Avatar: fixed TickFeed world (not changeable in this beta)")
    print("  Window: resizable, aspect locked — see docs/ProductBeta.md")
    print("  Smoke (host voice): python -m chorusface.host_client --voice \"Hello there\"")
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
