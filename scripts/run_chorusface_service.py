#!/usr/bin/env python3
"""Launch ChorusFace as a headless container/service face (web embed + PrismAPI).

Runs TickFeed world without a desktop window when the headless backend is
available (``MODERNGL_WINDOW=headless``). FaceBridge exposes ``/speak``,
``/prism/speak``, and ``/stream.mjpg`` for website embed.

See docs/FaceServiceEmbed.md.
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


def _preflight(world: Path) -> list[str]:
    fails: list[str] = []
    _, bds, face = _world_paths(world)
    required = [
        bds,
        face,
        world / "calibration_script.json",
        world / "face_cell_timeline.npz",
        world / "face_cell_timeline" / "meta.json",
        world / "ml" / "l3_face_motion.joblib",
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
        print(
            f"  timeline: schema={meta.get('schema')} n={meta.get('n_ticks')}"
        )
    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world",
        type=Path,
        default=Path(os.environ.get("CHORUSFACE_WORLD", str(DEFAULT_WORLD))),
    )
    parser.add_argument(
        "--window",
        default=os.environ.get("MODERNGL_WINDOW", "headless"),
        help="moderngl_window backend (default headless)",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Use a visible window instead of headless (local debug)",
    )
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--fidelity-hud", action="store_true")
    args = parser.parse_args()

    world, bds, face = _world_paths(Path(args.world))
    print("ChorusFace face service (container / embed)")
    print(f"  world: {world}")
    fails = _preflight(world)
    if fails:
        print("FAIL: service preflight", file=sys.stderr)
        for line in fails:
            print(f"  {line}", file=sys.stderr)
        return 2

    host = os.environ.get("CHORUSFACE_BRIDGE_HOST", "0.0.0.0")
    port = os.environ.get("CHORUSFACE_BRIDGE_PORT", "8766")
    token = os.environ.get("CHORUSFACE_BRIDGE_TOKEN", "chorusface-beta")
    cors = os.environ.get("CHORUSFACE_BRIDGE_CORS", "*")
    stream_fps = os.environ.get("CHORUSFACE_STREAM_FPS", "12")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["CHORUSFACE_PRODUCT_BETA"] = "1"
    env["CHORUSFACE_HEADLESS_SERVICE"] = "1"
    env["CHORUSFACE_BRIDGE_TOKEN"] = token
    env["CHORUSFACE_BRIDGE_PORT"] = str(port)
    env["CHORUSFACE_BRIDGE_HOST"] = host
    env["CHORUSFACE_BRIDGE_CORS"] = cors
    env["CHORUSFACE_STREAM_FPS"] = str(stream_fps)

    window = "pyglet" if args.visible else str(args.window or "headless")
    env["MODERNGL_WINDOW"] = window

    cmd = [
        sys.executable,
        "-m",
        "chorusface",
        "--window",
        window,
        "--product-beta",
        "--headless-service",
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
        "--allow-remote-bind",
        "--stream-fps",
        str(stream_fps),
        "--no-chat-box",
        "--world",
        str(bds),
        "--face-image",
        str(face),
        "--no-wire-loop",
    ]
    if not args.no_tts:
        cmd.append("--tts")
    if args.fidelity_hud:
        cmd.append("--fidelity-hud")

    url = f"http://{host}:{port}"
    print(f"  backend: {window}")
    print(f"  FaceBridge: {url}")
    print(f"  Authorization: Bearer {token}")
    print(f"  Embed: {url}/stream.mjpg?token={token}")
    print(f"  Prism speak: POST {url}/prism/speak")
    print("  Docs: docs/FaceServiceEmbed.md")
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
