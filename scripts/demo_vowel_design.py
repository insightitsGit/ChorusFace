#!/usr/bin/env python3
"""Complete VowelDesign GPU demo — biomech GA-16, not TickFeed plates.

Launches a dedicated ChorusFace window with ``--vowel-design`` (TickFeed LOOK
disabled; speech = BiomechanicalFace muscles + jaw), then drives clear
EE / OU / AA contrasts under several emotions.

Usage:
  python scripts/demo_vowel_design.py
  python scripts/demo_vowel_design.py --no-launch   # avatar already running
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8766"
DEFAULT_CLIENT = "vowel-demo-client"
DEFAULT_PORT = 8766
WORLD_BDS = ROOT / "output" / "worlds" / "tickfeed" / "avatar_face.bds"
WORLD_FACE = ROOT / "output" / "worlds" / "tickfeed" / "source_face.png"


def _default_token() -> str:
    """Prefer local handoff vault primary key over the obsolete beta stub."""
    handoff = ROOT / "secrets" / "api_keys.handoff.local.txt"
    if handoff.is_file():
        for line in handoff.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3 and parts[-1].strip():
                return parts[-1].strip()
    return "chorusface-beta"


DEFAULT_TOKEN = _default_token()

# Long holds so lip silhouettes read clearly (NEUTRAL first — no HAPPY grin mask).
_BLINK = {"blinks": True, "blink_interval_s": 1.6, "blink_seed": 7}
SHOW: list[dict] = [
    {
        "id": "contrast_neutral",
        "title": "1/5  NEUTRAL — wide EE → round OU → open AA (watch lips)",
        "wait_s": 5.8,
        "payload": {
            "utterance_id": "vd_neutral_contrast",
            "text": "EE OU AA",
            "play": True,
            **_BLINK,
            "emotion_track": [{"emotion": "NEUTRAL", "start_s": 0.0, "end_s": 5.4}],
            "spans": [
                {"tag": "EE", "start_s": 0.20, "end_s": 1.60},
                {"tag": "OU", "start_s": 1.90, "end_s": 3.40},
                {"tag": "AA", "start_s": 3.70, "end_s": 5.20},
            ],
        },
    },
    {
        "id": "contrast_angry",
        "title": "2/5  ANGRY — EE/OU/AA under knit brows",
        "wait_s": 5.2,
        "payload": {
            "utterance_id": "vd_angry_contrast",
            "text": "EE OU AA",
            "play": True,
            **{**_BLINK, "blink_seed": 11},
            "emotion_track": [{"emotion": "ANGRY", "start_s": 0.0, "end_s": 4.8}],
            "spans": [
                {"tag": "EE", "start_s": 0.20, "end_s": 1.40},
                {"tag": "OU", "start_s": 1.70, "end_s": 3.00},
                {"tag": "AA", "start_s": 3.30, "end_s": 4.60},
            ],
        },
    },
    {
        "id": "contrast_happy",
        "title": "3/5  HAPPY — same vowels + raised brows (oral still biomech)",
        "wait_s": 5.2,
        "payload": {
            "utterance_id": "vd_happy_contrast",
            "text": "EE OU AA",
            "play": True,
            **{**_BLINK, "blink_seed": 13},
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 4.8}],
            "spans": [
                {"tag": "EE", "start_s": 0.20, "end_s": 1.40},
                {"tag": "OU", "start_s": 1.70, "end_s": 3.00},
                {"tag": "AA", "start_s": 3.30, "end_s": 4.60},
            ],
        },
    },
    {
        "id": "blink_hold",
        "title": "4/5  NEUTRAL blink hold — EyeSystem (F9 schedule requests)",
        "wait_s": 3.8,
        "payload": {
            "utterance_id": "vd_blink_hold",
            "text": "AX",
            "play": True,
            "blinks": True,
            "blink_interval_s": 1.1,
            "blink_seed": 3,
            "emotion_track": [{"emotion": "NEUTRAL", "start_s": 0.0, "end_s": 3.4}],
            "spans": [
                {"tag": "AX", "start_s": 0.10, "end_s": 3.20},
            ],
        },
    },
    {
        "id": "sentence",
        "title": "5/5  Chat-style sentence — See you tomorrow",
        "wait_s": 3.8,
        "payload": {
            "utterance_id": "vd_happy_sentence",
            "text": "See you tomorrow",
            "play": True,
            **{**_BLINK, "blink_seed": 17},
            "emotion_track": [{"emotion": "NEUTRAL", "start_s": 0.0, "end_s": 3.4}],
            "spans": [
                {"tag": "EE", "start_s": 0.15, "end_s": 0.55},
                {"tag": "OU", "start_s": 0.70, "end_s": 1.10},
                {"tag": "AX", "start_s": 1.25, "end_s": 1.55},
                {"tag": "AA", "start_s": 1.70, "end_s": 2.20},
                {"tag": "OH", "start_s": 2.35, "end_s": 3.10},
            ],
        },
    },
]


def _headers(token: str, client_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "X-ChorusFace-Client-Id": client_id,
    }


def post_json(
    url: str,
    token: str,
    client_id: str,
    path: str,
    body: dict,
    *,
    timeout_s: float = 60.0,
) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers=_headers(token, client_id),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_bridge(url: str, token: str, client_id: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                url.rstrip("/") + "/health",
                headers=_headers(token, client_id),
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[ok] bridge ready {url}")
                    return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(0.5)
    raise SystemExit(f"bridge not ready: {last}")


def free_port(port: int) -> None:
    if sys.platform != "win32":
        return
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    f"$p=Get-NetTCPConnection -LocalPort {port} -State Listen "
                    "-ErrorAction SilentlyContinue | "
                    "Select-Object -ExpandProperty OwningProcess -Unique; "
                    "foreach($i in $p){ if($i){ Stop-Process -Id $i -Force "
                    "-ErrorAction SilentlyContinue } }"
                ),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        del out
    except Exception:  # noqa: BLE001
        pass
    time.sleep(1.0)


def launch_avatar(
    port: int,
    token: str,
    *,
    capture_dir: Path | None = None,
    capture_frames: int = 90,
) -> subprocess.Popen:
    if not WORLD_BDS.is_file() or not WORLD_FACE.is_file():
        raise SystemExit(
            f"missing world assets:\n  {WORLD_BDS}\n  {WORLD_FACE}\n"
            "Build TickFeed world first."
        )
    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["CHORUSFACE_VOWEL_DESIGN"] = "1"
    # Avoid OpenBLAS/OMP deadlocks between HTTP bridge threads and the GL loop.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    cmd = [
        py,
        "-u",
        "-m",
        "chorusface",
        "--vowel-design",
        "--bridge",
        "--bridge-direct-speak",
        "--bridge-token",
        token,
        "--bridge-host",
        "127.0.0.1",
        "--bridge-port",
        str(port),
        "--bridge-cors",
        "*",
        "--world",
        str(WORLD_BDS),
        "--face-image",
        str(WORLD_FACE),
        "--no-wire-loop",
        "--fidelity-hud",
    ]
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(
            [
                "--capture",
                str(capture_dir),
                "--capture-frames",
                str(max(1, int(capture_frames))),
            ]
        )
    log_path = ROOT / "output" / "teacher" / "vowel_design_demo.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")
    print("[launch]", " ".join(cmd))
    print(f"[log] {log_path}")
    # Detach stdio so the GL window does not share this console (avoids
    # interleaved logs / accidental Ctrl+C killing the avatar mid-POST).
    creation = 0
    if sys.platform == "win32":
        creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        creationflags=creation,
        close_fds=True,
    )


def run_show(url: str, token: str, client_id: str, only: str) -> int:
    try:
        post_json(url, token, client_id, "/auth/activate", {"client_id": client_id})
        print(f"[ok] activated client_id={client_id}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        # Same client already holds the exclusive lease — continue.
        if exc.code == 403 and client_id[:12] in body:
            print(f"[ok] lease already held by {client_id}")
        elif exc.code == 403:
            # Steal: release unknown holder then activate (demo only).
            print(f"[warn] activate 403: {body[:160]}")
            print("[warn] retrying with fresh client_id")
            client_id = f"vowel-design-{uuid.uuid4().hex[:8]}"
            # Force new token binding isn't available without server restart;
            # kill+relaunch is handled by caller. Fail clearly.
            raise SystemExit(
                "FaceBridge API key is leased to another client_id. "
                "Re-run without --no-launch (script frees the port / restarts avatar), "
                f"or pass --client-id matching the holder. Detail: {body}"
            ) from exc
        else:
            raise
    print()
    print("=" * 60)
    print("WATCH THE GPU WINDOW:  ChorusFace — VowelDesign (GA-16 biomech)")
    print("You should see wide EE, round OU, open AA — plates OFF (FIDELITY plate=0).")
    print("After the auto scenes: click the chat panel, Esc to focus, type + Enter.")
    print("Try:  EE OU AA   or   hello how are you")
    print("=" * 60)

    selected = [s for s in SHOW if not only or s["id"] == only]
    if not selected:
        print(f"no scenes match --only={only!r}", file=sys.stderr)
        return 2

    for scene in selected:
        print()
        print(">>>", scene["title"])
        try:
            reply = post_json(
                url, token, client_id, "/vowel/utterance", scene["payload"]
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"FAIL HTTP {exc.code}: {body}", file=sys.stderr)
            return 1
        tags = []
        spans = scene["payload"].get("spans") or []
        if spans:
            tags = [s["tag"] for s in spans]
        print(
            f"    scheduled={reply.get('scheduled')} "
            f"n_ticks={reply.get('n_ticks')} "
            f"emotion={reply.get('primary_emotion')} "
            f"tags={tags or '(g2p)'}"
        )
        # Confirm server returned biomech look when app supports it.
        if reply.get("look"):
            print(f"    look={reply.get('look')} drive={reply.get('drive')}")
        time.sleep(float(scene["wait_s"]))

    print()
    print("Done with auto scenes. Window stays open — chat in the panel under the face.")
    print("Re-run with --no-launch to replay scenes against the same window.")
    return 0


def main() -> int:
    # Unbuffered console so progress appears while the GPU window runs.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument("--client-id", default=DEFAULT_CLIENT)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-launch", action="store_true", help="Use existing avatar")
    ap.add_argument("--only", default="", help="Scene id filter")
    ap.add_argument(
        "--chat-only",
        action="store_true",
        help="Launch avatar and skip auto scenes (chat in the window)",
    )
    ap.add_argument("--wait-bridge", type=float, default=90.0)
    ap.add_argument(
        "--capture",
        type=Path,
        default=None,
        help="Write GPU frames under this directory while demo runs",
    )
    ap.add_argument("--capture-frames", type=int, default=120)
    args = ap.parse_args()
    client_id = args.client_id or str(uuid.uuid4())
    url = args.url or f"http://127.0.0.1:{args.port}"

    proc: subprocess.Popen | None = None
    if not args.no_launch:
        print(f"[prep] free port {args.port}", flush=True)
        free_port(args.port)
        proc = launch_avatar(
            args.port,
            args.token,
            capture_dir=args.capture,
            capture_frames=args.capture_frames,
        )
        print(f"[ok] avatar pid={proc.pid}", flush=True)

    try:
        wait_bridge(url, args.token, client_id, args.wait_bridge)
        if args.chat_only:
            try:
                post_json(
                    url, args.token, client_id, "/auth/activate", {"client_id": client_id}
                )
            except urllib.error.HTTPError:
                pass
            print()
            print("=" * 60)
            print("Chat-only: focus the GPU window chat panel (Esc), type, Enter.")
            print("Typed lines drive GA-16 biomech + Model A/B immediately.")
            print("=" * 60)
            return 0
        return run_show(url, args.token, client_id, args.only)
    finally:
        if proc is not None and proc.poll() is not None:
            print(f"[warn] avatar exited early code={proc.returncode}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
