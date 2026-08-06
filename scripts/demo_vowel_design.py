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
DEFAULT_TOKEN = "chorusface-beta"
DEFAULT_CLIENT = "vowel-demo-client"
DEFAULT_PORT = 8766
WORLD_BDS = ROOT / "output" / "worlds" / "tickfeed" / "avatar_face.bds"
WORLD_FACE = ROOT / "output" / "worlds" / "tickfeed" / "source_face.png"

# Long holds so a human can read lip silhouette on the photo face.
SHOW: list[dict] = [
    {
        "id": "contrast_happy",
        "title": "1/4  HAPPY — watch EE (wide) → OU (round) → AA (open)",
        "wait_s": 4.5,
        "payload": {
            "utterance_id": "vd_happy_contrast",
            "text": "EE OU AA",
            "play": True,
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 4.2}],
            "spans": [
                {"tag": "EE", "start_s": 0.20, "end_s": 1.20},
                {"tag": "OU", "start_s": 1.50, "end_s": 2.50},
                {"tag": "AA", "start_s": 2.80, "end_s": 4.00},
            ],
        },
    },
    {
        "id": "contrast_angry",
        "title": "2/4  ANGRY — same EE / OU / AA under furious brows",
        "wait_s": 4.5,
        "payload": {
            "utterance_id": "vd_angry_contrast",
            "text": "EE OU AA",
            "play": True,
            "emotion_track": [{"emotion": "ANGRY", "start_s": 0.0, "end_s": 4.2}],
            "spans": [
                {"tag": "EE", "start_s": 0.20, "end_s": 1.20},
                {"tag": "OU", "start_s": 1.50, "end_s": 2.50},
                {"tag": "AA", "start_s": 2.80, "end_s": 4.00},
            ],
        },
    },
    {
        "id": "contrast_sad",
        "title": "3/4  SAD — EE / OU / AA with sorrow eyes/brows",
        "wait_s": 4.5,
        "payload": {
            "utterance_id": "vd_sad_contrast",
            "text": "EE OU AA",
            "play": True,
            "emotion_track": [{"emotion": "SAD", "start_s": 0.0, "end_s": 4.2}],
            "spans": [
                {"tag": "EE", "start_s": 0.20, "end_s": 1.20},
                {"tag": "OU", "start_s": 1.50, "end_s": 2.50},
                {"tag": "AA", "start_s": 2.80, "end_s": 4.00},
            ],
        },
    },
    {
        "id": "sentence",
        "title": "4/4  HAPPY sentence — See you tomorrow (G2P GA-16)",
        "wait_s": 3.5,
        "payload": {
            "utterance_id": "vd_happy_sentence",
            "text": "See you tomorrow",
            "play": True,
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 3.0}],
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


def launch_avatar(port: int, token: str) -> subprocess.Popen:
    if not WORLD_BDS.is_file() or not WORLD_FACE.is_file():
        raise SystemExit(
            f"missing world assets:\n  {WORLD_BDS}\n  {WORLD_FACE}\n"
            "Build TickFeed world first."
        )
    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["CHORUSFACE_VOWEL_DESIGN"] = "1"
    cmd = [
        py,
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
        "--no-chat",
        "--fidelity-hud",
    ]
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
    print("You should see wide EE, round OU, open AA — not plate jaw-pump.")
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
    print("Done. Window stays open — re-run with --no-launch to replay.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument("--client-id", default=DEFAULT_CLIENT)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-launch", action="store_true", help="Use existing avatar")
    ap.add_argument("--only", default="", help="Scene id filter")
    ap.add_argument("--wait-bridge", type=float, default=90.0)
    args = ap.parse_args()
    client_id = args.client_id or str(uuid.uuid4())
    url = args.url or f"http://127.0.0.1:{args.port}"

    proc: subprocess.Popen | None = None
    if not args.no_launch:
        print(f"[prep] free port {args.port}")
        free_port(args.port)
        proc = launch_avatar(args.port, args.token)
        print(f"[ok] avatar pid={proc.pid}")

    try:
        wait_bridge(url, args.token, client_id, args.wait_bridge)
        return run_show(url, args.token, client_id, args.only)
    finally:
        if proc is not None and proc.poll() is not None:
            print(f"[warn] avatar exited early code={proc.returncode}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
