"""Drive the live GPU avatar with VowelDesign utterances (FaceBridge).

Requires ``python scripts/run_chorusface_beta.py`` already running.

Usage:
  python scripts/demo_vowel_avatar.py
  python scripts/demo_vowel_avatar.py --only angry_vowels
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8766"
DEFAULT_TOKEN = "chorusface-beta"
DEFAULT_CLIENT = "vowel-demo-client"

DEMOS = [
    {
        "id": "vowels",
        "label": "HAPPY EE/OU/AA",
        "payload": {
            "utterance_id": "live_vowels",
            "text": "see you ah",
            "play": True,
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 2.0}],
            "spans": [
                {"tag": "EE", "start_s": 0.05, "end_s": 0.45},
                {"tag": "OU", "start_s": 0.55, "end_s": 0.95},
                {"tag": "AA", "start_s": 1.10, "end_s": 1.60},
            ],
        },
        "wait_s": 2.2,
    },
    {
        "id": "angry_vowels",
        "label": "ANGRY EE/OU/AA",
        "payload": {
            "utterance_id": "live_angry_vowels",
            "text": "see you ah",
            "play": True,
            "emotion_track": [{"emotion": "ANGRY", "start_s": 0.0, "end_s": 2.0}],
            "spans": [
                {"tag": "EE", "start_s": 0.05, "end_s": 0.45},
                {"tag": "OU", "start_s": 0.55, "end_s": 0.95},
                {"tag": "AA", "start_s": 1.10, "end_s": 1.60},
            ],
        },
        "wait_s": 2.2,
    },
    {
        "id": "sad",
        "label": "SAD sentence",
        "payload": {
            "utterance_id": "live_sad",
            "text": "I miss you so much",
            "play": True,
            "emotion_track": [{"emotion": "SAD", "start_s": 0.0, "end_s": 2.8}],
        },
        "wait_s": 3.0,
    },
    {
        "id": "happy",
        "label": "HAPPY sentence",
        "payload": {
            "utterance_id": "live_happy",
            "text": "See you tomorrow",
            "play": True,
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 2.5}],
        },
        "wait_s": 2.8,
    },
]


def _headers(token: str, client_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "X-ChorusFace-Client-Id": client_id,
    }


def post_json(url: str, token: str, client_id: str, path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers=_headers(token, client_id),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def activate(url: str, token: str, client_id: str) -> None:
    post_json(
        url,
        token,
        client_id,
        "/auth/activate",
        {"client_id": client_id},
    )


def wait_bridge(
    url: str, token: str, client_id: str, timeout_s: float = 90.0
) -> None:
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
                    print(f"bridge ready: {url}")
                    return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(1.0)
    raise SystemExit(f"bridge not ready after {timeout_s:.0f}s: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument("--client-id", default=DEFAULT_CLIENT)
    ap.add_argument("--only", default="", help="demo id filter")
    ap.add_argument("--wait-bridge", type=float, default=90.0)
    args = ap.parse_args()
    client_id = args.client_id or str(uuid.uuid4())

    wait_bridge(args.url, args.token, client_id, args.wait_bridge)
    try:
        activate(args.url, args.token, client_id)
        print(f"activated client_id={client_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"activate warn: {exc}")

    selected = [d for d in DEMOS if not args.only or d["id"] == args.only]
    if not selected:
        print(f"no demos match --only={args.only!r}", file=sys.stderr)
        return 2

    for demo in selected:
        print(f"\n>>> {demo['label']} ({demo['id']})")
        try:
            reply = post_json(
                args.url,
                args.token,
                client_id,
                "/vowel/utterance",
                demo["payload"],
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"FAIL HTTP {exc.code}: {body}", file=sys.stderr)
            return 1
        print(json.dumps(reply, indent=2)[:800])
        time.sleep(float(demo["wait_s"]))
    print("\nDone — watch the GPU avatar window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
