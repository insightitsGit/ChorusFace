#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _bridge_token import bridge_token

BASE = "http://127.0.0.1:8766"
TOKEN = bridge_token()


def req(method: str, path: str, body=None) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as resp:
        return resp.read()


def main() -> int:
    raw = req(
        "POST",
        "/voice/timeline",
        {
            "spans": [
                {"phoneme": "AH", "start": 0.0, "end": 2.0},
                {"phoneme": "REST", "start": 2.0, "end": 2.3},
            ],
            "caption": "hold ah",
        },
    )
    print("SCHEDULE", raw.decode())
    t0 = time.perf_counter()
    for i in range(28):
        time.sleep(0.1)
        st = json.loads(req("GET", "/status").decode())
        tf = st.get("tickfeed") or {}
        print(
            f"wall={time.perf_counter() - t0:0.2f}s ph={st.get('phoneme')} "
            f"open={float(tf.get('plate_open') or 0):.2f} "
            f"pending={st.get('pending_visemes')} "
            f"jaw={float(st.get('jaw_angle') or 0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
