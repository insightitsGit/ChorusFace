#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _bridge_token import bridge_token

BASE = "http://127.0.0.1:8766"
TOKEN = bridge_token()
OUT = ROOT / "output" / "previews" / "mouth_review"
OUT.mkdir(parents=True, exist_ok=True)


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
    req(
        "POST",
        "/voice/timeline",
        {
            "spans": [
                {"phoneme": "EH", "start": 0.0, "end": 0.7},
                {"phoneme": "AA", "start": 0.7, "end": 1.5},
                {"phoneme": "PP", "start": 1.55, "end": 1.75},
                {"phoneme": "OU", "start": 1.8, "end": 2.5},
                {"phoneme": "REST", "start": 2.55, "end": 2.9},
            ],
            "caption": "eh aa pp ou",
        },
    )
    for i in range(70):
        time.sleep(0.04)
        st = json.loads(req("GET", "/status").decode())
        tf = st.get("tickfeed") or {}
        own = st.get("mouth_ownership") or {}
        ph = st.get("phoneme")
        po = float(tf.get("plate_open") or 0.0)
        jaw = float(st.get("jaw_angle") or 0.0)
        if i in {5, 12, 20, 30, 40, 50, 60}:
            raw = req("GET", "/preview")
            name = f"hold_{i:02d}_{ph}"
            (OUT / f"{name}.png").write_bytes(raw)
            im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                cv2.imwrite(str(OUT / f"crop_{name}.png"), im[480:860, 220:800])
            print(
                f"t={i * 0.04:.2f} ph={ph} open={po:.2f} jaw={jaw:.3f} "
                f"field={tf.get('field_gain_eff')} atlas={own.get('plate_atlas')} "
                f"amt={own.get('plate_amount')} owners={own.get('owners')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
