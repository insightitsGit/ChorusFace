#!/usr/bin/env python3
"""Quick mouth-motion QA against the running TickFeed bridge."""

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
from _bridge_token import bridge_token  # noqa: E402

BASE = "http://127.0.0.1:8766"
TOKEN = bridge_token()
OUT = ROOT / "output" / "previews" / "mouth_review"
OUT.mkdir(parents=True, exist_ok=True)


def req(method: str, path: str, body=None, timeout: float = 30) -> bytes:
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
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read()


def main() -> int:
    (OUT / "01_rest.png").write_bytes(req("GET", "/preview"))
    spans = [
        {"phoneme": "EH", "start": 0.00, "end": 0.18},
        {"phoneme": "OU", "start": 0.18, "end": 0.40},
        {"phoneme": "PP", "start": 0.48, "end": 0.62},
        {"phoneme": "AA", "start": 0.65, "end": 0.95},
        {"phoneme": "TH", "start": 1.00, "end": 1.14},
        {"phoneme": "EH", "start": 1.14, "end": 1.30},
        {"phoneme": "REST", "start": 1.40, "end": 1.70},
    ]
    req("POST", "/voice/timeline", {"spans": spans, "caption": "hello there"})
    best = None
    for i in range(48):
        time.sleep(0.04)
        st = json.loads(req("GET", "/status").decode())
        tf = st.get("tickfeed") or {}
        own = st.get("mouth_ownership") or {}
        row = {
            "t": i * 0.04,
            "phoneme": st.get("phoneme"),
            "jaw": float(st.get("jaw_angle") or 0.0),
            "plate_open": float(tf.get("plate_open") or 0.0),
            "field_gain": tf.get("field_gain_eff"),
            "atlas": own.get("plate_atlas"),
            "amount": own.get("plate_amount"),
            "owners": own.get("owners"),
        }
        score = row["plate_open"] + row["jaw"]
        if best is None or score > best[0]:
            best = (score, row)
        if i in {4, 8, 14, 18, 24, 30}:
            raw = req("GET", "/preview")
            name = f"t{i:02d}_{row['phoneme']}"
            (OUT / f"{name}.png").write_bytes(raw)
            im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                cv2.imwrite(str(OUT / f"crop_{name}.png"), im[480:860, 220:800])
        if row["plate_open"] > 0.2:
            print(
                f"t={row['t']:.2f} ph={row['phoneme']} open={row['plate_open']:.2f} "
                f"jaw={row['jaw']:.3f} field={row['field_gain']} "
                f"atlas={row['atlas']} amt={row['amount']}"
            )
    print("BEST", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
