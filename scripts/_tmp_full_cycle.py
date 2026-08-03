#!/usr/bin/env python3
"""Full-cycle TickFeed speak/status/preview capture for mouth QA."""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8766"
TOKEN = "tickfeed-lab"
OUT = Path("output/previews/full_cycle")


def req(method: str, path: str, body: dict | None = None, timeout: float = 60) -> bytes:
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


def status() -> dict:
    return json.loads(req("GET", "/status").decode())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.glob("*"):
        path.unlink()

    st = status()
    tf = st.get("tickfeed") or {}
    print(
        "START",
        json.dumps(
            {
                "tick": st.get("tick"),
                "fps": st.get("fps"),
                "speaking": st.get("speaking"),
                "pace": tf.get("speech_pace"),
                "open": tf.get("open"),
                "smile": tf.get("smile"),
                "viseme": tf.get("viseme"),
                "gain": tf.get("field_gain_eff"),
                "presence": tf.get("presence"),
            },
            indent=2,
        ),
    )
    (OUT / "00_rest.png").write_bytes(req("GET", "/preview"))
    cal = json.loads(
        req("POST", "/calibrate", {"mode": "normal", "speech_pace": 1.0}).decode()
    )
    print("CAL", cal)

    phrases = [
        ("01_ah", "Ah oh oo ee. Open wide now."),
        ("02_hello", "Hello there friend, how are you today?"),
        ("03_closed", "Mmm. Closed lips. Mm mm."),
    ]
    summary: list[dict] = []
    for tag, text in phrases:
        t0 = time.perf_counter()
        req("POST", "/speak", {"text": text})
        deadline = time.time() + 12
        frames = 0
        open_max = 0.0
        smile_max = 0.0
        gain_at_open = None
        while time.time() < deadline:
            st = status()
            tf = st.get("tickfeed") or {}
            open_v = float(tf.get("open") or 0.0)
            smile_v = float(tf.get("smile") or 0.0)
            open_max = max(open_max, open_v)
            smile_max = max(smile_max, smile_v)
            if open_v > 0.35 and frames < 8:
                gain = float(tf.get("field_gain_eff") or 0.0)
                png = req("GET", "/preview")
                name = (
                    f"{tag}_f{frames}_o{open_v:.2f}_s{smile_v:.2f}_g{gain:.2f}.png"
                )
                (OUT / name).write_bytes(png)
                if gain_at_open is None:
                    gain_at_open = gain
                frames += 1
            if frames >= 6 and open_v < 0.05 and (time.perf_counter() - t0) > 1.2:
                break
            time.sleep(0.07)
        # Wait for plate release so idle isn't a mid-hysteresis open frame.
        idle_deadline = time.time() + 4.0
        while time.time() < idle_deadline:
            st = status()
            tf = st.get("tickfeed") or {}
            if float(tf.get("open") or 0.0) < 0.05 and not st.get("speaking"):
                break
            time.sleep(0.1)
        st = status()
        tf = st.get("tickfeed") or {}
        (OUT / f"{tag}_idle.png").write_bytes(req("GET", "/preview"))
        row = {
            "tag": tag,
            "text": text,
            "open_max": open_max,
            "smile_max": smile_max,
            "frames": frames,
            "gain_at_open": gain_at_open,
            "idle_open": float(tf.get("open") or 0.0),
            "idle_smile": float(tf.get("smile") or 0.0),
            "pace": tf.get("speech_pace"),
            "wall_s": round(time.perf_counter() - t0, 2),
        }
        summary.append(row)
        print(tag, row)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", OUT)
    print("files", sorted(p.name for p in OUT.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
