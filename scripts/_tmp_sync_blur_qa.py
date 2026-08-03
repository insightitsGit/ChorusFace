#!/usr/bin/env python3
"""Phrase QA: word-sync (closure hits) + mid-open FIELD mute (blur proxy)."""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8766"
TOKEN = "tickfeed-lab"
OUT = Path("output/previews/sync_blur_qa")

PHRASES = [
    ("peter", "Peter Piper picked a peck of pickled peppers."),
    ("mmm", "Mmm. Closed lips. Mm mm."),
    ("open", "Ah oh oo ee. Open wide now."),
    ("hello", "Hello there friend, how are you today?"),
]


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
    cal = json.loads(
        req(
            "POST",
            "/calibrate",
            {"mode": "normal", "speech_pace": 1.0, "viseme_min_hold": 0.0},
        ).decode()
    )
    print("CAL", cal)

    summary: list[dict] = []
    for tag, text in PHRASES:
        req("POST", "/speak", {"text": text})
        t0 = time.perf_counter()
        deadline = time.time() + 14.0
        samples: list[dict] = []
        mid_gain: list[float] = []
        closure_hits = 0
        open_hits = 0
        viseme_seq: list[str] = []
        last_v = ""
        while time.time() < deadline:
            st = status()
            tf = st.get("tickfeed") or {}
            open_v = float(tf.get("open") or 0.0)
            gain = float(tf.get("field_gain_eff") or 0.0)
            vis = str(tf.get("viseme") or "")
            samples.append(
                {
                    "t": round(time.perf_counter() - t0, 3),
                    "open": open_v,
                    "smile": float(tf.get("smile") or 0.0),
                    "gain": gain,
                    "viseme": vis,
                    "presence": tf.get("presence"),
                }
            )
            if vis and vis != last_v:
                viseme_seq.append(vis)
                last_v = vis
            if vis in {"PP", "MM", "CLOSED"} and open_v < 0.12:
                closure_hits += 1
            if open_v >= 0.35:
                open_hits += 1
            if 0.15 <= open_v <= 0.55:
                mid_gain.append(gain)
            if (
                not st.get("speaking")
                and open_v < 0.05
                and (time.perf_counter() - t0) > 1.0
                and len(samples) > 8
            ):
                break
            time.sleep(0.05)
        # settle idle
        idle_deadline = time.time() + 3.0
        while time.time() < idle_deadline:
            st = status()
            tf = st.get("tickfeed") or {}
            if float(tf.get("open") or 0.0) < 0.05 and not st.get("speaking"):
                break
            time.sleep(0.08)
        (OUT / f"{tag}_idle.png").write_bytes(req("GET", "/preview"))
        row = {
            "tag": tag,
            "text": text,
            "closure_hits": closure_hits,
            "open_hits": open_hits,
            "mid_gain_mean": (
                round(sum(mid_gain) / len(mid_gain), 4) if mid_gain else None
            ),
            "mid_gain_max": round(max(mid_gain), 4) if mid_gain else None,
            "viseme_seq": viseme_seq[:40],
            "idle_open": float((status().get("tickfeed") or {}).get("open") or 0.0),
            "wall_s": round(time.perf_counter() - t0, 2),
            "samples": len(samples),
        }
        summary.append(row)
        print(tag, {k: row[k] for k in row if k != "viseme_seq"}, "seq", viseme_seq[:20])
        (OUT / f"{tag}_trace.json").write_text(
            json.dumps(samples, indent=2), encoding="utf-8"
        )

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
