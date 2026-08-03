"""Speak sentences via face bridge, capture previews, log Side-B LOOK vs FIELD.

Usage (demo must be running with --bridge --bridge-direct-speak):
  python scripts/calibrate_mouth_visibility.py
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "previews" / "mouth_calib"
TOKEN = "tickfeed-lab"
BASE = "http://127.0.0.1:8766"

SENTENCES = [
    ("rest", None),  # baseline capture only
    ("open_vowels", "Ah oh oo ee. Open wide now."),
    ("plosives", "Pop the big bubble. Put the book back."),
    ("natural", "Hello there, how are you doing today?"),
    ("count", "One two three four five six seven eight."),
]


def _req(method: str, path: str, body: dict | None = None) -> bytes:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _status() -> dict:
    return json.loads(_req("GET", "/status").decode("utf-8"))


def _wait_ready(timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            st = _status()
            if st.get("tick", 0) > 0:
                return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"bridge not ready: {last}")


def _wait_idle(timeout: float = 45.0) -> None:
    deadline = time.time() + timeout
    quiet = 0
    while time.time() < deadline:
        st = _status()
        speaking = bool(st.get("speaking")) or int(st.get("pending_visemes") or 0) > 0
        tf = st.get("tickfeed") or {}
        open_amt = float(tf.get("open") or 0.0)
        if not speaking and open_amt < 0.08:
            quiet += 1
            if quiet >= 6:
                return
        else:
            quiet = 0
        time.sleep(0.15)


def _wait_speech_active(timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = _status()
        tf = st.get("tickfeed") or {}
        if (
            bool(st.get("speaking"))
            or int(st.get("pending_visemes") or 0) > 0
            or float(tf.get("open") or 0.0) > 0.15
            or str(tf.get("viseme") or "REST") not in {"", "REST", "CLOSED"}
        ):
            return True
        time.sleep(0.08)
    return False


def capture_burst(tag: str, seconds: float = 2.2, hz: float = 8.0) -> list[dict]:
    rows: list[dict] = []
    folder = OUT / tag
    folder.mkdir(parents=True, exist_ok=True)
    n = max(1, int(seconds * hz))
    for i in range(n):
        st = _status()
        png = _req("GET", "/preview")
        path = folder / f"frame_{i:03d}.png"
        path.write_bytes(png)
        tf = st.get("tickfeed") or {}
        rows.append(
            {
                "i": i,
                "tick": st.get("tick"),
                "phoneme": st.get("phoneme"),
                "speaking": st.get("speaking"),
                "jaw": st.get("jaw_angle"),
                "muscles": st.get("active_muscles"),
                "mean_speed": st.get("mean_speed"),
                "peak_speed": st.get("peak_speed"),
                "active_cells": st.get("active_cells"),
                "open": tf.get("open"),
                "smile": tf.get("smile"),
                "plate_open": tf.get("plate_open"),
                "viseme": tf.get("viseme"),
                "path": str(path.relative_to(ROOT)),
            }
        )
        time.sleep(1.0 / hz)
    (folder / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def summarize(tag: str, rows: list[dict]) -> dict:
    opens = [float(r.get("open") or 0) for r in rows]
    speeds = [float(r.get("peak_speed") or 0) for r in rows]
    muscles = [int(r.get("muscles") or 0) for r in rows]
    return {
        "tag": tag,
        "n": len(rows),
        "open_max": max(opens) if opens else 0.0,
        "open_mean": sum(opens) / max(len(opens), 1),
        "peak_speed_max": max(speeds) if speeds else 0.0,
        "peak_speed_mean": sum(speeds) / max(len(speeds), 1),
        "muscles_max": max(muscles) if muscles else 0,
        "best_frame": max(rows, key=lambda r: float(r.get("open") or 0)).get("path")
        if rows
        else None,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Waiting for bridge {BASE} …")
    _wait_ready()
    print("Bridge OK", _status().get("avatar"))

    report = []
    for tag, text in SENTENCES:
        print(f"\n=== {tag} ===")
        if text:
            _wait_idle()
            _req("POST", "/speak", {"text": text})
            print("spoke:", text)
            if not _wait_speech_active():
                print("WARN: speech never became active")
            rows = capture_burst(tag, seconds=3.5, hz=6.0)
            _wait_idle()
        else:
            rows = capture_burst(tag, seconds=1.2, hz=5.0)
        summary = summarize(tag, rows)
        report.append(summary)
        print(json.dumps(summary, indent=2))

    out = OUT / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
