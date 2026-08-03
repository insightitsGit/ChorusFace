"""Speak a line, then analyze Side A ingest JSONL for blur-break points.

Requires demo with ``--bridge --bridge-direct-speak --tickfeed-debug``.
"""
from __future__ import annotations

import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "output" / "previews" / "tickfeed_side_a.jsonl"
BASE = "http://127.0.0.1:8766"
TOKEN = "tickfeed-lab"


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


def _wait_ready() -> None:
    for _ in range(60):
        try:
            if int(_status().get("tick") or 0) > 10:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    raise RuntimeError("bridge not ready")


def _load_rows(after_t: float) -> list[dict]:
    if not LOG.is_file():
        return []
    rows: list[dict] = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "ingest":
            continue
        if float(row.get("t") or 0) + 1e-6 >= after_t:
            rows.append(row)
    return rows


def analyze(rows: list[dict]) -> dict:
    flags: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    hot: list[dict] = []
    for r in rows:
        kinds[str(r.get("kind"))] += 1
        for f in r.get("blur_flags") or []:
            flags[f] += 1
        lab = r.get("labels") or {}
        gain = r.get("gain") or {}
        field = r.get("field") or {}
        if float(lab.get("plate_open") or 0) > 0.25 or float(gain.get("travel") or 0) > 0.05:
            hot.append(
                {
                    "t": r.get("t"),
                    "tick": r.get("master_tick"),
                    "kind": r.get("kind"),
                    "live": r.get("live_speech"),
                    "open": lab.get("plate_open"),
                    "smile": lab.get("ui_smile"),
                    "viseme": lab.get("viseme"),
                    "field_max": field.get("max"),
                    "mouth_max": field.get("mouth_max"),
                    "sep": field.get("sep_l_minus_u"),
                    "gain_eff": gain.get("effective"),
                    "travel": gain.get("travel"),
                    "gpu_peak": (r.get("gpu") or {}).get("peak"),
                    "muscles": (r.get("gpu") or {}).get("muscles"),
                    "flags": r.get("blur_flags"),
                }
            )
    # Rank by likely blur: plate+field stack first, then travel
    hot.sort(
        key=lambda h: (
            "plate+field_stack" in (h.get("flags") or []),
            "open+smile_stack" in (h.get("flags") or []),
            float(h.get("travel") or 0),
            float(h.get("open") or 0),
        ),
        reverse=True,
    )
    return {
        "n": len(rows),
        "kinds": dict(kinds),
        "blur_flag_counts": dict(flags),
        "hot_top": hot[:20],
        "verdict": _verdict(flags, rows),
    }


def _verdict(flags: Counter[str], rows: list[dict]) -> list[str]:
    out: list[str] = []
    if flags.get("plate+field_stack", 0) > 0:
        out.append(
            "BREAK: LOOK plate open AND FIELD travel both high — "
            "double-lip / blur ghost (plate+field_stack)."
        )
    if flags.get("open+smile_stack", 0) > 0:
        out.append(
            "BREAK: open plate stacked with smile plate — muddy mouth corners."
        )
    if flags.get("muscle_stack", 0) > 0:
        out.append("BREAK: muscle warp still active under TickFeed.")
    if flags.get("open_plate_field_not_muted", 0) > 0:
        out.append(
            "BREAK: open plate high but field_gain_eff not muted enough."
        )
    misses = sum(1 for r in rows if r.get("kind") == "MISS")
    if misses:
        out.append(f"BREAK: {misses} Side A MISS ticks (no package).")
    if not out:
        opens = [float((r.get("labels") or {}).get("plate_open") or 0) for r in rows]
        if max(opens or [0]) < 0.2:
            out.append(
                "No strong open seen in window — speech overlay may be dying early."
            )
        else:
            out.append(
                "No classic stack flags in this window; blur may be open.png matte/"
                "blend in the fragment shader (plate composite)."
            )
    return out


def main() -> int:
    _wait_ready()
    st0 = _status()
    print("bridge tick", st0.get("tick"), "debug", (st0.get("tickfeed") or {}).get("side_a_debug"))
    # Mark session time relative to debug log (log t is seconds since demo debug open).
    # Use wall wait + filter by master_tick instead.
    tick0 = int(st0.get("tick") or 0)
    t_mark = 0.0
    if LOG.is_file():
        # Read last session t
        for line in LOG.read_text(encoding="utf-8").splitlines()[::-1]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "ingest":
                t_mark = float(row.get("t") or 0)
                break

    text = "Ah oh oo ee. Open wide now. Say ah ah ah."
    print("SPEAK:", text)
    _req("POST", "/speak", {"text": text})

    # Wait for speech activity then settle
    deadline = time.time() + 8.0
    while time.time() < deadline:
        tf = _status().get("tickfeed") or {}
        if float(tf.get("open") or 0) > 0.3:
            break
        time.sleep(0.05)
    time.sleep(2.5)

    rows = [
        r
        for r in _load_rows(t_mark)
        if int(r.get("master_tick") or 0) >= tick0
    ]
    report = analyze(rows)
    out = ROOT / "output" / "previews" / "side_a_debug_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(f"Raw JSONL: {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
