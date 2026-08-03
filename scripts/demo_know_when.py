#!/usr/bin/env python3
"""Show how we know SMILE vs SAY_HI — contract time + labeled frames."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


def main() -> None:
    root = Path("output/worlds/avatar")
    look = json.loads(
        (root / "face_cell_timeline" / "look_drive.json").read_text(encoding="utf-8")
    )
    speech = json.loads(
        (root / "face_cell_timeline" / "speech_align.json").read_text(encoding="utf-8")
    )
    script = json.loads((root / "calibration_script.json").read_text(encoding="utf-8"))

    print("=== 8s SCRIPT (how we KNOW when) ===")
    print("We do NOT guess from pixels alone for beat identity.")
    print("The calibration contract says what happens in each time window:\n")
    for b in script["beats"]:
        print(
            f"  {b['t0']:3.1f}-{b['t1']:3.1f}s  {b['id']:8s}  "
            f"speech={b.get('speech')!r}"
        )

    print("\n=== LIVE LABELS AT KEY MOMENTS (woman take) ===")
    look_by = {int(r["tick"]): r for r in look["ticks"]}
    sp_by = {int(r["tick"]): r for r in speech["ticks"]}
    samples = [
        (0.5, "should be REST"),
        (1.5, "should SMILE"),
        (2.5, "should OPEN / ah"),
        (3.5, "should SAY_HI / hi"),
        (4.5, "should SURPRISE"),
        (5.5, "should ANGRY"),
        (6.75, "should TALK"),
        (7.75, "should REST"),
    ]
    for tsec, expect in samples:
        tick = int(round(tsec * 60))
        lk = look_by.get(tick, {})
        sp = sp_by.get(tick, {})
        print(
            f"  t={tsec:4.2f}s  expect={expect:18s}  "
            f"beat={str(lk.get('beat', '?')):8s}  "
            f"smile={float(lk.get('smile', 0)):.2f}  "
            f"open={float(lk.get('open', 0)):.2f}  "
            f"viseme={str(sp.get('viseme', '?')):6s}  "
            f"word={sp.get('word', '')!r}"
        )

    demo = root / "demo_know_when"
    demo.mkdir(exist_ok=True)
    vid = root / "calibration_take.mp4"
    cap = cv2.VideoCapture(str(vid))
    for label, tsec in [
        ("01_REST", 0.5),
        ("02_SMILE", 1.5),
        ("03_OPEN", 2.5),
        ("04_SAY_HI", 3.5),
        ("05_SURPRISE", 4.5),
        ("06_ANGRY", 5.5),
        ("07_TALK", 6.75),
        ("08_REST", 7.75),
    ]:
        cap.set(cv2.CAP_PROP_POS_MSEC, tsec * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        tick = int(round(tsec * 60))
        lk = look_by.get(tick, {})
        sp = sp_by.get(tick, {})
        line1 = (
            f"{label}  smile={float(lk.get('smile', 0)):.2f}  "
            f"open={float(lk.get('open', 0)):.2f}  word={sp.get('word', '')}"
        )
        line2 = "Contract time window = what we expect (not free guess)"
        cv2.putText(
            frame,
            line1,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line2,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        out = demo / f"{label}.jpg"
        cv2.imwrite(str(out), frame)
        print(f"wrote {out}")
    cap.release()

    # open folder for the user
    print(f"\nDEMO_DIR {demo.resolve()}")
    print("Open the JPGs — 02_SMILE should show smile labels; 04_SAY_HI should show word=hi")


if __name__ == "__main__":
    main()
