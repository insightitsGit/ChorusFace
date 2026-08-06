#!/usr/bin/env python3
"""F17 human-eval harness — checklist + scoring template (no fake scores).

Protocol (VowelDesignFinalAnswers):
  N = 10–15 raters
  muted video only
  emotion ID ≥ 80%
  MC vowel at marked times ≥ 50% and ≥ 20pt over jaw-pump baseline

This script writes a markdown checklist and optionally scores a CSV of
human responses. It never invents pass/fail without real rater data.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PROTOCOL = {
    "n_raters_min": 10,
    "n_raters_max": 15,
    "emotion_floor_pct": 80.0,
    "vowel_floor_pct": 50.0,
    "vowel_over_baseline_pts": 20.0,
    "conditions": [
        "muted video (no audio)",
        "randomize clip order",
        "mark vowel probe times on timeline",
        "include jaw-pump baseline condition",
    ],
    "items": [
        {
            "id": "E1",
            "kind": "emotion",
            "prompt": "Guess emotion",
            "options": ["NEUTRAL", "HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISED"],
            "truth": None,
        },
        {
            "id": "V1",
            "kind": "vowel",
            "prompt": "Which vowel at marker?",
            "options": ["EE", "OU", "AA", "OH", "AX", "other"],
            "truth": None,
        },
    ],
}

TEMPLATE_MD = """# F17 Human Eval Checklist

## Protocol
- Raters: N = {n_min}–{n_max}
- Conditions:
{conditions}

## Pass floors (do not invent scores)
- Emotion identification ≥ {emo}%
- MC vowel ≥ {vowel}% and ≥ {over} points above jaw-pump baseline

## Session log
| rater_id | clip_id | item_id | response | truth | correct |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## Aggregate (fill after scoring CSV)
- emotion_pct:
- vowel_pct:
- jaw_pump_baseline_pct:
- vowel_minus_baseline_pts:
- pass_emotion:
- pass_vowel:
- overall:

## Notes
- Use `scripts/vowel_human_eval_checklist.py --score responses.csv` once data exists.
- Until then status is MISSING, not PASS.
"""


def write_template(path: Path) -> None:
    body = TEMPLATE_MD.format(
        n_min=PROTOCOL["n_raters_min"],
        n_max=PROTOCOL["n_raters_max"],
        conditions="\n".join(f"  - {c}" for c in PROTOCOL["conditions"]),
        emo=PROTOCOL["emotion_floor_pct"],
        vowel=PROTOCOL["vowel_floor_pct"],
        over=PROTOCOL["vowel_over_baseline_pts"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    (path.parent / "f17_protocol.json").write_text(
        json.dumps(PROTOCOL, indent=2), encoding="utf-8"
    )


def score_csv(path: Path) -> dict:
    """Score responses CSV with columns: kind,correct,condition (optional).

    kind: emotion|vowel
    correct: 0|1
    condition: vowel|jaw_pump (for vowel rows)
    """
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return {"ok": False, "error": "empty CSV", "scored": False}

    emo = [int(r["correct"]) for r in rows if r.get("kind") == "emotion"]
    vow = [
        int(r["correct"])
        for r in rows
        if r.get("kind") == "vowel" and r.get("condition", "vowel") != "jaw_pump"
    ]
    base = [
        int(r["correct"])
        for r in rows
        if r.get("kind") == "vowel" and r.get("condition") == "jaw_pump"
    ]
    emotion_pct = 100.0 * sum(emo) / len(emo) if emo else None
    vowel_pct = 100.0 * sum(vow) / len(vow) if vow else None
    baseline_pct = 100.0 * sum(base) / len(base) if base else None
    over = (
        None
        if emotion_pct is None or vowel_pct is None or baseline_pct is None
        else vowel_pct - baseline_pct
    )
    pass_emotion = (
        emotion_pct is not None and emotion_pct >= PROTOCOL["emotion_floor_pct"]
    )
    pass_vowel = (
        vowel_pct is not None
        and baseline_pct is not None
        and vowel_pct >= PROTOCOL["vowel_floor_pct"]
        and (vowel_pct - baseline_pct) >= PROTOCOL["vowel_over_baseline_pts"]
    )
    return {
        "ok": True,
        "scored": True,
        "n_rows": len(rows),
        "emotion_pct": emotion_pct,
        "vowel_pct": vowel_pct,
        "jaw_pump_baseline_pct": baseline_pct,
        "vowel_minus_baseline_pts": over,
        "pass_emotion": pass_emotion,
        "pass_vowel": pass_vowel,
        "overall": bool(pass_emotion and pass_vowel),
        "counts": dict(Counter(r.get("kind") for r in rows)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output/worlds/tickfeed/vowel/f17_human_eval.md"),
    )
    ap.add_argument("--score", type=Path, default=None, help="CSV of rater responses")
    args = ap.parse_args()
    write_template(args.out)
    print(f"wrote {args.out}")
    if args.score is None:
        print(json.dumps({"ok": True, "scored": False, "status": "MISSING_RATINGS"}, indent=2))
        return 0
    report = score_csv(args.score)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
