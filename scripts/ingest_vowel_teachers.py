"""Ingest first teacher clips (any duration, e.g. 10s) into Teacher Package + D35.

Usage:
  python scripts/ingest_vowel_teachers.py
  python scripts/ingest_vowel_teachers.py --videos-dir path\\to\\folder --limit 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "output" / "teacher" / "teacher_package_v1" / "videos"
DEFAULT_PKG = ROOT / "output" / "teacher" / "teacher_package_v1"

# Prefer these names if present; else take newest mp4s.
PREFERRED = (
    "VowelTeacher_HAPPY",
    "VowelTeacher_SAD_Part1",
    "VowelTeacher_ANGRY_Part1",
    "VowelTeacher_SURPRISED",
    "VowelTeacher_THINKING",
    "VowelTeacher_SAD_Part2",
    "VowelTeacher_ANGRY_Part2",
)


def _stem_key(path: Path) -> str:
    return path.stem


def pick_videos(videos_dir: Path, limit: int) -> list[Path]:
    files = sorted(
        [p for p in videos_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return []
    chosen: list[Path] = []
    by_stem = {p.stem: p for p in files}
    for name in PREFERRED:
        if name in by_stem and by_stem[name] not in chosen:
            chosen.append(by_stem[name])
        if len(chosen) >= limit:
            return chosen
    for p in files:
        if p not in chosen:
            chosen.append(p)
        if len(chosen) >= limit:
            break
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--pkg", type=Path, default=DEFAULT_PKG)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--skip-d35", action="store_true")
    args = ap.parse_args(argv)

    args.videos_dir.mkdir(parents=True, exist_ok=True)
    (args.pkg / "landmarks").mkdir(parents=True, exist_ok=True)
    (args.pkg / "optical_flow").mkdir(parents=True, exist_ok=True)

    clips = pick_videos(args.videos_dir, args.limit)
    if not clips:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "no videos yet",
                    "paste_here": str(args.videos_dir.resolve()),
                    "hint": "Drop 3 mp4s (10s OK). Prefer VowelTeacher_HAPPY.mp4 first.",
                },
                indent=2,
            )
        )
        return 2

    from chorusface.vowel.teacher import run_d35

    results = []
    for clip in clips:
        out = args.pkg / "landmarks" / clip.stem
        out.mkdir(parents=True, exist_ok=True)
        entry = {"file": clip.name, "path": str(clip), "d35": None}
        if not args.skip_d35:
            try:
                metrics = run_d35(clip, out)
                entry["d35"] = metrics.to_dict()
                entry["d35_passed"] = metrics.passed
            except Exception as exc:  # noqa: BLE001
                entry["d35_error"] = str(exc)
                entry["d35_passed"] = False
        results.append(entry)

    report = {
        "ok": True,
        "count": len(results),
        "videos_dir": str(args.videos_dir.resolve()),
        "clips": results,
        "any_d35_pass": any(r.get("d35_passed") for r in results),
    }
    report_path = args.pkg / "ingest_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["any_d35_pass"] or args.skip_d35 else 3


if __name__ == "__main__":
    sys.exit(main())
