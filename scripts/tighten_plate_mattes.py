#!/usr/bin/env python3
"""Bake hybrid core+edge alphas onto TickFeed LOOK plates (B4-safe).

Does NOT invent RGB. Only rebuilds alpha:
  - core: alpha above core_lo → near-opaque oral interior
  - edge: thin soft falloff below core
  - garbage: very low alpha → 0 (kills wide veil)

Backs up originals next to each file as ``*.matte_bak.png`` (once).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = ROOT / "output" / "worlds" / "tickfeed"


def hybrid_alpha(
    alpha: np.ndarray,
    *,
    core_lo: float = 0.34,
    core_hi: float = 0.58,
    edge_lo: float = 0.10,
    edge_hi: float = 0.34,
    edge_weight: float = 0.50,
) -> np.ndarray:
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    # smoothstep helpers
    def _ss(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    core = _ss(core_lo, core_hi, a)
    edge = _ss(edge_lo, edge_hi, a) * (1.0 - core)
    out = np.clip(core + edge * edge_weight, 0.0, 1.0)
    return out


def process_rgba(path: Path, *, dry_run: bool) -> dict:
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im).astype(np.float32)
    rgb = arr[..., :3]
    a = arr[..., 3] / 255.0
    new_a = hybrid_alpha(a)
    stats = {
        "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "alpha_mean_before": float(a.mean()),
        "alpha_mean_after": float(new_a.mean()),
        "alpha_gt_0_5_before": float((a > 0.5).mean()),
        "alpha_gt_0_5_after": float((new_a > 0.5).mean()),
        "alpha_gt_0_2_before": float((a > 0.2).mean()),
        "alpha_gt_0_2_after": float((new_a > 0.2).mean()),
    }
    if dry_run:
        return stats
    bak = path.with_suffix(path.suffix + ".matte_bak.png")
    if not bak.is_file():
        shutil.copy2(path, bak)
    out = np.dstack([rgb, np.round(new_a * 255.0).astype(np.float32)])
    Image.fromarray(out.astype(np.uint8), mode="RGBA").save(path)
    return stats


def collect_paths(world: Path) -> list[Path]:
    paths: list[Path] = []
    for name in ("open.png", "smile.png", "surprise.png"):
        p = world / name
        if p.is_file():
            paths.append(p)
    meta = world / "plate_atlas.json"
    if meta.is_file():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for item in payload.get("plates") or []:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("file") or "")
            if not rel:
                continue
            p = world / rel
            if p.is_file():
                paths.append(p)
    # unique preserve order
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    world = args.world.resolve()
    paths = collect_paths(world)
    if not paths:
        print(f"No plates found under {world}")
        return 2
    report = []
    for path in paths:
        stats = process_rgba(path, dry_run=bool(args.dry_run))
        report.append(stats)
        print(
            f"{'[dry] ' if args.dry_run else ''}"
            f"{stats['file']}: mean {stats['alpha_mean_before']:.3f}→{stats['alpha_mean_after']:.3f} "
            f"gt0.5 {stats['alpha_gt_0_5_before']:.3f}→{stats['alpha_gt_0_5_after']:.3f}"
        )
    out = world / "plate_matte_tighten_report.json"
    if not args.dry_run:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
