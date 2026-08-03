#!/usr/bin/env python3
"""Rebuild LOOK plate alphas to an asymmetric oral ellipse (B4-safe).

Keeps plate RGB untouched. Fits an ellipse to |plate−source| content with a
*short* upward radius (avoids nose stamp / hard seam) and longer down/side
coverage so lips hide closed identity without a lower-face veil.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = ROOT / "output" / "worlds" / "tickfeed"


def _ss(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def oral_matte(
    rgb: np.ndarray,
    alpha: np.ndarray,
    source_rgb: np.ndarray,
    *,
    diff_lo: float = 18.0,
    diff_hi: float = 48.0,
) -> np.ndarray:
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    src = source_rgb.astype(np.float32)
    plate = rgb.astype(np.float32)
    diff = np.linalg.norm(plate - src, axis=2)
    lum = plate.mean(axis=2)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]

    content = _ss(diff_lo, diff_hi, diff) * _ss(0.06, 0.28, a)
    content = np.maximum(
        content,
        _ss(0.15, 0.40, a) * _ss(18.0, 40.0, diff) * (1.0 - _ss(50.0, 100.0, lum)),
    )
    content = np.maximum(
        content,
        _ss(0.12, 0.35, a) * _ss(12.0, 30.0, diff) * _ss(165.0, 210.0, lum),
    )
    pts = content > 0.35
    labels, count = ndimage.label(pts)
    if count > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        pts = labels == int(sizes.argmax())

    ys, xs = np.where(pts)
    if ys.size < 32:
        return _ss(0.22, 0.55, a).astype(np.float32)

    cy = float(ys.mean())
    cx = float(xs.mean())
    y0, y1 = np.percentile(ys, [8, 97])
    x0, x1 = np.percentile(xs, [3, 97])
    r_up = max((cy - y0) * 0.70, 16.0)
    r_dn = max((y1 - cy) * 1.30, 36.0)
    rx = max((x1 - x0) * 0.5 * 1.20, 58.0)

    dy = np.where(
        yy < cy, (cy - yy) / max(r_up, 1e-3), (yy - cy) / max(r_dn, 1e-3)
    )
    r = np.sqrt(dy**2 + ((xx - cx) / max(rx, 1e-3)) ** 2)
    # Solid core (fills oral cavity) + thin rim feather. A hollow matte left
    # closed-identity skin showing through the dark mouth interior.
    core = (r <= 0.82).astype(np.float32)
    rim = (1.0 - _ss(0.82, 1.05, r)) * (r > 0.82).astype(np.float32)
    matte = np.clip(core + rim, 0.0, 1.0)
    matte *= _ss(0.04, 0.16, a)
    filled = ndimage.binary_fill_holes(matte > 0.5)
    matte = np.maximum(matte, filled.astype(np.float32) * _ss(0.04, 0.16, a))
    return np.clip(matte, 0.0, 1.0).astype(np.float32)


def load_source(world: Path) -> np.ndarray:
    path = world / "source_face.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def process_plate(
    path: Path,
    source_rgb: np.ndarray,
    *,
    dry_run: bool,
    diff_lo: float,
    diff_hi: float,
) -> dict:
    bak = path.with_suffix(path.suffix + ".matte_bak.png")
    src_path = bak if bak.is_file() else path
    arr = np.asarray(Image.open(src_path).convert("RGBA")).astype(np.float32)
    cur = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)
    rgb = arr[..., :3] if bak.is_file() else cur[..., :3]
    a = arr[..., 3] / 255.0
    new_a = oral_matte(rgb, a, source_rgb, diff_lo=diff_lo, diff_hi=diff_hi)
    stats = {
        "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "alpha_mean_before": float((cur[..., 3] / 255.0).mean()),
        "alpha_mean_after": float(new_a.mean()),
        "alpha_gt_0_5_before": float((cur[..., 3] > 127).mean()),
        "alpha_gt_0_5_after": float((new_a > 0.5).mean()),
        "alpha_gt_0_2_before": float((cur[..., 3] > 51).mean()),
        "alpha_gt_0_2_after": float((new_a > 0.2).mean()),
        "source_alpha": "matte_bak" if bak.is_file() else "current",
    }
    if dry_run:
        return stats
    if not bak.is_file():
        shutil.copy2(path, bak)
    out = np.dstack([rgb, np.round(new_a * 255.0)])
    Image.fromarray(out.astype(np.uint8), mode="RGBA").save(path)
    return stats


def collect_paths(world: Path) -> list[Path]:
    return [p for name in ("open.png", "smile.png") if (p := world / name).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diff-lo", type=float, default=18.0)
    parser.add_argument("--diff-hi", type=float, default=48.0)
    args = parser.parse_args()
    world = args.world.resolve()
    source = load_source(world)
    paths = collect_paths(world)
    if not paths:
        print(f"No plates found under {world}")
        return 2
    report = []
    prev_dir = ROOT / "output" / "previews" / "blur_still"
    prev_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        dlo = float(args.diff_lo)
        if path.name.startswith("smile"):
            dlo = min(dlo, 16.0)
        stats = process_plate(
            path,
            source,
            dry_run=bool(args.dry_run),
            diff_lo=dlo,
            diff_hi=float(args.diff_hi),
        )
        report.append(stats)
        print(
            f"{'[dry] ' if args.dry_run else ''}"
            f"{stats['file']}: mean {stats['alpha_mean_before']:.3f}→{stats['alpha_mean_after']:.3f} "
            f"gt0.5 {stats['alpha_gt_0_5_before']:.3f}→{stats['alpha_gt_0_5_after']:.3f} "
            f"({stats['source_alpha']})"
        )
        if not args.dry_run:
            a = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
            Image.fromarray(a).save(prev_dir / f"{path.stem}_oral_alpha.png")
    out = world / "plate_oral_matte_report.json"
    if not args.dry_run:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
