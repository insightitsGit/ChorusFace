#!/usr/bin/env python3
"""Rebuild LOOK plate alphas + optional corner heal (B4-safe).

Full-cycle QA findings:
  - chin-bowl ellipse stamped a visible disc
  - resting smile corners peeked beside the open O
  - dark corner slits exist in capture RGB

This fits a wide/tall-enough oral ellipse (cover smile wings, not chin),
keeps a solid core, and softly heals dark corner outliers from local plate
skin (no new generative face RGB).
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
    x_pad: float = 2.05,
    up_pad: float = 1.35,
    diff_lo: float = 18.0,
    diff_hi: float = 48.0,
) -> tuple[np.ndarray, dict[str, float]]:
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    src = source_rgb.astype(np.float32)
    plate = rgb.astype(np.float32)
    diff = np.linalg.norm(plate - src, axis=2)
    lum = plate.mean(axis=2)
    h, _w = a.shape
    yy, xx = np.mgrid[0:h, 0 : a.shape[1]]

    content = _ss(diff_lo, diff_hi, diff) * _ss(0.06, 0.28, a)
    content = np.maximum(
        content,
        _ss(0.15, 0.40, a) * _ss(18.0, 40.0, diff) * (1.0 - _ss(50.0, 100.0, lum)),
    )
    content = np.maximum(
        content,
        _ss(0.12, 0.35, a) * _ss(12.0, 30.0, diff) * _ss(165.0, 210.0, lum),
    )
    seed = content > 0.32
    labels, count = ndimage.label(seed)
    if count > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        seed = labels == int(sizes.argmax())

    ys, xs = np.where(seed)
    if ys.size < 32:
        return _ss(0.22, 0.55, a).astype(np.float32), {"rx": 0.0, "r_up": 0.0, "r_dn": 0.0}

    cy = float(ys.mean())
    cx = float(xs.mean())
    y0, y1 = np.percentile(ys, [6, 92])
    x0, x1 = np.percentile(xs, [2, 98])
    r_up = max((cy - y0) * float(up_pad), 34.0)
    r_dn = max((y1 - cy) * 1.10, 28.0)
    rx = max((x1 - x0) * 0.5 * float(x_pad), 110.0)

    dy = np.where(
        yy < cy, (cy - yy) / max(r_up, 1e-3), (yy - cy) / max(r_dn, 1e-3)
    )
    r = np.sqrt(dy**2 + ((xx - cx) / max(rx, 1e-3)) ** 2)
    core = (r <= 0.80).astype(np.float32)
    rim = (1.0 - _ss(0.80, 1.03, r)) * (r > 0.80).astype(np.float32)
    matte = np.clip(core + rim, 0.0, 1.0) * _ss(0.04, 0.16, a)
    return np.clip(matte, 0.0, 1.0).astype(np.float32), {
        "rx": rx,
        "r_up": r_up,
        "r_dn": r_dn,
        "cx": cx,
        "cy": cy,
    }


def heal_corner_slits(
    rgb: np.ndarray, matte: np.ndarray, geom: dict[str, float]
) -> tuple[np.ndarray, int]:
    """Soften dark corner outliers using local plate fill (not generative RGB)."""
    out = rgb.astype(np.float32).copy()
    lum = out.mean(axis=2)
    med = ndimage.median_filter(lum, size=15)
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy = float(geom.get("cy", h * 0.75))
    cx = float(geom.get("cx", w * 0.5))
    r_up = float(geom.get("r_up", 40.0))
    rx = float(geom.get("rx", 120.0))
    corner = (
        (np.abs(yy - cy) < r_up * 1.1)
        & (np.abs(xx - cx) > rx * 0.22)
        & (matte > 0.25)
    )
    heal = corner & (lum < med - 12.0) & (lum < 110.0)
    heal = ndimage.binary_dilation(heal, iterations=2)
    if not np.any(heal):
        return out, 0
    fill = np.dstack(
        [ndimage.uniform_filter(out[..., c], size=17) for c in range(3)]
    )
    w = heal.astype(np.float32)[..., None] * 0.80
    out = out * (1.0 - w) + fill * w
    return out, int(heal.sum())


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
    x_pad: float,
    up_pad: float,
    diff_lo: float,
    diff_hi: float,
    heal: bool,
) -> dict:
    bak = path.with_suffix(path.suffix + ".matte_bak.png")
    src_path = bak if bak.is_file() else path
    arr = np.asarray(Image.open(src_path).convert("RGBA")).astype(np.float32)
    cur = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)
    rgb = arr[..., :3] if bak.is_file() else cur[..., :3]
    a = arr[..., 3] / 255.0
    new_a, geom = oral_matte(
        rgb,
        a,
        source_rgb,
        x_pad=x_pad,
        up_pad=up_pad,
        diff_lo=diff_lo,
        diff_hi=diff_hi,
    )
    healed = 0
    out_rgb = rgb
    if heal and path.name.startswith("open"):
        out_rgb, healed = heal_corner_slits(rgb, new_a, geom)
    stats = {
        "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "alpha_mean_before": float((cur[..., 3] / 255.0).mean()),
        "alpha_mean_after": float(new_a.mean()),
        "alpha_gt_0_5_before": float((cur[..., 3] > 127).mean()),
        "alpha_gt_0_5_after": float((new_a > 0.5).mean()),
        "heal_pixels": healed,
        "geom": geom,
        "source_alpha": "matte_bak" if bak.is_file() else "current",
    }
    if dry_run:
        return stats
    if not bak.is_file():
        shutil.copy2(path, bak)
    out = np.dstack([out_rgb, np.round(new_a * 255.0)])
    Image.fromarray(out.astype(np.uint8), mode="RGBA").save(path)
    return stats


def collect_paths(world: Path) -> list[Path]:
    return [p for name in ("open.png", "smile.png") if (p := world / name).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--x-pad", type=float, default=2.05)
    parser.add_argument("--up-pad", type=float, default=1.35)
    parser.add_argument("--diff-lo", type=float, default=18.0)
    parser.add_argument("--diff-hi", type=float, default=48.0)
    parser.add_argument("--no-heal", action="store_true")
    args = parser.parse_args()
    world = args.world.resolve()
    source = load_source(world)
    paths = collect_paths(world)
    if not paths:
        print(f"No plates found under {world}")
        return 2
    report = []
    prev_dir = ROOT / "output" / "previews" / "full_cycle"
    prev_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        dlo = float(args.diff_lo)
        if path.name.startswith("smile"):
            dlo = min(dlo, 16.0)
        stats = process_plate(
            path,
            source,
            dry_run=bool(args.dry_run),
            x_pad=float(args.x_pad),
            up_pad=float(args.up_pad),
            diff_lo=dlo,
            diff_hi=float(args.diff_hi),
            heal=not bool(args.no_heal),
        )
        report.append(stats)
        print(
            f"{'[dry] ' if args.dry_run else ''}"
            f"{stats['file']}: mean {stats['alpha_mean_before']:.3f}→{stats['alpha_mean_after']:.3f} "
            f"gt0.5 {stats['alpha_gt_0_5_before']:.3f}→{stats['alpha_gt_0_5_after']:.3f} "
            f"heal={stats['heal_pixels']}"
        )
        if not args.dry_run:
            a = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
            Image.fromarray(a).save(prev_dir / f"{path.stem}_alpha_tight.png")
    out = world / "plate_oral_matte_report.json"
    if not args.dry_run:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
