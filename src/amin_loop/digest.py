"""Step 4 — Digest image/video → .bds + plates (capture path on NWR cells).

Uses ``aiface.capture`` so identity comes from real frames — never invented teeth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aiface.capture import run_capture_from_video
from aiface.paths import ensure_output_tree
from aiface.runtime.bds import load_bds
from amin_loop.regions import digest_regions_from_grid, write_region_catalog


def digest_video_to_world(
    video: Path,
    *,
    world_dir: Path,
    sample_fps: float = 6.0,
    allow_soft: bool = True,
) -> dict[str, Any]:
    """Digest a frontal take into avatar world + region catalog."""
    video = Path(video).resolve()
    world_dir = Path(world_dir).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    ensure_output_tree()
    world_dir.mkdir(parents=True, exist_ok=True)
    world = world_dir / "avatar_face.bds"
    result = run_capture_from_video(
        video,
        output=world,
        sample_fps=float(sample_fps),
        preview=True,
        allow_soft=bool(allow_soft),
    )
    _header, grid = load_bds(result.world)
    arr = np.asarray(grid, dtype=np.float32)
    regions = digest_regions_from_grid(arr)
    catalog = write_region_catalog(
        regions,
        world_dir / "region_catalog.json",
        extra={"source_video": str(video), "world": str(result.world)},
    )
    return {
        "world": str(result.world),
        "portrait": str(result.portrait),
        "smile_plate": str(result.smile_plate),
        "open_plate": str(result.open_plate),
        "meta": str(result.meta),
        "region_catalog": str(catalog),
        "region_count": len(regions),
        "priors": {
            "jaw_travel_scale": float(result.priors.jaw_travel_scale),
            "lip_width_scale": float(result.priors.lip_width_scale),
            "lip_open_scale": float(result.priors.lip_open_scale),
        },
    }


__all__ = ["digest_video_to_world"]
