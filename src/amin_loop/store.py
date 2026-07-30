"""How Amin-loop persists walkthrough data without full per-frame grids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def describe_world_store(world_dir: Path) -> dict[str, Any]:
    """Summarize on-disk artifacts and their roles (for docs / CLI)."""
    world_dir = Path(world_dir)
    roles = {
        "avatar_face.bds": "Full 256×256×32 cell field (dense truth)",
        "source_face.png": "Immutable identity photo",
        "region_catalog.json": "Region objects: mean[32] + samples (not full grids)",
        "condition_maps.json": "Word/sound/emotion → drive tables",
        "gpu_display_recipe.json": "How GPU shows looks (same path at runtime)",
        "live_vector_dataset.npz": "Train set: audio features → control vectors",
        "live_vector_trajectory.json": "Time series of live vectors from video",
        "live_vector_model.joblib": "ML cover for unknown sounds",
        "open.png": "Condition look: open mouth (real frame)",
        "smile.png": "Condition look: smile (real frame)",
        "plate_atlas.json": "Viseme plate memory index",
        "amin_loop_report.json": "Last train pipeline report",
    }
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(world_dir.iterdir()) if world_dir.is_dir() else []:
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        files.append(
            {
                "name": path.name,
                "bytes": size,
                "role": roles.get(path.name, "support artifact"),
            }
        )
    return {
        "schema": "amin_loop.store.v1",
        "world_dir": str(world_dir),
        "total_bytes": total,
        "strategy": (
            "One dense .bds for all cells; side-cars are tables/vectors/plates. "
            "No per-frame full grids."
        ),
        "files": files,
    }


def write_store_manifest(world_dir: Path) -> Path:
    world_dir = Path(world_dir)
    payload = describe_world_store(world_dir)
    path = world_dir / "amin_data_store.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = ["describe_world_store", "write_store_manifest"]
