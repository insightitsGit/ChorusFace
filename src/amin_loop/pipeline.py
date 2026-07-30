"""Step 10 — Run all Amin walkthrough steps end-to-end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from amin_loop.digest import digest_video_to_world
from amin_loop.gpu_recipe import write_gpu_recipe
from amin_loop.live_vectors import train_from_video
from amin_loop.mapping import write_condition_maps
from amin_loop.store import write_store_manifest


def run_all_steps(
    video: Path,
    *,
    world_dir: Path,
    digest_fps: float = 6.0,
    vector_fps: float = 12.0,
    skip_digest: bool = False,
    landmarker_model: Path | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    """Implement steps 4–10: digest → maps → recipe → live vectors."""
    video = Path(video).resolve()
    world_dir = Path(world_dir).resolve()
    world_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "amin_loop.run_all_steps.v1",
        "video": str(video),
        "world_dir": str(world_dir),
        "steps": {},
    }

    world = world_dir / "avatar_face.bds"
    if skip_digest and world.is_file():
        report["steps"]["digest"] = {
            "skipped": True,
            "world": str(world),
        }
    else:
        report["steps"]["digest"] = digest_video_to_world(
            video,
            world_dir=world_dir,
            sample_fps=digest_fps,
        )

    maps_path = write_condition_maps(world_dir)
    report["steps"]["mapping"] = {"condition_maps": str(maps_path)}

    recipe_path = write_gpu_recipe(world_dir)
    report["steps"]["gpu_recipe"] = {"recipe": str(recipe_path)}

    report["steps"]["live_vectors"] = train_from_video(
        video,
        world_dir=world_dir,
        sample_fps=vector_fps,
        landmarker_model=landmarker_model,
        seed=seed,
    )

    store_path = write_store_manifest(world_dir)
    report["steps"]["store"] = {"manifest": str(store_path)}

    summary = world_dir / "amin_loop_report.json"
    summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(summary)
    return report


__all__ = ["run_all_steps"]
