"""Step 8 — GPU display recipe learned beside the digested look.

The recipe is not prose: it is the exact contract `AvatarFaceApp` +
`avatar.frag` execute at playback. `chorusface.runtime.recipe.DisplayRecipe`
is the single source of truth; this module serializes it (plus the plates
actually discovered next to the world) so train and play share one path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chorusface.runtime.recipe import RECIPE_NAME, DisplayRecipe


def build_gpu_recipe(
    *,
    world: Path,
    plates: dict[str, str] | None = None,
    recipe: DisplayRecipe | None = None,
) -> dict[str, Any]:
    world = Path(world)
    root = world.parent if world.is_file() or world.suffix else world
    discovered: dict[str, str] = {}
    for name in ("open.png", "smile.png", "surprise.png", "rest.png"):
        candidate = root / name
        if candidate.is_file():
            discovered[name.replace(".png", "")] = name
    discovered.update(plates or {})
    world_name = world.name if world.suffix else "avatar_face.bds"
    return (recipe or DisplayRecipe()).to_payload(
        world_name=world_name, plates=discovered
    )


def write_gpu_recipe(world_dir: Path, recipe: dict[str, Any] | None = None) -> Path:
    world_dir = Path(world_dir)
    world_dir.mkdir(parents=True, exist_ok=True)
    world = world_dir / "avatar_face.bds"
    payload = recipe or build_gpu_recipe(world=world)
    path = world_dir / RECIPE_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = ["RECIPE_NAME", "build_gpu_recipe", "write_gpu_recipe"]
