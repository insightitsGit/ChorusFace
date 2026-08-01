"""Avatar adoption contract — portable world dirs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aiface.avatar_profile import (
    MIN_MOUTH_CELLS,
    PROFILE_NAME,
    PROFILE_SCHEMA,
    AvatarAdoptionError,
    list_avatars,
    load_avatar_profile,
    meets_requirements,
    open_avatar,
    synthesize_avatar_profile,
    validate_avatar_root,
    write_avatar_profile,
)
from aiface.runtime.bds import HUMAN_LOCK_CHANNEL
from aiface.runtime.recipe import DisplayRecipe


def _write_minimal_bds(path: Path, *, mouth_cells: int = 64) -> None:
    """Tiny HxWx32 grid with a permeable unlocked mouth cluster."""
    from aiface.runtime.bds import save_bds

    h = w = 64
    grid = np.zeros((h, w, 32), dtype=np.float32)
    # Identity lock everywhere except a mouth block.
    grid[..., HUMAN_LOCK_CHANNEL] = 1.0
    grid[..., 11] = 1.0  # opacity
    if mouth_cells > 0:
        n = max(int(mouth_cells**0.5) + 1, 2)
        y0, x0 = 40, 20
        grid[y0 : y0 + n, x0 : x0 + n, HUMAN_LOCK_CHANNEL] = 0.0
        grid[y0 : y0 + n, x0 : x0 + n, 25] = 0.8  # permeability
        grid[y0 : y0 + n, x0 : x0 + n, 11] = 1.0
    metadata = {
        "avatar_seed": {
            "face_box": {"x": 4.0, "y": 4.0, "width": 56.0, "height": 56.0}
        }
    }
    save_bds(path, grid, metadata=metadata)


def _touch_png(path: Path) -> None:
    # Minimal valid 1×1 PNG.
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d494844520000000100000001080200000090"
            "7753de0000000a49444154789c63000100000500010d0a2db400000000"
            "49454e44ae426082"
        )
    )


def _build_world(tmp: Path, *, mouth_in_catalog: int = 80) -> Path:
    world = tmp / "friend_a"
    world.mkdir()
    _write_minimal_bds(world / "avatar_face.bds", mouth_cells=mouth_in_catalog)
    _touch_png(world / "source_face.png")
    _touch_png(world / "open.png")
    _touch_png(world / "smile.png")
    cells = [[20 + (i % 8), 40 + (i // 8)] for i in range(mouth_in_catalog)]
    catalog = {
        "schema": "amin_loop.regions.v1",
        "regions": [
            {
                "region_id": 0,
                "name": "mouth_unlocked",
                "cell_count": mouth_in_catalog,
                "cells_sample": cells[:64],
            }
        ],
    }
    (world / "region_catalog.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )
    recipe = DisplayRecipe().to_payload(
        world_name="avatar_face.bds",
        plates={"open": "open.png", "smile": "smile.png"},
    )
    (world / "gpu_display_recipe.json").write_text(
        json.dumps(recipe), encoding="utf-8"
    )
    (world / "condition_maps.json").write_text(
        json.dumps({"schema": "amin_loop.mapping.v1", "viseme_table": {"AH": {"jaw": 0.9}}}),
        encoding="utf-8",
    )
    return world


def test_validate_rejects_incomplete(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = validate_avatar_root(empty)
    assert not result.ok
    assert "source_face.png" in result.missing


def test_synthesize_and_write_round_trip(tmp_path: Path) -> None:
    world = _build_world(tmp_path)
    path = write_avatar_profile(world, avatar_id="friend_a")
    assert path.name == PROFILE_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == PROFILE_SCHEMA
    assert payload["id"] == "friend_a"
    assert payload["validation"]["ok"] is True
    assert payload["geometry"]["mouth_cell_count"] >= MIN_MOUTH_CELLS

    loaded = load_avatar_profile(world)
    assert loaded.id == "friend_a"
    assert loaded.validation.ok
    assert meets_requirements(world)


def test_open_avatar_bundle(tmp_path: Path) -> None:
    world = _build_world(tmp_path)
    write_avatar_profile(world)
    bundle = open_avatar(world, require=True)
    assert bundle.ok
    assert bundle.world_path.name == "avatar_face.bds"
    assert bundle.source_face.is_file()
    assert bundle.recipe.atlas_strength > 0.0
    assert bundle.condition_jaw.get("AH", 0.9) == pytest.approx(0.9)


def test_open_avatar_require_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(AvatarAdoptionError):
        open_avatar(bad, require=True)


def test_list_avatars_scans_worlds(tmp_path: Path) -> None:
    root = tmp_path / "worlds"
    root.mkdir()
    _build_world(root)
    found = list_avatars(root)
    assert any(p.id == "friend_a" and p.validation.ok for p in found)


def test_mouth_too_small_fails(tmp_path: Path) -> None:
    world = _build_world(tmp_path, mouth_in_catalog=80)
    _write_minimal_bds(world / "avatar_face.bds", mouth_cells=0)
    (world / "region_catalog.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "name": "mouth_unlocked",
                        "cell_count": 4,
                        "cells_sample": [[1, 1]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    profile = synthesize_avatar_profile(world)
    assert not profile.validation.ok
    assert any("mouth_unlocked" in m for m in profile.validation.missing)
