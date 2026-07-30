"""AminIntheLoop step modules — unit smoke tests."""

from __future__ import annotations

import numpy as np

from amin_loop.cells import CHANNEL_NAMES, VECTOR_DIM
from amin_loop.control import VelocityImpulse, impulses_to_commands, validate_impulse
from amin_loop.gpu_recipe import build_gpu_recipe
from amin_loop.mapping import build_condition_maps
from amin_loop.regions import connected_components, digest_regions_from_grid


def test_cell_schema_is_32() -> None:
    assert VECTOR_DIM == 32
    assert len(CHANNEL_NAMES) == 32


def test_validate_and_commands() -> None:
    raw = VelocityImpulse(x=10, y=20, vx=100, vy=0, radius=8, tick=2)
    ok = validate_impulse(raw, grid_width=256, grid_height=256, max_speed=4.0)
    assert abs(ok.vx) <= 4.0 + 1e-6
    cmds = impulses_to_commands([ok], grid_width=256, grid_height=256)
    assert len(cmds) == 1
    assert cmds[0].velocity_impulse is not None


def test_regions_from_grid() -> None:
    grid = np.zeros((32, 32, 32), dtype=np.float32)
    grid[8:16, 8:16, 11] = 1.0  # opacity
    grid[8:16, 8:16, 31] = 1.0  # locked identity
    grid[20:28, 20:28, 11] = 0.8
    grid[20:28, 20:28, 31] = 0.0  # unlocked
    regions = digest_regions_from_grid(grid, min_cells=8)
    assert len(regions) >= 2
    names = {r.name for r in regions}
    assert "identity" in names or any(r.locked_frac >= 0.5 for r in regions)


def test_connected_components() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True
    clusters = connected_components(mask)
    assert len(clusters) == 2


def test_maps_and_recipe() -> None:
    maps = build_condition_maps()
    assert "AH" in maps["viseme_table"] or "AA" in maps["viseme_table"]
    recipe = build_gpu_recipe(world=__import__("pathlib").Path("avatar_face.bds"))
    assert "display_path" in recipe
    assert "path_a_mouth_ownership_seals" in recipe["forbidden"]
