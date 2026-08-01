"""Per-cell / neighbor cluster control must address unlocked mouth cells."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiface.cell_cluster import (
    CELL_RADIUS,
    cell_impulse,
    cluster_flow,
    neighbor_wave,
    parse_drive_request,
    to_commands,
    toward_neighbor,
)
from aiface.cell_cluster import CellCluster, CellClusterIndex
from amin_loop.control import validate_impulse


def test_toward_neighbor_points_at_offset() -> None:
    impulse = toward_neighbor(10, 20, 0, -1, speed=2.0, tick=3)
    assert impulse.x == pytest.approx(10.5)
    assert impulse.y == pytest.approx(20.5)
    assert impulse.vx == pytest.approx(0.0)
    assert impulse.vy == pytest.approx(-2.0)
    assert impulse.radius == CELL_RADIUS


def test_cell_impulse_validates_inside_grid() -> None:
    raw = cell_impulse(2, 3, 10.0, 0.0, tick=1)  # speed will clamp
    ok = validate_impulse(raw, grid_width=256, grid_height=256, max_speed=4.0)
    assert ok.vx == pytest.approx(4.0)
    assert ok.vy == pytest.approx(0.0)


def test_cluster_flow_round_robins() -> None:
    cells = np.asarray([[i, 0] for i in range(10)], dtype=np.int32)
    cluster = CellCluster(region_id=0, name="mouth_unlocked", cells=cells)
    batch_a, cursor = cluster_flow(cluster, 1.0, 0.0, tick=1, cursor=0, budget=4)
    assert len(batch_a) == 4
    assert cursor == 4
    batch_b, cursor = cluster_flow(cluster, 1.0, 0.0, tick=2, cursor=cursor, budget=4)
    assert [int(i.x - 0.5) for i in batch_b] == [4, 5, 6, 7]


def test_neighbor_wave_stays_in_cluster() -> None:
    # 3x3 block so every interior cell has in-cluster neighbors.
    coords = [(x, y) for y in range(3) for x in range(3)]
    cells = np.asarray(coords, dtype=np.int32)
    cluster = CellCluster(region_id=1, name="mouth_unlocked", cells=cells)
    batch, _cursor = neighbor_wave(
        cluster, tick=1, speed=1.0, cursor=0, budget=9, include_diag=False
    )
    assert len(batch) == 9
    membership = set(coords)
    for impulse in batch:
        x = int(impulse.x - 0.5)
        y = int(impulse.y - 0.5)
        assert (x, y) in membership
        # Velocity aims toward a neighbor cell that is also in the cluster.
        nx = x + int(np.sign(impulse.vx) if abs(impulse.vx) > 0.1 else 0)
        ny = y + int(np.sign(impulse.vy) if abs(impulse.vy) > 0.1 else 0)
        if abs(impulse.vx) > 0.1 or abs(impulse.vy) > 0.1:
            assert (nx, ny) in membership


def test_to_commands_are_ai_velocity_rows() -> None:
    cmds = to_commands(
        [cell_impulse(5, 6, 0.5, -0.5, tick=9)],
        grid_width=256,
        grid_height=256,
    )
    assert len(cmds) == 1
    row = cmds[0].as_row()
    assert row[6] == pytest.approx(-4.0)  # AI ±4
    assert row[4] == pytest.approx(CELL_RADIUS)


def test_parse_drive_request_cell_and_batch() -> None:
    index = CellClusterIndex(width=32, height=32)
    cells = np.asarray([[8, 9], [9, 9], [10, 9]], dtype=np.int32)
    index.clusters.append(
        CellCluster(region_id=0, name="mouth_unlocked", cells=cells)
    )
    for x, y in cells.tolist():
        index._membership[(int(x), int(y))] = "mouth_unlocked"

    one = parse_drive_request(
        {"mode": "cell", "x": 8, "y": 9, "dx": 1, "dy": 0, "speed": 1.5},
        index=index,
        tick=2,
    )
    assert len(one) == 1
    assert one[0].vx > 0

    batch = parse_drive_request(
        {
            "mode": "batch",
            "cells": [
                {"x": 8, "y": 9, "vx": 0.2, "vy": 0.0},
                {"x": 9, "y": 9, "dx": 0, "dy": 1},
            ],
        },
        index=index,
        tick=3,
    )
    assert len(batch) == 2


@pytest.mark.skipif(
    not (Path("output/worlds/avatar/avatar_face.bds")).is_file(),
    reason="avatar world not present",
)
def test_index_loads_mouth_from_live_world() -> None:
    index = CellClusterIndex.from_world("output/worlds/avatar/avatar_face.bds")
    mouth = index.primary_mouth()
    assert mouth is not None
    assert mouth.cell_count >= 100
    assert index.region_of(int(mouth.cells[0, 0]), int(mouth.cells[0, 1])) == (
        "mouth_unlocked"
    )
