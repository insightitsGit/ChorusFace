"""Word-timed mouth cell plan: detect roles and step toward neighbors."""

from __future__ import annotations

import numpy as np

from aiface.cell_cluster import CellCluster, CellClusterIndex
from aiface.mouth_cell_plan import (
    MouthCellPlan,
    detect_mouth_cells,
    flow_velocity,
    velocity_to_neighbor_step,
    viseme_flow,
)


def _toy_mouth() -> CellCluster:
    # Ellipse of cells around (20, 20).
    cells = []
    for y in range(14, 27):
        for x in range(12, 29):
            if ((x - 20) / 8.0) ** 2 + ((y - 20) / 5.0) ** 2 <= 1.0:
                cells.append((x, y))
    return CellCluster(
        region_id=0,
        name="mouth_unlocked",
        cells=np.asarray(cells, dtype=np.int32),
    )


def test_detect_assigns_upper_and_lower() -> None:
    cluster = _toy_mouth()
    detected = detect_mouth_cells(cluster)
    assert len(detected) == cluster.cell_count
    assert any(c.lip > 0.2 for c in detected)
    assert any(c.lip < -0.2 for c in detected)
    assert any(c.side > 0.2 for c in detected)
    assert any(c.side < -0.2 for c in detected)


def test_ah_flow_opens_vertically() -> None:
    from aiface.mouth_cell_plan import DetectedCell

    open_n, width_n, round_n = viseme_flow("AH")
    assert open_n > 0.9
    u = DetectedCell(x=20, y=24, side=0.0, lip=0.8, radial=0.7)
    lo = DetectedCell(x=20, y=16, side=0.0, lip=-0.8, radial=0.7)
    _, uy = flow_velocity(u, open_n=open_n, width_n=width_n, round_n=round_n)
    _, ly = flow_velocity(lo, open_n=open_n, width_n=width_n, round_n=round_n)
    assert uy > 0.2
    assert ly < -0.2


def test_ee_flow_widens_corners() -> None:
    open_n, width_n, round_n = viseme_flow("EE")
    from aiface.mouth_cell_plan import DetectedCell

    left = DetectedCell(x=12, y=20, side=-0.9, lip=0.0, radial=0.8)
    right = DetectedCell(x=28, y=20, side=0.9, lip=0.0, radial=0.8)
    lvx, _ = flow_velocity(left, open_n=open_n, width_n=width_n, round_n=round_n)
    rvx, _ = flow_velocity(right, open_n=open_n, width_n=width_n, round_n=round_n)
    assert lvx < -0.2
    assert rvx > 0.2


def test_velocity_quantizes_to_neighbor() -> None:
    assert velocity_to_neighbor_step(0.0, 1.0) == (0, 1)
    assert velocity_to_neighbor_step(-1.0, 0.0) == (-1, 0)
    assert velocity_to_neighbor_step(0.8, 0.7) == (1, 1)


def test_plan_follows_word_timing() -> None:
    cluster = _toy_mouth()
    index = CellClusterIndex(width=64, height=64, clusters=[cluster])
    for x, y in cluster.cells.tolist():
        index._membership[(int(x), int(y))] = "mouth_unlocked"
    plan = MouthCellPlan(index, budget=32, speed=1.0)
    assert plan.cell_count == cluster.cell_count

    plan.sync_from_timeline("AH", active_until=1.0, now=0.1)
    steps = plan.plan_steps(tick=1)
    assert len(steps) > 0
    assert all(s.phoneme == "AH" for s in steps)
    assert any(s.dy != 0 or abs(s.vy) > 0.05 for s in steps)

    impulses = plan.impulses_for_tick(tick=2)
    assert len(impulses) > 0

    plan.sync_from_timeline("REST", active_until=0.0, now=2.0)
    assert plan.plan_steps(tick=3) == []


def test_snapshot_reports_detection() -> None:
    cluster = _toy_mouth()
    index = CellClusterIndex(width=64, height=64, clusters=[cluster])
    plan = MouthCellPlan(index, budget=16)
    plan.sync_from_timeline("OU", active_until=5.0, now=1.0)
    plan.plan_steps(tick=1)
    snap = plan.snapshot()
    assert snap["detected_cells"] == cluster.cell_count
    assert snap["phoneme"] == "OU"
    assert snap["flow"]["round"] > 0.5
