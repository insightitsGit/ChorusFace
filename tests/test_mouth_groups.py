"""Mouth cell groups: lips / teeth / cavity with retargetable membership."""

from __future__ import annotations

import numpy as np

from chorusface.cell_cluster import CellCluster, CellClusterIndex
from chorusface.mouth_cell_plan import MouthCellPlan, detect_mouth_cells
from chorusface.mouth_groups import (
    GroupMotion,
    assign_groups_geometric,
    build_mouth_group_plan,
)


def _toy_mouth() -> CellCluster:
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


def test_geometry_assigns_all_five_groups() -> None:
    detected = detect_mouth_cells(_toy_mouth())
    groups = assign_groups_geometric(detected)
    counts = groups.counts()
    assert counts["upper_lip"] > 0
    assert counts["lower_lip"] > 0
    assert counts["teeth"] > 0 or counts["cavity"] > 0
    assert sum(counts.values()) >= len(detected)


def test_ah_recipe_activates_teeth_pp_hides_teeth() -> None:
    plan = build_mouth_group_plan(detect_mouth_cells(_toy_mouth()))
    ah = plan.recipe_for("AH")
    assert ah["teeth"].active
    assert ah["teeth"].open_scale > 0.0
    pp = plan.recipe_for("PP")
    assert not pp["teeth"].active
    assert pp["upper_lip"].close_scale > 0.0


def test_retarget_moves_cells_into_teeth() -> None:
    detected = detect_mouth_cells(_toy_mouth())
    plan = build_mouth_group_plan(detected)
    sample = [(c.x, c.y) for c in detected[:12]]
    plan.retarget_group("teeth", sample)
    teeth = {(c.x, c.y) for c in plan.groups.cells_for("teeth")}
    assert set(sample).issubset(teeth)
    assert plan.groups.source.startswith("retarget")


def test_cell_plan_steps_carry_group_labels() -> None:
    cluster = _toy_mouth()
    index = CellClusterIndex(width=64, height=64, clusters=[cluster])
    for x, y in cluster.cells.tolist():
        index._membership[(int(x), int(y))] = "mouth_unlocked"
    plan = MouthCellPlan(index, budget=48, speed=1.0)
    plan.sync_from_timeline("AH", active_until=2.0, now=0.1)
    steps = plan.plan_steps(tick=1)
    assert steps
    assert any(s.group in {"upper_lip", "lower_lip", "teeth", "lip_corners", "cavity"} for s in steps)
    snap = plan.snapshot()
    assert "mouth_groups" in snap
    assert snap["mouth_groups"]["groups"]["groups"]["upper_lip"] > 0


def test_set_recipe_changes_group_weight() -> None:
    plan = build_mouth_group_plan(detect_mouth_cells(_toy_mouth()))
    plan.set_recipe("AH", "teeth", GroupMotion(open_scale=0.9, active=True))
    assert plan.recipe_for("AH")["teeth"].open_scale == 0.9
