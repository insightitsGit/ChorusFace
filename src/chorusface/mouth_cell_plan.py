"""DEPRECATED realtime path — replaced by ``chorusface.tickfeed`` (full-face KEY/DELTA).

Kept for unit tests and offline group QA only. ``AvatarFaceApp`` no longer
enqueues ±4 MouthCellPlan impulses; FIELD velocity comes from TickPackage
ingest (``tick_ingest.comp``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np

from chorusface.biomechanics.intent import PHONEME_JAW_TARGET
from chorusface.cell_cluster import (
    DEFAULT_CELL_BUDGET,
    CellCluster,
    CellClusterIndex,
    cell_impulse,
    toward_neighbor,
)
from chorusface.mouth_groups import MouthGroupPlan, build_mouth_group_plan
from chorusface.mouth_owner import CLOSED_VISEMES
from chorusface.plates import VISEME_OPENNESS
from chorusface.speech import canonical_viseme
from amin_loop.control import VelocityImpulse

#: Viseme → (open_n, width_n, round_n) in [0, 1].
#: open: part lips vertically; width: stretch corners out; round: pull corners in.
VISEME_FLOW: Final[dict[str, tuple[float, float, float]]] = {
    "REST": (0.0, 0.0, 0.0),
    "CLOSED": (0.0, 0.0, 0.15),
    "PP": (0.0, 0.0, 0.20),
    "MM": (0.0, 0.05, 0.10),
    "FF": (0.18, 0.10, 0.05),
    "TH": (0.28, 0.05, 0.0),
    "DD": (0.22, 0.08, 0.0),
    "KK": (0.16, 0.05, 0.0),
    "CH": (0.24, 0.12, 0.05),
    "SS": (0.20, 0.18, 0.0),
    "NN": (0.26, 0.10, 0.0),
    "RR": (0.30, 0.08, 0.10),
    "AH": (1.00, 0.05, 0.0),
    "AA": (0.90, 0.08, 0.0),
    "EH": (0.55, 0.20, 0.0),
    "IH": (0.35, 0.35, 0.0),
    "EE": (0.30, 0.85, 0.0),
    "OH": (0.65, 0.05, 0.55),
    "OU": (0.45, 0.0, 0.85),
}


@dataclass(frozen=True, slots=True)
class DetectedCell:
    """One mouth cell with local lip geometry (detection result)."""

    x: int
    y: int
    # -1 left … +1 right relative to mouth centroid.
    side: float
    # -1 lower lip … +1 upper lip.
    lip: float
    # 0 centre … 1 rim.
    radial: float


@dataclass(frozen=True, slots=True)
class CellPlanStep:
    """One planned cell change for the active word span."""

    x: int
    y: int
    vx: float
    vy: float
    dx: int
    dy: int
    phoneme: str
    group: str = ""


def detect_mouth_cells(cluster: CellCluster) -> list[DetectedCell]:
    """Detect geometric roles for every cell in the mouth cluster."""
    if cluster.cell_count == 0:
        return []
    xs = cluster.cells[:, 0].astype(np.float64)
    ys = cluster.cells[:, 1].astype(np.float64)
    cx = float(xs.mean())
    cy = float(ys.mean())
    # Half-spans for normalization (avoid divide-by-zero on thin clusters).
    half_w = max(float(xs.std()) * 2.0, float(xs.max() - xs.min()) * 0.5, 1.0)
    half_h = max(float(ys.std()) * 2.0, float(ys.max() - ys.min()) * 0.5, 1.0)
    detected: list[DetectedCell] = []
    for x_i, y_i in cluster.cells.tolist():
        x, y = int(x_i), int(y_i)
        side = float(np.clip((x - cx) / half_w, -1.0, 1.0))
        lip = float(np.clip((y - cy) / half_h, -1.0, 1.0))
        radial = float(
            np.clip(np.hypot((x - cx) / half_w, (y - cy) / half_h), 0.0, 1.5) / 1.5
        )
        detected.append(DetectedCell(x=x, y=y, side=side, lip=lip, radial=radial))
    return detected


def viseme_flow(phoneme: str) -> tuple[float, float, float]:
    """Return (open, width, round) for a canonical viseme."""
    key = canonical_viseme(phoneme)
    if key in VISEME_FLOW:
        return VISEME_FLOW[key]
    # Fallback from jaw / openness tables when an alias slips through.
    open_n = float(VISEME_OPENNESS.get(key, PHONEME_JAW_TARGET.get(key, 0.1)))
    return (
        float(np.clip(open_n, 0.0, 1.0)),
        0.1,
        0.0,
    )


def flow_velocity(
    cell: DetectedCell,
    *,
    open_n: float,
    width_n: float,
    round_n: float,
    close_n: float = 0.0,
    speed: float = 1.0,
) -> tuple[float, float]:
    """Map a detected cell + viseme flow → continuous velocity (grid space).

    Grid convention matches jaw warp: +y up, −y down. Opening sends the upper
    lip up and the lower lip down; width pushes corners outward; round pulls
    them in. ``close_n`` presses lips toward the midline / each other.
    """
    open_n = float(np.clip(open_n, 0.0, 1.0))
    width_n = float(np.clip(width_n, 0.0, 1.0))
    round_n = float(np.clip(round_n, 0.0, 1.0))
    close_n = float(np.clip(close_n, 0.0, 2.0))
    # Rim cells carry more of the articulator motion than the cavity centre.
    rim = 0.35 + 0.65 * float(cell.radial)
    vx = (width_n - round_n) * float(cell.side) * rim
    vy = open_n * float(cell.lip) * rim
    # Closed / press: gentle inward (toward midline) so lips meet.
    if open_n < 0.08 or close_n > 0.0:
        press = max(close_n, 0.35 if open_n < 0.08 else 0.0)
        vx += -0.35 * float(cell.side) * rim * press
        vy += -0.20 * float(cell.lip) * rim * press
    scale = float(speed)
    return vx * scale, vy * scale


def velocity_to_neighbor_step(vx: float, vy: float) -> tuple[int, int]:
    """Quantize a flow vector to a Moore neighbor offset (the 'next' cell)."""
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0, 0
    # Prefer the dominant axis; allow diagonals when both are strong.
    ax, ay = abs(vx), abs(vy)
    dx = 0 if ax < ay * 0.45 else (1 if vx > 0 else -1)
    dy = 0 if ay < ax * 0.45 else (1 if vy > 0 else -1)
    if dx == 0 and dy == 0:
        dx = 1 if vx >= 0 else -1
    return dx, dy


class MouthCellPlan:
    """Timed per-cell plan driven by the word/viseme clock + mouth groups."""

    def __init__(
        self,
        index: CellClusterIndex,
        *,
        budget: int = DEFAULT_CELL_BUDGET,
        speed: float = 0.85,
        group_plan: MouthGroupPlan | None = None,
    ) -> None:
        self._index = index
        self._cluster = index.primary_mouth()
        self._cells = (
            detect_mouth_cells(self._cluster) if self._cluster is not None else []
        )
        self._membership = (
            {(c.x, c.y) for c in self._cells} if self._cells else set()
        )
        self.groups = group_plan or build_mouth_group_plan(self._cells)
        self._active: list[tuple[DetectedCell, Any, str]] = []
        self._rebuild_active_cache("REST")
        self.budget = max(8, int(budget))
        self.speed = float(speed)
        self._cursor = 0
        self._phoneme = "REST"
        self._until = 0.0
        self._open = 0.0
        self._width = 0.0
        self._round = 0.0
        self._last_steps: list[CellPlanStep] = []

    def _rebuild_active_cache(self, phoneme: str) -> None:
        self._active = [
            (grouped.cell, motion, group_name)
            for grouped, motion, group_name in self.groups.iter_active_cells(
                phoneme
            )
        ]

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def phoneme(self) -> str:
        return self._phoneme

    @property
    def active_until(self) -> float:
        return self._until

    def retarget_group(
        self,
        group: str,
        coordinates: Sequence[tuple[int, int]],
        *,
        as_corner: bool = False,
    ) -> None:
        """Rewrite which cells belong to lips/teeth/cavity (capture retarget)."""
        self.groups.retarget_group(group, coordinates, as_corner=as_corner)
        self._rebuild_active_cache(self._phoneme)

    def sync_from_timeline(
        self,
        phoneme: str,
        *,
        active_until: float,
        now: float,
    ) -> None:
        """Follow the same span the layer timeline is showing."""
        key = canonical_viseme(phoneme)
        self._phoneme = key
        self._until = float(active_until)
        if key in CLOSED_VISEMES or key == "REST" or now >= self._until:
            if key == "REST" or now >= self._until:
                self._open, self._width, self._round = 0.0, 0.0, 0.0
                if now >= self._until:
                    self._phoneme = "REST"
            else:
                self._open, self._width, self._round = viseme_flow(key)
        else:
            self._open, self._width, self._round = viseme_flow(key)
        self._rebuild_active_cache(self._phoneme)

    def apply_behavior_flow(
        self,
        open_n: float,
        width_n: float,
        round_n: float,
        *,
        source: str = "behavior",
    ) -> None:
        """Override open/width/round from measured track or ML fill.

        Keeps the active phoneme / group membership; only the flow field
        changes. Used when avatar behavior data is more truthful than tables.
        """
        del source
        self._open = max(0.0, min(1.0, float(open_n)))
        self._width = max(0.0, min(1.0, float(width_n)))
        self._round = max(0.0, min(1.0, float(round_n)))
        if self._phoneme == "REST" and (
            self._open > 1e-3 or self._width > 1e-3 or self._round > 1e-3
        ):
            # Behavior motion without a speech tag — keep REST membership idle.
            return
        self._rebuild_active_cache(self._phoneme)

    def plan_steps(self, *, tick: int) -> list[CellPlanStep]:
        """Plan the next budget of group-targeted cell→neighbor changes."""
        del tick
        if not self._active or self._phoneme == "REST":
            if self._phoneme not in CLOSED_VISEMES:
                self._last_steps = []
                return []
        if (
            self._open <= 1e-4
            and self._width <= 1e-4
            and self._round <= 1e-4
            and self._phoneme not in CLOSED_VISEMES
        ):
            self._last_steps = []
            return []
        n = len(self._active)
        if n == 0:
            self._last_steps = []
            return []
        count = min(self.budget, n)
        start = self._cursor % n
        steps: list[CellPlanStep] = []
        for i in range(count):
            cell, motion, group_name = self._active[(start + i) % n]
            open_s, width_s, round_s, close_s = motion.scaled_flow(
                self._open, self._width, self._round
            )
            # Closed visemes still press via close_scale even when open≈0.
            if self._phoneme in CLOSED_VISEMES:
                close_s = max(close_s, motion.close_scale)
            vx, vy = flow_velocity(
                cell,
                open_n=open_s,
                width_n=width_s,
                round_n=round_s,
                close_n=close_s,
                speed=self.speed,
            )
            dx, dy = velocity_to_neighbor_step(vx, vy)
            if dx == 0 and dy == 0 and abs(vx) < 1e-4 and abs(vy) < 1e-4:
                continue
            nx, ny = cell.x + dx, cell.y + dy
            if (dx == 0 and dy == 0) or (nx, ny) not in self._membership:
                steps.append(
                    CellPlanStep(
                        x=cell.x,
                        y=cell.y,
                        vx=vx,
                        vy=vy,
                        dx=0,
                        dy=0,
                        phoneme=self._phoneme,
                        group=group_name,
                    )
                )
                continue
            steps.append(
                CellPlanStep(
                    x=cell.x,
                    y=cell.y,
                    vx=vx,
                    vy=vy,
                    dx=dx,
                    dy=dy,
                    phoneme=self._phoneme,
                    group=group_name,
                )
            )
        self._cursor = (start + count) % n
        self._last_steps = steps
        return steps

    def impulses_for_tick(self, *, tick: int) -> list[VelocityImpulse]:
        """Emit ±4 proposals that realize the planned neighbor steps."""
        steps = self.plan_steps(tick=tick)
        impulses: list[VelocityImpulse] = []
        for step in steps:
            if step.dx != 0 or step.dy != 0:
                speed = float(np.hypot(step.vx, step.vy))
                impulses.append(
                    toward_neighbor(
                        step.x,
                        step.y,
                        step.dx,
                        step.dy,
                        speed=max(speed, 0.15),
                        tick=tick,
                    )
                )
            else:
                impulses.append(
                    cell_impulse(step.x, step.y, step.vx, step.vy, tick=tick)
                )
        return impulses

    def snapshot(self) -> dict[str, Any]:
        """Observe detection + group plan (for /cells and /probe)."""
        return {
            "detected_cells": self.cell_count,
            "phoneme": self._phoneme,
            "active_until": self._until,
            "flow": {
                "open": self._open,
                "width": self._width,
                "round": self._round,
            },
            "budget": self.budget,
            "cursor": self._cursor,
            "last_steps": len(self._last_steps),
            "mouth_groups": self.groups.snapshot(),
            "sample_steps": [
                {
                    "x": s.x,
                    "y": s.y,
                    "dx": s.dx,
                    "dy": s.dy,
                    "vx": round(s.vx, 3),
                    "vy": round(s.vy, 3),
                    "phoneme": s.phoneme,
                    "group": s.group,
                }
                for s in self._last_steps[:8]
            ],
        }


def build_mouth_cell_plan(
    index: CellClusterIndex | None,
    *,
    budget: int = DEFAULT_CELL_BUDGET,
    speed: float = 0.85,
) -> MouthCellPlan | None:
    if index is None or index.primary_mouth() is None:
        return None
    return MouthCellPlan(index, budget=budget, speed=speed)


__all__ = [
    "VISEME_FLOW",
    "CellPlanStep",
    "DetectedCell",
    "MouthCellPlan",
    "build_mouth_cell_plan",
    "detect_mouth_cells",
    "flow_velocity",
    "velocity_to_neighbor_step",
    "viseme_flow",
]
