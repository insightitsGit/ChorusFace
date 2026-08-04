"""Per-cell / cluster control over unlocked NWR region objects.

Digest builds connected cell clusters (``region_catalog``). Runtime used to
collapse every lip/jaw writer onto one mouth disc — that threw away the object
address. This module restores addressability:

* load every unlocked mouth cell from the live ``.bds``
* propose ±4 velocity at a single cell (radius 0.5)
* aim a cell toward a Moore neighbor
* sweep a whole cluster under the per-tick command budget

Master Lock still rejects AI writes on identity cells on the GPU. Albedo is
never written here — only channels 0/1 (velocity).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import numpy.typing as npt

from amin_loop.control import (
    VelocityImpulse,
    impulses_to_commands,
    neighbor_offsets,
    validate_impulse,
)
from chorusface.runtime.bds import HUMAN_LOCK_CHANNEL, load_bds
from chorusface.runtime.commands import PaintCommand

#: Cell-precise disc — covers one cell centre without flooding neighbors.
CELL_RADIUS: Final = 0.5
#: Soft-tissue fingerprint used at digest time for ``mouth_unlocked``.
MOUTH_PERMEABILITY: Final = 0.5
#: Default per-tick budget for cell drives (shares the command SSBO).
DEFAULT_CELL_BUDGET: Final = 192


@dataclass(slots=True)
class CellCluster:
    """One connected unlocked region (object) with full cell membership."""

    region_id: int
    name: str
    cells: npt.NDArray[np.int32]  # (N, 2) columns = x, y

    @property
    def cell_count(self) -> int:
        return int(self.cells.shape[0])

    def centroid(self) -> tuple[float, float]:
        if self.cell_count == 0:
            return 0.0, 0.0
        return float(self.cells[:, 0].mean()), float(self.cells[:, 1].mean())

    def contains(self, x: int, y: int) -> bool:
        if self.cell_count == 0:
            return False
        return bool(np.any((self.cells[:, 0] == int(x)) & (self.cells[:, 1] == int(y))))


@dataclass(slots=True)
class CellClusterIndex:
    """Runtime index of controllable cell clusters loaded from a world."""

    width: int
    height: int
    clusters: list[CellCluster] = field(default_factory=list)
    _membership: dict[tuple[int, int], str] = field(default_factory=dict)
    _sweep_cursors: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_world(cls, world: Path | str) -> "CellClusterIndex":
        """Scan the seeded ``.bds`` and rebuild full mouth/unlocked clusters."""
        path = Path(world)
        header, grid = load_bds(path)
        height, width = int(grid.shape[0]), int(grid.shape[1])
        lock = grid[..., HUMAN_LOCK_CHANNEL]
        permeability = grid[..., 25]
        unlocked = lock < 0.5
        mouth = unlocked & (permeability >= MOUTH_PERMEABILITY)
        opacity = grid[..., 11]
        hard = grid[..., 24]
        matter = (opacity > 0.05) | (hard > 0.05)
        other = matter & unlocked & ~mouth

        index = cls(width=width, height=height)
        region_id = 0
        for name, mask in (("mouth_unlocked", mouth), ("unlocked_other", other)):
            for cells in _connected_components(mask):
                if len(cells) < 8:
                    continue
                arr = np.asarray(cells, dtype=np.int32).reshape(-1, 2)
                cluster = CellCluster(
                    region_id=region_id, name=name, cells=arr
                )
                index.clusters.append(cluster)
                for x, y in cells:
                    index._membership[(int(x), int(y))] = name
                region_id += 1
        # Prefer catalog names/ids when present (keeps digests aligned).
        catalog = path.with_name("region_catalog.json")
        if catalog.is_file():
            index._annotate_from_catalog(catalog)
        return index

    def _annotate_from_catalog(self, catalog: Path) -> None:
        try:
            payload = json.loads(catalog.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        # Catalog only carries samples — membership already came from .bds.
        del payload

    def by_name(self, name: str) -> list[CellCluster]:
        needle = (name or "").strip().lower()
        return [c for c in self.clusters if c.name.lower() == needle]

    def primary_mouth(self) -> CellCluster | None:
        mouths = self.by_name("mouth_unlocked")
        if not mouths:
            return None
        return max(mouths, key=lambda c: c.cell_count)

    def region_of(self, x: int, y: int) -> str | None:
        return self._membership.get((int(x), int(y)))

    def summary(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cluster in self.clusters:
            cx, cy = cluster.centroid()
            rows.append(
                {
                    "region_id": cluster.region_id,
                    "name": cluster.name,
                    "cell_count": cluster.cell_count,
                    "centroid": [cx, cy],
                }
            )
        return rows


def _connected_components(
    mask: npt.NDArray[np.bool_],
) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    offsets = neighbor_offsets(include_diag=True)
    clusters: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in offsets:
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            clusters.append(cells)
    clusters.sort(key=len, reverse=True)
    return clusters


def cell_impulse(
    x: int,
    y: int,
    vx: float,
    vy: float,
    *,
    tick: int = 1,
    radius: float = CELL_RADIUS,
) -> VelocityImpulse:
    """Propose a single-cell ±4 velocity write (grid space)."""
    return VelocityImpulse(
        x=float(x) + 0.5,
        y=float(y) + 0.5,
        vx=float(vx),
        vy=float(vy),
        radius=float(radius),
        tick=int(tick),
    )


def toward_neighbor(
    x: int,
    y: int,
    dx: int,
    dy: int,
    *,
    speed: float = 1.0,
    tick: int = 1,
) -> VelocityImpulse:
    """Aim cell ``(x,y)`` velocity toward neighbor offset ``(dx,dy)``."""
    if dx == 0 and dy == 0:
        raise ValueError("neighbor offset cannot be (0, 0)")
    length = (float(dx) ** 2 + float(dy) ** 2) ** 0.5
    vx = float(speed) * float(dx) / length
    vy = float(speed) * float(dy) / length
    return cell_impulse(x, y, vx, vy, tick=tick)


def cluster_flow(
    cluster: CellCluster,
    vx: float,
    vy: float,
    *,
    tick: int,
    cursor: int = 0,
    budget: int = DEFAULT_CELL_BUDGET,
) -> tuple[list[VelocityImpulse], int]:
    """Round-robin push the same velocity through up to ``budget`` cells."""
    n = cluster.cell_count
    if n == 0 or budget <= 0:
        return [], cursor
    count = min(int(budget), n)
    start = int(cursor) % n
    impulses: list[VelocityImpulse] = []
    for i in range(count):
        x, y = cluster.cells[(start + i) % n]
        impulses.append(cell_impulse(int(x), int(y), vx, vy, tick=tick))
    return impulses, (start + count) % n


def neighbor_wave(
    cluster: CellCluster,
    *,
    tick: int,
    speed: float = 0.8,
    cursor: int = 0,
    budget: int = DEFAULT_CELL_BUDGET,
    include_diag: bool = False,
) -> tuple[list[VelocityImpulse], int]:
    """Each cell steps toward a cyclic cardinal/diagonal neighbor."""
    offsets = neighbor_offsets(include_diag=include_diag)
    n = cluster.cell_count
    if n == 0 or budget <= 0:
        return [], cursor
    count = min(int(budget), n)
    start = int(cursor) % n
    membership = {(int(x), int(y)) for x, y in cluster.cells.tolist()}
    impulses: list[VelocityImpulse] = []
    for i in range(count):
        x, y = (int(v) for v in cluster.cells[(start + i) % n])
        dx, dy = offsets[(start + i) % len(offsets)]
        nx, ny = x + dx, y + dy
        if (nx, ny) not in membership:
            # Fall back to any in-cluster neighbor.
            for odx, ody in offsets:
                if (x + odx, y + ody) in membership:
                    dx, dy = odx, ody
                    break
            else:
                continue
        impulses.append(toward_neighbor(x, y, dx, dy, speed=speed, tick=tick))
    return impulses, (start + count) % n


def distribute_to_nearby_cells(
    cluster: CellCluster,
    center_xy: tuple[float, float],
    velocity: tuple[float, float],
    *,
    tick: int,
    radius_cells: float = 6.0,
    budget: int = 48,
) -> list[VelocityImpulse]:
    """Spread a muscle impulse across nearby cluster cells (no single-disc)."""
    if cluster.cell_count == 0:
        return []
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dx = cluster.cells[:, 0].astype(np.float32) - cx
    dy = cluster.cells[:, 1].astype(np.float32) - cy
    dist = np.sqrt(dx * dx + dy * dy)
    order = np.argsort(dist)
    impulses: list[VelocityImpulse] = []
    r = max(float(radius_cells), 0.5)
    for idx in order[: int(budget)]:
        if float(dist[idx]) > r:
            break
        # Falloff so the epicentre carries more of the push.
        fall = max(0.15, 1.0 - float(dist[idx]) / r)
        x, y = cluster.cells[int(idx)]
        impulses.append(
            cell_impulse(
                int(x),
                int(y),
                float(velocity[0]) * fall,
                float(velocity[1]) * fall,
                tick=tick,
            )
        )
    return impulses


def to_commands(
    impulses: Sequence[VelocityImpulse],
    *,
    grid_width: int,
    grid_height: int,
) -> list[PaintCommand]:
    """Validate and pack cell impulses into AI ±4 commands."""
    cleaned = [
        validate_impulse(item, grid_width=grid_width, grid_height=grid_height)
        for item in impulses
    ]
    return impulses_to_commands(
        cleaned, grid_width=grid_width, grid_height=grid_height
    )


def parse_drive_request(
    payload: dict[str, Any],
    *,
    index: CellClusterIndex,
    tick: int,
) -> list[VelocityImpulse]:
    """Parse a bridge ``/cells/drive`` body into validated impulse proposals."""
    kind = str(payload.get("mode") or payload.get("kind") or "cell").strip().lower()
    speed = float(payload.get("speed", 1.0))
    budget = int(payload.get("budget", DEFAULT_CELL_BUDGET))
    impulses: list[VelocityImpulse] = []

    if kind in {"cell", "point"}:
        x = int(payload["x"])
        y = int(payload["y"])
        if "dx" in payload or "dy" in payload:
            impulses.append(
                toward_neighbor(
                    x,
                    y,
                    int(payload.get("dx", 0)),
                    int(payload.get("dy", 0)),
                    speed=speed,
                    tick=tick,
                )
            )
        else:
            impulses.append(
                cell_impulse(
                    x,
                    y,
                    float(payload.get("vx", 0.0)),
                    float(payload.get("vy", 0.0)),
                    tick=tick,
                )
            )
        return impulses

    if kind in {"cluster", "flow", "region"}:
        name = str(payload.get("region") or "mouth_unlocked")
        clusters = index.by_name(name)
        if not clusters:
            raise KeyError(f"unknown region {name!r}")
        cluster = max(clusters, key=lambda c: c.cell_count)
        cursor = int(payload.get("cursor", index._sweep_cursors.get(name, 0)))
        batch, cursor = cluster_flow(
            cluster,
            float(payload.get("vx", 0.0)),
            float(payload.get("vy", 0.0)),
            tick=tick,
            cursor=cursor,
            budget=budget,
        )
        index._sweep_cursors[name] = cursor
        return batch

    if kind in {"neighbor", "wave", "neighbors"}:
        name = str(payload.get("region") or "mouth_unlocked")
        clusters = index.by_name(name)
        if not clusters:
            raise KeyError(f"unknown region {name!r}")
        cluster = max(clusters, key=lambda c: c.cell_count)
        cursor = int(payload.get("cursor", index._sweep_cursors.get(name, 0)))
        batch, cursor = neighbor_wave(
            cluster,
            tick=tick,
            speed=speed,
            cursor=cursor,
            budget=budget,
            include_diag=bool(payload.get("diag", False)),
        )
        index._sweep_cursors[name] = cursor
        return batch

    if kind == "batch":
        items = payload.get("cells")
        if not isinstance(items, list) or not items:
            raise ValueError("batch mode requires cells: [{x,y,vx,vy}|{x,y,dx,dy}]")
        for item in items[:budget]:
            if not isinstance(item, dict):
                continue
            x, y = int(item["x"]), int(item["y"])
            if "dx" in item or "dy" in item:
                impulses.append(
                    toward_neighbor(
                        x,
                        y,
                        int(item.get("dx", 0)),
                        int(item.get("dy", 0)),
                        speed=float(item.get("speed", speed)),
                        tick=tick,
                    )
                )
            else:
                impulses.append(
                    cell_impulse(
                        x,
                        y,
                        float(item.get("vx", 0.0)),
                        float(item.get("vy", 0.0)),
                        tick=tick,
                    )
                )
        return impulses

    if kind == "retarget":
        # Membership rewrite is handled by the app (MouthCellPlan); no impulses.
        return []

    raise ValueError(
        f"unknown drive mode {kind!r}; use cell|cluster|neighbor|batch|retarget"
    )


__all__ = [
    "CELL_RADIUS",
    "DEFAULT_CELL_BUDGET",
    "CellCluster",
    "CellClusterIndex",
    "cell_impulse",
    "cluster_flow",
    "distribute_to_nearby_cells",
    "neighbor_wave",
    "parse_drive_request",
    "to_commands",
    "toward_neighbor",
]
