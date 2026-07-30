"""Steps 5 + 7 — Region objects as connected cell clusters in x,y,(z),t.

A region is not a separate mesh. It is a set of related cells that share
material / lock / part labels. Z is an optional channel signal, not a voxel axis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from amin_loop.cells import CHANNEL_NAMES, HUMAN_LOCK_CHANNEL
from amin_loop.control import neighbor_offsets


@dataclass(slots=True)
class Region:
    region_id: int
    name: str
    cells: list[tuple[int, int]]
    mean_channels: list[float] = field(default_factory=list)
    locked_frac: float = 0.0
    z_signal_mean: float = 0.0
    t_span: tuple[float, float] = (0.0, 0.0)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cell_count"] = len(self.cells)
        # Keep catalog compact — sample up to 64 cell coords.
        if len(self.cells) > 64:
            step = max(1, len(self.cells) // 64)
            payload["cells_sample"] = self.cells[::step][:64]
            del payload["cells"]
        return payload


def connected_components(
    mask: npt.NDArray[np.bool_],
    *,
    include_diag: bool = True,
) -> list[list[tuple[int, int]]]:
    """Flood-fill connected True cells on a 2D mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    offsets = neighbor_offsets(include_diag=include_diag)
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


def digest_regions_from_grid(
    grid: npt.NDArray[np.float32],
    *,
    min_cells: int = 32,
    z_channel: int = 2,
) -> list[Region]:
    """Build region catalog from a loaded .bds grid (H,W,32)."""
    if grid.ndim != 3 or grid.shape[-1] < 32:
        raise ValueError(f"expected HxWx32 grid, got {grid.shape}")
    opacity = grid[..., 11]
    hard = grid[..., 24]
    permeability = grid[..., 25]
    lock = grid[..., HUMAN_LOCK_CHANNEL]
    # Soft tissue / visible face matter (not empty void).
    matter = (opacity > 0.05) | (hard > 0.05)
    unlocked = lock < 0.5
    # The seed marks the deliberate motion flesh (mouth cavity + lips) with
    # high permeability and locks identity at 0. "Not locked" alone is NOT an
    # address — background and cheeks are unlocked too, so clustering on lock
    # state merged the mouth into a half-grid blob and the object address was
    # lost. Permeability is the fingerprint of the cells impulses may feed.
    mouth_motion = unlocked & (permeability >= 0.5)
    masks = (
        ("identity", matter & ~unlocked),
        ("mouth_unlocked", mouth_motion),
        ("unlocked_other", matter & unlocked & ~mouth_motion),
    )
    regions: list[Region] = []
    region_id = 0
    for name, mask in masks:
        for cells in connected_components(mask):
            if len(cells) < min_cells:
                continue
            ys = np.asarray([c[1] for c in cells], dtype=np.int32)
            xs = np.asarray([c[0] for c in cells], dtype=np.int32)
            patch = grid[ys, xs]
            mean = patch.mean(axis=0)
            locked = float((patch[:, HUMAN_LOCK_CHANNEL] >= 0.5).mean())
            regions.append(
                Region(
                    region_id=region_id,
                    name=name,
                    cells=cells,
                    mean_channels=[float(v) for v in mean.tolist()],
                    locked_frac=locked,
                    z_signal_mean=float(mean[z_channel]),
                )
            )
            region_id += 1
    return regions


def write_region_catalog(
    regions: list[Region],
    path: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "amin_loop.regions.v1",
        "channel_names": list(CHANNEL_NAMES),
        "region_count": len(regions),
        "regions": [r.as_dict() for r in regions],
    }
    if extra:
        payload["extra"] = extra
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "Region",
    "connected_components",
    "digest_regions_from_grid",
    "write_region_catalog",
]
