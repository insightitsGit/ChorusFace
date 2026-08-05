"""Frozen group→cell expand matrix W (D8 / F11)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chorusface.vowel.schema import GROUP_DIM

_MAGIC = b"WEXP"
_VERSION = 1

# Which control channels drive which named regions (catalog keys, fuzzy).
_CHANNEL_REGIONS: dict[int, tuple[str, ...]] = {
    0: ("eye", "eyes", "lid"),
    1: ("eye", "eyes", "lid"),
    2: ("brow", "eyebrow", "eyebrows"),
    3: ("brow", "eyebrow", "eyebrows"),
    4: ("mouth", "cavity", "interior"),
    5: ("lip", "lips", "mouth"),
    6: ("lip", "lips", "mouth"),
    7: ("teeth", "tooth", "mouth"),
    8: ("jaw", "jaws", "chin", "mouth"),
}

# Primary displacement axis per channel: 0=vx, 1=vy
_CHANNEL_AXIS: dict[int, int] = {
    0: 1,
    1: 0,
    2: 1,
    3: 0,
    4: 1,
    5: 0,
    6: 1,
    7: 1,
    8: 1,
}


def _region_match(name: str, keys: tuple[str, ...]) -> bool:
    n = name.lower()
    return any(k in n for k in keys)


def author_w_from_catalog(
    catalog: dict[str, Any],
    *,
    grid_w: int = 256,
    grid_h: int = 256,
    sigma_scale: float = 0.15,
) -> tuple[NDArray[np.float32], list[tuple[int, int]]]:
    """Build dense W (9 × N) and cell xy list from region_catalog JSON.

    Catalog expected shape (amin_loop style): regions with name + cells [[x,y],…]
    or {name: {cells: ...}}.
    """
    regions = catalog.get("regions") if isinstance(catalog, dict) else None
    if regions is None and isinstance(catalog, dict):
        regions = catalog

    cell_set: dict[tuple[int, int], None] = {}
    region_cells: dict[str, list[tuple[int, int]]] = {}

    if isinstance(regions, list):
        items = regions
    elif isinstance(regions, dict):
        items = [{"name": k, **(v if isinstance(v, dict) else {"cells": v})} for k, v in regions.items()]
    else:
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("id") or "")
        cells_raw = item.get("cells") or item.get("points") or []
        pts: list[tuple[int, int]] = []
        for c in cells_raw:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                x, y = int(c[0]), int(c[1])
                pts.append((x, y))
                cell_set[(x, y)] = None
        if name and pts:
            region_cells[name] = pts

    if not cell_set:
        # synthetic mouth ROI for lab/tests when catalog empty
        for y in range(140, 200):
            for x in range(90, 166):
                cell_set[(x, y)] = None
        region_cells["mouth"] = list(cell_set.keys())
        region_cells["lips"] = [(x, y) for x, y in cell_set if 150 <= y <= 185]
        region_cells["jaw"] = [(x, y) for x, y in cell_set if y >= 175]
        region_cells["eyes"] = [(x, y) for y in range(70, 100) for x in range(80, 180)]
        region_cells["eyebrows"] = [(x, y) for y in range(55, 75) for x in range(80, 180)]
        region_cells["teeth"] = [(x, y) for x, y in cell_set if 155 <= y <= 170]
        for pts in region_cells.values():
            for p in pts:
                cell_set[p] = None

    cells = sorted(cell_set.keys())
    n = len(cells)
    index = {c: i for i, c in enumerate(cells)}
    W = np.zeros((GROUP_DIM, n), dtype=np.float32)

    for ch, keys in _CHANNEL_REGIONS.items():
        matched: list[tuple[int, int]] = []
        for rname, pts in region_cells.items():
            if _region_match(rname, keys):
                matched.extend(pts)
        if not matched:
            continue
        xs = np.array([p[0] for p in matched], dtype=np.float64)
        ys = np.array([p[1] for p in matched], dtype=np.float64)
        cx, cy = float(xs.mean()), float(ys.mean())
        rad = float(np.hypot(xs - cx, ys - cy).max()) + 1e-3
        sigma = max(1.0, sigma_scale * rad)
        for x, y in matched:
            i = index[(x, y)]
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            W[ch, i] = np.exp(-d2 / (2.0 * sigma * sigma))

    # column normalize Σ|W| <= 1
    col = np.sum(np.abs(W), axis=0)
    col = np.maximum(col, 1e-6)
    over = col > 1.0
    W[:, over] /= col[over]
    return W, cells


def expand_controls(
    controls: NDArray[np.floating],
    W: NDArray[np.floating],
    cells: list[tuple[int, int]],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Map 9D → per-cell vx, vy (phase-1 velocity)."""
    c = np.asarray(controls, dtype=np.float32).reshape(GROUP_DIM)
    # drive = W.T @ c  → N
    drive = W.T @ c
    n = len(cells)
    vx = np.zeros(n, dtype=np.float32)
    vy = np.zeros(n, dtype=np.float32)
    for ch in range(GROUP_DIM):
        axis = _CHANNEL_AXIS.get(ch, 1)
        contrib = W[ch] * float(c[ch])
        if axis == 0:
            vx += contrib
        else:
            vy += contrib
    # scale to modest grid velocity
    vx *= 0.35
    vy *= 0.35
    return vx, vy


def save_wexpand(
    path: str | Path,
    W: NDArray[np.floating],
    cells: list[tuple[int, int]],
    *,
    decoder_ver: int = 1,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    W32 = np.asarray(W, dtype=np.float32)
    n = len(cells)
    with path.open("wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<HHI", _VERSION, decoder_ver & 0xFFFF, n))
        f.write(struct.pack("<II", W32.shape[0], W32.shape[1]))
        for x, y in cells:
            f.write(struct.pack("<HH", x & 0xFFFF, y & 0xFFFF))
        f.write(W32.tobytes(order="C"))


def load_wexpand(
    path: str | Path,
) -> tuple[NDArray[np.float32], list[tuple[int, int]], int]:
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != _MAGIC:
            raise ValueError("bad wexpand magic")
        ver, decoder_ver, n = struct.unpack("<HHI", f.read(8))
        rows, cols = struct.unpack("<II", f.read(8))
        cells = [struct.unpack("<HH", f.read(4)) for _ in range(n)]
        raw = f.read(rows * cols * 4)
        W = np.frombuffer(raw, dtype=np.float32).reshape(rows, cols).copy()
    return W, [(int(x), int(y)) for x, y in cells], int(decoder_ver)


def load_catalog(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
