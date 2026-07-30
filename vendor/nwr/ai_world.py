"""AI-facing world inspection and export helpers.

These APIs never write GPU memory. They read a CPU-side grid snapshot and
return compact JSON suitable for external assistants. The runtime remains the
owner of truth; this module is an observation layer only.
"""

from __future__ import annotations

from typing import Any, Final, Mapping

import numpy as np
import numpy.typing as npt

from ai_commands import (
    CONTROL_ACTIONS,
    DEFAULT_PRIORITY,
    schema_for_authority,
)
from bds_format import ANCHORS, CHANNEL_SCHEMA, FORMAT_VERSION, PRIORITY_LEVELS
from entities import ENTITY_KINDS, entity_kind_catalog

SUPPORTED_COMMANDS: Final[tuple[str, ...]] = (
    "paint",
    "erase",
    "set_material",
    "increase_temperature",
    "decrease_temperature",
    "spawn_entity",
    "remove_entity",
    *CONTROL_ACTIONS,
)

# Capabilities models routinely invent. Naming them with a reason is cheaper
# than letting a request fail somewhere less legible.
UNSUPPORTED_COMMANDS: Final[dict[str, str]] = {
    "write_cells": (
        "no raw per-cell vector write exists; every write goes through "
        "paint/erase so authority and human locks can be enforced"
    ),
    "run_script": "the runtime executes no caller-supplied code",
    "set_gravity": (
        "there is no global force term; use paint velocity or spawn_entity "
        "with an emitter instead"
    ),
}

MATERIAL_NAMES: Final[tuple[str, ...]] = tuple(ANCHORS)


def build_ai_metadata(
    *,
    world_name: str = "Neural World",
    description: str = "Interactive semantic simulation",
    compact: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the optional ``ai_metadata`` block describing this world.

    Command reasons and the entity-kind catalog are build constants, not
    per-world facts, so ``compact=True`` reduces them to plain name lists. That
    is the form :func:`bds_format.make_header` embeds, which keeps the 4 KiB
    header budget spendable on world-specific metadata. Callers that want the
    detail read the full form from the runtime or ``GET /context``.
    """
    metadata: dict[str, Any] = {
        "schema": "nwr-1.0",
        "world_name": world_name,
        "description": description,
        "cell_schema": {
            "0-7": "physics",
            "8-15": "material",
            "16-23": "intent",
            "24-31": "rules",
        },
        "channel_schema": CHANNEL_SCHEMA,
        "materials": [
            {"id": index, "name": name.replace("_", " ").title()}
            for index, name in enumerate(MATERIAL_NAMES)
        ],
        "supported_commands": list(SUPPORTED_COMMANDS),
        "unsupported_commands": dict(UNSUPPORTED_COMMANDS),
        "entity_kinds": entity_kind_catalog(),
        "priority_levels": sorted(PRIORITY_LEVELS, key=PRIORITY_LEVELS.get),
        "architecture": {
            "world_owner": "runtime",
            "ai_role": "propose deterministic commands",
            "gpu_writes": "runtime validated commands only",
        },
    }
    if compact:
        metadata["unsupported_commands"] = sorted(UNSUPPORTED_COMMANDS)
        metadata["entity_kinds"] = sorted(ENTITY_KINDS)
        metadata["full_schema"] = (
            "ai_world.build_ai_metadata() or GET /context on the AI bridge"
        )
        del metadata["channel_schema"]
    if extra is not None:
        metadata.update(dict(extra))
    return metadata


def classify_cells(
    grid: npt.NDArray[np.float32],
    *,
    occupancy_threshold: float = 0.02,
) -> tuple[npt.NDArray[np.int32], dict[str, int]]:
    """Nearest-anchor labels for every cell, plus raw counts."""
    if grid.ndim != 3:
        raise ValueError("Grid must have shape (height, width, channels)")
    height, width, _channels = grid.shape
    labels = np.zeros((height, width), dtype=np.int32)
    density = grid[..., 3]
    material_norm = np.linalg.norm(grid[..., 8:16], axis=-1)
    occupied = (np.abs(density) > occupancy_threshold) | (
        material_norm > occupancy_threshold
    )
    counts = {name: 0 for name in MATERIAL_NAMES}
    counts["vacuum"] = int((~occupied).sum())
    if occupied.any():
        names = [name for name in MATERIAL_NAMES if name != "vacuum"]
        anchors = np.asarray([ANCHORS[name] for name in names], dtype=np.float32)
        samples = grid[occupied].reshape(-1, grid.shape[-1])
        distances = np.linalg.norm(samples[:, None, :] - anchors[None, :, :], axis=-1)
        nearest = np.argmin(distances, axis=1)
        occupied_labels = nearest + 1  # 0 reserved for vacuum
        labels[occupied] = occupied_labels
        for index, name in enumerate(names):
            counts[name] = int((nearest == index).sum())
    return labels, counts


def generate_ai_summary(
    grid: npt.NDArray[np.float32],
    *,
    tick: int = 0,
    occupancy_threshold: float = 0.02,
) -> dict[str, Any]:
    """Compact semantic summary for an LLM; never dumps every cell."""
    labels, counts = classify_cells(grid, occupancy_threshold=occupancy_threshold)
    height, width = labels.shape
    total = height * width
    locked = grid[..., 31] >= 0.5
    materials = {
        name.replace("_", " ").title(): round(100.0 * count / total, 4)
        for name, count in counts.items()
        if total
    }
    boxes: list[dict[str, Any]] = []
    for index, name in enumerate(MATERIAL_NAMES):
        if name == "vacuum":
            continue
        mask = labels == index
        box = _bounding_box(mask)
        if box is not None:
            boxes.append(
                {
                    "material": name.replace("_", " ").title(),
                    "min": box["min"],
                    "max": box["max"],
                    "cells": int(mask.sum()),
                }
            )
    return {
        "grid": [int(width), int(height)],
        "tick": int(tick),
        "cells": int(total),
        "locked_cells": int(locked.sum()),
        "materials": materials,
        "average_temperature": round(float(grid[..., 6].mean()), 5),
        "average_energy": round(float(grid[..., 7].mean()), 5),
        "bounding_boxes": boxes,
        "format_version": FORMAT_VERSION,
    }


def inspect_region(
    grid: npt.NDArray[np.float32],
    x: float,
    y: float,
    radius: float,
) -> dict[str, Any]:
    """Return semantic statistics for a circular region around ``(x, y)``."""
    if grid.ndim != 3:
        raise ValueError("Grid must have shape (height, width, channels)")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    height, width = grid.shape[0], grid.shape[1]
    if not 0.0 <= x <= width or not 0.0 <= y <= height:
        raise ValueError(f"Center ({x}, {y}) is outside the world")

    yy, xx = np.ogrid[:height, :width]
    mask = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= radius**2
    if not mask.any():
        raise ValueError("Region does not cover any cell centres")

    sample = grid[mask]
    labels, _counts = classify_cells(grid)
    region_labels = labels[mask]
    material_counts = {
        name.replace("_", " ").title(): int((region_labels == index).sum())
        for index, name in enumerate(MATERIAL_NAMES)
    }
    dominant_index = int(np.bincount(region_labels, minlength=len(MATERIAL_NAMES)).argmax())
    return {
        "center": [float(x), float(y)],
        "radius": float(radius),
        "cells": int(mask.sum()),
        "locked_cells": int((sample[..., 31] >= 0.5).sum()),
        "average_temperature": round(float(sample[..., 6].mean()), 5),
        "average_energy": round(float(sample[..., 7].mean()), 5),
        "average_velocity": [
            round(float(sample[..., 0].mean()), 5),
            round(float(sample[..., 1].mean()), 5),
            round(float(sample[..., 2].mean()), 5),
        ],
        "average_density": round(float(sample[..., 3].mean()), 5),
        "average_material_albedo": [
            round(float(sample[..., 8].mean()), 5),
            round(float(sample[..., 9].mean()), 5),
            round(float(sample[..., 10].mean()), 5),
        ],
        "materials": material_counts,
        "dominant_material": MATERIAL_NAMES[dominant_index].replace("_", " ").title(),
        "hard_surface_cells": int((sample[..., 24] >= 0.5).sum()),
    }


def export_ai_context(
    grid: npt.NDArray[np.float32],
    *,
    tick: int = 0,
    world_name: str = "Neural World",
    description: str = "Interactive semantic simulation",
    include_occupancy_map: bool = True,
    entities: Mapping[str, Any] | None = None,
    authority: int = DEFAULT_PRIORITY,
) -> dict[str, Any]:
    """Bundle metadata, summary, and schema for drag-into-assistant workflows.

    `authority` is the level the reader will be granted when it submits. The
    embedded grammar is narrowed to match, so a model planning from this bundle
    never builds on a command it would be refused.
    """
    from ai_bridge import summarize_world

    metadata = build_ai_metadata(world_name=world_name, description=description)
    summary = generate_ai_summary(grid, tick=tick)
    detailed = summarize_world(grid)
    if not include_occupancy_map:
        detailed = {key: value for key, value in detailed.items() if key != "occupancy_map"}
    schema = schema_for_authority(authority)
    return {
        "metadata": metadata,
        "summary": summary,
        "statistics": detailed,
        "materials": metadata["materials"],
        "supported_commands": metadata["supported_commands"],
        "unsupported_commands": metadata["unsupported_commands"],
        "command_schema": schema,
        "paintable_categories": list(schema["actions"]["paint"]["category"]),
        "entity_kinds": metadata["entity_kinds"],
        "entities": dict(entities) if entities is not None else {"count": 0},
    }


def render_preview(
    grid: npt.NDArray[np.float32],
    *,
    resolution: int = 1024,
    use_neural_material: bool = False,
) -> bytes:
    """Render an optional 1024×1024 PNG preview from a CPU grid snapshot.

    Requires an OpenGL 4.3 standalone context. Assistants should prefer
    :func:`generate_ai_summary` / :func:`export_ai_context` over pixels.
    """
    from export_video import HeadlessRenderer

    with HeadlessRenderer(
        resolution=resolution,
        grid=grid,
        use_neural_material=use_neural_material,
    ) as renderer:
        return renderer.render_frame()


def _bounding_box(mask: npt.NDArray[np.bool_]) -> dict[str, list[int]] | None:
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    return {
        "min": [int(columns[0]), int(rows[0])],
        "max": [int(columns[-1]), int(rows[-1])],
    }


__all__ = [
    "MATERIAL_NAMES",
    "SUPPORTED_COMMANDS",
    "UNSUPPORTED_COMMANDS",
    "build_ai_metadata",
    "classify_cells",
    "export_ai_context",
    "generate_ai_summary",
    "inspect_region",
    "render_preview",
]
