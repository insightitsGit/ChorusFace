"""Avatar observation dataset — measured look/cell truth the ML fills between.

The behavior model was filling gaps in landmark proxies. That left a hole:
we never stored how **this avatar's smile/open** shows on the GPU path
(``smile.png`` + ``avatar_mouth_pose.w``, etc.) or the rest→smile delta.

This module is that missing ground truth:

* **Look keyframes** from capture (rest / smile / open / surprise)
* **GPU display vectors** — the exact uniforms the shader reads for that look
* **Plate deltas** — measured pixel stats of smile/open vs rest (mouth ROI)
* **Cell geometry** — mouth_unlocked groups from the ``.bds`` (identity field)

ML fill may only interpolate / predict **between** these observations.
It must not invent a smile vector that was never measured on the avatar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

OBS_JSON: Final = "avatar_observations.json"
OBS_NPZ: Final = "avatar_observations.npz"
OBS_SCHEMA: Final = "chorusface.avatar_observations.v1"

LOOK_ROLES: Final[tuple[str, ...]] = ("rest", "smile", "open", "surprise")

#: GPU uniforms that make a look visible (avatar.frag / app upload).
GPU_UNIFORM_NAMES: Final[tuple[str, ...]] = (
    "avatar_mouth_pose.w",  # smile drive → smile.png
    "avatar_mouth_pose.y",  # openness
    "avatar_plate_blend.y",  # open/atlas amount
    "avatar_jaw.z",  # jaw angle
    "avatar_recipe.z",  # atlas strength
    "avatar_expr_state.z",  # surprise plate blend
    "avatar_expr_state.y",  # brow procedural
)


@dataclass(frozen=True, slots=True)
class GpuLookVector:
    """What we write to the GPU for one measured look (not invented RGB)."""

    smile_drive: float = 0.0
    open_drive: float = 0.0
    jaw: float = 0.0
    atlas_amount: float = 0.0
    expr_blend: float = 0.0
    brow_raise: float = 0.0
    plate_role: str = "rest"
    #: Texture the fragment shader samples for this look.
    plate_texture: str = "source_face.png"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_vector(self) -> list[float]:
        return [
            float(self.smile_drive),
            float(self.open_drive),
            float(self.jaw),
            float(self.atlas_amount),
            float(self.expr_blend),
            float(self.brow_raise),
        ]


@dataclass(frozen=True, slots=True)
class PlateDelta:
    """Measured mouth-ROI plate stats vs rest (from PNG pixels)."""

    mean_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    delta_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    delta_luma: float = 0.0
    mouth_energy: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_rgb": [round(v, 4) for v in self.mean_rgb],
            "delta_rgb": [round(v, 4) for v in self.delta_rgb],
            "delta_luma": round(float(self.delta_luma), 4),
            "mouth_energy": round(float(self.mouth_energy), 4),
        }


@dataclass(frozen=True, slots=True)
class CellGeometryObs:
    """Mouth cell geometry from the seeded ``.bds`` (field identity)."""

    mouth_cell_count: int = 0
    centroid: tuple[float, float] = (0.0, 0.0)
    group_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mouth_cell_count": int(self.mouth_cell_count),
            "centroid": [float(self.centroid[0]), float(self.centroid[1])],
            "group_counts": {k: int(v) for k, v in self.group_counts.items()},
        }


@dataclass(frozen=True, slots=True)
class LookObservation:
    """One measured avatar look (rest/smile/open/surprise or talk sample)."""

    role: str
    time_seconds: float
    frame_index: int
    # Landmark metrics from capture (avatar information).
    mouth_open: float
    smile_width: float
    teeth: float
    brow_raise: float
    lid_open: float
    gpu: GpuLookVector
    plate: PlateDelta
    # Group control anchors (same space as behavior track).
    controls: tuple[float, ...] = ()
    # Delta from rest controls (what gap-fill interpolates toward).
    delta_from_rest: tuple[float, ...] = ()
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "time_seconds": float(self.time_seconds),
            "frame_index": int(self.frame_index),
            "landmarks": {
                "mouth_open": round(float(self.mouth_open), 4),
                "smile_width": round(float(self.smile_width), 4),
                "teeth": round(float(self.teeth), 4),
                "brow_raise": round(float(self.brow_raise), 4),
                "lid_open": round(float(self.lid_open), 4),
            },
            "gpu": self.gpu.as_dict(),
            "gpu_uniforms": {
                "avatar_mouth_pose.w": self.gpu.smile_drive,
                "avatar_mouth_pose.y": self.gpu.open_drive,
                "avatar_plate_blend.y": self.gpu.open_drive,
                "avatar_jaw.z": self.gpu.jaw,
                "avatar_expr_state.z": self.gpu.expr_blend,
                "avatar_expr_state.y": self.gpu.brow_raise,
                "plate_texture": self.gpu.plate_texture,
            },
            "plate": self.plate.as_dict(),
            "controls": [round(float(v), 4) for v in self.controls],
            "delta_from_rest": [round(float(v), 4) for v in self.delta_from_rest],
            "notes": self.notes,
        }


@dataclass(slots=True)
class AvatarObservationSet:
    """Full observation package for one adopted world."""

    looks: list[LookObservation] = field(default_factory=list)
    cells: CellGeometryObs = field(default_factory=CellGeometryObs)
    smile_vector: list[float] = field(default_factory=list)
    open_vector: list[float] = field(default_factory=list)
    video: str = ""
    root: Path | None = None

    def look(self, role: str) -> LookObservation | None:
        for item in self.looks:
            if item.role == role:
                return item
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": OBS_SCHEMA,
            "video": self.video,
            "gpu_uniform_contract": list(GPU_UNIFORM_NAMES),
            "note": (
                "Measured avatar looks. smile_vector / open_vector are "
                "delta_from_rest for those roles. ML fill interpolates "
                "between these observations — it does not invent them."
            ),
            "cells": self.cells.as_dict(),
            "smile_vector": [round(float(v), 4) for v in self.smile_vector],
            "open_vector": [round(float(v), 4) for v in self.open_vector],
            "looks": [look.as_dict() for look in self.looks],
        }


def world_dir(world: Path | str) -> Path:
    path = Path(world)
    return path if path.is_dir() else path.parent


def obs_json_path(world: Path | str) -> Path:
    return world_dir(world) / OBS_JSON


def obs_npz_path(world: Path | str) -> Path:
    return world_dir(world) / OBS_NPZ


__all__ = [
    "GPU_UNIFORM_NAMES",
    "LOOK_ROLES",
    "OBS_JSON",
    "OBS_NPZ",
    "OBS_SCHEMA",
    "AvatarObservationSet",
    "CellGeometryObs",
    "GpuLookVector",
    "LookObservation",
    "PlateDelta",
    "obs_json_path",
    "obs_npz_path",
    "world_dir",
]
