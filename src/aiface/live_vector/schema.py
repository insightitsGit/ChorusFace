"""Live control vector schema — knobs for the GPU display path, not RGB."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

MODEL_NAME: Final = "live_vector_model.joblib"
META_NAME: Final = "live_vector_model.meta.json"
DATASET_NAME: Final = "live_vector_dataset.npz"
TRAJECTORY_NAME: Final = "live_vector_trajectory.json"
DATASET_CSV_NAME: Final = "live_vector_dataset.csv"

HISTORY: Final = 5
FEATURE_DIM: Final = 8
CONTROL_DIM: Final = 3
CONTROL_NAMES: Final = ("openness_n", "jaw_n", "width_n")

PLATE_OPEN_FLOOR: Final = 0.12
CLOSED_VISEMES: Final[frozenset[str]] = frozenset({"REST", "CLOSED", "PP", "MM"})
OPEN_VISEMES: Final[frozenset[str]] = frozenset({"AA", "AH", "OH", "OU"})


@dataclass(frozen=True, slots=True)
class LiveControlVector:
    """One-frame controls that the existing GPU recipe understands."""

    openness_n: float = 0.0
    jaw_n: float = 0.0
    width_n: float = 0.0
    plate_gate: float = 0.0
    source: str = "zero"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def world_dir(world: Path) -> Path:
    return world if world.is_dir() else world.parent


def model_path(world: Path) -> Path:
    return world_dir(world) / MODEL_NAME


def meta_path(world: Path) -> Path:
    return world_dir(world) / META_NAME


def dataset_path(world: Path) -> Path:
    return world_dir(world) / DATASET_NAME


def trajectory_path(world: Path) -> Path:
    return world_dir(world) / TRAJECTORY_NAME


def plate_gate(openness_n: float) -> float:
    return 1.0 if float(openness_n) >= PLATE_OPEN_FLOOR else 0.0
