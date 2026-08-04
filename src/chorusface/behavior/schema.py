"""Avatar behavior schema — measured transitions + ML fill targets.

Authority
---------
1. **Measured** ``cell_transition_track`` from the upload (group motion over time)
2. **ML fill** ``behavior_model`` trained on that track + audio (gaps / live speech)
3. Hand ``VISEME_FLOW`` tables only as last resort

We do **not** invent per-cell optical flow. Targets are mouth **group** controls
derived from landmarks on the user's video — honest, sparse, replayable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

TRACK_NPZ: Final = "cell_transition_track.npz"
TRACK_JSON: Final = "cell_transition_track.json"
BEHAVIOR_MODEL: Final = "behavior_model.joblib"
BEHAVIOR_META: Final = "behavior_model.meta.json"
BEHAVIOR_DATASET: Final = "behavior_dataset.npz"

#: Live-vector audio feature dim/history (shared train features).
HISTORY: Final = 5
FEATURE_DIM: Final = 8

#: Measured + predicted mouth-group controls (not RGB).
CONTROL_NAMES: Final[tuple[str, ...]] = (
    "openness_n",
    "jaw_n",
    "width_n",
    "upper_lip_dy",
    "lower_lip_dy",
    "corner_dx",
    "teeth_reveal",
    "cavity_n",
)
CONTROL_DIM: Final = len(CONTROL_NAMES)

#: If neighbors in the track are farther apart than this, treat as a gap for ML.
GAP_SECONDS: Final = 0.12


@dataclass(frozen=True, slots=True)
class BehaviorState:
    """One resolved behavior sample for the GPU / cell-plan path."""

    openness_n: float = 0.0
    jaw_n: float = 0.0
    width_n: float = 0.0
    upper_lip_dy: float = 0.0
    lower_lip_dy: float = 0.0
    corner_dx: float = 0.0
    teeth_reveal: float = 0.0
    cavity_n: float = 0.0
    #: measured | measured_lerp | ml_fill | table
    source: str = "zero"
    #: Transformation from previous measured sample (0 when unknown).
    delta_open: float = 0.0
    delta_width: float = 0.0

    def as_vector(self) -> tuple[float, ...]:
        return (
            self.openness_n,
            self.jaw_n,
            self.width_n,
            self.upper_lip_dy,
            self.lower_lip_dy,
            self.corner_dx,
            self.teeth_reveal,
            self.cavity_n,
        )

    def flow(self) -> tuple[float, float, float]:
        """Map group controls → MouthCellPlan (open, width, round)."""
        open_n = max(float(self.openness_n), float(self.cavity_n) * 0.85)
        width_n = float(self.width_n)
        # Round when corners pull in (low width with moderate open).
        round_n = float(
            max(0.0, 0.55 - width_n) * max(0.0, min(open_n, 0.8)) / 0.55
        )
        return (
            max(0.0, min(1.0, open_n)),
            max(0.0, min(1.0, width_n)),
            max(0.0, min(1.0, round_n)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def world_dir(world: Path | str) -> Path:
    path = Path(world)
    return path if path.is_dir() else path.parent


def track_npz_path(world: Path | str) -> Path:
    return world_dir(world) / TRACK_NPZ


def track_json_path(world: Path | str) -> Path:
    return world_dir(world) / TRACK_JSON


def model_path(world: Path | str) -> Path:
    return world_dir(world) / BEHAVIOR_MODEL


def meta_path(world: Path | str) -> Path:
    return world_dir(world) / BEHAVIOR_META


def dataset_path(world: Path | str) -> Path:
    return world_dir(world) / BEHAVIOR_DATASET


def landmarks_to_controls(
    *,
    openness_n: float,
    width_n: float,
    teeth_n: float = 0.0,
) -> list[float]:
    """Honest landmark → group controls (no invented tissue paths)."""
    open_n = float(max(0.0, min(1.0, openness_n)))
    width = float(max(0.0, min(1.0, width_n)))
    teeth = float(max(0.0, min(1.0, teeth_n)))
    # Upper lip rises (−y in image; +lip in detect) with open; lower drops.
    upper = -open_n
    lower = open_n
    corner = width
    teeth_reveal = max(teeth, open_n * 0.65)
    cavity = open_n
    jaw = open_n
    return [open_n, jaw, width, upper, lower, corner, teeth_reveal, cavity]


__all__ = [
    "BEHAVIOR_DATASET",
    "BEHAVIOR_META",
    "BEHAVIOR_MODEL",
    "CONTROL_DIM",
    "CONTROL_NAMES",
    "FEATURE_DIM",
    "GAP_SECONDS",
    "HISTORY",
    "TRACK_JSON",
    "TRACK_NPZ",
    "BehaviorState",
    "dataset_path",
    "landmarks_to_controls",
    "meta_path",
    "model_path",
    "track_json_path",
    "track_npz_path",
    "world_dir",
]
