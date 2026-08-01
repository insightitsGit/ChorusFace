"""End-to-end: video → measured transition track → behavior ML fill model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiface.behavior.track import extract_transition_track
from aiface.behavior.train import fit_behavior_model
from aiface.behavior.schema import BEHAVIOR_DATASET, TRACK_NPZ, dataset_path


def train_behavior_from_video(
    video: Path,
    *,
    world_dir: Path,
    sample_fps: float = 12.0,
    landmarker_model: Path | None = None,
    val_fraction: float = 0.2,
    seed: int = 17,
) -> dict[str, Any]:
    """Learn avatar behavior: measured transitions + ML for missing data."""
    video = Path(video).resolve()
    world_dir = Path(world_dir).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)

    track = extract_transition_track(
        video,
        world_dir=world_dir,
        sample_fps=sample_fps,
        landmarker_model=landmarker_model,
    )
    meta = fit_behavior_model(
        dataset_path(world_dir),
        world_dir=world_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    meta["track"] = str(world_dir / TRACK_NPZ)
    meta["dataset"] = str(world_dir / BEHAVIOR_DATASET)
    meta["n_track_samples"] = track.n_samples
    meta["track_duration"] = track.duration
    meta["video"] = str(video)
    return meta


__all__ = ["train_behavior_from_video"]
