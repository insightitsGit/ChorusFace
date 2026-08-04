"""End-to-end: video → measured transition track → behavior ML fill model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chorusface.behavior.track import extract_transition_track
from chorusface.behavior.train import fit_behavior_model
from chorusface.behavior.schema import BEHAVIOR_DATASET, TRACK_NPZ, dataset_path
from chorusface.observation.extract import (
    extract_avatar_observations,
    save_avatar_observations,
)
from chorusface.observation.schema import OBS_JSON


def train_behavior_from_video(
    video: Path,
    *,
    world_dir: Path,
    sample_fps: float = 12.0,
    landmarker_model: Path | None = None,
    val_fraction: float = 0.2,
    seed: int = 17,
) -> dict[str, Any]:
    """Learn avatar behavior: observations + transitions + ML for gaps."""
    video = Path(video).resolve()
    world_dir = Path(world_dir).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)

    # Ground truth looks (smile/open GPU vectors) from this avatar's plates.
    observations = extract_avatar_observations(world_dir)
    obs_path = save_avatar_observations(world_dir, observations)

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
    meta["observations"] = str(obs_path)
    meta["smile_vector"] = list(observations.smile_vector)
    meta["open_vector"] = list(observations.open_vector)
    meta["smile_gpu"] = (
        observations.look("smile").gpu.as_dict()
        if observations.look("smile")
        else {}
    )
    meta["n_track_samples"] = track.n_samples
    meta["track_duration"] = track.duration
    meta["video"] = str(video)
    meta["obs_schema"] = OBS_JSON
    return meta


__all__ = ["train_behavior_from_video"]
