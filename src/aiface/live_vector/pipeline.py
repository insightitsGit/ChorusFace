"""End-to-end: video → extract → train avatar live-vector model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiface.live_vector.extract import extract_live_vectors
from aiface.live_vector.train import fit_live_vector_model


def train_avatar_from_video(
    video: Path,
    *,
    world_dir: Path,
    sample_fps: float = 12.0,
    landmarker_model: Path | None = None,
    val_fraction: float = 0.2,
    seed: int = 17,
) -> dict[str, Any]:
    """From-scratch train on one capture take."""
    video = Path(video).resolve()
    world_dir = Path(world_dir).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)

    extracted = extract_live_vectors(
        video,
        world_dir=world_dir,
        sample_fps=sample_fps,
        landmarker_model=landmarker_model,
    )
    meta = fit_live_vector_model(
        extracted.dataset,
        world_dir=world_dir,
        val_fraction=val_fraction,
        seed=seed,
    )
    meta["trajectory"] = str(extracted.trajectory)
    meta["n_samples"] = extracted.n_samples
    meta["video"] = str(video)
    return meta
