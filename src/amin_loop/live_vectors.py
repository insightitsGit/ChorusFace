"""Step 9 — Video → live control vectors (re-export + train helper).

Delegates to ``aiface.live_vector`` — same GPU recipe at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiface.live_vector import LiveVectorDriver, train_avatar_from_video
from aiface.live_vector.schema import LiveControlVector


def train_from_video(
    video: Path,
    *,
    world_dir: Path,
    sample_fps: float = 12.0,
    landmarker_model: Path | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    return train_avatar_from_video(
        video,
        world_dir=world_dir,
        sample_fps=sample_fps,
        landmarker_model=landmarker_model,
        seed=seed,
    )


__all__ = [
    "LiveControlVector",
    "LiveVectorDriver",
    "train_from_video",
]
