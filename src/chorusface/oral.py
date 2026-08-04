"""Expression-plate path helpers (smile / open) for the capture converter.

Speech articulation is muscles + jaw warping ``source_face.png``. When
``chorusface-capture`` has written colocated ``smile.png`` / ``open.png``, the
runtime composites those real pixels inside ``mouth_gap``. Invented teeth are
not a product path.

This module keeps the filename contract and soft-shadow utilities used by tests
and tooling. Runtime loading lives in ``chorusface.app`` / ``avatar.frag``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from chorusface.landmarks import FaceLandmarks
    from chorusface.seed import FaceBox

# Written by chorusface-capture; loaded by the avatar runtime when both exist.
OPEN_MOUTH_PLATE_NAME: str = "open.png"
SMILE_PLATE_NAME: str = "smile.png"
# Legacy alias kept so older docs/tests that said open_mouth.png still resolve
# via default_open_mouth_path → open.png (the capture contract).
LEGACY_OPEN_MOUTH_PLATE_NAME: str = "open_mouth.png"
# Legacy name; ignored by the renderer.
ORAL_LAYER_NAME: str = "oral_layer.png"


def default_open_mouth_path(world: str | Path) -> Path:
    """Colocated open-mouth plate path (``open.png`` from chorusface-capture)."""
    return Path(world).with_name(OPEN_MOUTH_PLATE_NAME)


def default_smile_plate_path(world: str | Path) -> Path:
    """Colocated smile plate path (``smile.png`` from chorusface-capture)."""
    return Path(world).with_name(SMILE_PLATE_NAME)


def default_oral_layer_path(world: str | Path) -> Path:
    """Legacy soft-matte path; no longer loaded by the avatar runtime."""
    return Path(world).with_name(ORAL_LAYER_NAME)


def build_soft_mouth_shadow(
    height: int,
    width: int,
    face: "FaceBox",
    landmarks: "FaceLandmarks | None" = None,
) -> npt.NDArray[np.float32]:
    """Build a tight dark matte at the mouth seam (utility / tests only)."""
    from chorusface.seed import FaceBox

    if height <= 0 or width <= 0:
        raise ValueError("oral layer size must be positive")
    if not isinstance(face, FaceBox):
        face = FaceBox(int(face.x), int(face.y), int(face.width), int(face.height))

    if landmarks is not None:
        mx = float(landmarks.mouth[0])
        my = float(landmarks.mouth[1])
    else:
        mx = float(face.x) + 0.50 * float(face.width)
        my = float(face.y) + 0.78 * float(face.height)

    half_w = max(float(face.width) * 0.14, 6.0)
    half_h = max(float(face.height) * 0.035, 2.5)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u = (xx - mx) / half_w
    v = (yy - my) / half_h
    radial = np.sqrt(u * u + v * v)
    ellipse = radial <= 1.0

    rgba = np.zeros((height, width, 4), dtype=np.float32)
    if not np.any(ellipse):
        return rgba

    shade = np.array([0.12, 0.05, 0.055], dtype=np.float32)
    rgba[ellipse, :3] = shade
    alpha = np.clip(1.0 - radial, 0.0, 1.0) ** 1.4
    rgba[..., 3] = np.where(ellipse, alpha * 0.55, 0.0)
    return rgba


def save_oral_layer(path: str | Path, rgba: npt.NDArray[np.float32]) -> Path:
    """Write an RGBA float plate as PNG (0–1 → 8-bit)."""
    from PIL import Image

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(rgba, 0.0, 1.0)
    image = Image.fromarray((clipped * 255.0).astype(np.uint8), mode="RGBA")
    image.save(destination)
    return destination


def load_oral_layer(
    path: str | Path, *, height: int, width: int
) -> npt.NDArray[np.float32] | None:
    """Load a PNG plate resized to the field; ``None`` if missing."""
    file = Path(path)
    if not file.is_file():
        return None
    from PIL import Image

    image = Image.open(file).convert("RGBA")
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


__all__ = [
    "LEGACY_OPEN_MOUTH_PLATE_NAME",
    "OPEN_MOUTH_PLATE_NAME",
    "ORAL_LAYER_NAME",
    "SMILE_PLATE_NAME",
    "build_soft_mouth_shadow",
    "default_open_mouth_path",
    "default_oral_layer_path",
    "default_smile_plate_path",
    "load_oral_layer",
    "save_oral_layer",
]
