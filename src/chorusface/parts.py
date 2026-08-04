"""Anatomical face-part atlas for controllable piece-wise animation.

The photograph is split into labelled pieces (eyes, brows, upper/lower lips,
mouth cavity, nose, static face). The GPU renderer moves those pieces; the
Master-Locked field identity stays unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from chorusface.seed import FaceBox, build_region_masks

# Part IDs written into the atlas R channel (0..1 = id/PART_ID_SCALE).
PART_NONE: Final = 0
PART_FACE: Final = 1
PART_NOSE: Final = 2
PART_LEFT_BROW: Final = 3
PART_RIGHT_BROW: Final = 4
PART_LEFT_EYE: Final = 5
PART_RIGHT_EYE: Final = 6
PART_UPPER_LIP: Final = 7
PART_LOWER_LIP: Final = 8
PART_MOUTH_CAVITY: Final = 9
PART_ID_SCALE: Final = 10.0

PART_NAMES: Final[dict[int, str]] = {
    PART_FACE: "face",
    PART_NOSE: "nose",
    PART_LEFT_BROW: "left_brow",
    PART_RIGHT_BROW: "right_brow",
    PART_LEFT_EYE: "left_eye",
    PART_RIGHT_EYE: "right_eye",
    PART_UPPER_LIP: "upper_lip",
    PART_LOWER_LIP: "lower_lip",
    PART_MOUTH_CAVITY: "mouth_cavity",
}


@dataclass(frozen=True, slots=True)
class FacePartMasks:
    """Boolean masks in image space (y down) before the world y-flip."""

    face: npt.NDArray[np.bool_]
    nose: npt.NDArray[np.bool_]
    left_brow: npt.NDArray[np.bool_]
    right_brow: npt.NDArray[np.bool_]
    left_eye: npt.NDArray[np.bool_]
    right_eye: npt.NDArray[np.bool_]
    upper_lip: npt.NDArray[np.bool_]
    lower_lip: npt.NDArray[np.bool_]
    mouth_cavity: npt.NDArray[np.bool_]

    def counts(self) -> dict[str, int]:
        return {
            "face": int(self.face.sum()),
            "nose": int(self.nose.sum()),
            "left_brow": int(self.left_brow.sum()),
            "right_brow": int(self.right_brow.sum()),
            "left_eye": int(self.left_eye.sum()),
            "right_eye": int(self.right_eye.sum()),
            "upper_lip": int(self.upper_lip.sum()),
            "lower_lip": int(self.lower_lip.sum()),
            "mouth_cavity": int(self.mouth_cavity.sum()),
        }


@dataclass(frozen=True, slots=True)
class FacePartAtlas:
    """GPU-ready atlas: RGB photo + part-id alpha, y-up like the world grid."""

    rgba: npt.NDArray[np.float32]
    masks: FacePartMasks
    face: FaceBox
    anchors: dict[str, tuple[float, float]]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.rgba.shape)  # type: ignore[return-value]


def build_face_part_masks(
    height: int,
    width: int,
    face: FaceBox,
) -> FacePartMasks:
    """Split a frontal face box into independently movable anatomical pieces."""
    base = build_region_masks(height, width, face)
    yy, xx = np.mgrid[0:height, 0:width]
    fx0, fy0 = float(face.x), float(face.y)
    fw, fh = float(max(face.width, 1)), float(max(face.height, 1))
    u = (xx.astype(np.float64) - fx0) / fw
    v = (yy.astype(np.float64) - fy0) / fh

    # Fractions are face-box UV with v increasing downward (image space).
    # Centres sit on measured pupil darkness (~v=0.47 on the current seed).
    left_eye = base.face & (
        ((u - 0.30) / 0.10) ** 2 + ((v - 0.472) / 0.040) ** 2 <= 1.0
    )
    right_eye = base.face & (
        ((u - 0.70) / 0.10) ** 2 + ((v - 0.472) / 0.040) ** 2 <= 1.0
    )
    left_brow = base.face & (
        ((u - 0.30) / 0.12) ** 2 + ((v - 0.395) / 0.022) ** 2 <= 1.0
    )
    right_brow = base.face & (
        ((u - 0.70) / 0.12) ** 2 + ((v - 0.395) / 0.022) ** 2 <= 1.0
    )
    nose = base.nose

    # Lips are a horizontal split of the lip band around the known mouth centre.
    # Keep the band tight so chin / cheek flesh is not treated as a movable lip.
    lip_band = base.lip_interior | base.lip_outline
    lip_band = lip_band & (
        ((u - 0.50) / 0.20) ** 2 + ((v - 0.78) / 0.065) ** 2 <= 1.0
    )
    upper_lip = lip_band & (v <= 0.785) & (v >= 0.735)
    lower_lip = lip_band & (v > 0.785) & (v <= 0.835)
    mouth_cavity = base.mouth_cavity & ~upper_lip & ~lower_lip

    # Static face base excludes every movable piece.
    movable = (
        left_eye
        | right_eye
        | left_brow
        | right_brow
        | upper_lip
        | lower_lip
        | mouth_cavity
        | nose
    )
    face_base = base.face & ~movable

    return FacePartMasks(
        face=face_base,
        nose=nose,
        left_brow=left_brow,
        right_brow=right_brow,
        left_eye=left_eye,
        right_eye=right_eye,
        upper_lip=upper_lip,
        lower_lip=lower_lip,
        mouth_cavity=mouth_cavity,
    )


def _anchor(
    mask: npt.NDArray[np.bool_],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    if not mask.any():
        return fallback
    rows, columns = np.nonzero(mask)
    return (float(columns.mean()) + 0.5, float(rows.mean()) + 0.5)


def build_face_part_atlas(
    image_bgr: npt.NDArray[np.uint8],
    *,
    face: FaceBox | None = None,
) -> FacePartAtlas:
    """Build an RGBA atlas: RGB = photo, A = part id / PART_ID_SCALE."""
    from chorusface.seed import _cv2, detect_face

    cv2 = _cv2()
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    height, width = image_bgr.shape[:2]
    box = face if face is not None else detect_face(image_bgr)
    masks = build_face_part_masks(height, width, box)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    part_id = np.zeros((height, width), dtype=np.float32)
    layers = (
        (masks.face, PART_FACE),
        (masks.nose, PART_NOSE),
        (masks.left_brow, PART_LEFT_BROW),
        (masks.right_brow, PART_RIGHT_BROW),
        (masks.left_eye, PART_LEFT_EYE),
        (masks.right_eye, PART_RIGHT_EYE),
        (masks.upper_lip, PART_UPPER_LIP),
        (masks.lower_lip, PART_LOWER_LIP),
        (masks.mouth_cavity, PART_MOUTH_CAVITY),
    )
    for mask, code in layers:
        part_id[mask] = float(code) / PART_ID_SCALE

    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = rgb
    rgba[..., 3] = part_id
    # Match .bds / world y-up orientation.
    rgba = np.ascontiguousarray(np.flipud(rgba), dtype=np.float32)
    flipped_masks = FacePartMasks(
        face=np.flipud(masks.face),
        nose=np.flipud(masks.nose),
        left_brow=np.flipud(masks.left_brow),
        right_brow=np.flipud(masks.right_brow),
        left_eye=np.flipud(masks.left_eye),
        right_eye=np.flipud(masks.right_eye),
        upper_lip=np.flipud(masks.upper_lip),
        lower_lip=np.flipud(masks.lower_lip),
        mouth_cavity=np.flipud(masks.mouth_cavity),
    )
    anchors = {
        "left_eye": _anchor(flipped_masks.left_eye, (width * 0.38, height * 0.58)),
        "right_eye": _anchor(flipped_masks.right_eye, (width * 0.62, height * 0.58)),
        "left_brow": _anchor(flipped_masks.left_brow, (width * 0.38, height * 0.66)),
        "right_brow": _anchor(flipped_masks.right_brow, (width * 0.62, height * 0.66)),
        "upper_lip": _anchor(flipped_masks.upper_lip, (width * 0.50, height * 0.34)),
        "lower_lip": _anchor(flipped_masks.lower_lip, (width * 0.50, height * 0.28)),
        "mouth_cavity": _anchor(flipped_masks.mouth_cavity, (width * 0.50, height * 0.31)),
        "nose": _anchor(flipped_masks.nose, (width * 0.50, height * 0.48)),
    }
    return FacePartAtlas(rgba=rgba, masks=flipped_masks, face=box, anchors=anchors)


def save_face_part_atlas(path: str | Path, atlas: FacePartAtlas) -> Path:
    """Write a float RGBA .npy atlas plus a preview PNG."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, atlas.rgba)
    preview = destination.with_suffix(".png")
    try:
        from PIL import Image

        # Preview colours each part so you can verify the split by eye.
        ids = np.rint(atlas.rgba[..., 3] * PART_ID_SCALE).astype(np.int32)
        preview_rgb = (atlas.rgba[..., :3] * 255.0).astype(np.uint8).copy()
        tint = {
            PART_LEFT_EYE: (40, 180, 255),
            PART_RIGHT_EYE: (40, 180, 255),
            PART_LEFT_BROW: (255, 160, 40),
            PART_RIGHT_BROW: (255, 160, 40),
            PART_UPPER_LIP: (255, 70, 120),
            PART_LOWER_LIP: (255, 40, 90),
            PART_MOUTH_CAVITY: (30, 30, 30),
            PART_NOSE: (180, 255, 120),
        }
        for code, colour in tint.items():
            mask = ids == code
            if not mask.any():
                continue
            blend = preview_rgb[mask].astype(np.float32)
            colour_arr = np.asarray(colour, dtype=np.float32)
            preview_rgb[mask] = (blend * 0.45 + colour_arr * 0.55).astype(np.uint8)
        # PNG is y-up like the atlas; flip for normal image viewers.
        Image.fromarray(np.flipud(preview_rgb)).save(preview)
    except Exception:
        pass
    meta_path = destination.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "version": "face-parts-1.0",
                "part_id_scale": PART_ID_SCALE,
                "parts": PART_NAMES,
                "counts": atlas.masks.counts(),
                "anchors": {key: list(value) for key, value in atlas.anchors.items()},
                "face_box": {
                    "x": atlas.face.x,
                    "y": atlas.face.y,
                    "width": atlas.face.width,
                    "height": atlas.face.height,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def load_face_part_atlas(path: str | Path) -> npt.NDArray[np.float32]:
    array = np.load(path)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"face part atlas must be HxWx4, got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def default_parts_path(world_path: str | Path) -> Path:
    return Path(world_path).with_name("face_parts.npy")


__all__ = [
    "PART_FACE",
    "PART_ID_SCALE",
    "PART_LEFT_BROW",
    "PART_LEFT_EYE",
    "PART_LOWER_LIP",
    "PART_MOUTH_CAVITY",
    "PART_NAMES",
    "PART_NONE",
    "PART_NOSE",
    "PART_RIGHT_BROW",
    "PART_RIGHT_EYE",
    "PART_UPPER_LIP",
    "FacePartAtlas",
    "FacePartMasks",
    "build_face_part_atlas",
    "build_face_part_masks",
    "default_parts_path",
    "load_face_part_atlas",
    "save_face_part_atlas",
]
