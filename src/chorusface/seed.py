"""Build a locked face seed for the avatar.

Takes a front-facing face image and writes a ``256 × 256 × 32`` ``.bds`` world:

* Channel 24 (``hard_surface``) receives Sobel outlines for eyes, nose, skull,
  and lips — the structural contours.
* Channel 31 (``human_lock`` / ``M_human``) is set to ``1.0`` over the static
  facial regions: skull, cheeks, forehead, eyes, and nose. The GPU command path
  refuses AI writes on those cells, so identity cannot drift.
* The lower jaw, mouth cavity, and lip interior stay at ``M_human = 0.0`` so
  velocity impulses from the chat driver can move them.

OpenCV is optional and only imported when conversion runs. Install it with the
``seed`` extra: ``pip install "chorusface[seed]"``.

Example::

    chorusface-seed --synthetic
    chorusface-seed --input portrait.jpg --output output/worlds/avatar/avatar_face.bds
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import numpy.typing as npt

from chorusface.paths import DEFAULT_AVATAR_FACE, ensure_output_tree
from chorusface.runtime.bds import (
    DTYPE,
    GRID_HEIGHT,
    GRID_WIDTH,
    HARD_SURFACE_CHANNEL,
    HUMAN_LOCK_CHANNEL,
    PRIORITY_CHANNEL,
    PRIORITY_LEVELS,
    VECTOR_DIMENSIONS,
    save_bds,
)
from chorusface.runtime.shaders import normalize_priority

EDGE_THRESHOLD: Final = 0.28
FACE_PADDING: Final = 0.08


class AvatarSeedError(RuntimeError):
    """Raised when a face image cannot be converted into a seed world."""


@dataclass(frozen=True, slots=True)
class FaceBox:
    """Axis-aligned face rectangle in image coordinates (y down)."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        return self.x + self.width

    @property
    def y1(self) -> int:
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class RegionMasks:
    """Boolean masks in image space before the world y-flip."""

    face: npt.NDArray[np.bool_]
    skull: npt.NDArray[np.bool_]
    eyes: npt.NDArray[np.bool_]
    nose: npt.NDArray[np.bool_]
    jaw: npt.NDArray[np.bool_]
    mouth_cavity: npt.NDArray[np.bool_]
    lip_interior: npt.NDArray[np.bool_]
    lip_outline: npt.NDArray[np.bool_]

    @property
    def locked(self) -> npt.NDArray[np.bool_]:
        """Static Master-Lock regions: eyes + nose + skull."""
        return self.skull | self.eyes | self.nose

    @property
    def unlocked_motion(self) -> npt.NDArray[np.bool_]:
        """Lip and mouth cells the chat driver may move; chin stays locked."""
        return self.mouth_cavity | self.lip_interior | self.lip_outline


@dataclass(frozen=True, slots=True)
class AvatarSeedResult:
    """Converted tensor plus telemetry for the CLI."""

    tensor: npt.NDArray[np.float32]
    source_name: str
    face: FaceBox
    edge_fraction: float
    locked_cells: int
    unlocked_motion_cells: int
    synthetic: bool
    # The resized BGR pixels the tensor was built from. The part atlas and the
    # rendered portrait must come from exactly these pixels, not a re-read.
    image: npt.NDArray[np.uint8]
    landmarks: Any = None
    qa_score: float = 0.0


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise AvatarSeedError(
            "OpenCV is required for avatar seeding. Install it with "
            '`python -m pip install "chorusface[seed]"`.'
        ) from exc
    return cv2


def _bgr(red: int, green: int, blue: int) -> tuple[int, int, int]:
    """Write a colour in reading order; OpenCV draws into a BGR buffer."""
    return (blue, green, red)


def _blend(base: npt.NDArray[np.uint8], overlay: npt.NDArray[np.uint8], alpha: float) -> None:
    """In-place alpha blend of ``overlay`` onto ``base``."""
    amount = float(np.clip(alpha, 0.0, 1.0))
    if amount <= 0.0:
        return
    mixed = (
        base.astype(np.float32) * (1.0 - amount) + overlay.astype(np.float32) * amount
    )
    np.copyto(base, np.clip(mixed, 0, 255).astype(np.uint8))


def _radial_light(
    height: int,
    width: int,
    center: tuple[float, float],
    radii: tuple[float, float],
) -> npt.NDArray[np.float32]:
    """Soft elliptical falloff in [0, 1], peaking at ``center``."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    rx = max(float(radii[0]), 1.0)
    ry = max(float(radii[1]), 1.0)
    distance = np.sqrt(((xx - center[0]) / rx) ** 2 + ((yy - center[1]) / ry) ** 2)
    return np.clip(1.0 - distance, 0.0, 1.0).astype(np.float32)


def synthesize_face_bgr(
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT,
) -> npt.NDArray[np.uint8]:
    """Draw a polished stylized portrait aligned to the muscle UV layout.

    Features sit on the same face-box fractions the masks,
    ``face_definition.json``, and tissue bake use (eyes near ``v=0.472``,
    mouth at ``v=0.78``), so the synthetic seed exercises the real displacement
    path instead of a mismatched cartoon. Lighting, hair, brows, and irises
    make the demo face read as a character rather than a flat ellipse collage.
    """
    cv2 = _cv2()
    box = _fallback_face_box(width, height)
    fx, fy, fw, fh = float(box.x), float(box.y), float(box.width), float(box.height)

    def px(u: float, v: float) -> tuple[int, int]:
        return (int(round(fx + u * fw)), int(round(fy + v * fh)))

    def ax(u: float, v: float) -> tuple[int, int]:
        return (max(1, int(round(u * fw))), max(1, int(round(v * fh))))

    # Cool studio backdrop with a soft vignette.
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        t = row / max(height - 1, 1)
        tone = int(28 + 22 * (1.0 - t))
        image[row, :] = (tone + 8, tone + 4, tone)
    vignette = _radial_light(
        height, width, (width * 0.5, height * 0.42), (width * 0.62, height * 0.72)
    )
    image = np.clip(
        image.astype(np.float32) * (0.55 + 0.55 * vignette[..., None]), 0, 255
    ).astype(np.uint8)

    # Neck and shoulders under the head.
    neck = px(0.50, 0.92)
    cv2.ellipse(
        image, neck, ax(0.16, 0.22), 0, 0, 360, _bgr(168, 132, 112), -1, cv2.LINE_AA
    )
    cv2.ellipse(
        image,
        (width // 2, int(height * 0.98)),
        (int(width * 0.42), int(height * 0.18)),
        0,
        180,
        360,
        _bgr(54, 58, 72),
        -1,
        cv2.LINE_AA,
    )

    face_center = px(0.50, 0.50)
    face_axes = ax(0.48, 0.56)
    skin = _bgr(214, 176, 148)
    skin_shadow = _bgr(176, 132, 108)
    skin_warm = _bgr(228, 168, 142)

    # Hair mass behind the face oval.
    hair = _bgr(42, 32, 28)
    cv2.ellipse(
        image,
        px(0.50, 0.28),
        ax(0.52, 0.42),
        0,
        0,
        360,
        hair,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        px(0.18, 0.48),
        ax(0.14, 0.34),
        12,
        0,
        360,
        hair,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        px(0.82, 0.48),
        ax(0.14, 0.34),
        -12,
        0,
        360,
        hair,
        -1,
        cv2.LINE_AA,
    )

    # Base face oval.
    cv2.ellipse(image, face_center, face_axes, 0, 0, 360, skin, -1, cv2.LINE_AA)

    # Soft key light on the forehead/cheek and a cooler fill on the far side.
    light = _radial_light(
        height,
        width,
        (fx + 0.38 * fw, fy + 0.34 * fh),
        (0.34 * fw, 0.40 * fh),
    )
    shade = _radial_light(
        height,
        width,
        (fx + 0.72 * fw, fy + 0.58 * fh),
        (0.30 * fw, 0.38 * fh),
    )
    face_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(face_mask, face_center, face_axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    lit = image.astype(np.float32)
    highlight = np.array(skin_warm, dtype=np.float32)
    shadow = np.array(skin_shadow, dtype=np.float32)
    amount = (face_mask.astype(np.float32) / 255.0)[..., None]
    lit += amount * light[..., None] * (highlight - lit) * 0.35
    lit += amount * shade[..., None] * (shadow - lit) * 0.28
    image = np.clip(lit, 0, 255).astype(np.uint8)

    # Forehead hairline softener.
    cv2.ellipse(
        image,
        px(0.50, 0.12),
        ax(0.42, 0.16),
        0,
        0,
        180,
        hair,
        -1,
        cv2.LINE_AA,
    )

    # Brows.
    brow = _bgr(56, 40, 34)
    for u, tilt in ((0.30, 8), (0.70, -8)):
        cv2.ellipse(
            image,
            px(u, 0.30),
            ax(0.11, 0.025),
            tilt,
            0,
            360,
            brow,
            -1,
            cv2.LINE_AA,
        )

    # Eyes: socket, sclera, iris, pupil, catchlight. Centres match mask UVs.
    iris = _bgr(72, 108, 148)
    for u in (0.30, 0.70):
        eye = px(u, 0.472)
        cv2.ellipse(
            image, eye, ax(0.12, 0.055), 0, 0, 360, _bgr(64, 48, 44), -1, cv2.LINE_AA
        )
        cv2.ellipse(
            image, eye, ax(0.105, 0.045), 0, 0, 360, _bgr(236, 232, 228), -1, cv2.LINE_AA
        )
        cv2.circle(
            image, eye, max(3, int(fw * 0.035)), iris, -1, cv2.LINE_AA
        )
        cv2.circle(
            image, eye, max(2, int(fw * 0.016)), _bgr(18, 16, 20), -1, cv2.LINE_AA
        )
        catch = (eye[0] + max(1, int(fw * 0.012)), eye[1] - max(1, int(fh * 0.010)))
        cv2.circle(
            image, catch, max(1, int(fw * 0.008)), _bgr(250, 250, 252), -1, cv2.LINE_AA
        )
        # Upper lid shade.
        cv2.ellipse(
            image,
            (eye[0], eye[1] - max(1, int(fh * 0.012))),
            ax(0.105, 0.018),
            0,
            200,
            340,
            _bgr(150, 112, 98),
            max(1, int(fw * 0.008)),
            cv2.LINE_AA,
        )

    # Nose bridge and tip.
    cv2.line(
        image,
        px(0.50, 0.42),
        px(0.50, 0.58),
        _bgr(168, 128, 110),
        max(1, int(fw * 0.012)),
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image, px(0.50, 0.60), ax(0.055, 0.035), 0, 0, 360, _bgr(188, 140, 118), -1, cv2.LINE_AA
    )
    for u in (0.455, 0.545):
        cv2.ellipse(
            image,
            px(u, 0.615),
            ax(0.018, 0.012),
            0,
            0,
            360,
            _bgr(140, 100, 88),
            -1,
            cv2.LINE_AA,
        )

    # Mouth at v=0.78 — matches unlocked lip masks and mouth_center.
    mouth = px(0.50, 0.78)
    cv2.ellipse(
        image, mouth, ax(0.20, 0.070), 0, 0, 360, _bgr(168, 78, 88), -1, cv2.LINE_AA
    )
    cv2.ellipse(
        image,
        (mouth[0], mouth[1] - max(1, int(fh * 0.012))),
        ax(0.18, 0.035),
        0,
        0,
        360,
        _bgr(186, 92, 102),
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (mouth[0], mouth[1] + max(1, int(fh * 0.008))),
        ax(0.17, 0.030),
        0,
        0,
        360,
        _bgr(148, 64, 74),
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image, mouth, ax(0.12, 0.018), 0, 0, 360, _bgr(54, 22, 28), -1, cv2.LINE_AA
    )
    # Soft smile corners.
    for u, tilt in ((0.34, 18), (0.66, -18)):
        cv2.ellipse(
            image,
            px(u, 0.775),
            ax(0.035, 0.016),
            tilt,
            0,
            360,
            _bgr(158, 72, 82),
            -1,
            cv2.LINE_AA,
        )

    # Jaw contour for Sobel structure.
    cv2.ellipse(
        image,
        px(0.50, 0.72),
        ax(0.40, 0.34),
        0,
        25,
        155,
        _bgr(150, 112, 96),
        max(1, int(fw * 0.012)),
        cv2.LINE_AA,
    )

    # Subtle cheek warmth.
    for u in (0.22, 0.78):
        blush = np.zeros_like(image)
        cv2.ellipse(
            blush, px(u, 0.58), ax(0.10, 0.06), 0, 0, 360, _bgr(220, 120, 120), -1, cv2.LINE_AA
        )
        _blend(image, blush, 0.18)

    return image


def detect_face(image_bgr: npt.NDArray[np.uint8]) -> FaceBox:
    """Locate the primary frontal face, or fall back to a centred portrait box."""
    cv2 = _cv2()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.is_file():
        return _fallback_face_box(image_bgr.shape[1], image_bgr.shape[0])

    detector = cv2.CascadeClassifier(str(cascade_path))
    hits = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(24, image_bgr.shape[1] // 10), max(24, image_bgr.shape[0] // 10)),
    )
    if len(hits) == 0:
        return _fallback_face_box(image_bgr.shape[1], image_bgr.shape[0])

    # Largest detection wins for a single-avatar seed.
    x, y, width, height = max(hits, key=lambda box: int(box[2]) * int(box[3]))
    pad_x = int(width * FACE_PADDING)
    pad_y = int(height * FACE_PADDING)
    x0 = max(0, int(x) - pad_x)
    y0 = max(0, int(y) - pad_y)
    x1 = min(image_bgr.shape[1], int(x) + int(width) + pad_x)
    y1 = min(image_bgr.shape[0], int(y) + int(height) + pad_y)
    return FaceBox(x0, y0, x1 - x0, y1 - y0)


def _fallback_face_box(width: int, height: int) -> FaceBox:
    box_w = int(width * 0.62)
    box_h = int(height * 0.78)
    return FaceBox((width - box_w) // 2, int(height * 0.08), box_w, box_h)


def build_region_masks(
    height: int,
    width: int,
    face: FaceBox,
) -> RegionMasks:
    """Partition the face box into locked static regions and unlocked motion regions.

    Fractions follow a standard frontal portrait layout (image y points down).
    They are deterministic so the same image always yields the same locks.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    fx0, fy0 = float(face.x), float(face.y)
    fw, fh = float(max(face.width, 1)), float(max(face.height, 1))
    u = (xx.astype(np.float64) - fx0) / fw
    v = (yy.astype(np.float64) - fy0) / fh
    inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)

    # Soft face ellipse keeps background vacuum clean.
    face_ellipse = inside & (
        ((u - 0.5) / 0.48) ** 2 + ((v - 0.50) / 0.56) ** 2 <= 1.0
    )

    skull = face_ellipse & (v <= 0.48)
    left_eye = face_ellipse & (
        ((u - 0.30) / 0.13) ** 2 + ((v - 0.472) / 0.065) ** 2 <= 1.0
    )
    right_eye = face_ellipse & (
        ((u - 0.70) / 0.13) ** 2 + ((v - 0.472) / 0.065) ** 2 <= 1.0
    )
    eyes = left_eye | right_eye
    nose = face_ellipse & (
        ((u - 0.50) / 0.10) ** 2 + ((v - 0.50) / 0.15) ** 2 <= 1.0
    )

    # Wider / taller mouth unlock so the lip zip is not Master-Locked shut.
    # Previous ellipses left the outer lip seam in static_face → zipped look.
    mouth_cavity = face_ellipse & (
        ((u - 0.50) / 0.22) ** 2 + ((v - 0.78) / 0.08) ** 2 <= 1.0
    )
    lip_interior = face_ellipse & (
        ((u - 0.50) / 0.28) ** 2 + ((v - 0.78) / 0.12) ** 2 <= 1.0
    )
    # A thin ring around the mouth reads as lip outline in the Sobel pass.
    lip_band = face_ellipse & (
        ((u - 0.50) / 0.32) ** 2 + ((v - 0.78) / 0.14) ** 2 <= 1.0
    )
    lip_outline = lip_band & ~mouth_cavity

    jaw = face_ellipse & (v >= 0.70) & (v <= 1.05) & (u >= 0.16) & (u <= 0.84)
    # Motion regions must never inherit the Master Lock.
    locked_core = skull | eyes | nose
    jaw = jaw & ~locked_core
    mouth_cavity = mouth_cavity & ~locked_core
    lip_interior = lip_interior & ~locked_core
    lip_outline = lip_outline & ~locked_core
    # Lock static portrait, but keep full lip band + oral cavity free to move.
    motion = mouth_cavity | lip_interior | lip_outline
    static_face = face_ellipse & ~motion

    return RegionMasks(
        face=face_ellipse,
        skull=static_face,
        eyes=eyes,
        nose=nose,
        jaw=jaw,
        mouth_cavity=mouth_cavity,
        lip_interior=lip_interior,
        lip_outline=lip_outline,
    )


def sobel_edges(
    gray: npt.NDArray[np.float32],
    *,
    threshold: float = EDGE_THRESHOLD,
) -> npt.NDArray[np.float32]:
    """Return a binary hard-surface mask from a normalised Sobel gradient."""
    cv2 = _cv2()
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(sobel_x, sobel_y)
    peak = float(gradient.max(initial=0.0))
    if peak <= 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    normalized = gradient / peak
    return (normalized > threshold).astype(np.float32)


def load_face_image(
    path: str | Path,
    width: int,
    height: int,
    *,
    normalize: bool = True,
) -> tuple[npt.NDArray[np.uint8], FaceBox]:
    """Load a portrait and place it on the seed grid.

    With ``normalize=True`` (default) the face is square-cropped before resize
    so aspect warp cannot slide features off the muscle UV layout.
    """
    cv2 = _cv2()
    source = Path(path)
    if not source.is_file():
        raise AvatarSeedError(f"Input image does not exist: {source}")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise AvatarSeedError(f"OpenCV could not decode image: {source}")
    if not normalize:
        return (
            cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA),
            _fallback_face_box(width, height),
        )
    from chorusface.landmarks import normalize_face_image

    return normalize_face_image(image, width=width, height=height)


def build_avatar_tensor(
    image_bgr: npt.NDArray[np.uint8],
    *,
    edge_threshold: float = EDGE_THRESHOLD,
    face: FaceBox | None = None,
) -> tuple[npt.NDArray[np.float32], FaceBox, RegionMasks, float]:
    """Map a BGR face image onto the 32-channel substrate."""
    cv2 = _cv2()
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise AvatarSeedError("Face image must have shape (height, width, 3)")
    height, width = image_bgr.shape[:2]
    box = face if face is not None else detect_face(image_bgr)
    masks = build_region_masks(height, width, box)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / np.float32(
        255.0
    )
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(
        255.0
    )
    edges = sobel_edges(gray, threshold=edge_threshold)

    # Contours: keep Sobel hits on the face, and force lip/eye/nose/skull guides.
    contour = np.zeros((height, width), dtype=np.float32)
    guide = masks.skull | masks.eyes | masks.nose | masks.lip_outline | masks.face
    contour[(edges >= 0.5) & guide] = 1.0
    # Guarantee the semantic guides survive a soft photo with weak gradients.
    for region in (masks.eyes, masks.nose, masks.lip_outline):
        eroded = _erode(region, iterations=1)
        ring = region & ~eroded
        contour[ring] = 1.0

    tensor = np.zeros((height, width, VECTOR_DIMENSIONS), dtype=DTYPE)
    # Keep photographic albedo untouched. Contours live only in the rule
    # channels so the face identity cannot be rewritten by guide colours.
    tensor[..., 3] = gray
    tensor[..., 8:11] = rgb
    tensor[..., 11] = np.clip(gray * 0.55 + masks.face.astype(np.float32) * 0.45, 0.0, 1.0)
    tensor[..., 14] = 0.0

    tensor[..., HARD_SURFACE_CHANNEL] = contour
    tensor[..., HUMAN_LOCK_CHANNEL] = masks.locked.astype(np.float32)
    tensor[masks.locked, PRIORITY_CHANNEL] = np.float32(
        normalize_priority(PRIORITY_LEVELS["user"])
    )
    # Soft flesh in the unlocked mouth/jaw so impulses have mass to push.
    soft = masks.unlocked_motion
    tensor[soft, 3] = np.maximum(tensor[soft, 3], np.float32(0.35))
    tensor[soft, 25] = np.float32(0.7)  # permeability
    # Explicitly clear locks on motion regions (belt and suspenders).
    tensor[soft, HUMAN_LOCK_CHANNEL] = 0.0
    # Mouth/jaw/lips must not freeze under Sobel or outline guides — a hard
    # lip ring was sealing the mouth shut ("zipped") while field/warp moved.
    tensor[
        masks.mouth_cavity | masks.jaw | masks.lip_interior | masks.lip_outline,
        HARD_SURFACE_CHANNEL,
    ] = 0.0
    # Freeze static identity: no permeability and full priority on locked cells.
    tensor[masks.locked, 25] = 0.0

    # World grid y points up; image y points down.
    tensor = np.ascontiguousarray(np.flipud(tensor), dtype=DTYPE)
    return tensor, box, masks, float(contour.mean())


def _erode(mask: npt.NDArray[np.bool_], iterations: int = 1) -> npt.NDArray[np.bool_]:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(iterations, 0)):
        padded = np.pad(result, 1, constant_values=False)
        result = (
            padded[:-2, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 1:-1]
            & padded[1:-1, 2:]
            & padded[2:, 1:-1]
        )
    return result


def build_avatar_seed(
    source: str | Path | None = None,
    *,
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT,
    edge_threshold: float = EDGE_THRESHOLD,
    synthetic: bool = False,
    normalize: bool = True,
) -> AvatarSeedResult:
    """Convert a face image (or a synthetic portrait) into an avatar world."""
    from chorusface.landmarks import eye_aperture_score, measure_landmarks

    if width <= 0 or height <= 0:
        raise ValueError("Output dimensions must be positive")
    if not 0.0 <= edge_threshold <= 1.0:
        raise ValueError("edge_threshold must be in [0, 1]")
    if synthetic or source is None:
        image = synthesize_face_bgr(width, height)
        source_name = "synthetic_face"
        is_synthetic = True
        # Keep the authored UV layout — Haar detection would slide features off
        # the muscle anchors the stylized portrait was drawn against.
        face_hint = _fallback_face_box(width, height)
    else:
        image, face_hint = load_face_image(
            source, width, height, normalize=normalize
        )
        source_name = Path(source).name
        is_synthetic = False

    tensor, face, masks, edge_fraction = build_avatar_tensor(
        image,
        edge_threshold=edge_threshold,
        face=face_hint,
    )
    landmarks = measure_landmarks(image, face=face, synthetic=is_synthetic)
    qa_score = eye_aperture_score(image, landmarks)
    # Masks were built in image space; flip counts match the saved tensor.
    locked = int(np.flipud(masks.locked).sum())
    unlocked = int(np.flipud(masks.unlocked_motion).sum())
    return AvatarSeedResult(
        tensor=tensor,
        source_name=source_name,
        face=face,
        edge_fraction=edge_fraction,
        locked_cells=locked,
        unlocked_motion_cells=unlocked,
        synthetic=is_synthetic,
        image=image,
        landmarks=landmarks,
        qa_score=qa_score,
    )


def build_metadata(result: AvatarSeedResult) -> dict[str, Any]:
    landmarks = result.landmarks
    if landmarks is not None:
        mouth = {
            "x": float(landmarks.mouth[0]),
            "y": float(landmarks.mouth[1]),
        }
        landmark_meta = landmarks.as_metadata()
    else:
        mouth = {
            "x": float(result.face.x + result.face.width * 0.50),
            "y": float(result.face.y + result.face.height * 0.78),
        }
        landmark_meta = {}
    return {
        "world_name": f"Avatar face: {result.source_name}",
        "description": (
            "Front-facing face seed with Master-Locked skull/eyes/nose and an "
            "unlocked jaw/mouth for velocity-driven speech"
        ),
        "source": {
            "kind": "synthetic_face" if result.synthetic else "face_image",
            "filename": result.source_name,
        },
        "avatar_seed": {
            "version": "avatar-seed-1.1",
            "grid": [int(result.tensor.shape[1]), int(result.tensor.shape[0])],
            "face_box": {
                "x": result.face.x,
                "y": result.face.y,
                "width": result.face.width,
                "height": result.face.height,
            },
            "edge_fraction": round(result.edge_fraction, 6),
            "locked_cells": result.locked_cells,
            "unlocked_motion_cells": result.unlocked_motion_cells,
            "locked_regions": ["skull", "eyes", "nose", "cheeks", "forehead"],
            "unlocked_regions": ["mouth_cavity", "lip_interior"],
            "identity_stable": True,
            "mouth_center_image": mouth,
            "landmarks": landmark_meta,
            "qa_score": round(float(result.qa_score), 4),
            "hard_surface_channel": HARD_SURFACE_CHANNEL,
            "human_lock_channel": HUMAN_LOCK_CHANNEL,
        },
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a front-facing face image into a locked avatar .bds seed.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Front-facing face image (PNG/JPG). Omit with --synthetic.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_AVATAR_FACE,
        help=f"Output .bds path (default {DEFAULT_AVATAR_FACE})",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Build a deterministic drawn face instead of reading --input",
    )
    parser.add_argument("--width", type=int, default=GRID_WIDTH)
    parser.add_argument("--height", type=int, default=GRID_HEIGHT)
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=EDGE_THRESHOLD,
        help="Sobel peak fraction that becomes a hard-surface contour",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Stretch the whole frame to the grid (legacy; warps UV layout)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write a seed QA overlay next to the world (eyes/mouth marked)",
    )
    parser.add_argument(
        "--require-qa",
        type=float,
        default=0.0,
        help="Fail if the eye-registration score is below this (0 disables)",
    )
    return parser.parse_args(argv)


def write_portrait(path: str | Path, image_bgr: npt.NDArray[np.uint8]) -> Path:
    """Save the exact pixels the seed used as the renderer's immutable photo."""
    cv2 = _cv2()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image_bgr):
        raise AvatarSeedError(f"Could not write portrait: {destination}")
    return destination


def write_seed_bundle(
    result: AvatarSeedResult,
    output: str | Path,
    *,
    definition: str | Path | None = None,
    preview: bool = False,
) -> dict[str, Path]:
    """Write the colocated world, atlas, tissue maps, and portrait for one seed.

    The renderer finds all of them by name, so they must land in one directory:
    ``avatar_face.bds``, ``face_parts.npy``, ``face_tissue.npy`` (each with a
    sidecar ``.json``), and ``source_face.png``. Oral stamps are not part of
    this contract; speech warps the portrait via muscles + jaw only.
    """
    from chorusface.biomechanics.muscles import load_face_definition
    from chorusface.landmarks import render_seed_qa
    from chorusface.parts import build_face_part_atlas, save_face_part_atlas
    from chorusface.skinning import (
        build_tissue_maps,
        default_tissue_path,
        save_tissue_maps,
    )

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_bds(destination, result.tensor, metadata=build_metadata(result))
    written = {"world": destination}

    portrait = write_portrait(destination.with_name("source_face.png"), result.image)
    written["portrait"] = portrait

    atlas = build_face_part_atlas(result.image, face=result.face)
    written["parts"] = save_face_part_atlas(
        destination.with_name("face_parts.npy"), atlas
    )

    height, width = result.image.shape[:2]
    tissue = build_tissue_maps(
        height,
        width,
        result.face,
        load_face_definition(definition),
        landmarks=result.landmarks,
    )
    written["tissue"] = save_tissue_maps(
        default_tissue_path(destination), tissue, face=result.face
    )

    if preview and result.landmarks is not None:
        qa = write_portrait(
            destination.with_name("seed_qa.png"),
            render_seed_qa(result.image, result.landmarks),
        )
        written["preview"] = qa
    return written


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_arguments(argv)
    if options.input is None and not options.synthetic:
        print("error: provide --input FACE.png or pass --synthetic", flush=True)
        return 2

    started = time.perf_counter()
    try:
        result = build_avatar_seed(
            options.input,
            width=options.width,
            height=options.height,
            edge_threshold=options.edge_threshold,
            synthetic=options.synthetic,
            normalize=not options.no_normalize,
        )
    except (AvatarSeedError, ValueError, OSError) as exc:
        print(f"error: {exc}", flush=True)
        return 1

    if options.require_qa > 0.0 and result.qa_score < options.require_qa:
        print(
            f"error: eye registration score {result.qa_score:.2f} is below "
            f"--require-qa {options.require_qa:.2f}",
            flush=True,
        )
        return 1

    ensure_output_tree()
    try:
        written = write_seed_bundle(
            result, options.output, preview=options.preview or options.require_qa > 0.0
        )
    except (AvatarSeedError, ValueError, OSError) as exc:
        print(f"error: {exc}", flush=True)
        return 1

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    landmarks = result.landmarks

    print(f"Wrote {written['world']}")
    print(f"Wrote {written['parts']} (anatomical part atlas)")
    print(f"Wrote {written['tissue']} (mobility / lip parting / eye aperture)")
    print(f"Wrote {written['portrait']} (immutable render source)")
    if "preview" in written:
        print(f"Wrote {written['preview']} (seed QA overlay)")
    print(
        f"  source={result.source_name}  "
        f"locked={result.locked_cells}  "
        f"unlocked_motion={result.unlocked_motion_cells}  "
        f"edges={result.edge_fraction:.2%}  "
        f"{elapsed_ms:.0f} ms"
    )
    if landmarks is not None:
        left_uv, right_uv = landmarks.eye_uv()
        mouth_uv = landmarks.mouth_uv()
        print(
            f"  landmarks={landmarks.method}  qa={result.qa_score:.2f}  "
            f"eyes@({left_uv[0]:.3f},{left_uv[1]:.3f})/"
            f"({right_uv[0]:.3f},{right_uv[1]:.3f})  "
            f"mouth@({mouth_uv[0]:.3f},{mouth_uv[1]:.3f})"
        )
    print(f"Play with: chorusface --world {Path(written['world']).as_posix()}")
    return 0


__all__ = [
    "AvatarSeedError",
    "AvatarSeedResult",
    "EDGE_THRESHOLD",
    "FaceBox",
    "HARD_SURFACE_CHANNEL",
    "RegionMasks",
    "build_avatar_seed",
    "build_avatar_tensor",
    "build_metadata",
    "build_region_masks",
    "detect_face",
    "load_face_image",
    "main",
    "sobel_edges",
    "synthesize_face_bgr",
    "write_portrait",
    "write_seed_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
