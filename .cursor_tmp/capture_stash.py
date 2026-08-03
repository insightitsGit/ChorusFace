"""Avatar Capture Converter: short HQ video or stills → Path 1 seed + plates.

Unlike NWR ``video2game`` (terrain / optical flow), this digests a frontal face
take into a locked identity rest plate plus real smile/open expression plates
and lip/jaw travel priors learned from frames — never invented teeth.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import numpy.typing as npt

from aiface.paths import DEFAULT_AVATAR_FACE, ensure_output_tree
from aiface.runtime.bds import GRID_HEIGHT, GRID_WIDTH
from aiface.seed import AvatarSeedError, FaceBox, _cv2, write_portrait

SMILE_PLATE_NAME: Final = "smile.png"
OPEN_PLATE_NAME: Final = "open.png"
EYES_CLOSED_PLATE_NAME: Final = "eyes_closed.png"
SURPRISE_PLATE_NAME: Final = "surprise.png"
CAPTURE_META_NAME: Final = "capture_meta.json"
QA_CONTACT_NAME: Final = "capture_qa.png"

DEFAULT_SAMPLE_FPS: Final = 6.0
#: AMIN step 11 — display plates are re-cut from the source at this square
#: resolution. Analysis/registration stays at the grid size; the field never
#: grows. The face crop is deterministic per frame, so both stay registered.
DISPLAY_SIZE: Final = 1024
MIN_SHARPNESS: Final = 18.0
MIN_SHARPNESS_SOFT: Final = 10.0
MIN_LANDMARK_QUALITY: Final = 0.35
#: Eye-span / face-width below this ⇒ profile / strong yaw.
MIN_EYE_SPAN: Final = 0.18
#: Mouth must sit near face centre (fraction of face width).
MAX_MOUTH_OFFSET: Final = 0.18
#: Face must cover at least this fraction of the normalized canvas.
MIN_FACE_FRAC: Final = 0.42
#: Pupil aperture score below this ⇒ sunglasses / occluded eyes.
MIN_EYE_APERTURE: Final = 0.12
#: Open beat must exceed rest by at least this mouth_open delta.
MIN_OPEN_DELTA: Final = 0.035
#: Smile width must exceed rest by at least this amount (or fail closed).
MIN_SMILE_WIDTH_DELTA: Final = 0.015
#: Rest / identity must stay at or below this mouth_open (else lips look open forever).
MAX_REST_MOUTH_OPEN: Final = 0.18
#: Rest teeth visibility above this fails closed (visible enamel on identity).
MAX_REST_TEETH: Final = 0.12
#: Use scripted phase windows when the take is at least this long.
PHASE_SPLIT_SECONDS: Final = 8.0

_MP_UPPER_LIP: Final = 13
_MP_LOWER_LIP: Final = 14
_MP_MOUTH_LEFT: Final = 61
_MP_MOUTH_RIGHT: Final = 291
_MP_LEFT_EYE_UPPER: Final = 159
_MP_LEFT_EYE_LOWER: Final = 145
_MP_RIGHT_EYE_UPPER: Final = 386
_MP_RIGHT_EYE_LOWER: Final = 374
_MP_LEFT_BROW: Final = 70
_MP_RIGHT_BROW: Final = 300


class CaptureError(RuntimeError):
    """Raised when a capture take cannot become an avatar bundle."""


@dataclass(frozen=True, slots=True)
class ExpressionMetrics:
    """Scalar expression signals in face-box UV units."""

    mouth_open: float
    smile_width: float
    sharpness: float
    landmark_quality: float
    method: str
    eye_span: float = 0.0
    eye_aperture: float = 0.0
    mouth_offset: float = 0.0
    teeth: float = 0.0
    # Upper-face signals for the expression catalog (surprise / brows).
    brow_raise: float = 0.0
    lid_open: float = 0.0


@dataclass(frozen=True, slots=True)
class FrameSample:
    """One normalized frame with landmarks and expression scores."""

    index: int
    time_seconds: float
    image_bgr: npt.NDArray[np.uint8]
    face: FaceBox
    landmarks_meta: dict[str, Any]
    metrics: ExpressionMetrics


@dataclass(slots=True)
class CaptureSelection:
    rest: FrameSample
    smile: FrameSample
    open: FrameSample
    surprise: FrameSample | None = None
    talk_frames: list[FrameSample] = field(default_factory=list)
    phase_mode: str = "global"
    phase_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TravelPriors:
    """Scales baked from the talk segment into seed metadata."""

    jaw_travel_scale: float = 1.0
    lip_width_scale: float = 1.0
    lip_open_scale: float = 1.0
    peak_mouth_open: float = 0.0
    peak_smile_width: float = 0.0
    frame_count: int = 0


@dataclass(slots=True)
class RejectReport:
    """Per-reason frame drop counts for the CLI digest."""

    sampled: int = 0
    kept: int = 0
    reasons: Counter[str] = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.reasons[reason] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "sampled": self.sampled,
            "kept": self.kept,
            "rejected": int(sum(self.reasons.values())),
            "reasons": dict(self.reasons),
        }

    def format_lines(self) -> list[str]:
        lines = [
            f"Capture digest: sampled={self.sampled} kept={self.kept} "
            f"rejected={sum(self.reasons.values())}"
        ]
        if self.reasons:
            detail = ", ".join(f"{name}={count}" for name, count in sorted(self.reasons.items()))
            lines.append(f"  rejects: {detail}")
        return lines


@dataclass(frozen=True, slots=True)
class CaptureResult:
    world: Path
    portrait: Path
    smile_plate: Path
    open_plate: Path
    meta: Path
    qa: Path | None
    priors: TravelPriors
    written: dict[str, Path]
    reject_report: RejectReport | None = None


def default_smile_plate_path(world: str | Path) -> Path:
    return Path(world).with_name(SMILE_PLATE_NAME)


def default_open_plate_path(world: str | Path) -> Path:
    return Path(world).with_name(OPEN_PLATE_NAME)


def default_surprise_plate_path(world: str | Path) -> Path:
    return Path(world).with_name(SURPRISE_PLATE_NAME)


def default_eyes_closed_plate_path(world: str | Path) -> Path:
    """Photographed nearly-closed eyes LOOK plate (blink ownership)."""
    return Path(world).with_name(EYES_CLOSED_PLATE_NAME)


def default_capture_meta_path(world: str | Path) -> Path:
    return Path(world).with_name(CAPTURE_META_NAME)


def _brow_lid_from_landmarks(
    landmarks_meta: dict[str, Any], face: FaceBox
) -> tuple[float, float]:
    """Fallback brow_raise / lid_open from stored eye and brow points."""
    fh = max(float(face.height), 1.0)
    left_eye = landmarks_meta.get("left_eye_image") or {}
    right_eye = landmarks_meta.get("right_eye_image") or {}
    left_brow = landmarks_meta.get("left_brow_image") or left_eye
    right_brow = landmarks_meta.get("right_brow_image") or right_eye
    le_y = float(left_eye.get("y", face.y + face.height * 0.42))
    re_y = float(right_eye.get("y", face.y + face.height * 0.42))
    lb_y = float(left_brow.get("y", le_y - fh * 0.08))
    rb_y = float(right_brow.get("y", re_y - fh * 0.08))
    brow_raise = float(
        np.clip((((le_y - lb_y) + (re_y - rb_y)) * 0.5) / fh, 0.0, 0.22)
    )
    # Without eyelid landmarks, lid openness is unknown — mid default.
    lid_open = 0.045
    return brow_raise, lid_open


def _laplacian_sharpness(image_bgr: npt.NDArray[np.uint8]) -> float:
    cv2 = _cv2()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mouth_geometry_from_landmarks(
    landmarks_meta: dict[str, Any], face: FaceBox
) -> tuple[float, float, float]:
    """Return (mouth_open, smile_width, mouth_offset) from stored landmark points."""
    mouth = landmarks_meta.get("mouth_center_image") or {}
    left = landmarks_meta.get("left_eye_image") or {}
    right = landmarks_meta.get("right_eye_image") or {}
    mx = float(mouth.get("x", face.x + face.width * 0.5))
    my = float(mouth.get("y", face.y + face.height * 0.78))
    face_cx = float(face.x) + 0.5 * float(face.width)
    offset = abs(mx - face_cx) / max(float(face.width), 1.0)
    # Without lip corners, approximate open/width from eye span + mouth band later.
    eye_span = abs(float(right.get("x", 0.0)) - float(left.get("x", 0.0))) / max(
        float(face.width), 1.0
    )
    # Canonical mouth openness is unknown here; callers fill from band/MediaPipe.
    return 0.0, float(np.clip(0.28 + eye_span * 0.2, 0.2, 0.55)), float(offset)


def _expression_from_mediapipe(
    image_bgr: npt.NDArray[np.uint8], face: FaceBox
) -> ExpressionMetrics | None:
    try:
        import mediapipe as mp

        face_mesh = mp.solutions.face_mesh
    except (ImportError, AttributeError):
        return None

    cv2 = _cv2()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]
    try:
        with face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as mesh:
            result = mesh.process(rgb)
    except Exception:
        return None
    if not result.multi_face_landmarks:
        return None
    points = result.multi_face_landmarks[0].landmark

    def px(index: int) -> tuple[float, float]:
        point = points[index]
        return (float(point.x) * width, float(point.y) * height)

    upper, lower = px(_MP_UPPER_LIP), px(_MP_LOWER_LIP)
    left, right = px(_MP_MOUTH_LEFT), px(_MP_MOUTH_RIGHT)
    if left[0] > right[0]:
        left, right = right, left
    fh = max(float(face.height), 1.0)
    fw = max(float(face.width), 1.0)
    mouth_open = abs(lower[1] - upper[1]) / fh
    smile_width = abs(right[0] - left[0]) / fw
    eye_span = abs(px(473)[0] - px(468)[0]) / fw
    mouth_mid_x = 0.5 * (left[0] + right[0])
    face_cx = float(face.x) + 0.5 * fw
    mouth_offset = abs(mouth_mid_x - face_cx) / fw
    left_lid = abs(px(_MP_LEFT_EYE_LOWER)[1] - px(_MP_LEFT_EYE_UPPER)[1]) / fh
    right_lid = abs(px(_MP_RIGHT_EYE_LOWER)[1] - px(_MP_RIGHT_EYE_UPPER)[1]) / fh
    lid_open = float(np.clip(0.5 * (left_lid + right_lid), 0.0, 0.18))
    left_brow = px(_MP_LEFT_BROW)
    right_brow = px(_MP_RIGHT_BROW)
    left_eye = px(468)
    right_eye = px(473)
    brow_raise = float(
        np.clip(
            0.5
            * (
                (left_eye[1] - left_brow[1]) / fh
                + (right_eye[1] - right_brow[1]) / fh
            ),
            0.0,
            0.22,
        )
    )
    quality = float(np.clip(eye_span / 0.35, 0.0, 1.0))
    if quality < MIN_LANDMARK_QUALITY:
        return None
    return ExpressionMetrics(
        mouth_open=float(mouth_open),
        smile_width=float(smile_width),
        sharpness=_laplacian_sharpness(image_bgr),
        landmark_quality=quality,
        method="mediapipe",
        eye_span=float(eye_span),
        mouth_offset=float(mouth_offset),
        brow_raise=brow_raise,
        lid_open=lid_open,
    )


def _expression_from_opencv_lips(
    image_bgr: npt.NDArray[np.uint8],
    face: FaceBox,
    landmarks_quality: float,
    method: str,
) -> ExpressionMetrics:
    """OpenCV mouth-band geometry when MediaPipe lips are unavailable."""
    cv2 = _cv2()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mx = int(face.x + face.width * 0.5)
    my = int(face.y + face.height * 0.78)
    half_w = max(4, int(face.width * 0.16))
    half_h = max(3, int(face.height * 0.07))
    y0, y1 = max(0, my - half_h), min(gray.shape[0], my + half_h)
    x0, x1 = max(0, mx - half_w), min(gray.shape[1], mx + half_w)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        open_score, width_score = 0.02, 0.32
    else:
        # Vertical darkness gradient + overall darkness → openness proxy.
        top = patch[: max(1, patch.shape[0] // 2)]
        bot = patch[max(1, patch.shape[0] // 2) :]
        cavity = float(np.clip((95.0 - float(patch.mean())) / 95.0, 0.0, 0.40))
        split = float(np.clip((float(top.mean()) - float(bot.mean())) / 50.0, 0.0, 0.15))
        open_score = float(np.clip(cavity + split, 0.0, 0.40))
        # Horizontal edge energy as a crude smile-width proxy (keep headroom).
        sobelx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        width_score = float(
            np.clip(0.22 + float(np.mean(np.abs(sobelx))) / 120.0, 0.18, 0.62)
        )
    face_cx = float(face.x) + 0.5 * float(face.width)
    mouth_offset = abs(float(mx) - face_cx) / max(float(face.width), 1.0)
    return ExpressionMetrics(
        mouth_open=open_score,
        smile_width=width_score,
        sharpness=_laplacian_sharpness(image_bgr),
        landmark_quality=float(landmarks_quality),
        method=f"{method}+opencv-lips",
        eye_span=0.0,
        mouth_offset=float(mouth_offset),
        brow_raise=0.0,
        lid_open=0.0,
    )


def build_mouth_interior_matte(
    height: int,
    width: int,
    face: FaceBox,
    landmarks_meta: dict[str, Any],
    *,
    openness: float = 0.05,
) -> npt.NDArray[np.float32]:
    """Soft alpha matte over the oral interior (not full cheeks)."""
    mouth = landmarks_meta.get("mouth_center_image") or {}
    mx = float(mouth.get("x", face.x + face.width * 0.5))
    my = float(mouth.get("y", face.y + face.height * 0.78))
    half_w = max(float(face.width) * 0.13, 5.0)
    half_h = max(float(face.height) * (0.035 + 0.10 * float(np.clip(openness, 0.0, 0.4))), 2.5)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u = (xx - mx) / half_w
    v = (yy - my) / half_h
    radial = np.sqrt(u * u + v * v)
    alpha = np.clip(1.0 - radial, 0.0, 1.0) ** 1.35
    alpha = np.where(radial <= 1.0, alpha, 0.0).astype(np.float32)
    return alpha


def build_expression_region_matte(
    height: int,
    width: int,
    face: FaceBox,
    landmarks_meta: dict[str, Any],
) -> npt.NDArray[np.float32]:
    """Wider lower-face matte so open/smile capture looks actually show."""
    mouth = landmarks_meta.get("mouth_center_image") or {}
    mx = float(mouth.get("x", face.x + face.width * 0.5))
    my = float(mouth.get("y", face.y + face.height * 0.78))
    # Must cover lip corners + lower face; prior 0.30×0.18 left plates invisible.
    half_w = max(float(face.width) * 0.38, 12.0)
    half_h = max(float(face.height) * 0.24, 10.0)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u = (xx - mx) / half_w
    v = (yy - my) / half_h
    radial = np.sqrt(u * u + (v * 1.10) * (v * 1.10))
    alpha = np.clip(1.0 - radial, 0.0, 1.0) ** 1.05
    return np.where(radial <= 1.0, alpha, 0.0).astype(np.float32)


def build_upper_face_matte(
    height: int,
    width: int,
    face: FaceBox,
    landmarks_meta: dict[str, Any],
) -> npt.NDArray[np.float32]:
    """Soft matte over brows + eyes for surprise / upper-face plates."""
    left_eye = landmarks_meta.get("left_eye_image") or {}
    right_eye = landmarks_meta.get("right_eye_image") or {}
    left_brow = landmarks_meta.get("left_brow_image") or left_eye
    right_brow = landmarks_meta.get("right_brow_image") or right_eye
    mx = 0.5 * (
        float(left_eye.get("x", face.x + face.width * 0.35))
        + float(right_eye.get("x", face.x + face.width * 0.65))
    )
    my = 0.5 * (
        float(left_brow.get("y", face.y + face.height * 0.32))
        + float(right_eye.get("y", face.y + face.height * 0.42))
    )
    half_w = max(float(face.width) * 0.38, 12.0)
    half_h = max(float(face.height) * 0.16, 8.0)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u = (xx - mx) / half_w
    v = (yy - my) / half_h
    radial = np.sqrt(u * u + (v * 1.05) * (v * 1.05))
    alpha = np.clip(1.0 - radial, 0.0, 1.0) ** 1.1
    return np.where(radial <= 1.0, alpha, 0.0).astype(np.float32)


def build_eye_lid_matte(
    height: int,
    width: int,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    *,
    half_width: float,
    half_height: float,
) -> npt.NDArray[np.float32]:
    """Soft dual-ellipse matte over both eye sockets for ``eyes_closed.png``.

    Image space is y-down (same as ``source_face.png`` / expression plates).
    Do not use ``face_tissue.npy`` channel A here — on some takes that channel
    parks on cheek/nose rather than the iris.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    hw = max(float(half_width), 4.0)
    hh = max(float(half_height), 3.0)
    alpha = np.zeros((height, width), dtype=np.float32)
    for cx, cy in (left_eye, right_eye):
        u = (xx - float(cx)) / hw
        v = (yy - float(cy)) / hh
        radial = np.sqrt(u * u + v * v)
        disk = np.clip(1.0 - radial, 0.0, 1.0) ** 1.15
        alpha = np.maximum(alpha, np.where(radial <= 1.0, disk, 0.0))
    return alpha.astype(np.float32)


EYE_ANCHORS_NAME: Final = "eye_anchors.json"


def default_eye_anchors_path(world: str | Path) -> Path:
    """Colocated eye socket anchors written by the eyes-closed bake."""
    root = Path(world)
    if root.suffix.lower() == ".bds":
        root = root.parent
    return root / EYE_ANCHORS_NAME


def write_eye_anchors(
    path: str | Path,
    *,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    half_width: float,
    half_height: float,
    image_size: tuple[int, int],
    source_frame: int | None = None,
    method: str = "face_landmarker",
) -> Path:
    """Persist image-space eye sockets for runtime L09 (y-down pixels)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(image_size[0]), int(image_size[1])
    payload = {
        "schema": "aiface.eye_anchors.v1",
        "method": method,
        "image_width": width,
        "image_height": height,
        "source_frame": source_frame,
        "left_eye_image": {"x": float(left_eye[0]), "y": float(left_eye[1])},
        "right_eye_image": {"x": float(right_eye[0]), "y": float(right_eye[1])},
        "half_width": float(half_width),
        "half_height": float(half_height),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def match_plate_to_reference(
    plate_bgr: npt.NDArray[np.uint8],
    reference_bgr: npt.NDArray[np.uint8],
    alpha: npt.NDArray[np.float32],
) -> npt.NDArray[np.uint8]:
    """Affine color-transfer a plate onto the rest frame's lighting.

    Plates come from different video frames than the identity photo, so their
    exposure differs slightly; at partial blend the whole matte then reads as
    a washed veil over the skin (perceived as mouth blur). Matching per-channel
    mean/std over the matte's feather ring — where both frames show the same
    skin — makes mid-blends invisible everywhere except the actual mouth.
    """
    cv2 = _cv2()
    if reference_bgr.shape[:2] != plate_bgr.shape[:2]:
        reference_bgr = cv2.resize(
            reference_bgr,
            (plate_bgr.shape[1], plate_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    ring = (alpha > 0.03) & (alpha < 0.55)
    if int(ring.sum()) < 256:
        ring = alpha > 0.03
    if int(ring.sum()) < 256:
        return plate_bgr
    plate = plate_bgr.astype(np.float32)
    matched = plate.copy()
    for channel in range(3):
        p = plate[..., channel][ring]
        r = reference_bgr[..., channel][ring].astype(np.float32)
        p_std = float(p.std())
        if p_std < 1e-3:
            continue
        gain = float(np.clip(r.std() / p_std, 0.6, 1.6))
        matched[..., channel] = (plate[..., channel] - float(p.mean())) * gain + float(
            r.mean()
        )
    return np.clip(matched, 0.0, 255.0).astype(np.uint8)


def write_expression_plate(
    path: str | Path,
    image_bgr: npt.NDArray[np.uint8],
    alpha: npt.NDArray[np.float32],
) -> Path:
    """Write BGRA PNG: RGB from the plate, A = mouth-interior matte."""
    cv2 = _cv2()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise CaptureError("Expression plate requires a BGR image")
    if alpha.shape[:2] != image_bgr.shape[:2]:
        raise CaptureError("Mouth matte size must match plate image")
    bgra = np.dstack(
        [
            image_bgr,
            np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8),
        ]
    )
    if not cv2.imwrite(str(destination), bgra):
        raise CaptureError(f"Could not write expression plate: {destination}")
    return destination


def analyze_frame(
    image_bgr: npt.NDArray[np.uint8],
    *,
    index: int = 0,
    time_seconds: float = 0.0,
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT,
    min_sharpness: float = MIN_SHARPNESS,
    allow_closed_eyes: bool = False,
    report: RejectReport | None = None,
) -> FrameSample | None:
    """Normalize one frame and score expression; return ``None`` if rejected.

    ``allow_closed_eyes`` keeps blink frames for ``eyes_closed.png`` (normal
    capture rejects low eye aperture).
    """
    from aiface.landmarks import eye_aperture_score, measure_landmarks, normalize_face_image
    from aiface.seed import detect_face

    if report is not None:
        report.sampled += 1

    def drop(reason: str) -> None:
        if report is not None:
            report.reject(reason)

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        drop("format")
        return None
    try:
        faces = detect_face(image_bgr)
    except AvatarSeedError:
        drop("crop")
        return None

    cv2 = _cv2()
    cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if cascade.is_file():
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        hits = cv2.CascadeClassifier(str(cascade)).detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        if len(hits) > 1:
            drop("multi_face")
            return None

    try:
        normalized, face_box = normalize_face_image(
            image_bgr, width=width, height=height, face=faces
        )
    except (AvatarSeedError, ValueError):
        drop("crop")
        return None

    face_frac = (face_box.width * face_box.height) / float(max(width * height, 1))
    if face_frac < MIN_FACE_FRAC:
        drop("crop")
        return None

    landmarks = measure_landmarks(normalized, face=face_box)
    metrics = _expression_from_mediapipe(normalized, face_box)
    if metrics is None:
        metrics = _expression_from_opencv_lips(
            normalized, face_box, landmarks.quality, landmarks.method
        )

    eye_span = metrics.eye_span
    if eye_span <= 1e-6:
        left = landmarks.left_eye
        right = landmarks.right_eye
        eye_span = abs(right[0] - left[0]) / max(float(face_box.width), 1.0)
    aperture = eye_aperture_score(normalized, landmarks)
    mouth_offset = metrics.mouth_offset
    if mouth_offset <= 1e-6:
        _, _, mouth_offset = _mouth_geometry_from_landmarks(
            landmarks.as_metadata(), face_box
        )

    from aiface.plates import teeth_visibility_score

    teeth = teeth_visibility_score(
        normalized,
        landmarks.mouth,
        float(face_box.width),
        float(face_box.height),
    )
    brow_raise = float(metrics.brow_raise)
    lid_open = float(metrics.lid_open)
    if brow_raise <= 1e-6 or lid_open <= 1e-6:
        fallback_brow, fallback_lid = _brow_lid_from_landmarks(
            landmarks.as_metadata(), face_box
        )
        if brow_raise <= 1e-6:
            brow_raise = fallback_brow
        if lid_open <= 1e-6:
            lid_open = fallback_lid
    metrics = ExpressionMetrics(
        mouth_open=metrics.mouth_open,
        smile_width=metrics.smile_width,
        sharpness=metrics.sharpness,
        landmark_quality=max(metrics.landmark_quality, landmarks.quality),
        method=metrics.method,
        eye_span=float(eye_span),
        eye_aperture=float(aperture),
        mouth_offset=float(mouth_offset),
        teeth=float(teeth),
        brow_raise=float(brow_raise),
        lid_open=float(lid_open),
    )

    if metrics.sharpness < min_sharpness:
        drop("blur")
        return None
    if metrics.landmark_quality < MIN_LANDMARK_QUALITY and landmarks.method.startswith(
        "canonical"
    ):
        drop("landmarks")
        return None
    if eye_span < MIN_EYE_SPAN or mouth_offset > MAX_MOUTH_OFFSET:
        drop("yaw")
        return None
    if (
        aperture < MIN_EYE_APERTURE
        and not allow_closed_eyes
        and not landmarks.method.startswith("canonical")
    ):
        drop("eyes")
        return None

    if report is not None:
        report.kept += 1
    return FrameSample(
        index=index,
        time_seconds=float(time_seconds),
        image_bgr=normalized,
        face=face_box,
        landmarks_meta=landmarks.as_metadata(),
        metrics=metrics,
    )


def iter_video_frames(
    video: str | Path,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT,
    min_sharpness: float = MIN_SHARPNESS,
    report: RejectReport | None = None,
) -> list[FrameSample]:
    """Decode a short take and keep sharp frontal frames."""
    cv2 = _cv2()
    path = Path(video)
    if not path.is_file():
        raise CaptureError(f"Video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise CaptureError(f"Could not open video: {path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 1e-3:
        native_fps = 30.0
    stride = max(int(round(native_fps / max(sample_fps, 0.5))), 1)
    frames: list[FrameSample] = []
    index = 0
    digest = report if report is not None else RejectReport()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                sample = analyze_frame(
                    frame,
                    index=index,
                    time_seconds=index / native_fps,
                    width=width,
                    height=height,
                    min_sharpness=min_sharpness,
                    report=digest,
                )
                if sample is not None:
                    frames.append(sample)
            index += 1
    finally:
        capture.release()

    if not frames:
        detail = "; ".join(digest.format_lines())
        raise CaptureError(
            "No usable frontal frames. Need a sharp, front-facing face without "
            f"sunglasses; try better light or stills (--rest/--smile/--open). {detail}"
        )
    return frames


def resample_frames_hires(
    video: str | Path,
    samples: Sequence[FrameSample],
    *,
    size: int = DISPLAY_SIZE,
) -> dict[int, FrameSample]:
    """Re-cut already-accepted frames at display resolution (AMIN step 11).

    Selection and registration happen on the grid-sized pass; only the pixels
    that become display plates are re-cropped from the source video at high
    resolution. Sharpness gating is skipped — the frame already passed QA.
    Frames that fail re-detection simply keep their grid-sized version.
    """
    cv2 = _cv2()
    wanted = sorted({int(sample.index) for sample in samples})
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(Path(video)))
    if not capture.isOpened():
        return {}
    by_index = {int(sample.index): sample for sample in samples}
    hires: dict[int, FrameSample] = {}
    try:
        for index in wanted:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            sample = analyze_frame(
                frame,
                index=index,
                time_seconds=by_index[index].time_seconds,
                width=size,
                height=size,
                min_sharpness=0.0,
            )
            if sample is not None:
                hires[index] = sample
    finally:
        capture.release()
    print(
        f"capture: {len(hires)}/{len(wanted)} display frames re-cut at {size}px "
        "(grid registration unchanged)"
    )
    return hires


def load_still_as_sample(
    path: str | Path,
    *,
    role: str,
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT,
    allow_soft: bool = False,
    report: RejectReport | None = None,
) -> FrameSample:
    """Load a role still. Fail-closed unless ``allow_soft`` lowers sharpness only."""
    cv2 = _cv2()
    file = Path(path)
    if not file.is_file():
        raise CaptureError(f"{role} image not found: {file}")
    image = cv2.imread(str(file), cv2.IMREAD_COLOR)
    if image is None:
        raise CaptureError(f"Could not read {role} image: {file}")
    min_sharp = MIN_SHARPNESS_SOFT if allow_soft else MIN_SHARPNESS
    sample = analyze_frame(
        image,
        index=0,
        time_seconds=0.0,
        width=width,
        height=height,
        min_sharpness=min_sharp,
        report=report,
    )
    if sample is None:
        reasons = ""
        if report is not None and report.reasons:
            reasons = " (" + ", ".join(sorted(report.reasons.keys())) + ")"
        raise CaptureError(
            f"{role} still failed capture gates{reasons}. "
            "Retake frontal, sharp, eyes visible; use --allow-soft only for mild blur."
        )
    return sample


def _pick_rest(pool: Sequence[FrameSample]) -> FrameSample:
    # Prefer closed lips first — teeth-before-open used to pick a wide-open
    # "ah" with slightly lower teeth score as the immutable identity photo.
    # Prefer frames that already pass rest gates when any exist.
    closed = [
        f
        for f in pool
        if f.metrics.mouth_open <= MAX_REST_MOUTH_OPEN
        and f.metrics.teeth <= MAX_REST_TEETH
    ]
    use = closed if closed else list(pool)
    # Among equal mouth_open (landmark floor ~0.15), prefer low smile width so
    # a calm closed-lip frame beats a parted-lip worried look with less teeth.
    return min(
        use,
        key=lambda f: (
            f.metrics.mouth_open,
            f.metrics.smile_width,
            f.metrics.teeth,
            -f.metrics.sharpness,
        ),
    )


def _pick_open(pool: Sequence[FrameSample]) -> FrameSample:
    return max(pool, key=lambda f: (f.metrics.mouth_open, f.metrics.sharpness))


def _pick_smile(pool: Sequence[FrameSample]) -> FrameSample:
    return max(
        pool,
        key=lambda f: (
            f.metrics.smile_width - 1.8 * f.metrics.mouth_open,
            f.metrics.sharpness,
        ),
    )


def _surprise_score(frame: FrameSample) -> float:
    """Raised brows + wider lids; mild mouth open is fine, smile is not."""
    m = frame.metrics
    return (
        m.brow_raise * 3.2
        + m.lid_open * 4.0
        + min(m.mouth_open, 0.08) * 0.4
        - m.smile_width * 0.35
        + m.sharpness * 0.0004
    )


def _pick_surprise(
    pool: Sequence[FrameSample],
    *,
    exclude: set[int] | None = None,
    rest: FrameSample | None = None,
) -> FrameSample:
    """Best surprise / brow-up frame learned from the take."""
    banned = exclude or set()
    candidates = [f for f in pool if f.index not in banned] or list(pool)
    best = max(candidates, key=_surprise_score)
    if rest is None:
        return best
    # Prefer a frame that actually lifts brows/lids vs rest; else keep best.
    if (
        best.metrics.brow_raise >= rest.metrics.brow_raise + 0.008
        or best.metrics.lid_open >= rest.metrics.lid_open + 0.006
    ):
        return best
    lifted = [
        f
        for f in candidates
        if f.metrics.brow_raise >= rest.metrics.brow_raise + 0.004
        or f.metrics.lid_open >= rest.metrics.lid_open + 0.003
    ]
    if lifted:
        return max(lifted, key=_surprise_score)
    return best


def validate_selection(
    selection: CaptureSelection,
    *,
    stills: bool = False,
) -> None:
    """Fail closed when rest / smile / open are not distinct enough."""
    open_need = MIN_OPEN_DELTA * (0.55 if stills else 1.0)
    smile_need = MIN_SMILE_WIDTH_DELTA * (0.55 if stills else 1.0)
    rest_o = selection.rest.metrics.mouth_open
    open_o = selection.open.metrics.mouth_open
    rest_w = selection.rest.metrics.smile_width
    smile_w = selection.smile.metrics.smile_width
    open_delta = open_o - rest_o
    smile_delta = smile_w - rest_w
    if rest_o > MAX_REST_MOUTH_OPEN:
        raise CaptureError(
            f"Rest/identity mouth_open={rest_o:.3f} > {MAX_REST_MOUTH_OPEN:.3f} "
            "(identity would stay open when LOOK is REST). "
            "Retake with a true closed-mouth neutral."
        )
    if selection.rest.metrics.teeth > MAX_REST_TEETH:
        raise CaptureError(
            f"Rest/identity shows teeth (teeth={selection.rest.metrics.teeth:.3f} > "
            f"{MAX_REST_TEETH:.3f}). Retake with lips closed."
        )
    if (not stills and selection.open.index == selection.rest.index) or open_delta < open_need:
        raise CaptureError(
            f"Open beat too similar to rest "
            f"(delta open={open_delta:.3f} < {open_need:.3f}). "
            "Retake with a clear jaw drop / 'ah' so teeth are visible."
        )
    # Scripted video takes: smile phase already isolates the beat; smile_width
    # from OpenCV can saturate, so only require a distinct smile frame.
    scripted = selection.phase_mode == "scripted-quarters"
    if scripted and selection.smile.index != selection.rest.index:
        return
    if (not stills and selection.smile.index == selection.rest.index) or smile_delta < smile_need:
        raise CaptureError(
            f"Smile beat too similar to rest "
            f"(delta smile={smile_delta:.3f} < {smile_need:.3f}). "
            "Retake with a clear closed-lip smile."
        )


def _ensure_still_role_metrics(
    rest: FrameSample, smile: FrameSample, open_f: FrameSample
) -> tuple[FrameSample, FrameSample, FrameSample]:
    """Nudge still metrics from mouth-band darkness when MediaPipe is absent."""

    def mouth_dark_pct(sample: FrameSample) -> float:
        gray = sample.image_bgr.mean(axis=2)
        face = sample.face
        mx = int(face.x + face.width * 0.5)
        my = int(face.y + face.height * 0.78)
        half_w = max(4, int(face.width * 0.14))
        half_h = max(3, int(face.height * 0.07))
        patch = gray[my - half_h : my + half_h, mx - half_w : mx + half_w]
        return float(np.percentile(patch, 20)) if patch.size else 128.0

    rest_d, smile_d, open_d = (
        mouth_dark_pct(rest),
        mouth_dark_pct(smile),
        mouth_dark_pct(open_f),
    )
    open_boost = float(np.clip((rest_d - open_d) / 60.0, 0.0, 0.30))
    smile_boost = float(np.clip((rest_d - smile_d) / 90.0 + 0.02, 0.0, 0.14))
    open_need = MIN_OPEN_DELTA * 0.55
    smile_need = MIN_SMILE_WIDTH_DELTA * 0.55

    def with_metrics(sample: FrameSample, **updates: float) -> FrameSample:
        base = sample.metrics
        metrics = ExpressionMetrics(
            mouth_open=float(updates.get("mouth_open", base.mouth_open)),
            smile_width=float(updates.get("smile_width", base.smile_width)),
            sharpness=base.sharpness,
            landmark_quality=base.landmark_quality,
            method=base.method,
            eye_span=base.eye_span,
            eye_aperture=base.eye_aperture,
            mouth_offset=base.mouth_offset,
            teeth=base.teeth,
            brow_raise=float(updates.get("brow_raise", base.brow_raise)),
            lid_open=float(updates.get("lid_open", base.lid_open)),
        )
        return FrameSample(
            index=sample.index,
            time_seconds=sample.time_seconds,
            image_bgr=sample.image_bgr,
            face=sample.face,
            landmarks_meta=sample.landmarks_meta,
            metrics=metrics,
        )

    open_target = max(
        open_f.metrics.mouth_open,
        rest.metrics.mouth_open + open_boost,
    )
    # Role-labeled stills that differ in pixels must clear the stills open gate.
    if not np.array_equal(rest.image_bgr, open_f.image_bgr):
        open_target = max(open_target, rest.metrics.mouth_open + open_need + 0.002)
    open_f = with_metrics(open_f, mouth_open=open_target)

    smile_width = max(
        smile.metrics.smile_width, rest.metrics.smile_width + smile_boost
    )
    if not np.array_equal(rest.image_bgr, smile.image_bgr):
        smile_width = max(smile_width, rest.metrics.smile_width + smile_need + 0.002)
    smile = with_metrics(
        smile,
        smile_width=smile_width,
        mouth_open=min(smile.metrics.mouth_open, rest.metrics.mouth_open + 0.04),
    )
    return rest, smile, open_f


def select_expression_frames(
    frames: Sequence[FrameSample],
    *,
    validate: bool = True,
    calibration_script: dict[str, Any] | None = None,
) -> CaptureSelection:
    """Pick rest / smile / open; prefer calibration beats when provided.

    With a 9s script (REST/SMILE/OPEN/BLINK/…), open.png must come from the
    OPEN beat — global max openness often lands on SAY_HI/SURPRISE after BLINK
    and atlas AA previously won on closed-lid BLINK frames.
    """
    if not frames:
        raise CaptureError("No frames to select from")

    def sharp_ok(frame: FrameSample) -> bool:
        return frame.metrics.sharpness >= MIN_SHARPNESS * 0.75

    pool = [frame for frame in frames if sharp_ok(frame)] or list(frames)
    t_max = max(frame.time_seconds for frame in pool)
    t_min = min(frame.time_seconds for frame in pool)
    duration = t_max - t_min
    phase_counts = {"rest": 0, "smile": 0, "open": 0, "talk": 0}

    script = calibration_script
    if script is not None and script.get("beats"):
        from aiface.tickfeed.calibration import filter_frames_by_beats

        # Never mine LOOK mouth plates from the BLINK beat.
        mouth_pool = filter_frames_by_beats(
            pool, script, exclude={"BLINK"}
        ) or pool
        rest_pool = filter_frames_by_beats(
            mouth_pool, script, include={"REST"}
        ) or mouth_pool
        smile_pool = filter_frames_by_beats(
            mouth_pool, script, include={"SMILE"}
        ) or mouth_pool
        open_pool = filter_frames_by_beats(
            mouth_pool, script, include={"OPEN"}
        ) or mouth_pool
        talk_pool = filter_frames_by_beats(
            mouth_pool, script, include={"SAY_HI", "TALK", "SURPRISE", "ANGRY"}
        ) or mouth_pool
        surprise_pool = filter_frames_by_beats(
            mouth_pool, script, include={"SURPRISE", "SAY_HI"}
        ) or mouth_pool
        phase_counts = {
            "rest": len(rest_pool),
            "smile": len(smile_pool),
            "open": len(open_pool),
            "talk": len(talk_pool),
        }
        rest = _pick_rest(rest_pool)
        open_frame = _pick_open(open_pool)
        smile_widths = [f.metrics.smile_width for f in smile_pool]
        smile_flat = (
            len(smile_widths) >= 2
            and float(np.std(smile_widths)) < 0.012
        )
        if smile_flat:
            smile = min(
                smile_pool,
                key=lambda f: (f.metrics.mouth_open, -f.metrics.sharpness),
            )
        else:
            smile = _pick_smile(smile_pool)
        surprise = _pick_surprise(
            surprise_pool,
            exclude={rest.index, smile.index},
            rest=rest,
        )
        talk = [
            frame
            for frame in talk_pool
            if frame.index not in {rest.index, smile.index, open_frame.index}
        ] or list(mouth_pool)
        selection = CaptureSelection(
            rest=rest,
            smile=smile,
            open=open_frame,
            surprise=surprise,
            talk_frames=talk,
            phase_mode="calibration-beats",
            phase_counts=phase_counts,
        )
    elif duration >= PHASE_SPLIT_SECONDS and len(pool) >= 8:
        # Scripted quarters: rest | smile | open | talk
        edges = [
            t_min,
            t_min + 0.25 * duration,
            t_min + 0.45 * duration,
            t_min + 0.60 * duration,
            t_max + 1e-6,
        ]
        buckets: dict[str, list[FrameSample]] = {
            "rest": [],
            "smile": [],
            "open": [],
            "talk": [],
        }
        for frame in pool:
            t = frame.time_seconds
            if t < edges[1]:
                buckets["rest"].append(frame)
            elif t < edges[2]:
                buckets["smile"].append(frame)
            elif t < edges[3]:
                buckets["open"].append(frame)
            else:
                buckets["talk"].append(frame)
        phase_counts = {name: len(items) for name, items in buckets.items()}
        rest_pool = buckets["rest"] or pool
        smile_pool = buckets["smile"] or pool
        # Best open plate = global max openness (teeth), not only the open quarter —
        # generative clips often mistime the jaw-drop beat.
        open_frame = _pick_open(pool)
        rest = _pick_rest(rest_pool)
        smile_widths = [f.metrics.smile_width for f in smile_pool]
        smile_flat = (
            len(smile_widths) >= 2
            and float(np.std(smile_widths)) < 0.012
        )
        if smile_flat:
            # Closed-lip smile proxy: lowest openness in the smile quarter.
            smile = min(
                smile_pool,
                key=lambda f: (f.metrics.mouth_open, -f.metrics.sharpness),
            )
        else:
            smile = _pick_smile(smile_pool)
        talk = buckets["talk"] or [
            frame
            for frame in pool
            if frame.index not in {rest.index, smile.index, open_frame.index}
        ]
        surprise = _pick_surprise(
            pool,
            exclude={rest.index, smile.index},
            rest=rest,
        )
        selection = CaptureSelection(
            rest=rest,
            smile=smile,
            open=open_frame,
            surprise=surprise,
            talk_frames=talk or list(pool),
            phase_mode="scripted-quarters",
            phase_counts=phase_counts,
        )
    else:
        rest = _pick_rest(pool)
        open_frame = _pick_open(pool)
        smile = _pick_smile(pool)
        surprise = _pick_surprise(
            pool,
            exclude={rest.index, smile.index},
            rest=rest,
        )
        talk = [
            frame
            for frame in pool
            if frame.index
            not in {rest.index, smile.index, open_frame.index, surprise.index}
        ]
        if not talk:
            talk = list(pool)
        selection = CaptureSelection(
            rest=rest,
            smile=smile,
            open=open_frame,
            surprise=surprise,
            talk_frames=talk,
            phase_mode="global",
            phase_counts=phase_counts,
        )

    if validate:
        validate_selection(selection)
    return selection


def compute_travel_priors(talk_frames: Sequence[FrameSample]) -> TravelPriors:
    """Derive jaw/lip travel scales from talk-segment landmark curves."""
    if not talk_frames:
        return TravelPriors()
    opens = np.array([f.metrics.mouth_open for f in talk_frames], dtype=np.float64)
    widths = np.array([f.metrics.smile_width for f in talk_frames], dtype=np.float64)
    peak_open = float(np.percentile(opens, 95))
    peak_width = float(np.percentile(widths, 95))
    jaw_scale = float(np.clip(0.85 + peak_open * 3.5, 0.85, 1.35))
    width_scale = float(np.clip(0.90 + (peak_width - 0.30) * 1.5, 0.85, 1.30))
    open_scale = float(np.clip(0.90 + peak_open * 2.5, 0.85, 1.35))
    return TravelPriors(
        jaw_travel_scale=jaw_scale,
        lip_width_scale=width_scale,
        lip_open_scale=open_scale,
        peak_mouth_open=peak_open,
        peak_smile_width=peak_width,
        frame_count=len(talk_frames),
    )


def render_capture_qa(
    rest: FrameSample,
    smile: FrameSample,
    open_frame: FrameSample,
    surprise: FrameSample | None = None,
) -> npt.NDArray[np.uint8]:
    """Contact sheet: rest | smile | open | surprise with landmarks."""
    from aiface.landmarks import FaceLandmarks, render_seed_qa

    cv2 = _cv2()

    def sheet(sample: FrameSample, label: str) -> npt.NDArray[np.uint8]:
        meta = sample.landmarks_meta
        landmarks = FaceLandmarks(
            face=sample.face,
            left_eye=(
                float(meta["left_eye_image"]["x"]),
                float(meta["left_eye_image"]["y"]),
            ),
            right_eye=(
                float(meta["right_eye_image"]["x"]),
                float(meta["right_eye_image"]["y"]),
            ),
            mouth=(
                float(meta["mouth_center_image"]["x"]),
                float(meta["mouth_center_image"]["y"]),
            ),
            left_brow=(
                float(meta.get("left_brow_image", meta["left_eye_image"])["x"]),
                float(meta.get("left_brow_image", meta["left_eye_image"])["y"]),
            ),
            right_brow=(
                float(meta.get("right_brow_image", meta["right_eye_image"])["x"]),
                float(meta.get("right_brow_image", meta["right_eye_image"])["y"]),
            ),
            method=str(meta.get("method", sample.metrics.method)),
            quality=float(meta.get("quality", sample.metrics.landmark_quality)),
        )
        canvas = render_seed_qa(sample.image_bgr, landmarks)
        cv2.putText(
            canvas,
            f"{label} o={sample.metrics.mouth_open:.2f} "
            f"br={sample.metrics.brow_raise:.2f} "
            f"lid={sample.metrics.lid_open:.2f}",
            (8, canvas.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        return canvas

    panels = [sheet(rest, "REST"), sheet(smile, "SMILE"), sheet(open_frame, "OPEN")]
    if surprise is not None:
        panels.append(sheet(surprise, "SURPRISE"))
    return np.concatenate(panels, axis=1)


def _merge_capture_into_seed_metadata(
    world: Path,
    priors: TravelPriors,
    selection: CaptureSelection,
    source: str,
    reject_report: RejectReport | None,
) -> None:
    from aiface.runtime.bds import load_bds, save_bds

    header, tensor = load_bds(world)
    app_meta = dict(header.get("application_metadata") or {})
    avatar = dict(app_meta.get("avatar_seed") or {})
    # Keep the BDS header lean (4 KB cap). Full metrics / rejects live in
    # capture_meta.json and expression_catalog.json beside the world.
    plates = {"smile": SMILE_PLATE_NAME, "open": OPEN_PLATE_NAME}
    surprise = selection.surprise
    if surprise is not None:
        plates["surprise"] = SURPRISE_PLATE_NAME
    avatar["capture"] = {
        "version": "avatar-capture-1.3",
        "source": source,
        "frames": {
            "rest": selection.rest.index,
            "smile": selection.smile.index,
            "open": selection.open.index,
            "surprise": surprise.index if surprise is not None else None,
        },
        "phase_mode": selection.phase_mode,
        "plates": plates,
        "catalog": "expression_catalog.json",
        "priors": asdict(priors),
    }
    app_meta["avatar_seed"] = avatar
    save_bds(world, tensor, metadata=app_meta)


def _write_plate_atlas(
    destination: Path,
    frames: Sequence[FrameSample],
    *,
    source_label: str,
    hires: dict[int, FrameSample] | None = None,
    reference: FrameSample | None = None,
    exclude_time_ranges: Sequence[tuple[float, float]] | None = None,
    prefer_time_ranges: Sequence[tuple[float, float]] | None = None,
) -> list[Path]:
    from aiface.plates import (
        AtlasPlate,
        default_atlas_dir,
        default_atlas_meta_path,
        select_viseme_atlas_frames,
        write_plate_atlas_meta,
    )

    atlas_dir = default_atlas_dir(destination)
    if atlas_dir.is_dir():
        for old in atlas_dir.glob("plate_*.png"):
            old.unlink(missing_ok=True)
    else:
        atlas_dir.mkdir(parents=True, exist_ok=True)

    # AMIN step 13: landmark-match one real frame per viseme (unique plates).
    chosen, viseme_to_plate = select_viseme_atlas_frames(
        frames,
        exclude_time_ranges=exclude_time_ranges,
        prefer_time_ranges=prefer_time_ranges,
    )
    plate_visemes: dict[int, str] = {}
    for viseme, idx in viseme_to_plate.items():
        plate_visemes.setdefault(int(idx), viseme)
    records: list[AtlasPlate] = []
    paths: list[Path] = []
    for index, frame in enumerate(chosen):
        # AMIN step 11: write the display pixels from the hi-res re-cut when
        # available; selection metrics stay from the grid-sized analysis pass.
        plate = (hires or {}).get(int(frame.index), frame)
        tag = plate_visemes.get(index, "")
        # Closed/PP stay tight oral. Speech shapes need enough α that L07 can
        # own mid-band — tiny mattes + open mute = stuck rest smile.
        closed_tags = {"CLOSED", "PP", "REST", "MM"}
        speech_open = float(frame.metrics.mouth_open)
        if tag not in closed_tags and speech_open > 0.05:
            alpha = build_expression_region_matte(
                plate.image_bgr.shape[0],
                plate.image_bgr.shape[1],
                plate.face,
                plate.landmarks_meta,
            )
            oral = build_mouth_interior_matte(
                plate.image_bgr.shape[0],
                plate.image_bgr.shape[1],
                plate.face,
                plate.landmarks_meta,
                openness=max(speech_open, 0.30),
            )
            alpha = np.maximum(oral, alpha * 0.62).astype(np.float32)
        else:
            # Tight oral matte — wide cheek mattes stamp smile corners when closed.
            alpha = build_mouth_interior_matte(
                plate.image_bgr.shape[0],
                plate.image_bgr.shape[1],
                plate.face,
                plate.landmarks_meta,
                openness=max(speech_open, 0.04),
            )
        pixels = plate.image_bgr
        if reference is not None:
            pixels = match_plate_to_reference(pixels, reference.image_bgr, alpha)
        rel = f"plates/plate_{index:02d}.png"
        path = write_expression_plate(destination.parent / rel, pixels, alpha)
        paths.append(path)
        open_v = float(frame.metrics.mouth_open)
        # Closed / PP plates must index as sealed — capture floor (~0.15) lied.
        if tag in {"CLOSED", "PP", "REST", "MM"} or open_v <= 0.16:
            if tag in {"CLOSED", "PP", "REST", "MM"}:
                open_v = 0.0
        records.append(
            AtlasPlate(
                index=index,
                path=rel.replace("\\", "/"),
                openness=open_v,
                smile_width=float(frame.metrics.smile_width),
                frame_index=int(frame.index),
                time_seconds=float(frame.time_seconds),
                viseme=tag,
            )
        )
    write_plate_atlas_meta(
        default_atlas_meta_path(destination),
        records,
        source=source_label,
        viseme_to_plate=viseme_to_plate,
    )
    print(
        f"Plate atlas step 13: {len(records)} shapes, "
        f"{len(viseme_to_plate)} viseme->plate bindings",
        flush=True,
    )
    return paths


def _write_expression_catalog(
    destination: Path,
    selection: CaptureSelection,
    *,
    source_label: str,
) -> Path:
    """Write the fast expression DB next to the BDS."""
    from aiface.expression_catalog import (
        EMOTION_ROLE_DEFAULTS,
        ExpressionCatalog,
        ExpressionRole,
        default_catalog_path,
        write_expression_catalog,
    )
    from aiface.plates import VISEME_OPENNESS

    rest = selection.rest
    smile = selection.smile
    open_f = selection.open
    surprise = selection.surprise or rest

    def relative_brow(frame: FrameSample) -> float:
        return float(
            np.clip(
                (frame.metrics.brow_raise - rest.metrics.brow_raise) / 0.06,
                0.0,
                1.0,
            )
        )

    def relative_widen(frame: FrameSample) -> float:
        return float(
            np.clip(
                (frame.metrics.lid_open - rest.metrics.lid_open) / 0.05,
                0.0,
                1.0,
            )
        )

    roles = {
        "rest": ExpressionRole(
            name="rest",
            plate="source_face.png",
            frame_index=rest.index,
            time_seconds=rest.time_seconds,
            mouth_open=rest.metrics.mouth_open,
            smile_width=rest.metrics.smile_width,
            brow_raise=0.0,
            eye_widen=0.0,
            teeth=rest.metrics.teeth,
            notes="Immutable identity rest",
        ),
        "smile": ExpressionRole(
            name="smile",
            plate=SMILE_PLATE_NAME,
            frame_index=smile.index,
            time_seconds=smile.time_seconds,
            mouth_open=smile.metrics.mouth_open,
            smile_width=smile.metrics.smile_width,
            brow_raise=relative_brow(smile) * 0.25,
            eye_widen=relative_widen(smile) * 0.15,
            teeth=smile.metrics.teeth,
            notes="Closed-lip smile plate",
        ),
        "open": ExpressionRole(
            name="open",
            plate=OPEN_PLATE_NAME,
            frame_index=open_f.index,
            time_seconds=open_f.time_seconds,
            mouth_open=open_f.metrics.mouth_open,
            smile_width=open_f.metrics.smile_width,
            brow_raise=relative_brow(open_f) * 0.35,
            eye_widen=relative_widen(open_f) * 0.25,
            teeth=open_f.metrics.teeth,
            notes="Open mouth / teeth plate",
        ),
        "surprise": ExpressionRole(
            name="surprise",
            plate=SURPRISE_PLATE_NAME,
            frame_index=surprise.index,
            time_seconds=surprise.time_seconds,
            mouth_open=surprise.metrics.mouth_open,
            smile_width=surprise.metrics.smile_width,
            brow_raise=max(0.55, relative_brow(surprise)),
            eye_widen=max(0.55, relative_widen(surprise)),
            teeth=surprise.metrics.teeth,
            notes="Raised brows + wider lids from capture video",
        ),
    }
    catalog = ExpressionCatalog(
        version="expression-catalog-1.0",
        source=source_label,
        roles=roles,
        emotion_map=dict(EMOTION_ROLE_DEFAULTS),
        viseme_openness=dict(VISEME_OPENNESS),
    )
    return write_expression_catalog(default_catalog_path(destination), catalog)


def write_capture_bundle(
    selection: CaptureSelection,
    *,
    output: str | Path,
    priors: TravelPriors,
    source_label: str,
    preview: bool = True,
    reject_report: RejectReport | None = None,
    atlas_frames: Sequence[FrameSample] | None = None,
    hires: dict[int, FrameSample] | None = None,
    exclude_time_ranges: Sequence[tuple[float, float]] | None = None,
    prefer_time_ranges: Sequence[tuple[float, float]] | None = None,
) -> CaptureResult:
    """Seed from rest, write plates + expression catalog + atlas, meta, QA."""
    from aiface.seed import build_avatar_seed, write_seed_bundle

    def display(frame: FrameSample) -> FrameSample:
        """Hi-res re-cut of the same source frame, or the frame itself."""
        return (hires or {}).get(int(frame.index), frame)

    if (
        selection.rest.metrics.teeth > MAX_REST_TEETH
        or selection.rest.metrics.mouth_open > MAX_REST_MOUTH_OPEN
    ):
        print(
            "warning: rest/identity not closed "
            f"(mouth_open={selection.rest.metrics.mouth_open:.3f}, "
            f"teeth={selection.rest.metrics.teeth:.3f}). "
            "Retake with a true closed-mouth neutral for a calm identity.",
            flush=True,
        )

    destination = Path(output)
    if destination.suffix.lower() != ".bds":
        destination = destination / "avatar_face.bds"
    destination.parent.mkdir(parents=True, exist_ok=True)

    rest_path = destination.parent / "_capture_rest_tmp.png"
    write_portrait(rest_path, selection.rest.image_bgr)
    try:
        seed = build_avatar_seed(rest_path, normalize=False)
        written = write_seed_bundle(seed, destination, preview=preview)
    finally:
        if rest_path.is_file():
            rest_path.unlink(missing_ok=True)

    # AMIN step 11: the .bds seeded from the grid-sized rest above; the
    # renderer's photo is the same frame re-cut at display resolution.
    rest_display = display(selection.rest)
    if rest_display is not selection.rest and "portrait" in written:
        written["portrait"] = write_portrait(
            written["portrait"], rest_display.image_bgr
        )

    smile_display = display(selection.smile)
    open_display = display(selection.open)
    smile_alpha = build_expression_region_matte(
        smile_display.image_bgr.shape[0],
        smile_display.image_bgr.shape[1],
        smile_display.face,
        smile_display.landmarks_meta,
    )
    open_alpha = build_expression_region_matte(
        open_display.image_bgr.shape[0],
        open_display.image_bgr.shape[1],
        open_display.face,
        open_display.landmarks_meta,
    )
    smile_path = write_expression_plate(
        default_smile_plate_path(destination),
        match_plate_to_reference(
            smile_display.image_bgr, rest_display.image_bgr, smile_alpha
        ),
        smile_alpha,
    )
    open_path = write_expression_plate(
        default_open_plate_path(destination),
        match_plate_to_reference(
            open_display.image_bgr, rest_display.image_bgr, open_alpha
        ),
        open_alpha,
    )
    written["smile"] = smile_path
    written["open"] = open_path

    surprise = selection.surprise
    if surprise is None:
        surprise = _pick_surprise(
            [selection.rest, selection.smile, selection.open, *selection.talk_frames],
            rest=selection.rest,
        )
        selection.surprise = surprise
    surprise_display = display(surprise)
    surprise_alpha = build_upper_face_matte(
        surprise_display.image_bgr.shape[0],
        surprise_display.image_bgr.shape[1],
        surprise_display.face,
        surprise_display.landmarks_meta,
    )
    surprise_path = write_expression_plate(
        default_surprise_plate_path(destination),
        match_plate_to_reference(
            surprise_display.image_bgr, rest_display.image_bgr, surprise_alpha
        ),
        surprise_alpha,
    )
    written["surprise"] = surprise_path

    bank = list(atlas_frames or [])
    if not bank:
        bank = [
            selection.rest,
            selection.smile,
            selection.open,
            surprise,
            *selection.talk_frames,
        ]
    atlas_paths = _write_plate_atlas(
        destination,
        bank,
        source_label=source_label,
        hires=hires,
        reference=rest_display,
        exclude_time_ranges=exclude_time_ranges,
        prefer_time_ranges=prefer_time_ranges,
    )
    written["plate_atlas"] = destination.with_name("plate_atlas.json")
    for path in atlas_paths:
        written[path.name] = path

    catalog_path = _write_expression_catalog(
        destination, selection, source_label=source_label
    )
    written["expression_catalog"] = catalog_path

    _merge_capture_into_seed_metadata(
        destination, priors, selection, source_label, reject_report
    )

    meta_payload = {
        "version": "avatar-capture-1.3",
        "source": source_label,
        "selection": {
            "rest_frame": selection.rest.index,
            "smile_frame": selection.smile.index,
            "open_frame": selection.open.index,
            "surprise_frame": surprise.index,
            "rest_time": selection.rest.time_seconds,
            "smile_time": selection.smile.time_seconds,
            "open_time": selection.open.time_seconds,
            "surprise_time": surprise.time_seconds,
            "rest_teeth": selection.rest.metrics.teeth,
        },
        "phase_mode": selection.phase_mode,
        "phase_counts": selection.phase_counts,
        "expression_catalog": catalog_path.name,
        "priors": asdict(priors),
        "rejects": reject_report.as_dict() if reject_report else None,
        "talk_series": [
            {
                "index": frame.index,
                "t": round(frame.time_seconds, 4),
                "mouth_open": round(frame.metrics.mouth_open, 4),
                "smile_width": round(frame.metrics.smile_width, 4),
                "brow_raise": round(frame.metrics.brow_raise, 4),
                "lid_open": round(frame.metrics.lid_open, 4),
                "teeth": round(frame.metrics.teeth, 4),
                "sharpness": round(frame.metrics.sharpness, 2),
            }
            for frame in selection.talk_frames[:200]
        ],
    }
    meta_path = default_capture_meta_path(destination)
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    written["capture_meta"] = meta_path

    qa_path: Path | None = None
    if preview:
        qa_path = write_portrait(
            destination.with_name(QA_CONTACT_NAME),
            render_capture_qa(
                selection.rest, selection.smile, selection.open, surprise
            ),
        )
        written["capture_qa"] = qa_path

    return CaptureResult(
        world=destination,
        portrait=written["portrait"],
        smile_plate=smile_path,
        open_plate=open_path,
        meta=meta_path,
        qa=qa_path,
        priors=priors,
        written=written,
        reject_report=reject_report,
    )


def run_capture_from_video(
    video: str | Path,
    *,
    output: str | Path = DEFAULT_AVATAR_FACE,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    preview: bool = True,
    allow_soft: bool = False,
) -> CaptureResult:
    report = RejectReport()
    frames = iter_video_frames(
        video,
        sample_fps=sample_fps,
        min_sharpness=MIN_SHARPNESS_SOFT if allow_soft else MIN_SHARPNESS,
        report=report,
    )
    # Prefer TickFeed calibration beats when the script sits next to the world.
    script = None
    out_path = Path(output)
    world_dir = out_path.parent if out_path.suffix.lower() == ".bds" else out_path
    script_path = world_dir / "calibration_script.json"
    if script_path.is_file():
        try:
            script = json.loads(script_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            script = None
    selection = select_expression_frames(frames, calibration_script=script)
    priors = compute_travel_priors(selection.talk_frames)
    # Pre-pick surprise so its hi-res re-cut is available to the bundle too.
    if selection.surprise is None:
        selection.surprise = _pick_surprise(
            [selection.rest, selection.smile, selection.open, *selection.talk_frames],
            rest=selection.rest,
        )
    from aiface.plates import select_viseme_atlas_frames

    exclude_ranges = None
    prefer_ranges = None
    if script is not None:
        from aiface.tickfeed.calibration import beat_time_ranges

        exclude_ranges = beat_time_ranges(script, {"BLINK"})
        prefer_ranges = beat_time_ranges(script, {"OPEN", "SAY_HI", "TALK"})
    atlas_frames, _viseme_map = select_viseme_atlas_frames(
        frames,
        exclude_time_ranges=exclude_ranges,
        prefer_time_ranges=prefer_ranges,
    )
    display_targets: list[FrameSample] = [
        selection.rest,
        selection.smile,
        selection.open,
        *([selection.surprise] if selection.surprise is not None else []),
        *atlas_frames,
    ]
    hires = resample_frames_hires(video, display_targets)
    return write_capture_bundle(
        selection,
        output=output,
        priors=priors,
        source_label=str(Path(video).name),
        preview=preview,
        reject_report=report,
        atlas_frames=frames,
        hires=hires,
        exclude_time_ranges=exclude_ranges,
        prefer_time_ranges=prefer_ranges,
    )


def run_capture_from_stills(
    *,
    rest: str | Path,
    smile: str | Path,
    open_image: str | Path,
    output: str | Path = DEFAULT_AVATAR_FACE,
    preview: bool = True,
    allow_soft: bool = False,
    validate: bool = True,
) -> CaptureResult:
    report = RejectReport()
    rest_f = load_still_as_sample(
        rest, role="rest", allow_soft=allow_soft, report=report
    )
    smile_f = load_still_as_sample(
        smile, role="smile", allow_soft=allow_soft, report=report
    )
    open_f = load_still_as_sample(
        open_image, role="open", allow_soft=allow_soft, report=report
    )
    # Tag indices so selection validation can tell roles apart even when
    # metrics are close; stills are already role-labeled by the user.
    rest_f = FrameSample(
        index=0,
        time_seconds=0.0,
        image_bgr=rest_f.image_bgr,
        face=rest_f.face,
        landmarks_meta=rest_f.landmarks_meta,
        metrics=rest_f.metrics,
    )
    smile_f = FrameSample(
        index=1,
        time_seconds=1.0,
        image_bgr=smile_f.image_bgr,
        face=smile_f.face,
        landmarks_meta=smile_f.landmarks_meta,
        metrics=smile_f.metrics,
    )
    open_f = FrameSample(
        index=2,
        time_seconds=2.0,
        image_bgr=open_f.image_bgr,
        face=open_f.face,
        landmarks_meta=open_f.landmarks_meta,
        metrics=open_f.metrics,
    )
    rest_f, smile_f, open_f = _ensure_still_role_metrics(rest_f, smile_f, open_f)
    surprise_f = _pick_surprise(
        [rest_f, smile_f, open_f], exclude={rest_f.index}, rest=rest_f
    )
    selection = CaptureSelection(
        rest=rest_f,
        smile=smile_f,
        open=open_f,
        surprise=surprise_f,
        talk_frames=[rest_f, smile_f, open_f],
        phase_mode="stills",
        phase_counts={"rest": 1, "smile": 1, "open": 1, "talk": 0},
    )
    if validate:
        validate_selection(selection, stills=True)
    priors = compute_travel_priors(selection.talk_frames)
    # AMIN step 11: stills are already on disk — reload each at display res.
    hires: dict[int, FrameSample] = {}
    for tag_index, (still, role) in enumerate(
        ((rest, "rest"), (smile, "smile"), (open_image, "open"))
    ):
        try:
            sample = load_still_as_sample(
                still,
                role=role,
                width=DISPLAY_SIZE,
                height=DISPLAY_SIZE,
                allow_soft=True,
            )
        except (CaptureError, AvatarSeedError):
            continue
        hires[tag_index] = FrameSample(
            index=tag_index,
            time_seconds=float(tag_index),
            image_bgr=sample.image_bgr,
            face=sample.face,
            landmarks_meta=sample.landmarks_meta,
            metrics=sample.metrics,
        )
    return write_capture_bundle(
        selection,
        output=output,
        priors=priors,
        source_label="stills",
        preview=preview,
        reject_report=report,
        hires=hires,
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aiface-capture",
        description=(
            "Digest a short high-quality face video (or rest/smile/open stills) "
            "into a Path 1 avatar seed plus real expression plates. "
            "Not the NWR terrain video converter. See docs/AvatarCapture.md."
        ),
    )
    parser.add_argument("--video", type=Path, default=None, help="Short frontal MP4/MOV")
    parser.add_argument("--rest", type=Path, default=None, help="Rest still (closed mouth)")
    parser.add_argument("--smile", type=Path, default=None, help="Smile still")
    parser.add_argument(
        "--open", type=Path, default=None, dest="open_image", help="Open-mouth still"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_AVATAR_FACE,
        help=f"Output .bds or directory (default {DEFAULT_AVATAR_FACE})",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=DEFAULT_SAMPLE_FPS,
        help="Frame sample rate when reading --video",
    )
    parser.add_argument(
        "--allow-soft",
        action="store_true",
        help="Allow slightly lower sharpness (still enforces yaw/eyes/selection)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip capture_qa.png contact sheet",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_arguments(argv)
    ensure_output_tree()
    started = time.perf_counter()
    try:
        if options.video is not None:
            result = run_capture_from_video(
                options.video,
                output=options.output,
                sample_fps=options.sample_fps,
                preview=not options.no_preview,
                allow_soft=options.allow_soft,
            )
        elif options.rest and options.smile and options.open_image:
            result = run_capture_from_stills(
                rest=options.rest,
                smile=options.smile,
                open_image=options.open_image,
                output=options.output,
                preview=not options.no_preview,
                allow_soft=options.allow_soft,
            )
        else:
            print(
                "error: provide --video FACE.mp4 or --rest/--smile/--open stills",
                flush=True,
            )
            return 2
    except (CaptureError, AvatarSeedError, ValueError, OSError) as exc:
        message = str(exc).encode("ascii", errors="replace").decode("ascii")
        print(f"error: {message}", flush=True)
        return 1

    if result.reject_report is not None:
        for line in result.reject_report.format_lines():
            print(line)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(f"Wrote {result.world}")
    print(f"Wrote {result.portrait} (rest identity)")
    print(f"Wrote {result.smile_plate} (smile plate + mouth matte)")
    print(f"Wrote {result.open_plate} (open-mouth plate + mouth matte)")
    if "surprise" in result.written:
        print(f"Wrote {result.written['surprise']} (surprise / brow plate)")
    if "expression_catalog" in result.written:
        print(
            f"Wrote {result.written['expression_catalog']} "
            "(expression catalog DB)"
        )
    print(f"Wrote {result.meta}")
    if result.qa is not None:
        print(f"Wrote {result.qa} (inspect REST|SMILE|OPEN|SURPRISE before play)")
        print("  QA gate: OPEN teeth/jaw; SMILE wider; SURPRISE brows/lids up.")
    print(
        f"  priors jaw×{result.priors.jaw_travel_scale:.2f} "
        f"width×{result.priors.lip_width_scale:.2f} "
        f"open×{result.priors.lip_open_scale:.2f}  "
        f"{elapsed_ms:.0f} ms"
    )
    print(f"Play with: aiface --world {result.world.as_posix()}")
    return 0


__all__ = [
    "CAPTURE_META_NAME",
    "MIN_OPEN_DELTA",
    "MIN_SMILE_WIDTH_DELTA",
    "PHASE_SPLIT_SECONDS",
    "CaptureError",
    "CaptureResult",
    "CaptureSelection",
    "ExpressionMetrics",
    "FrameSample",
    "OPEN_PLATE_NAME",
    "RejectReport",
    "SMILE_PLATE_NAME",
    "SURPRISE_PLATE_NAME",
    "TravelPriors",
    "analyze_frame",
    "build_eye_lid_matte",
    "build_mouth_interior_matte",
    "build_upper_face_matte",
    "compute_travel_priors",
    "default_capture_meta_path",
    "default_eye_anchors_path",
    "default_open_plate_path",
    "default_smile_plate_path",
    "default_surprise_plate_path",
    "default_eyes_closed_plate_path",
    "EYE_ANCHORS_NAME",
    "EYES_CLOSED_PLATE_NAME",
    "write_eye_anchors",
    "iter_video_frames",
    "load_still_as_sample",
    "main",
    "render_capture_qa",
    "run_capture_from_stills",
    "run_capture_from_video",
    "select_expression_frames",
    "validate_selection",
    "write_capture_bundle",
    "write_expression_plate",
]
