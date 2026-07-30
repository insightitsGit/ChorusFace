"""Face landmarks and canonical UV for Path 1 portrait seeding.

The muscle definition, tissue bake, and synthetic portrait all share one layout:
eyes near ``v=0.472``, mouth at ``v=0.78`` in face-box UV (image y down). This
module measures those points on a real photo when it can, and falls back to the
canonical fractions when it cannot — never silently inventing a second layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from aiface.seed import AvatarSeedError, FaceBox, _cv2, _fallback_face_box, detect_face

#: Single source of truth for the authored frontal layout.
CANONICAL_LEFT_EYE_UV: Final = (0.30, 0.472)
CANONICAL_RIGHT_EYE_UV: Final = (0.70, 0.472)
CANONICAL_LEFT_BROW_UV: Final = (0.30, 0.395)
CANONICAL_RIGHT_BROW_UV: Final = (0.70, 0.395)
CANONICAL_MOUTH_UV: Final = (0.50, 0.78)
#: How much of the Haar box to expand before the square crop.
NORMALIZE_PAD: Final = 0.35
#: MediaPipe Face Mesh indices (approximate iris / lip centres).
_MP_LEFT_IRIS: Final = 468
_MP_RIGHT_IRIS: Final = 473
_MP_MOUTH: Final = 13
_MP_LEFT_BROW: Final = 70
_MP_RIGHT_BROW: Final = 300


@dataclass(frozen=True, slots=True)
class FaceLandmarks:
    """Measured feature centres in image pixel coordinates (y down)."""

    face: FaceBox
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    mouth: tuple[float, float]
    left_brow: tuple[float, float]
    right_brow: tuple[float, float]
    method: str
    quality: float

    def as_uv(self, point: tuple[float, float]) -> tuple[float, float]:
        fw = max(float(self.face.width), 1.0)
        fh = max(float(self.face.height), 1.0)
        return (
            (point[0] - float(self.face.x)) / fw,
            (point[1] - float(self.face.y)) / fh,
        )

    def eye_uv(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.as_uv(self.left_eye), self.as_uv(self.right_eye)

    def mouth_uv(self) -> tuple[float, float]:
        return self.as_uv(self.mouth)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "quality": round(float(self.quality), 4),
            "left_eye_image": {"x": self.left_eye[0], "y": self.left_eye[1]},
            "right_eye_image": {"x": self.right_eye[0], "y": self.right_eye[1]},
            "mouth_center_image": {"x": self.mouth[0], "y": self.mouth[1]},
            "left_brow_image": {"x": self.left_brow[0], "y": self.left_brow[1]},
            "right_brow_image": {"x": self.right_brow[0], "y": self.right_brow[1]},
            "eye_uv": {
                "left": list(self.as_uv(self.left_eye)),
                "right": list(self.as_uv(self.right_eye)),
            },
            "mouth_uv": list(self.mouth_uv()),
        }


def point_from_uv(face: FaceBox, uv: tuple[float, float]) -> tuple[float, float]:
    return (
        float(face.x) + float(uv[0]) * float(face.width),
        float(face.y) + float(uv[1]) * float(face.height),
    )


def canonical_landmarks(face: FaceBox, *, method: str = "canonical") -> FaceLandmarks:
    """Landmarks at the authored UV layout — used for synthetic and as fallback."""
    return FaceLandmarks(
        face=face,
        left_eye=point_from_uv(face, CANONICAL_LEFT_EYE_UV),
        right_eye=point_from_uv(face, CANONICAL_RIGHT_EYE_UV),
        mouth=point_from_uv(face, CANONICAL_MOUTH_UV),
        left_brow=point_from_uv(face, CANONICAL_LEFT_BROW_UV),
        right_brow=point_from_uv(face, CANONICAL_RIGHT_BROW_UV),
        method=method,
        quality=1.0 if method.startswith("canonical") else 0.55,
    )


def _square_crop_box(face: FaceBox, width: int, height: int) -> FaceBox:
    """Expand the face into a padded square that still fits the image."""
    cx = face.x + face.width * 0.5
    cy = face.y + face.height * 0.42
    side = max(face.width, face.height) * (1.0 + NORMALIZE_PAD)
    side = min(side, float(min(width, height)))
    x0 = int(round(cx - side * 0.5))
    y0 = int(round(cy - side * 0.5))
    x0 = max(0, min(x0, width - int(side)))
    y0 = max(0, min(y0, height - int(side)))
    size = int(round(side))
    size = max(8, min(size, width - x0, height - y0))
    return FaceBox(x0, y0, size, size)


def normalize_face_image(
    image_bgr: npt.NDArray[np.uint8],
    *,
    width: int,
    height: int,
    face: FaceBox | None = None,
) -> tuple[npt.NDArray[np.uint8], FaceBox]:
    """Crop/pad the face into a square, then resize to the seed grid.

    Stretching a whole non-square frame into 256×256 warps the UV layout the
    muscle definition assumes. A face-centred square crop keeps proportions.
    """
    cv2 = _cv2()
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise AvatarSeedError("Face image must have shape (height, width, 3)")
    src_h, src_w = image_bgr.shape[:2]
    detected = face if face is not None else detect_face(image_bgr)
    crop = _square_crop_box(detected, src_w, src_h)
    patch = image_bgr[crop.y : crop.y1, crop.x : crop.x1]
    if patch.size == 0:
        raise AvatarSeedError("Face crop was empty")
    resized = cv2.resize(patch, (width, height), interpolation=cv2.INTER_AREA)
    # After normalize the face fills most of the canvas; use the authored box.
    return resized, _fallback_face_box(width, height)


def _mediapipe_landmarks(
    image_bgr: npt.NDArray[np.uint8], face: FaceBox
) -> FaceLandmarks | None:
    try:
        import mediapipe as mp

        face_mesh = mp.solutions.face_mesh
    except (ImportError, AttributeError):
        # Missing package, or a MediaPipe build without the classic solutions API.
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

    left_eye, right_eye = px(_MP_LEFT_IRIS), px(_MP_RIGHT_IRIS)
    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye
    # Quality: eyes should sit inside the face box and be horizontally separated.
    span = abs(right_eye[0] - left_eye[0]) / max(float(face.width), 1.0)
    quality = float(np.clip(span / 0.35, 0.0, 1.0))
    if quality < 0.35:
        return None
    return FaceLandmarks(
        face=face,
        left_eye=left_eye,
        right_eye=right_eye,
        mouth=px(_MP_MOUTH),
        left_brow=px(_MP_LEFT_BROW),
        right_brow=px(_MP_RIGHT_BROW),
        method="mediapipe",
        quality=quality,
    )


def _opencv_eye_landmarks(
    image_bgr: npt.NDArray[np.uint8], face: FaceBox
) -> FaceLandmarks | None:
    """Haar eye pairs inside the upper face — better than pure fractions."""
    cv2 = _cv2()
    cascade = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
    if not cascade.is_file():
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi_y0 = face.y
    roi_y1 = face.y + int(face.height * 0.62)
    roi_x0, roi_x1 = face.x, face.x1
    roi = gray[roi_y0:roi_y1, roi_x0:roi_x1]
    if roi.size == 0:
        return None
    detector = cv2.CascadeClassifier(str(cascade))
    if detector.empty():
        return None
    hits = detector.detectMultiScale(
        roi, scaleFactor=1.1, minNeighbors=6, minSize=(12, 12)
    )
    if len(hits) < 2:
        return None
    centres = sorted(
        [
            (
                float(roi_x0 + x + w * 0.5),
                float(roi_y0 + y + h * 0.5),
            )
            for x, y, w, h in hits
        ],
        key=lambda point: point[0],
    )
    left_eye, right_eye = centres[0], centres[-1]
    span = abs(right_eye[0] - left_eye[0]) / max(float(face.width), 1.0)
    if span < 0.18:
        return None
    base = canonical_landmarks(face, method="opencv-eyes")
    return FaceLandmarks(
        face=face,
        left_eye=left_eye,
        right_eye=right_eye,
        mouth=base.mouth,
        left_brow=(left_eye[0], left_eye[1] - face.height * 0.08),
        right_brow=(right_eye[0], right_eye[1] - face.height * 0.08),
        method="opencv-eyes",
        quality=float(np.clip(span / 0.40, 0.4, 0.9)),
    )


def _pupil_eye_landmarks(
    image_bgr: npt.NDArray[np.uint8], face: FaceBox
) -> FaceLandmarks | None:
    """Dark-pupil search when Haar cascades are missing (OpenCV 5+)."""
    cv2 = _cv2()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    fw = max(float(face.width), 1.0)
    fh = max(float(face.height), 1.0)

    def darkest_in(u0: float, u1: float, v0: float, v1: float) -> tuple[float, float] | None:
        x0 = int(face.x + u0 * fw)
        x1 = int(face.x + u1 * fw)
        y0 = int(face.y + v0 * fh)
        y1 = int(face.y + v1 * fh)
        x0, x1 = max(0, x0), min(gray.shape[1], x1)
        y0, y1 = max(0, y0), min(gray.shape[0], y1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None
        patch = blur[y0:y1, x0:x1]
        # Ignore flat cheek: need a real dark spot.
        if float(patch.std()) < 6.0:
            return None
        flat = patch.reshape(-1)
        idx = int(np.argmin(flat))
        py, px = divmod(idx, patch.shape[1])
        return (float(x0 + px), float(y0 + py))

    # Eyes sit in the upper face; search left/right bands around the iris UV.
    left = darkest_in(0.18, 0.42, 0.36, 0.56)
    right = darkest_in(0.58, 0.82, 0.36, 0.56)
    if left is None or right is None:
        return None
    if left[0] > right[0]:
        left, right = right, left
    span = abs(right[0] - left[0]) / fw
    if span < 0.20 or span > 0.62:
        return None
    # Reject if pupils landed on the same vertical cheek band (too low/high).
    mid_v = 0.5 * (left[1] + right[1])
    expected = float(face.y) + CANONICAL_LEFT_EYE_UV[1] * fh
    if abs(mid_v - expected) > fh * 0.14:
        return None
    base = canonical_landmarks(face, method="pupil-eyes")
    quality = float(np.clip(span / 0.40, 0.45, 0.92))
    return FaceLandmarks(
        face=face,
        left_eye=left,
        right_eye=right,
        mouth=base.mouth,
        left_brow=(left[0], left[1] - fh * 0.08),
        right_brow=(right[0], right[1] - fh * 0.08),
        method="pupil-eyes",
        quality=quality,
    )


def measure_landmarks(
    image_bgr: npt.NDArray[np.uint8],
    *,
    face: FaceBox | None = None,
    synthetic: bool = False,
) -> FaceLandmarks:
    """Best available landmarks for this image."""
    box = face if face is not None else detect_face(image_bgr)
    if synthetic:
        return canonical_landmarks(box, method="canonical-synthetic")
    measured = _mediapipe_landmarks(image_bgr, box)
    if measured is not None:
        return measured
    measured = _opencv_eye_landmarks(image_bgr, box)
    if measured is not None:
        return measured
    measured = _pupil_eye_landmarks(image_bgr, box)
    if measured is not None:
        return measured
    return canonical_landmarks(box, method="canonical-fallback")


def eye_aperture_score(
    image_bgr: npt.NDArray[np.uint8], landmarks: FaceLandmarks
) -> float:
    """How well dark pupil pixels sit under the measured eye centres.

    Used as a seed QA signal: a wrong UV puts the aperture on cheek flesh and
    the score collapses.
    """
    cv2 = _cv2()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    scores: list[float] = []
    radius = max(2, int(landmarks.face.width * 0.04))
    for cx, cy in (landmarks.left_eye, landmarks.right_eye):
        x0 = max(0, int(cx) - radius)
        x1 = min(width, int(cx) + radius + 1)
        y0 = max(0, int(cy) - radius)
        y1 = min(height, int(cy) + radius + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            scores.append(0.0)
            continue
        # Pupils are darker than local cheek; high contrast → good registration.
        local = float(patch.mean())
        centre = float(
            gray[
                min(height - 1, max(0, int(cy))),
                min(width - 1, max(0, int(cx))),
            ]
        )
        scores.append(float(np.clip((local - centre) / 40.0, 0.0, 1.0)))
    return float(np.mean(scores)) if scores else 0.0


def render_seed_qa(
    image_bgr: npt.NDArray[np.uint8], landmarks: FaceLandmarks
) -> npt.NDArray[np.uint8]:
    """Overlay face box, eyes, brows, and mouth for human seed inspection."""
    cv2 = _cv2()
    canvas = image_bgr.copy()
    face = landmarks.face
    cv2.rectangle(
        canvas,
        (face.x, face.y),
        (face.x1, face.y1),
        (40, 220, 80),
        1,
        cv2.LINE_AA,
    )
    for point, color in (
        (landmarks.left_eye, (0, 200, 255)),
        (landmarks.right_eye, (0, 200, 255)),
        (landmarks.mouth, (0, 80, 255)),
        (landmarks.left_brow, (200, 160, 40)),
        (landmarks.right_brow, (200, 160, 40)),
    ):
        centre = (int(round(point[0])), int(round(point[1])))
        cv2.circle(canvas, centre, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, centre, 8, color, 1, cv2.LINE_AA)
    label = f"{landmarks.method} q={landmarks.quality:.2f}"
    cv2.putText(
        canvas,
        label,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas


__all__ = [
    "CANONICAL_LEFT_BROW_UV",
    "CANONICAL_LEFT_EYE_UV",
    "CANONICAL_MOUTH_UV",
    "CANONICAL_RIGHT_BROW_UV",
    "CANONICAL_RIGHT_EYE_UV",
    "FaceLandmarks",
    "canonical_landmarks",
    "eye_aperture_score",
    "measure_landmarks",
    "normalize_face_image",
    "point_from_uv",
    "render_seed_qa",
]
