"""Measure per-frame lid aperture (EAR → lid_amt) for LOOK teacher.

``lid_amt``: 1 = open, 0 = closed — same polarity as mouth ``open_amt``.
Uses MediaPipe Tasks Face Landmarker when ``face_landmarker.task`` is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Face Mesh topology (shared by classic + tasks).
_LEFT_EYE_UPPER = 159
_LEFT_EYE_LOWER = 145
_RIGHT_EYE_UPPER = 386
_RIGHT_EYE_LOWER = 374
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_OUTER = 263
_RIGHT_EYE_INNER = 362


def _ear_from_landmarks(pts: list) -> float:
    def dist(a: int, b: int) -> float:
        return float(np.hypot(pts[a].x - pts[b].x, pts[a].y - pts[b].y))

    left = dist(_LEFT_EYE_UPPER, _LEFT_EYE_LOWER) / max(
        dist(_LEFT_EYE_OUTER, _LEFT_EYE_INNER), 1e-6
    )
    right = dist(_RIGHT_EYE_UPPER, _RIGHT_EYE_LOWER) / max(
        dist(_RIGHT_EYE_OUTER, _RIGHT_EYE_INNER), 1e-6
    )
    return 0.5 * (left + right)


def resolve_landmarker_model(*candidates: Path) -> Path | None:
    for path in candidates:
        if path is not None and path.is_file():
            return path
    return None


def measure_lid_series(
    video: Path,
    *,
    model: Path | None = None,
) -> tuple[list[float], list[float]] | None:
    """Return ``(times_sec, lid_amt)`` or None if landmarker unavailable.

    ``lid_amt`` is normalized so the take's open baseline ≈ 1.0 and the most
    closed frames approach 0.0 (percentile-based, robust to makeup).
    """
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except Exception:
        return None

    root = Path(__file__).resolve().parents[3]
    candidates = [
        Path(video).resolve().parent / "face_landmarker.task",
        root / "models" / "face_landmarker.task",
        root / "output" / "models" / "face_landmarker.task",
    ]
    if model is not None:
        candidates.insert(0, Path(model))
    model_path = resolve_landmarker_model(*candidates)
    if model_path is None:
        print("TickFeed lid: no face_landmarker.task — lid_amt stays script/default")
        return None

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    times: list[float] = []
    ears: list[float] = []
    index = 0
    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(
                    image, int(round(index * 1000.0 / max(fps, 1e-6)))
                )
                if result.face_landmarks:
                    ears.append(_ear_from_landmarks(result.face_landmarks[0]))
                    times.append(index / max(fps, 1e-6))
                index += 1
    except Exception as exc:  # noqa: BLE001
        print(f"TickFeed lid: landmarker failed ({exc})")
        cap.release()
        return None
    cap.release()
    if len(ears) < 8:
        return None

    arr = np.asarray(ears, dtype=np.float64)
    # Open baseline = high percentile; closed floor = low percentile.
    open_ref = float(np.percentile(arr, 90))
    closed_ref = float(np.percentile(arr, 5))
    span = max(open_ref - closed_ref, 1e-4)
    lid = np.clip((arr - closed_ref) / span, 0.0, 1.0)
    # Light temporal smooth (3-tap) — keeps blink dips.
    if lid.size >= 3:
        pad = np.pad(lid, (1, 1), mode="edge")
        lid = 0.25 * pad[:-2] + 0.50 * pad[1:-1] + 0.25 * pad[2:]
    print(
        f"TickFeed lid: measured n={len(lid)} ear "
        f"min={arr.min():.4f} max={arr.max():.4f} -> "
        f"lid min={float(lid.min()):.3f} max={float(lid.max()):.3f} "
        f"model={model_path.name}"
    )
    return times, [float(x) for x in lid]


__all__ = ["measure_lid_series", "resolve_landmarker_model"]
