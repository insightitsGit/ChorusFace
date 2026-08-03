"""Bake photographed eyes_closed.png from the calibration take (LOOK region).

Picks the frame with the smallest lid aperture (~95% closed is fine),
color-matches to ``source_face.png``, and writes a dual-ellipse eye matte from
Face Landmarker sockets (not ``face_tissue.npy`` channel A — that channel can
sit on cheek/nose on some takes).

Also writes ``eye_anchors.json`` so runtime L09 centers match the plate.

If the take has no real blink evidence, exits without writing a plate so L09
holds the open photo (never invent lids / skin disks). Never bake an open-eye
frame as ``eyes_closed.png``.

Usage:
  python scripts/bake_eyes_closed_plate.py
  python scripts/bake_eyes_closed_plate.py --world output/worlds/tickfeed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# MediaPipe Face Mesh indices (tasks + classic share topology).
_LEFT_EYE_UPPER = 159
_LEFT_EYE_LOWER = 145
_RIGHT_EYE_UPPER = 386
_RIGHT_EYE_LOWER = 374
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_OUTER = 263
_RIGHT_EYE_INNER = 362
_LEFT_EYE_RING = (
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
)
_RIGHT_EYE_RING = (
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
)


def _ear_from_landmarks(pts: list) -> float:
    """Eye aspect ratio from Face Mesh points (lower → more closed)."""

    def dist(a: int, b: int) -> float:
        return float(np.hypot(pts[a].x - pts[b].x, pts[a].y - pts[b].y))

    left = dist(_LEFT_EYE_UPPER, _LEFT_EYE_LOWER) / max(
        dist(_LEFT_EYE_OUTER, _LEFT_EYE_INNER), 1e-6
    )
    right = dist(_RIGHT_EYE_UPPER, _RIGHT_EYE_LOWER) / max(
        dist(_RIGHT_EYE_OUTER, _RIGHT_EYE_INNER), 1e-6
    )
    return 0.5 * (left + right)


def _socket_from_ring(
    pts: list, ring: tuple[int, ...], width: int, height: int
) -> tuple[float, float, float, float]:
    xs = np.asarray([pts[i].x * width for i in ring], dtype=np.float64)
    ys = np.asarray([pts[i].y * height for i in ring], dtype=np.float64)
    cx = float(xs.mean())
    cy = float(ys.mean())
    # Cover closed lids + lashes; keep sockets on the eyes (not cheek/nose).
    half_w = max(float(0.52 * (xs.max() - xs.min())), 8.0) * 1.28
    half_h = max(float(0.58 * (ys.max() - ys.min())), 5.0) * 1.45
    return cx, cy, half_w, half_h


def _score_with_face_landmarker(
    video: Path, model: Path
) -> list[tuple[float, int, np.ndarray, object]] | None:
    """Return (ear, index, bgr, landmarks) per frame, or None if unavailable."""
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except Exception:
        return None
    if not model.is_file():
        return None

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    scored: list[tuple[float, int, np.ndarray, object]] = []
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
                    image, int(index * 1000.0 / max(fps, 1e-6))
                )
                if result.face_landmarks:
                    pts = result.face_landmarks[0]
                    ear = _ear_from_landmarks(pts)
                    scored.append((float(ear), index, frame.copy(), pts))
                index += 1
    except Exception as exc:
        print(f"face_landmarker failed: {exc}")
        cap.release()
        return None
    cap.release()
    return scored or None


def _landmarks_on_image(
    image_bgr: np.ndarray, model: Path
) -> object | None:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except Exception:
        return None
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def _has_blink_evidence(
    scores: list[float], *, relative_max: float, abs_max: float
) -> tuple[bool, float, float]:
    """True when the best (lowest) score is clearly closed vs the take median."""
    arr = np.asarray(scores, dtype=np.float64)
    best = float(arr.min())
    median = float(np.median(arr))
    ok = best <= abs_max or (median > 1e-6 and best <= relative_max * median)
    return ok, best, median


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world",
        type=Path,
        default=ROOT / "output" / "worlds" / "tickfeed",
    )
    parser.add_argument(
        "--relative-max",
        type=float,
        default=0.55,
        help="Accept if best_score <= this × median (blink vs open rest)",
    )
    parser.add_argument(
        "--abs-max",
        type=float,
        default=0.12,
        help="Accept if best EAR <= this (near-closed absolute)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write plate even without blink evidence (not recommended)",
    )
    args = parser.parse_args()
    world = args.world
    video = world / "calibration_take.mp4"
    source = world / "source_face.png"
    if not video.is_file():
        raise SystemExit(f"missing {video}")
    if not source.is_file():
        raise SystemExit(f"missing {source}")

    import cv2
    from aiface.capture import (
        DISPLAY_SIZE,
        analyze_frame,
        build_eye_lid_matte,
        default_eye_anchors_path,
        default_eyes_closed_plate_path,
        match_plate_to_reference,
        write_expression_plate,
        write_eye_anchors,
    )

    ref_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise SystemExit(f"could not read {source}")
    h, w = ref_bgr.shape[:2]
    out = default_eyes_closed_plate_path(world / "avatar_face.bds")
    anchors_path = default_eye_anchors_path(world)

    model_candidates = [
        world / "face_landmarker.task",
        ROOT / "models" / "face_landmarker.task",
        ROOT / "output" / "models" / "face_landmarker.task",
    ]
    model = next((p for p in model_candidates if p.is_file()), None)
    if model is None:
        raise SystemExit(
            "face_landmarker.task required to bake eyes_closed.png "
            "(tissue.a is not a reliable eye matte on this pipeline)"
        )

    scored = _score_with_face_landmarker(video, model)
    if scored is None:
        raise SystemExit("face_landmarker could not score the calibration take")

    scores = [s[0] for s in scored]
    ok, best, median = _has_blink_evidence(
        scores, relative_max=float(args.relative_max), abs_max=float(args.abs_max)
    )
    best_idx = int(np.argmin(np.asarray(scores)))
    lid, index, plate_bgr, pts = scored[best_idx]
    print(
        f"best blink candidate: frame={index} score={lid:.4f} "
        f"median={median:.4f} method=face_landmarker EAR"
    )
    if not ok and not args.force:
        if out.is_file():
            out.unlink()
            print(f"removed stale {out.name} (was open-eye / no blink evidence)")
        if anchors_path.is_file():
            anchors_path.unlink()
        print(
            "no photographed blink evidence in this take — "
            "L09 will hold the open photo (no invented lids). "
            "Re-capture with a BLINK beat, or pass --force to bake anyway."
        )
        return 0

    if not ok:
        print("warning: --force baking without blink evidence")

    # CRITICAL: same face-normalize crop as source_face / open.png.
    # Naive resize of the 1280x720 take onto 1024² misregisters lids → UV junk.
    sample = analyze_frame(
        plate_bgr,
        index=int(index),
        time_seconds=float(index) / 24.0,
        width=int(DISPLAY_SIZE),
        height=int(DISPLAY_SIZE),
        min_sharpness=0.0,
        allow_closed_eyes=True,
    )
    if sample is None:
        raise SystemExit(
            f"could not normalize blink frame {index} into source_face UV "
            "(detect/crop failed)"
        )
    plate_bgr = sample.image_bgr
    if plate_bgr.shape[:2] != (h, w):
        plate_bgr = cv2.resize(plate_bgr, (w, h), interpolation=cv2.INTER_AREA)

    # Prefer open-rest sockets from source_face so plate ownership matches
    # identity UV (blink frame can shift slightly).
    source_pts = _landmarks_on_image(ref_bgr, model)
    pts = source_pts or _landmarks_on_image(plate_bgr, model)
    if pts is None:
        # Fall back to capture landmark meta on the normalized blink sample.
        left = sample.landmarks_meta.get("left_eye_image") or {}
        right = sample.landmarks_meta.get("right_eye_image") or {}
        try:
            lcx = float(left["x"])
            lcy = float(left["y"])
            rcx = float(right["x"])
            rcy = float(right["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"no eye sockets for blink plate ({exc})") from exc
        half_w = max(float(sample.face.width) * 0.11, 18.0)
        half_h = max(float(sample.face.height) * 0.055, 10.0)
        print(
            f"eye sockets from capture landmarks: "
            f"L=({lcx:.1f},{lcy:.1f}) R=({rcx:.1f},{rcy:.1f}) "
            f"half=({half_w:.1f},{half_h:.1f})"
        )
    else:
        lcx, lcy, lhw, lhh = _socket_from_ring(pts, _LEFT_EYE_RING, w, h)
        rcx, rcy, rhw, rhh = _socket_from_ring(pts, _RIGHT_EYE_RING, w, h)
        half_w = 0.5 * (lhw + rhw)
        half_h = 0.5 * (lhh + rhh)
        origin = "source_face" if source_pts is not None else "blink frame"
        print(
            f"eye sockets from {origin}: "
            f"L=({lcx:.1f},{lcy:.1f}) R=({rcx:.1f},{rcy:.1f}) "
            f"half=({half_w:.1f},{half_h:.1f})"
        )

    matte = build_eye_lid_matte(
        h,
        w,
        (lcx, lcy),
        (rcx, rcy),
        half_width=half_w,
        half_height=half_h,
    )
    matched = match_plate_to_reference(plate_bgr, ref_bgr, matte)
    write_expression_plate(out, matched, matte)
    write_eye_anchors(
        anchors_path,
        left_eye=(lcx, lcy),
        right_eye=(rcx, rcy),
        half_width=half_w,
        half_height=half_h,
        image_size=(h, w),
        source_frame=int(index),
        method="face_landmarker",
    )
    qa = Path("output/previews/blink_qa")
    qa.mkdir(parents=True, exist_ok=True)
    rgba = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    if rgba is not None and rgba.shape[-1] == 4:
        rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        vis = (rgb.astype(np.float32) * a + ref_bgr[:, :, ::-1].astype(np.float32) * (1.0 - a))
        cv2.imwrite(
            str(qa / "eyes_closed_plate_vis.png"),
            cv2.cvtColor(np.clip(vis, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        overlay = ref_bgr.copy()
        mask = matte > 0.25
        overlay[mask] = (
            overlay[mask].astype(np.float32) * 0.35
            + np.array([0.0, 0.0, 255.0], dtype=np.float32) * 0.65
        ).astype(np.uint8)
        cv2.imwrite(str(qa / "eyes_closed_matte_on_source.png"), overlay)
    print(f"wrote {out} + {anchors_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
