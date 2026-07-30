"""Extract live control trajectories from a frontal capture video."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aiface.audio import DEFAULT_VOICE_THRESHOLD, decode_wav, rms_envelope
from aiface.live_vector.features import rms_history_features
from aiface.live_vector.schema import (
    CONTROL_NAMES,
    DATASET_CSV_NAME,
    HISTORY,
    dataset_path,
    trajectory_path,
)

_MP_UPPER_LIP = 13
_MP_LOWER_LIP = 14
_MP_MOUTH_LEFT = 61
_MP_MOUTH_RIGHT = 291


@dataclass(frozen=True, slots=True)
class ExtractResult:
    dataset: Path
    trajectory: Path
    csv_path: Path
    n_samples: int
    noise_floor: float
    peak_hint: float


def _extract_wav(video: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")


def _label_video(
    video: Path, *, sample_fps: float, landmarker_model: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times, raw mouth_open, raw smile_width."""
    if landmarker_model is not None and landmarker_model.is_file():
        labeled = _label_tasks(video, sample_fps=sample_fps, model=landmarker_model)
        if labeled is not None:
            return labeled
    from aiface.capture import MIN_SHARPNESS_SOFT, iter_video_frames

    frames = iter_video_frames(
        video, sample_fps=sample_fps, min_sharpness=MIN_SHARPNESS_SOFT
    )
    if not frames:
        raise RuntimeError("no usable frames for live-vector extract")
    times = np.asarray([f.time_seconds for f in frames], dtype=np.float64)
    opens = np.asarray([f.metrics.mouth_open for f in frames], dtype=np.float64)
    widths = np.asarray([f.metrics.smile_width for f in frames], dtype=np.float64)
    print(
        f"extract: capture-fallback  open "
        f"min={opens.min():.4f} max={opens.max():.4f} n={len(times)}"
    )
    return times, opens, widths


def _label_tasks(
    video: Path, *, sample_fps: float, model: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except Exception:
        return None

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    step = max(int(round(src_fps / max(sample_fps, 1.0))), 1)
    times: list[float] = []
    opens: list[float] = []
    widths: list[float] = []
    index = 0
    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step != 0:
                    index += 1
                    continue
                t = index / max(src_fps, 1e-6)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, int(t * 1000.0))
                mouth_open = 0.0
                smile_width = 0.35
                if result.face_landmarks:
                    pts = result.face_landmarks[0]
                    ys = [p.y for p in pts]
                    xs = [p.x for p in pts]
                    fh = max(max(ys) - min(ys), 1e-6)
                    fw = max(max(xs) - min(xs), 1e-6)
                    mouth_open = abs(pts[_MP_LOWER_LIP].y - pts[_MP_UPPER_LIP].y) / fh
                    smile_width = abs(
                        pts[_MP_MOUTH_RIGHT].x - pts[_MP_MOUTH_LEFT].x
                    ) / fw
                times.append(t)
                opens.append(float(mouth_open))
                widths.append(float(smile_width))
                index += 1
    except Exception as exc:
        print(f"extract: tasks failed ({exc})")
        cap.release()
        return None
    cap.release()
    if not times:
        return None
    print(
        f"extract: mediapipe-tasks  open "
        f"min={min(opens):.4f} max={max(opens):.4f} n={len(times)}"
    )
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(opens, dtype=np.float64),
        np.asarray(widths, dtype=np.float64),
    )


def _normalize(raw: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(raw, 10))
    hi = float(np.percentile(raw, 95))
    return np.clip((raw - lo) / max(hi - lo, 1e-4), 0.0, 1.0)


def extract_live_vectors(
    video: Path,
    *,
    world_dir: Path,
    sample_fps: float = 12.0,
    landmarker_model: Path | None = None,
) -> ExtractResult:
    """Video → dataset + trajectory JSON (live vectors only)."""
    world_dir.mkdir(parents=True, exist_ok=True)
    if landmarker_model is None:
        candidate = world_dir / "face_landmarker.task"
        landmarker_model = candidate if candidate.is_file() else None

    times, raw_open, raw_width = _label_video(
        video, sample_fps=sample_fps, landmarker_model=landmarker_model
    )
    openness = _normalize(raw_open)
    width = _normalize(raw_width)

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        _extract_wav(video, wav)
        clip = decode_wav(wav.read_bytes())
    envelope = rms_envelope(clip)
    noise = float(envelope.noise_floor())
    peak = float(envelope.peak)
    gate = max(DEFAULT_VOICE_THRESHOLD * peak, noise * 1.8)

    features: list[np.ndarray] = []
    labels: list[list[float]] = []
    rows: list[dict[str, float]] = []
    history: list[float] = []
    for t, open_n, width_n in zip(times, openness, width, strict=True):
        idx = int(t / max(envelope.hop, 1e-6))
        idx = min(max(idx, 0), max(envelope.frame_count - 1, 0))
        rms = float(envelope.values[idx]) if envelope.frame_count else 0.0
        history.append(rms)
        history = history[-HISTORY:]
        feat = rms_history_features(history, noise_floor=noise, peak_hint=peak)
        if rms >= gate:
            teach = [float(open_n), float(open_n), float(width_n)]
        else:
            teach = [0.0, 0.0, 0.0]
        features.append(feat)
        labels.append(teach)
        rows.append(
            {
                "t": float(t),
                "openness_n": teach[0],
                "jaw_n": teach[1],
                "width_n": teach[2],
                "video_openness": float(open_n),
                "rms": rms,
            }
        )

    x = np.stack(features, axis=0)
    y = np.asarray(labels, dtype=np.float64)
    for col in (0, 1):
        col_peak = float(y[:, col].max()) if len(y) else 0.0
        if col_peak > 1e-3:
            y[:, col] = np.clip(y[:, col] / col_peak, 0.0, 1.0)
    for row, vec in zip(rows, y, strict=True):
        row["openness_n"] = float(vec[0])
        row["jaw_n"] = float(vec[1])
        row["width_n"] = float(vec[2])

    npz = dataset_path(world_dir)
    np.savez_compressed(
        npz,
        X=x,
        y=y,
        times=times,
        noise_floor=np.asarray([noise]),
        peak_hint=np.asarray([peak]),
        video=np.asarray([str(video)]),
        control_names=np.asarray(list(CONTROL_NAMES)),
    )
    csv_path = world_dir / DATASET_CSV_NAME
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    traj = trajectory_path(world_dir)
    traj.write_text(
        json.dumps(
            {
                "version": "live-vector-1.0",
                "video": str(video),
                "controls": list(CONTROL_NAMES),
                "note": "From-scratch extract: video truth → live vectors",
                "frames": [
                    {
                        "t": r["t"],
                        "openness_n": r["openness_n"],
                        "jaw_n": r["jaw_n"],
                        "width_n": r["width_n"],
                        "rms": r["rms"],
                    }
                    for r in rows
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"extract: wrote {npz} ({len(y)} samples)")
    print(f"extract: wrote {traj}")
    return ExtractResult(
        dataset=npz,
        trajectory=traj,
        csv_path=csv_path,
        n_samples=len(y),
        noise_floor=noise,
        peak_hint=peak,
    )
