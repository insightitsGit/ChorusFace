"""Measured cell/group transition track from the user's upload video.

Stores timed group controls + frame-to-frame **deltas** (the transformation
between second N and N+ε). Per-cell optical flow is intentionally not invented;
groups are the honest unit learned from landmarks.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aiface.audio import DEFAULT_VOICE_THRESHOLD, decode_wav, rms_envelope
from aiface.behavior.schema import (
    CONTROL_DIM,
    CONTROL_NAMES,
    FEATURE_DIM,
    GAP_SECONDS,
    HISTORY,
    BehaviorState,
    dataset_path,
    landmarks_to_controls,
    track_json_path,
    track_npz_path,
)
from aiface.live_vector.extract import _label_video
from aiface.live_vector.features import rms_history_features


@dataclass(slots=True)
class TransitionTrack:
    """In-memory measured track for sampling / training."""

    times: np.ndarray  # (N,)
    controls: np.ndarray  # (N, CONTROL_DIM)
    deltas: np.ndarray  # (N, CONTROL_DIM) — transform from previous sample
    features: np.ndarray  # (N, FEATURE_DIM)
    video: str = ""
    sample_fps: float = 12.0
    noise_floor: float = 0.0
    peak_hint: float = 0.0

    @property
    def n_samples(self) -> int:
        return int(self.times.shape[0])

    @property
    def duration(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return float(self.times[-1] - self.times[0])

    def sample_at(self, t: float) -> BehaviorState | None:
        """Lerp measured controls at time ``t``. None if outside / empty."""
        if self.n_samples == 0:
            return None
        times = self.times
        t = float(t)
        if t < float(times[0]) - 1e-3 or t > float(times[-1]) + 1e-3:
            return None
        idx = int(np.searchsorted(times, t, side="right") - 1)
        idx = max(0, min(idx, self.n_samples - 1))
        if idx >= self.n_samples - 1:
            return _row_to_state(self.controls[-1], self.deltas[-1], "measured")
        t0 = float(times[idx])
        t1 = float(times[idx + 1])
        dt = t1 - t0
        # Gap too large → caller should ML-fill rather than trust long lerp.
        if dt > GAP_SECONDS * 1.75:
            return None
        u = 0.0 if dt <= 1e-9 else (t - t0) / dt
        ctrl = (1.0 - u) * self.controls[idx] + u * self.controls[idx + 1]
        delta = self.deltas[idx + 1] if u > 0.5 else self.deltas[idx]
        source = "measured" if abs(u) < 1e-3 or abs(u - 1.0) < 1e-3 else "measured_lerp"
        return _row_to_state(ctrl, delta, source)

    def gap_at(self, t: float) -> bool:
        """True when ``t`` falls in a hole larger than GAP_SECONDS."""
        if self.n_samples < 2:
            return True
        times = self.times
        t = float(t)
        if t < float(times[0]) or t > float(times[-1]):
            return True
        idx = int(np.searchsorted(times, t, side="right") - 1)
        idx = max(0, min(idx, self.n_samples - 2))
        return float(times[idx + 1] - times[idx]) > GAP_SECONDS * 1.75

    def as_dict(self) -> dict[str, Any]:
        frames = []
        for i in range(self.n_samples):
            row = {"t": float(self.times[i])}
            for name, value in zip(CONTROL_NAMES, self.controls[i], strict=True):
                row[name] = round(float(value), 4)
            row["delta_open"] = round(float(self.deltas[i, 0]), 4)
            row["delta_width"] = round(float(self.deltas[i, 2]), 4)
            frames.append(row)
        return {
            "schema": "aiface.cell_transition_track.v1",
            "video": self.video,
            "sample_fps": self.sample_fps,
            "controls": list(CONTROL_NAMES),
            "n_samples": self.n_samples,
            "duration": self.duration,
            "note": (
                "Measured group transitions from upload. "
                "deltas = transform from previous sample. "
                "ML behavior_model fills gaps / live speech."
            ),
            "frames": frames,
        }


def _row_to_state(
    ctrl: np.ndarray, delta: np.ndarray, source: str
) -> BehaviorState:
    c = [float(x) for x in ctrl.tolist()]
    while len(c) < CONTROL_DIM:
        c.append(0.0)
    return BehaviorState(
        openness_n=c[0],
        jaw_n=c[1],
        width_n=c[2],
        upper_lip_dy=c[3],
        lower_lip_dy=c[4],
        corner_dx=c[5],
        teeth_reveal=c[6],
        cavity_n=c[7],
        source=source,
        delta_open=float(delta[0]) if len(delta) else 0.0,
        delta_width=float(delta[2]) if len(delta) > 2 else 0.0,
    )


def _extract_wav(video: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")


def _normalize(raw: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(raw, 10))
    hi = float(np.percentile(raw, 95))
    return np.clip((raw - lo) / max(hi - lo, 1e-4), 0.0, 1.0)


def extract_transition_track(
    video: Path,
    *,
    world_dir: Path,
    sample_fps: float = 12.0,
    landmarker_model: Path | None = None,
) -> TransitionTrack:
    """Video landmarks → measured group transition track (+ audio features)."""
    video = Path(video).resolve()
    world_dir = Path(world_dir).resolve()
    world_dir.mkdir(parents=True, exist_ok=True)
    if landmarker_model is None:
        candidate = world_dir / "face_landmarker.task"
        landmarker_model = candidate if candidate.is_file() else None

    times, raw_open, raw_width = _label_video(
        video, sample_fps=sample_fps, landmarker_model=landmarker_model
    )
    openness = _normalize(raw_open)
    width = _normalize(raw_width)
    # Teeth proxy from open (capture path can enrich later via talk_series).
    teeth = np.clip(openness * 0.85, 0.0, 1.0)

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        _extract_wav(video, wav)
        clip = decode_wav(wav.read_bytes())
    envelope = rms_envelope(clip)
    noise = float(envelope.noise_floor())
    peak = float(envelope.peak)
    gate = max(DEFAULT_VOICE_THRESHOLD * peak, noise * 1.8)

    controls: list[list[float]] = []
    features: list[np.ndarray] = []
    history: list[float] = []
    for t, open_n, width_n, teeth_n in zip(
        times, openness, width, teeth, strict=True
    ):
        idx = int(t / max(envelope.hop, 1e-6))
        idx = min(max(idx, 0), max(envelope.frame_count - 1, 0))
        rms = float(envelope.values[idx]) if envelope.frame_count else 0.0
        history.append(rms)
        history = history[-HISTORY:]
        feat = rms_history_features(history, noise_floor=noise, peak_hint=peak)
        if rms >= gate:
            ctrl = landmarks_to_controls(
                openness_n=float(open_n),
                width_n=float(width_n),
                teeth_n=float(teeth_n),
            )
        else:
            ctrl = landmarks_to_controls(openness_n=0.0, width_n=0.35, teeth_n=0.0)
            ctrl[0] = ctrl[1] = ctrl[3] = ctrl[4] = ctrl[6] = ctrl[7] = 0.0
        features.append(feat)
        controls.append(ctrl)

    ctrl_arr = np.asarray(controls, dtype=np.float64)
    # Peak-normalize open/jaw like live_vector for stable ML targets.
    for col in (0, 1, 7):
        col_peak = float(ctrl_arr[:, col].max()) if len(ctrl_arr) else 0.0
        if col_peak > 1e-3:
            ctrl_arr[:, col] = np.clip(ctrl_arr[:, col] / col_peak, 0.0, 1.0)
            # Keep lip dy / cavity consistent with normalized open.
            if col == 0:
                ctrl_arr[:, 3] = -ctrl_arr[:, 0]
                ctrl_arr[:, 4] = ctrl_arr[:, 0]
                ctrl_arr[:, 7] = ctrl_arr[:, 0]
                ctrl_arr[:, 6] = np.maximum(ctrl_arr[:, 6], ctrl_arr[:, 0] * 0.65)

    deltas = np.zeros_like(ctrl_arr)
    if len(ctrl_arr) > 1:
        deltas[1:] = ctrl_arr[1:] - ctrl_arr[:-1]

    feat_arr = np.stack(features, axis=0) if features else np.zeros((0, FEATURE_DIM))
    track = TransitionTrack(
        times=np.asarray(times, dtype=np.float64),
        controls=ctrl_arr,
        deltas=deltas,
        features=feat_arr,
        video=str(video),
        sample_fps=float(sample_fps),
        noise_floor=noise,
        peak_hint=peak,
    )
    save_transition_track(track, world_dir)
    return track


def save_transition_track(track: TransitionTrack, world_dir: Path) -> Path:
    world_dir = Path(world_dir)
    world_dir.mkdir(parents=True, exist_ok=True)
    npz = track_npz_path(world_dir)
    np.savez_compressed(
        npz,
        times=track.times,
        controls=track.controls,
        deltas=track.deltas,
        features=track.features,
        noise_floor=np.asarray([track.noise_floor]),
        peak_hint=np.asarray([track.peak_hint]),
        sample_fps=np.asarray([track.sample_fps]),
        video=np.asarray([track.video]),
        control_names=np.asarray(list(CONTROL_NAMES)),
    )
    # Train-ready dataset (audio → group controls).
    ds = dataset_path(world_dir)
    np.savez_compressed(
        ds,
        X=track.features,
        y=track.controls,
        deltas=track.deltas,
        times=track.times,
        noise_floor=np.asarray([track.noise_floor]),
        peak_hint=np.asarray([track.peak_hint]),
        control_names=np.asarray(list(CONTROL_NAMES)),
    )
    meta = track_json_path(world_dir)
    meta.write_text(json.dumps(track.as_dict(), indent=2), encoding="utf-8")
    print(f"behavior-track: wrote {npz} ({track.n_samples} samples, {track.duration:.2f}s)")
    print(f"behavior-track: wrote {meta}")
    return npz


def load_transition_track(world: Path | str) -> TransitionTrack | None:
    path = track_npz_path(world)
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        times = np.asarray(data["times"], dtype=np.float64)
        controls = np.asarray(data["controls"], dtype=np.float64)
        deltas = np.asarray(data["deltas"], dtype=np.float64)
        features = np.asarray(data["features"], dtype=np.float64)
        video = str(np.asarray(data["video"]).reshape(-1)[0]) if "video" in data else ""
        fps = float(np.asarray(data["sample_fps"]).reshape(-1)[0]) if "sample_fps" in data else 12.0
        noise = float(np.asarray(data["noise_floor"]).reshape(-1)[0])
        peak = float(np.asarray(data["peak_hint"]).reshape(-1)[0])
    except (OSError, ValueError, KeyError) as exc:
        print(f"behavior-track: load failed ({exc})")
        return None
    if controls.ndim != 2 or controls.shape[1] < CONTROL_DIM:
        # Pad older / short vectors.
        padded = np.zeros((len(controls), CONTROL_DIM), dtype=np.float64)
        if controls.ndim == 2 and len(controls):
            cols = min(CONTROL_DIM, controls.shape[1])
            padded[:, :cols] = controls[:, :cols]
        controls = padded
    return TransitionTrack(
        times=times,
        controls=controls,
        deltas=deltas if deltas.shape == controls.shape else np.zeros_like(controls),
        features=features,
        video=video,
        sample_fps=fps,
        noise_floor=noise,
        peak_hint=peak,
    )


__all__ = [
    "TransitionTrack",
    "extract_transition_track",
    "load_transition_track",
    "save_transition_track",
]
