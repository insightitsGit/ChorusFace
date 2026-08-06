"""D35 teacher extract + go/no-go metrics (F12)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chorusface.vowel.schema import TICK_HZ


@dataclass(slots=True)
class D35Metrics:
    jitter_face_width_pct: float
    hold_stability_face_width_pct: float
    ee_ou_distance_face_width_pct: float
    velocity_continuity_ok: bool
    symmetry_ok: bool
    lip_closure_ok: bool
    passed: bool
    hard_fail: bool = False
    notes: list[str] = field(default_factory=list)
    mode: str = "vowel"

    def to_dict(self) -> dict[str, Any]:
        return {
            "jitter_face_width_pct": self.jitter_face_width_pct,
            "hold_stability_face_width_pct": self.hold_stability_face_width_pct,
            "ee_ou_distance_face_width_pct": self.ee_ou_distance_face_width_pct,
            "velocity_continuity_ok": self.velocity_continuity_ok,
            "symmetry_ok": self.symmetry_ok,
            "lip_closure_ok": self.lip_closure_ok,
            "passed": self.passed,
            "hard_fail": self.hard_fail,
            "mode": self.mode,
            "notes": list(self.notes),
        }


def savgol_landmarks_xy(
    landmarks: NDArray[np.floating],
    *,
    window: int = 5,
    poly: int = 2,
) -> NDArray[np.float64]:
    """F12 post-filter: Savitzky–Golay(5,2) on landmark x/y (z untouched)."""
    lm = np.asarray(landmarks, dtype=np.float64).copy()
    t = int(lm.shape[0])
    if t < window:
        return lm
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        # Fallback: same 3-tap temporal smooth used elsewhere.
        pad = np.pad(lm, ((1, 1), (0, 0), (0, 0)), mode="edge")
        return 0.25 * pad[:-2] + 0.50 * pad[1:-1] + 0.25 * pad[2:]
    w = window if window % 2 == 1 else window + 1
    w = min(w, t if t % 2 == 1 else t - 1)
    if w < poly + 2:
        return lm
    for axis in (0, 1):
        # filter along time for each landmark
        flat = lm[:, :, axis]
        lm[:, :, axis] = savgol_filter(flat, window_length=w, polyorder=poly, axis=0)
    return lm


def trim_to_lip_closure(
    landmarks: NDArray[np.floating],
    *,
    max_skip: int = 18,
) -> NDArray[np.float64]:
    """Drop leading frames until lips are near-closed (HAPPY open-start fix)."""
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape[0] < 4:
        return lm
    fw = _face_width(lm[0])
    upper = lm[:, 13, :2]
    lower = lm[:, 14, :2]
    aperture = np.linalg.norm(upper - lower, axis=1)
    thr = 0.08 * fw
    for i in range(min(max_skip, len(aperture))):
        if float(aperture[i]) < thr:
            return lm[i:]
    # Best near-closed in the lead-in window.
    best = int(np.argmin(aperture[: max_skip + 1]))
    return lm[best:]


def _face_width(landmarks: NDArray[np.floating]) -> float:
    # MediaPipe face oval approx: use min/max x of all points
    xs = landmarks[..., 0]
    return float(max(xs.max() - xs.min(), 1e-3))


def evaluate_landmarks(
    landmarks: NDArray[np.floating],
    *,
    fps: float = 30.0,
    ee_peak_frame: int | None = None,
    ou_peak_frame: int | None = None,
    mode: str = "vowel",
) -> D35Metrics:
    """Evaluate D35 metrics on (T, N, 2|3) landmark array.

    Uses lip-corner heuristic indices compatible with MediaPipe 478
    (61 / 291 outer corners; 13 / 14 upper/lower lip).

    ``mode="neutral"``: REST jitter + lip closure + symmetry only (blink clip).
    """
    del fps  # reserved for future temporal scaling
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim != 3 or lm.shape[1] < 292:
        raise ValueError("landmarks must be (T, >=292, 2|3)")
    mode_n = (mode or "vowel").strip().lower()
    T = lm.shape[0]
    fw = _face_width(lm[0])
    # lip corners
    left = lm[:, 61, :2]
    right = lm[:, 291, :2]
    upper = lm[:, 13, :2]
    lower = lm[:, 14, :2]
    width = np.linalg.norm(right - left, axis=1)
    aperture = np.linalg.norm(upper - lower, axis=1)

    # REST window: first 6 resampled ticks (~ if starts at rest)
    rest_n = min(6, T)
    corner = 0.5 * (left + right)
    jitter_px = float(np.std(corner[:rest_n], axis=0).mean())
    jitter_pct = 100.0 * jitter_px / fw

    # L/R symmetry of lip corners y
    sym_err = float(np.mean(np.abs(left[:, 1] - right[:, 1])))
    sym_ok = sym_err < 0.02 * fw

    # lip closure at start
    closure_ok = float(aperture[0]) < 0.08 * fw

    notes: list[str] = []
    pass_jitter = jitter_pct < 0.3
    if not pass_jitter:
        notes.append(f"jitter {jitter_pct:.3f}% >= 0.3% face width")
    if not sym_ok:
        notes.append("L/R lip symmetry unstable")
    if not closure_ok:
        notes.append("lip closure at start not near-closed")

    if mode_n == "neutral":
        # Blink / state-0 clip — no EE–OU / vowel-hold gates.
        hard_fail = jitter_pct > 0.8
        passed = pass_jitter and sym_ok and closure_ok and not hard_fail
        if hard_fail:
            notes.append("HARD FAIL: REST jitter > 0.8% face width")
        return D35Metrics(
            jitter_face_width_pct=jitter_pct,
            hold_stability_face_width_pct=0.0,
            ee_ou_distance_face_width_pct=0.0,
            velocity_continuity_ok=True,
            symmetry_ok=sym_ok,
            lip_closure_ok=closure_ok,
            passed=passed,
            hard_fail=hard_fail,
            notes=notes,
            mode="neutral",
        )

    # EE vs OU: use provided peaks or max-width vs min-width frames
    if ee_peak_frame is None:
        ee_peak_frame = int(np.argmax(width))
    if ou_peak_frame is None:
        # prefer a late-ish min so REST at start isn't mistaken for OU
        search = width[T // 5 :]
        ou_peak_frame = int(T // 5 + int(np.argmin(search)))
    ee_v = np.array([width[ee_peak_frame], aperture[ee_peak_frame]])
    ou_v = np.array([width[ou_peak_frame], aperture[ou_peak_frame]])
    sep_pct = 100.0 * float(np.linalg.norm(ee_v - ou_v)) / fw

    # hold stability: local ±3-tick windows around EE/OU peaks (not whole clip)
    def _local_hold_pct(idx: int) -> float:
        lo = max(0, idx - 3)
        hi = min(T, idx + 4)
        if hi - lo < 2:
            return 0.0
        return 100.0 * float(np.std(width[lo:hi]) + np.std(aperture[lo:hi])) / fw

    hold_pct = 0.5 * (_local_hold_pct(ee_peak_frame) + _local_hold_pct(ou_peak_frame))

    # velocity / accel continuity on width
    vel = np.diff(width)  # T-1
    acc = np.diff(vel)  # T-2
    # Align mask to accel length: use vel[1:] (also T-2)
    low = np.abs(vel[1:]) < np.median(np.abs(vel) + 1e-6)
    vel_ok = True
    if np.any(low) and acc.size:
        vel_ok = float(np.std(acc[low])) < 3.0 * float(np.median(np.abs(acc) + 1e-6))

    # Pass bars from FinalAnswers F12
    pass_hold = hold_pct < 2.0
    pass_sep = sep_pct > 8.0
    if not pass_hold:
        notes.append(f"hold σ {hold_pct:.3f}% >= 2% face width")
    if not pass_sep:
        notes.append(f"EE-OU sep {sep_pct:.3f}% <= 8% face width")
    if not vel_ok:
        notes.append("velocity/accel continuity weak on holds")

    hard_fail = jitter_pct > 0.8 or hold_pct > 5.0 or sep_pct < 4.0
    if hard_fail:
        notes.append("HARD FAIL: F12 hard threshold breached")
    passed = (
        pass_jitter
        and pass_hold
        and pass_sep
        and vel_ok
        and sym_ok
        and closure_ok
        and not hard_fail
    )
    return D35Metrics(
        jitter_face_width_pct=jitter_pct,
        hold_stability_face_width_pct=hold_pct,
        ee_ou_distance_face_width_pct=sep_pct,
        velocity_continuity_ok=vel_ok,
        symmetry_ok=sym_ok,
        lip_closure_ok=closure_ok,
        passed=passed,
        hard_fail=hard_fail,
        notes=notes,
        mode="vowel",
    )


def extract_mediapipe_landmarks(video_path: str | Path) -> tuple[NDArray[np.float64], float]:
    """Extract (T, 478, 3) landmarks; returns landmarks + source fps."""
    import cv2  # optional dep

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:
        raise ImportError("mediapipe required for D35 extract") from exc

    video_path = Path(video_path)
    model_path = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"
    if not model_path.is_file():
        # fall back to solutions API
        return _extract_solutions(video_path)

    base = mp_python.BaseOptions(model_asset_path=str(model_path))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[NDArray[np.float64]] = []
    t_ms = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, t_ms)
            t_ms += int(1000 / max(fps, 1e-3))
            if not result.face_landmarks:
                continue
            lm = result.face_landmarks[0]
            arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
            frames.append(arr)
    finally:
        cap.release()
        landmarker.close()
    if not frames:
        raise RuntimeError("no face landmarks detected")
    return np.stack(frames, axis=0), fps


def _extract_solutions(video_path: Path) -> tuple[NDArray[np.float64], float]:
    import cv2
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[NDArray[np.float64]] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
            frames.append(arr)
    finally:
        cap.release()
        face_mesh.close()
    if not frames:
        raise RuntimeError("no face landmarks detected")
    return np.stack(frames, axis=0), fps


def resample_to_60hz(
    landmarks: NDArray[np.floating], src_fps: float
) -> NDArray[np.float64]:
    """Linear interpolate landmark series to 60 Hz."""
    lm = np.asarray(landmarks, dtype=np.float64)
    T = lm.shape[0]
    duration = (T - 1) / max(src_fps, 1e-3)
    n_out = max(1, int(round(duration * TICK_HZ)) + 1)
    t_src = np.linspace(0.0, duration, T)
    t_dst = np.linspace(0.0, duration, n_out)
    out = np.zeros((n_out,) + lm.shape[1:], dtype=np.float64)
    for i in range(lm.shape[1]):
        for j in range(lm.shape[2]):
            out[:, i, j] = np.interp(t_dst, t_src, lm[:, i, j])
    return out


def prepare_landmarks_60hz(
    landmarks: NDArray[np.floating],
    src_fps: float,
    *,
    emotion: str = "NEUTRAL",
    apply_closure_trim: bool = True,
) -> tuple[NDArray[np.float64], D35Metrics]:
    """Resample → SG filter → optional lip-closure trim → evaluate D35."""
    lm60 = resample_to_60hz(landmarks, src_fps)
    lm60 = savgol_landmarks_xy(lm60)
    emo = (emotion or "NEUTRAL").strip().upper()
    mode = "neutral" if emo == "NEUTRAL" else "vowel"
    if apply_closure_trim and mode == "vowel":
        lm60 = trim_to_lip_closure(lm60)
    metrics = evaluate_landmarks(lm60, fps=TICK_HZ, mode=mode)
    return lm60, metrics


def run_d35(
    video_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    emotion: str | None = None,
) -> D35Metrics:
    """Full D35 pipeline on one teacher clip (F12 filtered)."""
    lm, fps = extract_mediapipe_landmarks(video_path)
    stem = Path(video_path).stem.upper()
    emo = (emotion or "NEUTRAL").strip().upper()
    if "NEUTRAL" in stem:
        emo = "NEUTRAL"
    lm60, metrics = prepare_landmarks_60hz(lm, fps, emotion=emo)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "landmarks_60hz.npy", lm60)
        (out / "d35_metrics.json").write_text(
            json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
        )
    return metrics


def write_teacher_package_skeleton(root: str | Path, version: str = "v1") -> Path:
    """Create Teacher Package directory layout (GPT ops lock)."""
    root = Path(root) / f"teacher_package_{version}"
    for sub in ("videos", "landmarks", "optical_flow"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    meta = {
        "version": version,
        "tick_hz": TICK_HZ,
        "ga16": True,
        "status": "skeleton",
    }
    (root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (root / "checksums").write_text("", encoding="utf-8")
    (root / "version").write_text(version + "\n", encoding="utf-8")
    return root
