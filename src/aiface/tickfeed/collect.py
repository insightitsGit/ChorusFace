"""Side B — prepare face_cell_timeline.npz from avatar video (landmark→face)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aiface.tickfeed.calibration import write_calibration_script
from aiface.tickfeed.driver import face_box_from_profile
from aiface.tickfeed.package import FaceBox
from aiface.tickfeed.schema import TICK_RATE_HZ
from aiface.tickfeed.synth import synthesize_velocity


def _optical_flow_face_series(
    video: Path,
    face: FaceBox,
    *,
    sample_fps: float,
) -> tuple[list[float], list[np.ndarray]] | None:
    """Dense Farneback flow on face crop → list of (t, HxWx2) patches."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    native = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(int(round(native / max(sample_fps, 0.5))), 1)
    prev_gray = None
    times: list[float] = []
    flows: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Approximate face crop: center box scaled to frame
        h, w = gray.shape[:2]
        x0 = int(w * (face.x / 256.0))
        y0 = int(h * (face.y / 256.0))
        x1 = int(w * ((face.x + face.w) / 256.0))
        y1 = int(h * ((face.y + face.h) / 256.0))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x0 + 8, x1)), min(h, max(y0 + 8, y1))
        crop = gray[y0:y1, x0:x1]
        crop = cv2.resize(crop, (face.w, face.h), interpolation=cv2.INTER_AREA)
        t = idx / native
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                crop,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            # Scale flow to grid velocity-ish units
            flow = flow.astype(np.float32) * 0.15
            times.append(t)
            flows.append(flow)
        prev_gray = crop
        idx += 1
    cap.release()
    if not flows:
        return None
    return times, flows


def prepare_face_timeline(
    world: Path | str,
    video: Path | str | None = None,
    *,
    sample_fps: float = 12.0,
) -> Path:
    """Build a 60 Hz face velocity timeline beside the world.

    Prefer optical-flow face patches when OpenCV can read the video; else
    landmark open/smile curves → synth; else 8s calibration script pattern.
    """
    root = Path(world)
    root = root if root.is_dir() else root.parent
    write_calibration_script(root)
    face = face_box_from_profile(root)
    mouth = (face.x + face.w * 0.5, face.y + face.h * 0.62)

    opens: list[float] = []
    smiles: list[float] = []
    times: list[float] = []

    vid = Path(video) if video else None
    if vid is None:
        for name in (
            "Generate_a_single_continuous_.mp4",
            "source.mp4",
            "avatar.mp4",
        ):
            cand = root / name
            if cand.is_file():
                vid = cand
                break
            cand = Path("assets/avatar_video_inputs") / name
            if cand.is_file():
                vid = cand
                break

    flow_series = None
    if vid is not None and vid.is_file():
        flow_series = _optical_flow_face_series(
            vid, face, sample_fps=float(sample_fps)
        )
        if flow_series is not None:
            print(
                f"TickFeed collect: optical flow patches n={len(flow_series[1])} "
                f"from {vid.name}"
            )
        try:
            from aiface.behavior.track import extract_transition_track

            track = extract_transition_track(
                vid, world_dir=root, sample_fps=float(sample_fps)
            )
            for i in range(track.n_samples):
                times.append(float(track.times[i]))
                opens.append(float(track.controls[i, 0]))
                smiles.append(float(track.controls[i, 2]))
        except Exception as exc:  # noqa: BLE001
            print(f"TickFeed collect: landmark track failed ({exc})")
            times, opens, smiles = [], [], []

    if flow_series is not None:
        flow_t, flow_v = flow_series
        duration = max(flow_t[-1], 1.0 / TICK_RATE_HZ)
        n_ticks = int(duration * TICK_RATE_HZ) + 1
        velocities = np.zeros((n_ticks, face.h, face.w, 2), dtype=np.float32)
        conf = np.zeros((n_ticks, face.h * face.w), dtype=np.uint8)
        tick_index = np.arange(n_ticks, dtype=np.int32)
        ft = np.asarray(flow_t, dtype=np.float64)
        # Stack flows for interpolation along time of each pixel is heavy;
        # nearest sample is enough for teacher.
        for t in tick_index:
            t_sec = float(t) / float(TICK_RATE_HZ)
            j = int(np.searchsorted(ft, t_sec, side="right") - 1)
            j = max(0, min(j, len(flow_v) - 1))
            velocities[t] = flow_v[j]
            # Blend landmark synth if available for open/smile emphasis
            if times:
                o = float(np.interp(t_sec, times, opens))
                s = float(np.interp(t_sec, times, smiles))
                syn = synthesize_velocity(
                    face, open_amt=o, smile_amt=s, mouth_uv=mouth
                )
                velocities[t] = 0.65 * velocities[t] + 0.35 * syn
            mag = np.linalg.norm(velocities[t], axis=-1).reshape(-1)
            # High motion → high conf; quiet cells still keep a floor (measured).
            conf[t] = np.clip(40 + mag * 400.0, 40, 255).astype(np.uint8)
        out = root / "face_cell_timeline.npz"
        np.savez_compressed(
            out,
            ticks=tick_index,
            velocity=velocities,
            conf=conf,
            face_box=np.asarray([face.x, face.y, face.w, face.h], dtype=np.int32),
            tick_rate=np.asarray([TICK_RATE_HZ], dtype=np.float64),
        )
        print(
            f"TickFeed collect: wrote {out} ticks={n_ticks} "
            f"face={face.w}x{face.h} source=optical_flow"
        )
        return out

    if not times:
        # 8s calibration script proxy @ sample_fps
        duration = 8.0
        n = int(duration * sample_fps)
        for i in range(n):
            t = i / sample_fps
            times.append(t)
            if 1.0 <= t < 2.0:
                opens.append(0.0)
                smiles.append(0.85)
            elif 2.0 <= t < 3.0:
                opens.append(0.9)
                smiles.append(0.1)
            elif 4.0 <= t < 5.0:
                opens.append(0.2)
                smiles.append(0.0)
            elif 5.0 <= t < 6.0:
                opens.append(0.1)
                smiles.append(0.0)
            elif 6.0 <= t < 7.5:
                opens.append(0.5 + 0.3 * np.sin(t * 8.0))
                smiles.append(0.2)
            else:
                opens.append(0.0)
                smiles.append(0.0)

    duration = max(times[-1], 1.0 / TICK_RATE_HZ)
    n_ticks = int(duration * TICK_RATE_HZ) + 1
    t_src = np.asarray(times, dtype=np.float64)
    open_s = np.asarray(opens, dtype=np.float64)
    smile_s = np.asarray(smiles, dtype=np.float64)

    velocities = np.zeros((n_ticks, face.h, face.w, 2), dtype=np.float32)
    conf = np.full((n_ticks, face.h * face.w), 180, dtype=np.uint8)
    tick_index = np.arange(n_ticks, dtype=np.int32)
    for t in tick_index:
        t_sec = float(t) / float(TICK_RATE_HZ)
        o = float(np.interp(t_sec, t_src, open_s))
        s = float(np.interp(t_sec, t_src, smile_s))
        sur = 0.55 if 4.0 <= t_sec < 5.0 else 0.0
        velocities[t] = synthesize_velocity(
            face,
            open_amt=o,
            smile_amt=s,
            surprise_amt=sur,
            mouth_uv=mouth,
        )
        # Synth teacher: mid confidence (not optical-flow measured).
        conf[t] = 160

    out = root / "face_cell_timeline.npz"
    np.savez_compressed(
        out,
        ticks=tick_index,
        velocity=velocities,
        conf=conf,
        face_box=np.asarray([face.x, face.y, face.w, face.h], dtype=np.int32),
        tick_rate=np.asarray([TICK_RATE_HZ], dtype=np.float64),
    )
    print(
        f"TickFeed collect: wrote {out} ticks={n_ticks} "
        f"face={face.w}x{face.h} video={vid.name if vid else 'script'}"
    )
    return out


__all__ = ["prepare_face_timeline"]
