"""Side B — prepare FaceCellTimeline from avatar video (dense UV-flow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aiface.tickfeed.calibration import write_calibration_script
from aiface.tickfeed.driver import face_box_from_profile
from aiface.tickfeed.package import FaceBox
from aiface.tickfeed.schema import TICK_RATE_HZ
from aiface.tickfeed.synth import synthesize_velocity
from aiface.tickfeed.timeline_io import write_face_cell_timeline


def _optical_flow_face_series(
    video: Path,
    face: FaceBox,
) -> tuple[list[float], list[np.ndarray]] | None:
    """Dense Farneback flow on **every** decoded frame → (t, HxWx2) patches."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    native = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    prev_gray = None
    times: list[float] = []
    flows: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Register: face_box fraction of 256² → same crop in frame (UV contract)
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
            # Scale flow to grid velocity (units / second ≈ flow * fps * scale)
            flow = flow.astype(np.float32) * float(native) * 0.01
            times.append(t)
            flows.append(flow)
        prev_gray = crop
        idx += 1
    cap.release()
    if not flows:
        return None
    return times, flows


def _interp_flow_to_60hz(
    flow_t: list[float],
    flow_v: list[np.ndarray],
    face: FaceBox,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolate dense flow patches onto the 60 Hz master clock."""
    ft = np.asarray(flow_t, dtype=np.float64)
    duration = max(float(ft[-1]), 1.0 / TICK_RATE_HZ)
    n_ticks = int(duration * TICK_RATE_HZ) + 1
    velocities = np.zeros((n_ticks, face.h, face.w, 2), dtype=np.float32)
    conf = np.zeros((n_ticks, face.h * face.w), dtype=np.uint8)
    stacked = np.stack(flow_v, axis=0).astype(np.float32)  # (F,H,W,2)
    for t in range(n_ticks):
        t_sec = float(t) / float(TICK_RATE_HZ)
        if t_sec <= ft[0]:
            velocities[t] = stacked[0]
        elif t_sec >= ft[-1]:
            velocities[t] = stacked[-1]
        else:
            j = int(np.searchsorted(ft, t_sec, side="right") - 1)
            j = max(0, min(j, len(ft) - 2))
            u = (t_sec - ft[j]) / max(ft[j + 1] - ft[j], 1e-9)
            velocities[t] = (1.0 - u) * stacked[j] + u * stacked[j + 1]
        mag = np.linalg.norm(velocities[t], axis=-1).reshape(-1)
        conf[t] = np.clip(40 + mag * 80.0, 40, 255).astype(np.uint8)
    return velocities, conf


def prepare_face_timeline(
    world: Path | str,
    video: Path | str | None = None,
    *,
    sample_fps: float = 0.0,
) -> Path:
    """Build FaceCellTimeline artifacts (dir + flat npz) beside the world.

    ``sample_fps`` is ignored for optical flow (every frame). Kept for API
    compatibility with older scripts.
    """
    del sample_fps
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
            "calibration_take.mp4",
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
        flow_series = _optical_flow_face_series(vid, face)
        if flow_series is not None:
            print(
                f"TickFeed collect: every-frame optical flow n={len(flow_series[1])} "
                f"from {vid.name}"
            )
        try:
            from aiface.behavior.track import extract_transition_track

            track = extract_transition_track(
                vid, world_dir=root, sample_fps=12.0
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
        velocities, conf = _interp_flow_to_60hz(flow_t, flow_v, face)
        n_ticks = velocities.shape[0]
        open_curve = [0.0] * n_ticks
        smile_curve = [0.0] * n_ticks
        if times:
            t_src = np.asarray(times, dtype=np.float64)
            open_s = np.asarray(opens, dtype=np.float64)
            smile_s = np.asarray(smiles, dtype=np.float64)
            for t in range(n_ticks):
                t_sec = float(t) / float(TICK_RATE_HZ)
                o = float(np.interp(t_sec, t_src, open_s))
                s = float(np.interp(t_sec, t_src, smile_s))
                # Script beats amplify weak closed-lip smile / angry (flow alone
                # under-reads those LOOK sections vs OPEN/TALK).
                if 1.0 <= t_sec < 2.0:
                    s = max(s, 0.85)
                if 2.0 <= t_sec < 3.0:
                    o = max(o, 0.85)
                if 4.0 <= t_sec < 5.0:
                    o = max(o, 0.25)
                sur = 0.75 if 4.0 <= t_sec < 5.0 else 0.0
                open_curve[t] = o
                smile_curve[t] = s
                syn = synthesize_velocity(
                    face,
                    open_amt=o,
                    smile_amt=s,
                    surprise_amt=sur,
                    mouth_uv=mouth,
                )
                # Closed-lip SMILE: lean synth harder (measured flow is subtle).
                w_syn = 0.55 if 1.0 <= t_sec < 2.0 else 0.35
                velocities[t] = (1.0 - w_syn) * velocities[t] + w_syn * syn
        out = write_face_cell_timeline(
            root,
            face=face,
            velocity=velocities,
            conf=conf,
            video_name=vid.name if vid else "",
            open_curve=open_curve,
            smile_curve=smile_curve,
        )
        print(
            f"TickFeed collect: wrote {out} ticks={n_ticks} "
            f"face={face.w}x{face.h} source=optical_flow_every_frame"
        )
        return out

    if not times:
        duration = 8.0
        n = int(duration * 12.0)
        for i in range(n):
            t = i / 12.0
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
    conf = np.full((n_ticks, face.h * face.w), 160, dtype=np.uint8)
    open_curve = [0.0] * n_ticks
    smile_curve = [0.0] * n_ticks
    for t in range(n_ticks):
        t_sec = float(t) / float(TICK_RATE_HZ)
        o = float(np.interp(t_sec, t_src, open_s))
        s = float(np.interp(t_sec, t_src, smile_s))
        sur = 0.55 if 4.0 <= t_sec < 5.0 else 0.0
        open_curve[t] = o
        smile_curve[t] = s
        velocities[t] = synthesize_velocity(
            face,
            open_amt=o,
            smile_amt=s,
            surprise_amt=sur,
            mouth_uv=mouth,
        )

    out = write_face_cell_timeline(
        root,
        face=face,
        velocity=velocities,
        conf=conf,
        video_name=vid.name if vid else "script",
        open_curve=open_curve,
        smile_curve=smile_curve,
    )
    print(
        f"TickFeed collect: wrote {out} ticks={n_ticks} "
        f"face={face.w}x{face.h} video={vid.name if vid else 'script'}"
    )
    return out


__all__ = ["prepare_face_timeline"]
