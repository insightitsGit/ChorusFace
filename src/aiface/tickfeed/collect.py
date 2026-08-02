"""Side B — prepare face_cell_timeline.npz from avatar video (landmark→face)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aiface.tickfeed.driver import face_box_from_profile
from aiface.tickfeed.package import FaceBox
from aiface.tickfeed.schema import TICK_RATE_HZ
from aiface.tickfeed.synth import synthesize_velocity


def prepare_face_timeline(
    world: Path | str,
    video: Path | str | None = None,
    *,
    sample_fps: float = 12.0,
) -> Path:
    """Build a 60 Hz face velocity timeline beside the world.

    Uses landmark-derived open/smile curves when video is available; otherwise
    a short synthetic calibration pattern (REST→SMILE→OPEN→…) for QA.
    """
    root = Path(world)
    root = root if root.is_dir() else root.parent
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

    if vid is not None and vid.is_file():
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
            print(f"TickFeed collect: video track failed ({exc}); using script")
            times, opens, smiles = [], [], []

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

    out = root / "face_cell_timeline.npz"
    np.savez_compressed(
        out,
        ticks=tick_index,
        velocity=velocities,
        face_box=np.asarray([face.x, face.y, face.w, face.h], dtype=np.int32),
        tick_rate=np.asarray([TICK_RATE_HZ], dtype=np.float64),
    )
    print(
        f"TickFeed collect: wrote {out} ticks={n_ticks} "
        f"face={face.w}x{face.h} video={vid.name if vid else 'script'}"
    )
    return out


__all__ = ["prepare_face_timeline"]
