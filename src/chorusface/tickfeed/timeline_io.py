"""FaceCellTimeline on-disk layout (Side B §4.5).

```text
face_cell_timeline/
  meta.json
  rest_ref.npz
  ticks_XXXX.npz          # batches of 60 ticks
  speech_align.json
  look_drive.json
  qa_report.json
```

Also mirrors a flat ``face_cell_timeline.npz`` beside the world for fast load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chorusface.tickfeed.package import FaceBox
from chorusface.tickfeed.qa import qa_beat_motion
from chorusface.tickfeed.schema import CHANNEL_MASK_VELOCITY, TICK_RATE_HZ
from chorusface.tickfeed.audio_feat import (
    extract_audio_feat_table,
    write_audio_feat,
)
from chorusface.tickfeed.force_align import force_align_speech
from chorusface.tickfeed.speech_align import (
    build_look_drive,
    write_look_drive,
    write_speech_align,
)

DIR_NAME = "face_cell_timeline"
BATCH = 60  # 1 second of ticks per chunk


def timeline_dir(world: Path | str) -> Path:
    root = Path(world)
    root = root if root.is_dir() else root.parent
    return root / DIR_NAME


# Per-tick FIELD provenance (design §10): never sell synth as measured.
SOURCE_MEASURED = 0
SOURCE_BLEND = 1  # reserved / legacy
SOURCE_SYNTH = 2


def write_face_cell_timeline(
    world: Path | str,
    *,
    face: FaceBox,
    velocity: NDArray[np.floating],
    conf: NDArray[np.uint8],
    video_name: str = "",
    open_curve: list[float] | None = None,
    smile_curve: list[float] | None = None,
    source: NDArray[np.uint8] | None = None,
    lid_curve: list[float] | None = None,
) -> Path:
    """Write full Side B artifact tree + flat npz mirror."""
    root = Path(world)
    root = root if root.is_dir() else root.parent
    out = timeline_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    vel = np.asarray(velocity, dtype=np.float32)
    conf_a = np.asarray(conf, dtype=np.uint8)
    n_ticks = int(vel.shape[0])
    ticks = np.arange(n_ticks, dtype=np.int32)
    if source is None:
        source_a = np.zeros(n_ticks, dtype=np.uint8)
    else:
        source_a = np.asarray(source, dtype=np.uint8).reshape(-1)
        if source_a.size != n_ticks:
            raise ValueError("source length must match n_ticks")

    meta = {
        "schema": "chorusface.face_cell_timeline.v2",
        "tick_rate": TICK_RATE_HZ,
        "n_ticks": n_ticks,
        "face_box": [face.x, face.y, face.w, face.h],
        "channel_mask": CHANNEL_MASK_VELOCITY,
        "channels": ["vx", "vy"],
        "video": video_name,
        "batch_ticks": BATCH,
        "source_codes": {
            "0": "measured_optical_flow",
            "1": "blend_legacy",
            "2": "synthetic_fallback",
        },
        "n_measured": int(np.sum(source_a == SOURCE_MEASURED)),
        "n_synth": int(np.sum(source_a == SOURCE_SYNTH)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    rest = np.zeros((face.h, face.w, 2), dtype=np.float32)
    np.savez_compressed(out / "rest_ref.npz", velocity=rest, face_box=meta["face_box"])

    # Clear old chunks
    for old in out.glob("ticks_*.npz"):
        old.unlink()
    for start in range(0, n_ticks, BATCH):
        end = min(start + BATCH, n_ticks)
        chunk = out / f"ticks_{start:04d}.npz"
        np.savez_compressed(
            chunk,
            ticks=ticks[start:end],
            velocity=vel[start:end],
            conf=conf_a[start:end],
            source=source_a[start:end],
        )

    video_path = root / "calibration_take.mp4"
    if not video_path.is_file() and video_name:
        cand = root / video_name
        if cand.is_file():
            video_path = cand
    speech = force_align_speech(
        root,
        video_path if video_path.is_file() else None,
        n_ticks=n_ticks,
    )
    look = build_look_drive(
        root,
        n_ticks=n_ticks,
        open_curve=open_curve,
        smile_curve=smile_curve,
        lid_curve=lid_curve,
    )
    write_speech_align(out / "speech_align.json", speech)
    write_look_drive(out / "look_drive.json", look)

    audio_feats, audio_source = extract_audio_feat_table(
        video_path if video_path.is_file() else None,
        n_ticks,
    )
    write_audio_feat(root, audio_feats, source=audio_source)
    meta["audio_feat_source"] = audio_source
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Flat mirror for fast driver load
    flat = root / "face_cell_timeline.npz"
    np.savez_compressed(
        flat,
        ticks=ticks,
        velocity=vel,
        conf=conf_a,
        source=source_a,
        face_box=np.asarray([face.x, face.y, face.w, face.h], dtype=np.int32),
        tick_rate=np.asarray([TICK_RATE_HZ], dtype=np.float64),
    )

    # QA against flat (driver path)
    qa = qa_beat_motion(root)
    qa["speech_align"] = {"n": speech["n_ticks"], "method": speech["method"]}
    qa["look_drive"] = {"n": look["n_ticks"]}
    qa["audio_feat_source"] = audio_source
    (out / "qa_report.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return out


def load_timeline_bundle(world: Path | str) -> dict[str, Any]:
    """Load velocity/conf + speech/look side tracks."""
    root = Path(world)
    root = root if root.is_dir() else root.parent
    flat = root / "face_cell_timeline.npz"
    tdir = timeline_dir(root)
    if not flat.is_file() and tdir.is_dir():
        # Reconstruct flat from chunks
        chunks = sorted(tdir.glob("ticks_*.npz"))
        if not chunks:
            raise FileNotFoundError(f"no timeline in {root}")
        parts_v, parts_c, parts_t, parts_s = [], [], [], []
        for c in chunks:
            d = np.load(c)
            parts_t.append(np.asarray(d["ticks"]))
            parts_v.append(np.asarray(d["velocity"]))
            parts_c.append(np.asarray(d["conf"]))
            if "source" in d.files:
                parts_s.append(np.asarray(d["source"], dtype=np.uint8))
        ticks = np.concatenate(parts_t)
        vel = np.concatenate(parts_v)
        conf = np.concatenate(parts_c)
        source = (
            np.concatenate(parts_s)
            if parts_s
            else np.zeros(len(ticks), dtype=np.uint8)
        )
    else:
        data = np.load(flat)
        ticks = np.asarray(data["ticks"], dtype=np.int32)
        vel = np.asarray(data["velocity"], dtype=np.float32)
        conf = (
            np.asarray(data["conf"], dtype=np.uint8)
            if "conf" in data.files
            else np.full((len(ticks), vel.shape[1] * vel.shape[2]), 200, dtype=np.uint8)
        )
        source = (
            np.asarray(data["source"], dtype=np.uint8)
            if "source" in data.files
            else np.zeros(len(ticks), dtype=np.uint8)
        )

    speech = None
    look = None
    if (tdir / "speech_align.json").is_file():
        speech = json.loads((tdir / "speech_align.json").read_text(encoding="utf-8"))
    if (tdir / "look_drive.json").is_file():
        look = json.loads((tdir / "look_drive.json").read_text(encoding="utf-8"))
    return {
        "ticks": ticks,
        "velocity": vel,
        "conf": conf,
        "source": source,
        "speech": speech,
        "look": look,
        "dir": tdir if tdir.is_dir() else None,
    }


__all__ = [
    "DIR_NAME",
    "SOURCE_BLEND",
    "SOURCE_MEASURED",
    "SOURCE_SYNTH",
    "load_timeline_bundle",
    "timeline_dir",
    "write_face_cell_timeline",
]
