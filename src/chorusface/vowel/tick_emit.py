"""Emit TickPackage KEY/Δ from PulseChunk 9D controls (Fabric lane)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chorusface.tickfeed.package import (
    FaceBox,
    TickLabels,
    TickPackage,
    build_delta,
    build_keyframe,
    encode,
)
from chorusface.tickfeed.schema import EmotionId as TickEmotion
from chorusface.vowel.expand import expand_controls
from chorusface.vowel.plates import plate_from_controls
from chorusface.vowel.pulsechunk import PulseChunk
from chorusface.vowel.schema import EMOTIONS

# VowelDesign emotion index → TickFeed EmotionId (different ordinals).
_TO_TICK_EMOTION = {
    "NEUTRAL": int(TickEmotion.NEUTRAL),
    "HAPPY": int(TickEmotion.HAPPY),
    "SAD": int(TickEmotion.SAD),
    "SURPRISED": int(TickEmotion.SURPRISED),
    "ANGRY": int(TickEmotion.ANGRY),
    "THINKING": int(TickEmotion.THINKING),
}


def _labels_from_control(
    c: NDArray[np.floating], emotion_id: int, plate: str
) -> TickLabels:
    # Map 9D → TickLabels LOOK drives
    eye_ap = float(c[0])
    brow_raise = float(c[2])
    brow_knit = float(c[3])
    mouth = float(c[4])
    spread = float(c[5])
    teeth = float(c[7])
    jaw = float(c[8])
    smile = float(np.clip(max(0.0, spread) * 0.8 + (0.2 if emotion_id == 1 else 0.0), 0, 1))
    open_amt = float(np.clip(max(mouth, jaw), 0, 1))
    surprise = float(np.clip(brow_raise, 0, 1))
    brow = float(np.clip(max(brow_raise, brow_knit), 0, 1))
    lid = float(np.clip(1.0 - eye_ap, 0, 1))
    return TickLabels(
        emotion_id=int(emotion_id) & 0xFF,
        viseme_id=TickLabels.viseme_index(plate),
        smile_amt=smile,
        open_amt=open_amt,
        surprise_amt=surprise,
        brow_amt=brow,
        lid_amt=lid,
        word=plate[:16],
        label_conf=255,
    )


def _dense_from_expand(
    face: FaceBox,
    vx: NDArray[np.float32],
    vy: NDArray[np.float32],
    cells: list[tuple[int, int]],
) -> NDArray[np.float32]:
    grid = np.zeros((face.h, face.w, 2), dtype=np.float32)
    for i, (x, y) in enumerate(cells):
        lx = x - face.x
        ly = y - face.y
        if 0 <= lx < face.w and 0 <= ly < face.h:
            grid[ly, lx, 0] = vx[i]
            grid[ly, lx, 1] = vy[i]
    return grid


@dataclass(slots=True)
class EmitConfig:
    face: FaceBox
    W: NDArray[np.float32] | None = None
    cells: list[tuple[int, int]] | None = None
    world_hash: int = 0


def controls_to_velocity_grid(
    c: NDArray[np.floating],
    cfg: EmitConfig,
) -> NDArray[np.float32]:
    face = cfg.face
    if cfg.W is None or cfg.cells is None:
        # label-only path: weak global mouth pump from jaw/mouth channels
        grid = np.zeros((face.h, face.w, 2), dtype=np.float32)
        amp = 0.25 * float(max(c[4], c[8]))
        cy = face.h // 2
        for y in range(max(0, cy - 8), min(face.h, cy + 12)):
            grid[y, :, 1] = amp * (1.0 if y >= cy else -0.3 * amp)
        return grid
    vx, vy = expand_controls(c, cfg.W, cfg.cells)
    return _dense_from_expand(face, vx, vy, cfg.cells)


def emit_tick_packages(
    chunk: PulseChunk,
    cfg: EmitConfig,
) -> list[TickPackage]:
    """Build KEY/Δ TickPackages for every tick in the PulseChunk."""
    key_set = set(chunk.key_ticks)
    packages: list[TickPackage] = []
    prev: NDArray[np.float32] | None = None
    emo_idx = int(chunk.primary_emotion)
    emo_name = EMOTIONS[emo_idx] if 0 <= emo_idx < len(EMOTIONS) else "NEUTRAL"
    tick_emo = _TO_TICK_EMOTION.get(emo_name, int(TickEmotion.NEUTRAL))

    for t in range(chunk.n_ticks):
        c = chunk.controls[t]
        plate = plate_from_controls(c, emo_name)
        labels = _labels_from_control(c, tick_emo, plate)
        grid = controls_to_velocity_grid(c, cfg)
        must_key = t in key_set or t == 0 or prev is None
        if not must_key and prev is not None:
            # size crossover: if dense delta would be huge, KEY (approx)
            delta = grid - prev
            if float(np.mean(np.abs(delta))) > 0.05:
                must_key = True
        if must_key:
            pkg = build_keyframe(
                t,
                cfg.face,
                grid,
                labels=labels,
                world_hash=cfg.world_hash,
            )
        else:
            pkg = build_delta(
                t,
                cfg.face,
                prev,
                grid,
                labels=labels,
                world_hash=cfg.world_hash,
            )
        packages.append(pkg)
        prev = grid
    return packages


def emit_encoded_bytes(chunk: PulseChunk, cfg: EmitConfig) -> list[bytes]:
    return [encode(p) for p in emit_tick_packages(chunk, cfg)]
