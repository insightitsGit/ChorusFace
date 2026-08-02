"""Audio energy force-align inside calibration beat windows (Side B speech).

Extracts WAV from the take (ffmpeg), computes RMS energy @ 60 Hz, and places
scripted words/visemes on energy peaks inside SAY_HI / TALK / OPEN / SURPRISE.
Beat windows stay authoritative; this only refines *when* inside the window.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from aiface.tickfeed.calibration import load_calibration_script
from aiface.tickfeed.schema import TICK_RATE_HZ
from aiface.tickfeed.speech_align import (
    _WORD_VISEMES,
    _viseme_id,
    _words_in_speech,
    build_speech_align,
)


def _extract_wav(video: Path, wav_path: Path) -> bool:
    ffmpeg = "ffmpeg"
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )
        return wav_path.is_file() and wav_path.stat().st_size > 44
    except (OSError, subprocess.CalledProcessError):
        return False


def _rms_at_60hz(wav_path: Path, n_ticks: int) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        rate = int(wf.getframerate())
        n = int(wf.getnframes())
        raw = wf.readframes(n)
        width = wf.getsampwidth()
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    hop = max(int(rate / TICK_RATE_HZ), 1)
    energy = np.zeros(n_ticks, dtype=np.float32)
    for t in range(n_ticks):
        a = t * hop
        b = min(a + hop, samples.size)
        if b > a:
            energy[t] = float(np.sqrt(np.mean(samples[a:b] ** 2) + 1e-12))
    if energy.max() > 1e-8:
        energy = energy / float(energy.max())
    return energy


def force_align_speech(
    world: Path | str,
    video: Path | str | None = None,
    *,
    n_ticks: int | None = None,
) -> dict[str, Any]:
    """Build speech_align with energy peaks inside script beat windows."""
    root = Path(world)
    root = root if root.is_dir() else root.parent
    script = load_calibration_script(root)
    if n_ticks is None:
        npz = root / "face_cell_timeline.npz"
        if npz.is_file():
            n_ticks = int(len(np.load(npz)["ticks"]))
        else:
            n_ticks = int(float(script.get("duration_s") or 8.0) * TICK_RATE_HZ)
    n_ticks = int(n_ticks)

    # Start from script schedule, then refine with audio energy.
    base = build_speech_align(root, n_ticks=n_ticks)
    vid = Path(video) if video else root / "calibration_take.mp4"
    if not vid.is_file():
        base["method"] = "script_force_align"
        return base

    with tempfile.TemporaryDirectory(prefix="aiface_align_") as tmp:
        wav = Path(tmp) / "take.wav"
        if not _extract_wav(vid, wav):
            base["method"] = "script_force_align"
            return base
        energy = _rms_at_60hz(wav, n_ticks)

    ticks_out = list(base["ticks"])
    for beat in script.get("beats") or []:
        bid = str(beat.get("id") or "")
        speech = str(beat.get("speech") or "")
        words = _words_in_speech(speech)
        if not words or bid not in {"SAY_HI", "TALK", "OPEN", "SURPRISE"}:
            continue
        t0 = float(beat["t0"])
        t1 = float(beat["t1"])
        i0 = max(0, int(t0 * TICK_RATE_HZ))
        i1 = min(n_ticks, int(np.ceil(t1 * TICK_RATE_HZ)))
        if i1 <= i0:
            continue
        window = energy[i0:i1].copy()
        # Soft gate so silence doesn't steal word placement
        thr = max(float(np.percentile(window, 55)), 0.08)
        active = np.flatnonzero(window >= thr)
        if active.size == 0:
            active = np.arange(window.size)

        chain: list[tuple[str, str]] = []
        for w in words:
            for v in _WORD_VISEMES.get(w, ("AA",)):
                chain.append((w, v))
        if not chain:
            continue
        # Map chain evenly across active energy frames
        for k, (word, viseme) in enumerate(chain):
            u = (k + 0.5) / len(chain)
            idx = int(active[min(int(u * active.size), active.size - 1)])
            tick = i0 + idx
            ticks_out[tick] = {
                "tick": tick,
                "viseme": "CLOSED" if viseme == "HH" else viseme,
                "viseme_id": _viseme_id(viseme),
                "word": word[:16],
                "beat_id": int(beat.get("beat_id", 0)),
                "beat": bid,
                "energy": float(energy[tick]),
            }
        # Fill remaining ticks in window toward nearest assigned chain item
        assigned = {
            int(r["tick"]): r
            for r in ticks_out[i0:i1]
            if str(r.get("beat")) == bid and r.get("word")
        }
        keys = sorted(assigned)
        for t in range(i0, i1):
            if t in assigned:
                continue
            if not keys:
                continue
            nearest = min(keys, key=lambda k: abs(k - t))
            src = assigned[nearest]
            ticks_out[t] = {
                "tick": t,
                "viseme": src["viseme"],
                "viseme_id": src["viseme_id"],
                "word": src["word"],
                "beat_id": int(beat.get("beat_id", 0)),
                "beat": bid,
                "energy": float(energy[t]),
            }

    return {
        "schema": "aiface.speech_align.v1",
        "tick_rate": TICK_RATE_HZ,
        "n_ticks": n_ticks,
        "method": "audio_energy_force_align",
        "video": str(vid.name),
        "ticks": ticks_out,
    }


def write_force_aligned_speech(
    world: Path | str,
    video: Path | str | None = None,
) -> Path:
    root = Path(world)
    root = root if root.is_dir() else root.parent
    out_dir = root / "face_cell_timeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = force_align_speech(root, video)
    path = out_dir / "speech_align.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"TickFeed force-align: {payload.get('method')} → {path} "
        f"n={payload.get('n_ticks')}"
    )
    return path


__all__ = ["force_align_speech", "write_force_aligned_speech"]
