"""Real per-tick audio features for L1 SpeechClock (not open/smile proxies)."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aiface.tickfeed.schema import TICK_RATE_HZ

AUDIO_FEAT = 8
AUDIO_FEAT_NAME = "audio_feat.npz"


def _extract_wav(video: Path, wav_path: Path) -> bool:
    import subprocess

    try:
        subprocess.run(
            [
                "ffmpeg",
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


def _rms_series(wav_path: Path, n_ticks: int) -> NDArray[np.float32]:
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
    peak = float(energy.max())
    if peak > 1e-8:
        energy = energy / peak
    return energy


def feats_from_energy(energy: NDArray[np.floating]) -> NDArray[np.float32]:
    """Build (N, 8) audio feature table from normalized RMS @ 60 Hz."""
    e = np.asarray(energy, dtype=np.float32).reshape(-1)
    n = int(e.size)
    out = np.zeros((n, AUDIO_FEAT), dtype=np.float32)
    if n == 0:
        return out
    delta = np.zeros(n, dtype=np.float32)
    delta[1:] = e[1:] - e[:-1]
    # 5-tick moving average
    ma = np.convolve(e, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    # Peak hold in ±3 tick window
    peak = np.zeros(n, dtype=np.float32)
    for t in range(n):
        a = max(0, t - 3)
        b = min(n, t + 4)
        peak[t] = float(e[a:b].max())
    silence = (e < 0.08).astype(np.float32)
    voiced = (e >= 0.12).astype(np.float32)
    # Soft onset: positive delta gated by energy
    onset = np.clip(delta, 0.0, 1.0) * voiced
    out[:, 0] = e
    out[:, 1] = delta
    out[:, 2] = ma
    out[:, 3] = peak
    out[:, 4] = silence
    out[:, 5] = voiced
    out[:, 6] = onset
    out[:, 7] = np.clip(e * voiced, 0.0, 1.0)
    return out


def extract_audio_feat_table(
    video: Path | str | None,
    n_ticks: int,
) -> tuple[NDArray[np.float32], str]:
    """Return ``(feats[N,8], source)`` where source is ``wav_rms`` or ``zeros``."""
    n = int(n_ticks)
    if video is None or not Path(video).is_file():
        return np.zeros((n, AUDIO_FEAT), dtype=np.float32), "zeros"
    with tempfile.TemporaryDirectory(prefix="aiface_audio_") as tmp:
        wav = Path(tmp) / "take.wav"
        if not _extract_wav(Path(video), wav):
            return np.zeros((n, AUDIO_FEAT), dtype=np.float32), "zeros"
        energy = _rms_series(wav, n)
    return feats_from_energy(energy), "wav_rms"


def write_audio_feat(
    world: Path | str,
    feats: NDArray[np.floating],
    *,
    source: str,
) -> Path:
    root = Path(world)
    root = root if root.is_dir() else root.parent
    out_dir = root / "face_cell_timeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / AUDIO_FEAT_NAME
    arr = np.asarray(feats, dtype=np.float32)
    np.savez_compressed(
        path,
        feats=arr,
        source=np.asarray([source]),
        tick_rate=np.asarray([TICK_RATE_HZ], dtype=np.float64),
    )
    # Flat mirror beside world for fast ML load
    flat = root / AUDIO_FEAT_NAME
    np.savez_compressed(
        flat,
        feats=arr,
        source=np.asarray([source]),
        tick_rate=np.asarray([TICK_RATE_HZ], dtype=np.float64),
    )
    return path


def load_audio_feat(world: Path | str) -> tuple[NDArray[np.float32], str] | None:
    root = Path(world)
    root = root if root.is_dir() else root.parent
    for path in (root / AUDIO_FEAT_NAME, root / "face_cell_timeline" / AUDIO_FEAT_NAME):
        if not path.is_file():
            continue
        data = np.load(path)
        feats = np.asarray(data["feats"], dtype=np.float32)
        source = "wav_rms"
        if "source" in data.files:
            raw = data["source"]
            source = str(raw[0]) if getattr(raw, "size", 0) else str(raw)
        return feats, source
    return None


__all__ = [
    "AUDIO_FEAT",
    "AUDIO_FEAT_NAME",
    "extract_audio_feat_table",
    "feats_from_energy",
    "load_audio_feat",
    "write_audio_feat",
]
