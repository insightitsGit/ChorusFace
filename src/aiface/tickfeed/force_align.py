"""Audio force-align inside calibration beat windows (Side B speech).

Priority:
1. Whisper word timestamps when an API key is present (lab MFA teacher).
2. WAV RMS energy peaks inside script beats.
3. Script schedule only (no video / no wav).

Beat windows stay authoritative; this only refines *when* inside the window.
"""

from __future__ import annotations

import json
import os
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


def _api_key() -> str:
    return (
        os.environ.get("AIFACE_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()


def _script_prompt(script: dict[str, Any]) -> str:
    parts: list[str] = []
    for beat in script.get("beats") or []:
        speech = str(beat.get("speech") or "").strip()
        if speech:
            parts.append(speech)
    return " ".join(parts)


def _whisper_word_spans(wav_path: Path, prompt: str) -> list[Any] | None:
    """Return Whisper word timestamps, or None when unavailable / failed."""
    key = _api_key()
    if not key:
        return None
    try:
        from aiface.audio import decode_wav
        from aiface.tts import (
            DEFAULT_BASE_URL,
            DEFAULT_TRANSCRIBE_MODEL,
            WhisperAligner,
        )
    except Exception:  # noqa: BLE001
        return None
    base_url = (
        os.environ.get("AIFACE_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = os.environ.get("AIFACE_TTS_TRANSCRIBE_MODEL") or DEFAULT_TRANSCRIBE_MODEL
    try:
        clip = decode_wav(wav_path.read_bytes())
        aligner = WhisperAligner(
            api_key=key, base_url=str(base_url), model=str(model)
        )
        return aligner.word_spans(clip, prompt=prompt)
    except Exception:  # noqa: BLE001
        return None


def _apply_whisper_words(
    ticks_out: list[dict[str, Any]],
    *,
    script: dict[str, Any],
    word_spans: list[Any],
    n_ticks: int,
    energy: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Stamp Whisper word/viseme chains onto ticks inside matching beat windows."""
    for beat in script.get("beats") or []:
        bid = str(beat.get("id") or "")
        speech = str(beat.get("speech") or "")
        words = _words_in_speech(speech)
        if not words or bid not in {
            "SAY_HI",
            "TALK",
            "OPEN",
            "SURPRISE",
            "TONGUE_TH",
        }:
            continue
        t0 = float(beat["t0"])
        t1 = float(beat["t1"])
        i0 = max(0, int(t0 * TICK_RATE_HZ))
        i1 = min(n_ticks, int(np.ceil(t1 * TICK_RATE_HZ)))
        if i1 <= i0:
            continue

        # Whisper words whose midpoint lands in this beat.
        in_beat: list[Any] = []
        for ws in word_spans:
            mid = 0.5 * (float(ws.start) + float(ws.end))
            if t0 <= mid < t1:
                in_beat.append(ws)
        # Keep script lexicon (teacher contract); use Whisper times for placement.
        if in_beat:
            for i, ws in enumerate(in_beat):
                word = words[min(i, len(words) - 1)]
                chain = list(_WORD_VISEMES.get(word, ("AA",)))
                if not chain:
                    chain = ["AA"]
                w0 = max(t0, float(ws.start))
                w1 = min(t1, float(ws.end))
                if w1 <= w0:
                    w1 = min(t1, w0 + 1.0 / TICK_RATE_HZ)
                for k, viseme in enumerate(chain):
                    u0 = k / len(chain)
                    u1 = (k + 1) / len(chain)
                    a = w0 + (w1 - w0) * u0
                    b = w0 + (w1 - w0) * u1
                    ta = max(i0, int(a * TICK_RATE_HZ))
                    tb = min(i1, max(ta + 1, int(np.ceil(b * TICK_RATE_HZ))))
                    for tick in range(ta, tb):
                        ticks_out[tick] = {
                            "tick": tick,
                            "viseme": "CLOSED" if viseme == "HH" else viseme,
                            "viseme_id": _viseme_id(viseme),
                            "word": word[:16],
                            "beat_id": int(beat.get("beat_id", 0)),
                            "beat": bid,
                            "energy": float(energy[tick]) if energy is not None else 0.0,
                            "teacher": "whisper_words",
                        }
        else:
            # No whisper hits in window — keep previous energy/script stamps.
            continue

        # Fill gaps in the beat toward nearest assigned tick.
        assigned = {
            int(r["tick"]): r
            for r in ticks_out[i0:i1]
            if str(r.get("beat")) == bid and r.get("word")
        }
        keys = sorted(assigned)
        for t in range(i0, i1):
            if t in assigned or not keys:
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
                "energy": float(energy[t]) if energy is not None else 0.0,
                "teacher": "whisper_words",
            }
    return ticks_out


def _apply_energy_words(
    ticks_out: list[dict[str, Any]],
    *,
    script: dict[str, Any],
    energy: np.ndarray,
    n_ticks: int,
) -> list[dict[str, Any]]:
    for beat in script.get("beats") or []:
        bid = str(beat.get("id") or "")
        speech = str(beat.get("speech") or "")
        words = _words_in_speech(speech)
        if not words or bid not in {
            "SAY_HI",
            "TALK",
            "OPEN",
            "SURPRISE",
            "TONGUE_TH",
        }:
            continue
        t0 = float(beat["t0"])
        t1 = float(beat["t1"])
        i0 = max(0, int(t0 * TICK_RATE_HZ))
        i1 = min(n_ticks, int(np.ceil(t1 * TICK_RATE_HZ)))
        if i1 <= i0:
            continue
        window = energy[i0:i1].copy()
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
                "teacher": "audio_energy",
            }
        assigned = {
            int(r["tick"]): r
            for r in ticks_out[i0:i1]
            if str(r.get("beat")) == bid and r.get("word")
        }
        keys = sorted(assigned)
        for t in range(i0, i1):
            if t in assigned or not keys:
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
                "teacher": "audio_energy",
            }
    return ticks_out


def force_align_speech(
    world: Path | str,
    video: Path | str | None = None,
    *,
    n_ticks: int | None = None,
) -> dict[str, Any]:
    """Build speech_align with Whisper words (keyed) or energy peaks in beats."""
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
        prompt = _script_prompt(script)
        whisper_spans = _whisper_word_spans(wav, prompt)
        method = "audio_energy_force_align"
        if whisper_spans:
            ticks_out = _apply_whisper_words(
                ticks_out,
                script=script,
                word_spans=whisper_spans,
                n_ticks=n_ticks,
                energy=energy,
            )
            method = "whisper_words_force_align"
        else:
            ticks_out = _apply_energy_words(
                ticks_out,
                script=script,
                energy=energy,
                n_ticks=n_ticks,
            )

    return {
        "schema": "aiface.speech_align.v1",
        "tick_rate": TICK_RATE_HZ,
        "n_ticks": n_ticks,
        "method": method,
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
        f"TickFeed force-align: {payload.get('method')} -> {path} "
        f"n={payload.get('n_ticks')}"
    )
    return path


__all__ = ["force_align_speech", "write_force_aligned_speech"]
