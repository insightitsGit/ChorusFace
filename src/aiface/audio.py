"""Waveform decoding, energy analysis, and real audio playback.

The face has to move with a voice it can actually hear, so this module keeps
three concerns separate and testable:

* **Decoding** — a small RIFF/WAVE reader that accepts integer PCM and IEEE
  float payloads, because synthesisers disagree about which one they emit.
* **Analysis** — a short-time RMS envelope and voiced-interval detection. This
  is the measurement that lets viseme timing follow the recording instead of a
  guess derived from letter counts.
* **Playback** — non-blocking sinks that start a clip and report roughly how
  long the start took, so the viseme clock can be offset to match.

No part of this module touches the GPU or the window, so the whole speech path
stays importable and testable headless.
"""

from __future__ import annotations

import io
import math
import os
import platform
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Iterable, Protocol, Sequence

import numpy as np

WAVE_FORMAT_PCM: Final = 0x0001
WAVE_FORMAT_IEEE_FLOAT: Final = 0x0003
WAVE_FORMAT_EXTENSIBLE: Final = 0xFFFE

DEFAULT_HOP_SECONDS: Final = 0.010
DEFAULT_WINDOW_SECONDS: Final = 0.025
# Fraction of the loudest frame that still counts as voice.
DEFAULT_VOICE_THRESHOLD: Final = 0.14
DEFAULT_MIN_VOICED: Final = 0.035
DEFAULT_MIN_SILENCE: Final = 0.075

# Rough time between asking a backend to start and the first sample leaving the
# device. Overridable per sink; the app exposes a further manual trim.
SOUNDDEVICE_LATENCY: Final = 0.035
WINSOUND_LATENCY: Final = 0.060
COMMAND_LATENCY: Final = 0.130


class AudioError(RuntimeError):
    """A waveform could not be decoded or played."""


@dataclass(frozen=True, slots=True)
class AudioClip:
    """Mono float32 audio in the range [-1, 1]."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.samples.ndim != 1:
            raise ValueError("samples must be mono (one dimension)")

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        return self.frame_count / float(self.sample_rate)

    @property
    def peak(self) -> float:
        return float(np.abs(self.samples).max()) if self.frame_count else 0.0

    def to_wav_bytes(self) -> bytes:
        return encode_wav(self)


def time_stretch(clip: AudioClip, pace: float) -> AudioClip:
    """Slow (``pace>1``) or speed (``pace<1``) a clip by resampling duration.

    Keeps sample rate; length scales by ``pace``. Pitch shifts with rate
    (record-slow) — acceptable for lab speech-clarity pacing.
    """
    pace = float(pace)
    if pace <= 1e-3 or abs(pace - 1.0) < 1e-4 or clip.frame_count == 0:
        return clip
    n = int(clip.frame_count)
    n_new = max(1, int(round(n * pace)))
    if n_new == n:
        return clip
    src = np.asarray(clip.samples, dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
    out = np.interp(x_new, x_old, src).astype(np.float32)
    return AudioClip(samples=out, sample_rate=int(clip.sample_rate))


def _find_chunks(data: bytes) -> dict[bytes, tuple[int, int]]:
    """Map chunk id → (offset, size) for a RIFF/WAVE payload."""
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("not a RIFF/WAVE payload")
    chunks: dict[bytes, tuple[int, int]] = {}
    offset = 12
    total = len(data)
    while offset + 8 <= total:
        chunk_id = data[offset : offset + 4]
        (size,) = struct.unpack_from("<I", data, offset + 4)
        body = offset + 8
        # A truncated final chunk is common in streamed audio; keep what is there.
        size = min(size, total - body)
        chunks.setdefault(chunk_id, (body, size))
        offset = body + size + (size & 1)
    if b"fmt " not in chunks or b"data" not in chunks:
        raise AudioError("RIFF payload is missing a fmt or data chunk")
    return chunks


def _pcm_to_float(raw: bytes, *, bits: int, format_tag: int) -> np.ndarray:
    if format_tag == WAVE_FORMAT_IEEE_FLOAT:
        dtype = {32: "<f4", 64: "<f8"}.get(bits)
        if dtype is None:
            raise AudioError(f"unsupported float width {bits}")
        return np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if format_tag != WAVE_FORMAT_PCM:
        raise AudioError(f"unsupported WAVE format tag {format_tag:#06x}")
    if bits == 8:
        # 8-bit PCM is unsigned by definition.
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (values - 128.0) / 128.0
    if bits == 16:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if bits == 24:
        usable = len(raw) - (len(raw) % 3)
        triples = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(-1, 3)
        values = (
            triples[:, 0].astype(np.int32)
            | (triples[:, 1].astype(np.int32) << 8)
            | (triples[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if bits == 32:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise AudioError(f"unsupported PCM bit depth {bits}")


def decode_wav(data: bytes) -> AudioClip:
    """Decode a WAVE payload into mono float32 samples.

    Handles integer PCM (8/16/24/32-bit), IEEE float, and the extensible
    wrapper, and downmixes any channel count to mono.
    """
    chunks = _find_chunks(data)
    fmt_offset, fmt_size = chunks[b"fmt "]
    if fmt_size < 16:
        raise AudioError("fmt chunk is too short")
    format_tag, channels, sample_rate, _bytes_per_second, _align, bits = (
        struct.unpack_from("<HHIIHH", data, fmt_offset)
    )
    if format_tag == WAVE_FORMAT_EXTENSIBLE:
        if fmt_size < 26:
            raise AudioError("extensible fmt chunk is too short")
        (format_tag,) = struct.unpack_from("<H", data, fmt_offset + 24)
    if channels <= 0:
        raise AudioError("fmt chunk declares no channels")

    data_offset, data_size = chunks[b"data"]
    frame_bytes = max(bits // 8, 1) * channels
    data_size -= data_size % frame_bytes if frame_bytes else 0
    flat = _pcm_to_float(
        data[data_offset : data_offset + data_size],
        bits=bits,
        format_tag=format_tag,
    )
    usable = flat.shape[0] - (flat.shape[0] % channels)
    mono = flat[:usable].reshape(-1, channels).mean(axis=1) if channels > 1 else flat
    return AudioClip(np.ascontiguousarray(mono, dtype=np.float32), int(sample_rate))


def encode_wav(clip: AudioClip) -> bytes:
    """Encode a clip as 16-bit PCM WAVE, the format every sink understands."""
    scaled = np.clip(clip.samples, -1.0, 1.0) * 32767.0
    payload = np.rint(scaled).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(clip.sample_rate)
        handle.writeframes(payload)
    return buffer.getvalue()


def decode_audio(data: bytes) -> AudioClip:
    """Decode audio bytes, requiring a WAVE container.

    Compressed containers (mp3, opus, aac) are deliberately not guessed at:
    silently mis-decoding them would desynchronise the whole face. Callers ask
    their synthesiser for WAVE instead.
    """
    if len(data) >= 12 and data[0:4] == b"RIFF":
        return decode_wav(data)
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        raise AudioError("mp3 payload received; request wav from the synthesiser")
    if data[:4] == b"OggS":
        raise AudioError("ogg payload received; request wav from the synthesiser")
    raise AudioError("unrecognised audio container; request wav")


@dataclass(frozen=True, slots=True)
class Envelope:
    """Short-time RMS energy of a clip on a uniform hop grid."""

    values: np.ndarray
    hop: float
    window: float

    @property
    def frame_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def duration(self) -> float:
        return self.frame_count * self.hop

    def time_of(self, index: int) -> float:
        return index * self.hop

    def segment(self, start: float, end: float) -> "Envelope":
        """The frames covering ``[start, end)``, as an envelope in its own right."""
        first = max(int(math.floor(start / self.hop)), 0)
        last = min(int(math.ceil(end / self.hop)), self.frame_count)
        return Envelope(self.values[first : max(last, first)], self.hop, self.window)

    @property
    def peak(self) -> float:
        return float(self.values.max()) if self.frame_count else 0.0

    def noise_floor(self, percentile: float = 15.0) -> float:
        if not self.frame_count:
            return 0.0
        return float(np.percentile(self.values, percentile))


def rms_envelope(
    clip: AudioClip,
    *,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> Envelope:
    """Measure short-time RMS energy with an O(n) prefix-sum sweep."""
    hop = max(int(round(hop_seconds * clip.sample_rate)), 1)
    window = max(int(round(window_seconds * clip.sample_rate)), hop)
    if clip.frame_count == 0:
        return Envelope(np.zeros(0, dtype=np.float32), hop / clip.sample_rate,
                        window / clip.sample_rate)

    power = np.square(clip.samples.astype(np.float64))
    prefix = np.concatenate(([0.0], np.cumsum(power)))
    starts = np.arange(0, clip.frame_count, hop, dtype=np.int64)
    ends = np.minimum(starts + window, clip.frame_count)
    widths = np.maximum(ends - starts, 1)
    values = np.sqrt((prefix[ends] - prefix[starts]) / widths)
    return Envelope(
        values.astype(np.float32),
        hop / clip.sample_rate,
        window / clip.sample_rate,
    )


def voiced_intervals(
    envelope: Envelope,
    *,
    threshold: float = DEFAULT_VOICE_THRESHOLD,
    min_voiced: float = DEFAULT_MIN_VOICED,
    min_silence: float = DEFAULT_MIN_SILENCE,
) -> list[tuple[float, float]]:
    """Find the spans where the clip is actually speaking.

    The gate sits above both a fraction of the loudest frame and the measured
    noise floor, so a quiet recording is not read as one long vowel and a hissy
    one is not read as continuous speech.
    """
    if envelope.frame_count == 0:
        return []
    peak = envelope.peak
    if peak <= 0.0:
        return []
    gate = max(threshold * peak, envelope.noise_floor() * 1.8)
    loud = envelope.values >= gate
    if not loud.any():
        return []

    spans: list[list[float]] = []
    start_index: int | None = None
    for index, active in enumerate(loud):
        if active and start_index is None:
            start_index = index
        elif not active and start_index is not None:
            spans.append([envelope.time_of(start_index), envelope.time_of(index)])
            start_index = None
    if start_index is not None:
        spans.append([envelope.time_of(start_index), envelope.duration])

    merged: list[list[float]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] < min_silence:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    return [
        (float(start), float(end))
        for start, end in merged
        if end - start >= min_voiced
    ]


def energy_warp(
    envelope: Envelope,
    positions: Sequence[float],
    *,
    strength: float = 0.65,
    floor: float = 0.05,
) -> list[float]:
    """Map normalised script positions onto clip time by cumulative energy.

    ``positions`` are fractions of the way through the written utterance. The
    returned times are seconds into the clip. Matching *energy* rather than
    duration is what makes leading silence, mid-sentence pauses, and a trailing
    breath fall out for free: no acoustic energy is spent there, so the script
    does not advance across them.

    ``strength`` blends the energy map with a plain linear map. Pure energy
    warping is brittle on clips with one dominant burst; a blend of two monotone
    maps is still monotone, and keeps the schedule sane.
    """
    duration = envelope.duration
    if duration <= 0.0 or envelope.frame_count == 0:
        return [0.0 for _ in positions]

    # A small floor keeps silence advancing slowly instead of stalling forever.
    weights = envelope.values.astype(np.float64)
    peak = float(weights.max())
    if peak > 0.0:
        weights = np.maximum(weights, peak * floor)
    else:
        weights = np.ones_like(weights)
    cumulative = np.concatenate(([0.0], np.cumsum(weights)))
    total = float(cumulative[-1])
    if total <= 0.0:
        return [float(np.clip(value, 0.0, 1.0)) * duration for value in positions]

    grid = cumulative / total
    times = np.arange(envelope.frame_count + 1, dtype=np.float64) * envelope.hop
    times = np.minimum(times, duration)

    blend = float(np.clip(strength, 0.0, 1.0))
    result: list[float] = []
    for value in positions:
        fraction = float(np.clip(value, 0.0, 1.0))
        warped = float(np.interp(fraction, grid, times))
        linear = fraction * duration
        result.append(warped * blend + linear * (1.0 - blend))
    return result


# --------------------------------------------------------------------- sinks


class AudioSink(Protocol):
    """A non-blocking speaker."""

    name: str
    startup_latency: float

    def play(self, clip: AudioClip) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def media_time(self) -> float | None:
        """Seconds since current clip started, or None if unknown."""
        ...


class NullSink:
    """Swallows audio. Used by ``--no-audio`` and by headless tests."""

    name = "null"
    startup_latency = 0.0

    def __init__(self) -> None:
        self.played: list[AudioClip] = []
        self._t0: float | None = None

    def play(self, clip: AudioClip) -> None:
        import time as _time

        self.played.append(clip)
        self._t0 = _time.perf_counter()

    def media_time(self) -> float | None:
        import time as _time

        if self._t0 is None:
            return None
        return max(0.0, _time.perf_counter() - self._t0)

    def stop(self) -> None:
        self._t0 = None

    def close(self) -> None:
        self.stop()


class SoundDeviceSink:
    """PortAudio playback via ``sounddevice``: lowest and most stable latency."""

    name = "sounddevice"

    def __init__(self) -> None:
        try:
            import sounddevice
        except (ImportError, OSError) as exc:  # OSError: PortAudio missing
            raise AudioError(f"sounddevice unavailable ({exc})") from exc
        self._module = sounddevice
        self.startup_latency = SOUNDDEVICE_LATENCY
        self._t0: float | None = None

    def play(self, clip: AudioClip) -> None:
        import time as _time

        self._module.stop()
        self._module.play(clip.samples, clip.sample_rate, blocking=False)
        # Anchor after play() returns — media_time tracks clip progress for
        # viseme fire instead of a fixed startup_latency guess alone.
        self._t0 = _time.perf_counter()

    def media_time(self) -> float | None:
        import time as _time

        if self._t0 is None:
            return None
        return max(0.0, _time.perf_counter() - self._t0)

    def stop(self) -> None:
        self._t0 = None
        self._module.stop()

    def close(self) -> None:
        self.stop()


class _TempWavSink:
    """Shared plumbing for sinks that need the clip as a file on disk."""

    name = "file"
    startup_latency = COMMAND_LATENCY

    def __init__(self) -> None:
        self._directory = Path(tempfile.mkdtemp(prefix="aiface-audio-"))
        self._lock = threading.Lock()
        self._slot = 0
        self._t0: float | None = None

    def _write(self, clip: AudioClip) -> Path:
        # Alternate two files: a backend may still hold the previous one open.
        self._slot ^= 1
        path = self._directory / f"speech{self._slot}.wav"
        path.write_bytes(clip.to_wav_bytes())
        return path

    def _mark_started(self) -> None:
        self._t0 = time.perf_counter()

    def media_time(self) -> float | None:
        if self._t0 is None:
            return None
        return max(0.0, time.perf_counter() - self._t0)

    def play(self, clip: AudioClip) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self._t0 = None

    def close(self) -> None:
        self.stop()
        shutil.rmtree(self._directory, ignore_errors=True)


class WinsoundSink(_TempWavSink):
    """Windows playback through the stdlib ``winsound`` module."""

    name = "winsound"
    startup_latency = WINSOUND_LATENCY

    def __init__(self) -> None:
        try:
            import winsound
        except ImportError as exc:
            raise AudioError("winsound is only available on Windows") from exc
        super().__init__()
        self._module = winsound

    def play(self, clip: AudioClip) -> None:
        with self._lock:
            path = self._write(clip)
            flags = (
                self._module.SND_FILENAME
                | self._module.SND_ASYNC
                | self._module.SND_NODEFAULT
            )
            self._module.PlaySound(str(path), flags)
            self._mark_started()

    def stop(self) -> None:
        with self._lock:
            self._module.PlaySound(None, self._module.SND_PURGE)
            self._t0 = None


class CommandSink(_TempWavSink):
    """Playback by spawning a system player (``afplay``, ``aplay``, ``ffplay``)."""

    name = "command"

    #: First entry that exists on PATH wins.
    CANDIDATES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
        ("afplay", ()),
        ("paplay", ()),
        ("aplay", ("-q",)),
        ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "quiet")),
        ("play", ("-q",)),
    )

    def __init__(self, executable: str | None = None, arguments: Iterable[str] = ()) -> None:
        resolved: tuple[str, tuple[str, ...]] | None = None
        if executable:
            found = shutil.which(executable)
            if found:
                resolved = (found, tuple(arguments))
        else:
            for candidate, defaults in self.CANDIDATES:
                found = shutil.which(candidate)
                if found:
                    resolved = (found, defaults)
                    break
        if resolved is None:
            raise AudioError("no system audio player found on PATH")
        super().__init__()
        self._executable, self._arguments = resolved
        self._process: subprocess.Popen[bytes] | None = None
        self.name = f"command:{Path(self._executable).stem}"

    def play(self, clip: AudioClip) -> None:
        with self._lock:
            self._terminate()
            path = self._write(clip)
            self._process = subprocess.Popen(
                [self._executable, *self._arguments, str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._mark_started()

    def _terminate(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            self._process = None
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
        self._process = None

    def stop(self) -> None:
        with self._lock:
            self._terminate()


SinkFactory = Callable[[], AudioSink]

SINK_FACTORIES: Final[dict[str, SinkFactory]] = {
    "sounddevice": SoundDeviceSink,
    "winsound": WinsoundSink,
    "command": CommandSink,
    "null": NullSink,
}


def open_audio_sink(preference: str = "auto") -> AudioSink:
    """Open the best available non-blocking sink.

    ``preference`` may name a specific backend, ``"null"`` to stay silent, or
    ``"auto"`` to try PortAudio first and fall back to the platform player.
    """
    choice = (preference or "auto").strip().lower()
    if choice != "auto":
        factory = SINK_FACTORIES.get(choice)
        if factory is None:
            raise AudioError(
                f"unknown audio backend {choice!r}; "
                f"expected one of {sorted(SINK_FACTORIES)} or 'auto'"
            )
        return factory()

    order: list[SinkFactory] = [SoundDeviceSink]
    if platform.system() == "Windows":
        order.append(WinsoundSink)
    order.append(CommandSink)
    failures: list[str] = []
    for factory in order:
        try:
            return factory()
        except AudioError as exc:
            failures.append(str(exc))
    raise AudioError("; ".join(failures) or "no audio backend available")


def default_sink_preference() -> str:
    """Backend name from ``AIFACE_AUDIO_BACKEND``, else automatic selection."""
    return os.environ.get("AIFACE_AUDIO_BACKEND", "auto")


__all__ = [
    "COMMAND_LATENCY",
    "DEFAULT_HOP_SECONDS",
    "DEFAULT_MIN_SILENCE",
    "DEFAULT_MIN_VOICED",
    "DEFAULT_VOICE_THRESHOLD",
    "DEFAULT_WINDOW_SECONDS",
    "SINK_FACTORIES",
    "SOUNDDEVICE_LATENCY",
    "WINSOUND_LATENCY",
    "AudioClip",
    "AudioError",
    "AudioSink",
    "CommandSink",
    "Envelope",
    "NullSink",
    "SoundDeviceSink",
    "WinsoundSink",
    "decode_audio",
    "decode_wav",
    "default_sink_preference",
    "encode_wav",
    "energy_warp",
    "open_audio_sink",
    "rms_envelope",
    "time_stretch",
    "voiced_intervals",
]
