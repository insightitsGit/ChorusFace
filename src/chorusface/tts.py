"""Text to speech, and lips locked to the audio it produces.

Letter-count heuristics can only guess at rhythm. Once a real waveform exists
the timing is no longer a guess, so this module owns two jobs:

**Synthesis.** :class:`OpenAISpeechVoice` calls an OpenAI-compatible
``/audio/speech`` endpoint; :class:`SapiVoice` uses Windows Speech API;
:class:`CommandVoice` runs a local synthesiser that writes WAVE to stdout
(``espeak-ng``, ``piper``, anything with that contract). All return a decoded
:class:`~chorusface.audio.AudioClip`.

**Alignment.** Three strategies, in descending fidelity:

``words``
    Ask a transcription model for word-level timestamps on the synthesised
    audio, match the heard words back onto the written script with a monotone
    diff, and distribute each word's visemes inside its measured span. This is
    forced alignment using the recogniser as the aligner.
``energy``
    Warp the script onto the clip by cumulative short-time energy. Silence
    costs no energy, so pauses, onsets, and the trailing breath land correctly
    without any transcription round trip. This is the default.
``linear``
    Stretch the script uniformly across the clip duration. The floor, used when
    a clip has no measurable energy structure at all.

Every strategy returns absolute :class:`PhonemeSpan` times measured from the
first audio sample, which :func:`chorusface.speech.schedule_spans` turns into
muscle events once playback actually starts.
"""

from __future__ import annotations

import difflib
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Final, Protocol, Sequence

from chorusface.audio import (
    AudioClip,
    AudioError,
    Envelope,
    decode_audio,
    energy_warp,
    rms_envelope,
    voiced_intervals,
)
from chorusface.speech import (
    DEFAULT_BASE_URL,
    PHONEME_DURATION_SCALE,
    PhonemeSpan,
    SpokenToken,
    strip_tags,
    tokenize_speech,
)

DEFAULT_SPEECH_MODEL: Final = "gpt-4o-mini-tts"
DEFAULT_SPEECH_VOICE: Final = "alloy"
DEFAULT_TRANSCRIBE_MODEL: Final = "whisper-1"
REQUEST_TIMEOUT: Final = 60.0
COMMAND_TIMEOUT: Final = 60.0
MAX_SPEECH_CHARACTERS: Final = 1200

ALIGN_WORDS: Final = "words"
ALIGN_ENERGY: Final = "energy"
ALIGN_LINEAR: Final = "linear"
ALIGNMENTS: Final = (ALIGN_WORDS, ALIGN_ENERGY, ALIGN_LINEAR)

#: Shortest viseme the solver is asked to hold. Below this the impulse queue
#: cannot express the shape and the mouth only flickers.
MIN_SPAN: Final = 0.035

#: How much of the alignment follows acoustic energy rather than the clock. The
#: streaming channel blends by the same figure, so the two paths agree on rhythm
#: and any difference between them is timing rather than taste.
DEFAULT_WARP_STRENGTH: Final = 0.65

#: Silence at least this long is heard as punctuation rather than as the closure
#: inside a stop consonant.
DEFAULT_PHRASE_SILENCE: Final = 0.12

#: Local synthesisers that write WAVE to stdout and read text from stdin.
LOCAL_VOICE_COMMANDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("espeak-ng", ("--stdout", "-s", "165")),
    ("espeak", ("--stdout", "-s", "165")),
)

POWERSHELL_CANDIDATES: Final[tuple[str, ...]] = (
    "powershell",
    "pwsh",
    "WindowsPowerShell\\v1.0\\powershell.exe",
)


class TTSError(RuntimeError):
    """Synthesis or alignment failed. The caller falls back to text timing."""


@dataclass(frozen=True, slots=True)
class WordSpan:
    """One measured word in the audio."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class PreparedSpeech:
    """A synthesised line plus the viseme timing measured from its audio."""

    text: str
    clip: AudioClip
    spans: tuple[PhonemeSpan, ...]
    words: tuple[WordSpan, ...] = ()
    voice: str = ""
    alignment: str = ALIGN_ENERGY

    @property
    def duration(self) -> float:
        return self.clip.duration

    def span_tuples(self) -> list[tuple[str, float, float]]:
        return [span.as_tuple() for span in self.spans]


def apply_speech_pace(
    speech: PreparedSpeech,
    pace: float,
    *,
    min_hold: float = 0.0,
) -> PreparedSpeech:
    """Stretch audio + viseme spans together so lips stay locked to the voice.

    ``pace`` 1.0 = realtime; 1.12 ≈ +12% duration (clearer mouth holds).
    ``min_hold`` expands any span shorter than that after pacing (seconds).
    """
    from chorusface.audio import time_stretch

    pace = max(0.85, min(1.60, float(pace)))
    min_hold = max(0.0, float(min_hold))
    if abs(pace - 1.0) < 1e-4 and min_hold <= 1e-6:
        return speech

    clip = time_stretch(speech.clip, pace)
    spans: list[PhonemeSpan] = [
        PhonemeSpan(
            span.phoneme,
            float(span.start) * pace,
            float(span.end) * pace,
        )
        for span in speech.spans
    ]
    # Enforce min_hold by pushing later spans forward (do not clip into next).
    if min_hold > 0.0 and spans:
        shift = 0.0
        for i, span in enumerate(spans):
            start = float(span.start) + shift
            end = float(span.end) + shift
            need = min_hold - (end - start)
            if need > 0.0:
                end += need
                shift += need
            spans[i] = PhonemeSpan(span.phoneme, start, end)
        # Pad silence if holds pushed past the paced clip.
        limit = float(spans[-1].end)
        if limit > float(clip.duration) + 1e-6:
            import numpy as np

            n_pad = int(round((limit - float(clip.duration)) * clip.sample_rate))
            if n_pad > 0:
                pad = np.zeros(n_pad, dtype=np.float32)
                samples = np.concatenate(
                    [np.asarray(clip.samples, dtype=np.float32), pad]
                )
                clip = AudioClip(samples=samples, sample_rate=int(clip.sample_rate))
    words = tuple(
        WordSpan(w.text, float(w.start) * pace, float(w.end) * pace)
        for w in speech.words
    )
    return PreparedSpeech(
        text=speech.text,
        clip=clip,
        spans=tuple(spans),
        words=words,
        voice=speech.voice,
        alignment=speech.alignment,
    )


# ------------------------------------------------------------------ alignment


def _viseme_weights(token: SpokenToken) -> list[float]:
    return [
        max(PHONEME_DURATION_SCALE.get(name, 1.0), 0.05) for name in token.visemes
    ]


#: Bilabial / closed-lip onset pin (seconds). Energy warp otherwise buries
#: PP inside the following vowel so "Peter/picked/pepper" never seals.
BILABIAL_ONSET_PIN: Final = 0.045
BILABIAL_VISEMES: Final = frozenset({"PP", "MM", "CLOSED"})


def _subdivide(
    token: SpokenToken, start: float, end: float
) -> list[PhonemeSpan]:
    """Split one token's window among its visemes by articulation weight."""
    weights = _viseme_weights(token)
    total = sum(weights)
    window = max(end - start, 0.0)
    spans: list[PhonemeSpan] = []
    cursor = start
    names = list(token.visemes)
    # Pin leading bilabial to word onset so P/B/M read before the vowel.
    if names and names[0] in BILABIAL_VISEMES and window > BILABIAL_ONSET_PIN + 0.02:
        pin = min(BILABIAL_ONSET_PIN, window * 0.45)
        spans.append(PhonemeSpan(names[0], start, start + pin))
        cursor = start + pin
        names = names[1:]
        weights = weights[1:]
        total = sum(weights)
        window = max(end - cursor, 0.0)
    for name, weight in zip(names, weights):
        share = window * (weight / total) if total > 0.0 else 0.0
        spans.append(PhonemeSpan(name, cursor, cursor + share))
        cursor += share
    if spans:
        spans[-1] = PhonemeSpan(spans[-1].phoneme, spans[-1].start, end)
    return spans


def bias_bilabial_onsets(
    spans: Sequence[PhonemeSpan],
    *,
    pin: float = BILABIAL_ONSET_PIN,
) -> list[PhonemeSpan]:
    """Ensure PP/MM/CLOSED spans keep a readable onset before the next vowel.

    Applied after energy warp: if a bilabial is shorter than ``pin`` and the
    following span is a vowel/open shape, borrow time from the follower
    without shifting later absolute starts of unrelated spans.
    """
    if not spans:
        return []
    out = [PhonemeSpan(s.phoneme, float(s.start), float(s.end)) for s in spans]
    for i, span in enumerate(out):
        if span.phoneme not in BILABIAL_VISEMES:
            continue
        width = span.end - span.start
        if width >= pin - 1e-6:
            continue
        if i + 1 >= len(out):
            continue
        nxt = out[i + 1]
        if nxt.phoneme in BILABIAL_VISEMES:
            continue
        need = pin - width
        borrow = min(need, max(0.0, (nxt.end - nxt.start) - MIN_SPAN))
        if borrow <= 1e-6:
            continue
        new_end = span.end + borrow
        out[i] = PhonemeSpan(span.phoneme, span.start, new_end)
        out[i + 1] = PhonemeSpan(nxt.phoneme, new_end, nxt.end)
    return out


def snap_bilabials_to_energy_valleys(
    spans: Sequence[PhonemeSpan],
    envelope: Envelope,
    *,
    search: float = 0.06,
) -> list[PhonemeSpan]:
    """Pull PP/MM/CLOSED starts toward a nearby energy trough (closure cue)."""
    if not spans or envelope.frame_count <= 2:
        return list(spans)
    energies = [float(v) for v in envelope.values]
    hop = float(envelope.hop)
    if not energies or hop <= 0.0:
        return list(spans)
    out = [PhonemeSpan(s.phoneme, float(s.start), float(s.end)) for s in spans]
    for i, span in enumerate(out):
        if span.phoneme not in BILABIAL_VISEMES:
            continue
        center = 0.5 * (span.start + span.end)
        lo = max(0.0, center - search)
        hi = center + search
        i0 = max(0, int(lo / hop))
        i1 = min(len(energies) - 1, int(hi / hop))
        if i1 <= i0:
            continue
        window = energies[i0 : i1 + 1]
        valley_i = i0 + int(min(range(len(window)), key=lambda j: window[j]))
        valley_t = valley_i * hop
        width = max(span.end - span.start, MIN_SPAN)
        new_start = max(0.0, valley_t - 0.5 * width)
        new_end = new_start + width
        if i > 0:
            prev = out[i - 1]
            new_start = max(new_start, float(prev.start) + MIN_SPAN)
            new_end = new_start + width
        if i + 1 < len(out):
            nxt = out[i + 1]
            new_end = min(new_end, float(nxt.end) - MIN_SPAN)
            new_start = new_end - width
            if i > 0:
                new_start = max(new_start, float(out[i - 1].start) + MIN_SPAN)
                new_end = new_start + width
        if new_end > new_start + 1e-4:
            out[i] = PhonemeSpan(span.phoneme, new_start, new_end)
    return out


def _enforce_minimum(spans: Sequence[PhonemeSpan], limit: float) -> list[PhonemeSpan]:
    """Give every surviving span a holdable length.

    A span with no width at all is dropped rather than inflated: it means the
    measured audio left no room for that sound, and stretching it would push the
    lips onto a shape the voice never made.
    """
    result: list[PhonemeSpan] = []
    for span in spans:
        if span.end <= span.start:
            continue
        end = max(span.end, span.start + MIN_SPAN)
        result.append(PhonemeSpan(span.phoneme, span.start, min(end, limit)))
    return bias_bilabial_onsets(result)


def _flatten(tokens: Sequence[SpokenToken]) -> tuple[list[str], list[float]]:
    """Flatten tokens into a viseme stream and its articulation weights."""
    names: list[str] = []
    weights: list[float] = []
    for token in tokens:
        for name, weight in zip(token.visemes, _viseme_weights(token)):
            names.append(name)
            weights.append(weight)
    return names, weights


def _warp_within(
    tokens: Sequence[SpokenToken],
    envelope: Envelope,
    *,
    start: float,
    end: float,
    strength: float,
) -> list[PhonemeSpan]:
    """Distribute a run of tokens across ``[start, end]`` by acoustic energy."""
    names, weights = _flatten(tokens)
    if not names:
        return []
    total = sum(weights)
    boundaries = [0.0]
    running = 0.0
    for weight in weights:
        running += weight
        boundaries.append(running / total)

    window = max(end - start, 0.0)
    segment = envelope.segment(start, end)
    if segment.frame_count == 0 or segment.duration <= 0.0 or window <= 0.0:
        times = [start + window * fraction for fraction in boundaries]
    else:
        # The slice snaps to hop boundaries, so rescale it back onto the window.
        scale = window / segment.duration
        warped = energy_warp(segment, boundaries, strength=strength)
        times = [start + value * scale for value in warped]
    times[0] = start
    times[-1] = end
    return [
        PhonemeSpan(name, times[index], times[index + 1])
        for index, name in enumerate(names)
    ]


def phrase_blocks(tokens: Sequence[SpokenToken]) -> list[tuple[bool, int, int]]:
    """Split a script into alternating phrase and break blocks.

    A phrase runs from one piece of punctuation to the next — spaces hold words
    together inside it, because fluent speech has no reliable silence between
    words, while a comma or a full stop really does interrupt the voice.
    Returns ``(is_phrase, begin, end)`` covering every token exactly once.
    """
    blocks: list[tuple[bool, int, int]] = []
    index = 0
    total = len(tokens)
    while index < total:
        start = index
        is_break = tokens[index].is_break
        while index < total and tokens[index].is_break == is_break:
            index += 1
        has_word = any(token.is_word for token in tokens[start:index])
        blocks.append((not is_break and has_word, start, index))
    return blocks


def silence_gaps(
    voiced: Sequence[tuple[float, float]], *, minimum: float
) -> list[tuple[float, float]]:
    """Silences long enough to be heard as punctuation rather than a stop consonant."""
    return [
        (float(left[1]), float(right[0]))
        for left, right in zip(voiced, voiced[1:])
        if right[0] - left[1] >= minimum
    ]


def align_by_energy(
    tokens: Sequence[SpokenToken],
    envelope: Envelope,
    *,
    duration: float,
    strength: float = DEFAULT_WARP_STRENGTH,
    phrase_silence: float = DEFAULT_PHRASE_SILENCE,
) -> list[PhonemeSpan]:
    """Warp the script onto the clip so the visemes follow acoustic energy.

    Two measurements do the work. The voiced extent of the clip bounds the
    script, so leading silence and the trailing breath are never articulated.
    Then, when the script's punctuation count matches the number of long
    silences inside that extent, each phrase is pinned to its own stretch of
    audio and the pauses take the measured gaps — word-timestamp quality with no
    recogniser. Inside every stretch, cumulative energy decides the rhythm.
    """
    if not tokens:
        return []
    voiced = voiced_intervals(envelope)
    if not voiced:
        return align_linear(tokens, duration=duration)

    blocks = phrase_blocks(tokens)
    phrases = [block for block in blocks if block[0]]
    if not phrases:
        return _enforce_minimum(
            _proportional(tokens, start=0.0, end=duration), duration
        )

    speech_start, speech_end = voiced[0][0], voiced[-1][1]
    gaps = silence_gaps(voiced, minimum=phrase_silence)
    if len(gaps) + 1 != len(phrases):
        # Punctuation and audible pauses disagree; time the script as one phrase.
        gaps = []
        phrases = [(True, phrases[0][1], phrases[-1][2])]

    windows: list[tuple[float, float]] = []
    cursor = speech_start
    for gap in gaps:
        windows.append((cursor, gap[0]))
        cursor = gap[1]
    windows.append((cursor, speech_end))

    spans: list[PhonemeSpan] = []
    previous_end = 0.0
    previous_token = 0
    for (_is_phrase, begin, end), (window_start, window_end) in zip(phrases, windows):
        start = max(window_start, previous_end)
        stop = max(window_end, start)
        spans.extend(
            _proportional(tokens[previous_token:begin], start=previous_end, end=start)
        )
        spans.extend(
            _warp_within(
                tokens[begin:end],
                envelope,
                start=start,
                end=stop,
                strength=strength,
            )
        )
        previous_end = stop
        previous_token = end
    spans.extend(
            _proportional(tokens[previous_token:], start=previous_end, end=duration)
    )
    spans = _enforce_minimum(spans, duration)
    return snap_bilabials_to_energy_valleys(spans, envelope)


def _proportional(
    tokens: Sequence[SpokenToken], *, start: float, end: float
) -> list[PhonemeSpan]:
    """Share a window between tokens by how long they take to say."""
    if not tokens:
        return []
    total = sum(token.nominal_duration for token in tokens) or 1.0
    window = max(end - start, 0.0)
    spans: list[PhonemeSpan] = []
    cursor = start
    for token in tokens:
        share = window * (token.nominal_duration / total)
        spans.extend(_subdivide(token, cursor, cursor + share))
        cursor += share
    return spans


def align_linear(
    tokens: Sequence[SpokenToken], *, duration: float
) -> list[PhonemeSpan]:
    """Stretch the script uniformly across the clip."""
    names, weights = _flatten(tokens)
    if not names:
        return []
    total = sum(weights)
    spans: list[PhonemeSpan] = []
    cursor = 0.0
    for name, weight in zip(names, weights):
        share = duration * (weight / total)
        spans.append(PhonemeSpan(name, cursor, cursor + share))
        cursor += share
    return _enforce_minimum(spans, duration)


def _normalise_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", word.lower())


def match_word_spans(
    tokens: Sequence[SpokenToken], words: Sequence[WordSpan]
) -> dict[int, WordSpan]:
    """Map script token indices onto measured words, monotonically.

    A recogniser drops, merges, and rewrites words. A longest-matching-blocks
    diff keeps every pair it is sure about and leaves the rest unanchored for
    interpolation, which is safer than forcing a one-to-one mapping that slides
    the whole utterance out of sync after a single mismatch.
    """
    script_indices = [
        index for index, token in enumerate(tokens) if token.is_word
    ]
    script = [_normalise_word(tokens[index].text) for index in script_indices]
    heard = [_normalise_word(word.text) for word in words]
    if not script or not heard:
        return {}

    matcher = difflib.SequenceMatcher(None, script, heard, autojunk=False)
    anchors: dict[int, WordSpan] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            token_index = script_indices[block.a + offset]
            anchors[token_index] = words[block.b + offset]
    return anchors


def align_by_words(
    tokens: Sequence[SpokenToken],
    words: Sequence[WordSpan],
    *,
    duration: float,
) -> list[PhonemeSpan]:
    """Place visemes inside measured word spans, interpolating the gaps."""
    anchors = match_word_spans(tokens, words)
    if not anchors:
        raise TTSError("no script word matched the transcript")

    windows: list[tuple[float, float] | None] = [None] * len(tokens)
    for index, span in anchors.items():
        start = max(0.0, min(float(span.start), duration))
        end = max(start, min(float(span.end), duration))
        windows[index] = (start, end)

    anchored = sorted(anchors)
    # Unanchored runs share whatever time sits between their neighbours,
    # proportionally to how long they take to say.
    segments: list[tuple[int, int, float, float]] = []
    first, last = anchored[0], anchored[-1]
    segments.append((0, first, 0.0, windows[first][0]))  # type: ignore[index]
    for left, right in zip(anchored, anchored[1:]):
        segments.append(
            (left + 1, right, windows[left][1], windows[right][0])  # type: ignore[index]
        )
    segments.append((last + 1, len(tokens), windows[last][1], duration))  # type: ignore[index]

    for begin, stop, region_start, region_end in segments:
        pending = list(range(begin, stop))
        if not pending:
            continue
        span_total = sum(tokens[index].nominal_duration for index in pending) or 1.0
        available = max(region_end - region_start, 0.0)
        cursor = region_start
        for index in pending:
            share = available * (tokens[index].nominal_duration / span_total)
            windows[index] = (cursor, cursor + share)
            cursor += share

    spans: list[PhonemeSpan] = []
    previous_end = 0.0
    for token, window in zip(tokens, windows):
        start, end = window if window is not None else (previous_end, previous_end)
        start = max(start, previous_end)
        end = max(end, start)
        spans.extend(_subdivide(token, start, end))
        previous_end = end
    return _enforce_minimum(spans, duration)


def align_text(
    text: str,
    clip: AudioClip,
    *,
    alignment: str = ALIGN_ENERGY,
    words: Sequence[WordSpan] = (),
    warp_strength: float = DEFAULT_WARP_STRENGTH,
) -> tuple[list[PhonemeSpan], str]:
    """Time the visemes of ``text`` against ``clip``.

    Returns the spans and the strategy that actually produced them, which may
    be weaker than the one requested when the audio or the transcript does not
    support it.
    """
    tokens = tokenize_speech(text)
    duration = clip.duration
    if not tokens or duration <= 0.0:
        return [], ALIGN_LINEAR

    if alignment == ALIGN_WORDS and words:
        try:
            return align_by_words(tokens, words, duration=duration), ALIGN_WORDS
        except TTSError:
            alignment = ALIGN_ENERGY

    if alignment != ALIGN_LINEAR:
        envelope = rms_envelope(clip)
        if envelope.peak > 0.0:
            spans = align_by_energy(
                tokens, envelope, duration=duration, strength=warp_strength
            )
            return spans, ALIGN_ENERGY

    return align_linear(tokens, duration=duration), ALIGN_LINEAR


# --------------------------------------------------------------------- voices


class TTSVoice(Protocol):
    """Anything that can turn a line of text into audio."""

    name: str

    def synthesize(self, text: str) -> AudioClip: ...


def _http_post(
    url: str,
    *,
    data: bytes,
    headers: dict[str, str],
    timeout: float = REQUEST_TIMEOUT,
) -> bytes:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:400].decode("utf-8", "replace")
        raise TTSError(f"{url} returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TTSError(f"{url} unreachable: {exc}") from exc


def _multipart_body(
    fields: Sequence[tuple[str, str]],
    *,
    file_field: str,
    file_name: str,
    file_bytes: bytes,
    content_type: str = "audio/wav",
) -> tuple[bytes, str]:
    boundary = f"----chorusface{secrets.token_hex(16)}"
    parts: list[bytes] = []
    for name, value in fields:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}";'
        f' filename="{file_name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


@dataclass(slots=True)
class OpenAISpeechVoice:
    """OpenAI-compatible ``/audio/speech`` synthesiser.

    WAVE is requested explicitly: a compressed container would have to be
    decoded before it could be measured, and a mis-decode desynchronises the
    face far more visibly than it would a plain audio player.
    """

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_SPEECH_MODEL
    voice: str = DEFAULT_SPEECH_VOICE
    speed: float = 1.0
    instructions: str = ""
    name: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not self.api_key:
            raise TTSError("an API key is required for hosted speech synthesis")
        self.name = f"openai:{self.model}/{self.voice}"

    def synthesize(self, text: str) -> AudioClip:
        spoken = strip_tags(text).strip()[:MAX_SPEECH_CHARACTERS]
        if not spoken:
            raise TTSError("nothing to speak")
        body: dict[str, object] = {
            "model": self.model,
            "voice": self.voice,
            "input": spoken,
            "response_format": "wav",
        }
        if abs(self.speed - 1.0) > 1e-3:
            body["speed"] = float(self.speed)
        if self.instructions:
            body["instructions"] = self.instructions
        payload = _http_post(
            self.base_url.rstrip("/") + "/audio/speech",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            return decode_audio(payload)
        except AudioError as exc:
            raise TTSError(f"could not decode synthesised audio: {exc}") from exc


@dataclass(slots=True)
class CommandVoice:
    """Local synthesiser invoked as a subprocess.

    The contract is deliberately narrow: text arrives on stdin, WAVE leaves on
    stdout. ``espeak-ng --stdout`` and ``piper --output_file -`` both satisfy
    it, so an offline face needs no network and no API key.
    """

    command: Sequence[str]
    name: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not self.command:
            raise TTSError("a synthesiser command is required")
        self.command = tuple(str(part) for part in self.command)
        self.name = f"command:{self.command[0]}"

    @classmethod
    def parse(cls, command_line: str) -> "CommandVoice":
        # Non-POSIX splitting keeps Windows backslashes intact, at the price of
        # leaving the quote characters on the token, so strip them back off.
        parts = [
            part[1:-1]
            if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'"
            else part
            for part in shlex.split(command_line, posix=False)
        ]
        return cls([part for part in parts if part])

    @classmethod
    def autodetect(cls) -> "CommandVoice":
        for executable, arguments in LOCAL_VOICE_COMMANDS:
            found = shutil.which(executable)
            if found:
                return cls((found, *arguments))
        raise TTSError(
            "no local synthesiser found; install espeak-ng or pass --tts-command"
        )

    def synthesize(self, text: str) -> AudioClip:
        spoken = strip_tags(text).strip()[:MAX_SPEECH_CHARACTERS]
        if not spoken:
            raise TTSError("nothing to speak")
        try:
            completed = subprocess.run(
                list(self.command),
                input=spoken.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=COMMAND_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TTSError(f"{self.command[0]} failed: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr[:400].decode("utf-8", "replace").strip()
            raise TTSError(
                f"{self.command[0]} exited {completed.returncode}: {detail}"
            )
        try:
            return decode_audio(completed.stdout)
        except AudioError as exc:
            raise TTSError(f"{self.command[0]} produced no usable wav: {exc}") from exc


def _powershell_executable() -> str | None:
    for candidate in POWERSHELL_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _sapi_script_path() -> Path:
    """Resolve the packaged PowerShell SAPI driver."""
    # Prefer the file beside this module — works for editable and wheel installs
    # that unpack package data onto disk.
    beside = Path(__file__).resolve().parent / "data" / "sapi_tts.ps1"
    if beside.is_file():
        return beside
    try:
        root = resources.files("chorusface")
        candidate = root.joinpath("data", "sapi_tts.ps1")
        if candidate.is_file():
            return Path(str(candidate))
    except (FileNotFoundError, ModuleNotFoundError, TypeError, AttributeError):
        pass
    raise TTSError("packaged sapi_tts.ps1 is missing")


@dataclass(slots=True)
class SapiVoice:
    """Windows Speech API voice via the packaged PowerShell driver.

    No extra pip dependency: System.Speech ships with Windows. The script writes
    a complete WAVE container to stdout so energy alignment can measure it.
    """

    rate: int | None = None
    voice: str = ""
    name: str = field(init=False, default="sapi")
    _shell: str = field(init=False, repr=False)
    _script: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if platform.system() != "Windows":
            raise TTSError("SAPI is only available on Windows")
        shell = _powershell_executable()
        if shell is None:
            raise TTSError("PowerShell is required for SAPI synthesis")
        self._shell = shell
        self._script = _sapi_script_path()
        self.name = "sapi:windows"

    def synthesize(self, text: str) -> AudioClip:
        spoken = strip_tags(text).strip()[:MAX_SPEECH_CHARACTERS]
        if not spoken:
            raise TTSError("nothing to speak")
        env = os.environ.copy()
        if self.rate is not None:
            env["CHORUSFACE_SAPI_RATE"] = str(int(self.rate))
        if self.voice:
            env["CHORUSFACE_SAPI_VOICE"] = self.voice
        try:
            completed = subprocess.run(
                [
                    self._shell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(self._script),
                ],
                input=spoken.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=COMMAND_TIMEOUT,
                check=False,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TTSError(f"SAPI synthesis failed: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr[:400].decode("utf-8", "replace").strip()
            raise TTSError(f"SAPI exited {completed.returncode}: {detail}")
        try:
            return decode_audio(completed.stdout)
        except AudioError as exc:
            raise TTSError(f"SAPI produced no usable wav: {exc}") from exc


def detect_local_voice() -> TTSVoice:
    """Pick the best offline synthesiser available on this machine."""
    failures: list[str] = []
    if platform.system() == "Windows":
        try:
            return SapiVoice()
        except TTSError as exc:
            failures.append(str(exc))
    try:
        return CommandVoice.autodetect()
    except TTSError as exc:
        failures.append(str(exc))
    raise TTSError(
        "; ".join(failures)
        or "no local synthesiser found; install espeak-ng or pass --tts-command"
    )


def local_voice_available() -> bool:
    """True when offline TTS can run without an API key."""
    try:
        detect_local_voice()
    except TTSError:
        return False
    return True


@dataclass(slots=True)
class WhisperAligner:
    """Word timestamps from an OpenAI-compatible transcription endpoint.

    Transcribing our own synthesised audio is forced alignment with a model we
    already have access to: the recogniser reports when each word was actually
    spoken, and the script is matched back onto those spans.
    """

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_TRANSCRIBE_MODEL
    language: str = "en"

    def word_spans(self, clip: AudioClip, *, prompt: str = "") -> list[WordSpan]:
        if not self.api_key:
            raise TTSError("an API key is required for word timestamps")
        fields: list[tuple[str, str]] = [
            ("model", self.model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
        ]
        if self.language:
            fields.append(("language", self.language))
        if prompt:
            fields.append(("prompt", strip_tags(prompt)[:900]))
        body, content_type = _multipart_body(
            fields,
            file_field="file",
            file_name="speech.wav",
            file_bytes=clip.to_wav_bytes(),
        )
        payload = _http_post(
            self.base_url.rstrip("/") + "/audio/transcriptions",
            data=body,
            headers={
                "Content-Type": content_type,
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        return parse_word_spans(payload)


def parse_word_spans(payload: bytes | str) -> list[WordSpan]:
    """Read ``verbose_json`` word timings, tolerating provider differences."""
    try:
        document = json.loads(
            payload.decode("utf-8") if isinstance(payload, bytes) else payload
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TTSError(f"transcription response was not json: {exc}") from exc
    if not isinstance(document, dict):
        raise TTSError("transcription response was not a json object")

    raw = document.get("words")
    if not isinstance(raw, list) or not raw:
        # Some providers only expose word timings nested inside segments.
        raw = [
            word
            for segment in document.get("segments", [])
            if isinstance(segment, dict)
            for word in segment.get("words", [])
            if isinstance(word, dict)
        ]
    spans: list[WordSpan] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("word") or entry.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(entry["start"])
            end = float(entry["end"])
        except (KeyError, TypeError, ValueError):
            continue
        spans.append(WordSpan(text, start, max(end, start)))
    if not spans:
        raise TTSError("transcription carried no word timestamps")
    return spans


# ---------------------------------------------------------------- the speaker


@dataclass(slots=True)
class SpeechSynthesizer:
    """A voice plus an alignment strategy: text in, timed speech out."""

    voice: TTSVoice
    alignment: str = ALIGN_ENERGY
    aligner: WhisperAligner | None = None
    warp_strength: float = DEFAULT_WARP_STRENGTH

    def __post_init__(self) -> None:
        if self.alignment not in ALIGNMENTS:
            raise TTSError(
                f"unknown alignment {self.alignment!r}; expected one of {ALIGNMENTS}"
            )
        if self.alignment == ALIGN_WORDS and self.aligner is None:
            raise TTSError("word alignment needs a transcription aligner")

    @property
    def description(self) -> str:
        return f"{self.voice.name} ({self.alignment} alignment)"

    def prepare(self, text: str) -> PreparedSpeech:
        """Synthesise ``text`` and measure when each viseme is due.

        Raises :class:`TTSError` when synthesis fails; alignment itself always
        degrades to a weaker strategy rather than failing, so audio that plays
        always has lips to go with it.
        """
        clip = self.voice.synthesize(text)
        if clip.frame_count == 0:
            raise TTSError("synthesiser returned an empty clip")

        words: list[WordSpan] = []
        requested = self.alignment
        if requested == ALIGN_WORDS and self.aligner is not None:
            try:
                words = self.aligner.word_spans(clip, prompt=text)
            except TTSError:
                requested = ALIGN_ENERGY

        spans, used = align_text(
            text,
            clip,
            alignment=requested,
            words=words,
            warp_strength=self.warp_strength,
        )
        return PreparedSpeech(
            text=text,
            clip=clip,
            spans=tuple(spans),
            words=tuple(words),
            voice=self.voice.name,
            alignment=used,
        )


def build_synthesizer(
    *,
    backend: str = "auto",
    api_key: str = "",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_SPEECH_MODEL,
    voice: str = DEFAULT_SPEECH_VOICE,
    speed: float = 1.0,
    instructions: str = "",
    command: str = "",
    alignment: str = ALIGN_ENERGY,
    transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL,
    warp_strength: float = DEFAULT_WARP_STRENGTH,
) -> SpeechSynthesizer:
    """Resolve CLI-style settings into a working synthesiser.

    ``backend`` is ``openai``, ``sapi``, ``command``, or ``auto`` — hosted when
    a key is present, otherwise the best local voice (Windows SAPI, then
    espeak-ng), so the face can speak with or without an account.
    """
    choice = (backend or "auto").strip().lower()
    if choice == "auto":
        if command:
            choice = "command"
        elif api_key:
            choice = "openai"
        else:
            choice = "local"

    engine: TTSVoice
    if choice == "openai":
        engine = OpenAISpeechVoice(
            api_key=api_key,
            base_url=base_url,
            model=model,
            voice=voice,
            speed=speed,
            instructions=instructions,
        )
    elif choice == "sapi":
        engine = SapiVoice(voice=voice if voice and voice != DEFAULT_SPEECH_VOICE else "")
    elif choice == "command":
        engine = CommandVoice.parse(command) if command else CommandVoice.autodetect()
    elif choice == "local":
        engine = detect_local_voice()
    else:
        raise TTSError(f"unknown tts backend {backend!r}")

    wanted = (alignment or ALIGN_ENERGY).strip().lower()
    if wanted not in ALIGNMENTS:
        raise TTSError(f"unknown alignment {alignment!r}; expected one of {ALIGNMENTS}")
    aligner: WhisperAligner | None = None
    if wanted == ALIGN_WORDS:
        if not api_key:
            # Word timestamps need a recogniser; energy needs nothing.
            wanted = ALIGN_ENERGY
        else:
            aligner = WhisperAligner(
                api_key=api_key, base_url=base_url, model=transcribe_model
            )
    return SpeechSynthesizer(
        voice=engine,
        alignment=wanted,
        aligner=aligner,
        warp_strength=warp_strength,
    )


__all__ = [
    "ALIGNMENTS",
    "ALIGN_ENERGY",
    "ALIGN_LINEAR",
    "ALIGN_WORDS",
    "DEFAULT_PHRASE_SILENCE",
    "DEFAULT_SPEECH_MODEL",
    "DEFAULT_SPEECH_VOICE",
    "DEFAULT_TRANSCRIBE_MODEL",
    "DEFAULT_WARP_STRENGTH",
    "LOCAL_VOICE_COMMANDS",
    "MIN_SPAN",
    "CommandVoice",
    "OpenAISpeechVoice",
    "PhonemeSpan",
    "PreparedSpeech",
    "SapiVoice",
    "SpeechSynthesizer",
    "TTSError",
    "TTSVoice",
    "WhisperAligner",
    "WordSpan",
    "align_by_energy",
    "align_by_words",
    "align_linear",
    "align_text",
    "apply_speech_pace",
    "bias_bilabial_onsets",
    "snap_bilabials_to_energy_valleys",
    "build_synthesizer",
    "detect_local_voice",
    "local_voice_available",
    "match_word_spans",
    "parse_word_spans",
    "phrase_blocks",
    "silence_gaps",
]
