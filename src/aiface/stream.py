"""Lips locked to audio that is *arriving*, not audio we already hold.

A realtime voice model hands you PCM in 20 ms chunks and a transcript alongside
it. Nobody hands you a face that moves with it. This module is that channel: push
audio in, get timed viseme decisions out, with a stated and measurable delay.

**What each side knows.** The text says *what* is being said — the viseme
sequence comes from :func:`aiface.speech.tokenize_speech`, the same phonetic
source of truth the offline path uses. The audio says *when*. Nothing here
pretends to recognise phonemes from a waveform.

**The online algorithm.** The offline aligner warps a script onto a clip by
cumulative energy, which it can do because it knows the clip's total energy in
advance. Streaming has no such luxury, so the same idea runs incrementally: each
analysed frame spends its energy against the articulation budget of the viseme
currently being held, and the next viseme is emitted the moment that budget runs
out. Loud frames advance the script quickly, quiet frames slowly, and silence
does not advance it at all — which is what keeps pauses from eating syllables.
The speaker's level is estimated as it goes, so the budget is in units of
"seconds of normally-loud speech" rather than raw amplitude.

**The cost, stated honestly.** A viseme's onset cannot be known before the audio
carrying it has arrived, so every decision lags by at most one hop plus the
configured lookahead. Each emitted span records the moment it was decided, so
that lag is a measurement rather than a claim. :mod:`aiface.sync` runs the same
utterance through this path and the offline path and reports the difference in
milliseconds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final, Iterable

import numpy as np

from aiface.speech import (
    PHONEME_DURATION_SCALE,
    PHONEME_IMPULSES,
    PhonemeSpan,
    VisemeEvent,
    canonical_viseme,
    phoneme_hold,
    schedule_spans,
    tokenize_speech,
)

#: What the realtime speech APIs emit, so it is the default here too.
DEFAULT_STREAM_RATE: Final = 24_000
#: Chunk size those APIs deliver, and what the oracle simulates by default.
DEFAULT_CHUNK_SECONDS: Final = 0.020

DEFAULT_HOP: Final = 0.010
DEFAULT_WINDOW: Final = 0.025
#: Audio we wait for beyond a frame's own window before trusting it. This is the
#: whole latency knob: zero reacts instantly and jitters, more is smoother.
DEFAULT_LOOKAHEAD: Final = 0.050
#: Speech one unit-weight viseme is expected to occupy. Only a prior: the
#: channel recalibrates it from the voice at every phrase boundary.
DEFAULT_SECONDS_PER_VISEME: Final = 0.09
#: Fraction of the running level estimate that still counts as voice. Tuned on
#: real synthesised speech with :mod:`aiface.sync`, not chosen by taste.
DEFAULT_GATE_RATIO: Final = 0.28
#: Absolute RMS below which audio is silence whatever the gate says.
DEFAULT_NOISE_FLOOR: Final = 3e-4
#: Silence this long is a real break in the voice, not a stop consonant.
DEFAULT_MIN_SILENCE: Final = 0.080
#: Silence this long is the end of the phrase whatever the script believes. A
#: hesitation inside a word is brief; quiet that goes on this long means the
#: words we are still holding were never going to get audio, and holding them
#: any longer only makes the catch-up later and more visible.
DEFAULT_LONG_SILENCE: Final = 0.300
#: How fast the level estimate forgets. Long enough to span a phrase, so a run
#: of loud syllables does not become the new definition of normal.
DEFAULT_LEVEL_HALFLIFE: Final = 0.85
#: How much of the advance follows energy rather than the clock. Much lower than
#: the offline warp strength, and deliberately so: the offline pass normalises
#: against a clip it already holds, while here the same energy has to be judged
#: against a running estimate, so leaning on the clock is measurably steadier.
DEFAULT_ENERGY_BLEND: Final = 0.28
#: Floor and ceiling on one frame's energy contribution. The floor mirrors the
#: offline warp's silence floor; the ceiling stops a plosive skipping syllables.
DEFAULT_RATIO_FLOOR: Final = 0.05
DEFAULT_RATE_CEILING: Final = 2.0
#: Script left in a phrase that a silence may be asked to absorb. Beyond this the
#: silence is read as a pause inside the phrase rather than the end of it.
DEFAULT_CATCHUP_SECONDS: Final = 0.25
#: Weight given to a freshly measured phrase when updating the rate estimate.
#: Well short of 1.0: one phrase is a small sample, and a voice does not change
#: pace as abruptly as a single measurement can suggest.
DEFAULT_RATE_TRUST: Final = 0.6
#: How far the estimate may stray from the prior, either way.
RATE_BOUNDS: Final = (0.5, 2.0)
#: Voiced audio a phrase may spend stalled at punctuation the voice did not
#: honour before the phrase is disqualified from measuring the speaking rate. A
#: frame or two is gate noise; more means the two clocks disagreed about where
#: the phrase ended, and a disagreement measures nothing.
STALL_TOLERANCE: Final = 0.02
#: Hold used for visemes flushed during catch-up: short, so the mouth closes the
#: gap instead of falling further behind.
CATCHUP_HOLD: Final = 0.05

_INT16_SCALE: Final = 32768.0


class StreamError(ValueError):
    """A chunk could not be interpreted as mono PCM."""


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Analysis and latency settings for one voice channel."""

    sample_rate: int = DEFAULT_STREAM_RATE
    hop_seconds: float = DEFAULT_HOP
    window_seconds: float = DEFAULT_WINDOW
    lookahead_seconds: float = DEFAULT_LOOKAHEAD
    seconds_per_viseme: float = DEFAULT_SECONDS_PER_VISEME
    gate_ratio: float = DEFAULT_GATE_RATIO
    noise_floor: float = DEFAULT_NOISE_FLOOR
    min_silence: float = DEFAULT_MIN_SILENCE
    long_silence: float = DEFAULT_LONG_SILENCE
    level_halflife: float = DEFAULT_LEVEL_HALFLIFE
    energy_blend: float = DEFAULT_ENERGY_BLEND
    ratio_floor: float = DEFAULT_RATIO_FLOOR
    rate_ceiling: float = DEFAULT_RATE_CEILING
    catchup_seconds: float = DEFAULT_CATCHUP_SECONDS
    rate_trust: float = DEFAULT_RATE_TRUST
    adapt: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise StreamError("sample_rate must be positive")
        if self.hop_seconds <= 0.0 or self.window_seconds <= 0.0:
            raise StreamError("hop and window must be positive")
        if self.window_seconds < self.hop_seconds:
            raise StreamError("window must cover at least one hop")
        if self.lookahead_seconds < 0.0:
            raise StreamError("lookahead cannot be negative")
        if self.seconds_per_viseme <= 0.0:
            raise StreamError("seconds_per_viseme must be positive")
        if not 0.0 < self.gate_ratio < 1.0:
            raise StreamError("gate_ratio must sit between 0 and 1")
        if self.long_silence < self.min_silence:
            raise StreamError("long_silence cannot be shorter than min_silence")
        if not 0.0 <= self.energy_blend <= 1.0:
            raise StreamError("energy_blend must sit between 0 and 1")
        if self.rate_ceiling <= self.ratio_floor:
            raise StreamError("rate_ceiling must exceed ratio_floor")
        if not 0.0 <= self.rate_trust <= 1.0:
            raise StreamError("rate_trust must sit between 0 and 1")

    @property
    def hop_samples(self) -> int:
        return max(int(round(self.hop_seconds * self.sample_rate)), 1)

    @property
    def window_samples(self) -> int:
        return max(int(round(self.window_seconds * self.sample_rate)), self.hop_samples)

    @property
    def lookahead_samples(self) -> int:
        return max(int(round(self.lookahead_seconds * self.sample_rate)), 0)

    @property
    def lookahead_frames(self) -> int:
        """Frames of arrived audio the channel holds back before judging one.

        A margin makes the decision on any single frame more robust — the frame is
        judged only once the audio around it has landed — at the cost of deciding
        that much later. It buys latency, not rhythm: the rhythm comes from the
        rate estimate, and :mod:`aiface.sync` measures both separately.
        """
        return self.lookahead_samples // self.hop_samples

    @property
    def worst_case_lag(self) -> float:
        """The most a decision can trail the audio that justified it."""
        return (self.window_samples + self.lookahead_samples) / float(self.sample_rate)


@dataclass(frozen=True, slots=True)
class StreamedSpan:
    """A viseme the channel decided on, and when it decided it.

    ``start`` is where the viseme belongs in the audio; ``emitted_at`` is how far
    the audio had arrived when the channel worked that out. The gap between them
    is the channel's latency, per decision, measured rather than assumed.
    """

    phoneme: str
    start: float
    hold: float
    emitted_at: float

    @property
    def end(self) -> float:
        return self.start + self.hold

    @property
    def lag(self) -> float:
        return max(self.emitted_at - self.start, 0.0)

    def as_span(self) -> PhonemeSpan:
        return PhonemeSpan(self.phoneme, self.start, self.end)

    def as_tuple(self) -> tuple[str, float, float]:
        return (self.phoneme, self.start, self.end)


@dataclass(slots=True)
class _Pending:
    """One queued viseme waiting for audio to justify it.

    ``absorbable`` marks a viseme a silence may take instead of energy — the
    relaxations between words and at punctuation. ``blocking`` is stronger: the
    script says the voice *stops* here, so no amount of energy may spend it. That
    is what pins a long utterance to the pauses the speaker actually takes.
    """

    phoneme: str
    weight: float
    absorbable: bool
    blocking: bool


@dataclass(frozen=True, slots=True)
class StreamStats:
    """What the channel has done so far, for HUD lines and status payloads."""

    received_seconds: float
    analysed_seconds: float
    emitted: int
    pending: int
    speaking: bool
    mean_lag: float
    peak_lag: float
    buffered_samples: int
    rate: float


def _expected_visemes(text: str) -> list[_Pending]:
    """Flatten a line into the viseme queue the channel will consume.

    Whitespace is absorbable but not blocking: fluent speech runs words together
    with no measurable silence, so a channel that waited for one would stall.
    Commas and full stops are blocking, because a speaker really does stop there.
    """
    return [
        _Pending(
            phoneme=canonical_viseme(name),
            weight=max(
                PHONEME_DURATION_SCALE.get(canonical_viseme(name), 1.0), 0.05
            ),
            absorbable=not token.is_word,
            blocking=token.is_break,
        )
        for token in tokenize_speech(text)
        for name in token.visemes
    ]


def decode_chunk(chunk: object, *, tail: bytes = b"") -> tuple[np.ndarray, bytes]:
    """Turn one arriving chunk into mono float32 samples.

    Accepts 16-bit little-endian PCM bytes (what the realtime APIs send) or a
    numpy array of floats or int16. Byte payloads may split a sample across
    chunks, so any trailing odd byte is returned for the next call rather than
    being dropped — one lost byte would shift every sample after it.
    """
    if isinstance(chunk, np.ndarray):
        if tail:
            raise StreamError("cannot mix byte and array chunks mid-sample")
        array = chunk
        if array.ndim == 2:
            array = array.mean(axis=1)
        if array.ndim != 1:
            raise StreamError("array chunks must be mono or (frames, channels)")
        if array.dtype == np.int16:
            return array.astype(np.float32) / _INT16_SCALE, b""
        return np.ascontiguousarray(array, dtype=np.float32), b""
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        payload = tail + bytes(chunk)
        usable = len(payload) - (len(payload) % 2)
        samples = np.frombuffer(payload[:usable], dtype="<i2").astype(np.float32)
        return samples / _INT16_SCALE, payload[usable:]
    raise StreamError(f"unsupported chunk type {type(chunk).__name__}")


class VoiceStream:
    """A push-based channel that locks visemes to audio as it arrives.

    Feed it PCM and it returns the viseme decisions that audio justified. Tell it
    what is being said with :meth:`expect` and the sequence is phonetic; without
    that it falls back to driving openness from loudness alone, which is honest
    about being a mouth flap rather than speech.
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self.config = config or StreamConfig()
        self._pending: deque[_Pending] = deque()
        self._samples = np.zeros(0, dtype=np.float32)
        self._frames: deque[float] = deque()
        self._byte_tail = b""
        self._received = 0
        self._frame_index = 0
        self._last_start = 0.0
        self._active: _Pending | None = None
        self._budget = 0.0
        self._level = 0.0
        self._speaking = False
        self._silence_started: float | None = None
        self._escalated = False
        self._hesitating = False
        self._emitted = 0
        self._lag_total = 0.0
        self._peak_lag = 0.0
        self._frame_ratio = 0.0
        self._acoustic_until = 0.0
        self._rate = self.config.seconds_per_viseme
        self._phrase_spend = 0.0
        self._phrase_weight = 0.0
        self._phrase_stall = 0.0
        self._scripted = False

    # ------------------------------------------------------------------ state

    @property
    def received_seconds(self) -> float:
        return self._received / float(self.config.sample_rate)

    @property
    def analysed_seconds(self) -> float:
        """Audio time whose frames have been processed."""
        return (self._frame_index * self.config.hop_samples) / float(
            self.config.sample_rate
        )

    @property
    def pending_visemes(self) -> int:
        return len(self._pending)

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def rate(self) -> float:
        """Current seconds-per-viseme-unit estimate for this voice."""
        return self._rate

    def stats(self) -> StreamStats:
        return StreamStats(
            received_seconds=self.received_seconds,
            analysed_seconds=self.analysed_seconds,
            emitted=self._emitted,
            pending=len(self._pending),
            speaking=self._speaking,
            mean_lag=self._lag_total / self._emitted if self._emitted else 0.0,
            peak_lag=self._peak_lag,
            buffered_samples=int(self._samples.shape[0]),
            rate=self._rate,
        )

    def reset(self) -> None:
        """Forget the utterance and the audio, keeping the configuration.

        The learned speaking rate survives: it describes the voice on the other
        end of the channel, not the sentence it just said.
        """
        self._pending.clear()
        self._samples = np.zeros(0, dtype=np.float32)
        self._frames.clear()
        self._byte_tail = b""
        self._received = 0
        self._frame_index = 0
        self._last_start = 0.0
        self._active = None
        self._budget = 0.0
        self._level = 0.0
        self._speaking = False
        self._silence_started = None
        self._escalated = False
        self._hesitating = False
        self._emitted = 0
        self._lag_total = 0.0
        self._peak_lag = 0.0
        self._frame_ratio = 0.0
        self._acoustic_until = 0.0
        self._phrase_spend = 0.0
        self._phrase_weight = 0.0
        self._phrase_stall = 0.0
        self._scripted = False

    # ------------------------------------------------------------------ input

    def expect(self, text: str) -> int:
        """Queue what is about to be spoken. Returns how many visemes that is.

        Safe to call while audio is already flowing: transcript deltas from a
        realtime model arrive in pieces, and each piece simply extends the queue.
        """
        queued = _expected_visemes(text)
        self._pending.extend(queued)
        self._scripted = self._scripted or bool(queued)
        return len(queued)

    def feed(self, chunk: object) -> list[StreamedSpan]:
        """Accept audio and return the viseme decisions it justified."""
        samples, self._byte_tail = decode_chunk(chunk, tail=self._byte_tail)
        if samples.size:
            self._samples = (
                samples.copy()
                if self._samples.size == 0
                else np.concatenate((self._samples, samples))
            )
            self._received += int(samples.shape[0])
        return self._drain(final=False)

    def finish(self) -> list[StreamedSpan]:
        """Close the utterance: analyse the tail and flush any unspoken visemes.

        The lookahead reserve is released here, since no more audio is coming.
        Visemes with no audio left to justify them are emitted anyway, spaced by
        their nominal holds — a face that stops mid-word looks broken, and the
        overshoot is visible in the report instead of being hidden.
        """
        spans = self._drain(final=True)
        cursor = max(self.received_seconds, self.analysed_seconds)
        frontier = self.received_seconds
        while self._pending:
            pending = self._pending.popleft()
            hold = phoneme_hold(pending.phoneme, base=self._rate)
            spans.append(self._record(pending.phoneme, cursor, hold, frontier))
            cursor += hold
        self._active = None
        self._budget = 0.0
        self._speaking = False
        return spans

    # -------------------------------------------------------------- analysis

    def _ingest(self, *, final: bool) -> None:
        """Turn buffered audio into one RMS value per hop.

        Measuring each frame once and queueing the result is what lets the state
        machine look forward into audio that has already arrived without paying
        for it twice.
        """
        hop = self.config.hop_samples
        window = self.config.window_samples
        while True:
            available = int(self._samples.shape[0])
            if available < (1 if final else window):
                break
            block = self._samples[:window].astype(np.float64)
            self._frames.append(float(np.sqrt(np.mean(np.square(block)))))
            self._samples = self._samples[hop:]

    def _drain(self, *, final: bool) -> list[StreamedSpan]:
        """Run the state machine over every frame the lookahead lets us judge."""
        self._ingest(final=final)
        config = self.config
        reserve = 0 if final else config.lookahead_frames
        spans: list[StreamedSpan] = []
        while len(self._frames) > reserve:
            rms = self._frames.popleft()
            frame_time = self._frame_index * config.hop_seconds
            spans.extend(self._step(rms, frame_time))
            self._frame_index += 1
        return spans

    def _step(self, rms: float, frame_time: float) -> list[StreamedSpan]:
        """Spend one frame of audio against the script."""
        config = self.config
        frontier = self.received_seconds
        spans: list[StreamedSpan] = []

        self._level = self._track_level(rms)
        gate = max(config.gate_ratio * self._level, config.noise_floor)
        voiced = rms >= gate
        self._frame_ratio = self._energy_ratio(rms)

        if not voiced:
            if self._silence_started is None:
                self._silence_started = frame_time
            confirmed = frame_time + config.hop_seconds - self._silence_started
            if confirmed >= config.min_silence and self._speaking:
                self._speaking = False
                spans.extend(self._on_silence(self._silence_started, frontier))
                return spans
            if (
                self._hesitating
                and not self._escalated
                and confirmed >= config.long_silence
                and self._pending
            ):
                self._escalated = True
                spans.extend(self._on_long_silence(self._silence_started, frontier))
                return spans
        else:
            self._silence_started = None
            self._escalated = False
            self._hesitating = False
            if not self._speaking:
                self._speaking = True
                if self._active is None:
                    # A phrase begins. Its first viseme is due at the onset we
                    # just heard, which is the one timing fact we measure exactly.
                    promoted = self._promote(frame_time, frontier)
                    if promoted is not None:
                        spans.append(promoted)
                    return spans

        if not self._speaking:
            return spans

        if self._pending and self._pending[0].blocking:
            # Stalled at a full stop, waiting for the silence the script promised.
            # Nothing is spent: only silence may spend a stop. Voiced audio during
            # the stall is noted, though, because it means the punctuation was not
            # honoured and this phrase cannot be trusted to measure the voice.
            if voiced:
                self._phrase_stall += config.hop_seconds
            return spans

        # Inside a phrase every frame spends, silence included, blending energy
        # with the clock exactly as the offline warp does. Frames louder than the
        # speaker's running level advance the script faster; quiet frames crawl.
        spend = (
            config.energy_blend * self._frame_ratio + (1.0 - config.energy_blend)
        ) * config.hop_seconds
        self._phrase_spend += spend
        self._budget -= spend

        if not self._pending:
            # Nothing left to say. The frame is still spent above, so a voice that
            # outran its transcript reads as slower than assumed rather than as
            # free time — that is what stops the estimate collapsing. With a
            # transcript the mouth now settles; without one the fallback keeps an
            # audio-only mouth roughly alive.
            fallback = self._acoustic_fallback(frame_time, frontier)
            return spans + ([fallback] if fallback is not None else [])

        while self._budget <= 0.0:
            promoted = self._promote(frame_time + config.hop_seconds, frontier)
            if promoted is None:
                break
            spans.append(promoted)
        return spans

    def _energy_ratio(self, rms: float) -> float:
        """This frame's loudness against the speaker's running level."""
        if self._level <= 0.0:
            return 0.0
        return float(
            np.clip(rms / self._level, self.config.ratio_floor, self.config.rate_ceiling)
        )

    def _track_level(self, rms: float) -> float:
        """Follow the speaker's level so the budget is in relative units.

        Only audible frames update it: letting silence pull the estimate down
        would make the next word read as shouting and race through the script.
        """
        if rms <= self.config.noise_floor:
            return self._level
        if self._level <= 0.0:
            return rms
        gate = max(self.config.gate_ratio * self._level, self.config.noise_floor)
        if rms < gate:
            return self._level
        decay = 0.5 ** (self.config.hop_seconds / self.config.level_halflife)
        return self._level * decay + rms * (1.0 - decay)

    def _promote(self, at: float, frontier: float) -> StreamedSpan | None:
        """Make the next queued viseme the one being held.

        Returns ``None`` when the script cannot advance: either it is exhausted,
        or the next viseme is a full stop that only silence may spend. In the
        second case the mouth holds its current shape until the voice pauses,
        which is what stops one fast phrase from desynchronising the rest.
        """
        if not self._pending:
            self._active = None
            self._budget = 0.0
            return self._acoustic_fallback(at, frontier)
        if self._pending[0].blocking and self._speaking:
            return None
        pending = self._pending.popleft()
        self._active = pending
        self._budget += pending.weight * self._rate
        self._phrase_weight += pending.weight
        hold = phoneme_hold(pending.phoneme, base=self._rate)
        return self._record(pending.phoneme, at, hold, frontier)

    # ---------------------------------------------------------------- phrasing

    def _phrase_remainder(self) -> tuple[float, int]:
        """Weight and count of script still owed before the next boundary.

        Any relaxation counts, not just punctuation: an audible gap between two
        words is as good an anchor as a comma, and a long sentence spoken without
        commas would otherwise offer nothing to re-anchor on at all.
        """
        weight = 0.0
        count = 0
        for pending in self._pending:
            if pending.absorbable:
                break
            weight += pending.weight
            count += 1
        return weight, count

    def _on_silence(self, silence_start: float, frontier: float) -> list[StreamedSpan]:
        """Decide what a confirmed silence means and act on it.

        A gap in the voice and a relaxation in the script are two independent
        statements that the same boundary has been reached. When they agree, the
        stretch that just ended measures how fast this voice really speaks and the
        next one starts from a corrected estimate — so error cannot accumulate
        past a boundary. When they disagree the silence is a hesitation inside a
        word, and the safe move is to change nothing.
        """
        remaining_weight, remaining_count = self._phrase_remainder()
        if remaining_weight * self._rate > self.config.catchup_seconds:
            # Too much script left for this to be a boundary: the voice paused
            # mid-word. Keep the script where it is and rest the lips. If the quiet
            # turns out to last, :meth:`_on_long_silence` revisits that judgement.
            self._hesitating = True
            return [
                self._record("REST", silence_start, self.config.min_silence, frontier)
            ]
        return self._flush_phrase(
            silence_start, frontier, remaining_count, measure=True
        )

    def _on_long_silence(
        self, silence_start: float, frontier: float
    ) -> list[StreamedSpan]:
        """Give up on words the audio was never going to carry.

        Reached only when a silence already read as a mid-word hesitation has gone
        on far too long for that to be true — most often the utterance simply
        ended with script still queued, because the transcript described more
        syllables than the voice articulated. Flushing at the moment the voice
        stopped keeps the error to the length of the leftover; waiting for
        :meth:`finish` would add the whole silence on top of it. Nothing here
        measures the voice: the two clocks plainly disagreed.
        """
        _, remaining_count = self._phrase_remainder()
        return self._flush_phrase(
            silence_start, frontier, remaining_count, measure=False
        )

    def _flush_phrase(
        self, at: float, frontier: float, count: int, *, measure: bool
    ) -> list[StreamedSpan]:
        """Close a phrase: place what is left of it, then take the punctuation."""
        spans: list[StreamedSpan] = []
        cursor = at
        for _ in range(count):
            pending = self._pending.popleft()
            self._phrase_weight += pending.weight
            spans.append(self._record(pending.phoneme, cursor, CATCHUP_HOLD, frontier))
            cursor += CATCHUP_HOLD
        spans.extend(self._consume_breaks(cursor, frontier))
        if measure:
            self._recalibrate()
        self._hesitating = False
        self._active = None
        self._budget = 0.0
        self._phrase_spend = 0.0
        self._phrase_weight = 0.0
        self._phrase_stall = 0.0
        return spans

    def _consume_breaks(self, at: float, frontier: float) -> list[StreamedSpan]:
        """Give the silence to the punctuation that predicted it."""
        spans: list[StreamedSpan] = []
        cursor = at
        while self._pending and self._pending[0].absorbable:
            pending = self._pending.popleft()
            hold = phoneme_hold(pending.phoneme, base=self._rate)
            spans.append(self._record(pending.phoneme, cursor, hold, frontier))
            cursor += hold
        if not spans:
            # The audio paused where the script did not. Rest anyway: a mouth
            # held mid-vowel through a silence is the worst artefact of all.
            spans.append(
                self._record("REST", at, self.config.min_silence, frontier)
            )
        return spans

    def _recalibrate(self) -> None:
        """Update the speaking-rate estimate from the phrase that just ended.

        A phrase only counts as a measurement if the script and the voice ended it
        together. When the voice talked straight through punctuation, the audio
        just measured covers more speech than the weight it is divided by, and
        believing it would push the estimate the wrong way — and, because the
        estimate then paces the next phrase, keep pushing.
        """
        if not self.config.adapt or self._phrase_weight <= 0.0:
            return
        if self._phrase_stall > STALL_TOLERANCE:
            return
        observed = self._phrase_spend / self._phrase_weight
        trust = self.config.rate_trust
        blended = self._rate * (1.0 - trust) + observed * trust
        low, high = RATE_BOUNDS
        nominal = self.config.seconds_per_viseme
        self._rate = float(np.clip(blended, nominal * low, nominal * high))

    def _acoustic_fallback(self, at: float, frontier: float) -> StreamedSpan | None:
        """Drive aperture from loudness when nothing said what the words are.

        This is not phoneme recognition and does not pretend to be: three
        aperture classes chosen by level, rate-limited to one per hold, so an
        audio-only source still moves a mouth roughly in time with its voice.

        A channel that has *ever* been given words never falls back to this. Once
        a caller has said what is being spoken, silence from the mouth is the
        right answer to audio the transcript does not cover — guessing would be a
        visible lie, and the transcript may simply be arriving a little behind.
        """
        if self._scripted or at < self._acoustic_until or self._frame_ratio <= 0.0:
            return None
        if self._frame_ratio >= 1.15:
            choice = "AA"
        elif self._frame_ratio >= 0.75:
            choice = "AH"
        else:
            choice = "EH"
        hold = phoneme_hold(choice, base=self._rate)
        self._acoustic_until = at + hold
        return self._record(choice, at, hold, frontier)

    def _record(
        self, phoneme: str, start: float, hold: float, frontier: float
    ) -> StreamedSpan:
        """Emit one decision, keeping the sequence monotone in time.

        Catch-up and syllable anchoring both place visemes relative to a measured
        moment, which can land before the previous decision. Consumers schedule
        these in order, so a start is never allowed to move backwards.
        """
        begin = max(float(start), self._last_start, 0.0)
        self._last_start = begin
        span = StreamedSpan(
            phoneme=phoneme,
            start=begin,
            hold=max(float(hold), 0.0),
            emitted_at=max(float(frontier), begin),
        )
        self._emitted += 1
        self._lag_total += span.lag
        self._peak_lag = max(self._peak_lag, span.lag)
        return span


def stream_visemes(
    spans: Iterable[StreamedSpan], emotion: str, *, start_at: float
) -> list[VisemeEvent]:
    """Place streamed spans on the app clock, anchored at the audio's first sample."""
    return schedule_spans(
        [span.as_tuple() for span in spans], emotion, start_at=start_at
    )


def chunk_clip(
    samples: np.ndarray, sample_rate: int, *, chunk_seconds: float = DEFAULT_CHUNK_SECONDS
) -> list[np.ndarray]:
    """Slice a clip the way a realtime API would deliver it."""
    size = max(int(round(chunk_seconds * sample_rate)), 1)
    return [samples[index : index + size] for index in range(0, samples.shape[0], size)]


def align_stream(
    text: str,
    samples: np.ndarray,
    *,
    config: StreamConfig | None = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
) -> list[StreamedSpan]:
    """Run a whole clip through the streaming channel, chunk by chunk.

    Used by the oracle and by tests. It is deliberately clock-free: chunk order
    is what the algorithm reacts to, not wall time, so a measurement is
    reproducible and does not take an utterance's worth of seconds to run.
    """
    stream = VoiceStream(config)
    stream.expect(text)
    spans: list[StreamedSpan] = []
    rate = stream.config.sample_rate
    for chunk in chunk_clip(samples, rate, chunk_seconds=chunk_seconds):
        spans.extend(stream.feed(chunk))
    spans.extend(stream.finish())
    return spans


__all__ = [
    "DEFAULT_CHUNK_SECONDS",
    "DEFAULT_LOOKAHEAD",
    "DEFAULT_STREAM_RATE",
    "StreamConfig",
    "StreamError",
    "StreamStats",
    "StreamedSpan",
    "VoiceStream",
    "align_stream",
    "chunk_clip",
    "decode_chunk",
    "stream_visemes",
]
