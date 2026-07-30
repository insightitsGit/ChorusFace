"""The sync oracle: how much timing does streaming actually cost?

The offline aligner in :mod:`aiface.tts` sees a whole clip before it decides
anything, so it is the best alignment this codebase can produce. The streaming
channel in :mod:`aiface.stream` sees the same audio arriving in chunks and has to
commit to each viseme as it goes. Run one utterance through both, match the two
viseme sequences, and the difference in onset times *is* the cost of causality —
in milliseconds, per viseme, with no room for opinion.

Three numbers matter, and they are not interchangeable:

``bias``
    The mean signed offset. A constant offset is the cheap kind of error: one
    playback trim cancels it, so it is a calibration constant, not a defect.
``jitter``
    Spread around that bias. This is the part no trim can fix, so it is the
    honest measure of streaming quality.
``lag``
    How far behind the audio the channel was when it committed to a viseme. Even
    a perfectly placed viseme is useless if it is decided after the sound has
    played, so this is what bounds how much audio a caller must buffer.

The oracle is clock-free: it replays a clip chunk by chunk in the order a
realtime API would deliver it, without sleeping. A measurement is therefore
reproducible and runs in milliseconds, which is what makes it usable as a CI
gate rather than a demo.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

from aiface.audio import AudioClip, rms_envelope
from aiface.speech import PhonemeSpan, tokenize_speech
from aiface.stream import (
    DEFAULT_CHUNK_SECONDS,
    StreamConfig,
    StreamedSpan,
    align_stream,
)
from aiface.tts import DEFAULT_WARP_STRENGTH, align_by_energy

#: Default gate for CI: streaming onsets must land within this of the offline
#: alignment once a constant trim is applied.
DEFAULT_BUDGET_MS: Final = 250.0

#: The fixture utterances. Chosen to cover the cases that behave differently:
#: several short phrases, one long stretch with a single comma, a one-word reply
#: with no internal structure to anchor to, and a long sentence whose punctuation
#: does not match where the voice actually pauses.
ORACLE_LINES: Final[tuple[str, ...]] = (
    "Hello there. I am listening to you now, carefully.",
    "The lips follow the voice, not a guess about the voice.",
    "Ask me anything, and watch the mouth land on every syllable.",
    "Yes.",
    "Sound and motion have to agree or the whole illusion falls apart, so we "
    "measure the disagreement instead of hoping it is small.",
)


@dataclass(frozen=True, slots=True)
class OnsetDelta:
    """One viseme, timed by both paths."""

    index: int
    phoneme: str
    batch_start: float
    stream_start: float
    emitted_at: float

    @property
    def delta(self) -> float:
        """Streaming onset minus offline onset. Negative means early."""
        return self.stream_start - self.batch_start

    @property
    def lag(self) -> float:
        return max(self.emitted_at - self.stream_start, 0.0)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


@dataclass(frozen=True, slots=True)
class SyncReport:
    """What the two paths did to one utterance."""

    text: str
    clip_seconds: float
    chunk_seconds: float
    lookahead_seconds: float
    batch_count: int
    stream_count: int
    deltas: tuple[OnsetDelta, ...]

    # ------------------------------------------------------------- statistics

    @property
    def matched(self) -> int:
        return len(self.deltas)

    @property
    def coverage(self) -> float:
        """Fraction of offline visemes the streaming path also produced."""
        return self.matched / self.batch_count if self.batch_count else 0.0

    @property
    def _milliseconds(self) -> list[float]:
        return [delta.delta * 1000.0 for delta in self.deltas]

    @property
    def bias_ms(self) -> float:
        values = self._milliseconds
        return float(np.mean(values)) if values else 0.0

    @property
    def median_ms(self) -> float:
        values = self._milliseconds
        return float(np.median(values)) if values else 0.0

    @property
    def abs_p95_ms(self) -> float:
        return _percentile([abs(value) for value in self._milliseconds], 95.0)

    @property
    def abs_max_ms(self) -> float:
        values = [abs(value) for value in self._milliseconds]
        return max(values) if values else 0.0

    @property
    def jitter_ms(self) -> float:
        """Standard deviation around the bias: the part a trim cannot remove."""
        values = self._milliseconds
        return float(np.std(values)) if values else 0.0

    @property
    def trimmed_p95_ms(self) -> float:
        """95th percentile error once the constant bias is compensated."""
        bias = self.bias_ms
        return _percentile([abs(value - bias) for value in self._milliseconds], 95.0)

    @property
    def lag_mean_ms(self) -> float:
        values = [delta.lag * 1000.0 for delta in self.deltas]
        return float(np.mean(values)) if values else 0.0

    @property
    def lag_p95_ms(self) -> float:
        return _percentile([delta.lag * 1000.0 for delta in self.deltas], 95.0)

    def within(self, budget_ms: float) -> bool:
        """Whether the trim-compensated 95th percentile fits a budget."""
        return self.matched > 0 and self.trimmed_p95_ms <= budget_ms

    # ----------------------------------------------------------------- output

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "clip_seconds": round(self.clip_seconds, 4),
            "chunk_ms": round(self.chunk_seconds * 1000.0, 2),
            "lookahead_ms": round(self.lookahead_seconds * 1000.0, 2),
            "visemes": {
                "offline": self.batch_count,
                "streaming": self.stream_count,
                "matched": self.matched,
                "coverage": round(self.coverage, 4),
            },
            "onset_error_ms": {
                "bias": round(self.bias_ms, 2),
                "median": round(self.median_ms, 2),
                "jitter": round(self.jitter_ms, 2),
                "abs_p95": round(self.abs_p95_ms, 2),
                "abs_max": round(self.abs_max_ms, 2),
                "trimmed_p95": round(self.trimmed_p95_ms, 2),
            },
            "decision_lag_ms": {
                "mean": round(self.lag_mean_ms, 2),
                "p95": round(self.lag_p95_ms, 2),
            },
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)

    def table(self) -> str:
        """A short human-readable summary."""
        lines = [
            f'utterance        "{self.text}"',
            f"clip             {self.clip_seconds:.3f}s"
            f"   chunk {self.chunk_seconds * 1000:.0f}ms"
            f"   lookahead {self.lookahead_seconds * 1000:.0f}ms",
            f"visemes          {self.matched} matched of {self.batch_count} offline"
            f" / {self.stream_count} streamed ({self.coverage:.0%} coverage)",
            f"onset bias       {self.bias_ms:+.1f} ms (constant; a trim cancels it)",
            f"onset jitter     {self.jitter_ms:.1f} ms sd",
            f"onset |error|    p95 {self.abs_p95_ms:.1f} ms   max {self.abs_max_ms:.1f} ms",
            f"after trim       p95 {self.trimmed_p95_ms:.1f} ms",
            f"decision lag     mean {self.lag_mean_ms:.1f} ms   p95 {self.lag_p95_ms:.1f} ms",
        ]
        return "\n".join(lines)

    def rows(self) -> str:
        """Per-viseme detail, for when a summary is not enough."""
        header = f"{'#':>4} {'viseme':>7} {'offline':>9} {'stream':>9} {'delta':>8} {'lag':>7}"
        lines = [header, "-" * len(header)]
        for delta in self.deltas:
            lines.append(
                f"{delta.index:>4} {delta.phoneme:>7} "
                f"{delta.batch_start:9.3f} {delta.stream_start:9.3f} "
                f"{delta.delta * 1000:8.1f} {delta.lag * 1000:7.1f}"
            )
        return "\n".join(lines)


def match_onsets(
    batch: Sequence[PhonemeSpan], streamed: Sequence[StreamedSpan]
) -> list[OnsetDelta]:
    """Pair the two viseme sequences without letting one slip against the other.

    Both paths read the same script, so the sequences normally agree exactly. They
    can still differ: the offline pass drops spans that measured no width, and the
    streaming pass inserts a rest when the voice pauses where the script did not.
    A longest-matching-blocks diff over the viseme names keeps only pairs it is
    sure about, so one inserted rest cannot shift every comparison after it.
    """
    matcher = difflib.SequenceMatcher(
        a=[span.phoneme for span in batch],
        b=[span.phoneme for span in streamed],
        autojunk=False,
    )
    pairs: list[OnsetDelta] = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            left = batch[block.a + offset]
            right = streamed[block.b + offset]
            pairs.append(
                OnsetDelta(
                    index=block.a + offset,
                    phoneme=left.phoneme,
                    batch_start=left.start,
                    stream_start=right.start,
                    emitted_at=right.emitted_at,
                )
            )
    return pairs


def align_offline(
    text: str, clip: AudioClip, *, warp_strength: float = DEFAULT_WARP_STRENGTH
) -> list[PhonemeSpan]:
    """The reference alignment: full lookahead over the whole clip."""
    return align_by_energy(
        tokenize_speech(text),
        rms_envelope(clip),
        duration=clip.duration,
        strength=warp_strength,
    )


def measure_sync(
    text: str,
    clip: AudioClip,
    *,
    config: StreamConfig | None = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    warp_strength: float = DEFAULT_WARP_STRENGTH,
) -> SyncReport:
    """Run one utterance through both paths and report the difference."""
    settings = config or StreamConfig(sample_rate=clip.sample_rate)
    if settings.sample_rate != clip.sample_rate:
        # Comparing a clip against a channel tuned for another rate would measure
        # the mismatch, not the algorithm.
        raise ValueError(
            f"stream config is {settings.sample_rate} Hz but the clip is "
            f"{clip.sample_rate} Hz"
        )
    batch = align_offline(text, clip, warp_strength=warp_strength)
    streamed = align_stream(
        text, clip.samples, config=settings, chunk_seconds=chunk_seconds
    )
    return SyncReport(
        text=text,
        clip_seconds=clip.duration,
        chunk_seconds=chunk_seconds,
        lookahead_seconds=settings.lookahead_seconds,
        batch_count=len(batch),
        stream_count=len(streamed),
        deltas=tuple(match_onsets(batch, streamed)),
    )


def summarise(reports: Sequence[SyncReport]) -> dict[str, object]:
    """Aggregate several utterances into the numbers a report leads with."""
    if not reports:
        return {"utterances": 0}
    trimmed = [report.trimmed_p95_ms for report in reports]
    return {
        "utterances": len(reports),
        "clip_seconds": round(sum(report.clip_seconds for report in reports), 3),
        "matched": sum(report.matched for report in reports),
        "coverage": round(
            sum(report.matched for report in reports)
            / max(sum(report.batch_count for report in reports), 1),
            4,
        ),
        "trimmed_p95_ms": {
            "mean": round(float(np.mean(trimmed)), 2),
            "worst": round(float(np.max(trimmed)), 2),
        },
        "abs_p95_ms": {
            "mean": round(
                float(np.mean([report.abs_p95_ms for report in reports])), 2
            ),
            "worst": round(float(np.max([report.abs_p95_ms for report in reports])), 2),
        },
        "bias_ms": {
            "mean": round(float(np.mean([report.bias_ms for report in reports])), 2),
            "spread": round(float(np.std([report.bias_ms for report in reports])), 2),
        },
        "decision_lag_ms": {
            "p95": round(
                float(np.max([report.lag_p95_ms for report in reports])), 2
            ),
        },
    }


__all__ = [
    "DEFAULT_BUDGET_MS",
    "ORACLE_LINES",
    "OnsetDelta",
    "SyncReport",
    "align_offline",
    "match_onsets",
    "measure_sync",
    "summarise",
]
