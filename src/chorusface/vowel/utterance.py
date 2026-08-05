"""Host utterance JSON parse/validate (F1–F2, F26–F27)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chorusface.vowel.schema import EMOTION_INDEX, GA16_INDEX, TICK_HZ


@dataclass(slots=True)
class EmotionSpan:
    emotion: str
    start_s: float
    end_s: float


@dataclass(slots=True)
class PhonemeSpan:
    tag: str
    start_s: float
    end_s: float

    @property
    def start_tick(self) -> int:
        return int(round(self.start_s * TICK_HZ))

    @property
    def end_tick(self) -> int:
        return max(self.start_tick + 1, int(round(self.end_s * TICK_HZ)))


@dataclass(slots=True)
class WordSpan:
    text: str
    start_s: float
    end_s: float


@dataclass(slots=True)
class UtterancePayload:
    utterance_id: str
    text: str
    emotion_track: list[EmotionSpan]
    spans: list[PhonemeSpan] = field(default_factory=list)
    words: list[WordSpan] = field(default_factory=list)
    duration_s: float | None = None
    speaker_id: str | None = None

    @property
    def primary_emotion(self) -> str:
        if not self.emotion_track:
            return "NEUTRAL"
        return self.emotion_track[0].emotion

    def emotion_at(self, t_s: float) -> str:
        for e in self.emotion_track:
            if e.start_s <= t_s < e.end_s or (
                t_s >= e.start_s and e is self.emotion_track[-1]
            ):
                if t_s >= e.start_s and (t_s < e.end_s or e is self.emotion_track[-1]):
                    return e.emotion
        return self.primary_emotion

    def total_ticks(self) -> int:
        if self.spans:
            return max(s.end_tick for s in self.spans) + 6  # release pad
        if self.duration_s is not None:
            return max(1, int(round(float(self.duration_s) * TICK_HZ)) + 6)
        if self.words:
            return max(1, int(round(max(w.end_s for w in self.words) * TICK_HZ)) + 6)
        # rough from text word count @ 150 WPM
        n_words = max(1, len(self.text.split()))
        return int(round(n_words * 0.4 * TICK_HZ)) + 6


def parse_utterance(payload: dict[str, Any]) -> UtterancePayload:
    """Parse host JSON into UtterancePayload. Raises ValueError on hard failures."""
    if not isinstance(payload, dict):
        raise ValueError("utterance must be a JSON object")
    uid = str(payload.get("utterance_id", "") or "").strip()
    text = str(payload.get("text", "") or "").strip()
    if not uid:
        raise ValueError("utterance_id is required")
    if not text:
        raise ValueError("text is required")

    raw_track = payload.get("emotion_track")
    if not isinstance(raw_track, list) or not raw_track:
        # single-emotion convenience
        emo = str(payload.get("emotion", "NEUTRAL") or "NEUTRAL").strip().upper()
        if emo not in EMOTION_INDEX:
            emo = "NEUTRAL"
        duration = float(payload.get("duration_s") or 1.0)
        raw_track = [{"emotion": emo, "start_s": 0.0, "end_s": duration}]

    emotion_track: list[EmotionSpan] = []
    for i, item in enumerate(raw_track):
        if not isinstance(item, dict):
            raise ValueError(f"emotion_track[{i}] must be object")
        emo = str(item.get("emotion", "") or "").strip().upper()
        if emo not in EMOTION_INDEX:
            raise ValueError(f"unknown emotion: {emo}")
        start_s = float(item.get("start_s", 0.0))
        end_s = item.get("end_s")
        if end_s is None:
            # hold until next or duration
            if i + 1 < len(raw_track):
                end_s = float(raw_track[i + 1].get("start_s", start_s + 0.1))
            else:
                end_s = float(payload.get("duration_s") or start_s + 1.0)
        emotion_track.append(
            EmotionSpan(emotion=emo, start_s=start_s, end_s=float(end_s))
        )

    spans: list[PhonemeSpan] = []
    raw_spans = payload.get("spans") or payload.get("phonemes") or []
    if isinstance(raw_spans, list):
        for item in raw_spans:
            if not isinstance(item, dict):
                continue
            tag = str(
                item.get("tag") or item.get("phoneme") or item.get("vowel") or ""
            ).strip().upper()
            if tag not in GA16_INDEX:
                continue  # consonants ignored in Phase-1 vowel spans
            spans.append(
                PhonemeSpan(
                    tag=tag,
                    start_s=float(item["start_s"] if "start_s" in item else item.get("start", 0.0)),
                    end_s=float(item["end_s"] if "end_s" in item else item.get("end", 0.0)),
                )
            )

    words: list[WordSpan] = []
    raw_words = payload.get("words") or []
    if isinstance(raw_words, list):
        for item in raw_words:
            if not isinstance(item, dict):
                continue
            w = str(item.get("text") or item.get("word") or "").strip()
            if not w:
                continue
            words.append(
                WordSpan(
                    text=w,
                    start_s=float(item.get("start_s", item.get("start", 0.0))),
                    end_s=float(item.get("end_s", item.get("end", 0.0))),
                )
            )

    duration_s = payload.get("duration_s")
    speaker_id = payload.get("speaker_id")
    return UtterancePayload(
        utterance_id=uid,
        text=text,
        emotion_track=emotion_track,
        spans=spans,
        words=words,
        duration_s=float(duration_s) if duration_s is not None else None,
        speaker_id=str(speaker_id) if speaker_id else None,
    )
