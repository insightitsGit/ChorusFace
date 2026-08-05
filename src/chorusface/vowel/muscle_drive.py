"""GA-16 → BiomechanicalFace muscle playback (NWR-correct vowel path).

Cell-group expand (W) is not the product delivery path for locked regions.
This module schedules ``submit_phoneme`` from composed utterance spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from chorusface.vowel.pipeline import ComposeResult, compose_utterance
from chorusface.vowel.schema import GA16_INDEX, TICK_HZ
from chorusface.vowel.utterance import UtterancePayload


class BiomechFace(Protocol):
    def submit_phoneme(
        self,
        phoneme: str,
        *,
        tick: int,
        emotion_label: str | None = None,
        duration: float = 0.1,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class MuscleDriveEvent:
    """One phoneme fire for the biomechanical face."""

    tag: str
    emotion: str
    start_s: float
    end_s: float
    start_tick: int

    @property
    def duration_s(self) -> float:
        return max(1.0 / TICK_HZ, self.end_s - self.start_s)


def emotion_at(payload: UtterancePayload, t: float) -> str:
    for em in payload.emotion_track:
        if em.start_s <= t < em.end_s or (
            abs(em.end_s - em.start_s) < 1e-9 and abs(t - em.start_s) < 1e-9
        ):
            return em.emotion
    return payload.primary_emotion


def spans_to_drive_events(payload: UtterancePayload) -> list[MuscleDriveEvent]:
    """Convert GA-16 spans into biomech fire events (REST skipped)."""
    events: list[MuscleDriveEvent] = []
    for span in payload.spans:
        tag = str(span.tag or "").strip().upper()
        if not tag or tag == "REST" or tag not in GA16_INDEX:
            continue
        start = float(span.start_s)
        end = float(span.end_s)
        if end <= start:
            end = start + 1.0 / TICK_HZ
        events.append(
            MuscleDriveEvent(
                tag=tag,
                emotion=emotion_at(payload, start),
                start_s=start,
                end_s=end,
                start_tick=max(0, int(round(start * TICK_HZ))),
            )
        )
    return events


def compose_muscle_drive(
    payload: Mapping[str, Any] | UtterancePayload,
) -> tuple[ComposeResult, list[MuscleDriveEvent]]:
    """Compose utterance + build biomech drive schedule."""
    result = compose_utterance(payload)
    return result, spans_to_drive_events(result.payload)


def play_muscle_drive(
    face: BiomechFace,
    events: Sequence[MuscleDriveEvent],
    *,
    tick_offset: int = 0,
) -> int:
    """Submit all events immediately (offline / test helper).

    Live app path uses ``schedule_spans`` + ``_fire_impulse`` instead; this is
    for acceptance / headless biomech smoke without the full App loop.
    """
    n = 0
    for ev in events:
        face.submit_phoneme(
            ev.tag,
            tick=tick_offset + ev.start_tick,
            emotion_label=ev.emotion,
            duration=ev.duration_s,
        )
        n += 1
    return n


def ga16_muscle_coverage(
    phoneme_muscles: Mapping[str, Mapping[str, float]],
) -> dict[str, bool]:
    """Report which GA-16 tags have a non-empty muscle table entry."""
    return {
        tag: bool(phoneme_muscles.get(tag)) and any(
            float(v) > 0.0 for v in phoneme_muscles.get(tag, {}).values()
        )
        for tag in GA16_INDEX
    }


__all__ = [
    "MuscleDriveEvent",
    "compose_muscle_drive",
    "emotion_at",
    "ga16_muscle_coverage",
    "play_muscle_drive",
    "spans_to_drive_events",
]
