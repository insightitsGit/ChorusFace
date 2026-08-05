"""NWR-correct GA-16 → biomech muscle drive coverage."""

from __future__ import annotations

import json
from pathlib import Path

from chorusface.speech import canonical_viseme
from chorusface.vowel.muscle_drive import (
    compose_muscle_drive,
    ga16_muscle_coverage,
    spans_to_drive_events,
)
from chorusface.vowel.schema import GA16
from chorusface.vowel.utterance import parse_utterance


def test_ga16_canonical_passthrough() -> None:
    for tag in GA16:
        assert canonical_viseme(tag) == tag


def test_phoneme_muscles_cover_ga16() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "chorusface"
        / "biomechanics"
        / "data"
        / "face_definition.json"
    )
    definition = json.loads(path.read_text(encoding="utf-8"))
    coverage = ga16_muscle_coverage(definition["phoneme_muscles"])
    missing = [tag for tag, ok in coverage.items() if not ok]
    assert not missing, f"missing phoneme_muscles for {missing}"


def test_compose_muscle_drive_keeps_ga16_tags() -> None:
    _, events = compose_muscle_drive(
        {
            "utterance_id": "t_drive",
            "text": "see you ah",
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 2.0}],
            "spans": [
                {"tag": "EE", "start_s": 0.05, "end_s": 0.45},
                {"tag": "OU", "start_s": 0.55, "end_s": 0.95},
                {"tag": "AE", "start_s": 1.10, "end_s": 1.50},
                {"tag": "OY", "start_s": 1.55, "end_s": 1.90},
            ],
        }
    )
    tags = [e.tag for e in events]
    assert tags == ["EE", "OU", "AE", "OY"]
    assert all(e.emotion == "HAPPY" for e in events)


def test_spans_skip_rest() -> None:
    payload = parse_utterance(
        {
            "utterance_id": "t_rest",
            "text": "x",
            "emotion_track": [{"emotion": "NEUTRAL", "start_s": 0.0, "end_s": 1.0}],
            "spans": [
                {"tag": "REST", "start_s": 0.0, "end_s": 0.2},
                {"tag": "AX", "start_s": 0.2, "end_s": 0.6},
            ],
        }
    )
    events = spans_to_drive_events(payload)
    assert [e.tag for e in events] == ["AX"]
