"""PrismAPI adapter → FaceBridge speak mapping."""

from __future__ import annotations

from chorusface.prism_adapter import SPEAK_SCHEMA, SpeakIntent, forward_speak


def test_speak_intent_from_payload_aliases() -> None:
    assert SpeakIntent.from_payload({"text": "a"}).text == "a"
    assert SpeakIntent.from_payload({"message": "b"}).text == "b"
    assert SpeakIntent.from_payload({"speak": {"response": "c"}}).text == "c"
    intent = SpeakIntent.from_payload(
        {"text": "hi", "requestId": "r1", "emotion": "HAPPY"}
    )
    assert intent.request_id == "r1"
    assert intent.emotion == "HAPPY"
    assert intent.schema == SPEAK_SCHEMA


def test_forward_speak_empty() -> None:
    result = forward_speak({"text": "  "})
    assert result.ok is False
    assert result.error == "empty text"


def test_forward_speak_face_down_no_raise() -> None:
    result = forward_speak(
        "Hello",
        base_url="http://127.0.0.1:1",
        token="x",
    )
    assert result.ok is False
