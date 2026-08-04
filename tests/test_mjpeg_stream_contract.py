"""MJPEG stream + Prism speak routes are wired on FaceBridge."""

from __future__ import annotations

from pathlib import Path

from chorusface.runtime.field import encode_jpeg
from chorusface.service.bridge import speak_text_from_payload


def test_encode_jpeg_roundtrip_shape() -> None:
    # 2x2 RGB bottom-up buffer
    pixels = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0])
    jpeg = encode_jpeg(pixels, 2, 2, 3, quality=80)
    assert jpeg[:2] == b"\xff\xd8"
    assert len(jpeg) > 20


def test_bridge_source_has_stream_and_prism_routes() -> None:
    text = Path("src/chorusface/service/bridge.py").read_text(encoding="utf-8")
    assert "/stream.mjpg" in text
    assert "preview_jpeg" in text
    assert "/prism/speak" in text
    assert "/voice/expect" in text
    assert "/voice/pcm" in text
    assert "/voice/end" in text
    assert "/voice/timeline" in text
    assert '"local_tts_default": False' in text
    assert 'query_token' in text or 'get("token"' in text


def test_service_launcher_exists() -> None:
    assert Path("scripts/run_chorusface_service.py").is_file()
    assert Path("Dockerfile").is_file()
    assert Path("docker-compose.yml").is_file()
    assert Path("docs/FaceServiceEmbed.md").is_file()
    assert Path("connectors/web/chorusface-embed.js").is_file()


def test_speak_payload_helpers() -> None:
    assert speak_text_from_payload({"response": "ok"}) == "ok"
