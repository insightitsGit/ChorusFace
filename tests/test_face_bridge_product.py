"""Product beta FaceBridge — CORS preflight + /speak text aliases."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

from chorusface.service.bridge import (
    FaceBridge,
    parse_cors_origins,
    speak_text_from_payload,
)


def test_speak_text_aliases() -> None:
    assert speak_text_from_payload({"text": "a"}) == "a"
    assert speak_text_from_payload({"speech": "b"}) == "b"
    assert speak_text_from_payload({"message": "c"}) == "c"
    assert speak_text_from_payload({"response": "d"}) == "d"
    assert speak_text_from_payload({"text": "  ", "message": "e"}) == "e"
    assert speak_text_from_payload({}) == ""


def test_parse_cors_origins() -> None:
    assert parse_cors_origins("*") == ("*",)
    assert parse_cors_origins("https://a.com, https://b.com") == (
        "https://a.com",
        "https://b.com",
    )


def test_bridge_cors_options_and_speak_alias() -> None:
    spoken: list[str] = []
    bridge = FaceBridge(
        status_provider=lambda: {"ok": True},
        preview_provider=lambda: b"",
        screenshot_provider=lambda: b"",
        speak_handler=spoken.append,
        token="test-token",
        host="127.0.0.1",
        port=0,
        cors_origins="*",
        job_timeout=2.0,
    )
    bridge.start()
    assert bridge._server is not None
    port = int(bridge._server.server_address[1])
    stop = threading.Event()

    def _drain() -> None:
        while not stop.is_set():
            bridge.service()
            stop.wait(0.01)

    worker = threading.Thread(target=_drain, daemon=True)
    worker.start()

    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=3.0)
        conn.request(
            "OPTIONS",
            "/speak",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        opt = conn.getresponse()
        opt.read()
        assert opt.status == 204
        assert opt.getheader("Access-Control-Allow-Origin") == "*"
        assert "POST" in (opt.getheader("Access-Control-Allow-Methods") or "")

        conn.request("GET", "/health")
        health = conn.getresponse()
        body = json.loads(health.read().decode("utf-8"))
        assert health.status == 200
        assert body["ok"] is True
        assert body["product"] == "beta"
        assert body["embed"] == "/stream.mjpg"
        assert body["prism_speak"] == "/prism/speak"
        assert body["host_voice"] == "/voice/expect|/voice/pcm|/voice/end"
        assert body["voice_timeline"] == "/voice/timeline"
        assert body["local_tts_default"] is False

        payload = json.dumps({"message": "Hello from host"}).encode("utf-8")
        conn.request(
            "POST",
            "/prism/speak",
            body=payload,
            headers={
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )
        speak_resp = conn.getresponse()
        speak_body = json.loads(speak_resp.read().decode("utf-8"))
        assert speak_resp.status == 200
        assert speak_body["queued"] is True
        assert speak_body["channel"] == "prism"
        assert speak_body["text"] == "Hello from host"
        for _ in range(50):
            if spoken:
                break
            stop.wait(0.02)
        assert spoken == ["Hello from host"]

        # Query-token auth for <img> embed paths.
        conn.request("GET", "/health")
        conn.getresponse().read()
        jpeg_frames: list[bytes] = []

        def _jpeg() -> bytes:
            jpeg_frames.append(b"\xff\xd8fake")
            return b"\xff\xd8fake"

        bridge._preview_jpeg_provider = _jpeg
        conn.request("GET", "/preview.jpg?token=test-token")
        jpg = conn.getresponse()
        data = jpg.read()
        assert jpg.status == 200
        assert data.startswith(b"\xff\xd8")
        conn.close()
    finally:
        stop.set()
        bridge.stop()
        worker.join(timeout=1.0)
