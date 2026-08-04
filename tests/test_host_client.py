"""host_client speak + host-voice path — success + face-down must not raise."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from chorusface.host_client import SpeakResult, drive_host_voice, speak, voice_expect


def test_speak_success_against_local_stub() -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if self.path.endswith("/auth/activate"):
                body = json.dumps({"ok": True}).encode("utf-8")
            else:
                assert self.path.endswith("/prism/speak")
                payload = json.loads(raw.decode("utf-8"))
                assert self.headers.get("Authorization") == "Bearer tok"
                assert self.headers.get("X-ChorusFace-Client-Id")
                body = json.dumps({"queued": True, "text": payload["text"]}).encode(
                    "utf-8"
                )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = speak(
            "Hi there",
            base_url=f"http://127.0.0.1:{port}",
            token="tok",
            client_id="11111111-2222-3333-4444-555555555555",
            timeout_s=1.5,
        )
        assert isinstance(result, SpeakResult)
        assert result.ok is True
        assert result.queued is True
        assert result.text == "Hi there"
    finally:
        server.shutdown()


def test_drive_host_voice_against_local_stub() -> None:
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            seen.append(self.path.split("?", 1)[0])
            assert self.headers.get("Authorization") == "Bearer tok"
            assert self.headers.get("X-ChorusFace-Client-Id")
            if self.path.startswith("/voice/pcm"):
                assert raw == b"\x01\x02\x03\x04"
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = drive_host_voice(
            "Hello there",
            b"\x01\x02\x03\x04",
            base_url=f"http://127.0.0.1:{port}",
            token="tok",
            client_id="11111111-2222-3333-4444-555555555555",
            sample_rate=24000,
            timeout_s=1.5,
        )
        assert result.ok is True
        assert "/auth/activate" in seen
        assert "/voice/expect" in seen
        assert "/voice/pcm" in seen
        assert "/voice/end" in seen
        expect = voice_expect(
            "Hi",
            base_url=f"http://127.0.0.1:{port}",
            token="tok",
            client_id="11111111-2222-3333-4444-555555555555",
            activate_first=False,
            timeout_s=1.5,
        )
        assert expect.ok is True
    finally:
        server.shutdown()


def test_speak_face_down_does_not_raise() -> None:
    result = speak(
        "Hello",
        base_url="http://127.0.0.1:1",
        token="x",
        timeout_s=0.3,
        raise_on_error=False,
    )
    assert result.ok is False
    assert result.queued is False
    assert result.error


def test_speak_empty_text() -> None:
    result = speak("   ")
    assert result.ok is False
    assert result.error == "empty text"
