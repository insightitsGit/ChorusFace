"""Prove host-owned TTS is the real product default (not docs-only)."""

from __future__ import annotations

import json
import re
import threading
from http.client import HTTPConnection
from pathlib import Path

import numpy as np

from chorusface.host_client import drive_host_voice, speak
from chorusface.service.bridge import FaceBridge
from chorusface.stream import StreamConfig, VoiceStream


ROOT = Path(__file__).resolve().parents[1]


def _pcm16_tone(seconds: float = 0.35, rate: int = 24_000, hz: float = 220.0) -> bytes:
    n = int(rate * seconds)
    t = np.arange(n, dtype=np.float32) / float(rate)
    samples = (0.45 * np.sin(2.0 * np.pi * hz * t) * 32767.0).astype("<i2")
    return samples.tobytes()


def test_launcher_scripts_default_local_tts_off() -> None:
    """Beta + service launchers must NOT enable --tts unless opted in."""
    for rel in (
        "scripts/run_chorusface_beta.py",
        "scripts/run_chorusface_service.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert 'add_argument(\n        "--tts"' in text or '--tts"' in text
        assert "if use_local_tts:" in text
        assert "cmd.append(\"--tts\")" in text
        # Old inverted default (TTS on unless --no-tts) must be gone.
        assert "if not args.no_tts" not in text
        assert "args.no_tts" not in text
        # Default path is host voice.
        assert "/voice/" in text or "host TTS" in text or "host-owned" in text.lower()


def test_voice_stream_aligns_host_pcm_to_transcript() -> None:
    """Core lip-lock: host transcript + host PCM → emitted viseme spans."""
    stream = VoiceStream(StreamConfig(sample_rate=24_000))
    queued = stream.expect("Hello there")
    assert queued >= 2
    # Feed enough voiced audio for the aligner to spend budget.
    audio = _pcm16_tone(0.8)
    emitted: list = []
    # Chunk like a realtime host TTS would.
    step = 24_000 // 50 * 2  # 20 ms of pcm16
    for i in range(0, len(audio), step):
        emitted.extend(stream.feed(audio[i : i + step]))
    emitted.extend(stream.finish())
    stats = stream.stats()
    assert stats.received_seconds > 0.5
    assert emitted, "host PCM + transcript must emit at least one viseme span"
    assert stats.emitted >= 1
    span = emitted[0]
    assert span.phoneme
    assert span.end >= span.start


def test_facebridge_host_voice_http_roundtrip() -> None:
    """Live FaceBridge HTTP: /voice/expect → /pcm → /end (product default path)."""
    calls: list[tuple[str, dict]] = []

    def voice_handler(kind: str, payload: dict) -> dict:
        calls.append((kind, dict(payload)))
        if kind == "expect":
            return {"expecting": 4, "pending": 4}
        if kind == "pcm":
            assert isinstance(payload.get("audio"), (bytes, bytearray))
            assert len(payload["audio"]) > 0
            return {"emitted": 2, "pending": 2, "chunks": 1}
        if kind == "end":
            return {"ok": True, "pending": 0}
        if kind == "timeline":
            return {"scheduled": len(payload.get("spans") or []), "mode": "timeline"}
        return {"ok": True}

    bridge = FaceBridge(
        status_provider=lambda: {"ok": True},
        preview_provider=lambda: b"",
        screenshot_provider=lambda: b"",
        speak_handler=lambda _t: None,
        voice_handler=voice_handler,
        token="host-voice-token",
        host="127.0.0.1",
        port=0,
        cors_origins="*",
        job_timeout=2.0,
    )
    bridge.start()
    assert bridge._server is not None
    port = int(bridge._server.server_address[1])
    base = f"http://127.0.0.1:{port}"
    cid = "11111111-2222-3333-4444-555555555555"

    try:
        # Product helper against a real FaceBridge (not a stub server).
        result = drive_host_voice(
            "Hello from the host LLM",
            _pcm16_tone(0.25),
            base_url=base,
            token="host-voice-token",
            client_id=cid,
            sample_rate=24_000,
            timeout_s=3.0,
            raise_on_error=True,
        )
        assert result.ok is True
        kinds = [k for k, _ in calls]
        assert "expect" in kinds
        assert "pcm" in kinds
        assert "end" in kinds
        expect_payload = next(p for k, p in calls if k == "expect")
        assert expect_payload["text"] == "Hello from the host LLM"
        pcm_payload = next(p for k, p in calls if k == "pcm")
        assert pcm_payload["format"] == "pcm16"
        assert int(pcm_payload["sample_rate"]) == 24_000

        # Timeline alt path (host-timed phonemes).
        conn = HTTPConnection("127.0.0.1", port, timeout=3.0)
        timeline = json.dumps(
            {
                "caption": "Hi",
                "spans": [{"phoneme": "OU", "start": 0.0, "end": 0.12}],
            }
        ).encode("utf-8")
        conn.request(
            "POST",
            "/voice/timeline",
            body=timeline,
            headers={
                "Authorization": "Bearer host-voice-token",
                "X-ChorusFace-Client-Id": cid,
                "Content-Type": "application/json",
                "Content-Length": str(len(timeline)),
            },
        )
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body.get("mode") == "timeline"
        assert body.get("scheduled") == 1

        # Mouth-cue speak must not require local TTS — still queues text only.
        spoken: list[str] = []
        bridge._speak_handler = spoken.append
        stop = threading.Event()

        def _drain() -> None:
            while not stop.is_set():
                bridge.service()
                stop.wait(0.01)

        worker = threading.Thread(target=_drain, daemon=True)
        worker.start()
        try:
            cue = speak(
                "Mouth cue only",
                base_url=base,
                token="host-voice-token",
                client_id=cid,
                timeout_s=2.0,
                activate_first=False,
            )
            assert cue.ok is True
            for _ in range(50):
                if spoken:
                    break
                stop.wait(0.02)
            assert spoken == ["Mouth cue only"]
        finally:
            stop.set()
            worker.join(timeout=1.0)
    finally:
        bridge.stop()


def test_health_advertises_host_voice_contract() -> None:
    bridge = FaceBridge(
        status_provider=lambda: {"ok": True},
        preview_provider=lambda: b"",
        screenshot_provider=lambda: b"",
        speak_handler=lambda _t: None,
        voice_handler=lambda k, p: {"ok": True},
        token="t",
        host="127.0.0.1",
        port=0,
    )
    bridge.start()
    assert bridge._server is not None
    port = int(bridge._server.server_address[1])
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=3.0)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body["ok"] is True
        # Contract surface hosts discover without reading source.
        assert body.get("host_voice") == "/voice/expect|/voice/pcm|/voice/end"
        assert body.get("voice_timeline") == "/voice/timeline"
        assert body.get("prism_speak") == "/prism/speak"
        assert body.get("local_tts_default") is False
    finally:
        bridge.stop()


def test_docs_and_readme_state_host_owns_tts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    product = (ROOT / "docs" / "ProductBeta.md").read_text(encoding="utf-8")
    readme_l = readme.lower()
    assert "/voice/expect" in readme
    assert "host owns" in readme_l and ("tts" in readme_l or "voice" in readme_l)
    assert "docs/ai-overview.md" in readme
    assert "docs/llm-context.md" in readme
    assert "product default" in product.lower()
    assert re.search(r"Host-owned TTS|/voice/\*", product)
    # Insightits must not mute host Web Speech when face is queued.
    voice_chat = Path(
        r"C:\code\InsightitsAIAgent\public\js\voice-chat.js"
    )
    if voice_chat.is_file():
        js = voice_chat.read_text(encoding="utf-8")
        assert "options.chorusfaceSpeakQueued" not in js
        assert "do not mute host audio" in js or "host TTS" in js
