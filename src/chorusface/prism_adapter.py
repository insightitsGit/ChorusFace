"""PrismAPI AI-to-AI adapter → FaceBridge /speak.

Integrators should drive the face through PrismAPI-shaped speak intents
(or ``POST /prism/speak``). This module:

1. Normalizes speak payloads (text / message / response / speech)
2. Forwards them to FaceBridge (``host_client.speak``)
3. Optionally registers a PrismAPI provider when ``prism.api`` is installed

The GPU face process stays authoritative; PrismAPI is the AI↔AI front door.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from chorusface.host_client import SpeakResult, speak, speak_async
from chorusface.service.bridge import speak_text_from_payload

LOGGER = logging.getLogger("chorusface.prism_adapter")

DEFAULT_BRIDGE_URL: Final = "http://127.0.0.1:8766"
SPEAK_SCHEMA: Final = "chorusface.prism.speak.v1"


@dataclass(frozen=True, slots=True)
class SpeakIntent:
    """Normalized AI-to-AI speak turn."""

    text: str
    schema: str = SPEAK_SCHEMA
    request_id: str = ""
    emotion: str = ""
    source: str = "prism"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SpeakIntent:
        text = speak_text_from_payload(payload)
        if not text:
            # Nested Prism-style envelope: { "speak": { "text": "..." } }
            nested = payload.get("speak") or payload.get("intent") or {}
            if isinstance(nested, dict):
                text = speak_text_from_payload(nested)
        return cls(
            text=text,
            request_id=str(
                payload.get("requestId")
                or payload.get("request_id")
                or payload.get("id")
                or ""
            ),
            emotion=str(payload.get("emotion") or "").strip(),
            source=str(payload.get("source") or "prism"),
        )


def forward_speak(
    intent: SpeakIntent | dict[str, Any] | str,
    *,
    base_url: str | None = None,
    token: str | None = None,
    async_: bool = False,
) -> SpeakResult:
    """Map a Prism speak intent onto FaceBridge ``/speak`` (or ``/prism/speak``)."""
    if isinstance(intent, str):
        parsed = SpeakIntent(text=intent.strip())
    elif isinstance(intent, dict):
        parsed = SpeakIntent.from_payload(intent)
    else:
        parsed = intent
    if not parsed.text:
        return SpeakResult(ok=False, status=0, queued=False, text="", error="empty text")

    url = (base_url or os.environ.get("CHORUSFACE_BRIDGE_URL") or DEFAULT_BRIDGE_URL).rstrip(
        "/"
    )
    # Prefer Prism-named route; FaceBridge aliases both.
    speak_url = url  # host_client appends /speak; prism path is optional
    if async_:
        speak_async(parsed.text, base_url=speak_url, token=token)
        return SpeakResult(ok=True, status=202, queued=True, text=parsed.text)
    return speak(parsed.text, base_url=speak_url, token=token)


def try_register_prism_provider(
    *,
    bridge_url: str | None = None,
    token: str | None = None,
) -> bool:
    """Register a PrismAPI provider that exposes face speak, if prism.api exists.

    Returns True when registration succeeded. Safe no-op when prismlib-plus is
    not installed — FaceBridge ``/prism/speak`` remains the HTTP contract.
    """
    try:
        from prism.api import provider  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        LOGGER.info("prism.api not installed — using HTTP /prism/speak only")
        return False

    base = bridge_url or os.environ.get("CHORUSFACE_BRIDGE_URL") or DEFAULT_BRIDGE_URL
    auth = token or os.environ.get("CHORUSFACE_BRIDGE_TOKEN") or "chorusface-beta"

    @provider.expose  # type: ignore[misc]
    def chorusface_speak(text: str = "", message: str = "", response: str = "") -> dict[str, Any]:
        """AI-to-AI: speak assistant text on the shared ChorusFace instance."""
        result = forward_speak(
            {"text": text, "message": message, "response": response},
            base_url=base,
            token=auth,
        )
        return {
            "ok": result.ok,
            "queued": result.queued,
            "text": result.text,
            "error": result.error,
            "schema": SPEAK_SCHEMA,
        }

    LOGGER.info("Registered PrismAPI provider chorusface_speak → %s", base)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forward a PrismAPI-shaped speak intent to FaceBridge"
    )
    parser.add_argument("text", nargs="*", help="Spoken text")
    parser.add_argument("--json", help="JSON speak payload file or '-' for stdin")
    parser.add_argument(
        "--url",
        default=os.environ.get("CHORUSFACE_BRIDGE_URL", DEFAULT_BRIDGE_URL),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CHORUSFACE_BRIDGE_TOKEN", "chorusface-beta"),
    )
    parser.add_argument(
        "--register-provider",
        action="store_true",
        help="Try to register prism.api provider then exit",
    )
    args = parser.parse_args(argv)

    if args.register_provider:
        ok = try_register_prism_provider(bridge_url=args.url, token=args.token)
        print("provider registered" if ok else "provider unavailable (HTTP-only)")
        return 0 if ok else 1

    payload: dict[str, Any]
    if args.json:
        raw = sys.stdin.read() if args.json == "-" else Path(args.json).read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            print("JSON object required", file=sys.stderr)
            return 2
    else:
        payload = {"text": " ".join(args.text)}

    result = forward_speak(payload, base_url=args.url, token=args.token)
    if result.ok:
        print(json.dumps({"ok": True, "queued": result.queued, "text": result.text}))
        return 0
    print(json.dumps({"ok": False, "error": result.error}), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SPEAK_SCHEMA",
    "SpeakIntent",
    "forward_speak",
    "main",
    "try_register_prism_provider",
]
