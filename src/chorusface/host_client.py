"""Thin host → FaceBridge client (LLM + TTS stay outside ChorusFace).

**Product default:** the host owns speech audio. Drive the face with
``/voice/expect`` + ``/voice/pcm`` + ``/voice/end``, or ``/voice/timeline`` when
you already timed phonemes. See ``docs/VoiceSync.md``.

``/speak`` / ``/prism/speak`` only cues mouth motion from text (no ChorusFace
audio). Local ``--tts`` on the face process is lab-only.

Failures never raise by default so a down face cannot break the chat product.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

LOGGER = logging.getLogger("chorusface.host_client")

DEFAULT_BASE_URL: Final = "http://127.0.0.1:8766"
DEFAULT_TIMEOUT_S: Final = 2.5
HEADER_CLIENT_ID: Final = "X-ChorusFace-Client-Id"


@dataclass(frozen=True, slots=True)
class SpeakResult:
    ok: bool
    status: int
    queued: bool
    text: str
    error: str = ""
    body: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class VoiceResult:
    ok: bool
    status: int
    error: str = ""
    body: dict[str, Any] | None = None


def _env_base_url() -> str:
    return (
        os.environ.get("CHORUSFACE_BRIDGE_URL")
        or os.environ.get("CHORUSFACE_HOST_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _env_token() -> str:
    return (
        os.environ.get("CHORUSFACE_BRIDGE_TOKEN")
        or os.environ.get("CHORUSFACE_HOST_TOKEN")
        or "chorusface-beta"
    )


def _client_id() -> str:
    return (
        os.environ.get("CHORUSFACE_CLIENT_ID")
        or os.environ.get("CHORUSFACE_HOST_CLIENT_ID")
        or ""
    ).strip()


def _resolve_client_id(client_id: str | None) -> str:
    import uuid

    cid = str(client_id if client_id is not None else _client_id()).strip()
    if not cid:
        cid = str(uuid.uuid4())
        os.environ.setdefault("CHORUSFACE_CLIENT_ID", cid)
    return cid


def _auth_headers(
    token: str,
    client_id: str,
    *,
    content_type: str | None = "application/json",
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        HEADER_CLIENT_ID: client_id,
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _http_json(
    method: str,
    url: str,
    *,
    token: str,
    client_id: str,
    payload: Mapping[str, Any] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=_auth_headers(token, client_id),
    )
    with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
        raw = response.read()
        status = int(getattr(response, "status", 200) or 200)
        body: dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(parsed, dict):
                    body = parsed
            except json.JSONDecodeError:
                body = {}
        return status, body


def activate(
    *,
    base_url: str | None = None,
    token: str | None = None,
    client_id: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Bind this API key exclusively to one client_id (one AI / one system)."""
    url = (base_url or _env_base_url()).rstrip("/") + "/auth/activate"
    auth = str(token if token is not None else _env_token()).strip()
    cid = _resolve_client_id(client_id)
    status, body = _http_json(
        "POST",
        url,
        token=auth,
        client_id=cid,
        payload={"client_id": cid},
        timeout_s=timeout_s,
    )
    if status >= 400:
        raise RuntimeError(f"activate failed HTTP {status}: {body}")
    body.setdefault("client_id", cid)
    body.setdefault("ok", True)
    return body


def speak(
    text: str,
    *,
    base_url: str | None = None,
    token: str | None = None,
    client_id: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    raise_on_error: bool = False,
    activate_first: bool = True,
) -> SpeakResult:
    """Cue mouth motion from text only (no ChorusFace audio).

    Prefer :func:`drive_host_voice` when the host already has TTS PCM.
    """
    spoken = str(text or "").strip()
    if not spoken:
        result = SpeakResult(ok=False, status=0, queued=False, text="", error="empty text")
        if raise_on_error:
            raise ValueError(result.error)
        return result

    auth = str(token if token is not None else _env_token()).strip()
    cid = _resolve_client_id(client_id)
    root = (base_url or _env_base_url()).rstrip("/")
    if activate_first:
        try:
            activate(base_url=root, token=auth, client_id=cid, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("chorusface activate failed: %s", exc)
    try:
        status, body = _http_json(
            "POST",
            root + "/prism/speak",
            token=auth,
            client_id=cid,
            payload={"text": spoken},
            timeout_s=timeout_s,
        )
        queued = bool(body.get("queued", status < 400))
        return SpeakResult(
            ok=status < 400 and queued,
            status=status,
            queued=queued,
            text=str(body.get("text") or spoken),
            body=body,
        )
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:240]
        except Exception:  # noqa: BLE001
            detail = str(exc.reason or exc)
        result = SpeakResult(
            ok=False,
            status=int(exc.code),
            queued=False,
            text=spoken,
            error=detail or f"HTTP {exc.code}",
        )
    except Exception as exc:  # noqa: BLE001 — face-down must not break host chat
        result = SpeakResult(
            ok=False,
            status=0,
            queued=False,
            text=spoken,
            error=str(exc),
        )

    LOGGER.warning("chorusface speak failed: %s", result.error)
    if raise_on_error:
        raise RuntimeError(result.error)
    return result


def speak_async(
    text: str,
    *,
    base_url: str | None = None,
    token: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> None:
    """Fire-and-forget text mouth-cue on a daemon thread."""
    import threading

    def _worker() -> None:
        speak(text, base_url=base_url, token=token, timeout_s=timeout_s)

    threading.Thread(target=_worker, name="chorusface-host-speak", daemon=True).start()


def voice_expect(
    text: str,
    *,
    base_url: str | None = None,
    token: str | None = None,
    client_id: str | None = None,
    sample_rate: int = 24000,
    emotion: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    activate_first: bool = True,
    raise_on_error: bool = False,
) -> VoiceResult:
    """Tell the face which words the host voice is about to speak."""
    spoken = str(text or "").strip()
    if not spoken:
        result = VoiceResult(ok=False, status=0, error="empty text")
        if raise_on_error:
            raise ValueError(result.error)
        return result
    auth = str(token if token is not None else _env_token()).strip()
    cid = _resolve_client_id(client_id)
    root = (base_url or _env_base_url()).rstrip("/")
    if activate_first:
        try:
            activate(base_url=root, token=auth, client_id=cid, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("chorusface activate failed: %s", exc)
    payload: dict[str, Any] = {"text": spoken, "sample_rate": int(sample_rate)}
    if emotion:
        payload["emotion"] = str(emotion)
    return _voice_json(
        root + "/voice/expect",
        token=auth,
        client_id=cid,
        payload=payload,
        timeout_s=timeout_s,
        raise_on_error=raise_on_error,
    )


def voice_pcm(
    audio: bytes,
    *,
    base_url: str | None = None,
    token: str | None = None,
    client_id: str | None = None,
    sample_rate: int = 24000,
    fmt: str = "pcm16",
    timeout_s: float = 30.0,
    raise_on_error: bool = False,
) -> VoiceResult:
    """Push host TTS PCM (product-default lip-lock path)."""
    if not audio:
        result = VoiceResult(ok=False, status=0, error="empty audio")
        if raise_on_error:
            raise ValueError(result.error)
        return result
    auth = str(token if token is not None else _env_token()).strip()
    cid = _resolve_client_id(client_id)
    root = (base_url or _env_base_url()).rstrip("/")
    layout = str(fmt or "pcm16").lower()
    url = f"{root}/voice/pcm?format={layout}&rate={int(sample_rate)}"
    request = urllib.request.Request(
        url,
        data=audio,
        method="POST",
        headers=_auth_headers(auth, cid, content_type="application/octet-stream"),
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
            raw = response.read()
            status = int(getattr(response, "status", 200) or 200)
            body: dict[str, Any] = {}
            if raw:
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                    if isinstance(parsed, dict):
                        body = parsed
                except json.JSONDecodeError:
                    body = {}
            return VoiceResult(ok=status < 400, status=status, body=body)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:240]
        except Exception:  # noqa: BLE001
            detail = str(exc.reason or exc)
        result = VoiceResult(
            ok=False, status=int(exc.code), error=detail or f"HTTP {exc.code}"
        )
    except Exception as exc:  # noqa: BLE001
        result = VoiceResult(ok=False, status=0, error=str(exc))
    LOGGER.warning("chorusface voice_pcm failed: %s", result.error)
    if raise_on_error:
        raise RuntimeError(result.error)
    return result


def voice_end(
    *,
    base_url: str | None = None,
    token: str | None = None,
    client_id: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    raise_on_error: bool = False,
) -> VoiceResult:
    """Close the current host-voice utterance."""
    auth = str(token if token is not None else _env_token()).strip()
    cid = _resolve_client_id(client_id)
    root = (base_url or _env_base_url()).rstrip("/")
    return _voice_json(
        root + "/voice/end",
        token=auth,
        client_id=cid,
        payload={},
        timeout_s=timeout_s,
        raise_on_error=raise_on_error,
    )


def voice_timeline(
    spans: Sequence[Mapping[str, Any]],
    *,
    caption: str = "",
    emotion: str = "",
    base_url: str | None = None,
    token: str | None = None,
    client_id: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    activate_first: bool = True,
    raise_on_error: bool = False,
) -> VoiceResult:
    """Host-timed phoneme spans — preferred when the LLM voice stack already aligned."""
    auth = str(token if token is not None else _env_token()).strip()
    cid = _resolve_client_id(client_id)
    root = (base_url or _env_base_url()).rstrip("/")
    if activate_first:
        try:
            activate(base_url=root, token=auth, client_id=cid, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("chorusface activate failed: %s", exc)
    payload: dict[str, Any] = {"spans": [dict(s) for s in spans]}
    if caption:
        payload["caption"] = str(caption)
    if emotion:
        payload["emotion"] = str(emotion)
    return _voice_json(
        root + "/voice/timeline",
        token=auth,
        client_id=cid,
        payload=payload,
        timeout_s=timeout_s,
        raise_on_error=raise_on_error,
    )


def drive_host_voice(
    text: str,
    audio: bytes,
    *,
    base_url: str | None = None,
    token: str | None = None,
    client_id: str | None = None,
    sample_rate: int = 24000,
    fmt: str = "pcm16",
    emotion: str = "",
    timeout_s: float = 30.0,
    raise_on_error: bool = False,
) -> VoiceResult:
    """Product-default path: host TTS bytes → face lip-lock.

    Plays nothing on ChorusFace — the host is already speaking this audio
    (or will). ChorusFace only moves the mouth to the PCM clock.
    """
    expect = voice_expect(
        text,
        base_url=base_url,
        token=token,
        client_id=client_id,
        sample_rate=sample_rate,
        emotion=emotion,
        timeout_s=min(timeout_s, DEFAULT_TIMEOUT_S),
        activate_first=True,
        raise_on_error=raise_on_error,
    )
    if not expect.ok:
        return expect
    pcm = voice_pcm(
        audio,
        base_url=base_url,
        token=token,
        client_id=client_id,
        sample_rate=sample_rate,
        fmt=fmt,
        timeout_s=timeout_s,
        raise_on_error=raise_on_error,
    )
    if not pcm.ok:
        return pcm
    return voice_end(
        base_url=base_url,
        token=token,
        client_id=client_id,
        timeout_s=min(timeout_s, DEFAULT_TIMEOUT_S),
        raise_on_error=raise_on_error,
    )


def _voice_json(
    url: str,
    *,
    token: str,
    client_id: str,
    payload: Mapping[str, Any],
    timeout_s: float,
    raise_on_error: bool,
) -> VoiceResult:
    try:
        status, body = _http_json(
            "POST",
            url,
            token=token,
            client_id=client_id,
            payload=payload,
            timeout_s=timeout_s,
        )
        return VoiceResult(ok=status < 400, status=status, body=body)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:240]
        except Exception:  # noqa: BLE001
            detail = str(exc.reason or exc)
        result = VoiceResult(
            ok=False, status=int(exc.code), error=detail or f"HTTP {exc.code}"
        )
    except Exception as exc:  # noqa: BLE001
        result = VoiceResult(ok=False, status=0, error=str(exc))
    LOGGER.warning("chorusface voice call failed: %s", result.error)
    if raise_on_error:
        raise RuntimeError(result.error)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Host → ChorusFace client. Default product path is host voice "
            "(--voice / --pcm-file). Plain text uses /prism/speak mouth-cue only."
        )
    )
    parser.add_argument("text", nargs="+", help="Assistant text")
    parser.add_argument(
        "--url",
        default=_env_base_url(),
        help="FaceBridge base URL (default CHORUSFACE_BRIDGE_URL)",
    )
    parser.add_argument(
        "--token",
        default=_env_token(),
        help="Bearer token (default CHORUSFACE_BRIDGE_TOKEN)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="HTTP timeout seconds",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Use host-voice path (requires --pcm-file with host TTS audio)",
    )
    parser.add_argument(
        "--pcm-file",
        type=str,
        default="",
        help="Raw pcm16 mono file from host TTS (with --voice)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=24000,
        help="PCM sample rate for --voice (default 24000)",
    )
    args = parser.parse_args(argv)
    text = " ".join(args.text)
    if args.voice:
        path = str(args.pcm_file or "").strip()
        if not path:
            print("FAIL: --voice requires --pcm-file <host-tts.pcm>", file=sys.stderr)
            return 2
        audio = open(path, "rb").read()
        result = drive_host_voice(
            text,
            audio,
            base_url=args.url,
            token=args.token,
            sample_rate=int(args.rate),
            timeout_s=max(float(args.timeout), 30.0),
            raise_on_error=False,
        )
        if result.ok:
            print(f"OK host-voice text={text!r} bytes={len(audio)}")
            return 0
        print(f"FAIL status={result.status} error={result.error}", file=sys.stderr)
        return 1

    result = speak(
        text,
        base_url=args.url,
        token=args.token,
        timeout_s=float(args.timeout),
        raise_on_error=False,
    )
    if result.ok:
        print(f"OK mouth-cue queued={result.queued} text={result.text!r}")
        print("(Host should play its own TTS; ChorusFace did not synthesize audio.)")
        return 0
    print(f"FAIL status={result.status} error={result.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_S",
    "HEADER_CLIENT_ID",
    "SpeakResult",
    "VoiceResult",
    "activate",
    "drive_host_voice",
    "main",
    "speak",
    "speak_async",
    "voice_end",
    "voice_expect",
    "voice_pcm",
    "voice_timeline",
]
