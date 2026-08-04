"""Thin host → FaceBridge client (LLM stays outside ChorusFace).

Hosts POST complete assistant text to ``/speak``. Failures never raise by
default so a down face cannot break the chat product.
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
from typing import Any, Final

LOGGER = logging.getLogger("chorusface.host_client")

DEFAULT_BASE_URL: Final = "http://127.0.0.1:8766"
DEFAULT_TIMEOUT_S: Final = 2.5


@dataclass(frozen=True, slots=True)
class SpeakResult:
    ok: bool
    status: int
    queued: bool
    text: str
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


def activate(
    *,
    base_url: str | None = None,
    token: str | None = None,
    client_id: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Bind this API key exclusively to one client_id (one AI / one system)."""
    import uuid

    url = (base_url or _env_base_url()).rstrip("/") + "/auth/activate"
    auth = str(token if token is not None else _env_token()).strip()
    cid = str(client_id if client_id is not None else _client_id() or uuid.uuid4())
    payload = json.dumps({"client_id": cid}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {auth}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-ChorusFace-Client-Id": cid,
        },
    )
    with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
        raw = response.read()
        body = json.loads(raw.decode("utf-8")) if raw else {}
        if isinstance(body, dict):
            body.setdefault("client_id", cid)
            return body
        return {"ok": True, "client_id": cid}


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
    """Queue assistant text on the face. Best-effort unless raise_on_error."""
    spoken = str(text or "").strip()
    if not spoken:
        result = SpeakResult(ok=False, status=0, queued=False, text="", error="empty text")
        if raise_on_error:
            raise ValueError(result.error)
        return result

    import uuid

    url = (base_url or _env_base_url()).rstrip("/") + "/speak"
    auth = str(token if token is not None else _env_token()).strip()
    cid = str(client_id if client_id is not None else _client_id()).strip()
    if not cid:
        cid = str(uuid.uuid4())
        os.environ.setdefault("CHORUSFACE_CLIENT_ID", cid)
    if activate_first:
        try:
            activate(base_url=base_url, token=auth, client_id=cid, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("chorusface activate failed: %s", exc)
    payload = json.dumps({"text": spoken}).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {auth}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if cid:
        headers["X-ChorusFace-Client-Id"] = cid
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers=headers,
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
    """Fire-and-forget speak on a daemon thread."""
    import threading

    def _worker() -> None:
        speak(text, base_url=base_url, token=token, timeout_s=timeout_s)

    threading.Thread(target=_worker, name="chorusface-host-speak", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test FaceBridge /speak from a host product"
    )
    parser.add_argument("text", nargs="+", help="Assistant text to speak")
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
    args = parser.parse_args(argv)
    result = speak(
        " ".join(args.text),
        base_url=args.url,
        token=args.token,
        timeout_s=float(args.timeout),
        raise_on_error=False,
    )
    if result.ok:
        print(f"OK queued={result.queued} text={result.text!r}")
        return 0
    print(f"FAIL status={result.status} error={result.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_S",
    "SpeakResult",
    "activate",
    "main",
    "speak",
    "speak_async",
]
