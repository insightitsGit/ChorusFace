"""Face control bridge — NWR's loopback HTTP pattern, for the avatar.

GPU work stays on the render thread. External clients (agents, voice drivers,
test harnesses) observe and speak through validated jobs that
:meth:`FaceBridge.service` drains each frame.

The voice routes are the exception, and deliberately so. ``/voice/pcm`` carries
audio arriving in real time from whatever is speaking — a realtime model, a
softphone, a file replay — and making each 20 ms chunk wait for a frame would
tie the audio clock to the render loop. Alignment is pure signal processing that
touches no GPU state, so those calls run on the request thread and hand the
render loop nothing but timed viseme events.

Cell-cluster control is available via ``GET /cells`` and ``POST /cells/drive``
(±4 velocity on unlocked cells only). There is still no route that resets,
saves, or loads a world, so a token holder cannot overwrite locked identity.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import secrets
import socket
import threading
from dataclasses import dataclass, field
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Final
from urllib.parse import parse_qs

LOGGER: Final = logging.getLogger("aiface.bridge")
MAX_BODY_BYTES: Final = 64 * 1024
MAX_PENDING_JOBS: Final = 64
DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8766
DEFAULT_JOB_TIMEOUT: Final = 8.0


# An empty host means "every interface" to the socket layer, same as 0.0.0.0.
_WILDCARD_HOSTS: Final[frozenset[str]] = frozenset({"", "*"})


def _is_loopback_literal(text: str) -> bool | None:
    """True/False for an IP literal, or None when ``text`` is not one."""
    try:
        return ipaddress.ip_address(text.split("%", 1)[0]).is_loopback
    except ValueError:
        return None


def is_loopback_host(host: object) -> bool:
    """Whether binding ``host`` keeps the bridge unreachable from the network.

    Mirrors NWR's ``net_guard`` policy, which owns this rule: names are resolved
    and every address they answer with has to be a loopback address, so a name
    pointing at both ``127.0.0.1`` and a routable address is not trusted. The
    guard fails closed — a name that will not resolve is treated as remote.
    """
    text = str(host or "").strip()
    if text.lower() in _WILDCARD_HOSTS:
        return False
    literal = _is_loopback_literal(text.strip("[]"))
    if literal is not None:
        return literal
    try:
        infos = socket.getaddrinfo(text, None)
    except (socket.gaierror, UnicodeError):
        return False
    resolved = {str(info[4][0]) for info in infos}
    if not resolved:
        return False
    return all(_is_loopback_literal(address) for address in resolved)

StatusProvider = Callable[[], dict[str, Any]]
BytesProvider = Callable[[], bytes]
SpeakHandler = Callable[[str], None]
#: Live mouth ownership + SSBO disc metrics (Path A probe).
ProbeProvider = Callable[[], dict[str, Any]]
#: Controllable cell-cluster index summary.
CellsProvider = Callable[[], dict[str, Any]]
#: ``payload -> {queued, impulses, ...}`` — per-cell / neighbor / cluster drive.
CellsDriveHandler = Callable[[dict[str, Any]], dict[str, Any]]
#: ``(kind, payload) -> response``, where kind is ``expect``, ``pcm`` or ``end``.
VoiceHandler = Callable[[str, dict[str, Any]], dict[str, Any]]

#: Audio layouts a caller may push. Both are mono; anything else is refused
#: rather than guessed at, because a wrong guess desynchronises the whole face.
PCM_FORMATS: Final[frozenset[str]] = frozenset({"pcm16", "float32"})


class BridgeError(Exception):
    """An HTTP-visible failure with an explicit status code."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(slots=True)
class _Job:
    kind: str
    event: threading.Event = field(default_factory=threading.Event)
    payload: Any = None
    error: str | None = None
    request: Any = None


class FaceBridge:
    """Loopback HTTP surface for observing and driving the live face."""

    def __init__(
        self,
        *,
        status_provider: StatusProvider,
        preview_provider: BytesProvider,
        screenshot_provider: BytesProvider,
        speak_handler: SpeakHandler,
        token: str,
        voice_handler: VoiceHandler | None = None,
        probe_provider: ProbeProvider | None = None,
        cells_provider: CellsProvider | None = None,
        cells_drive_handler: CellsDriveHandler | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        job_timeout: float = DEFAULT_JOB_TIMEOUT,
        allow_remote_bind: bool = False,
    ) -> None:
        if not token:
            raise ValueError("A non-empty access token is required")
        if not allow_remote_bind and not is_loopback_host(host):
            raise ValueError(
                f"Refusing to bind the face bridge to {host!r}: it is reachable "
                "from the network and would hand the face to anyone who can reach "
                "the port. Bind 127.0.0.1, or pass --allow-remote-bind if that is "
                "what you want."
            )
        self._status_provider = status_provider
        self._preview_provider = preview_provider
        self._screenshot_provider = screenshot_provider
        self._speak_handler = speak_handler
        self._voice_handler = voice_handler
        self._probe_provider = probe_provider
        self._cells_provider = cells_provider
        self._cells_drive_handler = cells_drive_handler
        self._token = token
        self._host = host
        self._port = int(port)
        self._job_timeout = float(job_timeout)
        self._allow_remote_bind = bool(allow_remote_bind)
        self._lock = threading.Lock()
        self._jobs: list[_Job] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._accepted = 0

    @property
    def token(self) -> str:
        return self._token

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        if self._server is not None:
            return
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            #: Whether this request's body has already been taken off the socket.
            _body_consumed = False

            def log_message(self, format: str, *args: object) -> None:
                LOGGER.debug("%s - %s", self.address_string(), format % args)

            def _auth(self) -> None:
                header = self.headers.get("Authorization", "")
                expected = f"Bearer {bridge._token}"
                # Constant-time compare: NWR uses the same pattern so a long
                # token cannot be walked character by character over the wire.
                if not compare_digest(header, expected):
                    raise BridgeError(HTTPStatus.UNAUTHORIZED, "invalid token")

            def _read_json(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise BridgeError(
                        HTTPStatus.BAD_REQUEST, "malformed Content-Length"
                    ) from exc
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise BridgeError(HTTPStatus.BAD_REQUEST, "body size out of range")
                self._body_consumed = True
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise BridgeError(HTTPStatus.BAD_REQUEST, f"bad json: {exc}") from exc
                if not isinstance(payload, dict):
                    raise BridgeError(HTTPStatus.BAD_REQUEST, "json object required")
                return payload

            def _read_body(self) -> bytes:
                """Read a raw body, for routes whose payload is not json."""
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise BridgeError(
                        HTTPStatus.BAD_REQUEST, "malformed Content-Length"
                    ) from exc
                if length < 0 or length > MAX_BODY_BYTES:
                    raise BridgeError(HTTPStatus.BAD_REQUEST, "body size out of range")
                self._body_consumed = True
                return self.rfile.read(length) if length else b""

            def _discard_body(self) -> None:
                """Swallow a body we are not going to read before answering.

                A reply sent while the request is still half-delivered leaves the
                socket out of step, and the client sees the connection drop
                instead of the error explaining what it did wrong.
                """
                if self._body_consumed:
                    return
                self._body_consumed = True
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                remaining = min(max(length, 0), MAX_BODY_BYTES)
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 65536))
                    if not chunk:
                        break
                    remaining -= len(chunk)

            def _query(self) -> dict[str, str]:
                _, _, query = self.path.partition("?")
                return {
                    key: values[0]
                    for key, values in parse_qs(query, keep_blank_values=True).items()
                }

            def _voice(self, kind: str, payload: dict[str, Any]) -> None:
                if bridge._voice_handler is None:
                    raise BridgeError(
                        HTTPStatus.SERVICE_UNAVAILABLE, "voice channel is not open"
                    )
                try:
                    result = bridge._voice_handler(kind, payload)
                except BridgeError:
                    raise
                except ValueError as exc:
                    raise BridgeError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
                self._send_json(HTTPStatus.OK, result)

            def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                self._send(
                    status,
                    json.dumps(payload).encode("utf-8"),
                    "application/json",
                )

            def do_GET(self) -> None:  # noqa: N802
                try:
                    self._auth()
                    path = self.path.split("?", 1)[0]
                    if path == "/health":
                        self._send_json(HTTPStatus.OK, {"ok": True, "service": "aiface"})
                        return
                    if path == "/status":
                        self._send_json(
                            HTTPStatus.OK, bridge._run_job("status")
                        )
                        return
                    if path == "/probe":
                        if bridge._probe_provider is None:
                            raise BridgeError(
                                HTTPStatus.NOT_FOUND, "probe not available"
                            )
                        self._send_json(
                            HTTPStatus.OK, bridge._run_job("probe")
                        )
                        return
                    if path == "/cells":
                        if bridge._cells_provider is None:
                            raise BridgeError(
                                HTTPStatus.NOT_FOUND, "cells not available"
                            )
                        self._send_json(
                            HTTPStatus.OK, bridge._run_job("cells")
                        )
                        return
                    if path == "/preview":
                        data = bridge._run_job("preview")
                        self._send(HTTPStatus.OK, data, "image/png")
                        return
                    if path == "/screenshot":
                        data = bridge._run_job("screenshot")
                        self._send(HTTPStatus.OK, data, "image/png")
                        return
                    raise BridgeError(HTTPStatus.NOT_FOUND, f"unknown path {path}")
                except BridgeError as exc:
                    self._discard_body()
                    self._send_json(exc.status, {"error": exc.message})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    self._auth()
                    path = self.path.split("?", 1)[0]
                    if path == "/speak":
                        payload = self._read_json()
                        text = str(payload.get("text", "")).strip()
                        if not text:
                            raise BridgeError(HTTPStatus.BAD_REQUEST, "text required")
                        bridge._run_job("speak", request=text)
                        self._send_json(HTTPStatus.OK, {"queued": True, "text": text})
                        return
                    if path == "/cells/drive":
                        if bridge._cells_drive_handler is None:
                            raise BridgeError(
                                HTTPStatus.NOT_FOUND, "cells drive not available"
                            )
                        payload = self._read_json()
                        result = bridge._run_job("cells_drive", request=payload)
                        self._send_json(HTTPStatus.OK, result)
                        return
                    if path == "/voice/expect":
                        payload = self._read_json()
                        text = str(payload.get("text", "")).strip()
                        if not text:
                            raise BridgeError(HTTPStatus.BAD_REQUEST, "text required")
                        self._voice(
                            "expect",
                            {
                                "text": text,
                                "emotion": str(payload.get("emotion", "")).strip(),
                                "sample_rate": payload.get("sample_rate"),
                            },
                        )
                        return
                    if path == "/voice/timeline":
                        # Host already owns audio + phoneme timing. AIFace only
                        # needs the absolute span list to drive the mouth.
                        payload = self._read_json()
                        spans = payload.get("spans")
                        if not isinstance(spans, list) or not spans:
                            raise BridgeError(
                                HTTPStatus.BAD_REQUEST, "spans required"
                            )
                        self._voice(
                            "timeline",
                            {
                                "spans": spans,
                                "emotion": str(payload.get("emotion", "")).strip(),
                                "caption": str(
                                    payload.get("caption") or payload.get("text") or ""
                                ).strip(),
                            },
                        )
                        return
                    if path == "/voice/pcm":
                        query = self._query()
                        layout = (query.get("format") or "pcm16").lower()
                        if layout not in PCM_FORMATS:
                            raise BridgeError(
                                HTTPStatus.BAD_REQUEST,
                                f"format must be one of {sorted(PCM_FORMATS)}",
                            )
                        try:
                            rate = int(query["rate"]) if "rate" in query else 0
                        except ValueError as exc:
                            raise BridgeError(
                                HTTPStatus.BAD_REQUEST, "rate must be an integer"
                            ) from exc
                        self._voice(
                            "pcm",
                            {
                                "audio": self._read_body(),
                                "sample_rate": rate,
                                "format": layout,
                            },
                        )
                        return
                    if path == "/voice/end":
                        self._discard_body()
                        self._voice("end", {})
                        return
                    raise BridgeError(HTTPStatus.NOT_FOUND, f"unknown path {path}")
                except BridgeError as exc:
                    self._discard_body()
                    self._send_json(exc.status, {"error": exc.message})

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="aiface-bridge",
            daemon=True,
        )
        self._thread.start()
        if self._allow_remote_bind and not is_loopback_host(self._host):
            LOGGER.warning(
                "Face bridge is reachable off this machine on %s", self.url
            )
        LOGGER.info("Face bridge listening on %s", self.url)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def service(self) -> None:
        """Drain queued GPU jobs on the render thread. Call once per frame."""
        with self._lock:
            jobs = list(self._jobs)
            self._jobs.clear()
        for job in jobs:
            try:
                if job.kind == "status":
                    job.payload = self._status_provider()
                elif job.kind == "probe":
                    if self._probe_provider is None:
                        raise RuntimeError("probe provider missing")
                    job.payload = self._probe_provider()
                elif job.kind == "preview":
                    job.payload = self._preview_provider()
                elif job.kind == "screenshot":
                    job.payload = self._screenshot_provider()
                elif job.kind == "speak":
                    self._speak_handler(str(job.request))
                    job.payload = {"queued": True}
                elif job.kind == "cells":
                    if self._cells_provider is None:
                        raise RuntimeError("cells provider missing")
                    job.payload = self._cells_provider()
                elif job.kind == "cells_drive":
                    if self._cells_drive_handler is None:
                        raise RuntimeError("cells drive handler missing")
                    if not isinstance(job.request, dict):
                        raise RuntimeError("cells drive requires a json object")
                    job.payload = self._cells_drive_handler(job.request)
                else:
                    job.error = f"unknown job {job.kind}"
            except Exception as exc:  # noqa: BLE001 — surface to HTTP client
                job.error = str(exc)
            finally:
                job.event.set()

    def _run_job(self, kind: str, request: Any = None) -> Any:
        job = _Job(kind=kind, request=request)
        with self._lock:
            # The render thread drains this list once a frame. Without a ceiling
            # an authenticated flood would grow it faster than frames retire it.
            if len(self._jobs) >= MAX_PENDING_JOBS:
                raise BridgeError(
                    HTTPStatus.SERVICE_UNAVAILABLE, "face bridge is saturated"
                )
            self._jobs.append(job)
            self._accepted += 1
        if not job.event.wait(timeout=self._job_timeout):
            raise BridgeError(HTTPStatus.GATEWAY_TIMEOUT, f"{kind} timed out")
        if job.error:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, job.error)
        return job.payload


def new_token() -> str:
    return secrets.token_urlsafe(24)


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "MAX_PENDING_JOBS",
    "PCM_FORMATS",
    "BridgeError",
    "CellsDriveHandler",
    "CellsProvider",
    "FaceBridge",
    "VoiceHandler",
    "is_loopback_host",
    "new_token",
]
