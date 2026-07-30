"""Local HTTP control bridge that lets an AI observe and command the world.

GPU work can only happen on the thread that owns the OpenGL context, so
requests that need world data are queued as jobs and fulfilled by the render
loop through :meth:`ControlBridge.service`.
"""

from __future__ import annotations

import io
import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Final

import numpy as np
import numpy.typing as npt

from ai_commands import (
    DEFAULT_BOUNDS,
    CommandError,
    Control,
    Operation,
    RemoveEntity,
    Segment,
    SpawnEntity,
    schema_for_authority,
)
from bds_format import ANCHORS, PRIORITY_LEVELS, PRIORITY_NAMES
from net_guard import require_loopback

LOGGER: Final = logging.getLogger("nwr.ai_bridge")
MAX_BODY_BYTES: Final = 256 * 1024
DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8765
DEFAULT_JOB_TIMEOUT: Final = 8.0
DEFAULT_QUEUE_LIMIT: Final = 65536

StateProvider = Callable[[], dict[str, Any]]
ScreenshotProvider = Callable[[], bytes]
StatusProvider = Callable[[], dict[str, Any]]


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


class ControlBridge:
    """Accepts validated commands and observation requests over loopback HTTP."""

    def __init__(
        self,
        *,
        status_provider: StatusProvider,
        state_provider: StateProvider,
        screenshot_provider: ScreenshotProvider,
        token: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        queue_limit: int = DEFAULT_QUEUE_LIMIT,
        job_timeout: float = DEFAULT_JOB_TIMEOUT,
        authority: int = PRIORITY_LEVELS["ai"],
        allow_remote_bind: bool = False,
        bounds: tuple[int, int] = DEFAULT_BOUNDS,
        context_provider: StateProvider | None = None,
        inspect_provider: Callable[[float, float, float], dict[str, Any]] | None = None,
        preview_provider: ScreenshotProvider | None = None,
    ) -> None:
        if not token:
            raise ValueError("A non-empty access token is required")
        if authority not in PRIORITY_LEVELS.values():
            raise ValueError(f"Unknown authority level: {authority}")
        if bounds[0] <= 0 or bounds[1] <= 0:
            raise ValueError(f"World bounds must be positive, got {bounds}")
        require_loopback(
            host,
            allow_remote=allow_remote_bind,
            service="AI bridge",
        )
        self._authority = authority
        self._bounds = (int(bounds[0]), int(bounds[1]))
        self._status_provider = status_provider
        self._state_provider = state_provider
        self._screenshot_provider = screenshot_provider
        self._context_provider = context_provider
        self._inspect_provider = inspect_provider
        self._preview_provider = preview_provider or screenshot_provider
        self._token = token
        self._host = host
        self._port = port
        self._queue_limit = queue_limit
        self._job_timeout = job_timeout
        self._lock = threading.Lock()
        self._operations: list[Any] = []
        self._jobs: list[_Job] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._accepted_requests = 0

    @property
    def token(self) -> str:
        return self._token

    @property
    def authority(self) -> int:
        return self._authority

    @property
    def bounds(self) -> tuple[int, int]:
        return self._bounds

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            return (self._host, self._port)
        host, port = self._server.server_address[:2]
        return (str(host), int(port))

    @property
    def base_url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    def start(self) -> str:
        """Bind the socket and serve requests on a daemon thread."""
        if self._server is not None:
            raise RuntimeError("Bridge is already running")
        bridge = self

        class Handler(_RequestHandler):
            bridge_instance = bridge

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="nwr-ai-bridge",
            daemon=True,
        )
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        """Stop serving and fail any request still waiting on the render loop."""
        server, self._server = self._server, None
        thread, self._thread = self._thread, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            pending, self._jobs = self._jobs, []
            self._operations.clear()
        for job in pending:
            job.error = "Bridge stopped before the request completed"
            job.event.set()

    def submit_request(self, payload: Any) -> dict[str, Any]:
        """Validate and queue a command request; raises :class:`BridgeError`."""
        from ai_command_compiler import TemperatureDelta, compile_ai_json

        try:
            operations = compile_ai_json(
                payload,
                default_priority=self._authority,
                bounds=self._bounds,
            )
        except CommandError as exc:
            raise BridgeError(HTTPStatus.BAD_REQUEST, str(exc)) from None

        with self._lock:
            if len(self._operations) + len(operations) > self._queue_limit:
                raise BridgeError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Command queue is full; retry after the simulation catches up",
                )
            self._operations.extend(operations)
            self._accepted_requests += 1
            queued = len(self._operations)
            request_id = self._accepted_requests

        segments = sum(1 for item in operations if isinstance(item, Segment))
        temperatures = sum(
            1 for item in operations if isinstance(item, TemperatureDelta)
        )
        controls = [item.action for item in operations if isinstance(item, Control)]
        entity_requests = sum(
            1
            for item in operations
            if isinstance(item, (SpawnEntity, RemoveEntity))
        )
        LOGGER.info(
            "Accepted request %d: %d segments, %d temperature, controls=%s",
            request_id,
            segments,
            temperatures,
            controls,
        )
        return {
            "request_id": request_id,
            "accepted_operations": len(operations),
            "segments": segments,
            "temperature_deltas": temperatures,
            "entity_requests": entity_requests,
            "controls": controls,
            "queued_operations": queued,
            "authority": PRIORITY_NAMES[self._authority],
        }

    def request_job(self, kind: str, request: Any = None) -> Any:
        """Ask the render loop for data and block until it responds."""
        if self._server is None:
            raise BridgeError(HTTPStatus.SERVICE_UNAVAILABLE, "Bridge is not running")
        job = _Job(kind=kind, request=request)
        with self._lock:
            self._jobs.append(job)
        if not job.event.wait(timeout=self._job_timeout):
            with self._lock:
                if job in self._jobs:
                    self._jobs.remove(job)
            raise BridgeError(
                HTTPStatus.GATEWAY_TIMEOUT,
                "The render loop did not respond; is the window running?",
            )
        if job.error is not None:
            raise BridgeError(HTTPStatus.INTERNAL_SERVER_ERROR, job.error)
        return job.payload

    def status(self) -> dict[str, Any]:
        with self._lock:
            queued = len(self._operations)
        payload = dict(self._status_provider())
        payload["queued_operations"] = queued
        return payload

    def service(self, max_operations: int = 512) -> list[Operation]:
        """Convenience wrapper that runs jobs and then drains operations."""
        self.run_jobs()
        return self.take_operations(max_operations)

    def take_operations(self, max_operations: int = 512) -> list[Operation]:
        """Hand a bounded batch of queued operations to the render loop."""
        with self._lock:
            batch = self._operations[:max_operations]
            del self._operations[: len(batch)]
        return batch

    def run_jobs(self) -> None:
        """Fulfil observation requests; must run on the OpenGL context thread."""
        with self._lock:
            jobs, self._jobs = self._jobs, []

        for job in jobs:
            try:
                if job.kind == "state":
                    job.payload = self._state_provider()
                elif job.kind == "screenshot":
                    job.payload = self._screenshot_provider()
                elif job.kind == "context":
                    if self._context_provider is None:
                        raise RuntimeError("No context provider is configured")
                    job.payload = self._context_provider()
                elif job.kind == "preview":
                    job.payload = self._preview_provider()
                elif job.kind == "inspect":
                    if self._inspect_provider is None:
                        raise RuntimeError("No inspect provider is configured")
                    request = job.request or {}
                    job.payload = self._inspect_provider(
                        float(request["x"]),
                        float(request["y"]),
                        float(request["radius"]),
                    )
                else:
                    job.error = f"Unknown job kind '{job.kind}'"
            except Exception as exc:
                LOGGER.exception("Job %s failed", job.kind)
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.event.set()


class _RequestHandler(BaseHTTPRequestHandler):
    bridge_instance: ControlBridge
    protocol_version = "HTTP/1.1"
    server_version = "NeuralWorldRuntime/1.0"

    def do_GET(self) -> None:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        routes: dict[str, Callable[[], tuple[HTTPStatus, Any]]] = {
            "/": lambda: (HTTPStatus.OK, _index_document()),
            "/health": lambda: (HTTPStatus.OK, self.bridge_instance.status()),
            "/schema": lambda: (
                HTTPStatus.OK,
                {
                    # Narrowed to this caller's authority, so the grammar a
                    # model reads is the grammar it can actually use.
                    **schema_for_authority(self.bridge_instance.authority),
                    "caller_authority": PRIORITY_NAMES[
                        self.bridge_instance.authority
                    ],
                    "world": {
                        "width": self.bridge_instance.bounds[0],
                        "height": self.bridge_instance.bounds[1],
                    },
                    "named_commands": [
                        "PaintMaterial",
                        "SetMaterial",
                        "Erase",
                        "IncreaseTemperature",
                        "DecreaseTemperature",
                    ],
                    "unsupported_commands": [
                        "SpawnEntity",
                        "RemoveEntity",
                    ],
                },
            ),
            "/state": lambda: (
                HTTPStatus.OK,
                self.bridge_instance.request_job("state"),
            ),
            "/context": lambda: (
                HTTPStatus.OK,
                self.bridge_instance.request_job("context"),
            ),
        }
        try:
            if path == "/screenshot" or path == "/preview":
                self._require_token()
                kind = "preview" if path == "/preview" else "screenshot"
                image = self.bridge_instance.request_job(kind)
                self._respond_bytes(HTTPStatus.OK, image, "image/png")
                return
            if path == "/inspect":
                self._require_token()
                try:
                    request = {
                        "x": float(query["x"][0]),
                        "y": float(query["y"][0]),
                        "radius": float(query.get("radius", ["8"])[0]),
                    }
                except (KeyError, IndexError, ValueError) as exc:
                    raise BridgeError(
                        HTTPStatus.BAD_REQUEST,
                        "Inspect requires query params x, y, and optional radius",
                    ) from exc
                payload = self.bridge_instance.request_job("inspect", request)
                self._respond_json(HTTPStatus.OK, payload)
                return
            route = routes.get(path)
            if route is None:
                raise BridgeError(HTTPStatus.NOT_FOUND, f"Unknown path '{path}'")
            if path not in ("/", "/health"):
                self._require_token()
            status, payload = route()
        except BridgeError as exc:
            self._respond_json(exc.status, {"error": exc.message})
        else:
            self._respond_json(status, payload)

    def do_POST(self) -> None:
        from urllib.parse import urlparse

        path = urlparse(self.path).path
        try:
            if path == "/inspect":
                self._require_token()
                body = self._read_json_body()
                if not isinstance(body, dict):
                    raise BridgeError(HTTPStatus.BAD_REQUEST, "Inspect body must be an object")
                try:
                    request = {
                        "x": float(body["x"]),
                        "y": float(body["y"]),
                        "radius": float(body.get("radius", 8.0)),
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    raise BridgeError(
                        HTTPStatus.BAD_REQUEST,
                        "Inspect requires numeric x, y, and optional radius",
                    ) from exc
                payload = self.bridge_instance.request_job("inspect", request)
                self._respond_json(HTTPStatus.OK, payload)
                return
            if path != "/commands":
                raise BridgeError(HTTPStatus.NOT_FOUND, f"Unknown path '{path}'")
            self._require_token()
            payload = self._read_json_body()
            result = self.bridge_instance.submit_request(payload)
        except BridgeError as exc:
            self._respond_json(exc.status, {"error": exc.message})
        else:
            self._respond_json(HTTPStatus.ACCEPTED, result)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug("%s - %s", self.address_string(), format % args)

    def _require_token(self) -> None:
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        supplied = header[len(prefix) :] if header.startswith(prefix) else ""
        if not supplied:
            supplied = self.headers.get("X-Access-Token", "")
        if not secrets.compare_digest(supplied, self.bridge_instance.token):
            raise BridgeError(
                HTTPStatus.UNAUTHORIZED,
                "Missing or invalid bearer token",
            )

    def _read_json_body(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from None
        if length <= 0:
            raise BridgeError(HTTPStatus.BAD_REQUEST, "Request body is required")
        if length > MAX_BODY_BYTES:
            raise BridgeError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"Request body exceeds {MAX_BODY_BYTES} bytes",
            )
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BridgeError(
                HTTPStatus.BAD_REQUEST,
                f"Body is not valid UTF-8 JSON: {exc}",
            ) from None

    def _respond_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, allow_nan=False, indent=2).encode("utf-8")
        self._respond_bytes(status, body, "application/json; charset=utf-8")

    def _respond_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _index_document() -> dict[str, Any]:
    return {
        "service": "Neural World Runtime AI bridge",
        "authentication": "Authorization: Bearer <token>",
        "endpoints": {
            "GET /health": "Runtime status without touching the GPU",
            "GET /schema": "Command grammar for command generation",
            "GET /state": "World observation summary",
            "GET /context": "Bundled AI context for assistant drag-in workflows",
            "GET /inspect?x&y&radius": "Semantic statistics for a circular region",
            "POST /inspect": "Same as GET inspect with a JSON body",
            "GET /screenshot": "PNG of the most recent frame",
            "GET /preview": "Optional PNG preview alias of the current frame",
            "POST /commands": "Submit {'commands': [...]} or a named command object",
        },
    }


def generate_token() -> str:
    return secrets.token_urlsafe(24)


def encode_png(pixels: bytes, width: int, height: int, components: int = 3) -> bytes:
    """Encode a bottom-up OpenGL pixel buffer as a top-down PNG."""
    from PIL import Image

    mode = {3: "RGB", 4: "RGBA"}.get(components)
    if mode is None:
        raise ValueError(f"Unsupported component count: {components}")
    expected = width * height * components
    if len(pixels) != expected:
        raise ValueError(f"Expected {expected} pixel bytes, received {len(pixels)}")
    image = Image.frombytes(mode, (width, height), pixels).transpose(
        Image.Transpose.FLIP_TOP_BOTTOM
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def summarize_world(
    grid: npt.NDArray[np.float32],
    *,
    occupancy_threshold: float = 0.02,
    map_resolution: int = 16,
) -> dict[str, Any]:
    """Describe a world grid in a compact, model-readable form."""
    if grid.ndim != 3:
        raise ValueError("Grid must have shape (height, width, channels)")
    height, width, _channels = grid.shape
    density = grid[..., 3]
    energy = grid[..., 7]
    emission = grid[..., 14]
    material_norm = np.linalg.norm(grid[..., 8:16], axis=-1)
    occupied = (np.abs(density) > occupancy_threshold) | (
        material_norm > occupancy_threshold
    )
    locked = grid[..., 31] >= 0.5

    counts = {"vacuum": int((~occupied).sum())}
    if occupied.any():
        names = [name for name in ANCHORS if name != "vacuum"]
        anchors = np.asarray([ANCHORS[name] for name in names], dtype=np.float32)
        samples = grid[occupied].reshape(-1, grid.shape[-1])
        distances = np.linalg.norm(samples[:, None, :] - anchors[None, :, :], axis=-1)
        nearest = np.argmin(distances, axis=1)
        for index, name in enumerate(names):
            counts[name] = int((nearest == index).sum())
    else:
        for name in ANCHORS:
            counts.setdefault(name, 0)

    return {
        "grid": {"width": int(width), "height": int(height)},
        "cells": {
            "total": int(height * width),
            "occupied": int(occupied.sum()),
            "human_locked": int(locked.sum()),
        },
        "categories": counts,
        "fields": {
            "density": _field_statistics(density),
            "energy": _field_statistics(energy),
            "emission": _field_statistics(emission),
        },
        "bounds": {
            "occupied": _bounding_box(occupied),
            "human_locked": _bounding_box(locked),
        },
        "occupancy_map": {
            "resolution": map_resolution,
            "legend": "0 is empty, 9 is fully covered; first row is the top of the grid",
            "rows": _occupancy_rows(occupied, map_resolution),
        },
    }


def _field_statistics(field_values: npt.NDArray[np.float32]) -> dict[str, float]:
    return {
        "mean": round(float(field_values.mean()), 5),
        "max": round(float(field_values.max()), 5),
        "min": round(float(field_values.min()), 5),
    }


def _bounding_box(mask: npt.NDArray[np.bool_]) -> dict[str, list[int]] | None:
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    return {
        "min": [int(columns[0]), int(rows[0])],
        "max": [int(columns[-1]), int(rows[-1])],
    }


def _occupancy_rows(
    occupied: npt.NDArray[np.bool_],
    resolution: int,
) -> list[str]:
    if resolution <= 0:
        raise ValueError("map_resolution must be positive")
    height, width = occupied.shape
    resolution = min(resolution, height, width)
    row_edges = np.linspace(0, height, resolution + 1).astype(int)[:-1]
    column_edges = np.linspace(0, width, resolution + 1).astype(int)[:-1]
    values = occupied.astype(np.float32)
    block_sums = np.add.reduceat(
        np.add.reduceat(values, row_edges, axis=0),
        column_edges,
        axis=1,
    )
    block_counts = np.add.reduceat(
        np.add.reduceat(np.ones_like(values), row_edges, axis=0),
        column_edges,
        axis=1,
    )
    coverage = np.clip(np.rint(block_sums / block_counts * 9.0), 0, 9).astype(int)
    return ["".join(str(value) for value in row) for row in coverage[::-1]]


__all__ = [
    "BridgeError",
    "ControlBridge",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "encode_png",
    "generate_token",
    "summarize_world",
]
