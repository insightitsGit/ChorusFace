"""Append-only `.bdl` command log for deterministic session replay.

A session is a base `.bds` snapshot plus an ordered log of tick-stamped
operations. History is stored as commands rather than tensors, so replaying the
log against the snapshot reproduces the world. The log is append-only and
tolerates a torn final record, which is the expected outcome when a process is
killed mid-write.

What "reproduces" is worth stating precisely, because the interesting failures
all hid in the gap between the claim and the format:

- **Every operation that reaches the GPU has a record.** Paint, erase, the five
  controls, temperature deltas, and velocity impulses. Anything else raises at
  record time rather than being skipped — a log that silently drops a write
  turns a replay guarantee into a guess.
- **Authority and writer identity are stored, not inferred.** A record keeps
  both the priority the command carried and whether a human or an AI issued it.
  These are separate facts, and the constraint shader needs the second one: it
  vetoes AI writes on human-locked cells by opcode, so replaying an AI erase as
  a human one applies an edit the live run refused.
- **`bdl-1.0` is best-effort.** It is byte-compatible and still loads, but it
  wrote the writer byte as zero, so its writer identity is unrecoverable and a
  replay of one falls back to inferring the writer from authority. Use
  :func:`is_lossy_version` before claiming fidelity for a replay.
- **Entity intents are not records.** They are resolved into segments before
  anything reaches the GPU, and those segments are logged. See
  :func:`_refuse_entity_intent`.

The claim this format supports is same-device bit-identical replay of a
`bdl-1.1` log, which `determinism.py` proves by replaying an actual log file
against the live run that produced it.
"""

from __future__ import annotations

import json
import math
import os
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Mapping, Sequence

from ai_commands import (
    CATEGORY_IDS,
    WRITER_SOURCES,
    WRITER_UNSPECIFIED,
    Control,
    Operation,
    RemoveEntity,
    Segment,
    SpawnEntity,
    TemperatureDelta,
    VelocityImpulse,
)
from bds_format import PRIORITY_LEVELS, TICK_RATE_HZ

MAGIC: Final = b"BDL1\0\0\0\0"
FORMAT_VERSION: Final = "bdl-1.1"
# ``bdl-1.0`` is byte-compatible: same magic, same 32-byte record, same field
# offsets. It differs only in what two bytes mean. It wrote the writer-identity
# byte as a literal ``0`` and left the trailing four bytes as padding, so a 1.0
# log reads back with ``source = WRITER_UNSPECIFIED`` and ``delta = 0.0``. That
# is genuinely lossy and no amount of reading can recover it, which is why 1.0
# replay falls back to inferring the writer from authority and is documented as
# best-effort rather than faithful.
SUPPORTED_VERSIONS: Final[tuple[str, ...]] = ("bdl-1.0", "bdl-1.1")
LEGACY_VERSIONS: Final[tuple[str, ...]] = ("bdl-1.0",)
HEADER_SIZE: Final = 1024
_PREFIX: Final = struct.Struct("<8sI")
_JSON_CAPACITY: Final = HEADER_SIZE - _PREFIX.size
# tick(I) kind(B) category(B) priority(B) source(B) + 6 floats = exactly 32
# bytes, unchanged from 1.0. The sixth float lives where 1.0 kept four pad
# bytes, so temperature and velocity became loggable without growing a record.
_RECORD: Final = struct.Struct("<IBBBBffffff")
RECORD_SIZE: Final = _RECORD.size


class OperationKind(IntEnum):
    """Wire encoding for logged operations."""

    PAINT = 0
    ERASE = 1
    RESET = 2
    SAVE = 3
    LOAD = 4
    PAUSE = 5
    RESUME = 6
    TEMPERATURE = 7
    VELOCITY = 8

    @property
    def is_write(self) -> bool:
        """Whether this kind rewrites a cell's material from an anchor."""
        return self in (OperationKind.PAINT, OperationKind.ERASE)

    @property
    def touches_cells(self) -> bool:
        """Whether this kind reaches the GPU at all, as opposed to a control."""
        return self in (
            OperationKind.PAINT,
            OperationKind.ERASE,
            OperationKind.TEMPERATURE,
            OperationKind.VELOCITY,
        )


_CONTROL_TO_KIND: Final[dict[str, OperationKind]] = {
    "reset": OperationKind.RESET,
    "save": OperationKind.SAVE,
    "load": OperationKind.LOAD,
    "pause": OperationKind.PAUSE,
    "resume": OperationKind.RESUME,
}
_KIND_TO_CONTROL: Final[dict[OperationKind, str]] = {
    kind: action for action, kind in _CONTROL_TO_KIND.items()
}


class HistoryError(ValueError):
    """Raised when a `.bdl` log is malformed or incompatible."""


@dataclass(frozen=True, slots=True)
class LoggedOperation:
    """One tick-stamped operation as stored on disk."""

    tick: int
    kind: OperationKind
    category: int = 0
    priority: int = PRIORITY_LEVELS["user"]
    start_x: float = 0.0
    start_y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    radius: float = 0.0
    # Occupies the record's writer-identity byte. ``bdl-1.0`` logs wrote it as a
    # literal zero, so they read back as ``WRITER_UNSPECIFIED`` and replay falls
    # back to inferring the writer from priority rather than calling it human.
    source: int = WRITER_UNSPECIFIED
    # The sixth float, meaningful only for ``TEMPERATURE``. ``VELOCITY`` reuses
    # ``end_x``/``end_y`` for the impulse vector, matching the shader, which
    # reads it from ``segment.zw``.
    delta: float = 0.0

    def to_operation(self) -> Operation:
        """Convert back into the runtime command representation.

        Priority and writer must both survive the round trip: the shader vetoes
        AI writes on human-locked cells by opcode, so replaying an AI erase as a
        human one would apply an edit the live run refused.
        """
        if self.kind.is_write:
            return Segment(
                start_x=self.start_x,
                start_y=self.start_y,
                end_x=self.end_x,
                end_y=self.end_y,
                radius=self.radius,
                category=self.category,
                erase=self.kind is OperationKind.ERASE,
                priority=self.priority,
                source=self.source,
            )
        if self.kind is OperationKind.TEMPERATURE:
            return TemperatureDelta(
                start_x=self.start_x,
                start_y=self.start_y,
                end_x=self.end_x,
                end_y=self.end_y,
                radius=self.radius,
                delta=self.delta,
                priority=self.priority,
            )
        if self.kind is OperationKind.VELOCITY:
            return VelocityImpulse(
                x=self.start_x,
                y=self.start_y,
                velocity_x=self.end_x,
                velocity_y=self.end_y,
                radius=self.radius,
                priority=self.priority,
                source=self.source,
            )
        return Control(action=_KIND_TO_CONTROL[self.kind])

    def pack(self) -> bytes:
        return _RECORD.pack(
            self.tick,
            int(self.kind),
            self.category,
            self.priority,
            self.source,
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
            self.radius,
            self.delta,
        )

    @classmethod
    def unpack(cls, raw: bytes, offset: int = 0) -> LoggedOperation:
        (
            tick,
            kind,
            category,
            priority,
            source,
            start_x,
            start_y,
            end_x,
            end_y,
            radius,
            delta,
        ) = _RECORD.unpack_from(raw, offset)
        try:
            resolved = OperationKind(kind)
        except ValueError:
            raise HistoryError(f"Unknown operation kind {kind}") from None
        if category not in set(CATEGORY_IDS.values()):
            raise HistoryError(f"Unknown category id {category}")
        if priority not in set(PRIORITY_LEVELS.values()):
            raise HistoryError(f"Unknown priority level {priority}")
        if source not in WRITER_SOURCES:
            raise HistoryError(f"Unknown writer source {source}")
        geometry = (start_x, start_y, end_x, end_y, radius, delta)
        if not all(math.isfinite(value) for value in geometry):
            raise HistoryError("Record carries a non-finite float")
        return cls(
            tick=tick,
            kind=resolved,
            category=category,
            priority=priority,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            radius=radius,
            source=source,
            delta=delta,
        )


def operation_to_logged(
    operation: Operation,
    *,
    tick: int,
    priority: int = PRIORITY_LEVELS["user"],
) -> LoggedOperation:
    """Convert a runtime command into its loggable form.

    A command that carries its own priority wins over the ``priority`` argument,
    which is only the fallback for operations that do not name one. Stamping an
    AI write with the caller's default would file it under human authority.

    Every operation that reaches the GPU has an encoding. Entity intents do not,
    and deliberately so — see :func:`_refuse_entity_intent`. Anything else raises
    rather than being dropped, because a log that quietly omits a write turns a
    replay guarantee into a guess.
    """
    if isinstance(operation, Segment):
        return LoggedOperation(
            tick=tick,
            kind=OperationKind.ERASE if operation.erase else OperationKind.PAINT,
            category=operation.category,
            priority=operation.priority,
            source=operation.source,
            start_x=operation.start_x,
            start_y=operation.start_y,
            end_x=operation.end_x,
            end_y=operation.end_y,
            radius=operation.radius,
        )
    if isinstance(operation, TemperatureDelta):
        return LoggedOperation(
            tick=tick,
            kind=OperationKind.TEMPERATURE,
            priority=operation.priority,
            start_x=operation.start_x,
            start_y=operation.start_y,
            end_x=operation.end_x,
            end_y=operation.end_y,
            radius=operation.radius,
            delta=operation.delta,
        )
    if isinstance(operation, VelocityImpulse):
        return LoggedOperation(
            tick=tick,
            kind=OperationKind.VELOCITY,
            priority=operation.priority,
            source=operation.source,
            start_x=operation.x,
            start_y=operation.y,
            end_x=operation.velocity_x,
            end_y=operation.velocity_y,
            radius=operation.radius,
        )
    if isinstance(operation, Control):
        kind = _CONTROL_TO_KIND.get(operation.action)
        if kind is None:
            raise HistoryError(f"Control action '{operation.action}' is not loggable")
        return LoggedOperation(tick=tick, kind=kind, priority=priority)
    if isinstance(operation, (SpawnEntity, RemoveEntity)):
        _refuse_entity_intent(operation)
    raise HistoryError(f"Unsupported operation type: {type(operation).__name__}")


def _refuse_entity_intent(operation: SpawnEntity | RemoveEntity) -> None:
    """Explain why entity intents are not log records, then raise.

    A `SpawnEntity` is an *intent*: it names a kind and a place, and the runtime's
    registry resolves it into ordinary segments at the caller's authority. Those
    segments are what reach the GPU, and every one of them is logged as it is
    enqueued. So a session using entities already replays faithfully, by replaying
    the writes rather than re-deriving them.

    Logging the intent as well would be worse than useless. It would double-apply
    on replay, and reproducing it would mean re-running the registry's own
    allocation — making replay depend on registry state rather than on the log.
    The string `kind`/`entity_id` also does not fit the 32-byte record, and the
    fix for that would be to make replay depend on creation order, which is the
    same problem wearing a hat.

    Recording an entity intent is therefore a caller error, not a format gap.
    """
    raise HistoryError(
        f"{type(operation).__name__} is an intent, not a cell write: the entity "
        "registry resolves it into segments and those are what the log records. "
        "Record the resolved operations instead."
    )


def build_header(
    *,
    grid_width: int,
    grid_height: int,
    snapshot_crc32: str | None = None,
    snapshot_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    header: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "grid_dimensions": [int(grid_width), int(grid_height), 1],
        "tick_rate_hz": TICK_RATE_HZ,
        "record_bytes": RECORD_SIZE,
        "base_snapshot": snapshot_name,
        "base_snapshot_crc32": snapshot_crc32,
    }
    if metadata is not None:
        json.dumps(metadata, allow_nan=False)
        header["metadata"] = dict(metadata)
    return header


class HistoryWriter:
    """Appends operations to a `.bdl` log, creating the header when needed."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        header: Mapping[str, Any] | None = None,
        autoflush: bool = True,
    ) -> None:
        self.autoflush = autoflush
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.path.exists() and self.path.stat().st_size >= HEADER_SIZE
        if existing:
            self.header = read_header(self.path)
        else:
            if header is None:
                raise HistoryError("A header is required to create a new log")
            self.header = dict(header)
            self._write_header()
        self._file = self.path.open("r+b")
        self._file.seek(0, os.SEEK_END)
        self._appended = 0

    def _write_header(self) -> None:
        encoded = json.dumps(
            self.header,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        if not encoded or len(encoded) > _JSON_CAPACITY:
            raise HistoryError(
                f"Encoded header is {len(encoded)} bytes; maximum is {_JSON_CAPACITY}"
            )
        prefix = _PREFIX.pack(MAGIC, len(encoded))
        with self.path.open("wb") as output:
            output.write(prefix)
            output.write(encoded)
            output.write(bytes(HEADER_SIZE - len(prefix) - len(encoded)))
            output.flush()
            os.fsync(output.fileno())

    @property
    def appended(self) -> int:
        return self._appended

    def append(self, operations: Iterable[LoggedOperation]) -> int:
        """Append operations in order and return how many were written.

        Records reach the operating system immediately by default, so an
        abruptly killed process still leaves a replayable log behind.
        """
        payload = b"".join(operation.pack() for operation in operations)
        if not payload:
            return 0
        self._file.write(payload)
        if self.autoflush:
            self._file.flush()
        written = len(payload) // RECORD_SIZE
        self._appended += written
        return written

    def flush(self, *, durable: bool = False) -> None:
        self._file.flush()
        if durable:
            os.fsync(self._file.fileno())

    def close(self) -> None:
        if not self._file.closed:
            self.flush(durable=True)
            self._file.close()

    def __enter__(self) -> HistoryWriter:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class HistoryLog:
    """A parsed log with its recovery state."""

    header: dict[str, Any]
    operations: tuple[LoggedOperation, ...]
    truncated_bytes: int

    def by_tick(self) -> dict[int, list[LoggedOperation]]:
        grouped: dict[int, list[LoggedOperation]] = {}
        for operation in self.operations:
            grouped.setdefault(operation.tick, []).append(operation)
        return grouped

    def to_operations(self) -> list[Operation]:
        return [operation.to_operation() for operation in self.operations]

    @property
    def last_tick(self) -> int:
        return self.operations[-1].tick if self.operations else 0


def read_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate only the log header."""
    source = Path(path)
    if source.stat().st_size < HEADER_SIZE:
        raise HistoryError("Log is shorter than the fixed header")
    with source.open("rb") as handle:
        block = handle.read(HEADER_SIZE)
    magic, length = _PREFIX.unpack_from(block)
    if magic != MAGIC:
        raise HistoryError("Invalid .bdl magic bytes")
    if length == 0 or length > _JSON_CAPACITY:
        raise HistoryError(f"Invalid header length: {length}")
    try:
        header = json.loads(block[_PREFIX.size : _PREFIX.size + length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoryError("Header is not valid UTF-8 JSON") from exc
    if not isinstance(header, dict):
        raise HistoryError("Header root must be a JSON object")
    if header.get("format_version") not in SUPPORTED_VERSIONS:
        raise HistoryError(f"Unsupported log version: {header.get('format_version')!r}")
    if header.get("record_bytes") != RECORD_SIZE:
        raise HistoryError("Log record size does not match this build")
    return header


def is_lossy_version(header: Mapping[str, Any]) -> bool:
    """Whether this log predates recorded writer identity and deltas.

    Callers that make a fidelity claim about a replay need to know this. A
    ``bdl-1.0`` log is readable but its writer identity is unrecoverable, so a
    replay of one is best-effort and must not be reported as faithful.
    """
    return header.get("format_version") in LEGACY_VERSIONS


def read_log(path: str | os.PathLike[str], *, strict: bool = False) -> HistoryLog:
    """Read every complete record, recovering from a torn final write."""
    source = Path(path)
    header = read_header(source)
    with source.open("rb") as handle:
        handle.seek(HEADER_SIZE)
        body = handle.read()

    remainder = len(body) % RECORD_SIZE
    if remainder and strict:
        raise HistoryError(
            f"Log ends with a partial {remainder}-byte record"
        )
    complete = len(body) - remainder
    operations = tuple(
        LoggedOperation.unpack(body, offset)
        for offset in range(0, complete, RECORD_SIZE)
    )
    ticks = [operation.tick for operation in operations]
    if any(later < earlier for earlier, later in zip(ticks, ticks[1:])):
        raise HistoryError("Log ticks are not monotonically ordered")
    return HistoryLog(
        header=header,
        operations=operations,
        truncated_bytes=remainder,
    )


class SessionRecorder:
    """Writes a base snapshot plus a command log for a running session."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        grid_width: int,
        grid_height: int,
        snapshot_bytes: bytes | None = None,
        name: str = "session",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.directory / f"{name}.bds"
        self.log_path = self.directory / f"{name}.bdl"
        snapshot_crc = None
        if snapshot_bytes is not None:
            self.snapshot_path.write_bytes(snapshot_bytes)
            snapshot_crc = f"{zlib.crc32(snapshot_bytes) & 0xFFFFFFFF:08x}"
        self.writer = HistoryWriter(
            self.log_path,
            header=build_header(
                grid_width=grid_width,
                grid_height=grid_height,
                snapshot_crc32=snapshot_crc,
                snapshot_name=self.snapshot_path.name if snapshot_crc else None,
                metadata=metadata,
            ),
        )

    def record(
        self,
        operations: Sequence[Operation],
        *,
        tick: int,
        priority: int = PRIORITY_LEVELS["user"],
    ) -> int:
        entries = [
            operation_to_logged(operation, tick=tick, priority=priority)
            for operation in operations
        ]
        return self.writer.append(entries)

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()


def replay_operations(log: HistoryLog) -> Iterator[tuple[int, list[Operation]]]:
    """Yield `(tick, operations)` groups in recorded order."""
    current_tick: int | None = None
    batch: list[Operation] = []
    for entry in log.operations:
        if current_tick is not None and entry.tick != current_tick:
            yield current_tick, batch
            batch = []
        current_tick = entry.tick
        batch.append(entry.to_operation())
    if current_tick is not None:
        yield current_tick, batch


def verify_snapshot(log: HistoryLog, snapshot_bytes: bytes) -> bool:
    """Check a snapshot against the checksum recorded in the log header."""
    expected = log.header.get("base_snapshot_crc32")
    if not expected:
        return False
    actual = f"{zlib.crc32(snapshot_bytes) & 0xFFFFFFFF:08x}"
    return actual == str(expected).lower()


__all__ = [
    "FORMAT_VERSION",
    "HEADER_SIZE",
    "LEGACY_VERSIONS",
    "SUPPORTED_VERSIONS",
    "HistoryError",
    "HistoryLog",
    "HistoryWriter",
    "LoggedOperation",
    "OperationKind",
    "RECORD_SIZE",
    "SessionRecorder",
    "build_header",
    "is_lossy_version",
    "operation_to_logged",
    "read_header",
    "read_log",
    "replay_operations",
    "verify_snapshot",
]
