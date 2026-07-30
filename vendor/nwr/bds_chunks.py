"""Chunked world storage with activity tracking and lossless paging.

A world is divided into fixed-size chunks. Chunks the simulation is touching
stay resident as plain arrays; quiescent chunks are deflated with the lossless
codec and dropped from the resident set. Paging must never alter data, so
eviction always uses :func:`bds_codec.pack_lossless` rather than the lossy
anchor codec, which is reserved for archival.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import Any, Final, Iterator

import numpy as np
import numpy.typing as npt

from bds_codec import pack_lossless, unpack_lossless
from bds_format import DTYPE, VECTOR_DIMENSIONS, FloatGrid, create_empty_grid

DEFAULT_CHUNK_SIZE: Final = 64
QUIESCENT_TICKS: Final = 120
ACTIVITY_EPSILON: Final = 1e-4


class ChunkError(ValueError):
    """Raised when chunk geometry or paging state is invalid."""


@dataclass(slots=True)
class ChunkState:
    """Residency and activity bookkeeping for one chunk."""

    index: tuple[int, int]
    origin: tuple[int, int]
    size: tuple[int, int]
    resident: FloatGrid | None = None
    compressed: bytes | None = None
    last_active_tick: int = 0
    checksum: int = 0

    @property
    def is_resident(self) -> bool:
        return self.resident is not None

    @property
    def cells(self) -> int:
        return self.size[0] * self.size[1]

    @property
    def resident_bytes(self) -> int:
        return 0 if self.resident is None else int(self.resident.nbytes)

    @property
    def compressed_bytes(self) -> int:
        return 0 if self.compressed is None else len(self.compressed)


@dataclass(slots=True)
class PagingStatistics:
    """Counters describing paging behaviour over a session."""

    evictions: int = 0
    restorations: int = 0
    bytes_written: int = 0
    bytes_read: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "evictions": self.evictions,
            "restorations": self.restorations,
            "bytes_written": self.bytes_written,
            "bytes_read": self.bytes_read,
        }


class ChunkedWorld:
    """A world of arbitrary size backed by independently paged chunks."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        grid: npt.ArrayLike | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ChunkError("World dimensions must be positive")
        if chunk_size <= 0:
            raise ChunkError("Chunk size must be positive")
        self.width = int(width)
        self.height = int(height)
        self.chunk_size = int(chunk_size)
        self.columns = -(-self.width // self.chunk_size)
        self.rows = -(-self.height // self.chunk_size)
        self.statistics = PagingStatistics()

        source = (
            create_empty_grid(self.width, self.height)
            if grid is None
            else self._validate_grid(grid)
        )
        self._chunks: dict[tuple[int, int], ChunkState] = {}
        for row in range(self.rows):
            for column in range(self.columns):
                origin = (column * self.chunk_size, row * self.chunk_size)
                size = (
                    min(self.chunk_size, self.width - origin[0]),
                    min(self.chunk_size, self.height - origin[1]),
                )
                block = np.ascontiguousarray(
                    source[
                        origin[1] : origin[1] + size[1],
                        origin[0] : origin[0] + size[0],
                    ],
                    dtype=DTYPE,
                )
                self._chunks[(column, row)] = ChunkState(
                    index=(column, row),
                    origin=origin,
                    size=size,
                    resident=block,
                    checksum=_checksum(block),
                )

    def _validate_grid(self, grid: npt.ArrayLike) -> FloatGrid:
        array = np.asarray(grid)
        expected = (self.height, self.width, VECTOR_DIMENSIONS)
        if array.shape != expected:
            raise ChunkError(f"Grid must have shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ChunkError("Grid cannot contain NaN or infinity")
        return np.ascontiguousarray(array, dtype=DTYPE)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def chunks(self) -> Iterator[ChunkState]:
        for key in sorted(self._chunks):
            yield self._chunks[key]

    def chunk_at(self, column: int, row: int) -> ChunkState:
        try:
            return self._chunks[(column, row)]
        except KeyError:
            raise ChunkError(f"No chunk at ({column}, {row})") from None

    def chunk_for_cell(self, x: int, y: int) -> ChunkState:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ChunkError(f"Cell ({x}, {y}) is outside the world")
        return self.chunk_at(x // self.chunk_size, y // self.chunk_size)

    def resident_indices(self) -> list[tuple[int, int]]:
        return [chunk.index for chunk in self.chunks() if chunk.is_resident]

    def memory_usage(self) -> dict[str, int]:
        resident = sum(chunk.resident_bytes for chunk in self.chunks())
        compressed = sum(chunk.compressed_bytes for chunk in self.chunks())
        return {
            "resident_chunks": len(self.resident_indices()),
            "total_chunks": self.chunk_count,
            "resident_bytes": resident,
            "compressed_bytes": compressed,
            "uncompressed_equivalent": self.width
            * self.height
            * VECTOR_DIMENSIONS
            * DTYPE.itemsize,
        }

    def ensure_resident(self, column: int, row: int) -> FloatGrid:
        """Return a chunk's array, inflating it first when it was paged out."""
        chunk = self.chunk_at(column, row)
        if chunk.resident is None:
            if chunk.compressed is None:
                raise ChunkError(f"Chunk {chunk.index} has no stored data")
            chunk.resident = unpack_lossless(chunk.compressed)
            self.statistics.restorations += 1
            self.statistics.bytes_read += len(chunk.compressed)
            chunk.compressed = None
        return chunk.resident

    def evict(self, column: int, row: int) -> int:
        """Compress a chunk out of the resident set and return the stored size."""
        chunk = self.chunk_at(column, row)
        if chunk.resident is None:
            return chunk.compressed_bytes
        blob = pack_lossless(chunk.resident)
        chunk.compressed = blob
        chunk.checksum = _checksum(chunk.resident)
        chunk.resident = None
        self.statistics.evictions += 1
        self.statistics.bytes_written += len(blob)
        return len(blob)

    def write_chunk(self, column: int, row: int, block: npt.ArrayLike, *, tick: int) -> None:
        """Replace a chunk's contents and refresh its activity state."""
        chunk = self.chunk_at(column, row)
        array = np.asarray(block, dtype=DTYPE)
        expected = (chunk.size[1], chunk.size[0], VECTOR_DIMENSIONS)
        if array.shape != expected:
            raise ChunkError(f"Chunk block must have shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ChunkError("Chunk block cannot contain NaN or infinity")
        contiguous = np.ascontiguousarray(array, dtype=DTYPE)
        previous = chunk.checksum
        chunk.resident = contiguous
        chunk.compressed = None
        chunk.checksum = _checksum(contiguous)
        if chunk.checksum != previous:
            chunk.last_active_tick = tick

    def mark_active(self, column: int, row: int, tick: int) -> None:
        self.chunk_at(column, row).last_active_tick = tick

    def mark_region_active(
        self,
        minimum: tuple[float, float],
        maximum: tuple[float, float],
        tick: int,
    ) -> list[tuple[int, int]]:
        """Mark every chunk overlapping a rectangle as active at ``tick``."""
        first_column = max(int(minimum[0]) // self.chunk_size, 0)
        last_column = min(int(maximum[0]) // self.chunk_size, self.columns - 1)
        first_row = max(int(minimum[1]) // self.chunk_size, 0)
        last_row = min(int(maximum[1]) // self.chunk_size, self.rows - 1)
        touched: list[tuple[int, int]] = []
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                self.mark_active(column, row, tick)
                touched.append((column, row))
        return touched

    def refresh_activity(self, tick: int) -> list[tuple[int, int]]:
        """Update activity from resident content and return changed chunks."""
        changed: list[tuple[int, int]] = []
        for chunk in self.chunks():
            if chunk.resident is None:
                continue
            checksum = _checksum(chunk.resident)
            if checksum != chunk.checksum:
                chunk.checksum = checksum
                chunk.last_active_tick = tick
                changed.append(chunk.index)
        return changed

    def evict_quiescent(
        self,
        tick: int,
        *,
        idle_ticks: int = QUIESCENT_TICKS,
        keep: int = 0,
    ) -> list[tuple[int, int]]:
        """Page out chunks untouched for ``idle_ticks``, keeping the newest few."""
        candidates = [
            chunk
            for chunk in self.chunks()
            if chunk.is_resident and tick - chunk.last_active_tick >= idle_ticks
        ]
        candidates.sort(key=lambda chunk: chunk.last_active_tick)
        if keep > 0:
            resident = len(self.resident_indices())
            allowed = max(resident - keep, 0)
            candidates = candidates[:allowed]
        evicted: list[tuple[int, int]] = []
        for chunk in candidates:
            self.evict(*chunk.index)
            evicted.append(chunk.index)
        return evicted

    def to_grid(self) -> FloatGrid:
        """Assemble the full world, inflating any paged-out chunks."""
        grid = create_empty_grid(self.width, self.height)
        for chunk in self.chunks():
            block = self.ensure_resident(*chunk.index)
            x, y = chunk.origin
            grid[y : y + chunk.size[1], x : x + chunk.size[0]] = block
        return grid

    def apply_grid(self, grid: npt.ArrayLike, *, tick: int) -> None:
        """Split a full world into chunks, refreshing activity where it changed."""
        array = self._validate_grid(grid)
        for chunk in self.chunks():
            x, y = chunk.origin
            self.write_chunk(
                *chunk.index,
                array[y : y + chunk.size[1], x : x + chunk.size[0]],
                tick=tick,
            )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the chunk table."""
        return {
            "world": {
                "width": self.width,
                "height": self.height,
                "chunk_size": self.chunk_size,
                "columns": self.columns,
                "rows": self.rows,
            },
            "memory": self.memory_usage(),
            "paging": self.statistics.as_dict(),
            "chunks": [
                {
                    "index": list(chunk.index),
                    "origin": list(chunk.origin),
                    "size": list(chunk.size),
                    "resident": chunk.is_resident,
                    "last_active_tick": chunk.last_active_tick,
                    "compressed_bytes": chunk.compressed_bytes,
                }
                for chunk in self.chunks()
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.describe(), separators=(",", ":"), sort_keys=True)


def _checksum(block: npt.NDArray[np.float32]) -> int:
    return zlib.crc32(np.ascontiguousarray(block, dtype=DTYPE).tobytes(order="C"))


__all__ = [
    "ACTIVITY_EPSILON",
    "ChunkError",
    "ChunkState",
    "ChunkedWorld",
    "DEFAULT_CHUNK_SIZE",
    "PagingStatistics",
    "QUIESCENT_TICKS",
]
