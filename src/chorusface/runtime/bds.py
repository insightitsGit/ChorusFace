"""Read, validate, and atomically write 32-channel ``.bds`` field worlds.

This is the deterministic substrate the avatar renders from. A cell is 32
float32 channels grouped as kinematics / material / intent / rules; channel 31
(``human_lock``) is the Master Lock that the constraint shader honours, and
channel 24 (``hard_surface``) carries structural contours.

The schema lives here and nowhere else: :mod:`chorusface.runtime.shaders` emits it
as GLSL constants so the CPU and GPU views of a cell cannot drift.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

MAGIC = b"BDS1\0\0\0\0"
FORMAT_VERSION = "bds-1.0"
HEADER_SIZE = 4096
GRID_WIDTH = 256
GRID_HEIGHT = 256
VECTOR_DIMENSIONS = 32
TICK_RATE_HZ = 60
DTYPE = np.dtype("<f4")
_PREFIX = struct.Struct("<8sI")
_JSON_CAPACITY = HEADER_SIZE - _PREFIX.size

FloatGrid = npt.NDArray[np.float32]

PRIORITY_LEVELS: dict[str, int] = {
    "background": 0,
    "ai": 1,
    "constraint": 2,
    "user": 3,
}
PRIORITY_NAMES: dict[int, str] = {
    value: name for name, value in PRIORITY_LEVELS.items()
}
HUMAN_LOCK_CHANNEL = 31
PRIORITY_CHANNEL = 30
HARD_SURFACE_CHANNEL = 24

CHANNEL_SCHEMA: dict[str, list[str]] = {
    "kinematics": [
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "density",
        "pressure",
        "shear",
        "temperature",
        "energy",
    ],
    "material": [
        "albedo_r",
        "albedo_g",
        "albedo_b",
        "opacity",
        "roughness",
        "metallic",
        "emission",
        "refraction",
    ],
    "intent": [
        "attraction",
        "alignment",
        "user_affinity",
        "growth",
        "decay",
        "lifespan",
        "reserved_22",
        "reserved_23",
    ],
    "rules": [
        "hard_surface",
        "permeability",
        "thermal_threshold",
        "phase_trigger",
        "reserved_28",
        "reserved_29",
        "authority_priority",
        "human_lock",
    ],
}


CHANNEL_NAMES: list[str] = [
    name for group in CHANNEL_SCHEMA.values() for name in group
]
#: Channels holding tissue motion rather than tissue identity. They are the only
#: ones speech writes, and the only ones that relax back to rest on their own.
VELOCITY_CHANNELS: tuple[int, ...] = tuple(
    index
    for index, name in enumerate(CHANNEL_NAMES)
    if name.startswith("velocity_")
)
if VELOCITY_CHANNELS != tuple(range(len(VELOCITY_CHANNELS))):
    raise AssertionError(
        "The GLSL prelude assumes the velocity channels lead the schema"
    )


def _anchor(**values: float) -> list[float]:
    channels = CHANNEL_NAMES
    unknown = set(values).difference(channels)
    if unknown:
        raise ValueError(f"Unknown anchor channels: {sorted(unknown)}")
    return [float(values.get(channel, 0.0)) for channel in channels]


ANCHORS: dict[str, list[float]] = {
    "vacuum": _anchor(),
    "human_barrier": _anchor(
        density=0.45,
        albedo_r=0.62,
        albedo_g=0.05,
        albedo_b=0.58,
        opacity=0.7,
        roughness=0.2,
        emission=0.4,
        hard_surface=0.8,
        human_lock=1.0,
    ),
    "active_fluid": _anchor(
        velocity_x=0.03,
        density=0.45,
        pressure=0.18,
        temperature=0.12,
        energy=0.32,
        albedo_r=0.04,
        albedo_g=0.34,
        albedo_b=0.74,
        opacity=0.88,
        roughness=0.2,
        emission=0.28,
        refraction=0.22,
        attraction=0.12,
        alignment=0.18,
        growth=0.06,
        permeability=0.65,
    ),
    # A wall that deflects flow without claiming human authority.
    "solid": _anchor(
        density=0.4,
        albedo_r=0.3,
        albedo_g=0.34,
        albedo_b=0.42,
        opacity=0.92,
        roughness=0.55,
        emission=0.06,
        hard_surface=1.0,
    ),
}


class BDSFormatError(ValueError):
    """Raised when a .bds file is malformed or incompatible."""


def create_empty_grid(
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT,
) -> FloatGrid:
    """Return a zeroed, C-contiguous world grid."""
    _validate_dimensions(width, height)
    return np.zeros((height, width, VECTOR_DIMENSIONS), dtype=DTYPE)


def make_header(
    grid: npt.ArrayLike,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical metadata for a validated grid."""
    array = _coerce_grid(grid)
    payload = array.tobytes(order="C")
    header: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "grid_dimensions": [int(array.shape[1]), int(array.shape[0]), 1],
        "vector_dimensions": VECTOR_DIMENSIONS,
        "tick_rate_hz": TICK_RATE_HZ,
        "dtype": "float32",
        "byte_order": "little",
        "payload_bytes": len(payload),
        "payload_crc32": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
        "layout": "height-width-channel",
        "channel_schema": CHANNEL_SCHEMA,
        "anchors": ANCHORS,
    }
    if metadata is not None:
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be finite and JSON serializable") from exc
        header["application_metadata"] = dict(metadata)
    return header


def save_bds(
    path: str | os.PathLike[str],
    grid: npt.ArrayLike,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically serialize a world and return its canonical header."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = _coerce_grid(grid)
    payload = array.tobytes(order="C")
    header = make_header(array, metadata=metadata)
    encoded = json.dumps(
        header,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if not encoded or len(encoded) > _JSON_CAPACITY:
        raise BDSFormatError(
            f"Encoded header is {len(encoded)} bytes; maximum is {_JSON_CAPACITY}"
        )

    prefix = _PREFIX.pack(MAGIC, len(encoded))
    padding = bytes(HEADER_SIZE - len(prefix) - len(encoded))
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_name = output.name
            output.write(prefix)
            output.write(encoded)
            output.write(padding)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return header


def load_bds(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], FloatGrid]:
    """Load a world after strict structural, compatibility, and CRC validation."""
    source = Path(path)
    file_size = source.stat().st_size
    if file_size < HEADER_SIZE:
        raise BDSFormatError("File is shorter than the fixed header")

    with source.open("rb") as input_file:
        fixed_header = input_file.read(HEADER_SIZE)
        magic, json_length = _PREFIX.unpack_from(fixed_header)
        if magic != MAGIC:
            raise BDSFormatError("Invalid .bds magic bytes")
        if json_length == 0 or json_length > _JSON_CAPACITY:
            raise BDSFormatError(f"Invalid JSON header length: {json_length}")
        encoded = fixed_header[_PREFIX.size : _PREFIX.size + json_length]
        try:
            header = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BDSFormatError("Header is not valid UTF-8 JSON") from exc
        _validate_header(header)

        payload_bytes = int(header["payload_bytes"])
        expected_size = HEADER_SIZE + payload_bytes
        if file_size != expected_size:
            raise BDSFormatError(
                f"File size is {file_size}; expected exactly {expected_size}"
            )
        payload = input_file.read(payload_bytes)

    actual_crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
    if actual_crc.lower() != str(header["payload_crc32"]).lower():
        raise BDSFormatError("Payload CRC32 mismatch")

    width, height, _depth = header["grid_dimensions"]
    grid = np.frombuffer(payload, dtype=DTYPE).reshape(
        (height, width, VECTOR_DIMENSIONS)
    )
    grid = np.array(grid, dtype=DTYPE, order="C", copy=True)
    if not np.isfinite(grid).all():
        raise BDSFormatError("Payload contains NaN or infinity")
    return header, grid


def _validate_dimensions(width: int, height: int) -> None:
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("Grid dimensions must be integers")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("Grid dimensions must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("Grid dimensions must be positive")


def _coerce_grid(grid: npt.ArrayLike) -> FloatGrid:
    array = np.asarray(grid)
    if array.ndim != 3 or array.shape[2] != VECTOR_DIMENSIONS:
        raise ValueError(
            f"Grid must have shape (height, width, {VECTOR_DIMENSIONS})"
        )
    _validate_dimensions(int(array.shape[1]), int(array.shape[0]))
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Grid must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError("Grid cannot contain NaN or infinity")
    return np.ascontiguousarray(array, dtype=DTYPE)


def _validate_header(header: Any) -> None:
    if not isinstance(header, dict):
        raise BDSFormatError("Header root must be a JSON object")
    required = {
        "format_version",
        "grid_dimensions",
        "vector_dimensions",
        "tick_rate_hz",
        "dtype",
        "byte_order",
        "payload_bytes",
        "payload_crc32",
    }
    missing = required.difference(header)
    if missing:
        raise BDSFormatError(f"Header is missing fields: {sorted(missing)}")
    if header["format_version"] != FORMAT_VERSION:
        raise BDSFormatError(
            f"Unsupported format version: {header['format_version']!r}"
        )
    dimensions = header["grid_dimensions"]
    if (
        not isinstance(dimensions, list)
        or len(dimensions) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in dimensions
        )
        or dimensions[0] <= 0
        or dimensions[1] <= 0
        or dimensions[2] != 1
    ):
        raise BDSFormatError(
            "grid_dimensions must be [positive width, positive height, 1]"
        )
    if header["vector_dimensions"] != VECTOR_DIMENSIONS:
        raise BDSFormatError("Unsupported vector dimension count")
    if header["tick_rate_hz"] != TICK_RATE_HZ:
        raise BDSFormatError("Unsupported tick rate")
    if header["dtype"] != "float32" or header["byte_order"] != "little":
        raise BDSFormatError("Only little-endian float32 payloads are supported")
    expected_payload = (
        dimensions[0] * dimensions[1] * VECTOR_DIMENSIONS * DTYPE.itemsize
    )
    if (
        isinstance(header["payload_bytes"], bool)
        or not isinstance(header["payload_bytes"], int)
        or header["payload_bytes"] != expected_payload
    ):
        raise BDSFormatError("payload_bytes does not match grid dimensions")
    checksum = header["payload_crc32"]
    if (
        not isinstance(checksum, str)
        or len(checksum) != 8
        or any(character not in "0123456789abcdefABCDEF" for character in checksum)
    ):
        raise BDSFormatError("payload_crc32 must be eight hexadecimal characters")


__all__ = [
    "ANCHORS",
    "BDSFormatError",
    "CHANNEL_NAMES",
    "CHANNEL_SCHEMA",
    "DTYPE",
    "FORMAT_VERSION",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "HARD_SURFACE_CHANNEL",
    "HEADER_SIZE",
    "HUMAN_LOCK_CHANNEL",
    "PRIORITY_CHANNEL",
    "PRIORITY_LEVELS",
    "PRIORITY_NAMES",
    "TICK_RATE_HZ",
    "VECTOR_DIMENSIONS",
    "VELOCITY_CHANNELS",
    "create_empty_grid",
    "load_bds",
    "make_header",
    "save_bds",
]
