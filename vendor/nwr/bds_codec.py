"""Compression codecs for world payloads.

Two codecs serve different needs:

* :func:`pack_lossless` and :func:`unpack_lossless` are bit-exact, and are used
  whenever data must survive a round trip unchanged, such as paging a chunk out
  of GPU memory and back.
* :func:`encode_anchor_residual` and :func:`decode_anchor_residual` store a
  per-cell anchor index plus a quantized residual. Most cells sit close to an
  anchor, so the residual planes compress well. The error is bounded by the
  requested quantization step and is reported rather than silently accepted.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from bds_format import ANCHORS, DTYPE, VECTOR_DIMENSIONS, FloatGrid

LOSSLESS_MAGIC: Final = b"BDZ1"
ANCHOR_MAGIC: Final = b"BDA1"
_LOSSLESS_HEADER: Final = struct.Struct("<4sHHHI")
_ANCHOR_HEADER: Final = struct.Struct("<4sHHHfII")
DEFAULT_QUANTIZATION_STEP: Final = 1.0 / 512.0
_MAX_RESIDUAL_CODE: Final = 127


class CodecError(ValueError):
    """Raised when encoded payload bytes are malformed or incompatible."""


@dataclass(frozen=True, slots=True)
class CompressionReport:
    """Measured outcome of an anchor/residual encode."""

    raw_bytes: int
    encoded_bytes: int
    max_absolute_error: float
    clipped_values: int
    step: float

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.encoded_bytes if self.encoded_bytes else 0.0


def anchor_matrix() -> npt.NDArray[np.float32]:
    """Return the codebook as an ``(anchors, channels)`` array."""
    return np.asarray(list(ANCHORS.values()), dtype=DTYPE)


def anchor_names() -> list[str]:
    return list(ANCHORS)


def pack_lossless(grid: npt.ArrayLike, *, level: int = 6) -> bytes:
    """Deflate a grid without any loss of precision."""
    array = _validate_grid(grid)
    height, width = array.shape[0], array.shape[1]
    payload = zlib.compress(array.tobytes(order="C"), level)
    return (
        _LOSSLESS_HEADER.pack(
            LOSSLESS_MAGIC,
            width,
            height,
            array.shape[2],
            len(payload),
        )
        + payload
    )


def unpack_lossless(blob: bytes) -> FloatGrid:
    """Restore a grid produced by :func:`pack_lossless`."""
    if len(blob) < _LOSSLESS_HEADER.size:
        raise CodecError("Payload is shorter than the lossless header")
    magic, width, height, channels, length = _LOSSLESS_HEADER.unpack_from(blob)
    if magic != LOSSLESS_MAGIC:
        raise CodecError("Invalid lossless payload magic")
    body = blob[_LOSSLESS_HEADER.size :]
    if len(body) != length:
        raise CodecError(f"Expected {length} compressed bytes, found {len(body)}")
    try:
        raw = zlib.decompress(body)
    except zlib.error as exc:
        raise CodecError(f"Payload could not be inflated: {exc}") from None
    expected = width * height * channels * DTYPE.itemsize
    if len(raw) != expected:
        raise CodecError(f"Expected {expected} decoded bytes, found {len(raw)}")
    grid = np.frombuffer(raw, dtype=DTYPE).reshape((height, width, channels))
    return np.array(grid, dtype=DTYPE, order="C", copy=True)


def encode_anchor_residual(
    grid: npt.ArrayLike,
    *,
    step: float | None = None,
    level: int = 6,
) -> tuple[bytes, CompressionReport]:
    """Encode a grid as anchor indices plus quantized residuals.

    With ``step`` left as ``None`` the quantization step is chosen so that no
    residual clips, which bounds the absolute error at half a step. Passing an
    explicit step trades that guarantee for a fixed precision.
    """
    array = _validate_grid(grid)
    if step is not None and not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1]")
    height, width, channels = array.shape
    anchors = anchor_matrix()
    if anchors.shape[1] != channels:
        raise CodecError("Anchor codebook does not match the grid channel count")

    flat = array.reshape(-1, channels)
    distances = np.linalg.norm(flat[:, None, :] - anchors[None, :, :], axis=-1)
    indices = np.argmin(distances, axis=1).astype(np.uint8)
    residual = flat - anchors[indices]

    if step is None:
        largest = float(np.abs(residual).max(initial=0.0))
        step = max(largest / _MAX_RESIDUAL_CODE, DEFAULT_QUANTIZATION_STEP)
    scaled = np.rint(residual / step)
    clipped = int(np.count_nonzero(np.abs(scaled) > _MAX_RESIDUAL_CODE))
    codes = np.clip(scaled, -_MAX_RESIDUAL_CODE, _MAX_RESIDUAL_CODE).astype(np.int8)
    error = float(np.abs(residual - codes.astype(np.float32) * step).max())

    index_blob = zlib.compress(indices.tobytes(order="C"), level)
    residual_blob = zlib.compress(
        np.ascontiguousarray(codes.T).tobytes(order="C"),
        level,
    )
    encoded = (
        _ANCHOR_HEADER.pack(
            ANCHOR_MAGIC,
            width,
            height,
            channels,
            float(step),
            len(index_blob),
            len(residual_blob),
        )
        + index_blob
        + residual_blob
    )
    report = CompressionReport(
        raw_bytes=array.nbytes,
        encoded_bytes=len(encoded),
        max_absolute_error=error,
        clipped_values=clipped,
        step=float(step),
    )
    return encoded, report


def decode_anchor_residual(blob: bytes) -> FloatGrid:
    """Restore a grid produced by :func:`encode_anchor_residual`."""
    if len(blob) < _ANCHOR_HEADER.size:
        raise CodecError("Payload is shorter than the anchor header")
    (
        magic,
        width,
        height,
        channels,
        step,
        index_length,
        residual_length,
    ) = _ANCHOR_HEADER.unpack_from(blob)
    if magic != ANCHOR_MAGIC:
        raise CodecError("Invalid anchor payload magic")
    if channels != VECTOR_DIMENSIONS:
        raise CodecError(f"Unsupported channel count: {channels}")
    body = blob[_ANCHOR_HEADER.size :]
    if len(body) != index_length + residual_length:
        raise CodecError("Anchor payload body length does not match its header")

    try:
        indices = np.frombuffer(
            zlib.decompress(body[:index_length]),
            dtype=np.uint8,
        )
        codes = np.frombuffer(
            zlib.decompress(body[index_length:]),
            dtype=np.int8,
        )
    except zlib.error as exc:
        raise CodecError(f"Payload could not be inflated: {exc}") from None

    cells = width * height
    if indices.size != cells:
        raise CodecError(f"Expected {cells} anchor indices, found {indices.size}")
    if codes.size != cells * channels:
        raise CodecError(
            f"Expected {cells * channels} residual codes, found {codes.size}"
        )

    anchors = anchor_matrix()
    if int(indices.max(initial=0)) >= anchors.shape[0]:
        raise CodecError("Payload references an anchor outside the codebook")
    residual = codes.reshape(channels, cells).T.astype(DTYPE) * DTYPE.type(step)
    grid = anchors[indices] + residual
    return np.ascontiguousarray(grid.reshape(height, width, channels), dtype=DTYPE)


def _validate_grid(grid: npt.ArrayLike) -> FloatGrid:
    array = np.asarray(grid)
    if array.ndim != 3:
        raise ValueError("Grid must have shape (height, width, channels)")
    if array.shape[2] != VECTOR_DIMENSIONS:
        raise ValueError(f"Grid must have {VECTOR_DIMENSIONS} channels")
    if not np.isfinite(array).all():
        raise ValueError("Grid cannot contain NaN or infinity")
    height, width = array.shape[0], array.shape[1]
    if not 0 < width <= 0xFFFF or not 0 < height <= 0xFFFF:
        raise ValueError("Grid dimensions must fit in an unsigned 16-bit integer")
    return np.ascontiguousarray(array, dtype=DTYPE)


__all__ = [
    "ANCHOR_MAGIC",
    "CodecError",
    "CompressionReport",
    "DEFAULT_QUANTIZATION_STEP",
    "LOSSLESS_MAGIC",
    "anchor_matrix",
    "anchor_names",
    "decode_anchor_residual",
    "encode_anchor_residual",
    "pack_lossless",
    "unpack_lossless",
]
