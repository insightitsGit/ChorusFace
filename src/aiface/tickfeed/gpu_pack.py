"""Pack TickPackage bodies for tick_ingest.comp (dense/sparse f16)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aiface.tickfeed.package import FaceBox, TickPackage, _as_hw2
from aiface.tickfeed.schema import DeltaEncoding, PackageKind, PHASE1_CHANNELS


def pack_half2_uints(vx_vy: NDArray[np.floating]) -> NDArray[np.uint32]:
    """Pack (..., 2) float pairs into uint32 via numpy float16 (GL unpackHalf2x16)."""
    pairs = np.asarray(vx_vy, dtype=np.float32).reshape(-1, PHASE1_CHANNELS)
    f16 = pairs.astype("<f2")
    # two f16 → one u32 little-endian
    return np.frombuffer(f16.tobytes(), dtype="<u4").copy()


def dense_uints_from_package(package: TickPackage) -> NDArray[np.uint32]:
    if package.values is None:
        raise ValueError("package has no dense values")
    hw2 = _as_hw2(package.values, package.face)
    return pack_half2_uints(hw2)


def sparse_buffers_from_package(
    package: TickPackage,
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    if package.sparse_idx is None or package.sparse_delta is None:
        raise ValueError("package is not sparse")
    idx = np.asarray(package.sparse_idx, dtype=np.uint32)
    packed = pack_half2_uints(package.sparse_delta)
    return idx, packed


def ingest_encoding(package: TickPackage) -> int:
    """Map package → shader ``encoding`` uniform."""
    if package.kind == PackageKind.KEYFRAME:
        return 0
    if package.delta_encoding == DeltaEncoding.DENSE_DELTA:
        return 1
    if package.delta_encoding == DeltaEncoding.SPARSE_DELTA:
        return 2
    if package.delta_encoding == DeltaEncoding.EMPTY:
        return 3
    return 3


def face_uniforms(face: FaceBox) -> dict[str, tuple[int, int]]:
    return {
        "face_offset": (int(face.x), int(face.y)),
        "face_size": (int(face.w), int(face.h)),
    }


__all__ = [
    "dense_uints_from_package",
    "face_uniforms",
    "ingest_encoding",
    "pack_half2_uints",
    "sparse_buffers_from_package",
]
