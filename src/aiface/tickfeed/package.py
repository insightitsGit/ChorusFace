"""Encode / decode TickPackage v1 (full-face KEY / DELTA)."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aiface.tickfeed.schema import (
    CHANNEL_MASK_VELOCITY,
    DELTA_EPS,
    DeltaEncoding,
    FLAG_HAS_CONF,
    FLAG_HAS_LABELS,
    HEADER_BYTES,
    LABELS_BYTES,
    MAGIC,
    PHASE1_CHANNELS,
    PackageKind,
    SPARSE_DENSE_THRESHOLD,
    TICK_RATE_HZ,
    VERSION,
    ValueDtype,
    VISEME_TABLE,
    BeatId,
    EmotionId,
)

_HEADER_STRUCT = struct.Struct("<IHHIf4H IBBBB II Q 16s")
assert _HEADER_STRUCT.size == HEADER_BYTES


@dataclass(slots=True)
class TickLabels:
    beat_id: int = int(BeatId.UNKNOWN)
    emotion_id: int = int(EmotionId.NEUTRAL)
    viseme_id: int = 0
    label_conf: int = 255
    smile_amt: float = 0.0
    open_amt: float = 0.0
    surprise_amt: float = 0.0
    word: str = ""

    def pack(self) -> bytes:
        word = self.word.encode("utf-8")[:16]
        word = word + b"\x00" * (16 - len(word))
        return struct.pack(
            "<BBBB fff 16s 16s",
            int(self.beat_id) & 0xFF,
            int(self.emotion_id) & 0xFF,
            int(self.viseme_id) & 0xFF,
            int(self.label_conf) & 0xFF,
            float(self.smile_amt),
            float(self.open_amt),
            float(self.surprise_amt),
            word,
            b"\x00" * 16,
        )

    @classmethod
    def unpack(cls, data: bytes) -> TickLabels:
        if len(data) < LABELS_BYTES:
            raise ValueError("labels block too short")
        beat, emo, vis, conf, smile, open_, surprise, word, _res = struct.unpack(
            "<BBBB fff 16s 16s", data[:LABELS_BYTES]
        )
        return cls(
            beat_id=beat,
            emotion_id=emo,
            viseme_id=vis,
            label_conf=conf,
            smile_amt=float(smile),
            open_amt=float(open_),
            surprise_amt=float(surprise),
            word=word.split(b"\x00", 1)[0].decode("utf-8", errors="replace"),
        )

    @staticmethod
    def viseme_index(name: str) -> int:
        key = (name or "REST").strip().upper()
        try:
            return VISEME_TABLE.index(key)
        except ValueError:
            return 0


@dataclass(slots=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def n_cells(self) -> int:
        return int(self.w) * int(self.h)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)


@dataclass(slots=True)
class TickPackage:
    """One master-clock package for full-face velocity."""

    kind: PackageKind
    tick: int
    face: FaceBox
    # Shape (H, W, 2) or (N, 2) — always vx, vy
    values: NDArray[np.floating] | None = None
    # Sparse delta: linear indices + (count, 2) deltas
    sparse_idx: NDArray[np.uint16] | None = None
    sparse_delta: NDArray[np.floating] | None = None
    conf: NDArray[np.uint8] | None = None
    labels: TickLabels | None = None
    value_dtype: ValueDtype = ValueDtype.F16
    delta_encoding: DeltaEncoding = DeltaEncoding.NONE
    channel_mask: int = CHANNEL_MASK_VELOCITY
    world_hash: int = 0
    flags: int = 0
    time_seconds: float | None = None

    def resolved_time(self) -> float:
        if self.time_seconds is not None:
            return float(self.time_seconds)
        return float(self.tick) / float(TICK_RATE_HZ)


def _as_hw2(values: NDArray[np.floating], face: FaceBox) -> NDArray[np.float32]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 3 and arr.shape == (face.h, face.w, PHASE1_CHANNELS):
        return arr
    if arr.ndim == 2 and arr.shape == (face.n_cells, PHASE1_CHANNELS):
        return arr.reshape(face.h, face.w, PHASE1_CHANNELS)
    raise ValueError(
        f"values shape {arr.shape} incompatible with face {face.w}x{face.h}x2"
    )


def _pack_values(arr_hw2: NDArray[np.float32], dtype: ValueDtype) -> bytes:
    flat = np.ascontiguousarray(arr_hw2.reshape(-1, PHASE1_CHANNELS))
    if dtype == ValueDtype.F32:
        return flat.astype("<f4").tobytes()
    if dtype == ValueDtype.F16:
        return flat.astype("<f2").tobytes()
    raise ValueError(f"unsupported dtype {dtype}")


def _unpack_values(
    data: bytes, n: int, dtype: ValueDtype
) -> NDArray[np.float32]:
    if dtype == ValueDtype.F32:
        need = n * PHASE1_CHANNELS * 4
        raw = np.frombuffer(data[:need], dtype="<f4")
    elif dtype == ValueDtype.F16:
        need = n * PHASE1_CHANNELS * 2
        raw = np.frombuffer(data[:need], dtype="<f2").astype(np.float32)
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    return raw.reshape(n, PHASE1_CHANNELS).copy()


def build_keyframe(
    tick: int,
    face: FaceBox,
    values: NDArray[np.floating],
    *,
    labels: TickLabels | None = None,
    conf: NDArray[np.uint8] | None = None,
    value_dtype: ValueDtype = ValueDtype.F16,
    world_hash: int = 0,
) -> TickPackage:
    hw2 = _as_hw2(values, face)
    flags = 0
    if labels is not None:
        flags |= FLAG_HAS_LABELS
    if conf is not None:
        flags |= FLAG_HAS_CONF
        conf = np.asarray(conf, dtype=np.uint8).reshape(face.n_cells)
    return TickPackage(
        kind=PackageKind.KEYFRAME,
        tick=int(tick),
        face=face,
        values=hw2,
        conf=conf,
        labels=labels,
        value_dtype=value_dtype,
        delta_encoding=DeltaEncoding.NONE,
        flags=flags,
        world_hash=int(world_hash) & 0xFFFFFFFFFFFFFFFF,
    )


def build_delta(
    tick: int,
    face: FaceBox,
    prev: NDArray[np.floating],
    curr: NDArray[np.floating],
    *,
    labels: TickLabels | None = None,
    value_dtype: ValueDtype = ValueDtype.F16,
    world_hash: int = 0,
    eps: float = DELTA_EPS,
) -> TickPackage:
    """Build DELTA of curr − prev (velocity snapshots)."""
    a = _as_hw2(prev, face)
    b = _as_hw2(curr, face)
    d = b - a
    flat = d.reshape(-1, PHASE1_CHANNELS)
    mag = np.max(np.abs(flat), axis=1)
    changed = np.flatnonzero(mag >= float(eps)).astype(np.uint16)
    flags = FLAG_HAS_LABELS if labels is not None else 0
    frac = float(changed.size) / float(max(face.n_cells, 1))

    if changed.size == 0:
        return TickPackage(
            kind=PackageKind.DELTA,
            tick=int(tick),
            face=face,
            labels=labels,
            value_dtype=value_dtype,
            delta_encoding=DeltaEncoding.EMPTY,
            flags=flags,
            world_hash=int(world_hash) & 0xFFFFFFFFFFFFFFFF,
        )

    if frac > SPARSE_DENSE_THRESHOLD:
        return TickPackage(
            kind=PackageKind.DELTA,
            tick=int(tick),
            face=face,
            values=d,
            labels=labels,
            value_dtype=value_dtype,
            delta_encoding=DeltaEncoding.DENSE_DELTA,
            flags=flags,
            world_hash=int(world_hash) & 0xFFFFFFFFFFFFFFFF,
        )

    return TickPackage(
        kind=PackageKind.DELTA,
        tick=int(tick),
        face=face,
        sparse_idx=changed,
        sparse_delta=flat[changed].astype(np.float32),
        labels=labels,
        value_dtype=value_dtype,
        delta_encoding=DeltaEncoding.SPARSE_DELTA,
        flags=flags,
        world_hash=int(world_hash) & 0xFFFFFFFFFFFFFFFF,
    )


def encode(package: TickPackage) -> bytes:
    """Serialize to wire bytes (header + optional labels + body)."""
    face = package.face
    body = bytearray()
    enc = package.delta_encoding

    if package.kind == PackageKind.KEYFRAME:
        if package.values is None:
            raise ValueError("KEYFRAME requires values")
        body += _pack_values(_as_hw2(package.values, face), package.value_dtype)
        if package.flags & FLAG_HAS_CONF:
            if package.conf is None:
                raise ValueError("HAS_CONF set but conf missing")
            body += np.asarray(package.conf, dtype=np.uint8).reshape(-1).tobytes()
    elif package.kind == PackageKind.DELTA:
        if enc == DeltaEncoding.EMPTY:
            pass
        elif enc == DeltaEncoding.DENSE_DELTA:
            if package.values is None:
                raise ValueError("DENSE_DELTA requires values")
            body += _pack_values(_as_hw2(package.values, face), package.value_dtype)
        elif enc == DeltaEncoding.SPARSE_DELTA:
            if package.sparse_idx is None or package.sparse_delta is None:
                raise ValueError("SPARSE_DELTA requires idx/delta")
            idx = np.asarray(package.sparse_idx, dtype="<u2")
            count = np.uint32(idx.size)
            body += struct.pack("<I", int(count))
            body += idx.tobytes()
            deltas = np.asarray(package.sparse_delta, dtype=np.float32).reshape(
                -1, PHASE1_CHANNELS
            )
            if package.value_dtype == ValueDtype.F16:
                body += deltas.astype("<f2").tobytes()
            else:
                body += deltas.astype("<f4").tobytes()
        else:
            raise ValueError(f"bad delta_encoding {enc}")
    else:
        raise ValueError(f"encode does not support kind {package.kind}")

    labels_blob = b""
    flags = package.flags
    if package.labels is not None:
        flags |= FLAG_HAS_LABELS
        labels_blob = package.labels.pack()
    elif flags & FLAG_HAS_LABELS:
        labels_blob = TickLabels().pack()

    payload = labels_blob + bytes(body)
    # crc over header fields that precede crc + payload — fill crc after
    header_wo_crc = _HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        int(package.kind),
        int(package.tick),
        float(package.resolved_time()),
        int(face.x),
        int(face.y),
        int(face.w),
        int(face.h),
        int(package.channel_mask) & 0xFFFFFFFF,
        int(package.value_dtype),
        int(enc if package.kind == PackageKind.DELTA else DeltaEncoding.NONE),
        int(flags) & 0xFF,
        0,
        len(payload),
        0,  # placeholder crc
        int(package.world_hash) & 0xFFFFFFFFFFFFFFFF,
        b"\x00" * 16,
    )
    # Recompute with real crc of (header with crc=0)[0:36] is awkward; crc payload only
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = _HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        int(package.kind),
        int(package.tick),
        float(package.resolved_time()),
        int(face.x),
        int(face.y),
        int(face.w),
        int(face.h),
        int(package.channel_mask) & 0xFFFFFFFF,
        int(package.value_dtype),
        int(enc if package.kind == PackageKind.DELTA else DeltaEncoding.NONE),
        int(flags) & 0xFF,
        0,
        len(payload),
        crc,
        int(package.world_hash) & 0xFFFFFFFFFFFFFFFF,
        b"\x00" * 16,
    )
    return header + payload


def decode(blob: bytes) -> TickPackage:
    if len(blob) < HEADER_BYTES:
        raise ValueError("buffer shorter than header")
    (
        magic,
        version,
        kind,
        tick,
        time_seconds,
        face_x,
        face_y,
        face_w,
        face_h,
        channel_mask,
        value_dtype,
        delta_encoding,
        flags,
        _reserved0,
        payload_bytes,
        crc,
        world_hash,
        _reserved1,
    ) = _HEADER_STRUCT.unpack(blob[:HEADER_BYTES])
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    payload = blob[HEADER_BYTES : HEADER_BYTES + payload_bytes]
    if len(payload) != payload_bytes:
        raise ValueError("truncated payload")
    if crc != 0 and (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
        raise ValueError("crc32 mismatch")

    face = FaceBox(face_x, face_y, face_w, face_h)
    offset = 0
    labels = None
    if flags & FLAG_HAS_LABELS:
        labels = TickLabels.unpack(payload[offset : offset + LABELS_BYTES])
        offset += LABELS_BYTES

    kind_e = PackageKind(kind)
    dtype_e = ValueDtype(value_dtype)
    enc_e = DeltaEncoding(delta_encoding)
    values = None
    sparse_idx = None
    sparse_delta = None
    conf = None
    body = payload[offset:]

    if kind_e == PackageKind.KEYFRAME:
        n = face.n_cells
        elem = 4 if dtype_e == ValueDtype.F32 else 2
        need = n * PHASE1_CHANNELS * elem
        flat = _unpack_values(body, n, dtype_e)
        values = flat.reshape(face.h, face.w, PHASE1_CHANNELS)
        rest = body[need:]
        if flags & FLAG_HAS_CONF:
            conf = np.frombuffer(rest[:n], dtype=np.uint8).copy()
    elif kind_e == PackageKind.DELTA:
        if enc_e == DeltaEncoding.EMPTY:
            pass
        elif enc_e == DeltaEncoding.DENSE_DELTA:
            flat = _unpack_values(body, face.n_cells, dtype_e)
            values = flat.reshape(face.h, face.w, PHASE1_CHANNELS)
        elif enc_e == DeltaEncoding.SPARSE_DELTA:
            (count,) = struct.unpack_from("<I", body, 0)
            idx_bytes = 4
            sparse_idx = np.frombuffer(
                body[idx_bytes : idx_bytes + count * 2], dtype="<u2"
            ).copy()
            val_off = idx_bytes + count * 2
            elem = 4 if dtype_e == ValueDtype.F32 else 2
            raw = body[val_off : val_off + count * PHASE1_CHANNELS * elem]
            if dtype_e == ValueDtype.F16:
                sparse_delta = (
                    np.frombuffer(raw, dtype="<f2")
                    .astype(np.float32)
                    .reshape(count, PHASE1_CHANNELS)
                    .copy()
                )
            else:
                sparse_delta = (
                    np.frombuffer(raw, dtype="<f4")
                    .reshape(count, PHASE1_CHANNELS)
                    .copy()
                )
        else:
            raise ValueError(f"bad delta encoding {enc_e}")
    else:
        raise ValueError(f"unsupported kind {kind_e}")

    return TickPackage(
        kind=kind_e,
        tick=int(tick),
        face=face,
        values=values,
        sparse_idx=sparse_idx,
        sparse_delta=sparse_delta,
        conf=conf,
        labels=labels,
        value_dtype=dtype_e,
        delta_encoding=enc_e,
        channel_mask=int(channel_mask),
        world_hash=int(world_hash),
        flags=int(flags),
        time_seconds=float(time_seconds),
    )


def apply_to_state(
    state: NDArray[np.floating],
    package: TickPackage,
) -> NDArray[np.float32]:
    """CPU reference apply: state is (H,W,2) face patch velocity."""
    face = package.face
    out = np.asarray(state, dtype=np.float32).reshape(face.h, face.w, PHASE1_CHANNELS).copy()
    if package.kind == PackageKind.KEYFRAME:
        if package.values is None:
            raise ValueError("KEYFRAME missing values")
        out[:] = _as_hw2(package.values, face)
        return out
    if package.kind != PackageKind.DELTA:
        raise ValueError("apply_to_state expects KEYFRAME or DELTA")
    if package.delta_encoding == DeltaEncoding.EMPTY:
        return out
    if package.delta_encoding == DeltaEncoding.DENSE_DELTA:
        if package.values is None:
            raise ValueError("DENSE_DELTA missing values")
        out += _as_hw2(package.values, face)
        return out
    if package.delta_encoding == DeltaEncoding.SPARSE_DELTA:
        if package.sparse_idx is None or package.sparse_delta is None:
            raise ValueError("SPARSE_DELTA missing data")
        flat = out.reshape(-1, PHASE1_CHANNELS)
        idx = np.asarray(package.sparse_idx, dtype=np.int64)
        flat[idx] += np.asarray(package.sparse_delta, dtype=np.float32)
        return out
    raise ValueError(f"bad encoding {package.delta_encoding}")


__all__ = [
    "FaceBox",
    "TickLabels",
    "TickPackage",
    "apply_to_state",
    "build_delta",
    "build_keyframe",
    "decode",
    "encode",
]
