"""PulseChunk PLS1 encode/decode (F5–F8)."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from chorusface.vowel.schema import (
    FLAG_HAS_EXT_HEADER,
    FLAG_HAS_WORD_SLICES,
    FLAG_IS_SPOOLED,
    GROUP_DIM,
    MAX_VOWELS_PER_SLICE,
    PLS_EXT_HEADER_BYTES,
    PLS_HEADER_BYTES,
    PLS_MAGIC,
    PLS_VERSION,
    TICK_HZ,
    VOWEL_PAD,
    WORD_SLICE_BYTES,
)

_CORE = struct.Struct("<I B B H Q I H B B I I")
assert _CORE.size == PLS_HEADER_BYTES

_EXT = struct.Struct("<6H")
assert _EXT.size == PLS_EXT_HEADER_BYTES

_SLICE = struct.Struct("<H H B B 6B")
assert _SLICE.size == WORD_SLICE_BYTES


def fnv1a64(text: str) -> int:
    h = 0xCBF29CE484222325
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


@dataclass(slots=True)
class WordSlice:
    start_tick: int
    end_tick: int
    vowel_ids: list[int]
    pause_flag: int = 0

    def pack(self) -> bytes:
        ids = list(self.vowel_ids[:MAX_VOWELS_PER_SLICE])
        n = len(ids)
        while len(ids) < MAX_VOWELS_PER_SLICE:
            ids.append(VOWEL_PAD)
        return _SLICE.pack(
            int(self.start_tick) & 0xFFFF,
            int(self.end_tick) & 0xFFFF,
            n & 0xFF,
            int(self.pause_flag) & 0xFF,
            *ids,
        )

    @classmethod
    def unpack(cls, data: bytes) -> WordSlice:
        vals = _SLICE.unpack(data[:WORD_SLICE_BYTES])
        start, end, n, pause = vals[0], vals[1], vals[2], vals[3]
        ids = [int(v) for v in vals[4 : 4 + n] if v != VOWEL_PAD]
        return cls(start_tick=start, end_tick=end, vowel_ids=ids, pause_flag=pause)


@dataclass(slots=True)
class VersionBlock:
    teacher_ver: int = 1
    dataset_ver: int = 1
    modelA_ver: int = 1
    modelB_ver: int = 1
    decoder_ver: int = 1

    def pack(self) -> bytes:
        return _EXT.pack(
            PLS_EXT_HEADER_BYTES,
            self.teacher_ver & 0xFFFF,
            self.dataset_ver & 0xFFFF,
            self.modelA_ver & 0xFFFF,
            self.modelB_ver & 0xFFFF,
            self.decoder_ver & 0xFFFF,
        )

    @classmethod
    def unpack(cls, data: bytes) -> VersionBlock:
        _len, t, d, a, b, dec = _EXT.unpack(data[:PLS_EXT_HEADER_BYTES])
        return cls(
            teacher_ver=t,
            dataset_ver=d,
            modelA_ver=a,
            modelB_ver=b,
            decoder_ver=dec,
        )


@dataclass(slots=True)
class PulseChunk:
    utterance_id: str
    n_ticks: int
    primary_emotion: int
    word_slices: list[WordSlice] = field(default_factory=list)
    controls: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, GROUP_DIM), dtype=np.float32)
    )
    versions: VersionBlock | None = None
    tick_hz: int = TICK_HZ
    is_spooled: bool = False
    key_ticks: list[int] = field(default_factory=list)

    @property
    def utterance_id_hash(self) -> int:
        return fnv1a64(self.utterance_id)


def _pack_payload(chunk: PulseChunk) -> bytes:
    body = bytearray()
    for ws in chunk.word_slices:
        body.extend(ws.pack())
    ctrl = np.asarray(chunk.controls, dtype=np.float32).reshape(-1, GROUP_DIM)
    if ctrl.shape[0] != chunk.n_ticks:
        raise ValueError(
            f"controls rows {ctrl.shape[0]} != n_ticks {chunk.n_ticks}"
        )
    body.extend(struct.pack("<I", int(chunk.n_ticks) & 0xFFFFFFFF))
    body.extend(ctrl.tobytes(order="C"))
    key_mask = np.zeros(chunk.n_ticks, dtype=np.uint8)
    for t in chunk.key_ticks:
        if 0 <= t < chunk.n_ticks:
            key_mask[t] = 1
    body.extend(key_mask.tobytes())
    return bytes(body)


def encode_pulsechunk(chunk: PulseChunk) -> bytes:
    """Serialize PulseChunk to PLS1 bytes."""
    flags = FLAG_HAS_WORD_SLICES
    ext = b""
    if chunk.versions is not None:
        flags |= FLAG_HAS_EXT_HEADER
        ext = chunk.versions.pack()
    if chunk.is_spooled:
        flags |= FLAG_IS_SPOOLED

    payload = _pack_payload(chunk)
    # payload_bytes = ext + payload (CRC covers pre[0:28] + ext + payload)
    payload_bytes = len(ext) + len(payload)
    pre = struct.pack(
        "<I B B H Q I H B B I",
        PLS_MAGIC,
        PLS_VERSION,
        flags & 0xFF,
        len(chunk.word_slices) & 0xFFFF,
        chunk.utterance_id_hash,
        int(chunk.n_ticks) & 0xFFFFFFFF,
        int(chunk.tick_hz) & 0xFFFF,
        int(chunk.primary_emotion) & 0xFF,
        0,
        payload_bytes & 0xFFFFFFFF,
    )
    assert len(pre) == 28
    crc = zlib.crc32(pre)
    crc = zlib.crc32(ext, crc)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    header = pre + struct.pack("<I", crc)
    assert len(header) == PLS_HEADER_BYTES
    return header + ext + payload


def decode_pulsechunk(data: bytes, utterance_id: str | None = None) -> PulseChunk:
    if len(data) < PLS_HEADER_BYTES:
        raise ValueError("PulseChunk too short")
    (
        magic,
        ver,
        flags,
        n_words,
        uid_hash,
        n_ticks,
        tick_hz,
        primary_emotion,
        _res,
        payload_bytes,
        crc,
    ) = _CORE.unpack(data[:PLS_HEADER_BYTES])
    if magic != PLS_MAGIC:
        raise ValueError(f"bad PulseChunk magic {magic:#x}")
    if ver != PLS_VERSION:
        raise ValueError(f"unsupported pulse_ver {ver}")

    offset = PLS_HEADER_BYTES
    versions: VersionBlock | None = None
    ext = b""
    if flags & FLAG_HAS_EXT_HEADER:
        ext = data[offset : offset + PLS_EXT_HEADER_BYTES]
        versions = VersionBlock.unpack(ext)
        offset += PLS_EXT_HEADER_BYTES

    payload_len = payload_bytes - len(ext)
    payload = data[offset : offset + payload_len]
    if len(payload) != payload_len:
        raise ValueError("PulseChunk truncated payload")

    pre = data[:28]
    check = zlib.crc32(pre)
    check = zlib.crc32(ext, check)
    check = zlib.crc32(payload, check) & 0xFFFFFFFF
    if check != crc:
        raise ValueError(f"PulseChunk CRC mismatch {check:#x} != {crc:#x}")

    pos = 0
    slices: list[WordSlice] = []
    for _ in range(n_words):
        slices.append(WordSlice.unpack(payload[pos : pos + WORD_SLICE_BYTES]))
        pos += WORD_SLICE_BYTES

    (n_ticks2,) = struct.unpack_from("<I", payload, pos)
    pos += 4
    if n_ticks2 != n_ticks:
        raise ValueError("n_ticks mismatch in payload")
    nbytes = n_ticks * GROUP_DIM * 4
    ctrl = (
        np.frombuffer(payload[pos : pos + nbytes], dtype=np.float32)
        .reshape(n_ticks, GROUP_DIM)
        .copy()
    )
    pos += nbytes
    key_mask = np.frombuffer(payload[pos : pos + n_ticks], dtype=np.uint8)
    key_ticks = [int(i) for i, v in enumerate(key_mask) if v]

    uid = utterance_id or f"hash:{uid_hash:016x}"
    return PulseChunk(
        utterance_id=uid,
        n_ticks=n_ticks,
        primary_emotion=primary_emotion,
        word_slices=slices,
        controls=ctrl,
        versions=versions,
        tick_hz=tick_hz,
        is_spooled=bool(flags & FLAG_IS_SPOOLED),
        key_ticks=key_ticks,
    )
