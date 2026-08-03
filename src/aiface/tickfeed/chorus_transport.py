"""CHORUS Fabric transport for TickFeed — two lanes (design §6.2).

Lane A: ``c_t`` float32[dim] compact codes via ``send_direct``.
Lane B: TickPackage bytes — zlib + framed into dim-vectors, or TPK_REF + spool
when the compressed body exceeds the inline budget.

**float32 wire rule:** every meta field must be exactly representable in IEEE-754
binary32 (≤ 2^24 for integers). Magics stay small; CRC32 is split into two
uint16 halves. Never store a full u32 CRC in one float32.
"""

from __future__ import annotations

import os
import zlib
from pathlib import Path
from typing import Any

import numpy as np

# Match L4 CODE_DIM; control plane must be started with CHORUS_DIM=64.
DEFAULT_DIM = 64
# Exact in float32 (integers with |n| ≤ 2^24). Do NOT use calendar dates > 2^24.
TPK_CHUNK_MAGIC = 64101.0
TPK_REF_MAGIC = 64102.0
# Reserve floats 0..7 for framing; remaining store bytes as 0..255 floats.
_META = 8
# Inline when compressed size fits a modest chunk count (lab default).
DEFAULT_INLINE_MAX = int(os.environ.get("AIFACE_CHORUS_TPK_INLINE_MAX", "4096"))
# Trim spool every N writes (not every push) — NTFS glob/sort is expensive.
DEFAULT_TRIM_EVERY = int(os.environ.get("AIFACE_CHORUS_SPOOL_TRIM_EVERY", "60"))
# PackageKind.KEYFRAME — avoid importing package (cycle risk).
_KIND_KEYFRAME = 1


def _f32(x: float) -> float:
    """Round-trip through float32 (what CHORUS actually sends)."""
    return float(np.float32(x))


def crc32_to_halves(crc: int) -> tuple[float, float]:
    """Split u32 CRC into two uint16 halves (each exact in float32)."""
    c = int(crc) & 0xFFFFFFFF
    return float(c & 0xFFFF), float((c >> 16) & 0xFFFF)


def crc32_from_halves(lo: float, hi: float) -> int:
    return (int(round(lo)) & 0xFFFF) | ((int(round(hi)) & 0xFFFF) << 16)


def assert_f32_exact_int(value: float, *, name: str = "meta") -> None:
    """Raise if ``value`` is not an exact float32 integer (receiver contract)."""
    if _f32(value) != float(value):
        raise ValueError(f"{name}={value!r} is not exactly representable in float32")


# Fail fast at import if magics regress.
assert_f32_exact_int(TPK_CHUNK_MAGIC, name="TPK_CHUNK_MAGIC")
assert_f32_exact_int(TPK_REF_MAGIC, name="TPK_REF_MAGIC")


class TickFeedTransport:
    """One-way push of codes + TickPackage bytes (producer → fabric / spool)."""

    def __init__(
        self,
        *,
        world: Path | str,
        dim: int = DEFAULT_DIM,
        use_chorus: bool = True,
        control_plane: str | None = None,
        target: str | None = None,
        spool_packages: bool = True,
        spool_codes: bool = False,
        spool_keys_only: bool = True,
        spool_keep: int = 240,
        inline_max: int = DEFAULT_INLINE_MAX,
        trim_every: int = DEFAULT_TRIM_EVERY,
    ) -> None:
        self.world = Path(world)
        self.world = self.world if self.world.is_dir() else self.world.parent
        self.dim = int(dim)
        self.spool_packages = bool(spool_packages)
        self.spool_codes = bool(spool_codes)
        self.spool_keys_only = bool(spool_keys_only)
        self.spool_keep = max(1, int(spool_keep))
        self.inline_max = max(64, int(inline_max))
        self.trim_every = max(1, int(trim_every))
        self.spool = self.world / "tickfeed_chorus_spool"
        self.spool.mkdir(parents=True, exist_ok=True)
        self._client: Any = None
        self.mode = "spool"
        self.control_plane = control_plane or os.environ.get(
            "AIFACE_CHORUS_CONTROL", "localhost:50051"
        )
        self.target = target or os.environ.get(
            "AIFACE_CHORUS_TARGET", "localhost:50053"
        )
        self.packages_sent = 0
        self.packages_inlined = 0
        self.packages_refed = 0
        self._writes_since_trim = 0
        # In-process latest — wire-loop pulls these without per-tick disk I/O.
        self._latest_code: np.ndarray | None = None
        self._latest_package: bytes | None = None
        # Lane-B framed CHUNK buffer for runtime reassemble (not tests-only).
        self._lane_b_frames: list[np.ndarray] = []
        self._lane_b_tick: int | None = None
        if use_chorus:
            self._try_chorus()

    def _try_chorus(self) -> None:
        os.environ.setdefault("CHORUS_DIM", str(self.dim))
        try:
            from chorus_fabric import ChorusClient

            client = ChorusClient(
                pod_id="aiface-tickfeed",
                control_plane=self.control_plane,
                relay=None,
                target=self.target,
                dim=self.dim,
            )
            try:
                client.handshake()
                self._client = client
                self.mode = "chorus"
                print(
                    f"TickFeed transport: CHORUS Fabric "
                    f"cp={self.control_plane} target={self.target} dim={self.dim}"
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"TickFeed transport: CHORUS unavailable ({exc}); using spool"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"TickFeed transport: chorus-fabric import failed ({exc})")

    def _trim(self, pattern: str) -> None:
        files = sorted(self.spool.glob(pattern))
        overflow = len(files) - self.spool_keep
        for path in files[: max(0, overflow)]:
            try:
                path.unlink()
            except OSError:
                pass

    def _maybe_trim(self, pattern: str) -> None:
        self._writes_since_trim += 1
        if self._writes_since_trim >= self.trim_every:
            self._writes_since_trim = 0
            self._trim(pattern)

    def _send_vec(self, vec: np.ndarray) -> bool:
        if self._client is None:
            return False
        try:
            import torch

            signal = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32))
            self._client.send_direct(signal)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"TickFeed CHORUS send failed ({exc}); spooling")
            self._client = None
            self.mode = "spool"
            return False

    def _bytes_per_chunk(self) -> int:
        return max(1, self.dim - _META)

    def _frame_chunks(self, blob: bytes, tick: int) -> list[np.ndarray]:
        """Pack zlib-compressed bytes into dim float32 vectors (safe metas + bytes)."""
        compressed = zlib.compress(blob, level=6)
        crc = zlib.crc32(blob) & 0xFFFFFFFF
        crc_lo, crc_hi = crc32_to_halves(crc)
        capacity = self._bytes_per_chunk()
        n_chunks = max(1, (len(compressed) + capacity - 1) // capacity)
        # Tick / lengths must also survive float32 (lab ticks << 2^24).
        assert_f32_exact_int(float(abs(int(tick))), name="tick")
        assert_f32_exact_int(float(n_chunks), name="n_chunks")
        assert_f32_exact_int(float(len(blob)), name="nbytes")
        assert_f32_exact_int(float(len(compressed)), name="compressed_len")
        out: list[np.ndarray] = []
        for i in range(n_chunks):
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = np.float32(TPK_CHUNK_MAGIC)
            vec[1] = np.float32(tick)
            vec[2] = np.float32(n_chunks)
            vec[3] = np.float32(i)
            vec[4] = np.float32(len(blob))
            vec[5] = np.float32(crc_lo)
            vec[6] = np.float32(crc_hi)
            vec[7] = np.float32(len(compressed))
            start = i * capacity
            piece = compressed[start : start + capacity]
            for j, b in enumerate(piece):
                vec[_META + j] = np.float32(b)
            out.append(vec)
        return out

    def _send_ref(self, tick: int, blob: bytes, path: Path) -> bool:
        crc = zlib.crc32(blob) & 0xFFFFFFFF
        crc_lo, crc_hi = crc32_to_halves(crc)
        assert_f32_exact_int(float(abs(int(tick))), name="tick")
        assert_f32_exact_int(float(len(blob)), name="nbytes")
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = np.float32(TPK_REF_MAGIC)
        vec[1] = np.float32(tick)
        vec[2] = np.float32(len(blob))
        vec[3] = np.float32(crc_lo)
        vec[4] = np.float32(crc_hi)
        # Encode spool basename as 0..255 floats for debug (not a security check).
        name = path.name.encode("utf-8")[:48]
        for j, b in enumerate(name):
            vec[_META + j] = np.float32(b)
        return self._send_vec(vec)

    def push_code(self, tick: int, code: list[float] | np.ndarray) -> Path | None:
        vec = np.asarray(code, dtype=np.float32).reshape(-1)
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        vec = vec[: self.dim].astype(np.float32)
        self._latest_code = vec.copy()
        sent = self._send_vec(vec)
        if not self.spool_codes:
            return None
        path = self.spool / f"tick_{int(tick):08d}.f32"
        path.write_bytes(np.ascontiguousarray(vec, dtype="<f4").tobytes())
        self._maybe_trim("tick_*.f32")
        return None if sent else path

    def push_package_bytes(
        self, tick: int, blob: bytes, *, kind: int | None = None
    ) -> Path | None:
        """Lane B: push TickPackage bytes on CHORUS (inline chunks or TPK_REF)."""
        self.packages_sent += 1
        self._latest_package = bytes(blob)
        # Always keep HELLO negotiate artifacts and a short package spool for QA.
        if int(tick) < 0:
            path = self.spool / f"hello_{abs(int(tick)):02d}.tpk"
            path.write_bytes(blob)
            if self._client is not None:
                compressed = zlib.compress(blob, level=6)
                if len(compressed) <= self.inline_max:
                    frames = self._frame_chunks(blob, tick)
                    self._lane_b_frames = [f.copy() for f in frames]
                    self._lane_b_tick = int(tick)
                    for vec in frames:
                        if not self._send_vec(vec):
                            break
                    else:
                        self.packages_inlined += 1
                else:
                    if self._send_ref(tick, blob, path):
                        self.packages_refed += 1
            return path

        is_key = kind is None or int(kind) == _KIND_KEYFRAME
        write_spool = self.spool_packages and (
            not self.spool_keys_only or is_key
        )
        path: Path | None = None
        if write_spool:
            path = self.spool / f"pkg_{int(tick):08d}.tpk"
            path.write_bytes(blob)
            self._maybe_trim("pkg_*.tpk")

        if self._client is None:
            return path

        compressed = zlib.compress(blob, level=6)
        if len(compressed) <= self.inline_max:
            frames = self._frame_chunks(blob, tick)
            # Always buffer frames for local reassemble consume path.
            self._lane_b_frames = [f.copy() for f in frames]
            self._lane_b_tick = int(tick)
            ok = True
            for vec in frames:
                if not self._send_vec(vec):
                    ok = False
                    break
            if ok:
                self.packages_inlined += 1
                return path
        else:
            # TPK_REF needs a durable spool file for the remote reader.
            if path is None:
                path = self.spool / f"pkg_{int(tick):08d}.tpk"
                path.write_bytes(blob)
                self._maybe_trim("pkg_*.tpk")
            if self._send_ref(tick, blob, path):
                self.packages_refed += 1
        return path

    def pull_latest_code(self) -> np.ndarray | None:
        if self._latest_code is not None:
            return self._latest_code.copy()
        files = sorted(self.spool.glob("tick_*.f32"))
        if not files:
            return None
        raw = files[-1].read_bytes()
        vec = np.frombuffer(raw, dtype="<f4").copy()
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        return vec[: self.dim].astype(np.float32)

    def pull_latest_package_bytes(self) -> bytes | None:
        if self._latest_package is not None:
            return self._latest_package
        # Prefer reassembled lane-B CHUNKs when memory blob was cleared.
        rebuilt = self.pull_package_from_lane_b_frames()
        if rebuilt is not None:
            return rebuilt
        files = sorted(self.spool.glob("pkg_*.tpk"))
        if not files:
            return None
        return files[-1].read_bytes()

    def pull_package_from_lane_b_frames(self) -> bytes | None:
        """Runtime lane-B consume: reassemble buffered CHUNK frames → TickPackage."""
        if not self._lane_b_frames:
            return None
        try:
            blob = reassemble_lane_b_chunks(self._lane_b_frames)
        except ValueError:
            return None
        self._latest_package = blob
        return blob


def parse_lane_b_header(vec: np.ndarray) -> dict[str, Any]:
    """Decode lane-B meta from one float32 vector (after wire float32 cast)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size < _META:
        raise ValueError("vector too short for lane-B header")
    magic = float(v[0])
    if magic == _f32(TPK_CHUNK_MAGIC):
        return {
            "kind": "chunk",
            "magic": magic,
            "tick": int(round(float(v[1]))),
            "n_chunks": int(round(float(v[2]))),
            "chunk_i": int(round(float(v[3]))),
            "nbytes": int(round(float(v[4]))),
            "crc32": crc32_from_halves(float(v[5]), float(v[6])),
            "compressed_len": int(round(float(v[7]))),
        }
    if magic == _f32(TPK_REF_MAGIC):
        return {
            "kind": "ref",
            "magic": magic,
            "tick": int(round(float(v[1]))),
            "nbytes": int(round(float(v[2]))),
            "crc32": crc32_from_halves(float(v[3]), float(v[4])),
        }
    raise ValueError(f"unknown lane-B magic {magic}")


def reassemble_lane_b_chunks(frames: list[np.ndarray]) -> bytes:
    """Reassemble zlib payload from CHUNK frames; verify CRC32 of original blob."""
    if not frames:
        raise ValueError("no frames")
    paired = [(parse_lane_b_header(f), np.asarray(f, dtype=np.float32)) for f in frames]
    if any(h["kind"] != "chunk" for h, _ in paired):
        raise ValueError("expected CHUNK frames only")
    paired.sort(key=lambda item: item[0]["chunk_i"])
    n = paired[0][0]["n_chunks"]
    if len(paired) != n:
        raise ValueError(f"expected {n} chunks, got {len(paired)}")
    capacity = max(1, int(paired[0][1].shape[0]) - _META)
    parts: list[bytes] = []
    for _h, v in paired:
        raw = bytes(int(round(float(v[_META + j]))) & 0xFF for j in range(capacity))
        parts.append(raw)
    header0 = paired[0][0]
    compressed = b"".join(parts)[: header0["compressed_len"]]
    blob = zlib.decompress(compressed)
    expect = int(header0["crc32"])
    got = zlib.crc32(blob) & 0xFFFFFFFF
    if got != expect:
        raise ValueError(f"lane-B CRC mismatch got={got:#x} expect={expect:#x}")
    if len(blob) != header0["nbytes"]:
        raise ValueError("lane-B nbytes mismatch")
    return blob


__all__ = [
    "DEFAULT_DIM",
    "DEFAULT_INLINE_MAX",
    "TPK_CHUNK_MAGIC",
    "TPK_REF_MAGIC",
    "TickFeedTransport",
    "assert_f32_exact_int",
    "crc32_from_halves",
    "crc32_to_halves",
    "parse_lane_b_header",
    "reassemble_lane_b_chunks",
]
