"""CHORUS Fabric transport for TickFeed — two lanes (design §6.2).

Lane A: ``c_t`` float32[dim] compact codes via ``send_direct``.
Lane B: TickPackage bytes — zlib + framed into dim-vectors, or TPK_REF + spool
when the compressed body exceeds the inline budget.
"""

from __future__ import annotations

import os
import zlib
from pathlib import Path
from typing import Any

import numpy as np

# Match L4 CODE_DIM; control plane must be started with CHORUS_DIM=64.
DEFAULT_DIM = 64
# Distinctive finite metas (must not be NaN — fabric matmul).
TPK_CHUNK_MAGIC = 20260802.0
TPK_REF_MAGIC = 20260803.0
# Reserve floats 0..7 for framing; remaining store bytes as 0..255 floats.
_META = 8
# Inline when compressed size fits a modest chunk count (lab default).
DEFAULT_INLINE_MAX = int(os.environ.get("AIFACE_CHORUS_TPK_INLINE_MAX", "4096"))


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
        spool_codes: bool = True,
        spool_keep: int = 240,
        inline_max: int = DEFAULT_INLINE_MAX,
    ) -> None:
        self.world = Path(world)
        self.world = self.world if self.world.is_dir() else self.world.parent
        self.dim = int(dim)
        self.spool_packages = bool(spool_packages)
        self.spool_codes = bool(spool_codes)
        self.spool_keep = max(8, int(spool_keep))
        self.inline_max = max(64, int(inline_max))
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
        """Pack zlib-compressed bytes into dim float32 vectors (safe 0..255)."""
        compressed = zlib.compress(blob, level=6)
        crc = zlib.crc32(blob) & 0xFFFFFFFF
        capacity = self._bytes_per_chunk()
        n_chunks = max(1, (len(compressed) + capacity - 1) // capacity)
        out: list[np.ndarray] = []
        for i in range(n_chunks):
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = TPK_CHUNK_MAGIC
            vec[1] = float(tick)
            vec[2] = float(n_chunks)
            vec[3] = float(i)
            vec[4] = float(len(blob))
            vec[5] = float(crc)
            vec[6] = float(len(compressed))
            start = i * capacity
            piece = compressed[start : start + capacity]
            for j, b in enumerate(piece):
                vec[_META + j] = float(b)
            out.append(vec)
        return out

    def _send_ref(self, tick: int, blob: bytes, path: Path) -> bool:
        crc = zlib.crc32(blob) & 0xFFFFFFFF
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = TPK_REF_MAGIC
        vec[1] = float(tick)
        vec[2] = float(len(blob))
        vec[3] = float(crc)
        # Encode spool basename hash lightly for debug (not a security check).
        name = path.name.encode("utf-8")[:48]
        for j, b in enumerate(name):
            vec[_META + j] = float(b)
        return self._send_vec(vec)

    def push_code(self, tick: int, code: list[float] | np.ndarray) -> Path | None:
        vec = np.asarray(code, dtype=np.float32).reshape(-1)
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        vec = vec[: self.dim].astype(np.float32)
        sent = self._send_vec(vec)
        if sent and self.spool_codes:
            path = self.spool / f"tick_{int(tick):08d}.f32"
            path.write_bytes(np.ascontiguousarray(vec, dtype="<f4").tobytes())
            self._trim("tick_*.f32")
            return None
        if not self.spool_codes:
            return None
        path = self.spool / f"tick_{int(tick):08d}.f32"
        path.write_bytes(np.ascontiguousarray(vec, dtype="<f4").tobytes())
        self._trim("tick_*.f32")
        return path

    def push_package_bytes(self, tick: int, blob: bytes) -> Path | None:
        """Lane B: push TickPackage bytes on CHORUS (inline chunks or TPK_REF)."""
        self.packages_sent += 1
        # Always keep HELLO negotiate artifacts and a short package spool for QA.
        if int(tick) < 0:
            path = self.spool / f"hello_{abs(int(tick)):02d}.tpk"
            path.write_bytes(blob)
            if self._client is not None:
                compressed = zlib.compress(blob, level=6)
                if len(compressed) <= self.inline_max:
                    for vec in self._frame_chunks(blob, tick):
                        if not self._send_vec(vec):
                            break
                    else:
                        self.packages_inlined += 1
                else:
                    if self._send_ref(tick, blob, path):
                        self.packages_refed += 1
            return path

        path: Path | None = None
        if self.spool_packages:
            path = self.spool / f"pkg_{int(tick):08d}.tpk"
            path.write_bytes(blob)
            self._trim("pkg_*.tpk")

        if self._client is None:
            return path

        compressed = zlib.compress(blob, level=6)
        if len(compressed) <= self.inline_max:
            ok = True
            for vec in self._frame_chunks(blob, tick):
                if not self._send_vec(vec):
                    ok = False
                    break
            if ok:
                self.packages_inlined += 1
                return path
        else:
            if path is None:
                path = self.spool / f"pkg_{int(tick):08d}.tpk"
                path.write_bytes(blob)
                self._trim("pkg_*.tpk")
            if self._send_ref(tick, blob, path):
                self.packages_refed += 1
        return path

    def pull_latest_code(self) -> np.ndarray | None:
        files = sorted(self.spool.glob("tick_*.f32"))
        if not files:
            return None
        raw = files[-1].read_bytes()
        vec = np.frombuffer(raw, dtype="<f4").copy()
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        return vec[: self.dim].astype(np.float32)

    def pull_latest_package_bytes(self) -> bytes | None:
        files = sorted(self.spool.glob("pkg_*.tpk"))
        if not files:
            return None
        return files[-1].read_bytes()


__all__ = [
    "DEFAULT_DIM",
    "DEFAULT_INLINE_MAX",
    "TPK_CHUNK_MAGIC",
    "TPK_REF_MAGIC",
    "TickFeedTransport",
]
