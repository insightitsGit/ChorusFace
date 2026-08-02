"""CHORUS Fabric transport for TickFeed compact codes / packages.

Starts against a live control plane + target when available; otherwise falls
back to a local binary spool (same float32 payload).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# Match L4 CODE_DIM; control plane must be started with CHORUS_DIM=64.
DEFAULT_DIM = 64


class TickFeedTransport:
    """One-way push of float32 vectors (design: producer → NWR master)."""

    def __init__(
        self,
        *,
        world: Path | str,
        dim: int = DEFAULT_DIM,
        use_chorus: bool = True,
        control_plane: str | None = None,
        target: str | None = None,
        spool_packages: bool = False,
        spool_codes: bool = True,
        spool_keep: int = 240,
    ) -> None:
        self.world = Path(world)
        self.world = self.world if self.world.is_dir() else self.world.parent
        self.dim = int(dim)
        self.spool_packages = bool(spool_packages)
        self.spool_codes = bool(spool_codes)
        self.spool_keep = max(8, int(spool_keep))
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
        if use_chorus:
            self._try_chorus()

    def _try_chorus(self) -> None:
        # Ensure client/server dim match TickFeed codes.
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

    def push_code(self, tick: int, code: list[float] | np.ndarray) -> Path | None:
        vec = np.asarray(code, dtype=np.float32).reshape(-1)
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        vec = vec[: self.dim]
        if self._client is not None:
            try:
                import torch

                signal = torch.from_numpy(vec.copy())
                self._client.send_direct(signal)
                # Also keep a short spool for QA replay
                if self.spool_codes:
                    path = self.spool / f"tick_{int(tick):08d}.f32"
                    path.write_bytes(
                        np.ascontiguousarray(vec, dtype="<f4").tobytes()
                    )
                    self._trim("tick_*.f32")
                return None
            except Exception as exc:  # noqa: BLE001
                print(f"TickFeed CHORUS send failed ({exc}); spooling")
                self._client = None
                self.mode = "spool"
        if not self.spool_codes:
            return None
        path = self.spool / f"tick_{int(tick):08d}.f32"
        path.write_bytes(np.ascontiguousarray(vec, dtype="<f4").tobytes())
        self._trim("tick_*.f32")
        return path

    def push_package_bytes(self, tick: int, blob: bytes) -> Path | None:
        if int(tick) < 0:
            path = self.spool / f"hello_{abs(int(tick)):02d}.tpk"
            path.write_bytes(blob)
            return path
        if not self.spool_packages:
            return None
        path = self.spool / f"pkg_{int(tick):08d}.tpk"
        path.write_bytes(blob)
        self._trim("pkg_*.tpk")
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


__all__ = ["DEFAULT_DIM", "TickFeedTransport"]
