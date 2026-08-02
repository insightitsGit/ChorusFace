"""3-tick lockstep ring + miss damp (bridge B3)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aiface.tickfeed.package import FaceBox, TickPackage, apply_to_state
from aiface.tickfeed.schema import RING_DEPTH, VELOCITY_MISS_DAMP, PackageKind


@dataclass
class TickRingBuffer:
    """Hold a few upcoming TickPackages keyed by tick index."""

    depth: int = RING_DEPTH
    _buf: OrderedDict[int, TickPackage] = field(default_factory=OrderedDict)

    def push(self, package: TickPackage) -> None:
        self._buf[int(package.tick)] = package
        while len(self._buf) > self.depth * 4:
            # Cap memory if producer runs ahead; keep highest ticks.
            self._buf.popitem(last=False)

    def pop_ready(self, tick: int) -> TickPackage | None:
        return self._buf.pop(int(tick), None)

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class FaceVelocityState:
    """CPU-side face patch state for tests / offline apply."""

    face: FaceBox
    velocity: NDArray[np.float32]
    last_tick: int = -1
    miss_damp: float = VELOCITY_MISS_DAMP

    @classmethod
    def zeros(cls, face: FaceBox) -> FaceVelocityState:
        vel = np.zeros((face.h, face.w, 2), dtype=np.float32)
        return cls(face=face, velocity=vel)

    def apply_or_damp(self, tick: int, package: TickPackage | None) -> str:
        """Apply package for ``tick`` or damp on miss. Returns source tag."""
        if package is None:
            self.velocity *= float(self.miss_damp)
            self.last_tick = int(tick)
            return "damp"
        if package.tick != int(tick):
            raise ValueError(f"package tick {package.tick} != master {tick}")
        if package.kind == PackageKind.KEYFRAME:
            self.velocity = apply_to_state(self.velocity, package)
            self.last_tick = int(tick)
            return "keyframe"
        if self.last_tick < 0 and package.kind == PackageKind.DELTA:
            # No prior KEY — treat dense/sparse as absolute if we must
            self.velocity = apply_to_state(self.velocity, package)
            self.last_tick = int(tick)
            return "delta_without_key"
        self.velocity = apply_to_state(self.velocity, package)
        self.last_tick = int(tick)
        return "delta"


@dataclass
class LockstepPlayer:
    """Master-clock consumer: ring → apply/damp each tick."""

    state: FaceVelocityState
    ring: TickRingBuffer = field(default_factory=TickRingBuffer)
    master_tick: int = 0

    def submit(self, package: TickPackage) -> None:
        self.ring.push(package)

    def step(self) -> str:
        pkg = self.ring.pop_ready(self.master_tick)
        tag = self.state.apply_or_damp(self.master_tick, pkg)
        self.master_tick += 1
        return tag


__all__ = [
    "FaceVelocityState",
    "LockstepPlayer",
    "TickRingBuffer",
]
