"""TickFeedDriver — produce KEY/DELTA packages and drive the master clock."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aiface.tickfeed.package import (
    FaceBox,
    TickPackage,
    build_delta,
    build_keyframe,
    decode,
)
from aiface.tickfeed.ring import LockstepPlayer, FaceVelocityState
from aiface.tickfeed.schema import KEY_REFRESH_TICKS, ValueDtype
from aiface.tickfeed.synth import labels_from_drives, synthesize_velocity


@dataclass
class TickFeedDriver:
    """Live full-face velocity packages for NWR ingest."""

    face: FaceBox
    mouth_uv: tuple[float, float]
    player: LockstepPlayer
    prev_velocity: NDArray[np.float32] | None = None
    sent_key: bool = False
    ticks_since_key: int = 0
    timeline: dict[int, NDArray[np.float32]] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def create(
        cls,
        face: FaceBox,
        mouth_uv: tuple[float, float],
    ) -> TickFeedDriver:
        state = FaceVelocityState.zeros(face)
        return cls(
            face=face,
            mouth_uv=mouth_uv,
            player=LockstepPlayer(state=state),
        )

    @classmethod
    def try_load_timeline(cls, world: Path | str, face: FaceBox, mouth_uv: tuple[float, float]) -> TickFeedDriver:
        driver = cls.create(face, mouth_uv)
        path = Path(world)
        root = path if path.is_dir() else path.parent
        npz = root / "face_cell_timeline.npz"
        if not npz.is_file():
            return driver
        data = np.load(npz)
        ticks = np.asarray(data["ticks"], dtype=np.int64)
        vel = np.asarray(data["velocity"], dtype=np.float32)
        for i, t in enumerate(ticks):
            driver.timeline[int(t)] = vel[i]
        print(
            f"TickFeedDriver: loaded timeline {npz.name} "
            f"({len(driver.timeline)} ticks)"
        )
        return driver

    def push_drives(
        self,
        *,
        tick: int,
        open_amt: float,
        smile_amt: float,
        surprise_amt: float = 0.0,
        phoneme: str = "REST",
        emotion: str = "NEUTRAL",
        word: str = "",
    ) -> TickPackage:
        labels = labels_from_drives(
            phoneme=phoneme,
            smile_amt=smile_amt,
            open_amt=open_amt,
            surprise_amt=surprise_amt,
            emotion=emotion,
            word=word,
        )
        if tick in self.timeline:
            curr = self.timeline[tick]
        else:
            curr = synthesize_velocity(
                self.face,
                open_amt=open_amt,
                smile_amt=smile_amt,
                surprise_amt=surprise_amt,
                mouth_uv=self.mouth_uv,
            )
        need_key = (
            not self.sent_key
            or self.ticks_since_key >= KEY_REFRESH_TICKS
            or self.prev_velocity is None
        )
        if need_key:
            pkg = build_keyframe(
                tick,
                self.face,
                curr,
                labels=labels,
                value_dtype=ValueDtype.F16,
            )
            self.sent_key = True
            self.ticks_since_key = 0
        else:
            pkg = build_delta(
                tick,
                self.face,
                self.prev_velocity,
                curr,
                labels=labels,
                value_dtype=ValueDtype.F16,
            )
            self.ticks_since_key += 1
        self.prev_velocity = np.asarray(curr, dtype=np.float32).copy()
        self.player.submit(pkg)
        return pkg

    def pop_for_master(self, master_tick: int) -> TickPackage | None:
        """Align ring to master tick and return package or None (miss → damp)."""
        # Advance internal player if behind
        while self.player.master_tick < master_tick:
            self.player.step()
        if self.player.master_tick == master_tick:
            pkg = self.player.ring.pop_ready(master_tick)
            self.player.state.apply_or_damp(master_tick, pkg)
            self.player.master_tick = master_tick + 1
            return pkg
        return None


def face_box_from_profile(world: Path | str, grid_w: int = 256, grid_h: int = 256) -> FaceBox:
    from aiface.avatar_profile import open_avatar

    bundle = open_avatar(world)
    box = bundle.profile.geometry.face_box or {}
    x = int(max(0, min(grid_w - 1, round(float(box.get("x", 0.0))))))
    y = int(max(0, min(grid_h - 1, round(float(box.get("y", 0.0))))))
    w = int(max(1, min(grid_w - x, round(float(box.get("width", grid_w))))))
    h = int(max(1, min(grid_h - y, round(float(box.get("height", grid_h))))))
    return FaceBox(x=x, y=y, w=w, h=h)


__all__ = ["TickFeedDriver", "face_box_from_profile"]
