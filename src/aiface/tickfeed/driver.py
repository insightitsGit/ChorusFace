"""TickFeedDriver — produce KEY/DELTA packages and drive the master clock."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aiface.tickfeed.calibration import beat_at_time, load_calibration_script
from aiface.tickfeed.chorus_transport import TickFeedTransport
from aiface.tickfeed.cosmetics import CosmeticPrefs, load_cosmetic_prefs
from aiface.tickfeed.ml.runtime import TickFeedMLStack
from aiface.tickfeed.package import (
    FaceBox,
    TickLabels,
    TickPackage,
    build_delta,
    build_keyframe,
    encode,
)
from aiface.tickfeed.ring import FaceVelocityState, LockstepPlayer
from aiface.tickfeed.schema import (
    KEY_REFRESH_TICKS,
    TICK_RATE_HZ,
    EmotionId,
    ValueDtype,
)
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
    timeline_conf: dict[int, NDArray[np.uint8]] = field(default_factory=dict)
    enabled: bool = True
    ml: TickFeedMLStack | None = None
    transport: TickFeedTransport | None = None
    last_code: list[float] = field(default_factory=list)
    calibration: dict | None = None
    cosmetics: CosmeticPrefs | None = None
    world: Path | None = None

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
    def try_load_timeline(
        cls, world: Path | str, face: FaceBox, mouth_uv: tuple[float, float]
    ) -> TickFeedDriver:
        driver = cls.create(face, mouth_uv)
        path = Path(world)
        root = path if path.is_dir() else path.parent
        driver.world = root
        driver.calibration = load_calibration_script(root)
        driver.cosmetics = load_cosmetic_prefs(root)
        npz = root / "face_cell_timeline.npz"
        if npz.is_file():
            data = np.load(npz)
            ticks = np.asarray(data["ticks"], dtype=np.int64)
            vel = np.asarray(data["velocity"], dtype=np.float32)
            conf = None
            if "conf" in data.files:
                conf = np.asarray(data["conf"], dtype=np.uint8)
            for i, t in enumerate(ticks):
                driver.timeline[int(t)] = vel[i]
                if conf is not None:
                    driver.timeline_conf[int(t)] = conf[i].reshape(-1)
            print(
                f"TickFeedDriver: loaded timeline {npz.name} "
                f"({len(driver.timeline)} ticks)"
            )
        driver.ml = TickFeedMLStack.try_load(root)
        try:
            from aiface.tickfeed.ml.train import CODE_DIM

            driver.transport = TickFeedTransport(
                world=root, dim=CODE_DIM, use_chorus=True, spool_packages=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"TickFeed transport: {exc}")
        return driver

    def _labels_for_tick(
        self,
        *,
        tick: int,
        open_amt: float,
        smile_amt: float,
        surprise_amt: float,
        phoneme: str,
        emotion: str,
        word: str,
        speech_viseme: int | None = None,
    ) -> TickLabels:
        labels = labels_from_drives(
            phoneme=phoneme,
            smile_amt=smile_amt,
            open_amt=open_amt,
            surprise_amt=surprise_amt,
            emotion=emotion,
            word=word,
        )
        if speech_viseme is not None:
            labels.viseme_id = int(speech_viseme)
        if self.calibration is not None:
            t = float(tick) / float(TICK_RATE_HZ)
            # Only stamp script beats inside the 8s calibration window.
            if t < float(self.calibration.get("duration_s") or 8.0):
                beat = beat_at_time(self.calibration, t)
                labels.beat_id = int(beat.get("beat_id", labels.beat_id))
                speech = str(beat.get("speech") or "")
                if speech and not word:
                    labels.word = speech[:16]
                bid = str(beat.get("id") or "")
                if bid == "ANGRY":
                    labels.emotion_id = int(EmotionId.ANGRY)
                elif bid == "SURPRISE":
                    labels.emotion_id = int(EmotionId.SURPRISED)
                elif bid == "SMILE":
                    labels.emotion_id = int(EmotionId.HAPPY)
        return labels

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
        labels = self._labels_for_tick(
            tick=tick,
            open_amt=open_amt,
            smile_amt=smile_amt,
            surprise_amt=surprise_amt,
            phoneme=phoneme,
            emotion=emotion,
            word=word,
        )
        curr: NDArray[np.float32] | None = None
        conf: NDArray[np.uint8] | None = None
        code: list[float] = []
        source = "synth"

        # Authority: measured timeline > ML decode > synth
        if tick in self.timeline:
            curr = self.timeline[tick]
            conf = self.timeline_conf.get(tick)
            source = "timeline"
            mean_conf = float(np.mean(conf)) if conf is not None else 255.0
            # Low-confidence measured cells → L5 gap prior blend when available
            if (
                mean_conf < 90.0
                and self.ml is not None
                and self.ml.l5 is not None
            ):
                speech, look, flat, code = self.ml.resolve(
                    tick=tick,
                    open_amt=open_amt,
                    smile_amt=smile_amt,
                    surprise_amt=surprise_amt,
                    use_gap_prior=True,
                )
                try:
                    gap = flat.reshape(self.face.h, self.face.w, 2)
                    alpha = 1.0 - (mean_conf / 255.0)
                    curr = (1.0 - alpha) * curr + alpha * gap
                    source = "timeline+l5"
                except ValueError:
                    pass
                labels = self._labels_for_tick(
                    tick=tick,
                    open_amt=float(look.open),
                    smile_amt=float(look.smile),
                    surprise_amt=float(look.surprise),
                    phoneme=phoneme,
                    emotion=emotion,
                    word=speech.word or word,
                    speech_viseme=int(speech.viseme_id),
                )
        elif self.ml is not None:
            speech, look, flat, code = self.ml.resolve(
                tick=tick,
                open_amt=open_amt,
                smile_amt=smile_amt,
                surprise_amt=surprise_amt,
            )
            labels = self._labels_for_tick(
                tick=tick,
                open_amt=float(look.open),
                smile_amt=float(look.smile),
                surprise_amt=float(look.surprise),
                phoneme=phoneme,
                emotion=emotion,
                word=speech.word or word,
                speech_viseme=int(speech.viseme_id),
            )
            try:
                curr = flat.reshape(self.face.h, self.face.w, 2)
                source = "ml"
                conf = np.full(self.face.n_cells, 140, dtype=np.uint8)
            except ValueError:
                curr = None
        if curr is None:
            curr = synthesize_velocity(
                self.face,
                open_amt=open_amt,
                smile_amt=smile_amt,
                surprise_amt=surprise_amt,
                mouth_uv=self.mouth_uv,
            )
            conf = np.full(self.face.n_cells, 100, dtype=np.uint8)
            source = "synth"

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
                conf=conf,
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
        self.last_code = code
        # Encode L4 code from measured/ML patch when not already produced
        if self.ml is not None and not code:
            try:
                code = self.ml.encode_patch(curr)
                self.last_code = code
            except Exception:  # noqa: BLE001
                pass
        if self.transport is not None:
            if self.last_code:
                self.transport.push_code(tick, self.last_code)
            self.transport.push_package_bytes(tick, encode(pkg))
        pkg.time_seconds = float(tick) / float(TICK_RATE_HZ)
        del source  # authority path used above; not on wire
        return pkg

    def pop_for_master(self, master_tick: int) -> TickPackage | None:
        while self.player.master_tick < master_tick:
            self.player.step()
        if self.player.master_tick == master_tick:
            pkg = self.player.ring.pop_ready(master_tick)
            self.player.state.apply_or_damp(master_tick, pkg)
            self.player.master_tick = master_tick + 1
            return pkg
        return None


def face_box_from_profile(
    world: Path | str, grid_w: int = 256, grid_h: int = 256
) -> FaceBox:
    from aiface.avatar_profile import open_avatar

    bundle = open_avatar(world)
    box = bundle.profile.geometry.face_box or {}
    x = int(max(0, min(grid_w - 1, round(float(box.get("x", 0.0))))))
    y = int(max(0, min(grid_h - 1, round(float(box.get("y", 0.0))))))
    w = int(max(1, min(grid_w - x, round(float(box.get("width", grid_w))))))
    h = int(max(1, min(grid_h - y, round(float(box.get("height", grid_h))))))
    return FaceBox(x=x, y=y, w=w, h=h)


__all__ = ["TickFeedDriver", "face_box_from_profile"]
