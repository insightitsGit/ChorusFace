"""TickFeedDriver — produce KEY/DELTA packages and drive the master clock."""

from __future__ import annotations

import os
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
    build_hello,
    build_keyframe,
    encode,
    negotiate_hello,
)
from aiface.tickfeed.ring import FaceVelocityState, LockstepPlayer
from aiface.tickfeed.schema import (
    KEY_REFRESH_TICKS,
    TICK_RATE_HZ,
    BeatId,
    EmotionId,
    ValueDtype,
)
from aiface.tickfeed.synth import labels_from_drives, synthesize_velocity
from aiface.tickfeed.timeline_io import (
    SOURCE_MEASURED,
    SOURCE_SYNTH,
    load_timeline_bundle,
)


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
    timeline_source: dict[int, int] = field(default_factory=dict)
    speech_by_tick: dict[int, dict] = field(default_factory=dict)
    look_by_tick: dict[int, dict] = field(default_factory=dict)
    enabled: bool = True
    ml: TickFeedMLStack | None = None
    transport: TickFeedTransport | None = None
    last_code: list[float] = field(default_factory=list)
    last_labels: TickLabels | None = None
    # LOOK freeze on ring miss — last package the master actually applied.
    last_applied_labels: TickLabels | None = None
    last_package: TickPackage | None = None
    hello_done: bool = False
    hello_ack_ok: bool = False
    calibration: dict | None = None
    cosmetics: CosmeticPrefs | None = None
    world: Path | None = None
    timeline_length: int = 0
    # One calibration pass then settle at REST (looping forever looked "never still").
    loop_timeline: bool = False
    # Master consumes from transport (c_t or lane-B bytes) instead of local ring produce.
    wire_loop: bool = False
    # Fidelity default: lane-B package. Lane-A c_t is bandwidth / lossy PCA.
    wire_loop_source: str = "package"  # "package" | "code"
    wire_prev_velocity: NDArray[np.float32] | None = None
    wire_sent_key: bool = False
    wire_ticks_since_key: int = 0

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

    def _is_rest_still_gate(
        self, labels: TickLabels, *, tick: int, live_speech: bool
    ) -> bool:
        """True only for neutral REST — never erase ANGRY/SMILE/SURPRISE motion."""
        if live_speech:
            return False
        emo = int(labels.emotion_id)
        if emo not in (int(EmotionId.NEUTRAL), int(EmotionId.UNKNOWN)):
            return False
        beat = int(labels.beat_id)
        if beat not in (int(BeatId.REST), int(BeatId.UNKNOWN)):
            return False
        teacher = self._teacher_tick(tick)
        brow = 0.0
        if teacher in self.look_by_tick:
            brow = float(self.look_by_tick[teacher].get("brow") or 0.0)
        return (
            float(labels.open_amt) < 0.08
            and float(labels.surprise_amt) < 0.08
            and float(labels.smile_amt) < 0.12
            and brow < 0.15
        )

    def _teacher_tick(self, tick: int) -> int:
        """Map master tick onto measured teacher length (loop for demo idle)."""
        if self.timeline_length <= 0:
            return int(tick)
        if self.loop_timeline:
            return int(tick) % int(self.timeline_length)
        return int(tick)

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
        try:
            bundle = load_timeline_bundle(root)
            ticks = np.asarray(bundle["ticks"], dtype=np.int64)
            vel = np.asarray(bundle["velocity"], dtype=np.float32)
            conf = np.asarray(bundle["conf"], dtype=np.uint8)
            source = np.asarray(
                bundle.get("source")
                if bundle.get("source") is not None
                else np.zeros(len(ticks), dtype=np.uint8),
                dtype=np.uint8,
            )
            for i, t in enumerate(ticks):
                driver.timeline[int(t)] = vel[i]
                driver.timeline_conf[int(t)] = conf[i].reshape(-1)
                driver.timeline_source[int(t)] = int(source[i]) if i < len(source) else 0
            driver.timeline_length = int(len(ticks))
            if bundle.get("speech"):
                for row in bundle["speech"].get("ticks") or []:
                    driver.speech_by_tick[int(row["tick"])] = row
            if bundle.get("look"):
                for row in bundle["look"].get("ticks") or []:
                    driver.look_by_tick[int(row["tick"])] = row
            n_meas = sum(1 for s in driver.timeline_source.values() if s == SOURCE_MEASURED)
            n_synth = sum(1 for s in driver.timeline_source.values() if s == SOURCE_SYNTH)
            print(
                f"TickFeedDriver: loaded timeline "
                f"({len(driver.timeline)} ticks, loop={driver.loop_timeline}, "
                f"speech={len(driver.speech_by_tick)}, "
                f"look={len(driver.look_by_tick)}, "
                f"source measured={n_meas} synth={n_synth})"
            )
        except FileNotFoundError:
            pass
        driver.ml = TickFeedMLStack.try_load(root)
        try:
            import os

            from aiface.tickfeed.ml.train import CODE_DIM

            spool_all = os.environ.get("AIFACE_CHORUS_SPOOL_ALL", "").lower() in (
                "1",
                "true",
                "yes",
            )
            driver.transport = TickFeedTransport(
                world=root,
                dim=CODE_DIM,
                use_chorus=True,
                # Memory always holds latest; disk spool is KEY-only unless QA env set.
                spool_packages=True,
                spool_codes=spool_all,
                spool_keys_only=not spool_all,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"TickFeed transport: {exc}")
        driver.run_hello()
        return driver

    def run_hello(self) -> TickPackage:
        """HELLO → local HELLO_ACK negotiate (design handshake §1)."""
        world_id = self.world.name if self.world is not None else "avatar"
        hello = build_hello(self.face, world_id=world_id)
        ack = negotiate_hello(hello)
        self.hello_done = True
        self.hello_ack_ok = bool(ack.hello and ack.hello.ok)
        if self.transport is not None:
            self.transport.push_package_bytes(-1, encode(hello))
            self.transport.push_package_bytes(-2, encode(ack))
        print(
            f"TickFeed HELLO: ack_ok={self.hello_ack_ok} "
            f"apply_mode={(ack.hello.apply_mode if ack.hello else '?')}"
        )
        return ack

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
        live_speech: bool = False,
        brow_amt: float = 0.0,
        lid_amt: float = 1.0,
    ) -> TickLabels:
        teacher = self._teacher_tick(tick)
        # Live chat/TTS owns amounts; otherwise Side B teachers are sole authority.
        if not live_speech:
            if teacher in self.look_by_tick:
                lk = self.look_by_tick[teacher]
                smile_amt = float(lk.get("smile") or 0.0)
                open_amt = float(lk.get("open") or 0.0)
                surprise_amt = float(lk.get("surprise") or 0.0)
                brow_amt = float(lk.get("brow") or brow_amt)
                if "lid" in lk:
                    lid_amt = float(lk.get("lid") if lk.get("lid") is not None else lid_amt)
                if int(lk.get("emotion_id", -1)) >= 0:
                    emotion_map = {
                        int(EmotionId.HAPPY): "HAPPY",
                        int(EmotionId.SURPRISED): "SURPRISED",
                        int(EmotionId.ANGRY): "ANGRY",
                    }
                    emotion = emotion_map.get(int(lk["emotion_id"]), emotion)
            if teacher in self.speech_by_tick:
                sp = self.speech_by_tick[teacher]
                phoneme = str(sp.get("viseme") or phoneme)
                word = str(sp.get("word") or word)
                speech_viseme = int(sp.get("viseme_id", speech_viseme or 0))

        labels = labels_from_drives(
            phoneme=phoneme,
            smile_amt=smile_amt,
            open_amt=open_amt,
            surprise_amt=surprise_amt,
            emotion=emotion,
            word=word,
            brow_amt=brow_amt,
            lid_amt=lid_amt,
        )
        if speech_viseme is not None:
            labels.viseme_id = int(speech_viseme)
        if self.calibration is not None and not live_speech:
            t = float(teacher) / float(TICK_RATE_HZ)
            duration = float(self.calibration.get("duration_s") or 8.0)
            if self.loop_timeline and duration > 0:
                t = t % duration
            if t < duration:
                beat = beat_at_time(self.calibration, t)
                labels.beat_id = int(beat.get("beat_id", labels.beat_id))
                speech = str(beat.get("speech") or "")
                if speech and not labels.word:
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
        live_speech: bool = False,
    ) -> TickPackage:
        teacher = self._teacher_tick(tick)
        past_end = (
            not live_speech
            and self.timeline_length > 0
            and not self.loop_timeline
            and int(tick) >= int(self.timeline_length)
        )
        if past_end:
            # Settled REST after one calibration pass — no ML thrash.
            open_amt = 0.0
            smile_amt = 0.0
            surprise_amt = 0.0
            phoneme = "REST"
            emotion = "NEUTRAL"
            word = ""
        labels = self._labels_for_tick(
            tick=tick,
            open_amt=open_amt,
            smile_amt=smile_amt,
            surprise_amt=surprise_amt,
            phoneme=phoneme,
            emotion=emotion,
            word=word,
            live_speech=live_speech,
        )
        curr: NDArray[np.float32] | None = None
        conf: NDArray[np.uint8] | None = None
        code: list[float] = []

        if past_end:
            curr = np.zeros((self.face.h, self.face.w, 2), dtype=np.float32)
            conf = np.full(self.face.n_cells, 255, dtype=np.uint8)
        # Authority: live speech synth/ML > measured timeline > ML decode > synth
        elif live_speech:
            # Hearing / 0-state overlay: closed lips own LOOK; keep FIELD still
            # so ML does not invent mouth motion while the user types.
            closed_live = (
                str(phoneme).upper() in {"REST", "CLOSED", "PP", "MM"}
                and float(open_amt) < 0.08
            )
            if closed_live:
                curr = np.zeros((self.face.h, self.face.w, 2), dtype=np.float32)
                conf = np.full(self.face.n_cells, 255, dtype=np.uint8)
            else:
                # Live chat/TTS: L3 owns FIELD when it produces real motion;
                # synth is fallback only (never sold as measured).
                curr = None
                conf = None
                if self.ml is not None:
                    try:
                        _s, _l, flat, code = self.ml.resolve(
                            tick=tick,
                            open_amt=open_amt,
                            smile_amt=smile_amt,
                            surprise_amt=surprise_amt,
                        )
                        del _s, _l
                        ml = flat.reshape(self.face.h, self.face.w, 2)
                        if float(np.abs(ml).max()) >= 0.04:
                            curr = ml.astype(np.float32)
                            conf = np.full(self.face.n_cells, 170, dtype=np.uint8)
                    except Exception:  # noqa: BLE001
                        pass
                if curr is None:
                    curr = synthesize_velocity(
                        self.face,
                        open_amt=float(open_amt),
                        smile_amt=float(smile_amt),
                        surprise_amt=float(surprise_amt),
                        mouth_uv=self.mouth_uv,
                    )
                    conf = np.full(self.face.n_cells, 140, dtype=np.uint8)
        elif teacher in self.timeline:
            src_code = int(self.timeline_source.get(teacher, SOURCE_MEASURED))
            curr = self.timeline[teacher]
            conf = self.timeline_conf.get(teacher)
            if conf is not None:
                conf = np.asarray(conf, dtype=np.uint8).reshape(-1).copy()
            mean_conf = float(np.mean(conf)) if conf is not None else 255.0
            # Provenance gate: synth ticks never keep measured-high conf.
            if src_code != SOURCE_MEASURED:
                mean_conf = min(mean_conf, 80.0)
                if conf is not None:
                    conf = np.minimum(conf, np.uint8(120))
            allow_gap = (
                mean_conf < 90.0
                or src_code != SOURCE_MEASURED
            )
            if allow_gap and self.ml is not None and self.ml.l5 is not None:
                speech, look, flat, code = self.ml.resolve(
                    tick=teacher,
                    open_amt=open_amt,
                    smile_amt=smile_amt,
                    surprise_amt=surprise_amt,
                    use_gap_prior=True,
                )
                try:
                    gap = flat.reshape(self.face.h, self.face.w, 2)
                    alpha = 1.0 - (mean_conf / 255.0)
                    if src_code != SOURCE_MEASURED:
                        alpha = max(alpha, 0.55)
                    curr = (1.0 - alpha) * curr + alpha * gap
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
                    live_speech=False,
                    brow_amt=float(look.brow),
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
                live_speech=False,
            )
            try:
                curr = flat.reshape(self.face.h, self.face.w, 2)
                conf = np.full(self.face.n_cells, 140, dtype=np.uint8)
            except ValueError:
                curr = None
        if curr is None:
            curr = synthesize_velocity(
                self.face,
                open_amt=float(labels.open_amt),
                smile_amt=float(labels.smile_amt),
                surprise_amt=float(labels.surprise_amt),
                mouth_uv=self.mouth_uv,
            )
            conf = np.full(self.face.n_cells, 100, dtype=np.uint8)

        # Still face only for neutral REST — never invent stillness for ANGRY/etc.
        if self._is_rest_still_gate(labels, tick=tick, live_speech=live_speech):
            curr = np.zeros((self.face.h, self.face.w, 2), dtype=np.float32)
            conf = np.full(self.face.n_cells, 255, dtype=np.uint8)

        curr_f = np.asarray(curr, dtype=np.float32)
        still = float(np.abs(curr_f).max()) < 1e-6
        prev_still = (
            self.prev_velocity is not None
            and float(np.abs(self.prev_velocity).max()) < 1e-6
        )
        # Steady design: KEY then Δ. Absolute KEY every tick is QA-only
        # (AIFACE_TICKFEED_ABSOLUTE=1) to diagnose Δ residue.
        absolute = os.environ.get("AIFACE_TICKFEED_ABSOLUTE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        need_key = (
            absolute
            or not self.sent_key
            or self.ticks_since_key >= KEY_REFRESH_TICKS
            or self.prev_velocity is None
            or (still and not prev_still)
        )
        if need_key:
            pkg = build_keyframe(
                tick,
                self.face,
                curr_f,
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
                curr_f,
                labels=labels,
                conf=conf,
                value_dtype=ValueDtype.F16,
            )
            self.ticks_since_key += 1
        self.prev_velocity = curr_f.copy()
        self.last_labels = labels
        self.last_code = code
        if self.ml is not None and not code:
            try:
                self.last_code = self.ml.encode_patch(curr_f)
            except Exception:  # noqa: BLE001
                pass
        if self.transport is not None:
            if self.last_code:
                self.transport.push_code(tick, self.last_code)
            self.transport.push_package_bytes(
                tick, encode(pkg), kind=int(pkg.kind)
            )
        # Local ring is fed here unless wire-loop (master pulls transport instead).
        if not self.wire_loop:
            self.player.submit(pkg)
        pkg.time_seconds = float(tick) / float(TICK_RATE_HZ)
        self.last_package = pkg
        return pkg

    def pop_for_master(self, master_tick: int) -> TickPackage | None:
        """Consume ring for master tick (None → GPU miss damp)."""
        # Advance any skipped ticks with damp on CPU state
        while self.player.master_tick < master_tick:
            self.player.step()
        if self.player.master_tick != master_tick:
            return None
        pkg = self.player.ring.pop_ready(master_tick)
        self.player.state.apply_or_damp(master_tick, pkg)
        self.player.master_tick = master_tick + 1
        if pkg is not None and pkg.labels is not None:
            self.last_labels = pkg.labels
            self.last_applied_labels = pkg.labels
        return pkg

    def expand_code_to_package(
        self,
        tick: int,
        code: list[float] | NDArray[np.floating],
        *,
        open_amt: float = 0.0,
        smile_amt: float = 0.0,
        surprise_amt: float = 0.0,
    ) -> TickPackage:
        """Side A receive path: L4 decode c_t → TickPackage → ring."""
        if self.ml is None:
            raise RuntimeError("ML stack required to expand c_t")
        flat = self.ml.decode_code(code)
        curr = flat.reshape(self.face.h, self.face.w, 2).astype(np.float32)
        labels = self._labels_for_tick(
            tick=tick,
            open_amt=open_amt,
            smile_amt=smile_amt,
            surprise_amt=surprise_amt,
            phoneme="REST",
            emotion="NEUTRAL",
            word="",
        )
        conf = np.full(self.face.n_cells, 130, dtype=np.uint8)
        # Wire-loop keeps a separate KEY/Δ chain so produce prev_velocity stays intact.
        if self.wire_loop:
            prev = self.wire_prev_velocity
            need_key = prev is None or not self.wire_sent_key
            if not need_key and self.wire_ticks_since_key >= KEY_REFRESH_TICKS:
                need_key = True
            if need_key:
                pkg = build_keyframe(
                    tick, self.face, curr, labels=labels, conf=conf
                )
                self.wire_sent_key = True
                self.wire_ticks_since_key = 0
            else:
                assert prev is not None
                pkg = build_delta(
                    tick, self.face, prev, curr, labels=labels, conf=conf
                )
                self.wire_ticks_since_key += 1
            self.wire_prev_velocity = curr.copy()
        elif self.prev_velocity is None or not self.sent_key:
            pkg = build_keyframe(
                tick, self.face, curr, labels=labels, conf=conf
            )
            self.sent_key = True
            self.ticks_since_key = 0
            self.prev_velocity = curr.copy()
        else:
            pkg = build_delta(
                tick,
                self.face,
                self.prev_velocity,
                curr,
                labels=labels,
                conf=conf,
            )
            self.ticks_since_key += 1
            self.prev_velocity = curr.copy()
        self.player.submit(pkg)
        self.last_code = list(np.asarray(code, dtype=np.float32).reshape(-1))
        self.last_labels = labels
        self.last_package = pkg
        return pkg

    def pull_remote_code_if_any(self, tick: int) -> TickPackage | None:
        """If transport has a newer c_t, expand and submit to the ring."""
        if self.transport is None or self.ml is None:
            return None
        code = self.transport.pull_latest_code()
        if code is None:
            return None
        return self.expand_code_to_package(tick, code)

    def ingest_from_wire(self, tick: int) -> TickPackage | None:
        """Wire-loop consumer: transport → ring (c_t expand or lane-B decode)."""
        if self.transport is None:
            return None
        source = (self.wire_loop_source or "package").strip().lower()
        if source == "package":
            from aiface.tickfeed.package import decode

            blob = self.transport.pull_latest_package_bytes()
            if blob is None:
                return None
            pkg = decode(blob)
            # Retarget to the produce tick so the producer-lead ring stays aligned.
            pkg.tick = int(tick)
            self.player.submit(pkg)
            if pkg.labels is not None:
                self.last_labels = pkg.labels
            self.last_package = pkg
            return pkg
        return self.pull_remote_code_if_any(tick)


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
