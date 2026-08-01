"""Runtime behavior driver — measured transitions first, ML fills gaps."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np

from aiface.behavior.schema import (
    CONTROL_DIM,
    HISTORY,
    BehaviorState,
    landmarks_to_controls,
    model_path,
)
from aiface.behavior.track import TransitionTrack, load_transition_track
from aiface.live_vector.features import (
    heuristic_vector,
    rms_history_features,
    sample_envelope_rms,
)


class BehaviorDriver:
    """Resolve mouth-group controls for cell plan + plates.

    Authority: measured track → ML fill → viseme table.
    """

    def __init__(
        self,
        *,
        track: TransitionTrack | None = None,
        pipeline: object | None = None,
        noise_floor: float = 0.0,
        peak_hint: float = 0.0,
    ) -> None:
        self.track = track
        self._pipeline = pipeline
        self.noise_floor = float(noise_floor)
        self.peak_hint = float(peak_hint)
        self._rms: deque[float] = deque(maxlen=HISTORY)
        self.last = BehaviorState()

    @classmethod
    def try_load(cls, world: Path | str) -> "BehaviorDriver":
        world = Path(world)
        track = load_transition_track(world)
        path = model_path(world)
        pipeline = None
        noise = 0.0
        peak = 0.0
        if path.is_file():
            try:
                import joblib

                payload = joblib.load(path)
                pipeline = (
                    payload.get("pipeline") if isinstance(payload, dict) else payload
                )
                if isinstance(payload, dict):
                    noise = float(payload.get("noise_floor", 0.0))
                    peak = float(payload.get("peak_hint", 0.0))
                print(f"BehaviorDriver: loaded ML fill {path}")
            except Exception as exc:
                print(f"BehaviorDriver: ML load failed ({exc})")
                pipeline = None
        else:
            print("BehaviorDriver: no behavior_model — measured/table only")
        if track is not None:
            noise = noise or track.noise_floor
            peak = peak or track.peak_hint
            print(
                f"BehaviorDriver: measured track {track.n_samples} samples "
                f"({track.duration:.2f}s)"
            )
        elif pipeline is None:
            print("BehaviorDriver: empty — viseme table fallback")
        return cls(
            track=track, pipeline=pipeline, noise_floor=noise, peak_hint=peak
        )

    @property
    def has_track(self) -> bool:
        return self.track is not None and self.track.n_samples > 0

    @property
    def using_ml(self) -> bool:
        return self._pipeline is not None

    @property
    def has_history(self) -> bool:
        return bool(self._rms)

    def clear_history(self) -> None:
        self._rms.clear()

    def push_rms(self, rms: float) -> None:
        self._rms.append(float(rms))

    def push_from_envelope(self, envelope: object | None, time_seconds: float) -> None:
        if envelope is None:
            self.push_rms(0.0)
            return
        self.noise_floor = float(envelope.noise_floor())
        self.peak_hint = float(envelope.peak)
        self.push_rms(sample_envelope_rms(envelope, time_seconds))

    def _ml_predict(self) -> BehaviorState | None:
        if self._pipeline is None:
            return None
        hist: Sequence[float] = list(self._rms) if self._rms else [0.0]
        try:
            feat = rms_history_features(
                hist, noise_floor=self.noise_floor, peak_hint=self.peak_hint
            )
            pred = np.asarray(
                self._pipeline.predict(np.asarray(feat, dtype=np.float64).reshape(1, -1)),
                dtype=np.float64,
            ).reshape(-1)
        except Exception:
            return None
        if pred.size < CONTROL_DIM:
            pad = np.zeros(CONTROL_DIM, dtype=np.float64)
            pad[: pred.size] = pred
            pred = pad
        pred = pred[:CONTROL_DIM]
        pred[0:3] = np.clip(pred[0:3], 0.0, 1.0)
        pred[3:5] = np.clip(pred[3:5], -1.0, 1.0)
        pred[5:8] = np.clip(pred[5:8], 0.0, 1.0)
        return BehaviorState(
            openness_n=float(pred[0]),
            jaw_n=float(pred[1]),
            width_n=float(pred[2]),
            upper_lip_dy=float(pred[3]),
            lower_lip_dy=float(pred[4]),
            corner_dx=float(pred[5]),
            teeth_reveal=float(pred[6]),
            cavity_n=float(pred[7]),
            source="ml_fill",
        )

    def _table_state(self, phoneme: str) -> BehaviorState:
        # Lazy import avoids amin_loop ↔ behavior circular init.
        from aiface.mouth_cell_plan import viseme_flow

        open_n, width_n, round_n = viseme_flow(phoneme)
        # Round reduces corner stretch.
        width = max(0.0, width_n - round_n * 0.35)
        vec = landmarks_to_controls(
            openness_n=open_n, width_n=width, teeth_n=open_n * 0.7
        )
        return BehaviorState(
            openness_n=vec[0],
            jaw_n=vec[1],
            width_n=vec[2],
            upper_lip_dy=vec[3],
            lower_lip_dy=vec[4],
            corner_dx=vec[5],
            teeth_reveal=vec[6],
            cavity_n=vec[7],
            source="table",
        )

    def resolve(
        self,
        *,
        phoneme: str = "REST",
        video_t: float | None = None,
        prefer_ml_when_gap: bool = True,
    ) -> BehaviorState:
        """Resolve controls: measured → ML → table.

        ``video_t``: capture clock when replaying the take. ``None`` for live
        speech (ML / table only).
        """
        # 1) Measured track at capture time.
        if video_t is not None and self.track is not None:
            measured = self.track.sample_at(float(video_t))
            if measured is not None:
                self.last = measured
                return measured
            if prefer_ml_when_gap and self.track.gap_at(float(video_t)):
                filled = self._ml_predict()
                if filled is not None:
                    self.last = filled
                    return filled

        # 2) Live / gap — ML fill from audio history.
        if self._pipeline is not None and self._rms:
            filled = self._ml_predict()
            if filled is not None:
                # Soft blend toward table so viseme clock still matters.
                table = self._table_state(phoneme)
                blend = BehaviorState(
                    openness_n=0.65 * filled.openness_n + 0.35 * table.openness_n,
                    jaw_n=0.65 * filled.jaw_n + 0.35 * table.jaw_n,
                    width_n=0.55 * filled.width_n + 0.45 * table.width_n,
                    upper_lip_dy=0.65 * filled.upper_lip_dy + 0.35 * table.upper_lip_dy,
                    lower_lip_dy=0.65 * filled.lower_lip_dy + 0.35 * table.lower_lip_dy,
                    corner_dx=0.55 * filled.corner_dx + 0.45 * table.corner_dx,
                    teeth_reveal=0.65 * filled.teeth_reveal + 0.35 * table.teeth_reveal,
                    cavity_n=0.65 * filled.cavity_n + 0.35 * table.cavity_n,
                    source="ml_fill",
                )
                self.last = blend
                return blend

        # 3) Heuristic audio if no ML, else pure table.
        if self._rms and self._pipeline is None:
            hist: Sequence[float] = list(self._rms)
            o, j, w = heuristic_vector(
                hist, noise_floor=self.noise_floor, peak_hint=self.peak_hint
            )
            table = self._table_state(phoneme)
            state = BehaviorState(
                openness_n=max(o, table.openness_n),
                jaw_n=max(j, table.jaw_n),
                width_n=max(w, table.width_n * 0.5),
                upper_lip_dy=-max(o, table.openness_n),
                lower_lip_dy=max(o, table.openness_n),
                corner_dx=max(w, table.width_n),
                teeth_reveal=max(o, table.teeth_reveal) * 0.7,
                cavity_n=max(o, table.cavity_n),
                source="heuristic",
            )
            self.last = state
            return state

        state = self._table_state(phoneme)
        self.last = state
        return state

    def snapshot(self) -> dict[str, object]:
        return {
            "has_track": self.has_track,
            "using_ml": self.using_ml,
            "track_samples": int(self.track.n_samples) if self.track else 0,
            "track_duration": float(self.track.duration) if self.track else 0.0,
            "last": self.last.as_dict(),
            "authority": "measured > ml_fill > table",
        }


__all__ = ["BehaviorDriver"]
