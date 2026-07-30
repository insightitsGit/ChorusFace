"""Runtime driver: audio RMS history → live controls for the GPU path.

NWR-first: no Path A ownership seal. Tables + model propose jaw/openness;
Master Lock on the GPU is the only hard reject for identity cells.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence

from aiface.live_vector.features import (
    heuristic_vector,
    rms_history_features,
    sample_envelope_rms,
)
from aiface.live_vector.schema import (
    HISTORY,
    OPEN_VISEMES,
    LiveControlVector,
    model_path,
    plate_gate,
)


class LiveVectorDriver:
    """From-scratch runtime authority for mouth live vectors."""

    def __init__(
        self,
        *,
        pipeline: object | None = None,
        noise_floor: float = 0.0,
        peak_hint: float = 0.0,
    ) -> None:
        self._pipeline = pipeline
        self.noise_floor = float(noise_floor)
        self.peak_hint = float(peak_hint)
        self._rms: deque[float] = deque(maxlen=HISTORY)
        self.last = LiveControlVector()

    @classmethod
    def try_load(cls, world: Path | str) -> "LiveVectorDriver":
        path = model_path(Path(world))
        if not path.is_file():
            print("LiveVectorDriver: no model — energy heuristic fallback")
            return cls()
        try:
            import joblib

            payload = joblib.load(path)
        except Exception as exc:
            print(f"LiveVectorDriver: load failed ({exc}) — heuristic")
            return cls()
        pipe = payload.get("pipeline") if isinstance(payload, dict) else payload
        noise = (
            float(payload.get("noise_floor", 0.0))
            if isinstance(payload, dict)
            else 0.0
        )
        peak = (
            float(payload.get("peak_hint", 0.0)) if isinstance(payload, dict) else 0.0
        )
        if pipe is None:
            return cls()
        print(f"LiveVectorDriver: loaded {path}")
        return cls(pipeline=pipe, noise_floor=noise, peak_hint=peak)

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

    def predict_raw(self) -> LiveControlVector:
        hist: Sequence[float] = list(self._rms) if self._rms else [0.0]
        heur_o, heur_j, heur_w = heuristic_vector(
            hist, noise_floor=self.noise_floor, peak_hint=self.peak_hint
        )
        if self._pipeline is not None:
            try:
                import numpy as np

                feats = rms_history_features(
                    hist,
                    noise_floor=self.noise_floor,
                    peak_hint=self.peak_hint,
                )
                current = float(feats[0])
                peak = max(float(self.peak_hint), float(feats[6]), 1e-6)
                # Outside the training energy range the regressor collapses to 0 —
                # fall back to energy heuristic instead of sealing the mouth.
                if current > peak * 1.15:
                    return LiveControlVector(
                        openness_n=heur_o,
                        jaw_n=heur_j,
                        width_n=heur_w,
                        plate_gate=plate_gate(heur_o),
                        source="heuristic",
                    )
                raw = np.asarray(
                    self._pipeline.predict(feats.reshape(1, -1)), dtype=np.float64
                ).reshape(-1)
                open_n = float(np.clip(raw[0], 0.0, 1.0))
                jaw_n = float(np.clip(raw[1] if raw.size > 1 else open_n, 0.0, 1.0))
                width_n = float(
                    np.clip(raw[2] if raw.size > 2 else open_n * 0.35, 0.0, 1.0)
                )
                # Never let OOD/zero ML undercut a clear voiced heuristic.
                if heur_o > 0.2 and open_n < 0.15 * heur_o:
                    open_n = max(open_n, heur_o)
                    jaw_n = max(jaw_n, heur_j)
                    width_n = max(width_n, heur_w)
                return LiveControlVector(
                    openness_n=open_n,
                    jaw_n=jaw_n,
                    width_n=width_n,
                    plate_gate=plate_gate(open_n),
                    source="ml",
                )
            except Exception:
                pass
        return LiveControlVector(
            openness_n=heur_o,
            jaw_n=heur_j,
            width_n=heur_w,
            plate_gate=plate_gate(heur_o),
            source="heuristic",
        )

    def resolve(
        self,
        *,
        phoneme: str,
        phoneme_jaw: float,
    ) -> LiveControlVector:
        """Blend table jaw with model — never ownership-seal the mouth to zero.

        Open vowels: table is a floor (model may raise).
        Other sounds: max(table, blended model) so motion is not killed.
        """
        from aiface.speech import canonical_viseme

        key = canonical_viseme(phoneme or "REST")
        ml = self.predict_raw()
        table = max(0.0, min(1.0, float(phoneme_jaw)))

        if key in OPEN_VISEMES or table >= 0.55:
            open_n = max(table, ml.openness_n)
            jaw_n = max(table, ml.jaw_n, ml.openness_n)
            width_n = max(0.15, ml.width_n)
            source = "table" if table >= ml.openness_n else ml.source
        else:
            # Follow speech table, let audio model add life — do not force 0.
            open_n = max(table, 0.35 * table + 0.65 * ml.openness_n)
            jaw_n = max(table, 0.35 * table + 0.65 * ml.jaw_n)
            width_n = max(table * 0.2, ml.width_n)
            source = ml.source if ml.openness_n > table else "table"

        self.last = LiveControlVector(
            openness_n=max(0.0, min(1.0, open_n)),
            jaw_n=max(0.0, min(1.0, jaw_n)),
            width_n=max(0.0, min(1.0, width_n)),
            plate_gate=plate_gate(open_n),
            source=source,
        )
        return self.last
