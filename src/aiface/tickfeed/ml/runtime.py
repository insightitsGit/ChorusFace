"""Load and run L1–L5 TickFeed layers at inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aiface.tickfeed.audio_feat import load_audio_feat
from aiface.tickfeed.ml.packets import LookDrive, SpeechClock
from aiface.tickfeed.ml.train import ML_DIR, _audio_proxy
from aiface.tickfeed.schema import VISEME_TABLE


@dataclass
class TickFeedMLStack:
    root: Path
    l1: Any = None
    l2: Any = None
    l3: Any = None
    l4: Any = None
    l5: Any = None
    meta: dict | None = None
    audio_feats: NDArray[np.float32] | None = None
    audio_feat_source: str = "proxy_fallback"

    @classmethod
    def try_load(cls, world: Path | str) -> TickFeedMLStack | None:
        import joblib

        root = Path(world)
        root = root if root.is_dir() else root.parent
        ml = root / ML_DIR
        if not (ml / "l3_face_motion.joblib").is_file():
            return None
        stack = cls(root=ml)
        stack.l1 = joblib.load(ml / "l1_speech_clock.joblib")
        stack.l2 = joblib.load(ml / "l2_look_drive.joblib")
        stack.l3 = joblib.load(ml / "l3_face_motion.joblib")
        stack.l4 = joblib.load(ml / "l4_tick_codec.joblib")
        if (ml / "l5_gap_prior.joblib").is_file():
            stack.l5 = joblib.load(ml / "l5_gap_prior.joblib")
        meta_path = ml / "tickfeed_ml.meta.json"
        if meta_path.is_file():
            import json

            stack.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            stack.audio_feat_source = str(
                (stack.meta or {}).get("audio_feat_source") or "proxy_fallback"
            )
        loaded = load_audio_feat(root)
        if loaded is not None:
            stack.audio_feats, src = loaded
            if src == "wav_rms":
                stack.audio_feat_source = "wav_rms"
        print(
            f"TickFeed ML: loaded L1–L5 from {ml} "
            f"(audio_feat={stack.audio_feat_source})"
        )
        return stack

    def _audio_feat(
        self,
        open_amt: float,
        smile_amt: float,
        t: float,
        *,
        tick: int | None = None,
    ) -> NDArray[np.float64]:
        if (
            self.audio_feats is not None
            and self.audio_feat_source == "wav_rms"
            and tick is not None
            and 0 <= int(tick) < len(self.audio_feats)
        ):
            return np.asarray(self.audio_feats[int(tick)], dtype=np.float64)
        # Live / off-timeline: no calibration WAV row — use drive proxy honestly.
        return _audio_proxy(open_amt, smile_amt, t)

    def resolve(
        self,
        *,
        tick: int,
        open_amt: float,
        smile_amt: float,
        surprise_amt: float = 0.0,
        time_seconds: float | None = None,
        use_gap_prior: bool = False,
    ) -> tuple[SpeechClock, LookDrive, NDArray[np.float32], list[float]]:
        """Return packets + decoded face velocity flat + compact code."""
        t = float(time_seconds if time_seconds is not None else tick / 60.0)
        audio = self._audio_feat(
            open_amt, smile_amt, t, tick=tick
        ).reshape(1, -1)
        viseme_id = int(self.l1.predict(audio)[0])
        speech = SpeechClock(
            tick=tick,
            viseme_id=viseme_id,
            word=VISEME_TABLE[viseme_id] if 0 <= viseme_id < len(VISEME_TABLE) else "REST",
            conf=0.85 if self.audio_feat_source == "wav_rms" else 0.55,
            audio_feat=audio.reshape(-1).tolist(),
        )
        x_l2 = np.concatenate(
            [audio, np.asarray([[viseme_id]], dtype=np.float64)], axis=1
        )
        look_v = np.asarray(self.l2.predict(x_l2)[0], dtype=np.float64)
        # Blend network look with live amounts (live wins when strong)
        look_v[0] = max(float(look_v[0]), float(smile_amt))
        look_v[1] = max(float(look_v[1]), float(open_amt))
        look_v[2] = max(float(look_v[2]), float(surprise_amt))
        look = LookDrive(
            tick=tick,
            smile=float(np.clip(look_v[0], 0, 1)),
            open=float(np.clip(look_v[1], 0, 1)),
            surprise=float(np.clip(look_v[2], 0, 1)),
            brow=float(np.clip(look_v[3], 0, 1)),
            conf=0.8,
        )
        x_l3 = np.concatenate([x_l2, look_v.reshape(1, -1)], axis=1)
        code = np.asarray(self.l3.predict(x_l3)[0], dtype=np.float64)
        # L5 recovers full code from an incomplete/holey code estimate.
        if use_gap_prior and self.l5 is not None:
            try:
                code = np.asarray(self.l5.predict(code.reshape(1, -1))[0], dtype=np.float64)
            except Exception:  # noqa: BLE001
                pass
        pca = self.l4["pca"]
        n = int(pca.n_components_)
        if code.shape[0] < n:
            code = np.pad(code, (0, n - code.shape[0]))
        code = code[:n]
        flat = pca.inverse_transform(code.reshape(1, -1))[0].astype(np.float32)
        return speech, look, flat, code.astype(np.float32).tolist()

    def encode_patch(self, patch: NDArray[np.floating]) -> list[float]:
        """L4 encode: face velocity patch → compact c_t."""
        if self.l4 is None:
            raise RuntimeError("L4 codec not loaded")
        pca = self.l4["pca"]
        flat = np.asarray(patch, dtype=np.float32).reshape(1, -1)
        code = pca.transform(flat)[0]
        return code.astype(np.float32).tolist()

    def decode_code(self, code: list[float] | NDArray[np.floating]) -> NDArray[np.float32]:
        """L4 decode: compact c_t → flat face velocity."""
        if self.l4 is None:
            raise RuntimeError("L4 codec not loaded")
        pca = self.l4["pca"]
        n = int(pca.n_components_)
        vec = np.asarray(code, dtype=np.float64).reshape(-1)
        if vec.size < n:
            vec = np.pad(vec, (0, n - vec.size))
        return pca.inverse_transform(vec[:n].reshape(1, -1))[0].astype(np.float32)


__all__ = ["TickFeedMLStack"]
