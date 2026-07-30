"""Semantic intent bridge: any LLM JSON → muscle impulses + subsystem targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from aiface.biomechanics.emotion import EmotionSystem
from aiface.biomechanics.jaw import JawSystem
from aiface.biomechanics.muscles import MuscleImpulse


# Phoneme → desired jaw openness in [0, 1] (Oculus/MPEG-4 inventory).
# Spaced so closed / fricative / mid / open read as distinct silhouettes.
PHONEME_JAW_TARGET: dict[str, float] = {
    "REST": 0.0,
    "CLOSED": 0.0,
    "PP": 0.0,
    "FF": 0.12,
    "TH": 0.22,
    "DD": 0.16,
    "KK": 0.12,
    "CH": 0.20,
    "SS": 0.14,
    "NN": 0.20,
    "RR": 0.22,
    "AH": 1.0,
    "AA": 0.98,
    "EH": 0.52,
    "IH": 0.30,
    "EE": 0.20,
    "OH": 0.74,
    "OU": 0.40,
    # Legacy aliases still accepted if a host bypasses canonical_viseme.
    "MM": 0.0,
    "FV": 0.12,
    "L": 0.22,
    "R": 0.22,
    "IY": 0.20,
    "OO": 0.40,
    "UW": 0.40,
}


@dataclass(slots=True)
class IntentSystem:
    """Converts semantic payloads into emotion patches and speech muscle drives."""

    phoneme_muscles: Mapping[str, Mapping[str, float]]

    def apply(
        self,
        payload: Mapping[str, Any],
        *,
        emotion: EmotionSystem,
        jaw: JawSystem,
        tick: int,
    ) -> list[MuscleImpulse]:
        impulses: list[MuscleImpulse] = []

        emotion_patch = payload.get("emotion")
        if isinstance(emotion_patch, Mapping):
            emotion.apply_patch(
                {str(key): float(value) for key, value in emotion_patch.items()},
                amount=1.0,
            )

        intent = payload.get("intent")
        if isinstance(intent, Mapping):
            thinking = float(intent.get("thinking", 0.0))
            emphasis = float(intent.get("emphasis", 0.0))
            if thinking:
                emotion.apply_patch({"thinking": thinking}, amount=0.8)
            if emphasis > 0.0:
                impulses.append(
                    MuscleImpulse(
                        tick=tick,
                        muscle="Frontalis",
                        strength=0.25 * emphasis,
                        duration=0.2,
                        falloff=1.1,
                        priority=2,
                        source="AI",
                    )
                )
                impulses.append(
                    MuscleImpulse(
                        tick=tick,
                        muscle="Masseter",
                        strength=0.2 * emphasis,
                        duration=0.18,
                        falloff=1.1,
                        priority=2,
                        source="AI",
                    )
                )

        speech = payload.get("speech")
        phonemes: Sequence[str] = ()
        if isinstance(speech, Mapping):
            raw = speech.get("phonemes", ())
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                phonemes = [str(item).upper() for item in raw]
        if phonemes:
            from aiface.speech import canonical_viseme

            key = canonical_viseme(phonemes[0])
            impulses.extend(self.speech_impulses(key, tick=tick))
            jaw.set_speech_target(PHONEME_JAW_TARGET.get(key, 0.1))

        return impulses

    def speech_impulses(
        self,
        phoneme: str,
        *,
        tick: int,
        duration: float = 0.1,
        source: str = "Speech",
        strength_scale: float = 1.0,
    ) -> list[MuscleImpulse]:
        from aiface.speech import canonical_viseme

        key = canonical_viseme(phoneme)
        mapping = self.phoneme_muscles.get(key, self.phoneme_muscles.get("REST", {}))
        scale = max(0.35, min(1.6, float(strength_scale)))
        hold = max(float(duration), 0.12)
        impulses: list[MuscleImpulse] = []
        for muscle, strength in mapping.items():
            impulses.append(
                MuscleImpulse(
                    tick=tick,
                    muscle=str(muscle),
                    strength=min(1.35, float(strength) * scale),
                    duration=hold,
                    falloff=1.05,
                    priority=4 if source == "Speech" else 2,
                    source=source,  # type: ignore[arg-type]
                )
            )
        return impulses

    def jaw_target_for_phoneme(self, phoneme: str) -> float:
        from aiface.speech import canonical_viseme

        return PHONEME_JAW_TARGET.get(canonical_viseme(phoneme), 0.1)

    def articulation_scale(self, phoneme: str, emotion: str = "NEUTRAL") -> float:
        """How hard to drive muscles for this viseme, from the MouthPose table.

        Geometry still comes from the solver; this only scales impulse strength
        so open vowels and rounded shapes actually reach the photo.
        """
        from aiface.speech import canonical_viseme, mouth_pose

        pose = mouth_pose(canonical_viseme(phoneme), emotion)
        open_n = min(pose.openness / 14.0, 1.0)
        round_n = float(pose.roundness)
        width_n = max(0.0, (pose.width - 14.0) / 8.0)
        return 0.75 + 0.55 * open_n + 0.20 * round_n + 0.10 * width_n


__all__ = ["IntentSystem", "PHONEME_JAW_TARGET"]
