"""Deterministic biomechanical face simulation.

Nothing here touches the GPU or the screen. Speech, emotion, blinks, breathing,
idle behaviour, and AI intent all inject :class:`MuscleImpulse` events into one
queue; a spring-damper solver integrates them into continuous activations, and
the renderer only visualises the result. Every system is seeded, so the same
inputs always produce the same face.

Character lives in ``data/face_definition.json`` — swap that file to swap the
performer without touching code.
"""

from chorusface.biomechanics.breathing import BreathingSystem
from chorusface.biomechanics.emotion import EMOTION_AXES, EmotionState, EmotionSystem
from chorusface.biomechanics.eyes import EyeState, EyeSystem
from chorusface.biomechanics.face import BiomechanicalFace, FaceRenderState
from chorusface.biomechanics.idle import IdleSystem
from chorusface.biomechanics.intent import PHONEME_JAW_TARGET, IntentSystem
from chorusface.biomechanics.jaw import JawState, JawSystem
from chorusface.biomechanics.muscles import (
    DEFAULT_FACE_DEFINITION,
    FieldImpulseSpec,
    Muscle,
    MuscleImpulse,
    MuscleImpulseQueue,
    MuscleRegistry,
    MuscleSolver,
    MuscleState,
    load_face_definition,
)

__all__ = [
    "DEFAULT_FACE_DEFINITION",
    "EMOTION_AXES",
    "PHONEME_JAW_TARGET",
    "BiomechanicalFace",
    "BreathingSystem",
    "EmotionState",
    "EmotionSystem",
    "EyeState",
    "EyeSystem",
    "FaceRenderState",
    "FieldImpulseSpec",
    "IdleSystem",
    "IntentSystem",
    "JawState",
    "JawSystem",
    "Muscle",
    "MuscleImpulse",
    "MuscleImpulseQueue",
    "MuscleRegistry",
    "MuscleSolver",
    "MuscleState",
    "load_face_definition",
]
