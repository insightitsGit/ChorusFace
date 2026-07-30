"""Deterministic biomechanical face simulation.

Nothing here touches the GPU or the screen. Speech, emotion, blinks, breathing,
idle behaviour, and AI intent all inject :class:`MuscleImpulse` events into one
queue; a spring-damper solver integrates them into continuous activations, and
the renderer only visualises the result. Every system is seeded, so the same
inputs always produce the same face.

Character lives in ``data/face_definition.json`` — swap that file to swap the
performer without touching code.
"""

from aiface.biomechanics.breathing import BreathingSystem
from aiface.biomechanics.emotion import EMOTION_AXES, EmotionState, EmotionSystem
from aiface.biomechanics.eyes import EyeState, EyeSystem
from aiface.biomechanics.face import BiomechanicalFace, FaceRenderState
from aiface.biomechanics.idle import IdleSystem
from aiface.biomechanics.intent import PHONEME_JAW_TARGET, IntentSystem
from aiface.biomechanics.jaw import JawState, JawSystem
from aiface.biomechanics.muscles import (
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
