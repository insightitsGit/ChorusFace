"""Biomechanical face orchestrator.

Speech, emotion, blink, breathing, idle, and AI intent all inject MuscleImpulse
events into one queue. The solver is the single source of truth; the renderer
only visualises the resulting continuous state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from aiface.biomechanics.breathing import BreathingSystem
from aiface.biomechanics.emotion import EmotionSystem
from aiface.biomechanics.eyes import EyeSystem
from aiface.biomechanics.idle import IdleSystem
from aiface.biomechanics.intent import PHONEME_JAW_TARGET, IntentSystem
from aiface.biomechanics.jaw import JawSystem
from aiface.biomechanics.muscles import (
    DEFAULT_FACE_DEFINITION,
    FieldImpulseSpec,
    MuscleImpulseQueue,
    MuscleRegistry,
    MuscleSolver,
    load_face_definition,
)


@dataclass(slots=True)
class FaceRenderState:
    """Everything the avatar renderer needs for one frame."""

    mouth_width: float = 14.0
    mouth_openness: float = 1.0
    mouth_roundness: float = 0.12
    expression: float = 0.0
    jaw_angle: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    pupil: float = 0.45
    lid_left: float = 1.0
    lid_right: float = 1.0
    brow_raise: float = 0.0
    brow_knit: float = 0.0
    eye_widen: float = 0.0
    breath_phase: float = 0.0
    # Per-muscle activation drives the renderer's displacement field; the group
    # means are what the rest of the system reasons about.
    muscle_activations: dict[str, float] = field(default_factory=dict)
    group_activations: dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = "valence"
    dominant_emotion_value: float = 0.0
    active_impulse_count: int = 0
    blink_timer: float = 0.0
    simulation_time: float = 0.0
    last_phoneme: str = "REST"
    speaking: bool = False


@dataclass(slots=True)
class BiomechanicalFace:
    """Modular biomechanical face. Character changes via face_definition.json."""

    definition: dict[str, Any]
    registry: MuscleRegistry
    solver: MuscleSolver
    impulses: MuscleImpulseQueue
    emotion: EmotionSystem
    eyes: EyeSystem
    jaw: JawSystem
    breathing: BreathingSystem
    idle: IdleSystem
    intent: IntentSystem
    simulation_time: float = 0.0
    last_phoneme: str = "REST"
    speaking_until: float = 0.0
    # Baked by aiface-capture from the talk segment (1.0 = default travel).
    speech_travel_scale: float = 1.0
    lip_width_travel_scale: float = 1.0
    _seed: int = 1

    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        *,
        seed: int = 1,
    ) -> "BiomechanicalFace":
        definition = load_face_definition(path or DEFAULT_FACE_DEFINITION)
        registry = MuscleRegistry.from_definition(definition)
        solver = MuscleSolver(registry)
        constraints = definition.get("constraints", {})
        jaw = JawSystem(
            max_opening=float(constraints.get("max_jaw_open_radians", 0.55)),
        )
        eyes = EyeSystem(seed=seed + 3)
        return cls(
            definition=definition,
            registry=registry,
            solver=solver,
            impulses=MuscleImpulseQueue(),
            emotion=EmotionSystem(),
            eyes=eyes,
            jaw=jaw,
            breathing=BreathingSystem(),
            idle=IdleSystem(seed=seed + 17, eyes=eyes),
            intent=IntentSystem(
                phoneme_muscles=definition.get("phoneme_muscles", {}),
            ),
            _seed=seed,
        )

    def reset(self) -> None:
        self.solver.reset()
        self.impulses.clear()
        self.simulation_time = 0.0
        self.last_phoneme = "REST"
        self.speaking_until = 0.0
        assert self.jaw.state is not None
        self.jaw.state.angle = 0.0
        self.jaw.state.velocity = 0.0
        self.jaw.state.angular_momentum = 0.0
        self.jaw.state.target = 0.0

    def apply_capture_priors(
        self,
        *,
        jaw_travel_scale: float = 1.0,
        lip_width_scale: float = 1.0,
        lip_open_scale: float = 1.0,
    ) -> None:
        """Scale jaw/lip travel from talk-segment landmark curves."""
        # Allow stronger travel when capture peaks were tiny (weak open takes).
        jaw = max(0.85, min(2.0, float(jaw_travel_scale)))
        width = max(0.85, min(1.8, float(lip_width_scale)))
        open_s = max(0.85, min(2.0, float(lip_open_scale)))
        self.jaw.max_opening = max(0.35, min(1.05, self.jaw.max_opening * jaw))
        self.speech_travel_scale = max(0.85, min(2.0, 0.5 * (open_s + width)))
        self.lip_width_travel_scale = width

    def submit_phoneme(
        self,
        phoneme: str,
        *,
        tick: int,
        emotion_label: str | None = None,
        duration: float = 0.1,
    ) -> None:
        from aiface.speech import canonical_viseme, mouth_pose

        key = canonical_viseme(phoneme)
        mood = (emotion_label or "NEUTRAL").upper()
        self.last_phoneme = key
        # Caller / UI preset sets the readable dwell; keep a small floor.
        hold = max(float(duration), 0.08)
        self.speaking_until = self.simulation_time + hold
        if emotion_label:
            self.emotion.from_label(emotion_label)
        scale = self.intent.articulation_scale(key, mood) * self.speech_travel_scale
        self.impulses.push_many(
            self.intent.speech_impulses(
                key, tick=tick, duration=hold, strength_scale=scale
            )
        )
        pose = mouth_pose(key, mood)
        jaw_base = PHONEME_JAW_TARGET.get(key, 0.1)
        jaw_open = min(1.0, jaw_base * (0.65 + 0.55 * min(pose.openness / 14.0, 1.0)))
        self.jaw.set_speech_target(jaw_open)

    def submit_intent(self, payload: Mapping[str, Any], *, tick: int) -> None:
        self.impulses.push_many(
            self.intent.apply(
                payload,
                emotion=self.emotion,
                jaw=self.jaw,
                tick=tick,
            )
        )
        speech = payload.get("speech")
        if isinstance(speech, Mapping):
            phonemes = speech.get("phonemes") or ()
            if phonemes:
                self.last_phoneme = str(phonemes[0]).upper()
                self.speaking_until = self.simulation_time + 0.2

    def step(self, dt: float, *, tick: int) -> tuple[FaceRenderState, list[FieldImpulseSpec]]:
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        self.simulation_time += dt
        speaking = self.simulation_time < self.speaking_until

        # Continuous layers inject into the same impulse queue.
        self.emotion.step(dt)
        self.impulses.push_many(self.emotion.impulses(tick))
        self.impulses.push_many(self.breathing.step(dt))
        self.impulses.push_many(self.idle.step(dt, speaking=speaking))
        self.impulses.push_many(
            self.eyes.step(dt, arousal=self.emotion.state.arousal)
        )

        drives = self.impulses.step(dt)
        self.solver.set_drives(drives)
        activations = self.solver.step(dt)
        groups = self.solver.group_activations()
        jaw_angle = self.jaw.step(dt)

        rest = self.definition.get("rest_pose", {})
        width = float(rest.get("mouth_width", 14.0))
        openness = float(rest.get("mouth_openness", 1.0))
        roundness = float(rest.get("mouth_roundness", 0.12))
        expression = float(rest.get("expression", 0.0))

        oris = groups.get("OrbicularisOris", 0.0)
        risorius = groups.get("Risorius", 0.0)
        zygo = groups.get("ZygomaticusMajor", 0.0)
        buccinator = groups.get("Buccinator", 0.0)
        frontalis = groups.get("Frontalis", 0.0)
        corrugator = groups.get("Corrugator", 0.0)
        depressor = groups.get("DepressorAnguliOris", 0.0)
        levator_anguli = groups.get("LevatorAnguliOris", 0.0)
        levator_palpebrae = groups.get("LevatorPalpebrae", 0.0)

        # Geometry is derived from muscle + jaw state, never from animation clips.
        openness = self.jaw.openness_units(
            rest=float(rest.get("mouth_openness", 1.0)),
            scale=18.0,
        )
        # Oris adds a little vertical cue; jaw owns true openness.
        openness += oris * 1.4 * self.speech_travel_scale
        depressor_labii = groups.get("DepressorLabii", 0.0)
        openness += depressor_labii * 3.2 * self.speech_travel_scale
        width += (
            (risorius * 6.0 + zygo * 1.2 - buccinator * 4.5)
            * self.lip_width_travel_scale
        )
        roundness = max(0.0, min(1.0, roundness + buccinator * 0.85 - risorius * 0.40))
        # Keep smile expression muscle-light during speech; HAPPY owns smile.
        expression += (
            zygo * 0.35
            + levator_anguli * 0.20
            - depressor * 0.80
            - corrugator * 0.45
            + self.emotion.state.valence * 0.20
        )
        expression = max(-1.0, min(1.0, expression))
        width = max(7.0, min(22.0, width))
        openness = max(0.7, min(14.0, openness))

        dominant_name, dominant_value = self.emotion.state.dominant()
        eye_widen = max(
            0.0,
            min(
                1.0,
                self.emotion.state.surprise * 0.95
                + self.emotion.state.curiosity * 0.35
                + self.emotion.state.arousal * 0.20
                + levator_palpebrae * 0.45
                + frontalis * 0.25,
            ),
        )
        brow_raise = max(
            0.0,
            min(
                1.0,
                frontalis * 1.15
                + self.emotion.state.surprise * 0.55
                + self.emotion.state.curiosity * 0.35
                + self.emotion.state.arousal * 0.22,
            ),
        )
        render = FaceRenderState(
            mouth_width=width,
            mouth_openness=openness,
            mouth_roundness=roundness,
            expression=expression,
            jaw_angle=jaw_angle,
            gaze_x=self.eyes.state.gaze_x,
            gaze_y=self.eyes.state.gaze_y,
            pupil=self.eyes.state.pupil,
            lid_left=self.eyes.state.lid_left,
            lid_right=self.eyes.state.lid_right,
            brow_raise=brow_raise,
            brow_knit=corrugator,
            eye_widen=eye_widen,
            breath_phase=self.breathing.phase,
            muscle_activations=dict(activations),
            group_activations=groups,
            dominant_emotion=dominant_name,
            dominant_emotion_value=dominant_value,
            active_impulse_count=self.impulses.count,
            blink_timer=self.eyes.state.blink_timer,
            simulation_time=self.simulation_time,
            last_phoneme=self.last_phoneme,
            speaking=speaking,
        )

        constraints = self.definition.get("constraints", {})
        field_specs = self.solver.field_impulse_specs(
            activations,
            source="Speech" if speaking else "Idle",
            radius=float(constraints.get("mouth_write_radius", 14.0)),
            budget=int(constraints.get("max_field_impulses_per_tick", 12)),
        )
        # Default: keep muscle anchors so lip/jaw writers address distinct
        # cells. Legacy single-disc remap is opt-in only — it destroyed
        # per-cell / neighbor control of the mouth cluster.
        if not bool(constraints.get("remap_field_to_mouth_center", False)):
            return render, field_specs
        mouth = self.definition.get("mouth_center", [0.5, 0.78])
        mouth_writers = {"OrbicularisOris", "JawOpener"}
        remapped: list[FieldImpulseSpec] = []
        for spec in field_specs:
            if self.registry.get(spec.muscle).group in mouth_writers:
                remapped.append(
                    FieldImpulseSpec(
                        muscle=spec.muscle,
                        center_uv=(float(mouth[0]), float(mouth[1])),
                        velocity=spec.velocity,
                        radius=spec.radius,
                        priority=spec.priority,
                        source=spec.source,
                    )
                )
            else:
                remapped.append(spec)
        return render, remapped


__all__ = ["BiomechanicalFace", "FaceRenderState"]
