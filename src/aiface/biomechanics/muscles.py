"""Facial muscle registry and deterministic impulse blending.

Muscles never paint pixels. They emit forces that the biomechanical simulator
integrates into continuous activation state; the avatar driver converts that
state into GPU field impulses (unlocked mouth cells only) and into the
continuous displacement field the fragment shader warps the portrait with.

Two coordinate conventions meet here, and they disagree about which way is up:

* ``anchor`` is face-box UV with ``v`` increasing **downward**, because it is
  authored against the portrait and matched against image-space region masks.
* ``force`` is a direction in **grid** space with ``y`` increasing upward,
  because it feeds velocity impulses and the renderer, both of which run on the
  y-up world grid.

:func:`aiface.skinning.pack_muscle_uniforms` is the single place that converts
between them.

Individual muscles are grouped so the rest of the system can keep addressing
anatomy at the level it cares about. Speech asks for ``OrbicularisOris``; the
registry fans that out to the four quadrants that actually contract.
"""

from __future__ import annotations

import json
from importlib.resources import files
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

# Packaged so an installed wheel ships the default character.
DEFAULT_FACE_DEFINITION: Final = Path(
    str(files("aiface.biomechanics").joinpath("data", "face_definition.json"))
)

ImpulseSource = str  # Speech | Emotion | Blink | Breathing | Idle | User | AI

# Which side of the lip parting a muscle is allowed to move. The mouth is the
# one place on the face where tissue genuinely separates, so gating there is
# anatomy rather than a rendering shortcut.
GATE_NONE: Final = "none"
GATE_UPPER_LIP: Final = "upper_lip"
GATE_LOWER_LIP: Final = "lower_lip"
GATE_CODES: Final[dict[str, float]] = {
    GATE_NONE: 0.0,
    GATE_UPPER_LIP: 1.0,
    GATE_LOWER_LIP: -1.0,
}
SIDES: Final = frozenset({"left", "right", "center"})


@dataclass(frozen=True, slots=True)
class Muscle:
    """One anatomically named actuator with deterministic dynamics."""

    name: str
    anchor: tuple[float, float]  # face-box UV (u right, v down)
    influence_radius: float  # grid cells; also the render falloff support
    stiffness: float
    damping: float
    max_contraction: float
    neighbors: tuple[str, ...]
    priority: int
    force: tuple[float, float]  # grid-space direction, y up
    group: str
    side: str = "center"
    gate: str = GATE_NONE
    # Static left/right drive imbalance. Real faces are never symmetric, and
    # perfect symmetry is one of the loudest "this is synthetic" cues.
    bias: float = 1.0
    # Peak skin displacement in grid cells at full activation. Kept separate
    # from ``force`` so render amplitude can be tuned without changing the
    # velocity impulses that drive the field.
    travel: float = 0.0
    writes_field: bool = False

    @property
    def gate_code(self) -> float:
        return GATE_CODES[self.gate]


@dataclass(frozen=True, slots=True)
class MuscleImpulse:
    """A time-bounded drive injected into one muscle."""

    tick: int
    muscle: str
    strength: float
    duration: float
    falloff: float
    priority: int
    source: ImpulseSource


@dataclass(slots=True)
class MuscleState:
    """Continuous per-muscle state integrated every simulation step."""

    activation: float = 0.0
    velocity: float = 0.0
    drive: float = 0.0


@dataclass(slots=True)
class FieldImpulseSpec:
    """GPU velocity impulse request for unlocked mouth tissue only."""

    muscle: str
    center_uv: tuple[float, float]
    velocity: tuple[float, float]
    radius: float
    priority: int
    source: ImpulseSource


class MuscleRegistry:
    """Loads and owns the character muscle graph.

    Muscles and groups share one namespace so callers can name either. A group
    is just the set of muscles that contract together — ``Frontalis`` resolves
    to its left and right bellies, ``MasseterLeft`` resolves to itself.
    """

    def __init__(self, muscles: Sequence[Muscle]) -> None:
        if not muscles:
            raise ValueError("Muscle registry requires at least one muscle")
        self._muscles: dict[str, Muscle] = {muscle.name: muscle for muscle in muscles}
        if len(self._muscles) != len(muscles):
            raise ValueError("Muscle names must be unique")

        groups: dict[str, list[str]] = {}
        for muscle in muscles:
            groups.setdefault(muscle.group, []).append(muscle.name)
        collisions = {
            name for name in groups if name in self._muscles and groups[name] != [name]
        }
        if collisions:
            raise ValueError(
                f"Group names collide with muscle names: {sorted(collisions)}"
            )
        self._groups: dict[str, tuple[str, ...]] = {
            name: tuple(members) for name, members in groups.items()
        }

        unknown = {
            neighbor
            for muscle in muscles
            for neighbor in muscle.neighbors
            if neighbor not in self._muscles and neighbor not in self._groups
        }
        if unknown:
            raise ValueError(f"Unknown neighbor muscles: {sorted(unknown)}")
        # Authoring names neighbours at group level; coupling runs per muscle.
        self._neighbors: dict[str, tuple[str, ...]] = {
            muscle.name: tuple(
                dict.fromkeys(
                    member
                    for neighbor in muscle.neighbors
                    for member in self.resolve(neighbor)
                    if member != muscle.name
                )
            )
            for muscle in muscles
        }

    @classmethod
    def from_definition(cls, payload: Mapping[str, Any]) -> "MuscleRegistry":
        muscles: list[Muscle] = []
        for entry in payload.get("muscles", []):
            anchor = entry["anchor"]
            force = entry.get("force", [0.0, 0.0])
            name = str(entry["name"])
            side = str(entry.get("side", "center"))
            if side not in SIDES:
                raise ValueError(f"{name}: side must be one of {sorted(SIDES)}")
            gate = str(entry.get("gate", GATE_NONE))
            if gate not in GATE_CODES:
                raise ValueError(f"{name}: gate must be one of {sorted(GATE_CODES)}")
            muscles.append(
                Muscle(
                    name=name,
                    anchor=(float(anchor[0]), float(anchor[1])),
                    influence_radius=float(entry["influence_radius"]),
                    stiffness=float(entry["stiffness"]),
                    damping=float(entry["damping"]),
                    max_contraction=float(entry["max_contraction"]),
                    neighbors=tuple(str(item) for item in entry.get("neighbors", [])),
                    priority=int(entry.get("priority", 1)),
                    force=(float(force[0]), float(force[1])),
                    group=str(entry.get("group", name)),
                    side=side,
                    gate=gate,
                    bias=float(entry.get("bias", 1.0)),
                    travel=float(entry.get("travel", 0.0)),
                    writes_field=bool(entry.get("writes_field", False)),
                )
            )
        return cls(muscles)

    def get(self, name: str) -> Muscle:
        return self._muscles[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._muscles)

    def group_names(self) -> tuple[str, ...]:
        return tuple(self._groups)

    def resolve(self, name: str) -> tuple[str, ...]:
        """Return the concrete muscles a muscle or group name addresses."""
        if name in self._muscles:
            return (name,)
        return self._groups.get(name, ())

    def neighbors_of(self, name: str) -> tuple[str, ...]:
        return self._neighbors.get(name, ())

    def expand_drives(self, drives: Mapping[str, float]) -> dict[str, float]:
        """Fan group-level drives out to member muscles, strongest wins.

        A group drive picks up each member's ``bias``, which is what keeps the
        two halves of the face from contracting in lockstep.
        """
        resolved: dict[str, float] = {}
        for name, value in drives.items():
            members = self.resolve(name)
            if not members:
                continue
            grouped = name not in self._muscles
            for member in members:
                scale = self._muscles[member].bias if grouped else 1.0
                candidate = float(value) * scale
                previous = resolved.get(member)
                if previous is None or abs(candidate) > abs(previous):
                    resolved[member] = candidate
        return resolved

    def __iter__(self) -> Iterable[Muscle]:
        return iter(self._muscles.values())

    def __len__(self) -> int:
        return len(self._muscles)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (
            name in self._muscles or name in self._groups
        )


class MuscleImpulseQueue:
    """Deterministic blend of overlapping muscle impulses."""

    __slots__ = ("_active",)

    def __init__(self) -> None:
        self._active: list[tuple[MuscleImpulse, float]] = []

    def clear(self) -> None:
        self._active.clear()

    def push(self, impulse: MuscleImpulse) -> None:
        if impulse.duration <= 0.0:
            return
        self._active.append((impulse, 0.0))

    def push_many(self, impulses: Sequence[MuscleImpulse]) -> None:
        for impulse in impulses:
            self.push(impulse)

    @property
    def count(self) -> int:
        return len(self._active)

    def step(self, dt: float) -> dict[str, float]:
        """Advance time and return blended drive per muscle name."""
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        drives: dict[str, float] = {}
        weights: dict[str, float] = {}
        surviving: list[tuple[MuscleImpulse, float]] = []
        for impulse, age in self._active:
            age += dt
            if age > impulse.duration:
                continue
            surviving.append((impulse, age))
            progress = age / max(impulse.duration, 1e-6)
            envelope = max(0.0, 1.0 - progress) ** max(impulse.falloff, 0.0)
            weight = (1.0 + float(impulse.priority)) * envelope
            drives[impulse.muscle] = drives.get(impulse.muscle, 0.0) + (
                impulse.strength * weight
            )
            weights[impulse.muscle] = weights.get(impulse.muscle, 0.0) + weight
        self._active = surviving
        return {
            name: value / max(weights[name], 1e-6)
            for name, value in drives.items()
        }


class MuscleSolver:
    """Second-order spring-damper activation dynamics for every muscle."""

    def __init__(self, registry: MuscleRegistry) -> None:
        self.registry = registry
        self.state: dict[str, MuscleState] = {
            muscle.name: MuscleState() for muscle in registry
        }

    def reset(self) -> None:
        for state in self.state.values():
            state.activation = 0.0
            state.velocity = 0.0
            state.drive = 0.0

    def set_drives(self, drives: Mapping[str, float]) -> None:
        resolved = self.registry.expand_drives(drives)
        for name, state in self.state.items():
            state.drive = float(resolved.get(name, 0.0))

    def step(self, dt: float) -> dict[str, float]:
        if dt <= 0.0:
            return {name: state.activation for name, state in self.state.items()}
        activations: dict[str, float] = {}
        for muscle in self.registry:
            state = self.state[muscle.name]
            # Neighbor coupling: stiff neighbors damp wild unilateral motion.
            neighbors = self.registry.neighbors_of(muscle.name)
            neighbor_mean = 0.0
            if neighbors:
                neighbor_mean = sum(
                    self.state[name].activation for name in neighbors
                ) / len(neighbors)
            target = state.drive * 0.92 + neighbor_mean * 0.08
            target = max(0.0, min(muscle.max_contraction, target))
            # Critically-damped spring with an extra chase term so speech
            # impulses become visible within a few 60 Hz ticks.
            force = (target - state.activation) * muscle.stiffness
            force -= state.velocity * muscle.damping
            force += (target - state.activation) * 28.0
            state.velocity += force * dt
            state.activation = max(
                0.0,
                min(muscle.max_contraction, state.activation + state.velocity * dt),
            )
            activations[muscle.name] = state.activation
        return activations

    def group_activations(self) -> dict[str, float]:
        """Mean activation per group, for callers that reason about anatomy."""
        totals: dict[str, float] = {}
        for name in self.registry.group_names():
            members = self.registry.resolve(name)
            totals[name] = sum(
                self.state[member].activation for member in members
            ) / len(members)
        return totals

    def active_muscles(self, *, threshold: float = 0.05) -> list[tuple[str, float]]:
        ranked = [
            (name, state.activation)
            for name, state in self.state.items()
            if state.activation >= threshold
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def field_impulse_specs(
        self,
        activations: Mapping[str, float],
        *,
        source: ImpulseSource = "Speech",
        scale: float = 0.45,
        radius: float = 14.0,
        budget: int = 12,
    ) -> list[FieldImpulseSpec]:
        """Convert activated field-writing muscles into GPU impulse requests."""
        specs: list[FieldImpulseSpec] = []
        ranked = sorted(
            (
                (muscle, float(activations.get(muscle.name, 0.0)))
                for muscle in self.registry
                if muscle.writes_field
            ),
            key=lambda item: (-item[1], -item[0].priority, item[0].name),
        )
        for muscle, activation in ranked:
            if activation < 0.04 or len(specs) >= budget:
                continue
            specs.append(
                FieldImpulseSpec(
                    muscle=muscle.name,
                    center_uv=muscle.anchor,
                    velocity=(
                        muscle.force[0] * activation * scale,
                        muscle.force[1] * activation * scale,
                    ),
                    radius=min(radius, muscle.influence_radius),
                    priority=muscle.priority,
                    source=source,
                )
            )
        return specs


def load_face_definition(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else DEFAULT_FACE_DEFINITION
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("face definition must be a JSON object")
    return payload


__all__ = [
    "DEFAULT_FACE_DEFINITION",
    "GATE_CODES",
    "GATE_LOWER_LIP",
    "GATE_NONE",
    "GATE_UPPER_LIP",
    "SIDES",
    "FieldImpulseSpec",
    "ImpulseSource",
    "Muscle",
    "MuscleImpulse",
    "MuscleImpulseQueue",
    "MuscleRegistry",
    "MuscleSolver",
    "MuscleState",
    "load_face_definition",
]
