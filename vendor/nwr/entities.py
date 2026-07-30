"""Named, tracked entities layered on top of the field substrate.

An entity is an addressable agent with an identity, a lifecycle, a position that
the field can carry, and a footprint of cells it owns. Entities never touch GPU
memory: the registry lowers every spawn, step, and removal into the same
validated :class:`~ai_commands.Segment` and :class:`~ai_commands.TemperatureDelta`
primitives the mouse produces, so authority and human locks still decide what
actually lands.

Identity is assigned from a monotonic counter and the registry holds no clocks
or randomness, so a replayed session produces the same entities with the same
identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ai_commands import (
    CATEGORY_IDS,
    DEFAULT_BOUNDS,
    CommandError,
    Operation,
    Segment,
    TemperatureDelta,
)
from bds_format import PRIORITY_LEVELS

MAX_ENTITIES: Final = 256
MIN_ENTITY_RADIUS: Final = 0.5
MAX_ENTITY_RADIUS: Final = 64.0
# Matches ADVECTION_SCALE in shaders/physics.comp so an entity rides the same
# flow field its cells do.
DRIFT_SCALE: Final = 0.5
DRIFT_RESTITUTION: Final = 0.85


@dataclass(frozen=True, slots=True)
class EntityKind:
    """Behaviour template shared by every entity of one kind."""

    name: str
    material: str | None
    radius: float
    drifts: bool
    restamps: bool
    clears_trail: bool
    temperature: float
    description: str


ENTITY_KINDS: Final[dict[str, EntityKind]] = {
    "emitter": EntityKind(
        name="emitter",
        material="active_fluid",
        radius=3.0,
        drifts=False,
        restamps=True,
        clears_trail=False,
        temperature=0.0,
        description="Anchored source that injects Active Fluid every step.",
    ),
    "blob": EntityKind(
        name="blob",
        material="active_fluid",
        radius=4.0,
        drifts=True,
        restamps=True,
        clears_trail=True,
        temperature=0.0,
        description=(
            "Fluid body carried by the velocity field; bounces off walls and "
            "clears the cells it leaves behind."
        ),
    ),
    "obstacle": EntityKind(
        name="obstacle",
        material="solid",
        radius=4.0,
        drifts=False,
        restamps=False,
        clears_trail=False,
        temperature=0.0,
        description="Static wall that deflects flow without a human lock.",
    ),
    "heater": EntityKind(
        name="heater",
        material=None,
        radius=5.0,
        drifts=False,
        restamps=False,
        clears_trail=False,
        temperature=0.05,
        description="Field effect that raises temperature; writes no material.",
    ),
    "chiller": EntityKind(
        name="chiller",
        material=None,
        radius=5.0,
        drifts=False,
        restamps=False,
        clears_trail=False,
        temperature=-0.05,
        description="Field effect that lowers temperature; writes no material.",
    ),
}


class EntityError(CommandError):
    """Raised when an entity request is malformed or the registry is full."""


@dataclass(frozen=True, slots=True)
class Entity:
    """One live entity record."""

    entity_id: str
    kind: str
    x: float
    y: float
    radius: float
    priority: int
    spawn_tick: int

    @property
    def template(self) -> EntityKind:
        return ENTITY_KINDS[self.kind]

    def moved_to(self, x: float, y: float) -> Entity:
        return Entity(
            entity_id=self.entity_id,
            kind=self.kind,
            x=x,
            y=y,
            radius=self.radius,
            priority=self.priority,
            spawn_tick=self.spawn_tick,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "position": [round(float(self.x), 4), round(float(self.y), 4)],
            "radius": round(float(self.radius), 4),
            "priority": int(self.priority),
            "spawn_tick": int(self.spawn_tick),
        }

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> Entity:
        try:
            kind = str(document["kind"])
            position = document["position"]
            entity = cls(
                entity_id=str(document["entity_id"]),
                kind=kind,
                x=float(position[0]),
                y=float(position[1]),
                radius=float(document["radius"]),
                priority=int(document["priority"]),
                spawn_tick=int(document["spawn_tick"]),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EntityError(f"Invalid entity record: {exc}") from None
        if kind not in ENTITY_KINDS:
            raise EntityError(f"Unknown entity kind '{kind}'")
        if entity.priority not in PRIORITY_LEVELS.values():
            raise EntityError(f"Unknown priority level {entity.priority}")
        return entity


class EntityRegistry:
    """Owns entity identity and lowers entity behaviour into field writes."""

    def __init__(
        self,
        *,
        bounds: tuple[int, int] = DEFAULT_BOUNDS,
        max_entities: int = MAX_ENTITIES,
    ) -> None:
        if bounds[0] <= 0 or bounds[1] <= 0:
            raise ValueError(f"World bounds must be positive, got {bounds}")
        if max_entities <= 0:
            raise ValueError("max_entities must be positive")
        self.bounds = (int(bounds[0]), int(bounds[1]))
        self.max_entities = int(max_entities)
        self._entities: dict[str, Entity] = {}
        self._counter = 0

    def __len__(self) -> int:
        return len(self._entities)

    def __iter__(self) -> Iterator[Entity]:
        return iter(tuple(self._entities.values()))

    def __contains__(self, entity_id: object) -> bool:
        return entity_id in self._entities

    @property
    def spawned(self) -> int:
        """Total spawns ever made; identifiers are never reused."""
        return self._counter

    def entities(self) -> tuple[Entity, ...]:
        return tuple(self._entities.values())

    def get(self, entity_id: str) -> Entity:
        try:
            return self._entities[entity_id]
        except KeyError:
            raise EntityError(f"No such entity: '{entity_id}'") from None

    def spawn(
        self,
        kind: str,
        position: Sequence[float],
        *,
        tick: int = 0,
        radius: float | None = None,
        priority: int = PRIORITY_LEVELS["ai"],
    ) -> tuple[Entity, list[Operation]]:
        """Create an entity and return it with the writes that realise it."""
        template = self._template(kind)
        if len(self._entities) >= self.max_entities:
            raise EntityError(
                f"Entity limit reached ({self.max_entities}); remove one first"
            )
        if priority not in PRIORITY_LEVELS.values():
            raise EntityError(f"Unknown priority level {priority}")
        x, y = self._coerce_position(position)
        extent = self._coerce_radius(template.radius if radius is None else radius)

        self._counter += 1
        entity = Entity(
            entity_id=f"{template.name}-{self._counter:04d}",
            kind=template.name,
            x=x,
            y=y,
            radius=extent,
            priority=priority,
            spawn_tick=int(tick),
        )
        self._entities[entity.entity_id] = entity
        return entity, self._stamp(entity)

    def remove(self, entity_id: str) -> list[Operation]:
        """Retire an entity and return the writes that release its cells."""
        entity = self.get(entity_id)
        del self._entities[entity_id]
        if entity.template.material is None:
            return []
        return [self._segment(entity, erase=True)]

    def clear(self) -> list[Operation]:
        """Retire every entity, newest first, and release all their cells."""
        operations: list[Operation] = []
        for entity_id in sorted(self._entities, reverse=True):
            operations.extend(self.remove(entity_id))
        return operations

    def advance(
        self,
        grid: npt.NDArray[np.float32] | None = None,
        *,
        tick: int = 0,
    ) -> list[Operation]:
        """Step every entity once and return this tick's writes.

        ``grid`` is a CPU snapshot used to sample the velocity field for
        drifting entities and to keep them out of walls. Without it, entities
        that only emit or heat still work; drifting entities hold position.
        """
        operations: list[Operation] = []
        for entity_id in sorted(self._entities):
            entity = self._entities[entity_id]
            template = entity.template
            moved = False
            if template.drifts and grid is not None:
                relocated = self._drift(entity, grid)
                moved = (relocated.x, relocated.y) != (entity.x, entity.y)
                if moved:
                    if template.clears_trail and template.material is not None:
                        operations.append(self._segment(entity, erase=True))
                    self._entities[entity_id] = relocated
                    entity = relocated
            if template.material is not None and (template.restamps or moved):
                operations.extend(self._stamp(entity))
            if template.temperature:
                operations.append(
                    TemperatureDelta(
                        start_x=entity.x,
                        start_y=entity.y,
                        end_x=entity.x,
                        end_y=entity.y,
                        radius=entity.radius,
                        delta=float(template.temperature),
                        priority=entity.priority,
                    )
                )
        return operations

    def describe(self) -> dict[str, Any]:
        """Compact, model-readable view of the live population."""
        counts: dict[str, int] = {}
        for entity in self._entities.values():
            counts[entity.kind] = counts.get(entity.kind, 0) + 1
        return {
            "count": len(self._entities),
            "limit": self.max_entities,
            "spawned_total": self._counter,
            "by_kind": counts,
            "entities": [entity.to_json() for entity in self.entities()],
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "bounds": [self.bounds[0], self.bounds[1]],
            "counter": self._counter,
            "entities": [entity.to_json() for entity in self.entities()],
        }

    @classmethod
    def from_json(
        cls,
        document: Mapping[str, Any],
        *,
        max_entities: int = MAX_ENTITIES,
    ) -> EntityRegistry:
        try:
            bounds = document["bounds"]
            registry = cls(
                bounds=(int(bounds[0]), int(bounds[1])),
                max_entities=max_entities,
            )
            registry._counter = int(document["counter"])
            records = document["entities"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EntityError(f"Invalid entity registry document: {exc}") from None
        if not isinstance(records, list):
            raise EntityError("'entities' must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise EntityError("Each entity record must be a JSON object")
            entity = Entity.from_json(record)
            registry._entities[entity.entity_id] = entity
        return registry

    def _template(self, kind: str) -> EntityKind:
        if not isinstance(kind, str):
            raise EntityError("Entity kind must be a string")
        key = kind.strip().lower().replace(" ", "_")
        template = ENTITY_KINDS.get(key)
        if template is None:
            raise EntityError(
                f"Unknown entity kind '{kind}'; expected one of "
                f"{sorted(ENTITY_KINDS)}"
            )
        return template

    def _coerce_position(self, position: Sequence[float]) -> tuple[float, float]:
        if isinstance(position, (str, bytes)) or len(position) != 2:
            raise EntityError("Entity position must be [x, y]")
        try:
            x = float(position[0])
            y = float(position[1])
        except (TypeError, ValueError) as exc:
            raise EntityError("Entity position must be two numbers") from exc
        if not (np.isfinite(x) and np.isfinite(y)):
            raise EntityError("Entity position must be finite")
        width, height = self.bounds
        if not (0.0 <= x <= width and 0.0 <= y <= height):
            raise EntityError(
                f"Entity position ({x}, {y}) is outside the world {self.bounds}"
            )
        return x, y

    @staticmethod
    def _coerce_radius(radius: float) -> float:
        try:
            extent = float(radius)
        except (TypeError, ValueError) as exc:
            raise EntityError("Entity radius must be a number") from exc
        if not np.isfinite(extent):
            raise EntityError("Entity radius must be finite")
        if not MIN_ENTITY_RADIUS <= extent <= MAX_ENTITY_RADIUS:
            raise EntityError(
                f"Entity radius must be in "
                f"[{MIN_ENTITY_RADIUS}, {MAX_ENTITY_RADIUS}]"
            )
        return extent

    def _stamp(self, entity: Entity) -> list[Operation]:
        if entity.template.material is None:
            return []
        return [self._segment(entity, erase=False)]

    def _segment(self, entity: Entity, *, erase: bool) -> Segment:
        material = entity.template.material
        category = CATEGORY_IDS["vacuum"] if erase else CATEGORY_IDS[str(material)]
        return Segment(
            start_x=entity.x,
            start_y=entity.y,
            end_x=entity.x,
            end_y=entity.y,
            radius=entity.radius,
            category=category,
            erase=erase,
            priority=entity.priority,
        )

    def _drift(
        self,
        entity: Entity,
        grid: npt.NDArray[np.float32],
    ) -> Entity:
        """Move an entity along the sampled velocity field, bouncing off walls."""
        if grid.ndim != 3 or grid.shape[2] < 32:
            raise EntityError("Grid must have shape (height, width, 32)")
        height, width = int(grid.shape[0]), int(grid.shape[1])
        column = int(np.clip(int(entity.x), 0, width - 1))
        row = int(np.clip(int(entity.y), 0, height - 1))
        velocity = np.asarray(grid[row, column, 0:2], dtype=np.float64)
        if not np.isfinite(velocity).all():
            return entity
        step = velocity * DRIFT_SCALE
        if not np.any(np.abs(step) > 1e-9):
            return entity

        target_x = entity.x + float(step[0])
        target_y = entity.y + float(step[1])
        blocked = (grid[..., 31] >= 0.5) | (grid[..., 24] >= 0.5)
        # Reflect each axis independently; the field's own reflection handles
        # the cell velocities, this keeps the tracked centre out of walls.
        if self._blocked_at(blocked, target_x, entity.y, width, height):
            target_x = entity.x - float(step[0]) * DRIFT_RESTITUTION
        if self._blocked_at(blocked, target_x, target_y, width, height):
            target_y = entity.y - float(step[1]) * DRIFT_RESTITUTION
        if self._blocked_at(blocked, target_x, target_y, width, height):
            return entity
        return entity.moved_to(
            float(np.clip(target_x, 0.0, width - 1e-4)),
            float(np.clip(target_y, 0.0, height - 1e-4)),
        )

    @staticmethod
    def _blocked_at(
        blocked: npt.NDArray[np.bool_],
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> bool:
        column = int(x)
        row = int(y)
        if column < 0 or row < 0 or column >= width or row >= height:
            return True
        return bool(blocked[row, column])


def entity_kind_catalog() -> list[dict[str, Any]]:
    """Describe every spawnable kind for AI metadata and documentation."""
    return [
        {
            "kind": template.name,
            "material": template.material or "none",
            "default_radius": template.radius,
            "drifts_with_field": template.drifts,
            "temperature_per_step": template.temperature,
            "description": template.description,
        }
        for template in ENTITY_KINDS.values()
    ]


__all__ = [
    "DRIFT_SCALE",
    "ENTITY_KINDS",
    "Entity",
    "EntityError",
    "EntityKind",
    "EntityRegistry",
    "MAX_ENTITIES",
    "MAX_ENTITY_RADIUS",
    "MIN_ENTITY_RADIUS",
    "entity_kind_catalog",
]
