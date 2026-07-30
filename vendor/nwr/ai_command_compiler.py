"""Deterministic AI command objects and JSON compiler.

This module is the named command API described by the AI interoperability
design. It validates and serializes commands, then lowers them into the same
GPU operations ``ai_commands.compile_request`` already produces. Entity commands
resolve against the runtime's :class:`~entities.EntityRegistry`; capabilities the
runtime genuinely lacks are rejected explicitly rather than simulated.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Final, Mapping, Sequence

from ai_commands import (
    CATEGORY_IDS,
    DEFAULT_BOUNDS,
    DEFAULT_PRIORITY,
    CommandError,
    Control,
    Operation,
    RemoveEntity,
    Segment,
    SpawnEntity,
    TemperatureDelta,
    compile_request,
)
from ai_world import UNSUPPORTED_COMMANDS
from bds_format import PRIORITY_LEVELS
from entities import ENTITY_KINDS, MAX_ENTITY_RADIUS, MIN_ENTITY_RADIUS

TEMPERATURE_CHANNEL: Final = 6
DEFAULT_TEMPERATURE_DELTA: Final = 0.08
MAX_TEMPERATURE_DELTA: Final = 0.5
MIN_TEMPERATURE_DELTA: Final = 0.01

# Retained for callers that annotated against the older, narrower union.
ExtendedOperation = Operation


@dataclass(frozen=True, slots=True)
class DeterministicCommand:
    """Serializable, CRC-stamped command envelope for replay and transport."""

    name: str
    tick: int
    payload: dict[str, Any]
    crc32: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> DeterministicCommand:
        try:
            name = str(document["name"])
            tick = int(document["tick"])
            payload = dict(document["payload"])
            crc32 = str(document["crc32"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise CommandError(f"Invalid deterministic command envelope: {exc}") from None
        expected = _payload_crc(name, tick, payload)
        if crc32 != expected:
            raise CommandError(
                f"CRC mismatch for '{name}' at tick {tick}: "
                f"got {crc32}, expected {expected}"
            )
        return cls(name=name, tick=tick, payload=payload, crc32=crc32)


def _payload_crc(name: str, tick: int, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"name": name, "tick": tick, "payload": payload},
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"{zlib.crc32(encoded) & 0xFFFFFFFF:08x}"


def _seal(name: str, tick: int, payload: Mapping[str, Any]) -> DeterministicCommand:
    body = dict(payload)
    return DeterministicCommand(
        name=name,
        tick=tick,
        payload=body,
        crc32=_payload_crc(name, tick, body),
    )


@dataclass(frozen=True, slots=True)
class PaintMaterialCommand:
    material: str
    center: tuple[float, float]
    radius: float
    tick: int = 0
    priority: str = "ai"

    def seal(self) -> DeterministicCommand:
        return _seal(
            "PaintMaterial",
            self.tick,
            {
                "material": self.material,
                "center": [self.center[0], self.center[1]],
                "radius": self.radius,
                "priority": self.priority,
            },
        )

    def to_operations(self, *, bounds: tuple[int, int] = DEFAULT_BOUNDS) -> list[ExtendedOperation]:
        return compile_ai_json(
            {
                "command": "PaintMaterial",
                "material": self.material,
                "center": [self.center[0], self.center[1]],
                "radius": self.radius,
                "priority": self.priority,
            },
            tick=self.tick,
            bounds=bounds,
        )


@dataclass(frozen=True, slots=True)
class EraseCommand:
    center: tuple[float, float]
    radius: float
    tick: int = 0
    priority: str = "ai"

    def seal(self) -> DeterministicCommand:
        return _seal(
            "Erase",
            self.tick,
            {
                "center": [self.center[0], self.center[1]],
                "radius": self.radius,
                "priority": self.priority,
            },
        )

    def to_operations(self, *, bounds: tuple[int, int] = DEFAULT_BOUNDS) -> list[ExtendedOperation]:
        return compile_ai_json(
            {
                "command": "Erase",
                "center": [self.center[0], self.center[1]],
                "radius": self.radius,
                "priority": self.priority,
            },
            tick=self.tick,
            bounds=bounds,
        )


@dataclass(frozen=True, slots=True)
class SetMaterialCommand:
    material: str
    center: tuple[float, float]
    radius: float
    tick: int = 0
    priority: str = "ai"

    def seal(self) -> DeterministicCommand:
        return _seal(
            "SetMaterial",
            self.tick,
            {
                "material": self.material,
                "center": [self.center[0], self.center[1]],
                "radius": self.radius,
                "priority": self.priority,
            },
        )

    def to_operations(self, *, bounds: tuple[int, int] = DEFAULT_BOUNDS) -> list[ExtendedOperation]:
        return PaintMaterialCommand(
            material=self.material,
            center=self.center,
            radius=self.radius,
            tick=self.tick,
            priority=self.priority,
        ).to_operations(bounds=bounds)


@dataclass(frozen=True, slots=True)
class IncreaseTemperatureCommand:
    center: tuple[float, float]
    radius: float
    amount: float = DEFAULT_TEMPERATURE_DELTA
    tick: int = 0
    priority: str = "ai"

    def seal(self) -> DeterministicCommand:
        return _seal(
            "IncreaseTemperature",
            self.tick,
            {
                "center": [self.center[0], self.center[1]],
                "radius": self.radius,
                "amount": self.amount,
                "priority": self.priority,
            },
        )

    def to_operations(self, *, bounds: tuple[int, int] = DEFAULT_BOUNDS) -> list[ExtendedOperation]:
        return compile_ai_json(
            {
                "command": "IncreaseTemperature",
                "center": [self.center[0], self.center[1]],
                "radius": self.radius,
                "amount": self.amount,
                "priority": self.priority,
            },
            tick=self.tick,
            bounds=bounds,
        )


@dataclass(frozen=True, slots=True)
class DecreaseTemperatureCommand:
    center: tuple[float, float]
    radius: float
    amount: float = DEFAULT_TEMPERATURE_DELTA
    tick: int = 0
    priority: str = "ai"

    def seal(self) -> DeterministicCommand:
        return _seal(
            "DecreaseTemperature",
            self.tick,
            {
                "center": [self.center[0], self.center[1]],
                "radius": self.radius,
                "amount": self.amount,
                "priority": self.priority,
            },
        )

    def to_operations(self, *, bounds: tuple[int, int] = DEFAULT_BOUNDS) -> list[ExtendedOperation]:
        return compile_ai_json(
            {
                "command": "DecreaseTemperature",
                "center": [self.center[0], self.center[1]],
                "radius": self.radius,
                "amount": self.amount,
                "priority": self.priority,
            },
            tick=self.tick,
            bounds=bounds,
        )


@dataclass(frozen=True, slots=True)
class SpawnEntityCommand:
    """Create a tracked entity through the runtime's registry."""

    kind: str
    position: tuple[float, float]
    radius: float | None = None
    tick: int = 0
    priority: str = "ai"

    def seal(self) -> DeterministicCommand:
        return _seal(
            "SpawnEntity",
            self.tick,
            {
                "kind": self.kind,
                "position": [self.position[0], self.position[1]],
                "radius": self.radius,
                "priority": self.priority,
            },
        )

    def to_operations(self, *, bounds: tuple[int, int] = DEFAULT_BOUNDS) -> list[ExtendedOperation]:
        return compile_ai_json(
            {
                "command": "SpawnEntity",
                "kind": self.kind,
                "position": [self.position[0], self.position[1]],
                **({} if self.radius is None else {"radius": self.radius}),
                "priority": self.priority,
            },
            tick=self.tick,
            bounds=bounds,
        )


@dataclass(frozen=True, slots=True)
class RemoveEntityCommand:
    """Retire a tracked entity and release the cells it owns."""

    entity_id: str
    tick: int = 0
    priority: str = "ai"

    def seal(self) -> DeterministicCommand:
        return _seal(
            "RemoveEntity",
            self.tick,
            {"entity_id": self.entity_id, "priority": self.priority},
        )

    def to_operations(self, *, bounds: tuple[int, int] = DEFAULT_BOUNDS) -> list[ExtendedOperation]:
        return compile_ai_json(
            {
                "command": "RemoveEntity",
                "entity_id": self.entity_id,
                "priority": self.priority,
            },
            tick=self.tick,
            bounds=bounds,
        )


def compile_ai_json(
    document: Any,
    *,
    tick: int = 0,
    default_priority: int = PRIORITY_LEVELS["ai"],
    bounds: tuple[int, int] = DEFAULT_BOUNDS,
) -> list[ExtendedOperation]:
    """Compile either the native command grammar or the named AI command form."""
    if isinstance(document, Mapping) and "commands" in document:
        return list(
            compile_request(
                document,
                default_priority=default_priority,
                bounds=bounds,
            )
        )
    if isinstance(document, Mapping) and "command" in document:
        return [_compile_named(document, tick=tick, default_priority=default_priority, bounds=bounds)]
    if isinstance(document, list):
        operations: list[ExtendedOperation] = []
        for index, item in enumerate(document):
            try:
                operations.extend(
                    compile_ai_json(
                        item,
                        tick=tick,
                        default_priority=default_priority,
                        bounds=bounds,
                    )
                )
            except CommandError as exc:
                raise CommandError(f"commands[{index}]: {exc}") from None
        return operations
    raise CommandError(
        "Expected {'commands': [...]} or a named {'command': '...'} object"
    )


def compile_sealed(
    command: DeterministicCommand,
    *,
    default_priority: int = PRIORITY_LEVELS["ai"],
    bounds: tuple[int, int] = DEFAULT_BOUNDS,
) -> list[ExtendedOperation]:
    """Re-validate a CRC-sealed envelope and lower it to GPU operations."""
    # Recompute CRC through from_mapping.
    DeterministicCommand.from_mapping(asdict(command))
    document = {"command": command.name, **command.payload}
    return compile_ai_json(
        document,
        tick=command.tick,
        default_priority=default_priority,
        bounds=bounds,
    )


def _compile_named(
    document: Mapping[str, Any],
    *,
    tick: int,
    default_priority: int,
    bounds: tuple[int, int],
) -> ExtendedOperation:
    name = document.get("command")
    if not isinstance(name, str):
        raise CommandError("'command' must be a string")
    normalized = name.strip()
    lowered = normalized.lower().replace(" ", "_")
    if lowered in UNSUPPORTED_COMMANDS:
        raise CommandError(
            f"'{normalized}' is unsupported: {UNSUPPORTED_COMMANDS[lowered]}"
        )

    priority = _read_priority(document, default_priority)
    if lowered in {"spawn_entity", "spawnentity"}:
        kind = document.get("kind")
        if not isinstance(kind, str):
            raise CommandError("'kind' must be a string")
        key = kind.strip().lower().replace(" ", "_")
        if key not in ENTITY_KINDS:
            raise CommandError(
                f"Unknown entity kind '{kind}'; expected one of "
                f"{sorted(ENTITY_KINDS)}"
            )
        position = _read_position(document, bounds)
        radius = (
            None
            if "radius" not in document
            else _read_radius(
                document,
                minimum=MIN_ENTITY_RADIUS,
                maximum=MAX_ENTITY_RADIUS,
            )
        )
        return SpawnEntity(
            kind=key,
            x=position[0],
            y=position[1],
            radius=radius,
            priority=priority,
        )

    if lowered in {"remove_entity", "removeentity"}:
        entity_id = document.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise CommandError("'entity_id' must be a non-empty string")
        return RemoveEntity(entity_id=entity_id.strip(), priority=priority)

    if normalized in {"PaintMaterial", "SetMaterial", "paint_material", "set_material"}:
        material = document.get("material")
        if not isinstance(material, str):
            raise CommandError("'material' must be a string")
        key = material.strip().lower().replace(" ", "_")
        if key not in CATEGORY_IDS or key == "vacuum":
            raise CommandError(
                f"Material '{material}' cannot be painted; use Erase to clear"
            )
        center = _read_center(document, bounds)
        radius = _read_radius(document)
        return Segment(
            start_x=center[0],
            start_y=center[1],
            end_x=center[0],
            end_y=center[1],
            radius=radius,
            category=CATEGORY_IDS[key],
            erase=False,
            priority=priority,
        )

    if normalized in {"Erase", "erase"}:
        center = _read_center(document, bounds)
        radius = _read_radius(document)
        return Segment(
            start_x=center[0],
            start_y=center[1],
            end_x=center[0],
            end_y=center[1],
            radius=radius,
            category=CATEGORY_IDS["vacuum"],
            erase=True,
            priority=priority,
        )

    if normalized in {
        "IncreaseTemperature",
        "DecreaseTemperature",
        "increase_temperature",
        "decrease_temperature",
    }:
        center = _read_center(document, bounds)
        radius = _read_radius(document)
        amount = _read_amount(document)
        if normalized.lower().startswith("decrease"):
            amount = -abs(amount)
        else:
            amount = abs(amount)
        return TemperatureDelta(
            start_x=center[0],
            start_y=center[1],
            end_x=center[0],
            end_y=center[1],
            radius=radius,
            delta=amount,
            priority=priority,
        )

    if normalized.lower() in {"reset", "save", "load", "pause", "resume"}:
        return Control(action=normalized.lower())

    raise CommandError(f"Unknown named command '{normalized}'")


def _read_priority(document: Mapping[str, Any], default: int) -> int:
    if "priority" not in document:
        return default
    value = document["priority"]
    if isinstance(value, str) and value in PRIORITY_LEVELS:
        level = PRIORITY_LEVELS[value]
    elif isinstance(value, int) and value in PRIORITY_LEVELS.values():
        level = value
    else:
        raise CommandError(f"'priority' must be one of {sorted(PRIORITY_LEVELS)}")
    if level > default:
        raise CommandError("Named command priority exceeds caller authority")
    return level


def _read_center(
    document: Mapping[str, Any],
    bounds: tuple[int, int],
) -> tuple[float, float]:
    return _read_coordinate(document, "center", bounds)


def _read_position(
    document: Mapping[str, Any],
    bounds: tuple[int, int],
) -> tuple[float, float]:
    key = "position" if "position" in document else "center"
    return _read_coordinate(document, key, bounds)


def _read_coordinate(
    document: Mapping[str, Any],
    key: str,
    bounds: tuple[int, int],
) -> tuple[float, float]:
    raw = document.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise CommandError(f"'{key}' must be [x, y]")
    try:
        x = float(raw[0])
        y = float(raw[1])
    except (TypeError, ValueError) as exc:
        raise CommandError(f"'{key}' values must be finite numbers") from exc
    if not (np_isfinite(x) and np_isfinite(y)):
        raise CommandError(f"'{key}' values must be finite numbers")
    if not (0.0 <= x <= bounds[0] and 0.0 <= y <= bounds[1]):
        raise CommandError(f"'{key}' {raw} is outside the world bounds {bounds}")
    return (x, y)


def _read_radius(
    document: Mapping[str, Any],
    *,
    default: float = 8.0,
    minimum: float = 0.5,
    maximum: float = 64.0,
) -> float:
    raw = document.get("radius", default)
    try:
        radius = float(raw)
    except (TypeError, ValueError) as exc:
        raise CommandError("'radius' must be a finite number") from exc
    if not np_isfinite(radius) or not minimum <= radius <= maximum:
        raise CommandError(f"'radius' must be in [{minimum}, {maximum}]")
    return radius


def _read_amount(document: Mapping[str, Any]) -> float:
    raw = document.get("amount", DEFAULT_TEMPERATURE_DELTA)
    try:
        amount = float(raw)
    except (TypeError, ValueError) as exc:
        raise CommandError("'amount' must be a finite number") from exc
    if not np_isfinite(amount):
        raise CommandError("'amount' must be a finite number")
    magnitude = abs(amount)
    if not MIN_TEMPERATURE_DELTA <= magnitude <= MAX_TEMPERATURE_DELTA:
        raise CommandError(
            f"'amount' magnitude must be in "
            f"[{MIN_TEMPERATURE_DELTA}, {MAX_TEMPERATURE_DELTA}]"
        )
    return magnitude


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


__all__ = [
    "DecreaseTemperatureCommand",
    "DeterministicCommand",
    "EraseCommand",
    "ExtendedOperation",
    "IncreaseTemperatureCommand",
    "PaintMaterialCommand",
    "RemoveEntity",
    "RemoveEntityCommand",
    "SetMaterialCommand",
    "SpawnEntity",
    "SpawnEntityCommand",
    "TEMPERATURE_CHANNEL",
    "TemperatureDelta",
    "compile_ai_json",
    "compile_sealed",
]
