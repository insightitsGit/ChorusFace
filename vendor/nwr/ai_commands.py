"""Validate and compile semantic world commands into GPU paint segments.

This module is the deterministic boundary between an AI model and the
simulation. Models emit high-level regions; this layer rasterizes them into
the same segment primitive the mouse produces, so no caller can write raw
cell vectors or bypass validation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Final, Iterable, Mapping, Sequence

from bds_format import GRID_HEIGHT, GRID_WIDTH, PRIORITY_LEVELS

DEFAULT_PRIORITY: Final = PRIORITY_LEVELS["user"]
DEFAULT_BOUNDS: Final[tuple[int, int]] = (GRID_WIDTH, GRID_HEIGHT)
CATEGORY_IDS: Final[dict[str, int]] = {
    "vacuum": 0,
    "human_barrier": 1,
    "active_fluid": 2,
    "solid": 3,
}
PAINTABLE_CATEGORIES: Final[tuple[str, ...]] = (
    "human_barrier",
    "active_fluid",
    "solid",
)
CONTROL_ACTIONS: Final[tuple[str, ...]] = (
    "reset",
    "save",
    "load",
    "pause",
    "resume",
)
# Control actions that read or write the world file. They swap the whole world
# outside the per-cell write path, so no amount of cell-level authority applies
# to them and a caller below human authority may not ask for them at all.
# ``reset`` is deliberately absent: it stays available to the AI because the
# runtime preserves human-locked cells across an AI reset.
OPERATOR_CONTROL_ACTIONS: Final[tuple[str, ...]] = ("save", "load")
# Painting this category mints a human lock on every cell it covers. Minting a
# boundary that its own writes can then never cross is a human prerogative, so
# only a caller holding human authority may ask for it. Agents that want a wall
# should paint ``solid``, which is a hard surface with no lock.
LOCK_MINTING_CATEGORIES: Final[tuple[str, ...]] = ("human_barrier",)
# Two descriptions of the same action, one per side of that boundary. A caller
# that may not paint a lock is told about the categories it has, not about the
# one it is missing, so the grammar it reads never dangles an unusable option.
PAINT_DESCRIPTION: Final = (
    "Write a category into cells. 'solid' is an immovable wall that flow "
    "bounces off; 'human_barrier' is the same wall plus a human lock that no AI "
    "write may ever overwrite, and painting it requires 'user' authority; "
    "'active_fluid' then evolves under simulation."
)
PAINT_DESCRIPTION_WITHOUT_LOCKS: Final = (
    "Write a category into cells. 'solid' is an immovable wall that flow "
    "bounces off; 'active_fluid' then evolves under simulation. Writes land "
    "only where the cell carries no human lock."
)
# Which class of writer produced a command. The constraint shader picks the
# opcode from this (`±1` human, `±2` AI) and only vetoes AI writes on locked
# cells, so a write that loses its writer identity also loses the Master Lock.
# ``UNSPECIFIED`` exists for logs written before the field was recorded.
WRITER_UNSPECIFIED: Final = 0
WRITER_HUMAN: Final = 1
WRITER_AI: Final = 2
WRITER_SOURCES: Final[tuple[int, ...]] = (
    WRITER_UNSPECIFIED,
    WRITER_HUMAN,
    WRITER_AI,
)

MAX_COMMANDS_PER_REQUEST: Final = 64
MAX_SEGMENTS_PER_REQUEST: Final = 4096
MAX_POINTS_PER_REGION: Final = 128
MAX_BRUSH_RADIUS: Final = 64.0
MIN_BRUSH_RADIUS: Final = 0.5
_FILL_RADIUS: Final = 0.5
_EPSILON: Final = 1e-9


class CommandError(ValueError):
    """Raised when an AI request is malformed, unsafe, or out of bounds."""


@dataclass(frozen=True, slots=True)
class Segment:
    """A capsule of cells to overwrite, in grid coordinates with y pointing up."""

    start_x: float
    start_y: float
    end_x: float
    end_y: float
    radius: float
    category: int
    erase: bool
    priority: int = DEFAULT_PRIORITY
    source: int = WRITER_UNSPECIFIED


@dataclass(frozen=True, slots=True)
class Control:
    """A runtime action that is not a cell write."""

    action: str


@dataclass(frozen=True, slots=True)
class TemperatureDelta:
    """Adjust temperature inside a capsule without rewriting other channels."""

    start_x: float
    start_y: float
    end_x: float
    end_y: float
    radius: float
    delta: float
    priority: int = DEFAULT_PRIORITY


@dataclass(frozen=True, slots=True)
class VelocityImpulse:
    """Add a velocity vector to a disc of cells without rewriting material.

    Not part of the request grammar — no `action` compiles to one. It exists so
    that the impulses the avatar and game drive through the runtime are a
    first-class operation the log can record and replay, rather than a shape
    only `PaintCommand` knows about. Coverage is a disc around `(x, y)`, matching
    the shader, which reads the impulse from `segment.zw` for `operation = ±4`.
    """

    x: float
    y: float
    velocity_x: float
    velocity_y: float
    radius: float
    priority: int = DEFAULT_PRIORITY
    source: int = WRITER_UNSPECIFIED


@dataclass(frozen=True, slots=True)
class SpawnEntity:
    """Ask the runtime's entity registry to create a tracked entity.

    The compiler cannot allocate an identity on its own, so this is an intent.
    :mod:`entities` resolves it into ordinary segments at the caller's authority.
    """

    kind: str
    x: float
    y: float
    radius: float | None = None
    priority: int = DEFAULT_PRIORITY


@dataclass(frozen=True, slots=True)
class RemoveEntity:
    """Retire a tracked entity and release the cells it owns."""

    entity_id: str
    priority: int = DEFAULT_PRIORITY


Operation = (
    Segment
    | Control
    | TemperatureDelta
    | VelocityImpulse
    | SpawnEntity
    | RemoveEntity
)

REGION_TYPES: Final[dict[str, dict[str, Any]]] = {
    "point": {
        "required": ("position",),
        "optional": ("radius",),
        "description": "Filled disc centred on a single cell coordinate.",
    },
    "line": {
        "required": ("start", "end"),
        "optional": ("thickness",),
        "description": "Straight stroke between two coordinates.",
    },
    "polyline": {
        "required": ("points",),
        "optional": ("thickness",),
        "description": "Open stroke through an ordered list of coordinates.",
    },
    "circle": {
        "required": ("center", "radius"),
        "optional": (),
        "description": "Filled disc.",
    },
    "ring": {
        "required": ("center", "radius"),
        "optional": ("thickness",),
        "description": "Hollow circular outline.",
    },
    "rectangle": {
        "required": ("min", "max"),
        "optional": (),
        "description": "Filled axis-aligned rectangle.",
    },
    "rectangle_outline": {
        "required": ("min", "max"),
        "optional": ("thickness",),
        "description": "Hollow axis-aligned rectangle, useful for walls.",
    },
    "polygon": {
        "required": ("points",),
        "optional": (),
        "description": "Filled polygon using even-odd scanline coverage.",
    },
    "polygon_outline": {
        "required": ("points",),
        "optional": ("thickness",),
        "description": "Closed polygon outline.",
    },
}

COMMAND_SCHEMA: Final[dict[str, Any]] = {
    "request": {
        "commands": "ordered list of 1..%d command objects" % MAX_COMMANDS_PER_REQUEST
    },
    "coordinates": {
        "space": "grid cells",
        "width": GRID_WIDTH,
        "height": GRID_HEIGHT,
        "origin": "bottom-left",
        "x_range": [0, GRID_WIDTH],
        "y_range": [0, GRID_HEIGHT],
    },
    "actions": {
        "paint": {
            "required": ["category", "region"],
            "optional": ["priority"],
            "category": list(PAINTABLE_CATEGORIES),
            "description": PAINT_DESCRIPTION,
        },
        "erase": {
            "required": ["region"],
            "optional": ["priority"],
            "description": "Clear cells to vacuum and release their authority.",
        },
        "reset": {
            "description": (
                "Restore the deterministic seed world. Human-locked cells are "
                "carried across when the caller is below 'user' authority."
            )
        },
        "save": {
            "description": (
                "Write the current world to its file. Requires 'user' authority."
            )
        },
        "load": {
            "description": (
                "Reload the world from its file. Requires 'user' authority."
            )
        },
        "pause": {"description": "Stop advancing simulation ticks."},
        "resume": {"description": "Resume advancing simulation ticks."},
    },
    "regions": {
        name: {
            "required": list(spec["required"]),
            "optional": list(spec["optional"]),
            "description": spec["description"],
        }
        for name, spec in REGION_TYPES.items()
    },
    "priority": {
        "levels": sorted(PRIORITY_LEVELS, key=PRIORITY_LEVELS.get),
        "description": (
            "A write only succeeds where the target cell's authority is at or "
            "below the command's. Callers cannot exceed their own authority."
        ),
    },
    "limits": {
        "max_commands_per_request": MAX_COMMANDS_PER_REQUEST,
        "max_segments_per_request": MAX_SEGMENTS_PER_REQUEST,
        "max_points_per_region": MAX_POINTS_PER_REGION,
        "brush_radius_range": [MIN_BRUSH_RADIUS, MAX_BRUSH_RADIUS],
    },
    "example": {
        "commands": [
            {
                "action": "paint",
                "category": "active_fluid",
                "region": {"type": "circle", "center": [128, 110], "radius": 34},
            },
            {
                "action": "paint",
                "category": "human_barrier",
                "region": {
                    "type": "rectangle_outline",
                    "min": [72, 54],
                    "max": [184, 166],
                    "thickness": 3,
                },
            },
        ]
    },
}

SYSTEM_PROMPT: Final = (
    "You control a 256x256 cell simulation grid. Translate the user's request "
    "into world commands. Reply with a single JSON object of the form "
    '{"commands": [...]} and no prose. Coordinates are grid cells with the '
    "origin at the bottom-left, x to the right and y upward. Use 'paint' with "
    "category 'solid' for walls that flow bounces off and 'active_fluid' for "
    "liquid or energy; use 'erase' to clear regions. Prefer a small number of "
    "large regions over many tiny strokes. Only paint 'human_barrier' if the "
    "grammar you were given lists it: it mints a human lock, so it is reserved "
    "for callers holding human authority, and 'solid' is the wall to use "
    "otherwise."
)


def compile_request(
    payload: Any,
    *,
    default_priority: int = DEFAULT_PRIORITY,
    bounds: tuple[int, int] = DEFAULT_BOUNDS,
) -> list[Operation]:
    """Validate a request and rasterize it into ordered GPU operations."""
    if default_priority not in PRIORITY_LEVELS.values():
        raise CommandError(f"Unknown default priority: {default_priority}")
    if bounds[0] <= 0 or bounds[1] <= 0:
        raise CommandError(f"World bounds must be positive, got {bounds}")
    commands = _extract_commands(payload)
    operations: list[Operation] = []
    segment_count = 0
    for index, command in enumerate(commands):
        try:
            produced = _compile_command(
                command,
                default_priority=default_priority,
                bounds=bounds,
            )
        except CommandError as exc:
            raise CommandError(f"commands[{index}]: {exc}") from None
        segment_count += sum(1 for item in produced if isinstance(item, Segment))
        if segment_count > MAX_SEGMENTS_PER_REQUEST:
            raise CommandError(
                f"commands[{index}]: request needs more than "
                f"{MAX_SEGMENTS_PER_REQUEST} segments; use larger or fewer regions"
            )
        operations.extend(produced)
    if not operations:
        raise CommandError("Request produced no operations")
    return operations


def _extract_commands(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if "commands" not in payload:
            raise CommandError("Request object must contain a 'commands' list")
        unexpected = set(payload).difference({"commands"})
        if unexpected:
            raise CommandError(f"Unexpected request fields: {sorted(unexpected)}")
        raw = payload["commands"]
    else:
        raw = payload
    if not isinstance(raw, list) or not raw:
        raise CommandError("'commands' must be a non-empty list")
    if len(raw) > MAX_COMMANDS_PER_REQUEST:
        raise CommandError(
            f"Request contains {len(raw)} commands; the limit is "
            f"{MAX_COMMANDS_PER_REQUEST}"
        )
    for command in raw:
        if not isinstance(command, Mapping):
            raise CommandError("Each command must be a JSON object")
    return list(raw)


def _compile_command(
    command: Mapping[str, Any],
    *,
    default_priority: int,
    bounds: tuple[int, int],
) -> list[Operation]:
    action = command.get("action")
    if not isinstance(action, str):
        raise CommandError("Missing string 'action'")
    if action in CONTROL_ACTIONS:
        unexpected = set(command).difference({"action"})
        if unexpected:
            raise CommandError(
                f"Action '{action}' accepts no other fields; got {sorted(unexpected)}"
            )
        if (
            action in OPERATOR_CONTROL_ACTIONS
            and default_priority < PRIORITY_LEVELS["user"]
        ):
            raise CommandError(
                f"Action '{action}' replaces the world through the file system and "
                "is reserved for callers with 'user' authority"
            )
        return [Control(action=action)]
    if action == "erase":
        allowed = {"action", "region", "priority"}
        category = "vacuum"
    elif action == "paint":
        allowed = {"action", "region", "category", "priority"}
        category = command.get("category")
        if not isinstance(category, str):
            raise CommandError("Action 'paint' requires a string 'category'")
        if category not in PAINTABLE_CATEGORIES:
            raise CommandError(
                f"Category '{category}' cannot be painted; choose one of "
                f"{list(PAINTABLE_CATEGORIES)} or use action 'erase'"
            )
        if (
            category in LOCK_MINTING_CATEGORIES
            and default_priority < PRIORITY_LEVELS["user"]
        ):
            raise CommandError(
                f"Category '{category}' mints a human lock and is reserved for "
                "callers with 'user' authority; paint 'solid' for a plain wall"
            )
    else:
        raise CommandError(
            f"Unknown action '{action}'; expected one of "
            f"{['paint', 'erase', *CONTROL_ACTIONS]}"
        )

    unexpected = set(command).difference(allowed)
    if unexpected:
        raise CommandError(
            f"Action '{action}' does not accept {sorted(unexpected)}"
        )
    region = command.get("region")
    if not isinstance(region, Mapping):
        raise CommandError(f"Action '{action}' requires a 'region' object")
    segments = _compile_region(
        region,
        category=CATEGORY_IDS[category],
        erase=action == "erase",
        priority=_read_priority(command, default_priority),
        bounds=bounds,
    )
    if not segments:
        raise CommandError("Region covers no cells inside the grid")
    return list(segments)


def schema_for_authority(authority: int) -> dict[str, Any]:
    """`COMMAND_SCHEMA` narrowed to what a caller at `authority` may actually do.

    A model reads the schema to find out what it is allowed to ask for. Serving
    the unrestricted grammar to a caller that cannot use all of it would just
    trade a clear refusal here for a rejected request later, so the paintable
    categories and the available actions are filtered to match.
    """
    if authority not in PRIORITY_LEVELS.values():
        raise ValueError(f"Unknown authority level: {authority}")
    if authority >= PRIORITY_LEVELS["user"]:
        return copy.deepcopy(COMMAND_SCHEMA)

    schema = copy.deepcopy(COMMAND_SCHEMA)
    categories = [
        name
        for name in PAINTABLE_CATEGORIES
        if name not in LOCK_MINTING_CATEGORIES
    ]
    schema["actions"]["paint"]["category"] = categories
    schema["actions"]["paint"]["description"] = PAINT_DESCRIPTION_WITHOUT_LOCKS
    for action in OPERATOR_CONTROL_ACTIONS:
        schema["actions"].pop(action, None)
    schema["restrictions"] = {
        "withheld_actions": list(OPERATOR_CONTROL_ACTIONS),
        "withheld_categories": list(LOCK_MINTING_CATEGORIES),
        "reason": (
            "These replace a world or mint a human lock outside the per-cell "
            "authority rules, so they are reserved for callers holding 'user' "
            "authority. Paint 'solid' for a wall that carries no lock."
        ),
    }
    example = schema.get("example", {})
    for command in example.get("commands", []):
        if command.get("category") in LOCK_MINTING_CATEGORIES:
            command["category"] = "solid"
    return schema


def _read_priority(command: Mapping[str, Any], default: int) -> int:
    if "priority" not in command:
        return default
    value = command["priority"]
    if not isinstance(value, str) or value not in PRIORITY_LEVELS:
        raise CommandError(
            f"'priority' must be one of {sorted(PRIORITY_LEVELS)}"
        )
    level = PRIORITY_LEVELS[value]
    if level > default:
        raise CommandError(
            f"'priority' cannot exceed the authority of this caller "
            f"({[name for name, value in PRIORITY_LEVELS.items() if value == default][0]})"
        )
    return level


@dataclass(frozen=True, slots=True)
class _WriteContext:
    """The authority, material, and world extent a region write applies to."""

    category: int
    erase: bool
    priority: int
    bounds: tuple[int, int]


def _compile_region(
    region: Mapping[str, Any],
    *,
    category: int,
    erase: bool,
    priority: int,
    bounds: tuple[int, int],
) -> list[Segment]:
    context = _WriteContext(
        category=category,
        erase=erase,
        priority=priority,
        bounds=bounds,
    )
    region_type = region.get("type")
    if not isinstance(region_type, str) or region_type not in REGION_TYPES:
        raise CommandError(
            f"Unknown region type {region_type!r}; expected one of "
            f"{sorted(REGION_TYPES)}"
        )
    specification = REGION_TYPES[region_type]
    allowed = {"type", *specification["required"], *specification["optional"]}
    unexpected = set(region).difference(allowed)
    if unexpected:
        raise CommandError(
            f"Region '{region_type}' does not accept {sorted(unexpected)}"
        )
    missing = [key for key in specification["required"] if key not in region]
    if missing:
        raise CommandError(f"Region '{region_type}' requires {missing}")

    def stroke(default: float) -> float:
        thickness = _read_number(region, "thickness", default=default, minimum=1.0)
        return _clamp(thickness / 2.0, MIN_BRUSH_RADIUS, MAX_BRUSH_RADIUS)

    if region_type == "point":
        position = _read_point(region, "position", bounds)
        radius = _read_number(
            region,
            "radius",
            default=2.0,
            minimum=MIN_BRUSH_RADIUS,
            maximum=MAX_BRUSH_RADIUS,
        )
        return _stroke_segments([position], radius, context, closed=False)

    if region_type == "line":
        start = _read_point(region, "start", bounds)
        end = _read_point(region, "end", bounds)
        return _stroke_segments([start, end], stroke(3.0), context, closed=False)

    if region_type == "polyline":
        points = _read_points(region, "points", bounds, minimum_count=2)
        return _stroke_segments(points, stroke(3.0), context, closed=False)

    if region_type == "circle":
        center = _read_point(region, "center", bounds)
        radius = _read_number(
            region,
            "radius",
            minimum=MIN_BRUSH_RADIUS,
            maximum=MAX_BRUSH_RADIUS,
        )
        return _stroke_segments([center], radius, context, closed=False)

    if region_type == "ring":
        center = _read_point(region, "center", bounds)
        radius = _read_number(
            region,
            "radius",
            minimum=1.0,
            maximum=float(max(bounds)),
        )
        thickness = stroke(3.0)
        return _stroke_segments(
            _circle_points(center, radius, thickness),
            thickness,
            context,
            closed=True,
        )

    if region_type == "rectangle":
        minimum, maximum = _read_box(region, bounds)
        return _fill_polygon(_box_points(minimum, maximum), context)

    if region_type == "rectangle_outline":
        minimum, maximum = _read_box(region, bounds)
        return _stroke_segments(
            _box_points(minimum, maximum),
            stroke(3.0),
            context,
            closed=True,
        )

    if region_type == "polygon":
        points = _read_points(region, "points", bounds, minimum_count=3)
        return _fill_polygon(points, context)

    points = _read_points(region, "points", bounds, minimum_count=3)
    return _stroke_segments(points, stroke(3.0), context, closed=True)


def _stroke_segments(
    points: Sequence[tuple[float, float]],
    radius: float,
    context: _WriteContext,
    *,
    closed: bool,
) -> list[Segment]:
    if len(points) == 1:
        single = points[0]
        return [
            Segment(
                start_x=single[0],
                start_y=single[1],
                end_x=single[0],
                end_y=single[1],
                radius=radius,
                category=context.category,
                erase=context.erase,
                priority=context.priority,
            )
        ]
    ordered = list(points)
    if closed and ordered[0] != ordered[-1]:
        ordered.append(ordered[0])
    return [
        Segment(
            start_x=start[0],
            start_y=start[1],
            end_x=end[0],
            end_y=end[1],
            radius=radius,
            category=context.category,
            erase=context.erase,
            priority=context.priority,
        )
        for start, end in zip(ordered, ordered[1:])
    ]


def _fill_polygon(
    points: Sequence[tuple[float, float]],
    context: _WriteContext,
) -> list[Segment]:
    width, height = context.bounds
    edges = list(zip(points, [*points[1:], points[0]]))
    lowest = min(point[1] for point in points)
    highest = max(point[1] for point in points)
    first_row = max(math.ceil(lowest - 0.5 - _EPSILON), 0)
    last_row = min(math.floor(highest - 0.5 + _EPSILON), height - 1)

    segments: list[Segment] = []
    for row in range(first_row, last_row + 1):
        center_y = row + 0.5
        crossings: list[float] = []
        for (x1, y1), (x2, y2) in edges:
            if (y1 <= center_y < y2) or (y2 <= center_y < y1):
                crossings.append(x1 + (center_y - y1) / (y2 - y1) * (x2 - x1))
        crossings.sort()
        for index in range(0, len(crossings) - 1, 2):
            span = _row_span(crossings[index], crossings[index + 1], width)
            if span is not None:
                segments.append(
                    Segment(
                        start_x=span[0],
                        start_y=center_y,
                        end_x=span[1],
                        end_y=center_y,
                        radius=_FILL_RADIUS,
                        category=context.category,
                        erase=context.erase,
                        priority=context.priority,
                    )
                )
    return segments


def _row_span(
    left: float,
    right: float,
    width: int,
) -> tuple[float, float] | None:
    first = max(math.ceil(left - 0.5 - _EPSILON), 0)
    last = min(math.floor(right - 0.5 + _EPSILON), width - 1)
    if first > last:
        return None
    return (first + 0.5, last + 0.5)


def _circle_points(
    center: tuple[float, float],
    radius: float,
    thickness: float,
) -> list[tuple[float, float]]:
    step = max(thickness, 1.0)
    count = int(_clamp(math.ceil(2.0 * math.pi * radius / step), 12, 180))
    return [
        (
            center[0] + radius * math.cos(2.0 * math.pi * index / count),
            center[1] + radius * math.sin(2.0 * math.pi * index / count),
        )
        for index in range(count)
    ]


def _box_points(
    minimum: tuple[float, float],
    maximum: tuple[float, float],
) -> list[tuple[float, float]]:
    return [
        (minimum[0], minimum[1]),
        (maximum[0], minimum[1]),
        (maximum[0], maximum[1]),
        (minimum[0], maximum[1]),
    ]


def _read_box(
    region: Mapping[str, Any],
    bounds: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    minimum = _read_point(region, "min", bounds)
    maximum = _read_point(region, "max", bounds)
    if maximum[0] <= minimum[0] or maximum[1] <= minimum[1]:
        raise CommandError("'max' must be greater than 'min' on both axes")
    return minimum, maximum


def _read_points(
    region: Mapping[str, Any],
    key: str,
    bounds: tuple[int, int],
    *,
    minimum_count: int,
) -> list[tuple[float, float]]:
    raw = region.get(key)
    if not isinstance(raw, list):
        raise CommandError(f"'{key}' must be a list of [x, y] coordinates")
    if len(raw) < minimum_count:
        raise CommandError(f"'{key}' needs at least {minimum_count} coordinates")
    if len(raw) > MAX_POINTS_PER_REGION:
        raise CommandError(
            f"'{key}' has {len(raw)} coordinates; the limit is "
            f"{MAX_POINTS_PER_REGION}"
        )
    return [
        _coerce_point(item, f"{key}[{index}]", bounds)
        for index, item in enumerate(raw)
    ]


def _read_point(
    region: Mapping[str, Any],
    key: str,
    bounds: tuple[int, int],
) -> tuple[float, float]:
    if key not in region:
        raise CommandError(f"Missing '{key}'")
    return _coerce_point(region[key], key, bounds)


def _coerce_point(
    value: Any,
    label: str,
    bounds: tuple[int, int],
) -> tuple[float, float]:
    if isinstance(value, Mapping):
        if set(value) != {"x", "y"}:
            raise CommandError(f"'{label}' object must have exactly 'x' and 'y'")
        pair: Iterable[Any] = (value["x"], value["y"])
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise CommandError(f"'{label}' must contain exactly two numbers")
        pair = value
    else:
        raise CommandError(f"'{label}' must be [x, y] or {{'x': .., 'y': ..}}")

    width, height = bounds
    x, y = (_coerce_number(item, label) for item in pair)
    if not 0.0 <= x <= width:
        raise CommandError(f"'{label}' x={x} is outside 0..{width}")
    if not 0.0 <= y <= height:
        raise CommandError(f"'{label}' y={y} is outside 0..{height}")
    return (x, y)


def _read_number(
    region: Mapping[str, Any],
    key: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if key not in region:
        if default is None:
            raise CommandError(f"Missing '{key}'")
        return default
    number = _coerce_number(region[key], key)
    if minimum is not None and number < minimum:
        raise CommandError(f"'{key}'={number} is below the minimum {minimum}")
    if maximum is not None and number > maximum:
        raise CommandError(f"'{key}'={number} is above the maximum {maximum}")
    return number


def _coerce_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandError(f"'{label}' must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CommandError(f"'{label}' must be finite")
    return number


def _clamp(value: float, lowest: float, highest: float) -> float:
    return max(lowest, min(highest, value))


__all__ = [
    "WRITER_AI",
    "WRITER_HUMAN",
    "WRITER_SOURCES",
    "WRITER_UNSPECIFIED",
    "CATEGORY_IDS",
    "COMMAND_SCHEMA",
    "CONTROL_ACTIONS",
    "LOCK_MINTING_CATEGORIES",
    "OPERATOR_CONTROL_ACTIONS",
    "PAINT_DESCRIPTION",
    "PAINT_DESCRIPTION_WITHOUT_LOCKS",
    "DEFAULT_PRIORITY",
    "CommandError",
    "Control",
    "MAX_COMMANDS_PER_REQUEST",
    "MAX_SEGMENTS_PER_REQUEST",
    "Operation",
    "PAINTABLE_CATEGORIES",
    "REGION_TYPES",
    "RemoveEntity",
    "SYSTEM_PROMPT",
    "Segment",
    "SpawnEntity",
    "TemperatureDelta",
    "VelocityImpulse",
    "compile_request",
    "schema_for_authority",
]
