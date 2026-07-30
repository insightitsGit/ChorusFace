"""Step 3 — Control API: propose → validate → GPU command rows.

Wraps the same command contract as ``aiface.runtime.commands`` / vendor NWR:
AI may emit ±4 velocity impulses; Master Lock (ch 31) rejects identity writes
on the GPU. This module never invents face RGB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from aiface.runtime.bds import PRIORITY_LEVELS
from aiface.runtime.commands import SOURCE_AI, PaintCommand


@dataclass(frozen=True, slots=True)
class VelocityImpulse:
    """Validated mouth-local velocity proposal (grid space)."""

    x: float
    y: float
    vx: float
    vy: float
    radius: float
    tick: int = 0


def validate_impulse(
    impulse: VelocityImpulse,
    *,
    grid_width: int,
    grid_height: int,
    max_speed: float = 4.0,
    max_radius: float = 32.0,
) -> VelocityImpulse:
    """Clamp a proposed impulse into the legal AI write envelope."""
    x = max(0.0, min(float(grid_width - 1), float(impulse.x)))
    y = max(0.0, min(float(grid_height - 1), float(impulse.y)))
    speed = (float(impulse.vx) ** 2 + float(impulse.vy) ** 2) ** 0.5
    scale = 1.0
    if speed > max_speed and speed > 1e-9:
        scale = max_speed / speed
    vx = float(impulse.vx) * scale
    vy = float(impulse.vy) * scale
    radius = max(0.5, min(float(max_radius), float(impulse.radius)))
    return VelocityImpulse(
        x=x, y=y, vx=vx, vy=vy, radius=radius, tick=int(impulse.tick)
    )


def impulses_to_commands(
    impulses: Sequence[VelocityImpulse],
    *,
    grid_width: int,
    grid_height: int,
) -> list[PaintCommand]:
    """Convert validated impulses into AI-sourced ±4 PaintCommands."""
    rows: list[PaintCommand] = []
    for raw in impulses:
        impulse = validate_impulse(
            raw, grid_width=grid_width, grid_height=grid_height
        )
        rows.append(
            PaintCommand(
                tick=max(1, int(impulse.tick)),
                start_x=impulse.x,
                start_y=impulse.y,
                end_x=impulse.vx,
                end_y=impulse.vy,
                radius=impulse.radius,
                category=0,
                operation=1.0,
                priority=PRIORITY_LEVELS["ai"],
                source=SOURCE_AI,
                velocity_impulse=(impulse.vx, impulse.vy),
            )
        )
    return rows


def neighbor_offsets(*, include_diag: bool = True) -> tuple[tuple[int, int], ...]:
    """2D cell neighbor offsets (Moore or 4-connected)."""
    if include_diag:
        return tuple(
            (dx, dy)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        )
    return ((-1, 0), (1, 0), (0, -1), (0, 1))


__all__ = [
    "VelocityImpulse",
    "impulses_to_commands",
    "neighbor_offsets",
    "validate_impulse",
]
