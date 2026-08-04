"""Deterministic grid-space commands scheduled onto simulation ticks.

A command is eight floats in an SSBO row. The operation magnitude marks the
writer and the kind, which is how the constraint shader decides whether a write
is allowed at all:

===========  ==========================================
``±1``       human paint / erase
``±2``       AI paint / erase
``±3``       temperature delta
``±4``       velocity impulse, ``(V_x, V_y)`` in a disc
===========  ==========================================

The avatar only ever emits ``±4``: speech moves soft tissue by adding velocity
inside the unlocked mouth region. Master-Locked cells reject AI writes on the
GPU, so identity cannot be overwritten by a bad command.

Only ``±1``/``±2`` encode the writer in the opcode magnitude. The shader gates
``±3`` and ``±4`` as AI writes whatever their sign, so temperature and velocity
can never reach a locked cell from either writer. ``source`` still has to be
right, because it decides both the authority a row claims in ``settings.w`` and
the order rows are packed in: writer and priority travelling separately is what
let a replayed AI erase acquire human authority in the parent runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from chorusface.runtime.bds import PRIORITY_LEVELS, PRIORITY_NAMES
from chorusface.runtime.shaders import normalize_priority

SOURCE_HUMAN: Final = 0
SOURCE_AI: Final = 1
#: The strongest authority an AI-sourced row may claim.
MAX_AI_PRIORITY: Final = PRIORITY_LEVELS["ai"]


@dataclass(frozen=True, slots=True)
class PaintCommand:
    """One validated command row bound for the GPU command buffer."""

    tick: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    radius: float
    category: int
    operation: float
    priority: int = PRIORITY_LEVELS["user"]
    source: int = SOURCE_HUMAN
    temperature_delta: float | None = None
    # When set, segment.zw carries (V_x, V_y) and operation becomes ±4.
    velocity_impulse: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.source not in (SOURCE_HUMAN, SOURCE_AI):
            raise ValueError(f"Unknown command source: {self.source}")
        if self.priority not in PRIORITY_NAMES:
            raise ValueError(f"Unknown priority level: {self.priority}")
        if self.source == SOURCE_AI and self.priority > MAX_AI_PRIORITY:
            raise ValueError(
                "An AI-sourced command may not claim "
                f"{PRIORITY_NAMES[self.priority]!r} authority; the strongest it "
                f"may hold is {PRIORITY_NAMES[MAX_AI_PRIORITY]!r}"
            )
        if self.temperature_delta is not None and self.velocity_impulse is not None:
            raise ValueError(
                "A command carries one payload: temperature or velocity, not both"
            )

    @property
    def is_ai(self) -> bool:
        return self.source == SOURCE_AI

    def as_row(self) -> tuple[float, ...]:
        if self.velocity_impulse is not None:
            vx, vy = self.velocity_impulse
            signed = -4.0 if self.is_ai else 4.0
            return (
                self.start_x,
                self.start_y,
                float(vx),
                float(vy),
                self.radius,
                0.0,
                signed,
                normalize_priority(self.priority),
            )
        if self.temperature_delta is not None:
            signed = 3.0 if self.temperature_delta >= 0.0 else -3.0
            return (
                self.start_x,
                self.start_y,
                self.end_x,
                self.end_y,
                self.radius,
                abs(float(self.temperature_delta)),
                signed,
                normalize_priority(self.priority),
            )
        magnitude = 2.0 if self.is_ai else 1.0
        signed = -magnitude if self.operation < 0.0 else magnitude
        return (
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
            self.radius,
            float(self.category),
            signed,
            normalize_priority(self.priority),
        )


__all__ = ["MAX_AI_PRIORITY", "SOURCE_AI", "SOURCE_HUMAN", "PaintCommand"]
