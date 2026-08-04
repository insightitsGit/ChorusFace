"""Minimal GPU field substrate the avatar renders on.

Only the parts a talking face needs: ``.bds`` world I/O, validated GPU command
rows, generated GLSL preludes, and a constraint-only render window.
"""

from chorusface.runtime.bds import (
    HARD_SURFACE_CHANNEL,
    HUMAN_LOCK_CHANNEL,
    PRIORITY_LEVELS,
    TICK_RATE_HZ,
    BDSFormatError,
    load_bds,
    save_bds,
)
from chorusface.runtime.commands import PaintCommand
from chorusface.runtime.field import FieldRuntime, encode_png

__all__ = [
    "HARD_SURFACE_CHANNEL",
    "HUMAN_LOCK_CHANNEL",
    "PRIORITY_LEVELS",
    "TICK_RATE_HZ",
    "BDSFormatError",
    "FieldRuntime",
    "PaintCommand",
    "encode_png",
    "load_bds",
    "save_bds",
]
