"""Step 2 — 32-float cell properties (NWR schema, shared with chorusface.runtime)."""

from __future__ import annotations

from typing import Final

# Prefer chorusface runtime schema (same contract as vendor/nwr).
try:
    from chorusface.runtime.bds import CHANNEL_NAMES, CHANNEL_SCHEMA, HUMAN_LOCK_CHANNEL
except ImportError:  # pragma: no cover
    CHANNEL_SCHEMA = {
        "kinematics": [
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "density",
            "pressure",
            "shear",
            "temperature",
            "energy",
        ],
        "material": [
            "albedo_r",
            "albedo_g",
            "albedo_b",
            "opacity",
            "roughness",
            "metallic",
            "emission",
            "refraction",
        ],
        "intent": [
            "attraction",
            "alignment",
            "user_affinity",
            "growth",
            "decay",
            "lifespan",
            "reserved_22",
            "reserved_23",
        ],
        "rules": [
            "hard_surface",
            "permeability",
            "thermal_threshold",
            "phase_trigger",
            "reserved_28",
            "reserved_29",
            "authority_priority",
            "human_lock",
        ],
    }
    CHANNEL_NAMES = [n for g in CHANNEL_SCHEMA.values() for n in g]
    HUMAN_LOCK_CHANNEL = 31

CELL_GROUPS: Final = CHANNEL_SCHEMA
VECTOR_DIM: Final = 32
assert len(CHANNEL_NAMES) == VECTOR_DIM

__all__ = [
    "CELL_GROUPS",
    "CHANNEL_NAMES",
    "HUMAN_LOCK_CHANNEL",
    "VECTOR_DIM",
]
