"""Step 6 — Condition looks + word / sound / emotion maps.

Known words/visemes use tables; unknowns are covered by the live-vector model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from aiface.biomechanics.intent import PHONEME_JAW_TARGET
from aiface.speech import EMOTION_IMPULSES, canonical_viseme, mouth_pose

MAPPING_NAME: Final = "condition_maps.json"

# Open vowels that must keep a floor under ML (never pull shut).
OPEN_SOUNDS: Final = frozenset({"AA", "AH", "OH", "OU", "AW", "AE"})


def build_condition_maps() -> dict[str, Any]:
    visemes: dict[str, dict[str, float]] = {}
    for key in (
        "REST",
        "PP",
        "FF",
        "TH",
        "DD",
        "KK",
        "CH",
        "SS",
        "NN",
        "RR",
        "AA",
        "E",
        "I",
        "O",
        "U",
        "AH",
        "OH",
        "OU",
    ):
        canon = canonical_viseme(key)
        pose = mouth_pose(canon, "NEUTRAL")
        visemes[canon] = {
            "width": float(pose.width),
            "openness": float(pose.openness),
            "roundness": float(pose.roundness),
            # Jaw table the runtime consumes at play (words own jaw timing).
            "jaw": float(PHONEME_JAW_TARGET.get(canon, 0.1)),
            "open_floor": canon in OPEN_SOUNDS or float(pose.openness) >= 0.55,
        }
    emotions = {
        name: {
            "expression": float(mouth_pose("REST", name).expression),
            "impulse": list(EMOTION_IMPULSES.get(name, (0.0, 0.0))),
        }
        for name in sorted(EMOTION_IMPULSES)
    }
    return {
        "schema": "amin_loop.mapping.v1",
        "viseme_table": visemes,
        "emotion_table": emotions,
        "policy": {
            "known_open_vowels_floor": sorted(OPEN_SOUNDS),
            "unknown_sounds": "live_vector_model",
            "identity": "master_lock_ch31",
        },
    }


def write_condition_maps(world_dir: Path) -> Path:
    world_dir = Path(world_dir)
    world_dir.mkdir(parents=True, exist_ok=True)
    path = world_dir / MAPPING_NAME
    path.write_text(json.dumps(build_condition_maps(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "MAPPING_NAME",
    "OPEN_SOUNDS",
    "build_condition_maps",
    "write_condition_maps",
]
