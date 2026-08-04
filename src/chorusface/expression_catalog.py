"""Expression catalog — fast lookup DB colocated with the avatar BDS.

Learned from ``chorusface-capture`` video/stills. Maps emotion labels and speech
roles onto real plates plus eye/brow parameters. The runtime reads this file
instead of inventing surprise brows or wide eyes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

CATALOG_NAME: Final = "expression_catalog.json"

# Emotion labels the chat / bridge already emit.
EMOTION_ROLE_DEFAULTS: Final[dict[str, str]] = {
    "NEUTRAL": "rest",
    "HAPPY": "smile",
    "JOY": "smile",
    "SURPRISE": "surprise",
    "SURPRISED": "surprise",
    "CURIOUS": "surprise",
    "SAD": "rest",
    "ANGRY": "rest",
    "THINKING": "rest",
}


@dataclass(frozen=True, slots=True)
class ExpressionRole:
    """One named look learned from capture frames."""

    name: str
    plate: str
    frame_index: int = 0
    time_seconds: float = 0.0
    mouth_open: float = 0.0
    smile_width: float = 0.0
    brow_raise: float = 0.0
    eye_widen: float = 0.0
    teeth: float = 0.0
    notes: str = ""


@dataclass(slots=True)
class ExpressionCatalog:
    """In-memory catalog loaded from disk."""

    version: str = "expression-catalog-1.0"
    source: str = ""
    roles: dict[str, ExpressionRole] = field(default_factory=dict)
    emotion_map: dict[str, str] = field(default_factory=dict)
    viseme_openness: dict[str, float] = field(default_factory=dict)

    def role_for_emotion(self, emotion: str) -> ExpressionRole | None:
        key = (emotion or "NEUTRAL").strip().upper()
        role_name = self.emotion_map.get(key) or EMOTION_ROLE_DEFAULTS.get(key, "rest")
        return self.roles.get(role_name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "roles": {name: asdict(role) for name, role in self.roles.items()},
            "emotion_map": dict(self.emotion_map),
            "viseme_openness": dict(self.viseme_openness),
        }


def default_catalog_path(world: str | Path) -> Path:
    return Path(world).with_name(CATALOG_NAME)


def write_expression_catalog(path: str | Path, catalog: ExpressionCatalog) -> Path:
    destination = Path(path)
    destination.write_text(json.dumps(catalog.as_dict(), indent=2), encoding="utf-8")
    return destination


def load_expression_catalog(world: str | Path) -> ExpressionCatalog | None:
    path = default_catalog_path(world)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    roles_raw = payload.get("roles") or {}
    roles: dict[str, ExpressionRole] = {}
    if isinstance(roles_raw, Mapping):
        for name, item in roles_raw.items():
            if not isinstance(item, Mapping):
                continue
            roles[str(name)] = ExpressionRole(
                name=str(item.get("name", name)),
                plate=str(item.get("plate", "")),
                frame_index=int(item.get("frame_index", 0)),
                time_seconds=float(item.get("time_seconds", 0.0)),
                mouth_open=float(item.get("mouth_open", 0.0)),
                smile_width=float(item.get("smile_width", 0.0)),
                brow_raise=float(item.get("brow_raise", 0.0)),
                eye_widen=float(item.get("eye_widen", 0.0)),
                teeth=float(item.get("teeth", 0.0)),
                notes=str(item.get("notes", "")),
            )
    emotion_map = {
        str(k).upper(): str(v)
        for k, v in (payload.get("emotion_map") or EMOTION_ROLE_DEFAULTS).items()
    }
    viseme = {
        str(k): float(v) for k, v in (payload.get("viseme_openness") or {}).items()
    }
    return ExpressionCatalog(
        version=str(payload.get("version", "expression-catalog-1.0")),
        source=str(payload.get("source", "")),
        roles=roles,
        emotion_map=emotion_map,
        viseme_openness=viseme,
    )


__all__ = [
    "CATALOG_NAME",
    "EMOTION_ROLE_DEFAULTS",
    "ExpressionCatalog",
    "ExpressionRole",
    "default_catalog_path",
    "load_expression_catalog",
    "write_expression_catalog",
]
