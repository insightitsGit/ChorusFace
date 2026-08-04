"""User cosmetic prefs on top of locked TickFeed scaffolding.

Cosmetics never rewrite identity albedo or Master Lock geometry.
Stored beside the world as ``cosmetic_prefs.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

PREFS_NAME: Final = "cosmetic_prefs.json"
PREFS_VERSION: Final = "chorusface.cosmetic_prefs.v1"


@dataclass
class CosmeticPrefs:
    skin_tint_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    eye_tint_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    makeup_strength: float = 0.0
    style_lut: str = ""

    def clamp(self) -> CosmeticPrefs:
        def _c(v: tuple[float, float, float]) -> tuple[float, float, float]:
            return (
                float(max(0.0, min(2.0, v[0]))),
                float(max(0.0, min(2.0, v[1]))),
                float(max(0.0, min(2.0, v[2]))),
            )

        return CosmeticPrefs(
            skin_tint_rgb=_c(self.skin_tint_rgb),
            eye_tint_rgb=_c(self.eye_tint_rgb),
            makeup_strength=float(max(0.0, min(1.0, self.makeup_strength))),
            style_lut=str(self.style_lut or "")[:64],
        )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self.clamp())
        d["schema"] = PREFS_VERSION
        return d

    def shader_uniforms(self) -> dict[str, float | tuple[float, float, float]]:
        c = self.clamp()
        return {
            "u_skin_tint": c.skin_tint_rgb,
            "u_eye_tint": c.eye_tint_rgb,
            "u_makeup_strength": c.makeup_strength,
        }


def default_prefs() -> CosmeticPrefs:
    return CosmeticPrefs()


def write_cosmetic_prefs(world: Path | str, prefs: CosmeticPrefs | None = None) -> Path:
    root = Path(world)
    root = root if root.is_dir() else root.parent
    path = root / PREFS_NAME
    payload = (prefs or default_prefs()).as_dict()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_cosmetic_prefs(world: Path | str) -> CosmeticPrefs:
    root = Path(world)
    root = root if root.is_dir() else root.parent
    path = root / PREFS_NAME
    if not path.is_file():
        write_cosmetic_prefs(root)
        return default_prefs()
    raw = json.loads(path.read_text(encoding="utf-8"))
    skin = raw.get("skin_tint_rgb") or [1.0, 1.0, 1.0]
    eye = raw.get("eye_tint_rgb") or [1.0, 1.0, 1.0]
    return CosmeticPrefs(
        skin_tint_rgb=(float(skin[0]), float(skin[1]), float(skin[2])),
        eye_tint_rgb=(float(eye[0]), float(eye[1]), float(eye[2])),
        makeup_strength=float(raw.get("makeup_strength") or 0.0),
        style_lut=str(raw.get("style_lut") or ""),
    ).clamp()


__all__ = [
    "CosmeticPrefs",
    "PREFS_NAME",
    "default_prefs",
    "load_cosmetic_prefs",
    "write_cosmetic_prefs",
]
