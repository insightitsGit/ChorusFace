"""Canonical locations for generated artifacts.

Source lives at the repository root. Everything a tool *writes* goes under
``output/`` with a name that says what produced it. See ``output/README.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent
OUTPUT: Final = ROOT / "output"

WORLDS: Final = OUTPUT / "worlds"
WORLDS_FROM_VIDEO: Final = WORLDS / "from-video"
WORLDS_PLAYGROUND: Final = WORLDS / "playground"
WORLDS_AVATAR: Final = WORLDS / "avatar"
PREVIEWS: Final = OUTPUT / "previews"
HANDOFFS: Final = OUTPUT / "handoffs"
RENDERS: Final = OUTPUT / "renders"
SESSIONS: Final = OUTPUT / "sessions"
REPORTS: Final = OUTPUT / "reports"

DEFAULT_WORLD: Final = WORLDS_PLAYGROUND / "world.bds"
DEFAULT_VIDEO_WORLD: Final = WORLDS_FROM_VIDEO / "game_world.bds"
DEFAULT_AVATAR_FACE: Final = WORLDS_AVATAR / "avatar_face.bds"
DEFAULT_HANDOFF: Final = HANDOFFS / "bundle"
DEFAULT_RENDER: Final = RENDERS / "frames"
DEFAULT_SESSION: Final = SESSIONS / "demo"


def ensure_output_tree() -> None:
    """Create the documented output subfolders if they are missing."""
    for directory in (
        WORLDS_FROM_VIDEO,
        WORLDS_PLAYGROUND,
        WORLDS_AVATAR,
        PREVIEWS,
        HANDOFFS,
        RENDERS,
        SESSIONS,
        REPORTS,
    ):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "DEFAULT_AVATAR_FACE",
    "DEFAULT_HANDOFF",
    "DEFAULT_RENDER",
    "DEFAULT_SESSION",
    "DEFAULT_VIDEO_WORLD",
    "DEFAULT_WORLD",
    "HANDOFFS",
    "OUTPUT",
    "PREVIEWS",
    "REPORTS",
    "RENDERS",
    "ROOT",
    "SESSIONS",
    "WORLDS",
    "WORLDS_AVATAR",
    "WORLDS_FROM_VIDEO",
    "WORLDS_PLAYGROUND",
    "ensure_output_tree",
]
