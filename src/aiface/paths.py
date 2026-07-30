"""Canonical locations for generated artifacts.

Everything AIFace *writes* goes under ``output/`` in the current working
directory, or under ``$AIFACE_OUTPUT`` when that is set. Nothing is written
beside the installed package, so a pip install stays read-only.

The avatar world, its part atlas, and the source portrait are colocated by
name. :func:`aiface.parts.default_parts_path` relies on that contract::

    output/worlds/avatar/avatar_face.bds
    output/worlds/avatar/face_parts.npy    (+ .json anchors, .png preview)
    output/worlds/avatar/source_face.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

OUTPUT: Final = Path(os.environ.get("AIFACE_OUTPUT", "output")).resolve()

WORLDS: Final = OUTPUT / "worlds"
WORLDS_AVATAR: Final = WORLDS / "avatar"
PREVIEWS: Final = OUTPUT / "previews"

DEFAULT_AVATAR_FACE: Final = WORLDS_AVATAR / "avatar_face.bds"
DEFAULT_AVATAR_SOURCE: Final = WORLDS_AVATAR / "source_face.png"


def ensure_output_tree() -> None:
    """Create the documented output subfolders if they are missing."""
    for directory in (WORLDS_AVATAR, PREVIEWS):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "DEFAULT_AVATAR_FACE",
    "DEFAULT_AVATAR_SOURCE",
    "OUTPUT",
    "PREVIEWS",
    "WORLDS",
    "WORLDS_AVATAR",
    "ensure_output_tree",
]
