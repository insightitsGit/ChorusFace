"""Generate a tennis racket and ball as a Neural World Runtime command file.

The command grammar has no ellipse primitive, so the racket head is emitted as a
polygon outline sampled from an ellipse, and the strings are clipped to that
ellipse so they stop at the frame instead of poking through it.

    python tools/make_tennis_racket.py --output examples/tennis-racket.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Final, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_commands import compile_request  # noqa: E402

HEAD_CENTER: Final = (100.0, 168.0)
HEAD_RADIUS_X: Final = 36.0
HEAD_RADIUS_Y: Final = 46.0
HEAD_SEGMENTS: Final = 36
FRAME_THICKNESS: Final = 3
STRING_INSET: Final = 0.88
BALL_CENTER: Final = (196.0, 206.0)
BALL_RADIUS: Final = 13.0
CANVAS_MIN: Final = (18, 18)
CANVAS_MAX: Final = (238, 238)


def rounded(x: float, y: float) -> list[int]:
    return [int(round(x)), int(round(y))]


def head_outline() -> list[list[int]]:
    cx, cy = HEAD_CENTER
    return [
        rounded(
            cx + HEAD_RADIUS_X * math.cos(2.0 * math.pi * index / HEAD_SEGMENTS),
            cy + HEAD_RADIUS_Y * math.sin(2.0 * math.pi * index / HEAD_SEGMENTS),
        )
        for index in range(HEAD_SEGMENTS)
    ]


def half_span(offset: float, along: float, across: float) -> float:
    """Half-chord of the ellipse at ``offset`` along one axis."""
    ratio = min(abs(offset) / along, 1.0)
    return across * math.sqrt(max(0.0, 1.0 - ratio * ratio))


def string_commands() -> list[dict[str, Any]]:
    cx, cy = HEAD_CENTER
    commands: list[dict[str, Any]] = []

    for offset in (-24.0, -12.0, 0.0, 12.0, 24.0):
        reach = half_span(offset, HEAD_RADIUS_X, HEAD_RADIUS_Y) * STRING_INSET
        commands.append(
            {
                "action": "paint",
                "category": "solid",
                "region": {
                    "type": "line",
                    "start": rounded(cx + offset, cy - reach),
                    "end": rounded(cx + offset, cy + reach),
                    "thickness": 1,
                },
            }
        )

    for offset in (-30.0, -15.0, 0.0, 15.0, 30.0):
        reach = half_span(offset, HEAD_RADIUS_Y, HEAD_RADIUS_X) * STRING_INSET
        commands.append(
            {
                "action": "paint",
                "category": "solid",
                "region": {
                    "type": "line",
                    "start": rounded(cx - reach, cy + offset),
                    "end": rounded(cx + reach, cy + offset),
                    "thickness": 1,
                },
            }
        )
    return commands


def build_commands(*, clear_canvas: bool = True) -> dict[str, Any]:
    cx, cy = HEAD_CENTER
    head_bottom = cy - HEAD_RADIUS_Y
    commands: list[dict[str, Any]] = []

    if clear_canvas:
        commands.append(
            {
                "action": "erase",
                "region": {
                    "type": "rectangle",
                    "min": list(CANVAS_MIN),
                    "max": list(CANVAS_MAX),
                },
            }
        )

    commands.append(
        {
            "action": "paint",
            "category": "solid",
            "region": {
                "type": "polygon_outline",
                "points": head_outline(),
                "thickness": FRAME_THICKNESS,
            },
        }
    )
    commands.extend(string_commands())

    # Throat: two struts from the frame down to the top of the handle.
    for side in (-1.0, 1.0):
        commands.append(
            {
                "action": "paint",
                "category": "solid",
                "region": {
                    "type": "line",
                    "start": rounded(cx + side * 20.0, head_bottom + 8.0),
                    "end": rounded(cx, head_bottom - 14.0),
                    "thickness": 3,
                },
            }
        )

    commands.append(
        {
            "action": "paint",
            "category": "solid",
            "region": {
                "type": "line",
                "start": rounded(cx, head_bottom - 12.0),
                "end": rounded(cx, 46.0),
                "thickness": 4,
            },
        }
    )
    # Grip is a fatter stub at the base of the handle.
    commands.append(
        {
            "action": "paint",
            "category": "solid",
            "region": {
                "type": "line",
                "start": rounded(cx, 78.0),
                "end": rounded(cx, 46.0),
                "thickness": 8,
            },
        }
    )

    commands.append(
        {
            "action": "paint",
            "category": "active_fluid",
            "region": {
                "type": "circle",
                "center": rounded(*BALL_CENTER),
                "radius": int(BALL_RADIUS),
            },
        }
    )
    # A brighter seam so the ball reads as a ball, not a dot.
    commands.append(
        {
            "action": "paint",
            "category": "solid",
            "region": {
                "type": "ring",
                "center": rounded(*BALL_CENTER),
                "radius": int(BALL_RADIUS) - 4,
                "thickness": 1,
            },
        }
    )
    return {"commands": commands}


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/tennis-racket.json"),
        help="Where to write the command file",
    )
    parser.add_argument(
        "--keep-background",
        action="store_true",
        help="Draw over the existing world instead of clearing a canvas first",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    payload = build_commands(clear_canvas=not arguments.keep_background)

    operations = compile_request(payload)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {arguments.output}: {len(payload['commands'])} commands "
        f"-> {len(operations)} GPU operations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
