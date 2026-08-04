"""Product window sizing — resizable, aspect locked.

Default composition is portrait square + bottom chat band (1024×1320).
Face-only mode (no chat panel) uses a square window.
"""

from __future__ import annotations

from typing import Final

# Portrait + chat band (matches AvatarFaceApp defaults).
DEFAULT_WINDOW_WIDTH: Final = 1024
DEFAULT_WINDOW_HEIGHT: Final = 1320
DEFAULT_WINDOW_SIZE: Final = (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
DEFAULT_ASPECT: Final = DEFAULT_WINDOW_WIDTH / DEFAULT_WINDOW_HEIGHT  # ~0.7758

FACE_ONLY_ASPECT: Final = 1.0
FACE_ONLY_WINDOW_SIZE: Final = (1024, 1024)

MIN_WINDOW_SIDE: Final = 480
MAX_WINDOW_SIDE: Final = 2400


def aspect_for_layout(*, chat_box_visible: bool) -> float:
    return DEFAULT_ASPECT if chat_box_visible else FACE_ONLY_ASPECT


def default_window_size(*, chat_box_visible: bool) -> tuple[int, int]:
    return DEFAULT_WINDOW_SIZE if chat_box_visible else FACE_ONLY_WINDOW_SIZE


def snap_size_to_aspect(
    width: int,
    height: int,
    aspect: float,
    *,
    prev_width: int | None = None,
    prev_height: int | None = None,
    min_side: int = MIN_WINDOW_SIDE,
    max_side: int = MAX_WINDOW_SIDE,
) -> tuple[int, int]:
    """Return a size with the given width/height ratio.

    Chooses the candidate (from width or from height) closer to the requested
    size. When ``prev_*`` is set, prefer the axis that moved more so edge drags
    feel natural. Clamps the longest side into ``[min_side, max_side]``.
    """
    aspect = float(aspect)
    if aspect <= 1e-6:
        aspect = DEFAULT_ASPECT
    width = max(1, int(width))
    height = max(1, int(height))
    min_side = max(64, int(min_side))
    max_side = max(min_side, int(max_side))

    from_w = (width, max(1, int(round(width / aspect))))
    from_h = (max(1, int(round(height * aspect))), height)

    prefer_width: bool | None = None
    if prev_width is not None and prev_height is not None:
        dw = abs(width - int(prev_width))
        dh = abs(height - int(prev_height))
        if dw > dh + 1:
            prefer_width = True
        elif dh > dw + 1:
            prefer_width = False

    if prefer_width is True:
        w, h = from_w
    elif prefer_width is False:
        w, h = from_h
    else:
        # Corner drag / unknown — pick nearer candidate.
        err_w = abs(from_w[0] - width) + abs(from_w[1] - height)
        err_h = abs(from_h[0] - width) + abs(from_h[1] - height)
        w, h = from_w if err_w <= err_h else from_h

    long_side = max(w, h)
    if long_side < min_side:
        scale = min_side / max(long_side, 1)
        w = max(1, int(round(w * scale)))
        h = max(1, int(round(h * scale)))
    elif long_side > max_side:
        scale = max_side / long_side
        w = max(1, int(round(w * scale)))
        h = max(1, int(round(h * scale)))

    # Exact re-snap after clamp — keep the preferred axis stable.
    if prefer_width is False:
        w = max(1, int(round(h * aspect)))
    else:
        h = max(1, int(round(w / aspect)))
    return max(1, w), max(1, h)


__all__ = [
    "DEFAULT_ASPECT",
    "DEFAULT_WINDOW_HEIGHT",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_WINDOW_WIDTH",
    "FACE_ONLY_ASPECT",
    "FACE_ONLY_WINDOW_SIZE",
    "MAX_WINDOW_SIDE",
    "MIN_WINDOW_SIDE",
    "aspect_for_layout",
    "default_window_size",
    "snap_size_to_aspect",
]
