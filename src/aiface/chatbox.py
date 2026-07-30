"""The in-window chat panel: editing state, transcript, and how it is painted.

Typing into the GL window rather than a terminal is what makes the face feel
answered rather than scripted — the reply lands beside the portrait while the
muscles are still moving. All the state here is ordinary Python with no GL or
PIL import at module scope, so the editing rules are testable headlessly and
the painting stays optional.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Final

MAX_INPUT_CHARS: Final = 400
MAX_ENTRIES: Final = 200
CARET_BLINK_SECONDS: Final = 1.06

SPEAKER_YOU: Final = "you"
SPEAKER_FACE: Final = "face"
SPEAKER_SYSTEM: Final = "system"

# Panel chrome, in the same linear-ish palette as the runtime's clear colour.
PANEL_FILL: Final = (9, 14, 26, 232)
PANEL_EDGE: Final = (38, 78, 108, 255)
INPUT_FILL: Final = (5, 9, 18, 245)
TITLE_COLOR: Final = (108, 150, 178, 255)
CONTROL_FILL: Final = (16, 28, 44, 245)
CONTROL_EDGE: Final = (70, 130, 170, 255)
CONTROL_TEXT: Final = (210, 230, 245, 255)
CONTROL_ACTIVE: Final = (120, 210, 255, 255)
SPEAKER_COLORS: Final[dict[str, tuple[int, int, int, int]]] = {
    SPEAKER_YOU: (150, 214, 255, 255),
    SPEAKER_FACE: (170, 255, 214, 255),
    SPEAKER_SYSTEM: (200, 170, 120, 255),
}


_GUTTERS: Final[dict[str, str]] = {
    SPEAKER_YOU: "you  ",
    SPEAKER_FACE: "face ",
    SPEAKER_SYSTEM: "     ",
}


@dataclass(frozen=True, slots=True)
class ChatEntry:
    """One line of transcript."""

    speaker: str
    text: str


class ChatBox:
    """Single-line input with history, plus a scrolling transcript.

    The input is deliberately a one-line editor that scrolls horizontally
    instead of wrapping. A wrapping editor has to reconcile caret position with
    a layout that changes as you type, and none of that earns anything for
    utterances a face is going to speak in a breath or two.
    """

    def __init__(
        self,
        *,
        max_entries: int = MAX_ENTRIES,
        max_input: int = MAX_INPUT_CHARS,
    ) -> None:
        self.entries: deque[ChatEntry] = deque(maxlen=max(int(max_entries), 1))
        self.max_input = max(int(max_input), 1)
        self.text = ""
        self.caret = 0
        self.focused = True
        self.pending = False
        self.history: list[str] = []
        self._history_index: int | None = None
        self._draft = ""

    # ------------------------------------------------------------- editing

    def insert(self, characters: str) -> bool:
        """Insert printable text at the caret. Returns whether anything changed."""
        clean = "".join(
            character
            for character in characters
            if character.isprintable() and character not in "\r\n\t"
        )
        if not clean:
            return False
        room = self.max_input - len(self.text)
        if room <= 0:
            return False
        clean = clean[:room]
        self.text = self.text[: self.caret] + clean + self.text[self.caret :]
        self.caret += len(clean)
        return True

    def backspace(self) -> bool:
        if self.caret <= 0:
            return False
        self.text = self.text[: self.caret - 1] + self.text[self.caret :]
        self.caret -= 1
        return True

    def delete(self) -> bool:
        if self.caret >= len(self.text):
            return False
        self.text = self.text[: self.caret] + self.text[self.caret + 1 :]
        return True

    def delete_word(self) -> bool:
        """Ctrl+Backspace: drop trailing spaces, then the word before the caret."""
        if self.caret <= 0:
            return False
        cut = self.caret
        while cut > 0 and self.text[cut - 1].isspace():
            cut -= 1
        while cut > 0 and not self.text[cut - 1].isspace():
            cut -= 1
        self.text = self.text[:cut] + self.text[self.caret :]
        self.caret = cut
        return True

    def move_caret(self, delta: int) -> bool:
        target = max(0, min(len(self.text), self.caret + int(delta)))
        if target == self.caret:
            return False
        self.caret = target
        return True

    def caret_to_start(self) -> bool:
        return self.move_caret(-len(self.text))

    def caret_to_end(self) -> bool:
        return self.move_caret(len(self.text))

    def clear_input(self) -> bool:
        if not self.text:
            return False
        self.text = ""
        self.caret = 0
        self._history_index = None
        return True

    def submit(self) -> str | None:
        """Take the typed line, record it in history, and empty the editor."""
        spoken = self.text.strip()
        self.text = ""
        self.caret = 0
        self._history_index = None
        self._draft = ""
        if not spoken:
            return None
        if not self.history or self.history[-1] != spoken:
            self.history.append(spoken)
        return spoken

    # ------------------------------------------------------------- history

    def recall_previous(self) -> bool:
        if not self.history:
            return False
        if self._history_index is None:
            self._draft = self.text
            self._history_index = len(self.history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return False
        self._apply_history(self.history[self._history_index])
        return True

    def recall_next(self) -> bool:
        if self._history_index is None:
            return False
        if self._history_index < len(self.history) - 1:
            self._history_index += 1
            self._apply_history(self.history[self._history_index])
        else:
            self._history_index = None
            self._apply_history(self._draft)
        return True

    def _apply_history(self, value: str) -> None:
        self.text = value[: self.max_input]
        self.caret = len(self.text)

    # ---------------------------------------------------------- transcript

    def add(self, speaker: str, text: str) -> None:
        cleaned = " ".join(str(text).split())
        if cleaned:
            self.entries.append(ChatEntry(speaker=speaker, text=cleaned))

    def transcript_lines(
        self, columns: int, rows: int
    ) -> list[tuple[str, str]]:
        """Wrap the tail of the transcript into at most ``rows`` display lines."""
        columns = max(int(columns), 8)
        rows = max(int(rows), 1)
        # Every line carries a fixed-width speaker gutter, so wrap the body to
        # the remaining width and indent continuations into that same gutter.
        gutter = 5
        body_columns = max(columns - gutter, 4)
        lines: list[tuple[str, str]] = []
        for entry in self.entries:
            prefix = _GUTTERS.get(entry.speaker, " " * gutter)
            for index, chunk in enumerate(_wrap(entry.text, body_columns)):
                head = prefix if index == 0 else " " * gutter
                lines.append((entry.speaker, head + chunk))
        return lines[-rows:]

    def input_view(self, columns: int) -> tuple[str, int]:
        """The visible slice of the input line plus the caret's column in it.

        Scrolls to keep the caret in view so a long sentence stays editable in
        a panel narrower than the text.
        """
        columns = max(int(columns), 8)
        if len(self.text) < columns:
            return self.text, self.caret
        start = max(0, min(self.caret - columns + 1, len(self.text) - columns + 1))
        return self.text[start : start + columns], self.caret - start

    def caret_visible(self, now: float) -> bool:
        if not self.focused:
            return False
        return (float(now) % CARET_BLINK_SECONDS) < (CARET_BLINK_SECONDS * 0.55)

    def status_text(self) -> str:
        if self.pending:
            return "thinking…"
        if not self.focused:
            return "Esc: type"
        return "Enter: speak"


def _wrap(text: str, columns: int) -> list[str]:
    """Greedy word wrap that never drops a long unbroken token."""
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        while len(word) > columns:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:columns])
            word = word[columns:]
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= columns:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current or not lines:
        lines.append(current)
    return lines


def measure_columns(font: Any, width: int) -> int:
    """How many monospace-ish characters fit across ``width`` pixels."""
    try:
        advance = float(font.getlength("0"))
    except AttributeError:  # very old Pillow
        advance = float(font.getbbox("0")[2])
    return max(int(width / max(advance, 1.0)), 8)


def paint_panel(
    draw: Any,
    box: ChatBox,
    *,
    rect: tuple[int, int, int, int],
    font: Any,
    title_font: Any,
    now: float,
    title: str = "chat",
    mouth_speed_label: str | None = None,
    mouth_menu_open: bool = False,
    mouth_options: tuple[str, ...] = (),
) -> dict[str, tuple[int, int, int, int]]:
    """Draw the chat frame into an RGBA overlay.

    ``rect`` is ``(x, y, width, height)`` in top-down overlay pixels, matching
    the HUD texture's orientation. Returns hit rectangles for UI controls
    (``mouth_button`` and ``mouth_opt_<label>``) in the same pixel space.
    """
    hits: dict[str, tuple[int, int, int, int]] = {}
    x, y, width, height = (int(value) for value in rect)
    if width <= 24 or height <= 48:
        return hits

    line_height = _line_height(font)
    padding = 12
    input_height = line_height + 14

    draw.rectangle((x, y, x + width, y + height), fill=PANEL_FILL)
    draw.line((x, y, x + width, y), fill=PANEL_EDGE, width=2)

    header = f"{title}   {box.status_text()}"
    draw.text((x + padding, y + 6), header, fill=TITLE_COLOR, font=title_font)
    header_height = _line_height(title_font) + 8

    if mouth_speed_label:
        hits.update(
            _paint_mouth_speed_dropdown(
                draw,
                panel_x=x,
                panel_y=y,
                panel_width=width,
                font=title_font,
                label=mouth_speed_label,
                open_menu=mouth_menu_open,
                options=mouth_options,
                padding=padding,
            )
        )

    transcript_top = y + 6 + header_height
    input_top = y + height - input_height - 6
    transcript_height = input_top - transcript_top - 6
    rows = max(int(transcript_height // line_height), 1)
    columns = measure_columns(font, width - padding * 2)

    lines = box.transcript_lines(columns, rows)
    # Grow upward from the input line: the newest turn belongs next to where
    # you are typing, not stranded at the top of an empty panel.
    cursor_y = max(transcript_top, input_top - 6 - len(lines) * line_height)
    for speaker, line in lines:
        draw.text(
            (x + padding, cursor_y),
            line,
            fill=SPEAKER_COLORS.get(speaker, SPEAKER_COLORS[SPEAKER_SYSTEM]),
            font=font,
        )
        cursor_y += line_height
    draw.rectangle(
        (x + padding // 2, input_top, x + width - padding // 2, input_top + input_height),
        fill=INPUT_FILL,
    )
    prompt = "> "
    visible, caret_column = box.input_view(columns - len(prompt))
    draw.text(
        (x + padding, input_top + 7),
        prompt + visible,
        fill=(226, 240, 255, 255) if box.focused else (128, 148, 166, 255),
        font=font,
    )
    if box.caret_visible(now):
        caret_x = x + padding + _advance(font, prompt + visible[:caret_column])
        draw.rectangle(
            (caret_x, input_top + 6, caret_x + 2, input_top + input_height - 6),
            fill=(150, 232, 255, 255),
        )
    return hits


def _paint_mouth_speed_dropdown(
    draw: Any,
    *,
    panel_x: int,
    panel_y: int,
    panel_width: int,
    font: Any,
    label: str,
    open_menu: bool,
    options: tuple[str, ...],
    padding: int,
) -> dict[str, tuple[int, int, int, int]]:
    """Mouth-speed control in the chat header (clickable dropdown)."""
    hits: dict[str, tuple[int, int, int, int]] = {}
    text = f"Mouth: {label} ▾"
    text_w = _advance(font, text) + 16
    btn_h = _line_height(font) + 4
    btn_x = panel_x + panel_width - padding - text_w
    btn_y = panel_y + 4
    draw.rectangle(
        (btn_x, btn_y, btn_x + text_w, btn_y + btn_h),
        fill=CONTROL_FILL,
        outline=CONTROL_EDGE,
    )
    draw.text((btn_x + 8, btn_y + 2), text, fill=CONTROL_TEXT, font=font)
    hits["mouth_button"] = (btn_x, btn_y, text_w, btn_h)

    if open_menu and options:
        row_h = _line_height(font) + 6
        menu_h = row_h * len(options) + 4
        menu_w = max(text_w, max(_advance(font, f"  {name}") for name in options) + 20)
        menu_x = btn_x + text_w - menu_w
        menu_y = btn_y + btn_h + 2
        draw.rectangle(
            (menu_x, menu_y, menu_x + menu_w, menu_y + menu_h),
            fill=CONTROL_FILL,
            outline=CONTROL_EDGE,
        )
        for index, name in enumerate(options):
            oy = menu_y + 2 + index * row_h
            selected = name == label
            draw.text(
                (menu_x + 10, oy),
                name,
                fill=CONTROL_ACTIVE if selected else CONTROL_TEXT,
                font=font,
            )
            hits[f"mouth_opt_{name}"] = (menu_x, oy, menu_w, row_h)
    return hits


def _line_height(font: Any) -> int:
    try:
        ascent, descent = font.getmetrics()
        return int(ascent + descent) + 4
    except AttributeError:
        return 20


def _advance(font: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return int(font.getlength(text))
    except AttributeError:
        return int(font.getbbox(text)[2])


def frame_layout(
    width: int,
    height: int,
    *,
    panel_fraction: float = 0.28,
    min_panel: int = 170,
    max_panel: int = 340,
) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]]:
    """Split the window into a square portrait frame and the chat panel.

    Returns the portrait rectangle in GL UV (origin bottom-left, what the
    shader wants) and the panel rectangle in top-down overlay pixels (what PIL
    wants). The portrait stays square so reframing never stretches the face.
    """
    width = max(int(width), 1)
    height = max(int(height), 1)
    panel = int(min(max(height * float(panel_fraction), min_panel), max_panel))
    panel = max(0, min(panel, height - 64))

    face_area = height - panel
    side = max(min(width, face_area), 1)
    face_x = (width - side) // 2
    face_y_top = (face_area - side) // 2

    uv_rect = (
        face_x / width,
        # UV origin is bottom-left; the panel occupies the bottom band.
        (height - face_y_top - side) / height,
        side / width,
        side / height,
    )
    panel_rect = (0, height - panel, width, panel)
    return uv_rect, panel_rect


def hit_test(
    hits: dict[str, tuple[int, int, int, int]], x: int, y: int
) -> str | None:
    """Return the control id under ``(x, y)``, or ``None``."""
    for name, (hx, hy, hw, hh) in hits.items():
        if hx <= x < hx + hw and hy <= y < hy + hh:
            return name
    return None


__all__ = [
    "CARET_BLINK_SECONDS",
    "MAX_ENTRIES",
    "MAX_INPUT_CHARS",
    "SPEAKER_FACE",
    "SPEAKER_SYSTEM",
    "SPEAKER_YOU",
    "ChatBox",
    "ChatEntry",
    "frame_layout",
    "hit_test",
    "measure_columns",
    "paint_panel",
]
