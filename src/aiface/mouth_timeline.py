"""Viseme-clock authority for mouth-layer appear / disappear.

Open.png, smile.png, and atlas visibility follow the *scheduled* viseme span
(`due_at` → `due_at + duration`). Muscle hold floors and jaw springs may lag
for tissue motion, but they must not keep GPU plates painted after the word
has moved on.

Timing polish (after the first snap timeline):
- Minimum speech dwell so 1–2 tick consonants actually read
- Bridge short gaps to the next due_at so AH→REST→OH flashes die
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from aiface.biomechanics.intent import PHONEME_JAW_TARGET
from aiface.mouth_owner import CLOSED_VISEMES, snap_smile_drive
from aiface.plates import VISEME_OPENNESS
from aiface.speech import canonical_viseme

# Visemes at or above this table openness show open.png (hard snap).
_OPEN_PLATE_MIN: Final = 0.15
# Short energy-aligned spans are ~30ms; plates need ~80ms to read.
MIN_SPEECH_DWELL_S: Final = 0.08
# Hold last speech plate across schedule gaps shorter than this.
MAX_BRIDGE_GAP_S: Final = 0.20


@dataclass(frozen=True, slots=True)
class LayerCommand:
    """One-frame GPU layer directive from the viseme timeline."""

    phoneme: str
    atlas_viseme: str
    open_amount: float
    smile_amount: float
    jaw_target: float
    plate_openness: float
    source: str
    active_until: float


class MouthLayerTimeline:
    """Snap mouth layers to the active viseme event."""

    def __init__(
        self,
        *,
        min_dwell_s: float = MIN_SPEECH_DWELL_S,
        max_bridge_s: float = MAX_BRIDGE_GAP_S,
    ) -> None:
        self._phoneme: str = "REST"
        self._until: float = 0.0
        self._emotion: str = "NEUTRAL"
        self._bridged: bool = False
        self.min_dwell_s = float(min_dwell_s)
        self.max_bridge_s = float(max_bridge_s)

    @property
    def phoneme(self) -> str:
        return self._phoneme

    @property
    def active_until(self) -> float:
        return self._until

    def set_hold_timing(self, *, min_dwell_s: float, max_bridge_s: float) -> None:
        """Realtime UI: how long speech plates stay painted."""
        self.min_dwell_s = max(0.02, float(min_dwell_s))
        self.max_bridge_s = max(0.0, float(max_bridge_s))

    def clear(self) -> None:
        self._phoneme = "REST"
        self._until = 0.0
        self._emotion = "NEUTRAL"
        self._bridged = False

    def fire(
        self,
        phoneme: str,
        *,
        now: float,
        duration: float,
        emotion: str = "NEUTRAL",
        next_due_at: float | None = None,
        due_at: float | None = None,
    ) -> None:
        """Start a layer span on the wall-clock / audio timeline.

        Prefer absolute ``due_at + duration`` when the scheduler provides it —
        ``now + duration`` drifts when fires are late under frame jitter.
        """
        key = canonical_viseme(phoneme)
        self._phoneme = key
        self._emotion = (emotion or "NEUTRAL").strip().upper() or "NEUTRAL"
        self._bridged = False

        if due_at is not None:
            span_end = float(due_at) + max(float(duration), 1e-3)
            # Late fire: still show at least one frame from now.
            span_end = max(span_end, float(now) + 1.0 / 60.0)
        else:
            span_end = float(now) + max(float(duration), 1e-3)
        if key not in CLOSED_VISEMES and key != "REST":
            # Absolute path already used span length; only pad when due_at missing.
            if due_at is None:
                span_end = max(span_end, float(now) + float(self.min_dwell_s))
            # Fill tiny holes before the next scheduled speech event.
            if next_due_at is not None:
                nxt = float(next_due_at)
                gap = nxt - span_end
                if 0.0 < gap <= float(self.max_bridge_s):
                    span_end = nxt
                    self._bridged = True
        elif key in CLOSED_VISEMES:
            # Tight lips clear open immediately — do not min-dwell an open plate.
            pass
        self._until = span_end

    def tick(
        self,
        now: float,
        *,
        width_n: float = 0.0,
        jaw_table: Mapping[str, float] | None = None,
        smile_width_start: float = 0.12,
        smile_width_span: float = 0.35,
        smile_happy_floor: float = 0.0,
        hard_snap: bool = True,
        upcoming_due_at: float | None = None,
        upcoming_phoneme: str | None = None,
    ) -> LayerCommand:
        now = float(now)
        if now > self._until:
            self._expire_or_bridge(
                now,
                upcoming_due_at=upcoming_due_at,
                upcoming_phoneme=upcoming_phoneme,
            )

        phoneme = self._phoneme
        table = jaw_table or {}
        jaw = float(table.get(phoneme, PHONEME_JAW_TARGET.get(phoneme, 0.1)))
        open_n = float(VISEME_OPENNESS.get(phoneme, 0.0))

        if phoneme in CLOSED_VISEMES or phoneme == "REST":
            open_amount = 0.0
            jaw = 0.0
            atlas = phoneme
            source = "timeline-closed" if phoneme in CLOSED_VISEMES else "timeline-rest"
        else:
            if hard_snap:
                open_amount = 1.0 if open_n >= _OPEN_PLATE_MIN else 0.0
            else:
                open_amount = max(0.0, min(1.0, open_n))
            atlas = phoneme
            source = "timeline-bridge" if self._bridged else "timeline-viseme"

        # Smile plate is a wide soft matte — never park it on closed / speech.
        smile = 0.0
        if phoneme == "REST" and open_amount <= 0.0:
            floor = (
                float(smile_happy_floor)
                if self._emotion == "HAPPY"
                else 0.0
            )
            width_smile = max(
                0.0,
                min(
                    1.0,
                    (float(width_n) - float(smile_width_start))
                    / max(float(smile_width_span), 1e-6),
                ),
            )
            smile = max(floor, width_smile)
            if floor <= 0.0:
                smile = 0.0
            else:
                smile = snap_smile_drive(smile, hard_snap=hard_snap)

        return LayerCommand(
            phoneme=phoneme,
            atlas_viseme=atlas,
            open_amount=float(open_amount),
            smile_amount=float(smile),
            jaw_target=float(jaw),
            plate_openness=float(open_amount),
            source=source,
            active_until=float(self._until),
        )

    def _expire_or_bridge(
        self,
        now: float,
        *,
        upcoming_due_at: float | None,
        upcoming_phoneme: str | None,
    ) -> None:
        """Drop to REST, or hold the last speech plate across a tiny gap."""
        if (
            self._phoneme not in CLOSED_VISEMES
            and self._phoneme != "REST"
            and upcoming_due_at is not None
            and upcoming_phoneme is not None
        ):
            nxt_key = canonical_viseme(upcoming_phoneme)
            gap = float(upcoming_due_at) - float(now)
            if (
                nxt_key not in CLOSED_VISEMES
                and nxt_key != "REST"
                and 0.0 <= gap <= float(self.max_bridge_s)
            ):
                # Keep painting the current speech plate until the next fires.
                self._until = float(upcoming_due_at)
                self._bridged = True
                return
        self._phoneme = "REST"
        self._bridged = False


__all__ = [
    "LayerCommand",
    "MAX_BRIDGE_GAP_S",
    "MIN_SPEECH_DWELL_S",
    "MouthLayerTimeline",
]
