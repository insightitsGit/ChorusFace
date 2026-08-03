"""Text to visemes, mouth poses, conversation memory, and the chat backend.

Everything here is pure Python with no GPU or window dependency, so the whole
speech path is testable without an OpenGL context. The app layer consumes it:
text becomes a timed viseme stream, each viseme becomes a muscle impulse, and
the biomechanical solver turns those into a face.

Conversation memory is deliberate NWR-grade control surface power applied to
chat: emotion and spoken context carry across turns so the face does not reset
to neutral every reply.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Final, Sequence

DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"
DEFAULT_MODEL: Final = "gpt-4o-mini"
REQUEST_TIMEOUT: Final = 60.0
IMPULSE_SCALE: Final = 0.55
# How many prior user/assistant turns stay in the LLM body.
DEFAULT_HISTORY_TURNS: Final = 8

# Oculus / MPEG-4 style inventory (15 visual units of intelligibility), plus a
# few retained aliases hosts still emit. Canonical names are uppercase.
# Reference: Meta Oculus Lipsync viseme set (sil, PP, FF, TH, DD, kk, CH, SS,
# nn, RR, aa, E, ih, oh, ou).
CANONICAL_VISEMES: Final[frozenset[str]] = frozenset(
    {
        "REST",  # sil
        "PP",  # p, b, m
        "FF",  # f, v
        "TH",  # th
        "DD",  # t, d
        "KK",  # k, g
        "CH",  # ch, sh, j
        "SS",  # s, z
        "NN",  # n, l
        "RR",  # r
        "AA",  # aa / open A
        "AH",  # more open central (kept for jaw contrast)
        "EH",  # E / bed
        "IH",  # ih / tip
        "EE",  # wide smile vowel (iy family)
        "OH",  # oh / toe
        "OU",  # ou / book / moon
        "CLOSED",  # full stop / bilabial seal
    }
)

# Hosts and older tags may use Oculus lowercase ids or prior AIFace names.
VISEME_ALIASES: Final[dict[str, str]] = {
    "SIL": "REST",
    "SILENCE": "REST",
    "NEUTRAL": "REST",
    "FV": "FF",
    "MM": "PP",
    "L": "NN",
    "R": "RR",
    "OO": "OU",
    "UW": "OU",
    "IY": "EE",
    "E": "EH",
    "I": "IH",
    "O": "OH",
    "U": "OU",
    "KKK": "KK",
    "K": "KK",
    "G": "KK",
    "T": "DD",
    "D": "DD",
    "N": "NN",
    "S": "SS",
    "Z": "SS",
    "F": "FF",
    "V": "FF",
    "P": "PP",
    "B": "PP",
    "M": "PP",
}

# Grid y points up. Positive V_y lifts soft tissue; negative drops the jaw open.
# Magnitudes are tuned for lip-reading contrast, not subtlety.
PHONEME_IMPULSES: Final[dict[str, tuple[float, float]]] = {
    "REST": (0.0, 0.0),
    "CLOSED": (0.0, 0.65),
    "PP": (0.0, 0.70),
    "FF": (0.20, 0.40),
    "TH": (0.25, -0.15),
    "DD": (0.10, -0.12),
    "KK": (0.05, 0.20),
    "CH": (0.45, 0.10),
    "SS": (0.50, 0.05),
    "NN": (0.15, -0.18),
    "RR": (0.08, 0.12),
    "AH": (0.0, -1.05),
    "AA": (0.0, -0.90),
    "EH": (0.40, -0.45),
    "IH": (0.30, -0.20),
    "EE": (0.80, 0.12),
    "OH": (0.05, -0.65),
    "OU": (0.0, 0.35),
}

EMOTION_IMPULSES: Final[dict[str, tuple[float, float]]] = {
    "NEUTRAL": (0.0, 0.0),
    "HAPPY": (0.45, 0.35),
    "SAD": (0.0, -0.25),
    "SURPRISED": (0.0, -1.10),
    "ANGRY": (0.20, -0.40),
    "THINKING": (0.10, 0.05),
}

# Relative hold times. Open vowels linger; stops are snappy; REST is a breath.
PHONEME_DURATION_SCALE: Final[dict[str, float]] = {
    "REST": 0.70,
    "CLOSED": 0.85,
    "PP": 0.85,
    "FF": 0.75,
    "TH": 0.75,
    "DD": 0.50,
    "KK": 0.50,
    "CH": 0.70,
    "SS": 0.80,
    "NN": 0.70,
    "RR": 0.75,
    "AH": 1.45,
    "AA": 1.35,
    "EH": 1.05,
    "IH": 0.95,
    "EE": 1.10,
    "OH": 1.25,
    "OU": 1.20,
}

#: Visemes that carry a syllable. One of these is the loud, open centre of a
#: syllable, which is what an energy peak in a recording actually marks — so this
#: is the set the streaming aligner pins to measured peaks.
VOWEL_VISEMES: Final[frozenset[str]] = frozenset(
    {"AH", "AA", "EH", "IH", "EE", "OH", "OU"}
)

VOWEL_MAP: Final[dict[str, str]] = {
    "A": "AH",
    "E": "EE",
    "I": "EE",
    "O": "OH",
    "U": "OU",
    "Y": "EE",
}

# Grapheme fallback: only consonants with a strong external lip cue. Full
# Oculus inventory (DD, KK, NN from /n/, …) is still accepted on the host
# timeline; spelling must stay sparse enough that energy alignment can pin
# syllables.
LETTER_VISEMES: Final[dict[str, str]] = {
    "B": "PP",
    "F": "FF",
    "L": "NN",
    "M": "PP",
    "P": "PP",
    "R": "RR",
    "S": "SS",
    "V": "FF",
    "Z": "SS",
}

# Letter pairs that speak as one sound. Vowel pairs matter as much as consonant
# pairs: a diphthong is one syllable, and spelling it as two visemes would tell
# the aligner the voice is saying twice as much as it is.
DIGRAPH_VISEMES: Final[dict[str, str]] = {
    "OO": "OU",
    "OU": "OU",
    "OW": "OH",
    "OA": "OH",
    "OI": "OH",
    "OY": "OH",
    "AU": "OH",
    "AW": "OH",
    "AI": "EH",
    "AY": "EH",
    "EE": "EE",
    "EA": "EE",
    "EI": "EE",
    "EY": "EE",
    "IE": "EE",
    "UE": "OU",
    "UI": "OU",
    "TH": "TH",
    "SH": "CH",
    "CH": "CH",
    "PH": "FF",
    "CK": "CLOSED",
    "NG": "CLOSED",
    "QU": "OU",
    "WH": "OU",
}

# Endings where a final ``e`` is spoken as its own syllable, so it keeps its
# viseme: "little" is two syllables, "voice" is one.
SYLLABIC_ENDINGS: Final = ("LE",)
#: Shortest word in which a trailing ``e`` is assumed silent. Below this the
#: vowel is usually the whole point of the word — "the", "be", "we".
SILENT_E_MINIMUM: Final = 4

# Punctuation that stops the voice: the lips close and the phrase lands.
STOP_PUNCTUATION: Final = ".!?"
# Punctuation that only breaks the airflow.
BREATH_PUNCTUATION: Final = ",;:\u2014-"

TOKEN_WORD: Final = "word"
#: Whitespace between words: the lips relax but the phrase continues.
TOKEN_BREATH: Final = "breath"
#: Comma-level punctuation: a real break in the airflow.
TOKEN_PAUSE: Final = "pause"
#: Sentence-final punctuation: the voice stops and the lips close.
TOKEN_STOP: Final = "stop"
#: Breaks that split an utterance into separately timed phrases.
HARD_BREAKS: Final = frozenset({TOKEN_PAUSE, TOKEN_STOP})

# Numbers first so a decimal point is not mistaken for a full stop; words keep
# their internal apostrophes so "don't" stays one word for timestamp matching.
_TOKEN_PATTERN: Final = re.compile(
    r"\d+(?:[.,]\d+)*"
    r"|[^\W\d_]+(?:['\u2019][^\W\d_]+)*"
    r"|\s+"
    r"|[^\w\s]"
)


@dataclass(frozen=True, slots=True)
class MouthPose:
    """Readable render-space mouth shape for one viseme."""

    width: float
    openness: float
    roundness: float
    expression: float = 0.0


# Width / openness / roundness deliberately exaggerate closures vs open vowels
# so a lip reader can tell PP from AA from OU at a glance on a still frame.
MOUTH_POSES: Final[dict[str, MouthPose]] = {
    "REST": MouthPose(14.0, 1.0, 0.12),
    "CLOSED": MouthPose(15.0, 0.6, 0.10),
    "PP": MouthPose(14.0, 0.5, 0.12),
    "FF": MouthPose(16.5, 2.8, 0.08),
    "TH": MouthPose(16.0, 4.2, 0.08),
    "DD": MouthPose(15.5, 3.2, 0.10),
    "KK": MouthPose(14.5, 2.4, 0.15),
    "CH": MouthPose(17.5, 3.6, 0.35),
    "SS": MouthPose(18.5, 3.2, 0.05),
    "NN": MouthPose(15.0, 4.0, 0.08),
    "RR": MouthPose(12.0, 3.6, 0.55),
    "AH": MouthPose(16.5, 14.0, 0.18),
    "AA": MouthPose(17.5, 12.0, 0.14),
    "EH": MouthPose(18.5, 8.0, 0.05),
    "IH": MouthPose(17.5, 5.8, 0.05),
    "EE": MouthPose(22.0, 4.8, 0.00),
    "OH": MouthPose(10.0, 10.0, 0.90),
    "OU": MouthPose(7.0, 7.5, 1.00),
}


@dataclass(frozen=True, slots=True)
class VisemeEvent:
    """One scheduled mouth impulse."""

    phoneme: str
    emotion: str
    due_at: float
    duration: float


@dataclass(frozen=True, slots=True)
class PhonemeSpan:
    """One viseme, timed in seconds from the first audio sample.

    Both aligners speak in these: the offline one measures a clip it already
    holds, the streaming one measures audio as it arrives. Sharing the type is
    what makes the two directly comparable.
    """

    phoneme: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_tuple(self) -> tuple[str, float, float]:
        return (self.phoneme, self.start, self.end)


@dataclass(frozen=True, slots=True)
class SpokenToken:
    """One written unit of speech and the visemes it articulates.

    Alignment against real audio needs word boundaries, not a flat viseme
    stream: a timestamp from a speech recogniser covers a *word*, and the
    visemes inside it have to be distributed across that span.
    """

    text: str
    kind: str
    visemes: tuple[str, ...]

    @property
    def is_word(self) -> bool:
        return self.kind == TOKEN_WORD

    @property
    def is_break(self) -> bool:
        """Whether this token ends a phrase rather than joining two words."""
        return self.kind in HARD_BREAKS

    @property
    def nominal_duration(self) -> float:
        """Unstretched articulation time, in `phoneme_hold` base units."""
        return sum(PHONEME_DURATION_SCALE.get(name, 1.0) for name in self.visemes)


@dataclass
class ConversationSession:
    """Rolling chat memory so emotion and topic survive across turns.

    NWR's control bridge keeps world state coherent for an external agent; this
    does the same for spoken dialogue. Without it every reply starts from a
    blank face and a blank mind.
    """

    max_turns: int = DEFAULT_HISTORY_TURNS
    messages: list[dict[str, str]] = field(default_factory=list)
    last_emotion: str = "NEUTRAL"
    turn_count: int = 0

    def remember_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def remember_assistant(self, text: str, emotion: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        if emotion in EMOTION_IMPULSES:
            self.last_emotion = emotion
        self.turn_count += 1
        self._trim()

    def _trim(self) -> None:
        max_messages = max(self.max_turns, 1) * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def system_prompt(self) -> str:
        emotion_hint = (
            f"The face is currently holding {self.last_emotion}. "
            "Prefer continuity unless the user clearly shifts mood. "
            if self.last_emotion != "NEUTRAL"
            else ""
        )
        return (
            "You are a face avatar in a live biomechanical runtime. "
            "Reply in one or two short spoken sentences that a mouth can "
            "actually articulate. "
            f"{emotion_hint}"
            "Prefix the reply with one [EMOTION:NAME] tag where NAME is one of "
            f"{sorted(EMOTION_IMPULSES)} and optionally sprinkle [PHONEME:CODE] "
            f"tags from {sorted(PHONEME_IMPULSES)}. Keep the audible words too."
        )


_ONES: Final[tuple[str, ...]] = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS: Final[tuple[str, ...]] = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES: Final[tuple[tuple[int, str], ...]] = (
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
    (100, "hundred"),
)


def _integer_words(value: int) -> list[str]:
    if value < 0:
        return ["minus", *_integer_words(-value)]
    if value < 20:
        return [_ONES[value]]
    if value < 100:
        tens, rest = divmod(value, 10)
        return [_TENS[tens], *([_ONES[rest]] if rest else [])]
    for scale, name in _SCALES:
        if value >= scale:
            count, rest = divmod(value, scale)
            return [
                *_integer_words(count),
                name,
                *(_integer_words(rest) if rest else []),
            ]
    # Beyond the named scales a speaker reads the digits out.
    return [_ONES[int(digit)] for digit in str(value)]


def spoken_number(raw: str) -> list[str]:
    """Read a written number the way a voice engine will read it aloud.

    Alignment counts words, so ``25`` has to become ``twenty five`` before the
    lips can be matched to what the synthesiser actually says.
    """
    cleaned = raw.replace(",", "").strip()
    if not cleaned:
        return []
    integer_part, _, fraction = cleaned.partition(".")
    words: list[str] = []
    if integer_part:
        try:
            words.extend(_integer_words(int(integer_part)))
        except ValueError:
            return []
    if fraction:
        words.append("point")
        words.extend(_ONES[int(digit)] for digit in fraction if digit.isdigit())
    return words


def strip_tags(text: str) -> str:
    """Remove `[EMOTION:...]`-style control tags from a spoken line."""
    return re.sub(r"\[[^\]]*\]", " ", text)


def canonical_viseme(name: str) -> str:
    """Map any host / legacy / Oculus id onto the inventory we animate.

    Unknown names become ``REST`` rather than inventing a mouth shape — a closed
    mouth is merely early; a wrong shape is a lie a lip reader will catch.
    """
    key = str(name or "").strip().upper()
    if not key:
        return "REST"
    key = VISEME_ALIASES.get(key, key)
    if key in PHONEME_IMPULSES:
        return key
    return "REST"


def _drops_final_e(upper: str) -> bool:
    """Whether a written trailing ``e`` is silent in speech.

    Spelling generates far more vowels than a voice pronounces, and every spare
    vowel tells the aligner about a syllable that was never spoken. English is
    largely consistent here: a final ``e`` after a consonant is silent unless it
    carries the ``-le`` ending.
    """
    if len(upper) < SILENT_E_MINIMUM or not upper.endswith("E"):
        return False
    if upper.endswith(SYLLABIC_ENDINGS):
        return False
    return upper[-2] not in VOWEL_MAP


def word_visemes(word: str) -> list[str]:
    """Articulate one written word as a viseme stream.

    Digraphs win over single letters (`TH`, `SH`, `OO`, `AI`), vowels map to open
    shapes, and letters with no visible lip signature are silent. Two rules keep
    the stream close to what a voice actually does: a silent final ``e`` is
    dropped, and a shape is never articulated twice in a row — a doubled letter
    is one movement of the mouth, not two.
    """
    upper = word.upper()
    if _drops_final_e(upper):
        upper = upper[:-1]
    result: list[str] = []
    index = 0
    while index < len(upper):
        pair = upper[index : index + 2]
        if pair in DIGRAPH_VISEMES:
            viseme = DIGRAPH_VISEMES[pair]
            index += 2
        else:
            character = upper[index]
            index += 1
            if character in VOWEL_MAP:
                viseme = VOWEL_MAP[character]
            elif character in LETTER_VISEMES:
                viseme = LETTER_VISEMES[character]
            else:
                continue
        if not result or result[-1] != viseme:
            result.append(viseme)
    return result


def tokenize_speech(text: str) -> list[SpokenToken]:
    """Split a spoken line into words and pauses with their visemes.

    This is the single phonetic source of truth: :func:`text_to_visemes`
    flattens these tokens, and audio alignment stretches them onto the
    timeline of a real waveform.
    """
    stripped = strip_tags(text)
    tokens: list[SpokenToken] = []

    def push_breath(raw: str) -> None:
        # Whitespace around punctuation is part of that pause, not another one.
        if tokens and tokens[-1].kind in (TOKEN_BREATH, TOKEN_PAUSE, TOKEN_STOP):
            return
        tokens.append(SpokenToken(raw, TOKEN_BREATH, ("REST",)))

    def push_pause(raw: str) -> None:
        if tokens and tokens[-1].kind in (TOKEN_PAUSE, TOKEN_STOP):
            return
        pause = SpokenToken(raw, TOKEN_PAUSE, ("REST",))
        if tokens and tokens[-1].kind == TOKEN_BREATH:
            # Punctuation outranks the space that preceded it.
            tokens[-1] = pause
            return
        tokens.append(pause)

    def push_word(raw: str) -> None:
        visemes = word_visemes(raw)
        if visemes:
            tokens.append(SpokenToken(raw, TOKEN_WORD, tuple(visemes)))

    for match in re.finditer(_TOKEN_PATTERN, stripped):
        raw = match.group(0)
        head = raw[0]
        if head.isspace():
            push_breath(raw)
        elif head in STOP_PUNCTUATION:
            tokens.append(SpokenToken(raw, TOKEN_STOP, ("CLOSED",)))
        elif head in BREATH_PUNCTUATION:
            push_pause(raw)
        elif head.isdigit():
            for word in spoken_number(raw):
                push_word(word)
        elif head.isalpha():
            push_word(raw)
    return tokens


def text_to_visemes(text: str, *, limit: int = 48) -> list[str]:
    """Convert ordinary written words into a compact deterministic viseme stream."""
    result = [
        viseme for token in tokenize_speech(text) for viseme in token.visemes
    ]
    if not result:
        return ["REST"]
    if len(result) > limit:
        step = math.ceil(len(result) / limit)
        result = result[::step]
    return result


def mouth_pose(phoneme: str, emotion: str) -> MouthPose:
    """Resolve a viseme and emotion into a stable render-space mouth pose."""
    base = MOUTH_POSES.get(canonical_viseme(phoneme), MOUTH_POSES["REST"])
    expression = {
        "HAPPY": 0.85,
        "SAD": -0.70,
        "SURPRISED": 0.0,
        "ANGRY": -0.25,
        "THINKING": 0.15,
    }.get(emotion, 0.0)
    openness = base.openness
    roundness = base.roundness
    if emotion == "SURPRISED":
        openness = max(openness, 9.0)
        roundness = max(roundness, 0.8)
    return MouthPose(base.width, openness, roundness, expression)


def extract_states(text: str) -> tuple[list[str], str]:
    """Pull phonemes and one emotion from an AI reply.

    Accepts explicit tags the model may emit::

        [PHONEME:AH] [EMOTION:HAPPY] Hello there!

    and otherwise derives a phoneme stream from Latin vowels in the text.
    """
    upper = text.upper()
    emotion = "NEUTRAL"
    emotion_match = re.search(r"\[EMOTION:\s*([A-Z]+)\]", upper)
    if emotion_match and emotion_match.group(1) in EMOTION_IMPULSES:
        emotion = emotion_match.group(1)

    tagged = re.findall(r"\[PHONEME:\s*([A-Z]+)\]", upper)
    phonemes = [canonical_viseme(name) for name in tagged]
    phonemes = [name for name in phonemes if name != "REST" or len(tagged) == 1]
    if tagged:
        return phonemes or ["REST"], emotion

    for name in ("SURPRISED", "HAPPY", "ANGRY", "SAD", "THINKING"):
        if re.search(rf"\b{name}\b", upper):
            emotion = name
            break

    return text_to_visemes(text), emotion


def phoneme_hold(phoneme: str, *, base: float = 0.09) -> float:
    """How long one viseme should drive the muscle solver."""
    key = canonical_viseme(phoneme)
    return base * PHONEME_DURATION_SCALE.get(key, 1.0)


def schedule_visemes(
    phonemes: Sequence[str],
    emotion: str,
    *,
    start_at: float,
    seconds_per_phoneme: float = 0.09,
    coarticulate: float = 0.22,
) -> list[VisemeEvent]:
    """Lay a viseme stream on a timeline with coarticulation.

    ``coarticulate`` is the fraction of each hold that anticipates the next
    impulse. Real speech anticipates the coming shape; firing impulses that
    abut with a hard edge is what makes a face look typed rather than spoken.
    """
    events: list[VisemeEvent] = []
    cursor = float(start_at)
    mood = emotion if emotion in EMOTION_IMPULSES else "NEUTRAL"
    overlap = max(0.0, min(0.45, float(coarticulate)))
    for index, phoneme in enumerate(phonemes):
        key = canonical_viseme(phoneme)
        duration = phoneme_hold(key, base=seconds_per_phoneme)
        if index > 0 and events and events[-1].phoneme != key:
            cursor -= events[-1].duration * overlap * 0.5
        events.append(
            VisemeEvent(
                phoneme=key,
                emotion=mood,
                due_at=cursor,
                duration=duration,
            )
        )
        cursor += duration * (1.0 - overlap * 0.35)
    return events


def schedule_spans(
    spans: Sequence[tuple[str, float, float]],
    emotion: str,
    *,
    start_at: float,
    minimum_hold: float = 0.045,
) -> list[VisemeEvent]:
    """Turn absolute-timed phoneme spans into muscle events.

    Unlike :func:`schedule_visemes` this adds no rhythm of its own: the spans
    already carry timing measured from real audio, and inventing coarticulation
    on top of a measured schedule would slide the lips off the voice.
    """
    events: list[VisemeEvent] = []
    mood = emotion if emotion in EMOTION_IMPULSES else "NEUTRAL"
    for phoneme, span_start, span_end in spans:
        key = canonical_viseme(phoneme)
        events.append(
            VisemeEvent(
                phoneme=key,
                emotion=mood,
                due_at=float(start_at) + float(span_start),
                duration=max(float(span_end) - float(span_start), minimum_hold),
            )
        )
    return events


def speech_overlay_until(
    *,
    now: float,
    due_at: float | None,
    duration: float,
    next_due_at: float | None,
    frame: float = 1.0 / 60.0,
) -> float:
    """Absolute audio-clock release time for a TickFeed live LOOK overlay.

    Fire times already use ``due_at = start_at + span.start``. Display must
    release at the scheduled span end (capped by the next event), not at
    ``now + vowel_hold_floor`` — those floors drifted lips past word closures.
    """
    due = float(due_at) if due_at is not None else float(now)
    scheduled_end = due + max(float(duration), 0.0)
    # Late fires still get one visible frame.
    until = max(scheduled_end, float(now) + max(float(frame), 1e-4))
    if next_due_at is not None:
        # Never overrun the next scheduled speech event (interruptible).
        until = min(until, max(float(next_due_at), float(now) + max(float(frame), 1e-4)))
    return float(until)


def compose_impulse(phoneme: str, emotion: str) -> tuple[float, float]:
    """Blend a viseme and a mood into one grid-space velocity impulse."""
    base = PHONEME_IMPULSES.get(canonical_viseme(phoneme), PHONEME_IMPULSES["REST"])
    mood = EMOTION_IMPULSES.get(emotion, EMOTION_IMPULSES["NEUTRAL"])
    return (
        (base[0] + mood[0] * 0.35) * IMPULSE_SCALE,
        (base[1] + mood[1] * 0.45) * IMPULSE_SCALE,
    )


def parse_timeline_spans(
    raw: Sequence[object],
) -> list[tuple[str, float, float]]:
    """Validate host-provided timed visemes for :meth:`AvatarFaceApp` timeline.

    Each item is ``{"phoneme"|"viseme": "OU", "start": 0.0, "end": 0.12}``.
    Times are seconds from utterance start (the host's audio clock). Spans must
    be ordered, non-empty, and ``end > start``.
    """
    if not raw:
        raise ValueError("timeline spans required")
    spans: list[tuple[str, float, float]] = []
    previous_end = -1.0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"span {index} must be an object")
        name = item.get("phoneme") or item.get("viseme") or item.get("name")
        if name is None:
            raise ValueError(f"span {index} needs phoneme or viseme")
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"span {index} needs numeric start and end") from exc
        if end <= start:
            raise ValueError(f"span {index} end must be after start")
        if start < previous_end - 1e-6:
            raise ValueError(f"span {index} starts before the previous span ends")
        previous_end = end
        spans.append((canonical_viseme(str(name)), start, end))
    return spans


def local_reply(user_text: str, *, last_emotion: str = "NEUTRAL") -> str:
    """Deterministic offline reply so the driver works without an API key.

    Replies are plain spoken sentences. Visemes come from the words themselves
    via :func:`text_to_visemes` — never from a canned phoneme filler, which a
    real voice would read aloud as gibberish.
    """
    lowered = user_text.lower()
    snippet = re.sub(r"\s+", " ", strip_tags(user_text)).strip()
    if len(snippet) > 72:
        snippet = snippet[:69].rstrip() + "..."

    if any(word in lowered for word in ("happy", "great", "love", "win", "smile")):
        return "[EMOTION:HAPPY] That makes me smile. Tell me more about it."
    if any(word in lowered for word in ("wow", "surprise", "really", "amazing")):
        return "[EMOTION:SURPRISED] Oh! That caught me off guard."
    if any(word in lowered for word in ("sad", "sorry", "miss", "hurt")):
        return "[EMOTION:SAD] I hear you. I'm still right here with you."
    if any(word in lowered for word in ("mad", "angry", "hate", "furious")):
        return "[EMOTION:ANGRY] Easy now. Let's slow that down a little."
    if any(word in lowered for word in ("hello", "hi ", "hey", "greetings")):
        return "[EMOTION:HAPPY] Hello there. It's good to see you."
    if any(word in lowered for word in ("move", "realtime", "real time", "living")):
        return (
            "[EMOTION:HAPPY] Yes. My lips, jaw, and eyes move in real time "
            "while we talk."
        )
    if any(word in lowered for word in ("how", "what", "why", "who", "?")):
        return (
            "[EMOTION:THINKING] Interesting question. Ask me again with a "
            "language model key for a deeper answer."
        )
    if last_emotion == "HAPPY" and any(
        word in lowered for word in ("and", "also", "more", "tell")
    ):
        return "[EMOTION:HAPPY] Still smiling with you. Keep going."
    if last_emotion == "SAD":
        return "[EMOTION:SAD] I'm still with you. Take your time."
    if not snippet:
        return "[EMOTION:NEUTRAL] I'm listening. Go ahead whenever you're ready."
    return f"[EMOTION:NEUTRAL] Got it. You mentioned: {snippet}."


def llm_reply(
    user_text: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    session: ConversationSession | None = None,
) -> str:
    """Ask an OpenAI-compatible chat model for a tagged spoken reply.

    Falls back to :func:`local_reply` whenever the network, the key, or the
    response shape is unavailable, so the face never stalls on a chat failure.
    When ``session`` is provided, prior turns ride along and the reply is
    recorded so the next call inherits emotion and topic.
    """
    conversation = session if session is not None else ConversationSession()
    conversation.remember_user(user_text)

    if not api_key:
        spoken = local_reply(user_text, last_emotion=conversation.last_emotion)
        _, emotion = extract_states(spoken)
        conversation.remember_assistant(spoken, emotion)
        return spoken

    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": conversation.system_prompt()},
            *conversation.messages,
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        spoken = str(payload["choices"][0]["message"]["content"])
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        spoken = (
            local_reply(user_text, last_emotion=conversation.last_emotion)
            + f" (llm fallback: {exc})"
        )

    _, emotion = extract_states(spoken)
    conversation.remember_assistant(spoken, emotion)
    return spoken


__all__ = [
    "BREATH_PUNCTUATION",
    "DEFAULT_BASE_URL",
    "DEFAULT_HISTORY_TURNS",
    "DEFAULT_MODEL",
    "DIGRAPH_VISEMES",
    "EMOTION_IMPULSES",
    "HARD_BREAKS",
    "IMPULSE_SCALE",
    "LETTER_VISEMES",
    "MOUTH_POSES",
    "PHONEME_DURATION_SCALE",
    "PHONEME_IMPULSES",
    "STOP_PUNCTUATION",
    "TOKEN_BREATH",
    "TOKEN_PAUSE",
    "TOKEN_STOP",
    "TOKEN_WORD",
    "CANONICAL_VISEMES",
    "VISEME_ALIASES",
    "VOWEL_MAP",
    "VOWEL_VISEMES",
    "ConversationSession",
    "MouthPose",
    "PhonemeSpan",
    "SpokenToken",
    "VisemeEvent",
    "canonical_viseme",
    "compose_impulse",
    "extract_states",
    "llm_reply",
    "local_reply",
    "mouth_pose",
    "parse_timeline_spans",
    "phoneme_hold",
    "schedule_spans",
    "schedule_visemes",
    "speech_overlay_until",
    "spoken_number",
    "strip_tags",
    "text_to_visemes",
    "tokenize_speech",
    "word_visemes",
]
