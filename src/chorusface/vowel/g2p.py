"""Face-owned G2P fallback (F3 / D25) — dict → rules → REST (no invent)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path

from chorusface.vowel.schema import GA16_INDEX

# Minimal high-frequency GA-16 vowel sequences (not full CMU — Phase-1 bootstrap).
_BUILTIN: dict[str, tuple[str, ...]] = {
    "hi": ("AH", "EE"),
    "hello": ("EH", "OH"),
    "how": ("AW",),
    "are": ("AH", "ER"),
    "you": ("OU",),
    "i": ("AY",),
    "can": ("AE",),
    "help": ("EH",),
    "with": ("IH",),
    "that": ("AE",),
    "stop": ("AA",),
    "right": ("AY",),
    "now": ("AW",),
    "see": ("EE",),
    "tomorrow": ("AX", "AA", "OH"),
    "the": ("AX",),
    "a": ("AX",),
    "to": ("OU",),
    "and": ("AE",),
    "is": ("IH",),
    "it": ("IH",),
    "for": ("AO", "ER"),
    "on": ("AA",),
    "my": ("AY",),
    "me": ("EE",),
    "we": ("EE",),
    "what": ("AH",),
    "when": ("EH",),
    "where": ("EH", "ER"),
    "why": ("AY",),
    "yes": ("EH",),
    "no": ("OH",),
    "ok": ("OH", "EY"),
    "okay": ("OH", "EY"),
    "thanks": ("AE",),
    "thank": ("AE",),
    "please": ("EE",),
    "sorry": ("AO", "EE"),
    "good": ("UH",),
    "great": ("EY",),
    "news": ("OU",),
    "already": ("AO", "EH", "EE"),
    "told": ("OH",),
    "wouldn": ("UH",),
    "wouldn't": ("UH", "IH"),
    "work": ("ER",),
    "such": ("AH",),
    "congratulations": ("AA", "AE", "OU", "EY", "IH", "AH"),
    "today": ("OU", "EY"),
}


@lru_cache(maxsize=1)
def _load_dict() -> dict[str, tuple[str, ...]]:
    out = dict(_BUILTIN)
    try:
        ref = resources.files("chorusface.vowel").joinpath("data", "ga16_dict.json")
        raw = json.loads(ref.read_text(encoding="utf-8"))
        for k, v in raw.items():
            if isinstance(v, list) and v:
                tags = tuple(str(x).upper() for x in v if str(x).upper() in GA16_INDEX)
                if tags:
                    out[str(k).lower()] = tags
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
        pass
    # optional external override
    env_path = Path(__file__).resolve().parents[3] / "data" / "ga16_dict.json"
    if env_path.is_file():
        try:
            raw = json.loads(env_path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                if isinstance(v, list) and v:
                    tags = tuple(
                        str(x).upper() for x in v if str(x).upper() in GA16_INDEX
                    )
                    if tags:
                        out[str(k).lower()] = tags
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return out


_VOWEL_RULES: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"ee|ea|ie|ey$"), ("EE",)),
    (re.compile(r"oo|ou|u$"), ("OU",)),
    (re.compile(r"ow|ou"), ("AW",)),
    (re.compile(r"oy|oi"), ("OY",)),
    (re.compile(r"ay|ai"), ("AY",)),
    (re.compile(r"oa|oe|o$"), ("OH",)),
    (re.compile(r"au|aw"), ("AO",)),
    (re.compile(r"ar|er|ir|ur|or"), ("ER",)),
    (re.compile(r"a$|ah"), ("AA",)),
    (re.compile(r"i$|ih|y$"), ("IH",)),
    (re.compile(r"e$|eh"), ("EH",)),
    (re.compile(r"a"), ("AE",)),
    (re.compile(r"o"), ("AA",)),
    (re.compile(r"u"), ("AH",)),
    (re.compile(r"e"), ("EH",)),
    (re.compile(r"i"), ("IH",)),
]


def normalize_word(word: str) -> str:
    w = word.strip().lower()
    w = re.sub(r"[^a-z']", "", w)
    return w


def g2p_word(word: str) -> tuple[str, ...] | None:
    """Return GA-16 vowel tags for a word, or None → caller must REST-hold."""
    key = normalize_word(word)
    if not key:
        return None
    d = _load_dict()
    if key in d:
        return d[key]
    # strip common clitics
    for suffix in ("'s", "'t", "'re", "'ve", "'ll", "'d"):
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if base in d:
                return d[base]
    # rule-based: collect vowel-ish hits in order (coarse)
    found: list[str] = []
    rest = key
    pos = 0
    while pos < len(rest):
        matched = False
        for pat, tags in _VOWEL_RULES:
            m = pat.match(rest, pos)
            if m:
                found.extend(tags)
                pos = m.end()
                matched = True
                break
        if not matched:
            pos += 1
    # de-dupe adjacent identical
    out: list[str] = []
    for t in found:
        if t in GA16_INDEX and (not out or out[-1] != t):
            out.append(t)
    if out:
        return tuple(out)
    return None  # unresolved → REST


def g2p_text(text: str) -> list[tuple[str, tuple[str, ...] | None]]:
    """Split text into (word, vowels|None) pairs."""
    words = re.findall(r"[A-Za-z']+", text)
    return [(w, g2p_word(w)) for w in words]
