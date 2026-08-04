"""SpeechAlign — force-align scripted speech onto 60 Hz ticks (Side B §[6]).

Uses the calibration script as beat authority. Inside SAY_HI / TALK windows,
distributes scripted words/visemes evenly across the beat (deterministic
force-align). When WAV+transcript MFA is unavailable this is the accurate
contract-aligned teacher — not free ASR guessing outside beat windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chorusface.tickfeed.calibration import load_calibration_script
from chorusface.tickfeed.package import TickLabels
from chorusface.tickfeed.schema import TICK_RATE_HZ, BeatId, EmotionId

# Simple grapheme→viseme schedule for English script words (phase-1 teacher).
_WORD_VISEMES: dict[str, tuple[str, ...]] = {
    "hi": ("HH", "EE"),
    "hello": ("HH", "EH", "LL", "OH"),
    "there": ("TH", "EH", "RR"),
    "how": ("HH", "OU"),
    "are": ("AA", "RR"),
    "you": ("EE", "OU"),
    "today": ("TT", "OH", "DD", "EE"),
    "ah": ("AA",),
    "oh": ("OH",),
    "think": ("TH", "IH", "NN", "KK"),
}

# Map informal → VISEME_TABLE names
_VISEME_ALIAS: dict[str, str] = {
    "HH": "CLOSED",
    "LL": "NN",
    "TT": "DD",
}


def _viseme_id(name: str) -> int:
    key = _VISEME_ALIAS.get(name.upper(), name.upper())
    return TickLabels.viseme_index(key)


def _words_in_speech(speech: str) -> list[str]:
    out: list[str] = []
    for raw in (speech or "").replace("?", " ").replace(".", " ").split():
        w = raw.strip().lower()
        if w:
            out.append(w)
    return out


def build_speech_align(
    world: Path | str,
    *,
    n_ticks: int,
) -> dict[str, Any]:
    """Return speech_align.json payload: tick → viseme/word."""
    script = load_calibration_script(world)
    ticks: list[dict[str, Any]] = []
    for t in range(int(n_ticks)):
        t_sec = float(t) / float(TICK_RATE_HZ)
        beat = None
        for b in script.get("beats") or []:
            if float(b["t0"]) <= t_sec < float(b["t1"]):
                beat = b
                break
        if beat is None:
            ticks.append(
                {
                    "tick": t,
                    "viseme": "REST",
                    "viseme_id": 0,
                    "word": "",
                    "beat_id": int(BeatId.REST),
                }
            )
            continue
        bid = str(beat.get("id") or "REST")
        speech = str(beat.get("speech") or "")
        words = _words_in_speech(speech)
        viseme = "REST"
        word = ""
        if words and bid in {"SAY_HI", "TALK", "OPEN", "SURPRISE", "TONGUE_TH"}:
            span = max(float(beat["t1"]) - float(beat["t0"]), 1e-6)
            u = (t_sec - float(beat["t0"])) / span
            # Expand words into viseme chain
            chain: list[tuple[str, str]] = []
            for w in words:
                for v in _WORD_VISEMES.get(w, ("AA",)):
                    chain.append((w, v))
            if chain:
                idx = min(int(u * len(chain)), len(chain) - 1)
                word, viseme = chain[idx]
        ticks.append(
            {
                "tick": t,
                "viseme": viseme if viseme != "HH" else "CLOSED",
                "viseme_id": _viseme_id(viseme),
                "word": word[:16],
                "beat_id": int(beat.get("beat_id", BeatId.UNKNOWN)),
                "beat": bid,
            }
        )
    return {
        "schema": "chorusface.speech_align.v1",
        "tick_rate": TICK_RATE_HZ,
        "n_ticks": int(n_ticks),
        "method": "script_force_align",
        "ticks": ticks,
    }


def build_look_drive(
    world: Path | str,
    *,
    n_ticks: int,
    open_curve: list[float] | None = None,
    smile_curve: list[float] | None = None,
    lid_curve: list[float] | None = None,
) -> dict[str, Any]:
    """Per-tick LOOK amounts from calibration beats + optional measured curves.

    Measured curves refine amounts **inside the matching beat only**. They must
    not max-merge into REST (that left smile/open high on every tick and looked
    like a permanently open mouth).
    """
    script = load_calibration_script(world)
    ticks: list[dict[str, Any]] = []
    for t in range(int(n_ticks)):
        t_sec = float(t) / float(TICK_RATE_HZ)
        beat_id = "REST"
        for b in script.get("beats") or []:
            if float(b["t0"]) <= t_sec < float(b["t1"]):
                beat_id = str(b["id"])
                break
        smile = 0.0
        open_ = 0.0
        surprise = 0.0
        brow = 0.0
        lid = (
            float(lid_curve[t])
            if lid_curve is not None and t < len(lid_curve)
            else 1.0
        )
        emotion = int(EmotionId.NEUTRAL)
        curve_o = (
            float(open_curve[t])
            if open_curve is not None and t < len(open_curve)
            else 0.0
        )
        curve_s = (
            float(smile_curve[t])
            if smile_curve is not None and t < len(smile_curve)
            else 0.0
        )
        if beat_id == "SMILE":
            # Closed-lip smile — never open from curve bleed.
            smile = max(0.85, curve_s)
            open_ = 0.0
            emotion = int(EmotionId.HAPPY)
        elif beat_id == "OPEN":
            open_ = max(0.9, curve_o)
            smile = min(curve_s, 0.25)
        elif beat_id == "SAY_HI":
            open_ = max(0.22, min(curve_o, 0.55))
            smile = min(max(curve_s, 0.15), 0.45)
        elif beat_id == "TONGUE_TH":
            # Mild open, no smile — tongue tip owns the oral disk.
            open_ = max(0.18, min(curve_o if curve_o > 0.05 else 0.28, 0.40))
            smile = 0.0
        elif beat_id == "SURPRISE":
            open_ = max(0.2, min(curve_o, 0.45))
            surprise = 0.8
            brow = 0.7
            emotion = int(EmotionId.SURPRISED)
            # Keep measured EAR blinks — do not force lids open.
            lid = max(lid, 0.85) if lid > 0.5 else lid
        elif beat_id == "ANGRY":
            open_ = 0.0
            smile = 0.0
            brow = 0.7
            emotion = int(EmotionId.ANGRY)
        elif beat_id == "BLINK":
            # Deliberate full close for lid teacher + eyes_closed plate bake.
            open_ = 0.0
            smile = 0.0
            surprise = 0.0
            brow = 0.0
            # Prefer measured close; only force if EAR missed the blink.
            if lid > 0.45:
                lid = 0.05
            else:
                lid = min(lid, 0.12)
        elif beat_id == "TALK":
            open_ = max(0.35, min(curve_o if curve_o > 0.05 else 0.45, 0.75))
            smile = min(curve_s, 0.3)
        else:
            # REST — fully closed, ignore noisy landmark curves
            open_ = 0.0
            smile = 0.0
            surprise = 0.0
        ticks.append(
            {
                "tick": t,
                "smile": float(smile),
                "open": float(open_),
                "surprise": float(surprise),
                "brow": float(brow),
                "lid": float(max(0.0, min(1.0, lid))),
                "emotion_id": emotion,
                "beat": beat_id,
            }
        )
    return {
        "schema": "chorusface.look_drive.v3",
        "tick_rate": TICK_RATE_HZ,
        "n_ticks": int(n_ticks),
        "ticks": ticks,
    }


def write_speech_align(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_look_drive(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "build_look_drive",
    "build_speech_align",
    "write_look_drive",
    "write_speech_align",
]
