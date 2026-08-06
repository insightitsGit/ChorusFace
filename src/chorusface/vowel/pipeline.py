"""Utterance → 9D controls → PulseChunk (Phase-1 compose path)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chorusface.vowel.blinks import apply_blinks
from chorusface.vowel.g2p import g2p_text, g2p_word
from chorusface.vowel.model_a import ModelA
from chorusface.vowel.model_b import ModelB, diphthong_end_tag
from chorusface.vowel.priors import rest_9d
from chorusface.vowel.pulsechunk import PulseChunk, VersionBlock, WordSlice, encode_pulsechunk
from chorusface.vowel.schema import (
    CROSSFADE_TICKS,
    EMOTION_INDEX,
    GA16_INDEX,
    GROUP_DIM,
    MICRO_REST_MAX_TICKS,
    RELEASE_TICKS,
    TICK_DT,
    TICK_HZ,
)
from chorusface.vowel.utterance import PhonemeSpan, UtterancePayload, WordSpan, parse_utterance


@dataclass(slots=True)
class ComposeResult:
    chunk: PulseChunk
    payload: UtterancePayload
    controls: NDArray[np.float32]
    plates: list[str] = field(default_factory=list)


def _ensure_spans(payload: UtterancePayload) -> UtterancePayload:
    """Fill spans via G2P when host omitted them (F3)."""
    if payload.spans:
        return payload
    pairs = g2p_text(payload.text)
    if not pairs:
        return payload
    # allocate ~150 ms per vowel @ 150 WPM fallback
    t = 0.0
    spans: list[PhonemeSpan] = []
    words: list[WordSpan] = []
    for word, tags in pairs:
        w_start = t
        if tags is None:
            # REST hold ~100 ms (6 ticks)
            t += 6 * TICK_DT
            words.append(WordSpan(text=word, start_s=w_start, end_s=t))
            continue
        for tag in tags:
            spans.append(PhonemeSpan(tag=tag, start_s=t, end_s=t + 0.15))
            t += 0.15
        words.append(WordSpan(text=word, start_s=w_start, end_s=t))
        t += 2 * TICK_DT  # micro gap
    payload.spans = spans
    if not payload.words:
        payload.words = words
    if payload.duration_s is None:
        payload.duration_s = t + RELEASE_TICKS * TICK_DT
    # extend last emotion end
    if payload.emotion_track:
        last = payload.emotion_track[-1]
        payload.emotion_track[-1] = type(last)(
            emotion=last.emotion,
            start_s=last.start_s,
            end_s=max(last.end_s, float(payload.duration_s)),
        )
    return payload


def _word_slices_from_payload(payload: UtterancePayload) -> list[WordSlice]:
    slices: list[WordSlice] = []
    if payload.words:
        for w in payload.words:
            tags = g2p_word(w.text)
            start = int(round(w.start_s * TICK_HZ))
            end = max(start + 1, int(round(w.end_s * TICK_HZ)))
            if tags is None:
                slices.append(
                    WordSlice(start_tick=start, end_tick=end, vowel_ids=[], pause_flag=0)
                )
                continue
            ids = [GA16_INDEX[t] for t in tags if t in GA16_INDEX][:6]
            slices.append(
                WordSlice(start_tick=start, end_tick=end, vowel_ids=ids, pause_flag=0)
            )
    else:
        # one slice per span group
        for s in payload.spans:
            slices.append(
                WordSlice(
                    start_tick=s.start_tick,
                    end_tick=s.end_tick,
                    vowel_ids=[GA16_INDEX.get(s.tag, GA16_INDEX["AX"])],
                    pause_flag=0,
                )
            )
    return slices


class VowelComposer:
    def __init__(
        self,
        model_a: ModelA | None = None,
        model_b: ModelB | None = None,
        *,
        versions: VersionBlock | None = None,
        blinks: bool = True,
        blink_interval_s: float = 3.2,
        blink_seed: int | None = 0,
    ) -> None:
        self.model_a = model_a or ModelA()
        self.model_b = model_b or ModelB()
        self.versions = versions or VersionBlock()
        self.blinks = bool(blinks)
        self.blink_interval_s = float(blink_interval_s)
        self.blink_seed = blink_seed

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> VowelComposer:
        d = Path(model_dir)
        a_path = d / "model_a.npz"
        b_path = d / "model_b.npz"
        a = ModelA.load(a_path) if a_path.is_file() else ModelA()
        b = ModelB.load(b_path) if b_path.is_file() else ModelB()
        if not a.trained:
            a.fit()
        return cls(a, b)

    def compose(self, payload: UtterancePayload | dict) -> ComposeResult:
        if isinstance(payload, dict):
            payload = parse_utterance(payload)
        payload = _ensure_spans(payload)
        n_ticks = payload.total_ticks()
        controls = np.zeros((n_ticks, GROUP_DIM), dtype=np.float64)
        plates = ["REST"] * n_ticks
        key_ticks: list[int] = [0]
        emotion0 = payload.primary_emotion
        controls[:] = rest_9d(emotion0)

        # emotion boundaries → KEY
        for e in payload.emotion_track:
            key_ticks.append(int(round(e.start_s * TICK_HZ)))

        spans = list(payload.spans)
        cursor_state = rest_9d(emotion0)
        from chorusface.vowel.plates import plate_for

        for i, span in enumerate(spans):
            emo = payload.emotion_at(0.5 * (span.start_s + span.end_s))
            c_tgt = self.model_a.predict(span.tag, emo)
            end_tag = diphthong_end_tag(span.tag)
            c_end = self.model_a.predict(end_tag, emo) if end_tag else None
            n = max(1, span.end_tick - span.start_tick)
            # conflict bridge
            if i > 0 and self.model_b.needs_conflict_bridge(spans[i - 1].tag, span.tag):
                bridge = self.model_b.bridge(cursor_state, emo)
                b0 = max(0, span.start_tick - bridge.shape[0])
                for k in range(bridge.shape[0]):
                    t = b0 + k
                    if t < n_ticks:
                        controls[t] = bridge[k]
                        key_ticks.append(t)
                cursor_state = bridge[-1]

            seg = self.model_b.generate_segment(
                cursor_state,
                c_tgt,
                n,
                emo,
                c_end=c_end,
                release=False,
            )
            for k in range(seg.shape[0]):
                t = span.start_tick + k
                if t < n_ticks:
                    controls[t] = seg[k]
                    plates[t] = plate_for(span.tag, emo)
            key_ticks.append(span.start_tick)
            cursor_state = controls[min(span.end_tick - 1, n_ticks - 1)]

            # micro-rest toward next
            if i + 1 < len(spans):
                gap = spans[i + 1].start_tick - span.end_tick
                if 0 < gap <= MICRO_REST_MAX_TICKS:
                    rest = rest_9d(emo)
                    for k in range(gap):
                        t = span.end_tick + k
                        if t < n_ticks:
                            blend = 0.5 * (k + 1) / gap
                            controls[t] = (1.0 - blend) * cursor_state + blend * (
                                0.5 * cursor_state + 0.5 * rest
                            )

        # release pad
        rest = rest_9d(payload.emotion_at(payload.duration_s or 0.0))
        for k in range(RELEASE_TICKS):
            t = n_ticks - RELEASE_TICKS + k
            if 0 <= t < n_ticks:
                blend = (k + 1) / RELEASE_TICKS
                controls[t] = (1.0 - blend) * controls[t] + blend * rest
                key_ticks.append(t)

        # WordSlice boundary crossfade (3 ticks)
        word_slices = _word_slices_from_payload(payload)
        for wi in range(1, len(word_slices)):
            boundary = word_slices[wi].start_tick
            key_ticks.append(boundary)
            if boundary >= CROSSFADE_TICKS and boundary < n_ticks:
                a = controls[boundary - 1]
                b = controls[min(boundary, n_ticks - 1)]
                xf = self.model_b.crossfade(a, b)
                for k in range(min(CROSSFADE_TICKS, n_ticks - (boundary - 1))):
                    t = boundary - 1 + k
                    if 0 <= t < n_ticks:
                        controls[t] = xf[min(k, len(xf) - 1)]

        controls = apply_blinks(
            controls,
            interval_s=self.blink_interval_s,
            seed=self.blink_seed,
            enabled=self.blinks,
        )
        ctrl_f32 = np.asarray(controls, dtype=np.float32)
        chunk = PulseChunk(
            utterance_id=payload.utterance_id,
            n_ticks=n_ticks,
            primary_emotion=EMOTION_INDEX.get(emotion0, 0),
            word_slices=word_slices,
            controls=ctrl_f32,
            versions=self.versions,
            tick_hz=TICK_HZ,
            key_ticks=sorted(set(int(t) for t in key_ticks if 0 <= t < n_ticks)),
        )
        return ComposeResult(
            chunk=chunk, payload=payload, controls=ctrl_f32, plates=plates
        )


def _default_model_dir() -> Path | None:
    """Prefer trained teacher models next to the TickFeed world."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "output" / "worlds" / "tickfeed" / "vowel",
        Path.cwd() / "output" / "worlds" / "tickfeed" / "vowel",
    ]
    for d in candidates:
        if (d / "model_a.npz").is_file():
            return d
    return None


def compose_utterance(
    payload: UtterancePayload | dict,
    *,
    model_dir: str | Path | None = None,
    blinks: bool | None = None,
) -> ComposeResult:
    resolved = Path(model_dir) if model_dir is not None else _default_model_dir()
    composer = (
        VowelComposer.from_dir(resolved) if resolved is not None else VowelComposer()
    )
    if not composer.model_a.trained:
        composer.model_a.fit()
    if isinstance(payload, dict):
        if blinks is None and "blinks" in payload:
            blinks = bool(payload.get("blinks"))
        if "blink_interval_s" in payload:
            composer.blink_interval_s = float(payload["blink_interval_s"])
        if "blink_seed" in payload:
            composer.blink_seed = int(payload["blink_seed"])
        # Default: blinks on for play demos (eyes are part of Dataset A/B path).
        if blinks is None:
            blinks = True
    if blinks is not None:
        composer.blinks = bool(blinks)
    return composer.compose(payload)


def compose_utterance_bytes(
    payload: UtterancePayload | dict,
    *,
    model_dir: str | Path | None = None,
) -> bytes:
    return encode_pulsechunk(compose_utterance(payload, model_dir=model_dir).chunk)
