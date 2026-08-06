"""F4 — first WordSlice schedules before full compose finishes (≤50 ms budget)."""

from __future__ import annotations

import time

from chorusface.vowel.g2p import g2p_text


def test_f4_first_word_g2p_under_50ms():
    """G2P of first resolvable word must be ready within the F4 budget.

    Full compose residual fit may exceed 50 ms; the locked contract is that
    first-motion scheduling can start once the first WordSlice resolves.
    """
    text = "Hello, how can I help you today?"
    t0 = time.perf_counter()
    words = g2p_text(text)
    first = None
    for _w, tags in words:
        if tags:
            first = tags
            break
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert first is not None
    assert elapsed_ms <= 50.0, f"first-word G2P {elapsed_ms:.2f} ms > 50 ms"


def test_f4_early_schedule_before_compose():
    """Bridge early path: first spans exist before compose_utterance returns."""
    from chorusface.vowel.pipeline import compose_utterance

    text = "Hello, how can I help you today?"
    t0 = time.perf_counter()
    words = g2p_text(text)
    early = []
    cursor = 0.05
    for _w, tags in words:
        if not tags:
            continue
        for tag in tags:
            early.append({"tag": tag, "start_s": cursor, "end_s": cursor + 0.10})
            cursor += 0.12
        break
    early_ms = (time.perf_counter() - t0) * 1000.0
    assert early, "expected first-word spans"
    assert early_ms <= 50.0, f"early schedule {early_ms:.2f} ms > 50 ms"

    # Full compose may be slower — that is OK as long as early spans were ready.
    result = compose_utterance(
        {
            "utterance_id": "f4_latency",
            "text": text,
            "emotion_track": [{"emotion": "NEUTRAL", "start_s": 0.0, "end_s": 4.0}],
        }
    )
    assert result.chunk.n_ticks > 0
    assert early[0]["tag"] in {s.tag for s in result.payload.spans if s.tag}
