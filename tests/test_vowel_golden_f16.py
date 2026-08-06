"""F16 — golden conversational lines resolve to stable GA-16 scripts."""

from __future__ import annotations

from chorusface.vowel.g2p import g2p_text, g2p_word
from chorusface.vowel.pipeline import compose_utterance
from chorusface.vowel.schema import GA16_INDEX

# Locked product lines (VowelDesignFinalAnswers F16).
F16_LINES = (
    ("NEUTRAL", "Hello, how can I help you today?"),
    ("ANGRY", "I already told you that wouldn't work."),
    ("HAPPY", "That's such great news, congratulations!"),
)


def _flatten_tags(text: str) -> list[str]:
    out: list[str] = []
    for word, tags in g2p_text(text):
        assert tags is not None, f"unresolved word {word!r} (must not invent REST silently)"
        assert tags, f"empty tags for {word!r}"
        for t in tags:
            assert t in GA16_INDEX
            out.append(t)
    return out


def test_f3_junk_does_not_invent():
    assert g2p_word("xyzq") is None
    assert g2p_word("bcdfg") is None


def test_f16_every_word_resolves():
    for emotion, text in F16_LINES:
        tags = _flatten_tags(text)
        assert tags, text
        result = compose_utterance(
            {
                "utterance_id": f"f16_{emotion.lower()}",
                "text": text,
                "emotion_track": [
                    {"emotion": emotion, "start_s": 0.0, "end_s": 4.0}
                ],
            }
        )
        assert result.chunk.n_ticks > 0
        assert result.controls.shape[1] == 9
        # Snapshot: every resolved vowel appears in composed span tags.
        span_tags = [s.tag.upper() for s in result.payload.spans if s.tag]
        for t in tags:
            assert t in span_tags or t == "AX", (t, span_tags)


def test_f16_ga16_sequence_snapshot():
    """Authoritative scripts = G2P output (frozen once dict-stable)."""
    expected = {
        "NEUTRAL": ["EH", "OH", "AW", "AE", "AY", "EH", "OU", "OU", "EY"],
        "ANGRY": ["AY", "AO", "EH", "EE", "OH", "OU", "AE", "UH", "IH", "ER"],
        "HAPPY": ["AE", "AH", "EY", "OU", "AA", "AE", "OU", "EY", "IH", "AH"],
    }
    for emotion, text in F16_LINES:
        got = _flatten_tags(text)
        assert got == expected[emotion], (emotion, got)
