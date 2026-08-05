"""VowelDesign Phase-1 — GA-16 compose → PulseChunk + biomech muscle drive.

Primary face delivery: ``BiomechanicalFace.submit_phoneme`` (see
``docs/VowelDesignNWRReconciliation.md``). Cell-group expand is debug/legacy.
Architecture freeze: ``docs/VowelDesignFinalAnswers.md``.
"""

from __future__ import annotations

from chorusface.vowel.pipeline import compose_utterance, compose_utterance_bytes
from chorusface.vowel.schema import EMOTIONS, GA16, GROUP_DIM, TICK_HZ
from chorusface.vowel.utterance import UtterancePayload, parse_utterance

__all__ = [
    "EMOTIONS",
    "GA16",
    "GROUP_DIM",
    "TICK_HZ",
    "UtterancePayload",
    "compose_utterance",
    "compose_utterance_bytes",
    "parse_utterance",
]
