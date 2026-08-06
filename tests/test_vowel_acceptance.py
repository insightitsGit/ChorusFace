"""F15 — Model A acceptance against shipped teacher-trained weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chorusface.vowel.acceptance import evaluate_model_a
from chorusface.vowel.model_a import ModelA

ROOT = Path(__file__).resolve().parents[1]
MODEL_A = ROOT / "output" / "worlds" / "tickfeed" / "vowel" / "model_a.npz"


@pytest.mark.skipif(not MODEL_A.is_file(), reason="model_a.npz not shipped yet")
def test_f15_shipped_model_a_passes():
    model = ModelA.load(MODEL_A)
    report = evaluate_model_a(model)
    assert report.passed, report.to_dict()
    angry = model.predict("AA", "ANGRY")
    happy = model.predict("AA", "HAPPY")
    upper_l2 = float(np.linalg.norm(angry[:4] - happy[:4]))
    assert upper_l2 >= 0.12, upper_l2
