"""VowelDesign lip path: width/round + muscle FIELD specs must be non-empty."""

from __future__ import annotations

from chorusface.biomechanics.face import BiomechanicalFace
from chorusface.skinning import muscle_anchor_grid


def _drive(face: BiomechanicalFace, phoneme: str, *, frames: int = 24):
    face.submit_phoneme(phoneme, tick=1, emotion_label="NEUTRAL", duration=0.6)
    render = None
    specs = []
    for i in range(frames):
        render, specs = face.step(1.0 / 60.0, tick=i + 2, tickfeed_field=False)
    return render, specs


def test_ee_widens_and_emits_field_specs() -> None:
    face = BiomechanicalFace.from_file()
    face.speech_owns_oral = True
    face.speech_travel_scale = 1.65
    face.lip_width_travel_scale = 2.05
    render, specs = _drive(face, "EE")
    assert render is not None
    assert render.mouth_width > 15.0, render.mouth_width
    assert render.group_activations.get("Risorius", 0.0) >= 0.04
    assert specs, "EE must emit muscle FIELD impulse specs"


def test_ou_rounds_and_emits_oris_field() -> None:
    face = BiomechanicalFace.from_file()
    face.speech_owns_oral = True
    face.speech_travel_scale = 1.65
    face.lip_width_travel_scale = 2.05
    render, specs = _drive(face, "OU")
    assert render is not None
    assert render.mouth_roundness > 0.20, render.mouth_roundness
    assert render.group_activations.get("OrbicularisOris", 0.0) >= 0.04
    assert specs
    writers = {s.muscle for s in specs}
    assert any("Orbicularis" in m or "Buccinator" in m for m in writers), writers


def test_aa_opens_jaw() -> None:
    face = BiomechanicalFace.from_file()
    face.speech_owns_oral = True
    render, _ = _drive(face, "AA", frames=30)
    assert render is not None
    assert render.jaw_angle > 0.05, render.jaw_angle
    assert render.mouth_openness > 2.0, render.mouth_openness


def test_field_spec_anchors_map_into_face_box() -> None:
    face = BiomechanicalFace.from_file()
    _, specs = _drive(face, "EE", frames=20)
    assert specs
    box = {"x": 40.0, "y": 20.0, "width": 160.0, "height": 200.0}
    grid_h = 256
    for spec in specs:
        muscle = face.registry.get(spec.muscle)
        gx, gy = muscle_anchor_grid(muscle, box, grid_h)
        assert box["x"] <= gx <= box["x"] + box["width"]
        assert 0.0 <= gy <= float(grid_h)
