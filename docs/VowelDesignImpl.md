# VowelDesign — Phase-1 implementation (vowelBrnach)

**Branch:** `vowelBrnach`  
**Freeze:** [`VowelDesignFinalAnswers.md`](VowelDesignFinalAnswers.md)  
**NWR retarget:** [`VowelDesignNWRReconciliation.md`](VowelDesignNWRReconciliation.md)  
**Package:** `src/chorusface/vowel/`

## What’s implemented

| Piece | Module / surface |
| --- | --- |
| GA-16 + 9D F9 contract | `schema.py`, `priors.py` (9D = compose/debug; not primary cell writer) |
| Host utterance JSON (F1) | `utterance.py` |
| G2P → REST fallback (F3) | `g2p.py`, `data/ga16_dict.json` |
| Model A MLP 22→64→64→9 | `model_a.py` |
| Model B attack/hold/release + diphthong + bridges | `model_b.py` |
| PulseChunk PLS1 + WordSlice | `pulsechunk.py` |
| Compose pipeline | `pipeline.py` |
| **Biomech muscle drive** | `muscle_drive.py` + app `kind=="vowel"` → GA-16 tags → `_fire_impulse` |
| Full GA-16 `phoneme_muscles` | `biomechanics/data/face_definition.json` + jaw/`MOUTH_POSES`/`plates` |
| Expand matrix \(W\) | `expand.py` — **legacy/debug**, not product path for locked cells |
| Plate lookup stub | `plates.py` (vowel) + `chorusface/plates.py` LOOK overlay |
| TickPackage KEY/Δ emit | `tick_emit.py` — optional / mouth-only later |
| D35 GO/NO-GO | `teacher.py` |
| Acceptance F15 | `acceptance.py` |
| Runtime helper | `runtime.py` |
| FaceBridge | `POST /vowel/utterance` |
| CLI | `chorusface-vowel` |

## Commands

```powershell
cd C:\code\AIFace
pip install -e ".[dev]"

# Train Model A from §5.2 priors (until Teacher Package exists)
chorusface-vowel train --out output/worlds/tickfeed/vowel

# Compose a PulseChunk
chorusface-vowel compose --json path\to\utt.json --out pulse.pls1 --models output/worlds/tickfeed/vowel

# Author W from region_catalog (or synthetic ROI)
chorusface-vowel author-w --catalog output/worlds/tickfeed/region_catalog.json --out output/worlds/tickfeed/vowel/expand_matrix_v1.wexpand

# D35 (needs a Veo/teacher mp4 + mediapipe)
chorusface-vowel d35 --video path\to\VowelTeacher_HAPPY.mp4 --out output/d35

# Teacher package skeleton
chorusface-vowel teacher-skeleton --root output/teacher

pytest tests/test_vowel_pulsechunk.py tests/test_vowel_expand_emit.py -q
```

## Host API

```http
POST /vowel/utterance
{
  "utterance_id": "u_001",
  "text": "Hi, how are you?",
  "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 1.5}],
  "play": true
}
```

Returns `pulsechunk_b64`, schedules **first-class GA-16** spans on the mouth timeline when `play` is true. Each fire calls `BiomechanicalFace.submit_phoneme` (muscle impulses + jaw), not a coarse 7-vowel collapse.

```powershell
pytest tests/test_vowel_muscle_drive.py tests/test_vowel_pulsechunk.py -q
```

## Still gated by teacher videos

Full SAD/ANGRY (and remaining emotion) teacher clips refine the muscle table / future Model A. Until then, GA-16 uses authored `phoneme_muscles` + priors. Consonants remain on TickFeed §14.
