# VowelDesign — Phase-1 implementation (vowelBrnach)

**Branch:** `vowelBrnach`  
**Freeze:** [`VowelDesignFinalAnswers.md`](VowelDesignFinalAnswers.md)  
**Package:** `src/chorusface/vowel/`

## What’s implemented

| Piece | Module / surface |
| --- | --- |
| GA-16 + 9D F9 contract | `schema.py`, `priors.py` |
| Host utterance JSON (F1) | `utterance.py` |
| G2P → REST fallback (F3) | `g2p.py`, `data/ga16_dict.json` |
| Model A MLP 22→64→64→9 | `model_a.py` |
| Model B attack/hold/release + diphthong + bridges | `model_b.py` |
| PulseChunk PLS1 + WordSlice | `pulsechunk.py` |
| Compose pipeline | `pipeline.py` |
| Expand matrix \(W\) | `expand.py` |
| Plate lookup stub | `plates.py` |
| TickPackage KEY/Δ emit | `tick_emit.py` |
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

Returns `pulsechunk_b64`, schedules GA-16→viseme events on the existing mouth timeline when `play` is true.

## Still gated by D35

Full teacher shoot + Dataset A/B from video are **not** substituted. Until D35 passes, Model A trains from design priors (§5.2). Consonants remain on TickFeed §14.
