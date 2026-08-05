# VowelDesign — NWR Reconciliation + Step 0

**Status:** design locked for output retarget · Step 0 **complete** (2026-08-05)  
**Parent:** [`VowelDesign.md`](VowelDesign.md) · freeze: [`VowelDesignFinalAnswers.md`](VowelDesignFinalAnswers.md)

Teacher shoot / D35 / host API continue. Cell-group expand \(W\) is **not** the primary delivery path.

---

## 1. The wall (confirmed in AIFace)

VowelDesign F9–F11 assumed six cell-writable groups + expand matrix \(W\).

Real ChorusFace:

- Master Lock at seed: eyes/nose/skull/cheeks/forehead/jaw/chin locked; mouth cavity / lip interior unlocked.
- `constraint.comp` vetoes AI writes to locked cells.
- Expression on locked regions = **muscle spring-damper** (`biomechanics/`) → render uniforms / warp — not cell KEY geometry writes.
- Speech injects **`MuscleImpulse`** via `BiomechanicalFace.submit_phoneme` (see `app._fire_impulse`).
- TickFeed still owns FIELD velocity for unlocked mouth when LOOK authority is on; that is **mouth-only**, not brows/jaw cells.

**Conclusion:** PulseChunk-as-6-group-cell-writer would be silently vetoed for 4/6 groups. Fix = retarget output.

---

## 2. Fix (one sentence)

**PulseChunk / vowel compose targets `BiomechanicalFace` muscle impulses + jaw/emotion/eye subsystems; cell velocity impulses only for unlocked mouth interior when needed.**

---

## 3. Step 0 findings (this repo)

| Question | Finding |
| --- | --- |
| Muscle count | **39** named muscles in `face_definition.json` (~20 groups with L/R) |
| Solver input | `MuscleImpulse(tick, muscle, strength, duration, falloff, priority, source)` → `MuscleImpulseQueue` → `MuscleSolver` spring-damper |
| Phoneme table | `phoneme_muscles` already has REST/CLOSED + consonants + **AH AA EH IH EE OH OU** (+ aliases IY/OO/UW) |
| GA-16 gaps (before patch) | Missing first-class: **AE AO UH AX ER EY AY AW OY** (often collapsed by speech digraphs) |
| Jaw | `JawSystem.set_speech_target` + `PHONEME_JAW_TARGET` in `intent.py` |
| Emotion hold | `EmotionSystem.from_label` + continuous axes; not utterance-timed sustain yet — vowel path calls `from_label` per phoneme; good enough Phase-1 |
| Blinks | **`EyeSystem` already owns blinks** — do not invent a parallel cell blink stream for product |
| Teeth / plates | LOOK plates + `OPEN_TOOTH_VISEMES` still exist for TickFeed path; part atlas / uniforms also derive openness from muscles — prefer muscle geometry; plates as overlay when TickFeed LOOK authority is on |
| Model B | Spring-damper already shapes attack/release from impulse duration — **Model B may shrink** to timing/strength modulation; keep A/B split conceptually, implement B as impulse schedule over existing solver first |
| Wire integration | Same point as today: schedule spans → `_fire_impulse` → `submit_phoneme` — **not** a parallel cell Fabric writer for brows/jaw |

### Revised estimate

| Work | New vs reuse |
| --- | --- |
| Host API / G2P / GA-16 compose | Keep (done) |
| Teacher / D35 | Keep |
| Extend `phoneme_muscles` + jaw for full GA-16 | **New data** (small) |
| Drive face via existing biomech | **Reuse** — primary fix |
| Group→cell \(W\) for eyes/brows/jaw | **Drop as primary** |
| PLS1 cell KEY/Δ for full face | **Deprioritize**; optional mouth-only later |
| Model A as muscle-target predictor | Phase-1.5 — table + teacher refine first |
| Model B full residual net | Only if solver curves fail acceptance |

---

## 4. Locked premise delta (merge into VowelDesign)

| Premise | New value |
| --- | --- |
| Output target | Muscle impulses + jaw/emotion/eyes; mouth cell ±4 optional |
| Expand \(W\) | Not required for Phase-1 vowel delivery |
| Plate lookup F10 | Not a second system — use biomech geometry; TickFeed LOOK optional overlay |
| PulseChunk role | Utterance clock + GA-16/emotion schedule (+ optional debug 9D); **playback via biomech** |
| Model A/B dims | Phase-1: drive existing `phoneme_muscles`; learn later if needed |
| Blinks | `EyeSystem` (product); HTML demo blink overlay is visualization-only |

---

## 5. Implementation order (done / next)

1. [x] Step 0 report (this file)  
2. [x] Extend GA-16 → `phoneme_muscles` + jaw + speech pass-through  
3. [x] Vowel app path passes GA-16 tags (no coarse collapse)  
4. [x] `vowel/muscle_drive.py` offline biomech playback helper  
5. [ ] Tomorrow: ingest remaining SAD/ANGRY teacher videos → refine table from landmarks  
6. [ ] Acceptance on live avatar (EE/OU/AA distinguishable on photo face)

---

## 6. Dev brief

Before expanding PulseChunk cell binary: **drive `BiomechanicalFace.submit_phoneme` with GA-16 + duration + emotion**. Teacher videos improve the muscle table / future Model A — they do not justify writing locked cells.
