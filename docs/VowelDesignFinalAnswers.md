# VowelDesign — Final handoff answers (MERGED · ARCHITECTURE CLOSED)

**Status:** Phase-1 **architecture CLOSED** (GPT · Gemini · Claude three-way merge).  
**Output retarget (post-close):** cell-group \(W\) is **not** the NWR delivery path — see [`VowelDesignNWRReconciliation.md`](VowelDesignNWRReconciliation.md). Host API / GA-16 / teacher / A·B *concept* stay; playback = biomech muscle impulses.  
**Questions:** [`VowelDesignFinalHandoff.md`](VowelDesignFinalHandoff.md)  
**Parent:** [`VowelDesign.md`](VowelDesign.md) · Round-2: [`VowelDesignDetailAnswers.md`](VowelDesignDetailAnswers.md)

---

## Sign-off

```text
PHASE-1 DESIGN STATUS: ARCHITECTURE CLOSED

Blockers: NONE for architecture.

Experimental gate (does not reopen design unless FAIL + Plan B):
- D35 / F12 must PASS before full teacher shoot and before training A/B.

Ready for:
- [x] D35 one-day experiment (protocol below)
- [ ] Full teacher shoot (only if D35 pass)
- [x] Implementation of frozen contracts (after Amin says go)
- [ ] MILESTONE_VOWEL_VECTOR_CORE E2E

Anti-thrash: No architectural revision from subjective visual preference alone.
Evidence required: D35, F15 metrics, or hard implementation constraint.

Amin ack: _______________  date: _______________
```

---

## Conflict resolutions (final picks)

| Topic | Pick | Why |
| --- | --- | --- |
| **F1 required fields** | `utterance_id`, `text`, `emotion_track` — **`spans` optional** | Claude + Round-2 G2P; Gemini required-spans rejected |
| **Span field** | `spans[].tag` (GA-16 enum) | Scope-honest; not full IPA “phonemes” |
| **Emotion track** | `{emotion, start_s, end_s}` required per span | Claude; clearer than start-only |
| **F3 unresolved word** | **REST hold** that WordSlice | Never invent AX/schwa shape (Round-2 D25) |
| **F5 header** | New table below (LE + magic + CRC like TPK family) | PulseChunk ≠ TickPackage 64B; nested TPK keeps [`TickPackageHandshake.md`](TickPackageHandshake.md) |
| **F6 WordSlice** | Fixed **12B**, `vowel_ids[6]`, u16 ticks | Gemini; 3-vowel Claude pad too tight for real words |
| **F7 versions** | Optional **12B extended header** after core 32B | Claude hot-path lean; D36 satisfied |
| **F8 KEY crossover** | Δ ≥ **KEY size** | Claude/Round-2; plus mandatory KEY list |
| **F9 9D map** | **Claude / Round-2** Eyes2 Brows2 Mouth1 Lips2 Teeth1 Jaws1 | Matches D7; Gemini MouthCorner/JawShift deferred to optional +3 |
| **F10 plates** | Claude **4×6 = 24** class×emotion stub | Deterministic D10; rename to TickFeed asset names at implement |
| **F12 D35** | % face-width floors + GPT temporal suite + MP/RAFT/SG steps | Claude resolution-independent + GPT continuity + Gemini pipeline |
| **F13 Plan B** | **Primary: real camera** same prompts; **B2:** split all emotions 2×8s | Claude diagnosis value; Gemini B2 if no camera |
| **F15 floors** | Claude ship floors; Gemini stretch as targets | Ship vs aspire |
| **F16 golden** | Claude conversational 3 lines; scripts from G2P when live | Product-real; Gemini short lines as alt smoke |
| **F17 human** | N=10–15; emotion ≥80%; vowel ≥50%; ≥20pt over baseline | Claude; Gemini N=5 preference as optional A/B |
| **F18 memory** | Claude conservative ballpark | Safer than Gemini’s larger ONNX guesses |
| **F20** | GPT + Claude combined | Architecture vs parameter tuning |

---

## Frozen contracts (F1–F20)

### F1–F4 Host API

**Required:** `utterance_id` (string), `text` (string), `emotion_track` (min 1 item).  
**Optional:** `duration_s`, `spans[]`, `words[]`, `speaker_id`.

**Emotion enum (exact):** `NEUTRAL` `HAPPY` `SAD` `SURPRISED` `ANGRY` `THINKING`

**emotion_track item:** `{ "emotion", "start_s", "end_s" }`

**spans item:** `{ "tag": <GA-16>, "start_s", "end_s" }`  
GA-16: `EE IH EY EH AE AA AO OH UH OU AH AX ER AY AW OY`

**Minimal example:**
```json
{
  "utterance_id": "u_001",
  "text": "Hi, how are you?",
  "emotion_track": [{ "emotion": "HAPPY", "start_s": 0.0, "end_s": 1.5 }]
}
```

**G2P:** Face engine owns. Chain: bundled dict → rule G2P → **REST WordSlice** for unresolved word; utterance continues.  
**First KEY:** when `utterance_id` + first emotion entry + first WordSlice vowels resolvable — **do not wait** for full utterance. Latency clock starts at first-word-resolvable.

---

### F5 PulseChunk core header (32 bytes, little-endian)

PulseChunk is the **utterance envelope**. Inner per-tick bodies remain TickPackage (`TPK1`, 64B) per TickPackageHandshake.

| Offset | Type | Name | Notes |
| --- | --- | --- | --- |
| 0 | u32 | `magic` | `'PLS1'` = `0x31534C50` |
| 4 | u8 | `pulse_ver` | start at 1 |
| 5 | u8 | `flags` | bit0=HAS_EXT_HEADER, bit1=IS_SPOOLED, bit2=HAS_WORD_SLICES |
| 6 | u16 | `n_words` | WordSlice count |
| 8 | u64 | `utterance_id_hash` | FNV-1a 64 of string `utterance_id` |
| 16 | u32 | `n_ticks` | total ticks @ tick_hz |
| 20 | u16 | `tick_hz` | 60 |
| 22 | u8 | `primary_emotion` | 0..5 (first / dominant) |
| 23 | u8 | `reserved0` | 0 |
| 24 | u32 | `payload_bytes` | bytes after core (+ext if any) header |
| 28 | u32 | `crc32` | IEEE CRC32 of bytes **0..27** + payload (same “exclude self” style as TPK) |

---

### F6 WordSlice record (fixed 12 bytes)

| Offset | Type | Name |
| --- | --- | --- |
| 0 | u16 | `start_tick` |
| 2 | u16 | `end_tick` |
| 4 | u8 | `n_vowels` (0..6) |
| 5 | u8 | `pause_flag` (0=word, 1=micro-pause) |
| 6 | u8[6] | `vowel_ids` GA-16 index 0..15; unused = `0xFF` |

Words needing &gt;6 vowel slots: split across WordSlices + D16 bridge.

---

### F7 Extended version header (12 bytes, if `flags.HAS_EXT_HEADER`)

Immediately after core 32B:

| Off | Type | Name |
| --- | --- | --- |
| 0 | u16 | `ext_header_len` (=12) |
| 2 | u16 | `teacher_ver` |
| 4 | u16 | `dataset_ver` |
| 6 | u16 | `modelA_ver` |
| 8 | u16 | `modelB_ver` |
| 10 | u16 | `decoder_ver` (`W` / expand) |

`pulse_ver` lives only in core header (not duplicated).

---

### F8 KEY vs Δ

**Rule:** Emit KEY when Δ payload size ≥ KEY size **or** a mandatory KEY event fires.

**Mandatory KEY:** (1) tick 0 · (2) every WordSlice start · (3) hold→release · (4) emotion_track boundary · (5) first tick after TPK_REF spool segment.

---

### F9 ONNX 9D contract (immutable without version bump)

| i | Name | Group | Range | Domain |
| --- | --- | --- | --- | --- |
| 0 | `eye_aperture` | Eyes | [0,1] | sigmoid |
| 1 | `eye_gaze_or_blink` | Eyes | [-1,1] | tanh |
| 2 | `brow_raise` | Eyebrows | [0,1] | sigmoid |
| 3 | `brow_knit` | Eyebrows | [0,1] | sigmoid |
| 4 | `mouth_cavity_gap` | Mouth | [0,1] | sigmoid |
| 5 | `lip_spread` | Lips | [-1,1] | tanh (−round lean / +spread) |
| 6 | `lip_round` | Lips | [0,1] | sigmoid |
| 7 | `teeth_visibility` | Teeth | [0,1] | sigmoid |
| 8 | `jaw_drop` | Jaws | [0,1] | sigmoid |

Optional Phase-1.5 (+3 only if F15 OH/OU fail): lip_protrusion, lip_press, jaw_lateral — **new model/decoder ver**, not silent reshape.

---

### F10 Plate lookup stub (4×6)

Lip/jaw class ∈ {spread, open, round, close} × emotion → plate family.  
Diphthongs use **end** class for plate select; D14 blend handles fine motion.  
Asset names are placeholders — rename to existing TickFeed LOOK plate IDs at implement.

| Class | Vowels (examples) |
| --- | --- |
| spread | EE IH EY |
| open | AA AH AE AO |
| round | OH OU UH |
| close | AX / REST |

Full 24-name stub: see Claude answer in chat history; implementer maps to real plate assets.

---

### F11 \(W\) authoring checklist

1. Load `region_catalog` + `.bds` rest geometry  
2. Per-group centroid + falloff (Gaussian σ≈0.15×radius **or** linear `max(0,1−d/r)`)  
3. Mask so mouth controls never bleed into eyes/brows  
4. Map each of C[0..8] to displacement axis in group  
5. Normalize per cell: \(\sum_k |W_{k,i}| \le 1\)  
6. Store sparse `{avatar_id}.wexpand` (+ version in file header)  
7. Mesh / region / falloff change → bump `decoder_ver` + recompile \(W\)

---

### F12 D35 protocol (GO / NO-GO)

**Clip:** one Veo HAPPY (Part1 if already split).  
**Steps:** MediaPipe 478 @ native fps → resample/interp 60 Hz → optional RAFT lip ROI → Savitzky–Golay(5,2) → metrics.

| Metric | PASS | Hard FAIL |
| --- | --- | --- |
| REST lip-corner jitter | &lt; 0.3% face width (≈&lt;1.5 px @ typ. res) | &gt; 0.8% fw |
| Hold σ (aperture/width) | &lt; 2% face width | &gt; 5% fw |
| EE vs OU peak distance | &gt; 8% face width | &lt; 4% fw |
| Vel / accel continuity (GPT) | no sustained spikes on holds | broken continuity |
| L/R lip symmetry | stable on REST/holds | large asymmetric flicker |
| Lip closure at REST | near-closed stable | flutter / identity drift |
| Optical flow vs landmarks | agree on lip ROI | diverge badly |

**All PASS → GO full shoot. Any hard FAIL → NO-GO → F13. Do not train.**

---

### F13 Plan B

1. **Primary:** real-camera retake, same prompts / structure (isolates Veo vs extraction).  
2. **B2 (no camera):** morphological 2×8s for **all** emotions (10 clips) + heavier temporal filter.  
Neither changes Model A/B architecture.

---

### F14 Shoot package

- Default: **7 clips** (HAPPY, SURPRISED, THINKING ×1; SAD×2; ANGRY×2 morphological).  
- Source fps: **30 preferred, 24 minimum**.  
- If F12 shows rushed end-of-walk holds → also split HAPPY/SURPRISED/THINKING → up to 10.  
- Premise wording: “5 emotions, 7 clips (SAD/ANGRY split)”.

---

### F15 Ship acceptance floors

| Metric | Ship floor | Stretch target |
| --- | --- | --- |
| Pairwise dist EE/OH/OU/AA (any pair) | ≥ 0.25 | ≥ 0.50 (EE–OU) |
| Jaw↔lip_spread \(r\) (open vowels) | &lt; 0.75 | &lt; 0.35 |
| Hold σ (9D) | &lt; 0.08 / ch | &lt; 0.03 |
| Jerk (2nd deriv / tick²) | &lt; 0.15 | &lt; 0.05 |
| E2E first-motion | ≤ 50 ms from first-word-resolvable | 35 ms |

---

### F16 Golden utterances (3)

1. NEUTRAL: “Hello, how can I help you today?”  
2. ANGRY: “I already told you that wouldn't work.”  
3. HAPPY: “That's such great news, congratulations!”  

**Authoritative GA-16 scripts = G2P output once F3 runs** (hand tags are placeholders).

---

### F17 Human eval

N=10–15; muted video; guess emotion (≥80%) + MC vowel at marked times (≥50%, ≥20pt over jaw-pump baseline).

---

### F18 Footprint ballpark

| Asset | Estimate |
| --- | --- |
| Teacher 7×8s | 200–500 MB |
| Dataset A | &lt; 1 MB |
| Dataset B | ~1–3 MB |
| Model A ONNX | ~30–50 KB |
| Model B ONNX | ~50–100 KB |
| \(W\) sparse | ~50–200 KB (mesh-dependent) |
| 5s PulseChunk | ~5–20 KB |

---

### F19 Out of scope Phase 1

Latent `c_t` · consonants in PulseChunk · multi-avatar · mid-flight utterance rewrite · MFA · website TTS · real gaze logic (C[1] idle/blink only) · required W1 lexicon · continuous emotion dial.

---

### F20 Design 100% closed means

Interfaces, ownership, binary contracts, model responsibilities, datasets, timing, acceptance floors, and fallback behavior are frozen. **May tune without redesign:** loss weights, thresholds, attack ticks, falloff radii, plate asset names, SG window, learned weights. **Requires Design Rev 2.0:** 9D set/order, PulseChunk/WordSlice layout, A/B split, group-vs-cell ownership, host required fields, PulseChunk semantics.

---

## Ops: Teacher Package (GPT — adopt)

```text
teacher_package_v1/
  videos/
  landmarks/
  optical_flow/
  pose_targets.npz      # Dataset A
  transfer.npz          # Dataset B
  metadata.json
  checksums
  version
```

One immutable package → all downstream training/eval.

---

## Next sequence (stop new design docs)

1. Freeze acknowledged (this file + VowelDesign §6.3)  
2. Run **F12 / D35**  
3. If PASS → full shoot → extract into Teacher Package  
4. Dataset A → train Model A → static separability  
5. Dataset B → train Model B  
6. One PulseChunk → Fabric → NWR → GPU E2E  
7. `MILESTONE_VOWEL_VECTOR_CORE`
