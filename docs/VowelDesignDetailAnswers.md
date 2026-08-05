# VowelDesign — Round 2 answers (engineering locks)

**Status:** three-way merge — Gemini · Claude · GPT (D36–D42).  
**Do not implement** until Amin asks.  
**Questions:** [`VowelDesignDetailQuestions.md`](VowelDesignDetailQuestions.md)  
**Parent:** [`VowelDesign.md`](VowelDesign.md)

---

## Top 5 to lock next (merged)

1. **D35 first** — one-day Veo landmark stability (+ optional RAFT / Savitzky–Golay) before full shoot  
2. **D7** — Phase-1 **9D** group vector (Claude); optional +3 if D30 fails OH/OU  
3. **D24 / D25 / D27** — host utterance schema + G2P fallback + timing authority  
4. **D10 / D11** — LOOK plate lookup vs cell Δ residual split  
5. **D6 + D3** — morphological Part1/Part2 + REST-under-emotion nuance (shoot-day)

---

## Conflict scorecard (must-read)

| ID | Gemini | Claude | **Merged lock** |
| --- | --- | --- | --- |
| **D1 hold** | ≥12 ticks | ≥6 ticks | **≥6** gate; prefer **≥12** for high-confidence Dataset A peaks |
| **D1 stack** | MP + RAFT hybrid primary | MP primary; dense flow validate | **MP primary**; RAFT/Farneback validate + lip-margin velocity; note 24–30 fps → 60 Hz **interpolated** |
| **D3 REST** | mouth 0±0.05; upper L2≥0.40 | small HAPPY/ANGRY lip bias OK; upper L2≥0.15 | Mouth≈0; **tiny** static lip bias OK for HAPPY/ANGRY; upper L2 **≥0.15** floor, target **≥0.40** after shoot |
| **D4 vs prior** | clamp to prior±σ | measure wins; reject implausible | **Measure wins**; reject if >1.2× take range or low MP conf; prior = soft review flag only |
| **D6 AY/ER** | AY∈P1, ER∈P2 | ER∈P1, AY∈P2 | **Claude**: ER with central/open; AY with diphthongs (see below) |
| **D7 dims** | **12D** | **9D** | **Phase-1 = 9D**; grow to 12 only if silhouette tests fail |
| **D12** | 26→128→12 | 22→64→9 | **22→64→64→9** (~6.5k); optional context 4 later |
| **D18 endian** | big-endian 32B | little-endian ~19B | **Little-endian** (match TickPackage); padded **32B** header + CRC |
| **D25 missing phonemes** | never refuse; Model A grapheme OK | G2P then **hold REST** | Dict → rule G2P → **REST hold** for that span (never invent vowel shape) |
| **D29 micro-REST** | ≤6 ticks full 0 | 2–3 ticks partial | **≤3** ticks @ ~50% toward REST; **4–6** fuller REST; **>6** = sentence pause |
| **D33 consonants** | reserve PulseChunk override slots | TickFeed only; **don’t reserve** | **TickFeed §14 only**; extend schema later |
| **D32 name** | `MILESTONE_VOWEL_VECTOR_CORE` | “Measured Vowel Motion” | Code: **`MILESTONE_VOWEL_VECTOR_CORE`**; title: Measured Vowel Motion |

---

## Block 1 — Teacher

| ID | Merged lock | Pri |
| --- | --- | --- |
| **D1** | MediaPipe Face Landmarker (478) primary @ source fps; resample/interp to 60 Hz (state that mid-frame ticks are **interpolated**). Dense flow (RAFT if GPU else Farneback) on face ROI to **validate** motion + catch landmark jitter. Hold gate ≥**6** ticks; Dataset A “strong peak” prefer ≥**12** with mouth variance &lt;0.5 px | P0 |
| **D2** | Script time ±**150 ms** (~±9 ticks); peak = max distance-from-REST in landmark space (aperture for open; lip-width for spread/round) that passes hold gate; also check aperture velocity near zero. FP: reject if landmark vel &gt;1.5× clip median or lip RAFT &gt;0.8 px/tick. FN: closest-to-script frame + **low-confidence flag** (keep sample) | P0 |
| **D3** | Vowel channels (mouth/lips/teeth/jaw) ≈0; eyes/brows keep emotion. Optional tiny static bias: HAPPY slight corner lift, ANGRY slight thin press. Validate \(\mathrm{L2}(\mathrm{ANGRY},\mathrm{HAPPY})_{\mathrm{upper}} \ge 0.15\) (floor); aim ≥0.40 post-tune | P0 |
| **D4** | Teacher measurement is truth over §5.2 prior. Reject frame if channel &gt;1.2× take max or MediaPipe low-conf — do **not** average garbage. Prior may flag for visual review only | P1 |
| **D5** | Phase-2: segment-level DTW on landmarks (±5 tick slack); discard take if cost &gt; **2×** (Claude) / **2.5×** (Gemini) median pairwise — use **2.5×** for fewer false discards | P2 |
| **D6** | Morphological 2×8s — **Part1** EE IH EY EH AE AA **AX ER**; **Part2** AO OH UH OU AH **AY** AW OY | P0 |

---

## Block 2 — Groups / NWR

| ID | Merged lock | Pri |
| --- | --- | --- |
| **D7** | **9 floats/tick:** Eyes[aperture, blink/gaze], Brows[raise, knit], Mouth[cavity_gap], Lips[spread, round], Teeth[visibility], Jaws[drop]. Optional +3 later: lip protrusion, lip press, jaw lateral / mouth corner — only if D30 can’t separate OH/OU | P0 |
| **D8** | Frozen sparse expand: per-axis Gaussian/radial falloff × `region_catalog` mask; authored once per `.bds`; \(W \in \mathbb{R}^{9 \times N}\) | P0 |
| **D9** | Phase-1: **vx, vy** per cell (+ openness/plate α for mouth/teeth). Align TickFeed phase-1 velocity mask. Eyes continuous stream stays simple; blink may stay TickFeed eye path | P0 |
| **D10** | LOOK plates = coarse discrete (KEY / vowel-class change); cell Δ = fine continuous on plate base. **Plate select = deterministic lookup** `(vowel_class, emotion)` — not a second model | P0 |
| **D11** | Plate α = coarse teeth show; cell/lip Δ = fine cover/exposure (same split as D10) | P1 |

### D7 layout (Phase-1 locked)

| Group | Dims | Channels |
| --- | --- | --- |
| Eyes | 2 | aperture, blink/gaze |
| Eyebrows | 2 | raise, knit |
| Mouth | 1 | cavity_gap |
| Lips | 2 | spread, round |
| Teeth | 1 | visibility |
| Jaws | 1 | drop |
| **Total** | **9** | |

---

## Block 3 — Models

| ID | Merged lock | Pri |
| --- | --- | --- |
| **D12** | Model A: one-hot 16+6 = **22D** → MLP `22→64→64→9`, ReLU/GELU, dropout 0.2–0.3; ~6.5k params. Optional +4 context later (stress, WPM, prev/next) | P1 |
| **D13** | Predict **Δ residual**; integrate from state0. Loss: endpoint L2 (w≈1–10), smoothness/jerk (w≈0.1–0.5), jaw hinge limit (w≈0.5–5). Gemini’s heavier endpoint OK if holds undershoot | P1 |
| **D14** | Diphthong: start9 + end9 + blend; default **smoothstep** \(3\tau^2-2\tau^3\) over vowel ticks (Claude). Alternate Gemini cosine window 0.2N…0.8N if attacks look soft | P1 |
| **D15** | Attack priors (correct from Dataset B after shoot): ANGRY **4**, SURPRISED **4**, HAPPY **5**, NEUTRAL/THINKING **6**, SAD **9** (clamp 3–12) | P1 |
| **D16** | Compatible → 4-tick cosine; spread↔round conflict → **2-tick** mouth-neutral bridge (EE↔OU). Open↔close alone → 1-tick OK | P1 |
| **D17** | A: proximity to measured (vowel,emotion) cells. B: small ensemble variance or Δ-smoothness. Cascade: high → full A/B; mid → A + spline; low (&lt;0.25–0.5) → W1 or **REST hold** (align D25/D38 — never invent) | P1–P2 |

---

## Block 4 — PulseChunk / Fabric

| ID | Merged lock | Pri |
| --- | --- | --- |
| **D18** | Fixed **32-byte little-endian** header (TickPackage family): utterance_id u64, emotion/track fields, tick_hz, n_words, flags, total_ticks, payload_size, **crc32**, pad | P0–P1 |
| **D19** | Explicit WordSlice records: start/end tick, pause_flag, vowel_ids (fixed pad to 6 × uint8 OK for Phase-1) | P1 |
| **D20** | KEY at t=0; WordSlice starts (esp. after pause); hold→release; emotion shift; or when Δ size ≥ **KEY size** (Claude self-threshold) / &gt;0.6×KEY (Gemini) — use **≥ KEY size** | P0 |
| **D21** | Phase-1: **defer learned `c_t`**. Lane A optional = same **9D** group vector (or PCA 4–6). float32[64] reserved for Phase-2 latent | P1 |
| **D22** | ≤**300 ticks** inline; else TPK_REF spool `{utterance_id}_{crc32}` / part index; CRC before play past inline prefix | P1 |
| **D23** | Compute path ~5–15 ms realistic (tiny nets). Keep Gemini stage table as planning budget; **Fabric/topology is the real risk** — smoke-test, don’t only estimate. Present slot ~16.7 ms | P0 |

---

## Block 5 — LLM / API

| ID | Merged lock | Pri |
| --- | --- | --- |
| **D24** | Required: `utterance_id`, `text`, `emotion_track[{emotion,start_s,end_s?}]`. Preferred: `phonemes`/`spans[{tag,start_s,end_s}]`. Optional: `duration_s`, `words[]`, `speaker_id` | P0 |
| **D25** | Missing phonemes: **CMUdict/G2P dict → rule G2P → REST hold** for unresolved spans. Do **not** smuggle grapheme guessing into Model A. Utterance still plays (never hard-fail API) | P0 |
| **D26** | Time-keyed `emotion_track` in seconds; single emotion → one span covering utterance | P1 |
| **D27** | Host **absolute seconds** primary; `tick = round(s * 60)`. WPM (~150) only if no timings at all | P0 |

---

## Block 6 — Stitch / QA

| ID | Merged lock | Pri |
| --- | --- | --- |
| **D28** | 3-tick crossfade in **9D group space** then expand: 0.75/0.25 → 0.5/0.5 → 0.25/0.75 | P0 |
| **D29** | Micro-pause ≤3 ticks: lower face ~40–60% toward REST; eyes/brows hold emotion. 4–6 ticks: fuller mouth REST. &gt;6: sentence pause span | P1 |
| **D30** | Pairwise lips distance EE/OH/OU/AA ≥**0.30** floor (Claude); stretch goals Dist(EE,OU)≥0.60, Dist(OH,AA)≥0.45 (Gemini). Jaw↔lip-width \(r &lt; 0.70\) floor / \(r &lt; 0.35\) stretch. Hold σ &lt; **0.05** floor / **0.02** stretch. + jerk continuity (D40) | P1 |
| **D31** | Golden suite (both sets OK): Gemini fox/dog + angry/happy lines; Claude help/angry/happy conversational. Prefer machine G2P scripts once D25 runs | P1–P2 |
| **D32** | **`MILESTONE_VOWEL_VECTOR_CORE`** (“Measured Vowel Motion”): videos → MP extract → A/B datasets → train A/B → \(W\) authored → PulseChunk round-trip → Fabric smoke → D30 pass/gap doc | P0 |

---

## Block 7 — Scope / risks

| ID | Merged lock | Pri |
| --- | --- | --- |
| **D33** | Consonants **only** on existing TickFeed §14 hard-snap path in Phase 1. **Do not** reserve PulseChunk consonant slots yet | P0 |
| **D34** | Models on normalized 9D may transfer as a start; **\(W\) must be re-authored** per `.bds`. Phase-1 product = teacher identity | P1 |
| **D35** | Biggest risk: **Veo landmark / texture jitter** corrupting lip spread/round. **1-day:** generate HAPPY (or Part1); MediaPipe; eyeball lip corners on EE/OU/AA/AH; optional RAFT + Savitzky–Golay(5,2) SNR. **Go/no-go before full shoot** | P0 |

---

## GPT additions D36–D42 (adopt)

| ID | Topic | Adopted lock | Pri |
| --- | --- | --- | --- |
| **D36** | Versioning | PulseChunk carries teacher/dataset/modelA/B/decoder(\(W\))/pulse versions | P0 |
| **D37** | Avatar portability | Phase-1: same teacher identity; other faces need new \(W\) | P1 |
| **D38** | Failure recovery | NaN/low conf → D17; last good or mouth REST + hold emotion; never invent | P0 |
| **D39** | Memory footprint | TBD estimate table before mobile/browser claim | P1 |
| **D40** | Temporal metrics | Jerk / velocity continuity with D30 | P1 |
| **D41** | Incremental chunks | Phase-1 full utterance only; Phase-2 appendable | P2 |
| **D42** | Human eval | Silent video: guess vowel / emotion / word vs baseline | P0 |

---

## Next gate

Final handoff merged → [`VowelDesignFinalAnswers.md`](VowelDesignFinalAnswers.md) (**ARCHITECTURE CLOSED**).  
Next: **D35 / F12** → (pass) shoot → Teacher Package → train A then B → one PulseChunk E2E.
