# VowelDesign — detail / accuracy questions (round 2)

**Purpose:** Paste this file (or Part A) to Gemini / GPT / Claude for a **precision review**.  
**Status:** **D1–D35 answered by Gemini + Claude** (merged) · GPT **D36–D42** adopted · Round-2 closed for design.  
**Answers:** [`VowelDesignDetailAnswers.md`](VowelDesignDetailAnswers.md)  
**Context:** [`VowelDesign.md`](VowelDesign.md) · merge [`VowelDesignReviewMerge.md`](VowelDesignReviewMerge.md) · prompts [`VowelTeacherPrompts.md`](VowelTeacherPrompts.md)

**Ask the model to:** answer each question with a concrete recommendation, numbers where possible, and call out conflicts with the locked premises below.

---

## Locked premises (do not reopen unless broken)

| Premise | Value |
| --- | --- |
| Vowels | GA-16 tags (EE…OY) |
| Articulators | 6 groups = NWR objects → all known cells |
| Emotions | 6 tags; teacher = 5 videos (NEUTRAL from REST-under-emotion) |
| Clock | 60 Hz (~16.67 ms), state 0 at start |
| Datasets | A = targets (576 space); B = full-set transfer (attack+hold+release) |
| Wire | PulseChunk → WordSlices → TickPackage KEY/Δ on **CHORUS Fabric**; optional `c_t` lane A |
| ML Phase 1 | Two models: A state, B transfer; train **6-group controls** + frozen group→cell expand |
| ML Phase 2 (later) | Latent facial space |
| Product | No TTS ownership; LLM/API sync; NWR rigid geometry + vectorized motion |
| Teacher | SAD/ANGRY = 2×8s; REST = mouth 0 + eyes/brows keep emotion |

**Pipeline:**

```text
LLM utterance_id + GA-16 script + emotion_track + timings
  → Model A (state) + Model B (path)
  → PulseChunk @ 16.7 ms
  → CHORUS Fabric
  → NWR ROI cells (.bds)
  → GPU (photo/plates rigid; motion = Δ)
```

---

# Part A — Questions (answer these)

## Block 1 — Teacher data accuracy

**D1.** For Side B collect from Veo teachers, what **landmark + dense motion** stack do you recommend (MediaPipe only vs MediaPipe + Farneback/RAFT/etc.), and what **minimum hold length in ticks** is required before a frame is accepted as a Dataset A peak?

**D2.** How should we **detect vowel hold peaks** automatically (max mouth aperture? landmark stability window? script timestamps only)? Give a false-positive / false-negative strategy.

**D3.** REST-under-emotion: propose a numeric definition — which of the 6 group channels must be ≈0, which may stay high — and how to validate that ANGRY-REST ≠ HAPPY-REST in Dataset A.

**D4.** When teacher measurement **disagrees** with the §5.2 pose prior table, what win rule? (always measure / blend / reject take)

**D5.** Phase-2 “3 performances per emotion”: how to **align** three takes of the same vowel for training (DTW on landmarks? script time only? reject outliers)?

**D6.** Exact **Part A/B vowel split** for SAD/ANGRY 2×8s — keep 1–8 / 9–16, or regroup by open vs round to reduce mid-clip difficulty for Veo?

---

## Block 2 — Group controls & NWR cells

**D7.** Specify the **Phase-1 control vector** per tick: how many floats per group (eyes, brows, mouth, lips, teeth, jaws)? Propose dims and semantic meaning (e.g. lips = [spread, round]).

**D8.** Design the **frozen group→cell expand**: linear matrix? radial falloff from region centroid? neighbor graph? How is it authored from `region_catalog` / `.bds`?

**D9.** Which NWR **channels** does PulseChunk write per cell in Phase 1 (vx, vy only? openness? lid?)? Align with TickFeed phase-1 velocity mask.

**D10.** How do **LOOK plates** (open/smile/surprise) combine with cell Δ so we don’t double-drive the mouth (relationship to TickFeed §14 hard-snap)?

**D11.** Teeth visibility: plate alpha vs cell Δ — which owns “teeth show” for AA vs OU?

---

## Block 3 — Models A & B (numbers)

**D12.** Model A architecture: input dims (GA-16 one-hot + emotion + ?); hidden size; output = group control vector. Recommend a small net suitable for **~5–15 minutes** of teacher video.

**D13.** Model B: predict **absolute group state** vs **velocity/residual**? Loss terms and weights (endpoint L2, smoothness, jaw limit).

**D14.** Diphthong encoding: exact tensor layout for (start, end, blend∈[0,1]) and how blend advances over the vowel’s N ticks (linear vs ease-in-out).

**D15.** Emotion-conditioned attack: formula or table for attack_ticks(emotion) beyond ANGRY≈4 / SAD≈8–10 / default 6.

**D16.** Coarticulation: exact rule for “compatible vs conflicting” lip pairs; when to insert mouth-neutral bridge (1–3 ticks) vs 4-tick cosine blend. Give a small conflict matrix (EE×OU, AA×OU, …).

**D17.** Confidence: how to compute confidence for A and B; thresholds for fallback (smooth / prior / W1).

---

## Block 4 — PulseChunk & Fabric binary accuracy

**D18.** Propose a **fixed binary header** for PulseChunk metadata (byte layout): `utterance_id`, `emotion_track` summary, `n_words`, `n_ticks`, `tick_hz`, flags.

**D19.** WordSlice on-wire or in-band only? If on-wire, header fields: `start_tick`, `end_tick`, `vowel_ids[]`, `pause_flag`.

**D20.** KEY emission policy with numbers: at t=0; each WordSlice start; hold→release; when Δ payload would exceed **X** bytes vs KEY. Suggest X.

**D21.** Lane A `c_t` float32[64]: who produces it in Phase 1 (PCA of group controls? learned encoder?)? Or defer `c_t` until Phase 2 latent?

**D22.** Inline vs TPK_REF: confirm ~300 ticks / ~5 s threshold or propose better; spool naming + CRC rules if needed.

**D23.** End-to-end latency budget breakdown (LLM → models → Fabric → NWR apply → present) targeting **&lt; 50 ms** face path after utterance known. What’s realistic?

---

## Block 5 — LLM / API contract

**D24.** JSON (or protobuf) schema for one utterance from the host API: required fields (`utterance_id`, `text`, `emotion_track`, `phonemes[]` or GA-16 spans with `start_s`/`end_s`). Mark optional fields.

**D25.** If the LLM omits phonemes, fallback order: G2P dict → rule → Model A with grapheme features? Or refuse to animate?

**D26.** Multi-emotion track: wire format for mid-utterance emotion change (time or tick keyed). Default when only one emotion is sent.

**D27.** Timing authority: host provides absolute seconds vs tick indices vs “estimate from text rate WPM”. Pick one primary.

---

## Block 6 — Runtime stitch & acceptance

**D28.** WordSlice boundary **3-tick crossfade**: exact weights per tick; applied in group space or cell space?

**D29.** Between-word micro-REST: max ticks; which groups zero vs which keep emotion.

**D30.** Acceptance tests (numeric): minimum pairwise silhouette distance for EE/OH/OU/AA; max jaw-pump metric (correlation of all open vowels’ lip width); hold stability (std of group vector over hold window).

**D31.** Golden utterance for QA: propose 3 English lines (neutral / angry / happy) + expected vowel scripts.

**D32.** Milestone name + Phase-1 done criteria (checklist of artifacts: videos, npz, model weights, Fabric smoke).

---

## Block 7 — Scope & risks

**D33.** Consonants (PP/FF/TH): keep entirely on old TickFeed path for Phase 1, or reserve PulseChunk slots now?

**D34.** Identity transfer: what breaks if we swap `.bds` / photo but keep Model A/B trained on blonde teacher?

**D35.** Biggest accuracy risk in this design — pick one and propose a mitigation experiment (one day of work).

---

# Part B — How to answer (instructions for the helper model)

For each **D1–D35**:

1. **Recommendation** (concrete)  
2. **Numbers** (ticks, dims, bytes, thresholds) when relevant  
3. **Conflicts** with locked premises (if any)  
4. **Priority:** P0 must-decide before shoot / P1 before train / P2 before ship  

End with a short **“Top 5 decisions Amin should lock next”** list.

---

# Part B2 — GPT follow-ons (D36–D42) — adopted in answers

These were proposed after GPT’s meta-review of the question packet. Locked/adopted in [`VowelDesignDetailAnswers.md`](VowelDesignDetailAnswers.md).

| ID | Topic |
| --- | --- |
| **D36** | Version fields on PulseChunk (teacher / dataset / modelA / modelB / decoder \(W\) / pulse) |
| **D37** | Avatar portability scope for Phase 1 (same teacher vs new \(W\) only) |
| **D38** | Runtime failure recovery (NaN / low conf — never invent) |
| **D39** | Memory footprint estimate table |
| **D40** | Temporal acceptance metrics (jerk / velocity continuity) |
| **D41** | Incremental / appendable PulseChunks (Phase-2) |
| **D42** | Human silent-video eval protocol |

---

# Part C — Pointers (optional reading)

| Doc | Why |
| --- | --- |
| [`VowelDesign.md`](VowelDesign.md) §3.2 | NWR scoping |
| [`VowelDesign.md`](VowelDesign.md) §5.2 | Pose prior table |
| [`VowelDesign.md`](VowelDesign.md) §6.1–6.2 | Teacher + seven gaps |
| [`TickFeedDesign.md`](TickFeedDesign.md) §6.2 | CHORUS Fabric / TickPackage |
| [`TickPackageHandshake.md`](TickPackageHandshake.md) | Binary KEY/Δ |
| [`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) | Video → cells |
| [`NWRDataDesign.md`](NWRDataDesign.md) | Layers / region catalog |
