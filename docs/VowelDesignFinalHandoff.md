# VowelDesign — Final handoff (close to 100%)

**Purpose:** One last precision pass. Answer **F1–F20** to freeze every remaining design gap. After answers are merged, Phase-1 design is **closed** — next step is D35 experiment + implementation, not more architecture review.

**Status:** design only — **do not implement** until Amin asks.  
**Answers:** [`VowelDesignFinalAnswers.md`](VowelDesignFinalAnswers.md) — **ARCHITECTURE CLOSED** (GPT · Gemini · Claude). Next: D35, not more questions.  
**Already locked (do not reopen):** [`VowelDesign.md`](VowelDesign.md) · [`VowelDesignDetailAnswers.md`](VowelDesignDetailAnswers.md) · [`VowelTeacherPrompts.md`](VowelTeacherPrompts.md)

**Ask the model (or Amin) to:** for each F-item give (1) Recommendation, (2) exact numbers/schema/bytes, (3) Conflicts with locks, (4) Priority. End with **“Sign-off: Phase-1 design CLOSED / NOT CLOSED”** and list any blockers.

---

## Locked premises (readonly)

| Premise | Value |
| --- | --- |
| Vowels | GA-16 |
| Groups | 6 → **9D** control / tick (D7) |
| Emotions | 6; teacher = 5 Veo (SAD/ANGRY morphological 2×8s) |
| Clock | 60 Hz; state 0 start |
| Datasets | A targets · B full-set transfer |
| Models | A state MLP 22→64→64→9 · B residual Δ |
| Wire | PulseChunk → WordSlices → KEY/Δ on Fabric; LE; defer learned `c_t` |
| Fallback | G2P → REST hold; never invent |
| Consonants | TickFeed §14 only in Phase 1 |
| Milestone | `MILESTONE_VOWEL_VECTOR_CORE` |

---

# Part A — Closeout questions (answer these)

## Block F1 — Host API freeze (product boundary)

**F1.** Paste the **final canonical JSON Schema** (draft-07 or similar) for one utterance. Required vs optional fields must be explicit. Include one **minimal valid example** and one **full example** (multi-emotion + phoneme spans + words).

**F2.** Confirm phoneme field name: `phonemes` vs `spans` — pick **one** forever for Phase 1. Map emotion enum strings exactly (case, spelling).

**F3.** When host sends **text only** (no phonemes, no timings): exact fallback chain + who owns G2P (host vs face engine) + what the face emits for unresolved words.

**F4.** Sync contract: does the face wait for a full utterance JSON before first KEY, or can it start after first WordSlice? State the rule for &lt;50 ms first-motion.

---

## Block F2 — Binary wire freeze (Fabric)

**F5.** Give a **byte-offset table** for the PulseChunk 32-byte little-endian header (field, type, offset, size). Must not fight [`TickPackageHandshake.md`](TickPackageHandshake.md) endianness/CRC style.

**F6.** WordSlice on-wire: fixed 12-byte vs length-prefixed — pick one; give offset table + max vowels per word pad rule.

**F7.** Version fields (D36): where do `teacher_ver`, `dataset_ver`, `modelA_ver`, `modelB_ver`, `decoder_ver`, `pulse_ver` live — inside the 32B header, extended header, or first KEY payload? Exact layout.

**F8.** KEY vs Δ rule in one sentence + numeric crossover (Δ ≥ KEY size). List every mandatory KEY event.

---

## Block F3 — Control / plate / expand freeze

**F9.** Confirm the **9D channel order** as a fixed index map `C[0]…C[8]` (names + ranges + tanh/sigmoid domain). This becomes the ONNX I/O contract.

**F10.** Plate lookup table stub: for each of {spread, open, round, close} × {emotion or NEUTRAL}, which LOOK plate family activates? Enough rows to implement Phase 1 without a second model.

**F11.** Group→cell expand \(W\): authoring checklist for one avatar (inputs from `region_catalog` / `.bds`, normalize rule, where binary \(W\) is stored, version bump rule when mesh changes).

---

## Block F4 — Teacher / D35 go–no-go (must close before shoot)

**F12.** Write the **D35 one-day experiment protocol** as a checklist: clip choice, extract steps, metrics, pass/fail thresholds (numeric). Include: landmark jitter px, hold stability, EE vs OU separable in landmark space.

**F13.** If D35 **FAILS**, what is Plan B within one week? (real camera retake of same prompts / different generator / lower ambition Dataset A / other). Pick one primary Plan B.

**F14.** Final shoot package: confirm clip count (5 vs 7), fps request to Veo, and whether HAPPY/SURPRISED/THINKING also split to morphological 2×8s if D35 shows rushed holds.

---

## Block F5 — Acceptance / QA freeze

**F15.** Final Phase-1 **numeric acceptance gate** table (use floor numbers that ship, not stretch goals): pairwise distances, jaw-pump correlation, hold σ, jerk limit (D40).

**F16.** Golden utterance suite: finalize **exactly 3 lines** + expected GA-16 scripts (tags only). Prefer conversational product lines over pangrams if both conflict.

**F17.** Human eval (D42): minimum protocol — N raters, task (guess vowel / emotion / word), pass bar vs current jaw-pump baseline.

---

## Block F6 — Ops / scope freeze

**F18.** Memory footprint ballpark (D39): fill a table — teacher video GB, Dataset A/B npz MB, Model A/B ONNX KB, \(W\) MB, one 5s PulseChunk KB. Mark “unknown” only if impossible without measuring; else estimate.

**F19.** Explicit **out of scope for Phase 1** list (confirm): latent `c_t`, consonants in PulseChunk, multi-avatar, incremental chunks, MFA aligner, website TTS. Add anything else that must stay out.

**F20.** Definition of **design 100% closed**: one paragraph. What may still change after D35/prototype without reopening architecture? What requires a new design rev?

---

# Part B — Sign-off template (fill at end)

```text
PHASE-1 DESIGN STATUS: CLOSED | NOT CLOSED

Blockers (if NOT CLOSED):
- …

Ready for:
- [ ] D35 one-day experiment
- [ ] Full teacher shoot (only if D35 pass)
- [ ] Implementation toward MILESTONE_VOWEL_VECTOR_CORE

Amin ack: _______________  date: _______________
```

---

# Part C — How to use

1. Paste **Part A + locked premises** to Gemini / GPT / Claude (or answer yourself).  
2. Merge answers into [`VowelDesignFinalAnswers.md`](VowelDesignFinalAnswers.md) (create when answers arrive).  
3. Update [`VowelDesign.md`](VowelDesign.md) §6.2–6.3 / §8 checkboxes.  
4. Stop design review. Run **F12/D35**. Then implement.

**Do not** invent new architecture in this round — only freeze remaining details.
