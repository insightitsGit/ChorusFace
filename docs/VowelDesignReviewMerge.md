# VowelDesign — review merge (Gemini · GPT · Claude)

**Date:** 2026-08-05  
**Status:** design synthesis — **do not implement** until Amin asks.  
**Inputs:** Gemini Part A+B · GPT architect review · Claude Q1–Q7 review  
**Parents:** [`VowelDesign.md`](VowelDesign.md) · [`VowelDesignHelp.md`](VowelDesignHelp.md)

**Round 2 (precision D1–D35):** Gemini + Claude answered; GPT added D36–D42 → three-way merge in [`VowelDesignDetailAnswers.md`](VowelDesignDetailAnswers.md). This Round-1 file stays the Q1–Q7 source; Round-2 numbers live in DetailAnswers / §6.2–6.3 of VowelDesign.

This file records **what each model said** and the **reconciled lock** we adopt
into VowelDesign. Where models disagree, the merge picks one path and notes why.

---

## Scorecard (consensus)

| Theme | Gemini | GPT | Claude | Merge |
| --- | --- | --- | --- | --- |
| A vs B split | yes | yes (strong) | yes | **Keep** |
| PulseChunk | yes | yes (strong) | yes | **Keep** + hierarchical |
| No TTS / LLM sync / Fabric | yes | yes | yes | **Keep** |
| 5 emotion teachers | yes | yes | yes | **Keep**, refine pacing |
| Two models | yes | yes | yes + curriculum | **Keep** |

---

## Q1 — Teacher video

| Source | Recommendation |
| --- | --- |
| Gemini | Always **2×8s per emotion** (~500 ms/vowel cycle); match blonde calibration framing |
| GPT | Keep 5-emotion plan; add **3 performances** per prompt for variation |
| Claude | Split **ANGRY/SAD** to 2×8s by default; HAPPY/SURPRISED/THINKING may stay 1×8s; pin **REST-under-emotion** (brows/eyes keep emotion, mouth at 0) |

### Merged lock

1. **5 emotion prompts** (HAPPY, SAD, SURPRISED, ANGRY, THINKING).  
2. **Pacing:** ANGRY + SAD = **2×8s each** by default; others = **1×8s**, split if hold shapes are unreadable.  
3. **REST-under-emotion:** mouth/jaw/lips/teeth → state 0; eyes/brows **keep** the video’s emotion. REST is **not** global NEUTRAL face — so Dataset A does not collapse.  
4. **NEUTRAL column:** measured from mouth-zero REST peaks **per emotion video** (emotion×rest), plus optional later true-neutral take if needed.  
5. **Variation (phase 2):** regenerate each prompt **3 times** after the first clean take (GPT). Phase 1 = one take per slot is OK to start.  
6. Framing/light must match existing blonde calibration take.

Update prompts in [`VowelTeacherPrompts.md`](VowelTeacherPrompts.md) accordingly.

---

## Q2 — PulseChunk / Fabric wire

| Source | Recommendation |
| --- | --- |
| Gemini | Lane A `c_t` every tick; Lane B KEY/Δ; KEY at t=0 + word/emotion shifts; spool >~5s; header fields |
| GPT | **Hierarchical** PulseChunk: metadata → WordSlices → tick packages → cell Δ |
| Claude | Reuse existing TickPackage KEY/Δ; metadata **once** at chunk start; KEY at attack / phase changes / when Δ loses |

### Merged lock

```text
PulseChunk
  metadata (once): utterance_id, emotion_track, tick_hz, …
  WordSlice[]     (each owns start/end tick, vowel script, timing)
    → TickPackage stream (KEY / Δ) on CHORUS Fabric lane B
  optional lane A: c_t each tick for compact live path
```

- **Reuse** TickFeed TickPackage machinery — no second delta scheme.  
- **KEY** at utterance start, WordSlice starts, hold↔release phase changes, or when Δ size loses.  
- **Δ** elsewhere (omit zero cells).  
- Long utterances: **TPK_REF** spool; short: inline.  
- Metadata not repeated every tick on lane A.

---

## Q3 — ML I/O

| Source | Recommendation |
| --- | --- |
| Gemini | GA-16 + diphthong transition vector; train **6-group** → frozen expand to cells |
| GPT | Model A → **latent** 64/128 → decoder → cells; B → velocity |
| Claude | Prefer **6-group** train target + hand/frozen expand (small teacher set); diphthongs as 2-point trajectory |

### Merged lock (phased)

**Phase 1 (now):**

- Train target = **6-group control vectors** (+ few dims each if needed)  
- **Frozen / authored** group→cell expansion (not learned from 5 videos alone)  
- Diphthongs = start tag + end tag + blend fraction over vowel duration  
- Input = LLM-emitted GA-16 script + emotion + tick progress / N  

**Phase 2 (signature / later — GPT):**

- Introduce **latent facial state** (e.g. 64-d) between group space and cells  
- Model A predicts latent; deterministic or lightly trained decoder → groups/cells  
- Model B predicts **velocity / residual** in latent or group space  

Phase 1 does **not** block PulseChunk; Phase 2 is the compression / style path.

---

## Q4 — Model packaging & curriculum

| Source | Recommendation |
| --- | --- |
| Gemini | Phase1 A on holds; Phase2 freeze A, train B on trajectories |
| GPT | Latent + velocity (see Q3) |
| Claude | Train A first; B gets `(S0, S1, N)` **and emotion** (attack speed by mood) |

### Merged lock

1. **Two separate models** (A then B).  
2. Train **A on hold peaks** first.  
3. Freeze A; train **B** on `(S0, S1, N_ticks, emotion)` → path / velocity-style ticks.  
4. B loss: endpoint match to A’s S1 + smoothness (velocity/accel penalty).

---

## Q5 — Statement boundary

| Source | Recommendation |
| --- | --- |
| Gemini | utterance_id; micro-pauses as AX/REST inside chunk; emotion holds |
| GPT | Emotion as **track** (future multi-emotion), even if constant today |
| Claude | Explicit `utterance_id`; in-band pause markers; keep one PulseChunk |

### Merged lock

- Boundary = host **`utterance_id`** span (not a naive sentence splitter).  
- Micro-pauses = in-chunk AX/REST (or pause spans); **emotion_track stays up**.  
- Metadata field = **`emotion_track[]`** (today often one constant value; format allows mid-utterance shifts later).

---

## Q6 — Attack / hold / release & coarticulation

| Source | Recommendation |
| --- | --- |
| Gemini | Fixed 6/hold/6; 4-tick cosine blend between vowels |
| GPT | (less specific; velocity framing) |
| Claude | Default 6; **ANGRY ~4**, **SAD ~8–10**; large lip conflicts force brief neutral pass |

### Merged lock

| Phase | Default | Emotion override |
| --- | --- | --- |
| Attack | 6 ticks (~100 ms) | ANGRY ≈ 4; SAD ≈ 8–10 |
| Hold | `N_word - attack - release` | emotion held |
| Release | 6 ticks | may shorten if next attack overlaps |

**Coarticulation:** 4-tick cosine blend when consecutive vowel targets are compatible; if conflict is large (e.g. tight round → wide spread), insert a short **mouth-neutral** bridge (1–3 ticks) instead of a mushy average.

---

## Q7 — Word path

| Source | Recommendation |
| --- | --- |
| Gemini | G2P dict + 3-tick crossfade; hot list of 8 words |
| GPT | WordSlice owns timing; hierarchical chunk |
| Claude | **LLM emits GA-16** (avoid English spelling traps); domain hot-list 20–30 |

### Merged lock

1. **Primary:** LLM/API emits **GA-16 script** (+ emotion + timings) — not grapheme guessing.  
2. Fallback G2P only if API omits phonemes.  
3. **WordSlice owns** `start_tick`, `end_tick`, vowel script, local coarticulation flags.  
4. PulseChunk concatenates WordSlices.  
5. Boundary stitch: **3-tick** crossfade on group/cell Δ.  
6. Optional W1 hot list (beta): conversational anchors  
   `hi, hello, yes, no, thanks, okay, sorry, bye` (+ expand to ~20–30 domain words).

---

## Extra (outside the 7)

| Source | Point | Merge |
| --- | --- | --- |
| GPT | Latent facial space as signature | **Phase 2** architecture goal |
| GPT | Confidence scores | Add optional `confidence` on A/B outputs; low → smooth / W1 / prior |
| GPT | General facial behavior (not only speech) | Design note: intentional facial states — keep door open |
| Claude | UTF-8 corruption in help paste | Re-save help packet as UTF-8 |
| Claude | Changelog still said 3×8s | Clean changelog / §6.1 |

---

## Pose matrix (§5.2)

Gemini supplied a full 16×6 numeric table. **Adopt as design reference** in
`VowelDesign.md` §5.2 (replace blanks). Values are prior for Veo direction and
Model A supervision — measured teacher data still wins when it disagrees.

---

## NWR scoping recommendation (added)

**Adopted into [`VowelDesign.md`](VowelDesign.md) §3.2.** Summary:

- NWR = structured **cell ROI** substrate (not per-frame generative video @ 60 Hz).  
- Advantages: spatial precision, micro-latency via Fabric vectors, anatomical independence of the 6 groups (anti jaw-pump).  
- Rules: rigid geometry + vectorized motion; Side B teacher mapped into `.bds` digest; LLM script → vectors → Fabric → same cells → GPU.

---

## What to do next (design, not code)

1. Update `VowelDesign.md` locks from this merge (done).  
2. Retarget `VowelTeacherPrompts.md` for 2×8s on ANGRY/SAD + REST-under-emotion (done).  
3. Refresh `VowelDesignHelp.md` Part A from this merge + UTF-8 clean (partial — prefer ReviewMerge).  
4. Amin: confirm Phase 1 (6-group) vs jump early to latent (GPT Phase 2).  
5. NWR scoping locked in §3.2 — keep identity photo rigid; PulseChunk = motion only.
