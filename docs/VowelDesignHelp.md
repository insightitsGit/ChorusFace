# VowelDesign — help packet (7 questions + full design)

**Purpose:** Give this file to another person or agent when you want help thinking
through or reviewing the design. **Design only — do not implement** until Amin asks.

**Sibling docs:**
- Working design: [`VowelDesign.md`](VowelDesign.md)
- Veo teacher prompts: [`VowelTeacherPrompts.md`](VowelTeacherPrompts.md)
- **Review merge (Gemini · GPT · Claude):** [`VowelDesignReviewMerge.md`](VowelDesignReviewMerge.md) ← start here after the three reviews
- Related (do not replace): TickFeedDesign, PhoneticFidelity, VoiceSync, SideB_VideoCellCollection

> Part A below may lag the merge. Prefer **VowelDesignReviewMerge.md** + current
> **VowelDesign.md §6.1–6.2** for the locked answers to the 7 questions.

**One-line story:**  
Shoot 5 teacher videos → train Model A (pose) + Model B (motion) → LLM utterance →
PulseChunk vectors @ 16.7 ms → CHORUS Fabric → NWR/GPU cells. **No TTS ownership.**

---

# Part A — The 7 questions (explain + locked default)

These were the gaps after the architecture locked. Each section: **what it means**,
**why it matters**, **locked default**, **what you can still help decide**.

---

## Q1 — Teacher video (how we shoot truth)

**What it means:**  
How do we get *measured* face motion for *this* avatar so Dataset A (targets) and
Dataset B (transfers) are real, not invented?

**Why it matters:**  
ML and PulseChunks must match the same NWR cells / GPU recipe as the product face.
Video of this face is the ground truth.

**Locked default:**  
**5 Veo prompts → 5 videos.** One emotion per video (HAPPY, SAD, SURPRISED, ANGRY,
THINKING). Each walks all **16** GA vowels with **REST = state 0** between vowels.
NEUTRAL is not a sixth shoot — it comes from REST holds. Target **8 s** per video;
if too dense, split *that emotion only* into 2×8s.  
Paste prompts: [`VowelTeacherPrompts.md`](VowelTeacherPrompts.md).

**Help welcome on:**  
Pacing (is 16 vowels in 8s readable?), whether any emotion should always be 2×8s,
subject/framing tweaks to match the existing blonde calibration take.

---

## Q2 — PulseChunk on the wire (how data is sent)

**What it means:**  
What binary shape carries one statement’s ordered 16.7 ms samples from the API
side to the face?

**Why it matters:**  
Realtime face drive should be **vectors**, not JSON mouth cues. Must fit
**CHORUS Fabric** (existing TickFeed transport).

**Locked default:**  
Logical **PulseChunk** = ordered **TickPackages** (KEY then Δ) on Fabric **lane B**;
optional compact `c_t` on **lane A** each tick. Not JSON-primary.

**Help welcome on:**  
Exact header fields, when to KEY vs Δ, inline vs TPK_REF spool thresholds,
how PulseChunk metadata (statement id, emotion) rides with the packages.

---

## Q3 — ML inputs / outputs

**What it means:**  
What goes into Model A/B, and what vectors come out per tick?

**Why it matters:**  
We do **not** require TTS/audio. The LLM/API supplies meaning and timing;
models supply face motion.

**Locked default:**  
**In:** GA-16 (or word→vowel) script + emotion + tick count/durations from
**LLM/API**.  
**Out:** per-tick **vectors** (group drives → known cells, and/or Fabric-compatible
`c_t`). No audio required.

**Help welcome on:**  
Prefer dense cell Δ vs 6-group controls vs `float32[64] c_t` as the primary
training target; feature encoding for diphthongs (EY, AY, …).

---

## Q4 — One model or two

**What it means:**  
Dataset A (held “state 1” pose) vs Dataset B (tick path 0 → hold → release) —
same network or separate?

**Why it matters:**  
Pose match and trajectory match want different losses; one net often muddies both.

**Locked default:**  
**Two separate models** — Model A = state/target, Model B = transfer. Shared
trunk later only if needed.

**Help welcome on:**  
Training curriculum (train A first, then B conditioned on A), and whether B
should take explicit (state0, state1, N_ticks) as inputs.

---

## Q5 — Statement boundary

**What it means:**  
What starts/ends one PulseChunk?

**Why it matters:**  
Emotion must **hold for the whole set** (e.g. anger until the line ends). Wrong
boundary → emotion drops mid-thought.

**Locked default:**  
One **LLM/API utterance** (one assistant speak span / one “say this” job) =
one PulseChunk.

**Help welcome on:**  
API shape: explicit `utterance_id` vs whole chat message vs sentence splitter;
how pauses inside one utterance are marked.

---

## Q6 — Attack / hold / release timing

**What it means:**  
How many ~16.7 ms ticks to enter a shape, stay, and leave?

**Why it matters:**  
No teleport from state 0; Dataset B must include sustain (keep emotion) not only
the ramp-in.

**Locked default:**  
Attack ≈ **6 ticks (~100 ms)**; **hold** = rest of word/utterance with emotion
kept; release ≈ **6 ticks**. KEY at big changes; Δ otherwise.

**Help welcome on:**  
Per-emotion attack (angry slower?), per-vowel hold floors, coarticulation overlap
when the next vowel starts before release finishes.

---

## Q7 — How each word gets its ticks

**What it means:**  
For a word inside the utterance, do we look up a stored clip or build from vowels + ML?

**Why it matters:**  
Teacher video cannot cover the English lexicon; runtime must generalize.

**Locked default:**  
**W2-first:** word → GA-16 vowels → Model A/B → WordSlice → stitch into
PulseChunk. Optional W1 lexicon later for hot words (“hi”, “thanks”).

**Help welcome on:**  
Grapheme→GA-16 mapping quality, minimum hot-word list for beta, boundary blend
between WordSlices.

---

### Quick reference — locked defaults

| # | Topic | Default |
| --- | --- | --- |
| 1 | Teacher | 5 Veo videos (emotion × 16 vowels) |
| 2 | Wire | PulseChunk → TickPackages on CHORUS Fabric |
| 3 | ML I/O | LLM script+emotion+duration → tick vectors |
| 4 | Models | Two separate (A state, B transfer) |
| 5 | Boundary | One LLM/API utterance |
| 6 | Phases | ~6 tick attack / hold / ~6 tick release |
| 7 | Words | W2-first (vowels → models → stitch) |

---

# Part B — Full VowelDesign (copy for context)

Everything below is the full contents of `VowelDesign.md` so helpers have one file.

---
# VowelDesign

**Status:** design in progress â€” **do not implement** until explicitly asked.  
**Created:** 2026-08-05  
**Owns:** General American vowel inventory (16), six face articulators, per-tick (60 Hz) targets.  
**Does not replace:** older design docs below â€” they stay the source of truth until this doc is accepted and linked from [`docs/README.md`](README.md).

### Locked premises (Amin, 2026-08-05)

| Premise | Value |
| --- | --- |
| Speech inventory | **General American â‰ˆ 16 vowel phonemes** (incl. common diphthongs) |
| Face articulators | **6:** eyes Â· eyebrows Â· mouth Â· lips Â· teeth Â· jaws |
| Emotions | **6:** NEUTRAL Â· HAPPY Â· SAD Â· SURPRISED Â· ANGRY Â· THINKING |
| Time base | TickFeed **60 Hz** (â‰ˆ **16.67 ms**); cell/LOOK values may change every tick |
| State-0 | At utterance / segment start, face articulator state is **0** (rest) |
| Two datasets | **(A) targets** 16Ã—6Ã—6 = **576** end-states Â· **(B) transfer** tick path for whole set |
| Wire unit | **PulseChunk** â€” one statementâ€™s ordered 16.7 ms samples for all cells/articulators |
| Product focus | **No TTS ownership** â€” sync with **LLM/API**; payload is **vectors**; transport **CHORUS Fabric** |

## Relationship to existing docs (keep)

| Doc | Still owns |
| --- | --- |
| [`TickFeedDesign.md`](TickFeedDesign.md) | TickFeed Side A/B, LOOK/FIELD, post-initial mouth band (Â§14) |
| [`PhoneticFidelity.md`](PhoneticFidelity.md) | Full viseme set, host timeline vs PCM aligner, lip-reading goals |
| [`VoiceSync.md`](VoiceSync.md) | `/voice/expect` â†’ `/pcm` â†’ `/end`, energy placement |
| [`DisplayLayers.md`](DisplayLayers.md) | L00â€“L11 plate / field order |
| [`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md) | 8s Side B teacher take (includes OPEN / hi / TH) |
| [`ProductBeta.md`](ProductBeta.md) Â· [`FaceServiceEmbed.md`](FaceServiceEmbed.md) | Host TTS default, FaceBridge surface |

This file is the **vowel-only** redesign workspace. Consonants (PP, FF, TH, â€¦), full TickFeed packaging, and Docker/embed are out of scope here unless a vowel decision forces a cross-link.

---

## 1. Problem (why this doc)

Observed failure mode in dogfood: mouth reads as **simple open/close**, not as distinct spoken vowels. Vowels are where lip readers and casual viewers judge â€œis this talking?â€ â€” if AH / EE / OH / OU collapse to one jaw pump, the face looks broken even when closures are correct.

**Design goal:** each canonical vowel must produce a **visibly different lip silhouette** (width Ã— openness Ã— roundness), driven by host timing, without inventing face RGB.

---

## 2. Goals

1. Stable **vowel inventory** (names + aliases) agreed for host timelines and grapheme fallback.
2. Per-vowel **pose targets** (jaw / open / width / round) that survive TickFeed plates + muscle field.
3. Clear **authority** when host PCM, host timeline, and text-only cue disagree.
4. Calibration / plate requirements for vowels called out separately from consonant kit.
5. Acceptance tests that a human (or HUD) can score: â€œAH â‰  EE â‰  OUâ€ at a glance.

## 3. Non-goals

- **TTS** â€” ChorusFace does **not** own speech audio for this design. No requirement
  to synthesize or play voice. (Host may speak separately; we only need sync cues
  from the **LLM/API** â€” text, emotion, timing â€” not a face TTS pipeline.)
- Acoustic vowel ASR inside ChorusFace.
- Rewriting full Oculus consonant set (see PhoneticFidelity).
- Generative mouth interior / painted teeth.
- Insightits landing bridge heuristics (website dogfood owns that integration).
- JSON-heavy REST as the primary realtime face drive (vectors + Fabric instead).

### 3.1 Product focus (Amin â€” locked)

```text
LLM / host API  â†’  (text + emotion + timing)  â†’  PulseChunk as vectors
                â†’  CHORUS Fabric  â†’  NWR / GPU cells
```

| We care about | We do **not** center |
| --- | --- |
| Sync with **LLM + API** (what was said, emotion, when) | Owning or implementing **TTS** |
| **Vectors** on cells / groups / compact codes | Mouth-cue-only `/prism/speak` as the long-term path |
| **CHORUS Fabric** transport (see TickFeedDesign Â§6.2) | Browser Web Speech as the face clock |
| Teacherâ†’Model A/B â†’ PulseChunk | Local `--tts` lab path |

PulseChunk samples are **vector data** (groupâ†’cell drives, or compact \(c_t\) +
expand). Fabric carries those vectors (TickFeed: fixed-dim float lanes + KEY/Î”
packages). Exact binding: PulseChunk â†” Fabric lanes TBD, same family as
existing CHORUS â†” TickPackage.

---

## 4. Current baseline (as of product code)

Canonical vowels today (`chorusface.speech`):

| Name | Family (informal) | Role |
| --- | --- | --- |
| `AH` | open central | max jaw contrast |
| `AA` | open back/central | Oculus `aa` family |
| `EH` | mid front | â€œbedâ€ |
| `IH` | near-close front | â€œbitâ€ |
| `EE` | close front / wide | smile-wide outline |
| `OH` | mid back round | â€œtoeâ€ |
| `OU` | close back round | â€œbootâ€ / book |

Grapheme map (fallback only): Aâ†’AH, E/I/Yâ†’EE, Oâ†’OH, Uâ†’OU (+ digraphs OO/OU/OWâ€¦).

**Known gap:** dogfood paths that only pump â€œopen vs CLOSEDâ€ discard this inventory. Energy without a vowel script also collapses motion to amplitude.

---

## 5. Proposed vowel contract (draft â€” fill with Amin)

### 5.1 Inventory â€” General American 16 (locked count)

**Count locked: 16.** Names below are the working phoneme set for design
(Wells-style keywords). Host timelines may still send ChorusFace/Oculus
aliases; mapping 16 â†’ visual drives is Â§5.2â€“5.3.

| # | Keyword | IPA (approx.) | Working tag | Notes |
| --- | --- | --- | --- | --- |
| 1 | fleece | i | `EE` | close front |
| 2 | kit | Éª | `IH` | |
| 3 | face | eÉª | `EY` | diphthong |
| 4 | dress | É› | `EH` | |
| 5 | trap | Ã¦ | `AE` | |
| 6 | lot / palm | É‘ | `AA` | |
| 7 | thought | É” | `AO` | may merge with lot in some speakers |
| 8 | goat | oÊŠ | `OH` | diphthong |
| 9 | foot | ÊŠ | `UH` | |
| 10 | goose | u | `OU` | |
| 11 | strut | ÊŒ | `AH` | open-ish central |
| 12 | comma / about | É™ | `AX` | schwa â€” often visually weak |
| 13 | nurse | É / Éš | `ER` | rhotic |
| 14 | price | aÉª | `AY` | diphthong |
| 15 | mouth | aÊŠ | `AW` | diphthong |
| 16 | choice | É”Éª | `OY` | diphthong |

**Design rule:** all 16 are first-class in the **speech script**. They do **not**
all need 16 unique photographs â€” several may share a visual class (e.g. schwa
near `AH`/`EH`) as long as the script still names the phoneme.

### 5.1b Six face articulators (locked)

For every tick (and every vowel), design may drive these **six** parts:

| # | Part | What changes (design intent) | Typical TickFeed / LOOK hook |
| --- | --- | --- | --- |
| 1 | **Eyes** | gaze / aperture (blink lids) | lid / eye LOOK |
| 2 | **Eyebrows** | raise / knit | brow label |
| 3 | **Mouth** | cavity / interior show (gap) | open plate / mouth_gap |
| 4 | **Lips** | width, round, seal / spread | lip outline / oris |
| 5 | **Teeth** | visibility (upper/lower show) | open/smile plates â€” photographed |
| 6 | **Jaws** | drop / close | jaw drive |

Yes â€” **6**. Note: **mouth** (cavity/gap) and **lips** (rim shape) are separate
on purpose so â€œopen jaw + rounded lipsâ€ (OH) â‰  â€œopen jaw + wide lipsâ€ (AE/AH).

Vowels mainly stress **lips Â· teeth Â· jaws Â· mouth**; eyes/eyebrows stay mostly
idle unless the beat is expressive (surprise, smile eyes). Consonants will reuse
the same 6 later â€” not redefined here.

### 5.2 Pose targets (design numbers â€” not code yet)

Per vowel Ã— six articulators. Use `0` = rest/off, `1` = full for that part;
leave blank until we fill together. Sparse is OK (eyes/brows often `0` for pure speech).

| Tag | eyes | brows | mouth | lips | teeth | jaws | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EE | | | | | | | |
| IH | | | | | | | |
| EY | | | | | | | diphthong |
| EH | | | | | | | |
| AE | | | | | | | |
| AA | | | | | | | |
| AO | | | | | | | |
| OH | | | | | | | diphthong |
| UH | | | | | | | |
| OU | | | | | | | |
| AH | | | | | | | |
| AX | | | | | | | schwa |
| ER | | | | | | | |
| AY | | | | | | | diphthong |
| AW | | | | | | | diphthong |
| OY | | | | | | | diphthong |

### 5.3 Timing rules (vowels only)

- Host timeline spans win for start/end.
- With PCM + expect: energy **places** the next scripted vowel; it does not invent a new vowel name.
- Closures (PP/MM/CLOSED) always interrupt vowel holds (already in TickFeed Â§14 â€” reaffirm here).
- Max vowel hold / coarticulation into next vowel: **TBD**.

### 5.4 Host API surface (vowels)

Preferred:

```http
POST /voice/timeline
{ "caption": "â€¦", "spans": [
  {"phoneme": "AH", "start": 0.00, "end": 0.12},
  {"phoneme": "EE", "start": 0.12, "end": 0.20}
], "replace": true }
```

Alternate: `/voice/expect` + `/voice/pcm` with caption that expands to the vowel script via `polygon_speech` (or successor).

Mouth-cue-only `/prism/speak` remains **non-authoritative** for vowel QA.

---

## 5.5 Two datasets: targets + transfer (Amin â€” locked intent)

Yes: we need **more than** the 576 end poses.

### Dataset A â€” Target states (what we aim for)

```text
16 vowels Ã— 6 articulators Ã— 6 emotions  =  576  target values
```

Each cell is the **arrived** face drive for that (vowel, emotion) on that articulator  
when the shape is â€œthereâ€ (hold / peak), not how we got there.

### Dataset B â€” Transfer data for the **whole segment** @ 16.7 ms

At **time 0** the face articulator state is **0** (rest / zero).  
Speech does not teleport to a target; every **â‰ˆ 16.67 ms** we emit the next
transfer sample. Dataset B is **not only the ramp-in** â€” it covers the
**entire set / segment** until we leave that delivery:

```text
phase ATTACK   0 â†’ target     (enter vowelÃ—emotion, e.g. start ANGRY)
phase HOLD     stay at / near target for the rest of the segment
               (e.g. keep ANGRY through the whole angry line â€” do not drop)
phase RELEASE  target â†’ 0 or â†’ next segment   (only when the set ends)
```

Example â€” one angry utterance lasting ~1.0 s (~60 ticks):

```text
t = 0          state = 0
t = 1â€¦K Â·Î”t    attack â†’ ANGRYÃ—vowel target     Î”t = 16.67 ms
t = K+1â€¦M Â·Î”t  HOLD â€” keep anger (and active vowel shaping) for the whole set
t = M+1â€¦End    release / handoff when the segment actually ends
```

So if the emotion is **ANGRY**, Dataset B must **hold that anger to the end of
the set**, not fade after the first vowel peak. Vowel changes inside the set
still retarget lips/jaws/mouth tick-by-tick; emotion sustain stays up until
the segment boundary.

```text
â€œmotionâ€   = how values change across ticks
targets A  = where a vowelÃ—emotion wants to sit (576)
transfer B = full tick series for the segment: attack + hold + release
```

Same 60 Hz TickFeed clock as [`TickFeedDesign.md`](TickFeedDesign.md).

### Design consequences (still open numerically)

| Topic | Intent | Open |
| --- | --- | --- |
| Segment = â€œsetâ€ | One emotion span (e.g. whole ANGRY line) or one vowel only? | Prefer **emotion span** â€” hold emotion for whole set |
| Attack length | Ticks 0 â†’ first target | TBD (~50â€“200 ms) |
| Hold | Repeat / sustain samples while set continues | Required â€” not optional |
| Release | Only at set end â†’ 0 or next set | TBD |
| Stored form | Absolute state each tick vs delta from previous | Align with KEY/Î” if possible |
| Vowel changes inside hold | New vowel targets while emotion stays | Yes â€” coarticulation inside the set |

**Not the same as** â€œ576 video clips.â€ A is the pose table; B is the **full
60 Hz time series** for each played set (attack + hold through the end + release).

### PulseChunk â€” one statement on the wire (Amin)

Join A + B into a single sendable unit for **one term / statement**.

**Example:** AI should say *â€œHi, how are you?â€* with **HAPPY**.

One **PulseChunk** carries **everything** needed to play that statement on the
face: ordered samples every **â‰ˆ 16.67 ms** from state **0**, through the whole
line (attack + hold HAPPY + vowel changes per word), to release â€” so Side A can
drive **cells** (and the six articulators) without asking for more mid-utterance.

```text
PulseChunk {
  statement:  "Hi, how are you?"
  emotion:    HAPPY          // held for the whole set
  tick_hz:    60
  t0_state:   0              // rest at start
  samples: [                 // strict time order
    { t: 0,    â€¦ cell / articulator drives â€¦ },
    { t: 1,    â€¦ },          // +16.7 ms
    { t: 2,    â€¦ },
    â€¦
    { t: M,    â€¦ }           // end of statement / release
  ]
}
```

| Property | Rule |
| --- | --- |
| Scope | **One statement** (one term), not the whole conversation |
| Emotion | One primary emotion for the chunk (e.g. HAPPY) â€” held per Dataset B |
| Ordering | Samples **must** be in tick order; no gaps in the play clock |
| Cadence | One sample per tick (60 Hz) unless we later allow sparse Î” with explicit tick index |
| Groups | **Every sample includes all 6 articulator groups** â€” not mouth-only |
| Payload | Enough to update face **cells** / LOOK+FIELD for that tick (exact bytes TBD) |
| Join | Built from Dataset A targets + Dataset B transfer for this statement |

**Per tick, a PulseChunk sample covers all six groups together:**

| Group | In every 16.7 ms sample? |
| --- | --- |
| Eyes | Yes |
| Eyebrows | Yes |
| Mouth | Yes |
| Lips | Yes |
| Teeth | Yes |
| Jaws | Yes |

Idle groups may be **0** for that tick, but they are still present in the sample
so the face never has to invent a missing channel mid-statement.

### Groups = NWR objects â†’ cells inside objects (Amin)

Yes. **Eye** and **mouth** (and the other four) are **groups**, and in NWR terms a
group is an **object**: a connected **cluster of cells**, not a separate mesh
(see [`AvatarCellDataflow.md`](AvatarCellDataflow.md), [`NWRDataDesign.md`](NWRDataDesign.md)).

Hierarchy inside one PulseChunk sample:

```text
PulseChunk sample @ tick t
  â””â”€ group / object  (e.g. mouth, eyes, â€¦)     â† one of the 6
       â””â”€ all known cells for that object       â† from NWR region / catalog
            â””â”€ channel values (phase-1: vx, vy; later as needed)
```

| Rule | Meaning |
| --- | --- |
| Object | Named group = cell cluster addressable in the world (`.bds` + `region_catalog`) |
| Membership | PulseChunk carries data for **all known cells** of that group, not a single centroid only |
| Source of â€œknown cellsâ€ | NWR digest / region catalog for this avatar world (TickFeed face ROI cells in that object) |
| Nesting | Objects can contain sub-structure later (e.g. mouth â†’ lips/teeth/cavity) without breaking the 6 top-level groups |
| Empty motion | A cell may have Î”=0 this tick; it stays listed so membership is stable |

So: **PulseChunk â†’ groups (objects) â†’ every known cell in that object**, every
16.7 ms, in order for the statement.

Relation to existing TickFeed:

```text
PulseChunk   =  statement-level bundle (host â†’ face)
TickPackage  =  one tickâ€™s KEY/Î” bytes (already in TickFeedDesign)
```

A PulseChunk is the **outer envelope**: many TickPackages (or equivalent sample
records) **in order** for one spoken line. Exact binary layout TBD â€” design
name and meaning are locked here first.

```text
Host AI:  "Hi, how are you?" + HAPPY
    â†’ build PulseChunk (all 16.7 ms steps)
    â†’ Face plays samples in order from state 0
```

---

## 5.6 How do we provide data for each word? (design question)

We know the datasets and PulseChunk shape. The next design problem:

> For each **word** inside a statement, where do the 16.7 ms group/cell samples
> come from before they are joined into one PulseChunk?

Example statement: *â€œHi, how are you?â€* + HAPPY â†’ one PulseChunk, but built from
word (or vowel) pieces in order:

```text
[hi] + [how] + [are] + [you]   (+ spaces / coarticulation)
        â†“ join on 60 Hz timeline
     PulseChunk(statement, HAPPY)
```

### Layered answer (aligns with product rule: known â†’ table, unknown â†’ ML)

| Layer | When | What we store / run | Provides |
| --- | --- | --- | --- |
| **W1 â€” Word table** | Word is **known** | Precomputed transfer series for that word (all 6 groups Ã— cells, or compact code) at 60 Hz, often at NEUTRAL; emotion applied as hold overlay | Best fidelity for frequent words (â€œhiâ€, â€œyouâ€, â€¦) |
| **W2 â€” Vowel compose** | Word unknown or no row | Grapheme/phoneme â†’ GA-16 tags â†’ Dataset **A** targets + Dataset **B** attack/hold/release curves | Default path for any English word |
| **W3 â€” Measured take** | Side B teacher / calibration | Video â†’ tick samples for beats/words (existing TickFeed collect) | Seeds W1/W2 tables; not required every live chat |
| **W4 â€” Host timing** | Live product | Host gives text + emotion (+ optional PCM or phoneme times) | **When** each word/vowel sits on the clock; face fills **what** from W1/W2 |
| **W5 â€” ML cover** | Gaps / unseen transitions | Fill missing transfer ticks (same spirit as behavior ML) | Only where W1â€“W2 have no sample |

```text
Host:  text + emotion (+ optional audio clock)
  â†’ tokenize words
  â†’ for each word:  W1 lookup  or  W2 compose from vowels
  â†’ stitch on 60 Hz (coarticulation at boundaries)
  â†’ apply emotion HOLD for whole statement (Dataset B)
  â†’ emit one PulseChunk
```

### Per-word payload (design intent)

For word \(w\) with duration \(D\) seconds â‰ˆ \(N = \mathrm{round}(D \times 60)\) ticks:

```text
WordSlice {
  word: "how"
  vowels: [AW] or [AH, OU] â€¦     // from GA-16 script
  ticks: N
  samples[0..N): each sample = 6 groups â†’ all known cells
}
```

WordSlices concatenate into the statement PulseChunk (with short boundary
blends â€” TBD).

### Word-path decisions (locked in Â§6.2)

- [x] **Primary live path:** W2-first (vowel compose via Model A/B)  
- [x] **Duration:** LLM/API timing or text estimate â€” not TTS/PCM  
- [x] **Emotion:** held on every tick for the utterance  
- [x] **Between words:** micro-REST / short coarticulation  
- [ ] Minimum W1 lexicon size for beta â€” optional later

### 5.7 Lock to avatar + GPU via video teacher (Amin â€” intent)

**Yes â€” that makes sense, and it is possible.**

To make Dataset A/B and PulseChunks **realistic for this avatarâ€™s GPU path**
(same identity photo, same cell objects, same LOOK/FIELD recipe), we need a
**video avatar sample** (teacher take of *this* face), then **detect / measure**
from that video:

| Extract from video | Becomes |
| --- | --- |
| Held shapes for vowel Ã— emotion (articulator drives on known cells) | **Dataset A** targets (toward the 576 space) |
| Frameâ†’tick motion from rest through hold to release @ ~60 Hz | **Dataset B** transfer series |
| Region membership (mouth, eyes, â€¦) on the NWR grid | Group â†’ cell lists for PulseChunk samples |

```text
Video avatar (this face)
  â†’ detect landmarks / optical flow / lids (Side B collect family)
  â†’ resample to 60 Hz ticks on known cells per group
  â†’ Dataset A peaks + Dataset B full-set transfers
  â†’ bake into world (tables / tracks) used by GPU display recipe
  â†’ live: compose PulseChunk from those locked measurements
```

This is the same **Side B** idea as [`TickFeedDesign.md`](TickFeedDesign.md) /
[`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) /
[`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md): video teaches
**this** avatar; runtime does not invent a different face.

#### Possible? Yes â€” with a practical shoot plan

| Claim | Reality |
| --- | --- |
| Need a video of the avatar face | **Yes** â€” required for realistic lock |
| Need all 576 as *separate* clips | **No** â€” one (or few) structured takes can cover the matrix |
| Detect 576 target points | **Yes** â€” as **measurements** at hold frames (vowelÃ—emotion), not 576 files |
| Dataset B from video | **Yes** â€” resample video timeline â†’ 16.7 ms ticks (attack/hold/release) |
| Lock to GPU schematic | **Yes** â€” same `.bds` cells, plates, region catalog, display recipe |

**Shoot design (not 576 videos):** e.g. per emotion block, walk GA-16 vowels
(rest â†’ vowel â†’ hold â†’ rest) on camera; six emotion blocks (or denser single
script). Detector fills A at holds and B along the ticks. Sparse cells /
unseen combos â†’ W2 curves or W5 ML later â€” still design-valid.

**Possible risks (design, not blockers):** thought/lot merger in speech vs face;
schwa visually weak; some emotions barely move mouth; need clear REST=0 between
blocks so transfers start from state 0.

### 5.8 Teacher video â†’ ML model â†’ any word (Amin â€” intent)

**Yes â€” that makes sense.**

The teacher video **cannot** speak every English word. It **can** feed **true**
measured motion on **this** avatar (cells â†” GPU vectors). That truth trains an
**ML model**. At runtime, a **new word** is run through the model, which outputs
the PulseChunk materials (toward Dataset A space + Dataset B transfers).

```text
Teacher video (limited script, this face)
  â†’ Side B measure: true ticks on NWR cells / groups / GPU vectors
  â†’ train ML  (input: word or GA-16(+emotion) script Â· output: tick series)
  â†’ Runtime: new word â†’ model â†’ samples @ 16.7 ms â†’ PulseChunk
```

| Role | What |
| --- | --- |
| Teacher video | **Ground truth** only â€” how *this* avatar actually moves; locks identity + cell topology |
| Dataset A/B from video | Supervision targets (holds + full-set transfers), not the full English lexicon |
| ML model | Learns mapping from speech units â†’ avatar/GPU tick drives |
| New word at runtime | Model inference (W5 / extended W2) â€” not â€œfind that word in the videoâ€ |
| Tables W1 | Optional cache of frequent words **after** model or from measured beats |

**Train (offline):**

```text
video ticks (true) + labels (vowel / emotion / beat / word if known)
  â†’ fit model so outputs match measured group/cell drives @ 60 Hz
```

**Infer (live):**

```text
"chrysanthemum" + HAPPY
  â†’ phoneme/vowel script (GA-16)
  â†’ model(script, emotion, duration)
  â†’ ordered samples (6 groups Ã— known cells)
  â†’ PulseChunk
```

So: **video teaches truth connected to avatar+GPU; ML generalizes to all words.**
Same spirit as existing â€œknown â†’ table, unknown â†’ ML coverâ€ â€” here the teacher
is the source of truth for training, not a dictionary of every word.

Open (design): model input features (phoneme one-hots vs audio embedding vs text);
output (dense cells vs compact code \(c_t\) vs 6-group controls expanded to cells).

### 5.9 Two models (or two heads): state vs transfer (Amin â€” intent)

**Yes â€” splitting is right.** Dataset A and Dataset B are different jobs; one
monolithic net often muddies both.

| Model | Learns | Answers | Output |
| --- | --- | --- | --- |
| **Model A â€” State (target)** | Dataset A | â€œWhat is **state 1** / the held pose for this vowelÃ—emotion on the 6 groupsÃ—cells?â€ | Peak / hold target (toward 576 space) |
| **Model B â€” Transfer** | Dataset B | â€œHow do we **walk** tick-by-tick from state **0** â†’ state 1, hold for the set, release?â€ | Ordered samples @ 16.7 ms (attack + hold + release) |

```text
New word + emotion
  â†’ Model A  â†’  target state (state 1)
  â†’ Model B  â†’  transfer series given (state 0, state 1, duration / set length)
  â†’ stitch â†’ PulseChunk
```

**Why two:**

- A is mostly **static / peak** (where the face sits).
- B is **dynamics** (how values change every tick) â€” including keeping anger for
  the whole set.
- Teacher video supervises **both**, but loss/metrics differ (pose match vs
  trajectory match).

**Packaging options (either is fine in design):**

1. **Two models** (clear ownership â€” preferred for VowelDesign clarity)  
2. **One trunk, two heads** (shared encoder, A-head + B-head) â€” same split, shared features  

Locked intent: **separate A vs B predictors**, not one net that only emits a
single pose and hopes an ad-hoc ease replaces Dataset B.

---

## 6. Calibration / plates (vowels)

Teacher take today ([`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md)): REST â†’ SMILE â†’ OPEN(â€œahâ€) â†’ hi â†’ think â†’ â€¦  
That **8s dense kit stays** for the old path. VowelDesign adds a **separate teacher shoot** aimed at GA-16 Ã— emotions Ã— transfers.

### 6.1 Teacher shoot plan (Amin â€” locked)

**Yes â€” 5 prompts â†’ 5 videos is the straightforward collect plan.**

One **emotion per video**, each walks all **16 vowels**, with **REST = state 0**
between vowels (so Dataset B always sees 0 â†’ target â†’ 0).  
**NEUTRAL** is not a sixth shoot: it is the REST holds inside every video
(and Model Aâ€™s NEUTRAL targets come from those REST peaks).

| # | Prompt / video | Emotion held for whole clip | Vowels |
| --- | --- | --- | --- |
| 1 | `VowelTeacher_HAPPY` | HAPPY | GA-16 walk |
| 2 | `VowelTeacher_SAD` | SAD | GA-16 walk |
| 3 | `VowelTeacher_SURPRISED` | SURPRISED | GA-16 walk |
| 4 | `VowelTeacher_ANGRY` | ANGRY | GA-16 walk |
| 5 | `VowelTeacher_THINKING` | THINKING | GA-16 walk |

```text
Per video (same face, framing, light):
  REST â†’ V1 â†’ REST â†’ V2 â†’ â€¦ â†’ REST â†’ V16 â†’ REST
  emotion face held the whole time (except pure REST frames = state 0 mouth)
```

| Rule | Choice |
| --- | --- |
| Generator | Veo (same style as [`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md)) |
| Count | **5 prompts â†’ 5 videos** (simple collect) |
| Length | Target **8.0 s** each; if 16 vowels too dense, **that emotion only** splits to 2Ã—8s (still same prompt family) |
| Identity | Same avatar as product world |
| Consonants / old kit | Stay on existing 8s calibration prompt â€” out of these 5 |

Prompt paste text: [`VowelTeacherPrompts.md`](VowelTeacherPrompts.md).

---

## 6.2 The seven gaps â€” locked defaults (Amin)

Addresses the open list under â€œno TTS / LLM+API / vectors / Fabricâ€.

| # | Topic | Locked default |
| --- | --- | --- |
| 1 | **Teacher video** | **5 Veo prompts / 5 videos** â€” one per emotion Ã— GA-16 (Â§6.1) |
| 2 | **PulseChunk on wire** | Logical PulseChunk = **ordered TickPackages** (KEY then Î”) pushed on **CHORUS Fabric** lane B; optional lane A `c_t` each tick. Not JSON-primary. |
| 3 | **ML I/O** | **In:** GA-16 (or wordâ†’vowel) script + emotion + tick count / durations from **LLM/API**. **Out:** per-tick **vectors** (prefer group drives â†’ known cells, or `c_t` compatible with Fabric). **No audio/TTS required.** |
| 4 | **Model packaging** | **Two separate models** (A state, B transfer) for clear train/loss; shared trunk later only if needed |
| 5 | **Statement / set boundary** | One **LLM/API utterance** (one assistant speak span / one API â€œsay thisâ€ job) = one PulseChunk |
| 6 | **Attack / hold / release** | Attack â‰ˆ **6 ticks (~100 ms)** 0â†’target; **hold** = remainder of word/set with emotion kept; release â‰ˆ **6 ticks** â†’ 0 or next. KEY at segment start / big changes; Î” otherwise |
| 7 | **Word path** | **W2-first**: vowels â†’ Model A/B â†’ slices; stitch to PulseChunk. W1 lexicon optional later for hot words |

**Duration source:** LLM/API timing or text-rate estimate â€” **not** face TTS/PCM.

**Emotion in samples:** carried every tick (hold tracks); vowel retargets lips/jaws/mouth while emotion stays up for the utterance.

**Between words:** short coarticulation / micro-REST (â‰¤ a few ticks), not a long silence unless the API marks a pause.

---

## 7. Acceptance (design-level)

A change that claims â€œVowelDesign doneâ€ must satisfy:

1. Still photo or HUD: AH, EE, OH, OU outlines are **pairwise distinguishable**.
2. Same caption + host timeline on two runs â†’ same vowel sequence (deterministic script).
3. LLM/API timing does not rename vowels; Model A/B supply shape + transfer.
4. No regression to consonant closures (PP still seals) when consonants return.
5. Old docs remain valid; this doc links any deliberate override.
6. Wire path is vectors on **CHORUS Fabric**, not TTS-centered.

---

## 8. Open decisions (work through with Amin)

- [x] Inventory count: **GA 16** (tags in Â§5.1)  
- [x] Face articulators: **6** (eyes, eyebrows, mouth, lips, teeth, jaws)  
- [x] Emotions: **6** (same product set)  
- [x] Time base: **60 Hz / 16.67 ms** ticks  
- [x] State-0 + **two datasets**: targets (576) + transfer for **whole set** (attack+hold+release)  
- [x] **PulseChunk** = one statementâ€™s ordered 16.7 ms cell/articulator samples  
- [x] PulseChunk on wire = ordered TickPackages on **CHORUS Fabric** (Â§6.2)  
- [x] Attack â‰ˆ6 ticks / hold / release â‰ˆ6 ticks; KEY then Î” (Â§6.2)  
- [x] Statement = one **LLM/API utterance** (Â§6.2)  
- [x] Word path **W2-first** (Â§6.2)  
- [x] Realistic lock: video â†’ A+B (Â§5.7)  
- [x] Teacher = truth; ML â†’ new words (Â§5.8)  
- [x] **Two separate models** A state / B transfer (Â§5.9, Â§6.2)  
- [x] Teacher shoot: **5 prompts â†’ 5 videos** (Â§6.1)  
- [x] Veo prompt drafts: [`VowelTeacherPrompts.md`](VowelTeacherPrompts.md)  
- [x] ML I/O: LLM script+emotion+duration â†’ per-tick vectors (Â§6.2)  
- [ ] Numeric pose table (Â§5.2) â€” optional detail once video measured  
- [ ] Visual clustering of vowels â€” optional  
- [ ] Relationship to TickFeed Â§14 plate hard-snap  
- [ ] Name of implementation milestone when we leave design-only  
- [ ] Link this doc from [`README.md`](README.md)

---

## 9. Out of scope for first acceptance

- Website SpeechSynthesis boundary wiring  
- Azure vs Docker Desktop GL  
- Full MFA / lab aligner  

---

## 10. Change log

| Date | Note |
| --- | --- |
| 2026-08-05 | Doc created â€” skeleton only; awaiting Amin decisions in Â§5â€“Â§8 |
| 2026-08-05 | Locked GA-16, 6 articulators, 6 emotions; Â§5.5 targets + transfer @ 16.7 ms |
| 2026-08-05 | Dataset B = whole set (attack + hold emotion to end + release), not ramp-only |
| 2026-08-05 | PulseChunk named: one statement, ordered 60 Hz samples for cells |
| 2026-08-05 | PulseChunk sample = all 6 groups each tick (eyesâ€¦jaws), zeros allowed |
| 2026-08-05 | Group = NWR object; sample includes all known cells per group |
| 2026-08-05 | Â§5.6 word â†’ PulseChunk layers W1â€“W5 (table / compose / host clock) |
| 2026-08-05 | Â§5.7 video teacher measures A+B; possible without 576 separate clips |
| 2026-08-05 | Â§5.8 teacher video trains ML; runtime new words â†’ model â†’ PulseChunk |
| 2026-08-05 | Â§5.9 Model A = state 1 targets; Model B = transfer trajectories |
| 2026-08-05 | Â§6.1 teacher shoot: Veo prompt; 1Ã—8s or 3Ã—8s parts |
| 2026-08-05 | Focus: no TTS; LLM/API sync; vectors; CHORUS Fabric |
| 2026-08-05 | Â§6.1 five emotion videos; Â§6.2 locks seven gaps; VowelTeacherPrompts.md |
