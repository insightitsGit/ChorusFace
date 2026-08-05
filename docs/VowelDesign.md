# VowelDesign

**Status:** Phase-1 **architecture CLOSED**; **NWR output retarget locked** — see [`VowelDesignNWRReconciliation.md`](VowelDesignNWRReconciliation.md). Implementation on branch `vowelBrnach` — [`VowelDesignImpl.md`](VowelDesignImpl.md).  
**Created:** 2026-08-05  
**Owns:** General American vowel inventory (16), six face articulators, per-tick (60 Hz) targets.  
**Does not replace:** older design docs below — they stay the source of truth until this doc is accepted and linked from [`docs/README.md`](README.md).

**Help packet (7 questions + full copy for sharing):** [`VowelDesignHelp.md`](VowelDesignHelp.md) · teacher prompts: [`VowelTeacherPrompts.md`](VowelTeacherPrompts.md)  
**Review merge (Gemini · GPT · Claude):** [`VowelDesignReviewMerge.md`](VowelDesignReviewMerge.md)  
**Round-2 questions:** [`VowelDesignDetailQuestions.md`](VowelDesignDetailQuestions.md) · **answers (Gemini · Claude · GPT D36–D42):** [`VowelDesignDetailAnswers.md`](VowelDesignDetailAnswers.md)  
**Final handoff:** [`VowelDesignFinalHandoff.md`](VowelDesignFinalHandoff.md) · **answers (ARCHITECTURE CLOSED):** [`VowelDesignFinalAnswers.md`](VowelDesignFinalAnswers.md)  
**NWR reconciliation (Step 0):** [`VowelDesignNWRReconciliation.md`](VowelDesignNWRReconciliation.md)

### Locked premises (Amin, 2026-08-05)

| Premise | Value |
| --- | --- |
| Speech inventory | **General American ≈ 16 vowel phonemes** (incl. common diphthongs) |
| Face articulators | **6:** eyes · eyebrows · mouth · lips · teeth · jaws |
| Emotions | **6:** NEUTRAL · HAPPY · SAD · SURPRISED · ANGRY · THINKING |
| Time base | TickFeed **60 Hz** (≈ **16.67 ms**); cell/LOOK values may change every tick |
| State-0 | At utterance / segment start, face articulator state is **0** (rest) |
| Two datasets | **(A) targets** 16×6×6 = **576** end-states · **(B) transfer** tick path for whole set |
| Wire unit | **PulseChunk** — utterance clock + GA-16/emotion schedule (debug 9D optional) |
| Product focus | **No TTS ownership** — sync with **LLM/API**; payload is **vectors**; transport **CHORUS Fabric** |
| Render substrate | **NWR** — geometry+identity rigid; **motion via biomechanics muscle impulses** (not cell KEY writes on Master-Locked regions) |
| Delivery path | Host utterance → compose → `schedule_spans` → `_fire_impulse` → `BiomechanicalFace.submit_phoneme` |

## Relationship to existing docs (keep)

| Doc | Still owns |
| --- | --- |
| [`TickFeedDesign.md`](TickFeedDesign.md) | TickFeed Side A/B, LOOK/FIELD, post-initial mouth band (§14) |
| [`PhoneticFidelity.md`](PhoneticFidelity.md) | Full viseme set, host timeline vs PCM aligner, lip-reading goals |
| [`VoiceSync.md`](VoiceSync.md) | `/voice/expect` → `/pcm` → `/end`, energy placement |
| [`DisplayLayers.md`](DisplayLayers.md) | L00–L11 plate / field order |
| [`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md) | 8s Side B teacher take (includes OPEN / hi / TH) |
| [`ProductBeta.md`](ProductBeta.md) · [`FaceServiceEmbed.md`](FaceServiceEmbed.md) | Host TTS default, FaceBridge surface |

This file is the **vowel-only** redesign workspace. Consonants (PP, FF, TH, …), full TickFeed packaging, and Docker/embed are out of scope here unless a vowel decision forces a cross-link.

---

## 1. Problem (why this doc)

Observed failure mode in dogfood: mouth reads as **simple open/close**, not as distinct spoken vowels. Vowels are where lip readers and casual viewers judge “is this talking?” — if AH / EE / OH / OU collapse to one jaw pump, the face looks broken even when closures are correct.

**Design goal:** each canonical vowel must produce a **visibly different lip silhouette** (width × openness × roundness), driven by host timing, without inventing face RGB.

---

## 2. Goals

1. Stable **vowel inventory** (names + aliases) agreed for host timelines and grapheme fallback.
2. Per-vowel **pose targets** (jaw / open / width / round) that survive TickFeed plates + muscle field.
3. Clear **authority** when host PCM, host timeline, and text-only cue disagree.
4. Calibration / plate requirements for vowels called out separately from consonant kit.
5. Acceptance tests that a human (or HUD) can score: “AH ≠ EE ≠ OU” at a glance.

## 3. Non-goals

- **TTS** — ChorusFace does **not** own speech audio for this design. No requirement
  to synthesize or play voice. (Host may speak separately; we only need sync cues
  from the **LLM/API** — text, emotion, timing — not a face TTS pipeline.)
- Acoustic vowel ASR inside ChorusFace.
- Rewriting full Oculus consonant set (see PhoneticFidelity).
- Generative mouth interior / painted teeth.
- Insightits landing bridge heuristics (website dogfood owns that integration).
- JSON-heavy REST as the primary realtime face drive (vectors + Fabric instead).

### 3.1 Product focus (Amin — locked)

```text
LLM / host API  →  (text + emotion + timing)  →  PulseChunk as vectors
                →  CHORUS Fabric  →  NWR cells / GPU
```

| We care about | We do **not** center |
| --- | --- |
| Sync with **LLM + API** (what was said, emotion, when) | Owning or implementing **TTS** |
| **Vectors** on cells / groups / compact codes | Mouth-cue-only `/prism/speak` as the long-term path |
| **CHORUS Fabric** transport (see TickFeedDesign §6.2) | Browser Web Speech as the face clock |
| **NWR** addressable ROI cells (6 articulator objects) | Frame-by-frame generative video faces |
| Teacher→Model A/B → PulseChunk | Local `--tts` lab path |

PulseChunk samples are **vector data** (group→cell drives, or compact \(c_t\) +
expand). Fabric carries those vectors (TickFeed: fixed-dim float lanes + KEY/Δ
packages). Exact binding: PulseChunk ↔ Fabric lanes — same family as
existing CHORUS ↔ TickPackage.

### 3.2 Why NWR (scoped recommendation — locked)

Using **NWR** (Neural World Representation / cell field under ChorusFace) inside
VowelDesign is a **massive upgrade if scoped correctly** — not as a full-frame
generative video model every 16.7 ms, but as the **structured face substrate**.

| Advantage | What it means for us |
| --- | --- |
| **1. Spatial precision** | Face = addressable ROI **cells** on objects (mouth, lips, teeth, jaw, eyes, brows) — not blurry full-frame pixels. Model A/B output Δ drives per **cell cluster**; GPU deforms regions deterministically → less flicker / soft lips / floating teeth. |
| **2. Micro-latency @ 60 Hz** | Do **not** regenerate 1080p with heavy generative models each tick. Send lightweight float arrays (`c_t` / KEY·Δ packages) on Fabric to the shader/render node; keep end-to-end sync in the ~tens of ms. |
| **3. Anatomical independence** | The 6 articulators are **distinct NWR cell clusters**. Jaw drop (AA) without forcing lip-wide (EE) — fixes the classic **jaw-pump** collapse of all open vowels. |

**Setup rules (locked):**

1. **Keep geometry rigid, motion vectorized** — NWR + baseline avatar photo own lighting, skin, mouth-interior shading; PulseChunk supplies **motion deltas only**.  
2. **Anchor Side B collect to NWR digest** — optical flow / landmarks from the 5 Veo teachers map into this avatar’s **`.bds` / region catalog** ROI cells so train and render share one coordinate system.  
3. **Bridge** high-level LLM phoneme script ↔ low-level GPU cell execution through that shared NWR address space.

```text
LLM phoneme script
  → Model A/B (group / latent vectors)
  → PulseChunk Δ @ 16.7 ms
  → CHORUS Fabric
  → NWR cells (same .bds ROI as Side B teacher)
  → GPU (identity photo + plates stay; motion is vectorized)
```

---

## 4. Current baseline (as of product code)

Canonical vowels today (`chorusface.speech`):

| Name | Family (informal) | Role |
| --- | --- | --- |
| `AH` | open central | max jaw contrast |
| `AA` | open back/central | Oculus `aa` family |
| `EH` | mid front | “bed” |
| `IH` | near-close front | “bit” |
| `EE` | close front / wide | smile-wide outline |
| `OH` | mid back round | “toe” |
| `OU` | close back round | “boot” / book |

Grapheme map (fallback only): A→AH, E/I/Y→EE, O→OH, U→OU (+ digraphs OO/OU/OW…).

**Known gap:** dogfood paths that only pump “open vs CLOSED” discard this inventory. Energy without a vowel script also collapses motion to amplitude.

---

## 5. Proposed vowel contract (draft — fill with Amin)

### 5.1 Inventory — General American 16 (locked count)

**Count locked: 16.** Names below are the working phoneme set for design
(Wells-style keywords). Host timelines may still send ChorusFace/Oculus
aliases; mapping 16 → visual drives is §5.2–5.3.

| # | Keyword | IPA (approx.) | Working tag | Notes |
| --- | --- | --- | --- | --- |
| 1 | fleece | i | `EE` | close front |
| 2 | kit | ɪ | `IH` | |
| 3 | face | eɪ | `EY` | diphthong |
| 4 | dress | ɛ | `EH` | |
| 5 | trap | æ | `AE` | |
| 6 | lot / palm | ɑ | `AA` | |
| 7 | thought | ɔ | `AO` | may merge with lot in some speakers |
| 8 | goat | oʊ | `OH` | diphthong |
| 9 | foot | ʊ | `UH` | |
| 10 | goose | u | `OU` | |
| 11 | strut | ʌ | `AH` | open-ish central |
| 12 | comma / about | ə | `AX` | schwa — often visually weak |
| 13 | nurse | ɝ / ɚ | `ER` | rhotic |
| 14 | price | aɪ | `AY` | diphthong |
| 15 | mouth | aʊ | `AW` | diphthong |
| 16 | choice | ɔɪ | `OY` | diphthong |

**Design rule:** all 16 are first-class in the **speech script**. They do **not**
all need 16 unique photographs — several may share a visual class (e.g. schwa
near `AH`/`EH`) as long as the script still names the phoneme.

### 5.1b Six face articulators (locked)

For every tick (and every vowel), design may drive these **six** parts:

| # | Part | What changes (design intent) | Typical TickFeed / LOOK hook |
| --- | --- | --- | --- |
| 1 | **Eyes** | gaze / aperture (blink lids) | lid / eye LOOK |
| 2 | **Eyebrows** | raise / knit | brow label |
| 3 | **Mouth** | cavity / interior show (gap) | open plate / mouth_gap |
| 4 | **Lips** | width, round, seal / spread | lip outline / oris |
| 5 | **Teeth** | visibility (upper/lower show) | open/smile plates — photographed |
| 6 | **Jaws** | drop / close | jaw drive |

Yes — **6**. Note: **mouth** (cavity/gap) and **lips** (rim shape) are separate
on purpose so “open jaw + rounded lips” (OH) ≠ “open jaw + wide lips” (AE/AH).

Vowels mainly stress **lips · teeth · jaws · mouth**; eyes/eyebrows stay mostly
idle unless the beat is expressive (surprise, smile eyes). Consonants will reuse
the same 6 later — not redefined here.

### 5.2 Pose targets (design prior — Gemini table; teacher measurement wins)

Per vowel × six articulators. `0` = rest/off, `1` = max. Sparse eyes/brows OK for
pure speech. **Measured Side B holds override these numbers** when they disagree.

| Tag | eyes | brows | mouth | lips | teeth | jaws | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EE | 0.0 | 0.1 | 0.2 | 0.9 wide | 0.8 | 0.2 | close front |
| IH | 0.0 | 0.0 | 0.3 | 0.5 | 0.5 | 0.3 | |
| EY | 0.0 | 0.1 | 0.4 | 0.7 | 0.6 | 0.4 | diphthong EH→EE |
| EH | 0.0 | 0.0 | 0.5 | 0.4 | 0.5 | 0.5 | |
| AE | 0.0 | 0.1 | 0.7 | 0.8 | 0.8 | 0.7 | |
| AA | 0.0 | 0.0 | 0.9 | 0.2 | 0.4 | 0.9 | |
| AO | 0.0 | 0.0 | 0.8 | 0.6 | 0.2 | 0.8 | |
| OH | 0.0 | 0.0 | 0.6 | 0.8 round | 0.1 | 0.6 | diphthong AO→OU |
| UH | 0.0 | 0.0 | 0.3 | 0.5 | 0.1 | 0.3 | |
| OU | 0.0 | 0.0 | 0.2 | 1.0 tight | 0.0 | 0.2 | |
| AH | 0.0 | 0.0 | 0.6 | 0.3 | 0.3 | 0.6 | |
| AX | 0.0 | 0.0 | 0.2 | 0.1 | 0.1 | 0.2 | schwa |
| ER | 0.0 | 0.0 | 0.3 | 0.4 | 0.2 | 0.3 | |
| AY | 0.0 | 0.1 | 0.8 | 0.6 | 0.7 | 0.8 | diphthong AA→IH |
| AW | 0.0 | 0.1 | 0.8 | 0.7 | 0.5 | 0.8 | diphthong AA→UH |
| OY | 0.0 | 0.0 | 0.7 | 0.8 | 0.5 | 0.7 | diphthong AO→IH |

### 5.3 Timing rules (vowels only)

- Host timeline spans win for start/end.
- With PCM + expect: energy **places** the next scripted vowel; it does not invent a new vowel name.
- Closures (PP/MM/CLOSED) always interrupt vowel holds (already in TickFeed §14 — reaffirm here).
- Max vowel hold / coarticulation into next vowel: **TBD**.

### 5.4 Host API surface (vowels)

Preferred:

```http
POST /voice/timeline
{ "caption": "…", "spans": [
  {"phoneme": "AH", "start": 0.00, "end": 0.12},
  {"phoneme": "EE", "start": 0.12, "end": 0.20}
], "replace": true }
```

Alternate: `/voice/expect` + `/voice/pcm` with caption that expands to the vowel script via `polygon_speech` (or successor).

Mouth-cue-only `/prism/speak` remains **non-authoritative** for vowel QA.

---

## 5.5 Two datasets: targets + transfer (Amin — locked intent)

Yes: we need **more than** the 576 end poses.

### Dataset A — Target states (what we aim for)

```text
16 vowels × 6 articulators × 6 emotions  =  576  target values
```

Each cell is the **arrived** face drive for that (vowel, emotion) on that articulator  
when the shape is “there” (hold / peak), not how we got there.

### Dataset B — Transfer data for the **whole segment** @ 16.7 ms

At **time 0** the face articulator state is **0** (rest / zero).  
Speech does not teleport to a target; every **≈ 16.67 ms** we emit the next
transfer sample. Dataset B is **not only the ramp-in** — it covers the
**entire set / segment** until we leave that delivery:

```text
phase ATTACK   0 → target     (enter vowel×emotion, e.g. start ANGRY)
phase HOLD     stay at / near target for the rest of the segment
               (e.g. keep ANGRY through the whole angry line — do not drop)
phase RELEASE  target → 0 or → next segment   (only when the set ends)
```

Example — one angry utterance lasting ~1.0 s (~60 ticks):

```text
t = 0          state = 0
t = 1…K ·Δt    attack → ANGRY×vowel target     Δt = 16.67 ms
t = K+1…M ·Δt  HOLD — keep anger (and active vowel shaping) for the whole set
t = M+1…End    release / handoff when the segment actually ends
```

So if the emotion is **ANGRY**, Dataset B must **hold that anger to the end of
the set**, not fade after the first vowel peak. Vowel changes inside the set
still retarget lips/jaws/mouth tick-by-tick; emotion sustain stays up until
the segment boundary.

```text
“motion”   = how values change across ticks
targets A  = where a vowel×emotion wants to sit (576)
transfer B = full tick series for the segment: attack + hold + release
```

Same 60 Hz TickFeed clock as [`TickFeedDesign.md`](TickFeedDesign.md).

### Design consequences (still open numerically)

| Topic | Intent | Open |
| --- | --- | --- |
| Segment = “set” | One emotion span (e.g. whole ANGRY line) or one vowel only? | Prefer **emotion span** — hold emotion for whole set |
| Attack length | Ticks 0 → first target | TBD (~50–200 ms) |
| Hold | Repeat / sustain samples while set continues | Required — not optional |
| Release | Only at set end → 0 or next set | TBD |
| Stored form | Absolute state each tick vs delta from previous | Align with KEY/Δ if possible |
| Vowel changes inside hold | New vowel targets while emotion stays | Yes — coarticulation inside the set |

**Not the same as** “576 video clips.” A is the pose table; B is the **full
60 Hz time series** for each played set (attack + hold through the end + release).

### PulseChunk — one statement on the wire (Amin)

Join A + B into a single sendable unit for **one term / statement**.

**Example:** AI should say *“Hi, how are you?”* with **HAPPY**.

One **PulseChunk** carries **everything** needed to play that statement on the
face: ordered samples every **≈ 16.67 ms** from state **0**, through the whole
line (attack + hold HAPPY + vowel changes per word), to release — so Side A can
drive **cells** (and the six articulators) without asking for more mid-utterance.

```text
PulseChunk {
  statement:  "Hi, how are you?"
  emotion:    HAPPY          // held for the whole set
  tick_hz:    60
  t0_state:   0              // rest at start
  samples: [                 // strict time order
    { t: 0,    … cell / articulator drives … },
    { t: 1,    … },          // +16.7 ms
    { t: 2,    … },
    …
    { t: M,    … }           // end of statement / release
  ]
}
```

| Property | Rule |
| --- | --- |
| Scope | **One statement** (one term), not the whole conversation |
| Emotion | One primary emotion for the chunk (e.g. HAPPY) — held per Dataset B |
| Ordering | Samples **must** be in tick order; no gaps in the play clock |
| Cadence | One sample per tick (60 Hz) unless we later allow sparse Δ with explicit tick index |
| Groups | **Every sample includes all 6 articulator groups** — not mouth-only |
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

### Groups = NWR objects → cells inside objects (Amin)

Yes. **Eye** and **mouth** (and the other four) are **groups**, and in NWR terms a
group is an **object**: a connected **cluster of cells**, not a separate mesh
(see [`AvatarCellDataflow.md`](AvatarCellDataflow.md), [`NWRDataDesign.md`](NWRDataDesign.md)).

Hierarchy inside one PulseChunk sample:

```text
PulseChunk sample @ tick t
  └─ group / object  (e.g. mouth, eyes, …)     ← one of the 6
       └─ all known cells for that object       ← from NWR region / catalog
            └─ channel values (phase-1: vx, vy; later as needed)
```

| Rule | Meaning |
| --- | --- |
| Object | Named group = cell cluster addressable in the world (`.bds` + `region_catalog`) |
| Membership | PulseChunk carries data for **all known cells** of that group, not a single centroid only |
| Source of “known cells” | NWR digest / region catalog for this avatar world (TickFeed face ROI cells in that object) |
| Nesting | Objects can contain sub-structure later (e.g. mouth → lips/teeth/cavity) without breaking the 6 top-level groups |
| Empty motion | A cell may have Δ=0 this tick; it stays listed so membership is stable |

So: **PulseChunk → groups (objects) → every known cell in that object**, every
16.7 ms, in order for the statement.

Relation to existing TickFeed:

```text
PulseChunk   =  statement-level bundle (host → face)
TickPackage  =  one tick’s KEY/Δ bytes (already in TickFeedDesign)
```

A PulseChunk is the **outer envelope**: many TickPackages (or equivalent sample
records) **in order** for one spoken line. Exact binary layout TBD — design
name and meaning are locked here first.

```text
Host AI:  "Hi, how are you?" + HAPPY
    → build PulseChunk (all 16.7 ms steps)
    → Face plays samples in order from state 0
```

---

## 5.6 How do we provide data for each word? (design question)

We know the datasets and PulseChunk shape. The next design problem:

> For each **word** inside a statement, where do the 16.7 ms group/cell samples
> come from before they are joined into one PulseChunk?

Example statement: *“Hi, how are you?”* + HAPPY → one PulseChunk, but built from
word (or vowel) pieces in order:

```text
[hi] + [how] + [are] + [you]   (+ spaces / coarticulation)
        ↓ join on 60 Hz timeline
     PulseChunk(statement, HAPPY)
```

### Layered answer (aligns with product rule: known → table, unknown → ML)

| Layer | When | What we store / run | Provides |
| --- | --- | --- | --- |
| **W1 — Word table** | Word is **known** | Precomputed transfer series for that word (all 6 groups × cells, or compact code) at 60 Hz, often at NEUTRAL; emotion applied as hold overlay | Best fidelity for frequent words (“hi”, “you”, …) |
| **W2 — Vowel compose** | Word unknown or no row | Grapheme/phoneme → GA-16 tags → Dataset **A** targets + Dataset **B** attack/hold/release curves | Default path for any English word |
| **W3 — Measured take** | Side B teacher / calibration | Video → tick samples for beats/words (existing TickFeed collect) | Seeds W1/W2 tables; not required every live chat |
| **W4 — Host timing** | Live product | Host gives text + emotion (+ optional PCM or phoneme times) | **When** each word/vowel sits on the clock; face fills **what** from W1/W2 |
| **W5 — ML cover** | Gaps / unseen transitions | Fill missing transfer ticks (same spirit as behavior ML) | Only where W1–W2 have no sample |

```text
Host:  text + emotion (+ optional audio clock)
  → tokenize words
  → for each word:  W1 lookup  or  W2 compose from vowels
  → stitch on 60 Hz (coarticulation at boundaries)
  → apply emotion HOLD for whole statement (Dataset B)
  → emit one PulseChunk
```

### Per-word payload (design intent)

For word \(w\) with duration \(D\) seconds ≈ \(N = \mathrm{round}(D \times 60)\) ticks:

```text
WordSlice {
  word: "how"
  vowels: [AW] or [AH, OU] …     // from GA-16 script
  ticks: N
  samples[0..N): each sample = 6 groups → all known cells
}
```

WordSlices concatenate into the statement PulseChunk (with short boundary
blends — TBD).

### Word-path decisions (locked in §6.2)

- [x] **Primary live path:** W2-first (vowel compose via Model A/B)  
- [x] **Duration:** LLM/API timing or text estimate — not TTS/PCM  
- [x] **Emotion:** held on every tick for the utterance  
- [x] **Between words:** micro-REST / short coarticulation  
- [ ] Minimum W1 lexicon size for beta — optional later

### 5.7 Lock to avatar + GPU via video teacher (Amin — intent)

**Yes — that makes sense, and it is possible.**

To make Dataset A/B and PulseChunks **realistic for this avatar’s GPU path**
(same identity photo, same cell objects, same LOOK/FIELD recipe), we need a
**video avatar sample** (teacher take of *this* face), then **detect / measure**
from that video:

| Extract from video | Becomes |
| --- | --- |
| Held shapes for vowel × emotion (articulator drives on known cells) | **Dataset A** targets (toward the 576 space) |
| Frame→tick motion from rest through hold to release @ ~60 Hz | **Dataset B** transfer series |
| Region membership (mouth, eyes, …) on the NWR grid | Group → cell lists for PulseChunk samples |

```text
Video avatar (this face)
  → detect landmarks / optical flow / lids (Side B collect family)
  → resample to 60 Hz ticks on known cells per group
  → Dataset A peaks + Dataset B full-set transfers
  → bake into world (tables / tracks) used by GPU display recipe
  → live: compose PulseChunk from those locked measurements
```

This is the same **Side B** idea as [`TickFeedDesign.md`](TickFeedDesign.md) /
[`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) /
[`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md): video teaches
**this** avatar; runtime does not invent a different face.

#### Possible? Yes — with a practical shoot plan

| Claim | Reality |
| --- | --- |
| Need a video of the avatar face | **Yes** — required for realistic lock |
| Need all 576 as *separate* clips | **No** — one (or few) structured takes can cover the matrix |
| Detect 576 target points | **Yes** — as **measurements** at hold frames (vowel×emotion), not 576 files |
| Dataset B from video | **Yes** — resample video timeline → 16.7 ms ticks (attack/hold/release) |
| Lock to GPU schematic | **Yes** — same `.bds` cells, plates, region catalog, display recipe |

**Shoot design (not 576 videos):** e.g. per emotion block, walk GA-16 vowels
(rest → vowel → hold → rest) on camera; six emotion blocks (or denser single
script). Detector fills A at holds and B along the ticks. Sparse cells /
unseen combos → W2 curves or W5 ML later — still design-valid.

**Possible risks (design, not blockers):** thought/lot merger in speech vs face;
schwa visually weak; some emotions barely move mouth; need clear REST=0 between
blocks so transfers start from state 0.

### 5.8 Teacher video → ML model → any word (Amin — intent)

**Yes — that makes sense.**

The teacher video **cannot** speak every English word. It **can** feed **true**
measured motion on **this** avatar (cells ↔ GPU vectors). That truth trains an
**ML model**. At runtime, a **new word** is run through the model, which outputs
the PulseChunk materials (toward Dataset A space + Dataset B transfers).

```text
Teacher video (limited script, this face)
  → Side B measure: true ticks on NWR cells / groups / GPU vectors
  → train ML  (input: word or GA-16(+emotion) script · output: tick series)
  → Runtime: new word → model → samples @ 16.7 ms → PulseChunk
```

| Role | What |
| --- | --- |
| Teacher video | **Ground truth** only — how *this* avatar actually moves; locks identity + cell topology |
| Dataset A/B from video | Supervision targets (holds + full-set transfers), not the full English lexicon |
| ML model | Learns mapping from speech units → avatar/GPU tick drives |
| New word at runtime | Model inference (W5 / extended W2) — not “find that word in the video” |
| Tables W1 | Optional cache of frequent words **after** model or from measured beats |

**Train (offline):**

```text
video ticks (true) + labels (vowel / emotion / beat / word if known)
  → fit model so outputs match measured group/cell drives @ 60 Hz
```

**Infer (live):**

```text
"chrysanthemum" + HAPPY
  → phoneme/vowel script (GA-16)
  → model(script, emotion, duration)
  → ordered samples (6 groups × known cells)
  → PulseChunk
```

So: **video teaches truth connected to avatar+GPU; ML generalizes to all words.**
Same spirit as existing “known → table, unknown → ML cover” — here the teacher
is the source of truth for training, not a dictionary of every word.

Open (design): model input features (phoneme one-hots vs audio embedding vs text);
output (dense cells vs compact code \(c_t\) vs 6-group controls expanded to cells).

### 5.9 Two models (or two heads): state vs transfer (Amin — intent)

**Yes — splitting is right.** Dataset A and Dataset B are different jobs; one
monolithic net often muddies both.

| Model | Learns | Answers | Output |
| --- | --- | --- | --- |
| **Model A — State (target)** | Dataset A | “What is **state 1** / the held pose for this vowel×emotion on the 6 groups×cells?” | Peak / hold target (toward 576 space) |
| **Model B — Transfer** | Dataset B | “How do we **walk** tick-by-tick from state **0** → state 1, hold for the set, release?” | Ordered samples @ 16.7 ms (attack + hold + release) |

```text
New word + emotion
  → Model A  →  target state (state 1)
  → Model B  →  transfer series given (state 0, state 1, duration / set length)
  → stitch → PulseChunk
```

**Why two:**

- A is mostly **static / peak** (where the face sits).
- B is **dynamics** (how values change every tick) — including keeping anger for
  the whole set.
- Teacher video supervises **both**, but loss/metrics differ (pose match vs
  trajectory match).

**Packaging options (either is fine in design):**

1. **Two models** (clear ownership — preferred for VowelDesign clarity)  
2. **One trunk, two heads** (shared encoder, A-head + B-head) — same split, shared features  

Locked intent: **separate A vs B predictors**, not one net that only emits a
single pose and hopes an ad-hoc ease replaces Dataset B.

---

## 6. Calibration / plates (vowels)

Teacher take today ([`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md)): REST → SMILE → OPEN(“ah”) → hi → think → …  
That **8s dense kit stays** for the old path. VowelDesign adds a **separate teacher shoot** aimed at GA-16 × emotions × transfers.

### 6.1 Teacher shoot plan (locked — after Gemini/GPT/Claude merge)

**5 emotion prompts**; pacing and REST refined by review merge
([`VowelDesignReviewMerge.md`](VowelDesignReviewMerge.md)).

| # | Prompt family | Emotion | Default length |
| --- | --- | --- | --- |
| 1 | `VowelTeacher_HAPPY` | HAPPY | 1×8s (split if needed) |
| 2 | `VowelTeacher_SAD` | SAD | **2×8s** morphological Part1/Part2 (D6) |
| 3 | `VowelTeacher_SURPRISED` | SURPRISED | 1×8s (split if needed) |
| 4 | `VowelTeacher_ANGRY` | ANGRY | **2×8s** morphological Part1/Part2 (D6) |
| 5 | `VowelTeacher_THINKING` | THINKING | 1×8s (split if needed) |

```text
Per clip:
  REST → V1 → REST → V2 → … → REST
  REST-under-emotion:
    mouth/lips/teeth/jaws = state 0
    eyes/brows KEEP the clip emotion   ← critical (Claude)
```

| Rule | Choice |
| --- | --- |
| Generator | Veo; framing matches blonde calibration take |
| Phase-1 takes | One clean performance per slot |
| Phase-2 variation | Up to **3 performances** per prompt (GPT) |
| Consonants / old kit | Existing [`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md) |

Prompt paste text: [`VowelTeacherPrompts.md`](VowelTeacherPrompts.md).

---

## 6.2 The seven gaps — locked defaults (after review merge)

Full rationale: [`VowelDesignReviewMerge.md`](VowelDesignReviewMerge.md).

| # | Topic | Locked default |
| --- | --- | --- |
| 1 | **Teacher video** | 5 emotion prompts; SAD/ANGRY = **morphological** 2×8s (D6); REST-under-emotion keeps eyes/brows |
| 2 | **PulseChunk on wire** | Hierarchical: metadata → **WordSlices** → TickPackages KEY/Δ on Fabric lane B; Phase-1 **defer** learned `c_t` (D21); **32-byte little-endian** header + CRC (D18; match TickPackage) |
| 3 | **ML I/O (Phase 1)** | LLM → GA-16 + emotion + timing; train **9D group controls** (D7) + frozen radial \(W\) expand (D8); diphthongs = 2-point + smoothstep blend (D14). Phase 2: latent `c_t` |
| 4 | **Model packaging** | Model A MLP **22→64→64→9** (D12); Model B Δ residuals + endpoint/smooth/jaw losses (D13); train A on holds; freeze A; train B on paths |
| 5 | **Statement boundary** | Host **`utterance_id`**; absolute `start_s`/`end_s` → `tick=round(s*60)` (D27); **`emotion_track[]`** |
| 6 | **Attack / hold / release** | Attack ticks (D15): ANGRY 4, SURPRISED 4, HAPPY 5, NEUTRAL/THINKING 6, SAD 9; spread↔round → 2-tick neutral bridge else 4-tick cosine (D16) |
| 7 | **Word path** | LLM phoneme script primary; WordSlice timing; 3-tick **9D** crossfade (D28); fallback dict→G2P→**REST hold** (D25); never invent vowel shapes |

**Duration source:** LLM/API — **not** face TTS/PCM.  
**Confidence:** cascade D17 (full ML → A+spline → W1 / REST).  
**Latency:** E2E ≤50 ms planning budget; Fabric topology is the real risk (D23).  
**Milestone:** `MILESTONE_VOWEL_VECTOR_CORE` — “Measured Vowel Motion” (D32).

Full Round-2 merge: [`VowelDesignDetailAnswers.md`](VowelDesignDetailAnswers.md).

---

### 6.3 Round-2 + Final handoff (ARCHITECTURE CLOSED)

Full freeze: [`VowelDesignFinalAnswers.md`](VowelDesignFinalAnswers.md) (F1–F20 merge).

| Area | Lock |
| --- | --- |
| Host API | Required `utterance_id`+`text`+`emotion_track`; optional `spans[].tag`; face owns G2P → REST on fail; stream first WordSlice |
| Group vector | **9D ONNX map F9** (Eyes2 Brows2 Mouth1 Lips2 Teeth1 Jaws1) |
| Plates vs cells | Deterministic 4×6 class×emotion plate stub (F10); cell Δ residual |
| Wire | PulseChunk **PLS1** 32B LE + optional 12B version ext; WordSlice 12B; inner TickPackages stay `TPK1` 64B |
| KEY | Δ ≥ KEY size; KEY at t0, WordSlice starts, hold→release, emotion boundary, spool boundary |
| Teacher | 5 emotions / **7 clips** default; Teacher Package v1 layout; D35 GO/NO-GO before shoot |
| Next gate | **D35 / F12** only — then implement; no more architecture docs |

---

## 7. Acceptance (design-level)

A change that claims “VowelDesign done” must satisfy:

1. Still photo or HUD: AH, EE, OH, OU outlines are **pairwise distinguishable**.
2. Same caption + host timeline on two runs → same vowel sequence (deterministic script).
3. LLM/API timing does not rename vowels; Model A/B supply shape + transfer.
4. No regression to consonant closures (PP still seals) when consonants return.
5. Old docs remain valid; this doc links any deliberate override.
6. Wire path is vectors on **CHORUS Fabric**, not TTS-centered.

---

## 8. Open decisions (work through with Amin)

- [x] Inventory count: **GA 16** (tags in §5.1)  
- [x] Face articulators: **6** (eyes, eyebrows, mouth, lips, teeth, jaws)  
- [x] Emotions: **6** (same product set)  
- [x] Time base: **60 Hz / 16.67 ms** ticks  
- [x] State-0 + **two datasets**: targets (576) + transfer for **whole set** (attack+hold+release)  
- [x] **PulseChunk** = one statement’s ordered 16.7 ms cell/articulator samples  
- [x] PulseChunk on wire = ordered TickPackages on **CHORUS Fabric** (§6.2)  
- [x] Attack ≈6 ticks / hold / release ≈6 ticks; KEY then Δ (§6.2)  
- [x] Statement = one **LLM/API utterance** (§6.2)  
- [x] Word path **W2-first** (§6.2)  
- [x] Realistic lock: video → A+B (§5.7)  
- [x] Teacher = truth; ML → new words (§5.8)  
- [x] **Two separate models** A state / B transfer (§5.9, §6.2)  
- [x] Teacher shoot refined after Gemini/GPT/Claude (§6.1, ReviewMerge)  
- [x] Veo prompts updated: [`VowelTeacherPrompts.md`](VowelTeacherPrompts.md)  
- [x] Seven gaps re-locked after review merge (§6.2)  
- [x] Pose prior table filled (§5.2)  
- [x] Round-2 three-way merge (Gemini · Claude · GPT D36–D42) → [`VowelDesignDetailAnswers.md`](VowelDesignDetailAnswers.md)  
- [x] Morphological SAD/ANGRY Part1/Part2 (D6 Claude AY/ER placement) in teacher prompts  
- [x] Phase-1 control vector = **9D** (D7); optional +3 if D30 fails  
- [x] Milestone: **`MILESTONE_VOWEL_VECTOR_CORE`** / Measured Vowel Motion (D32)  
- [x] Consonants Phase-1 = TickFeed §14 only; no reserved PulseChunk slots (D33)  
- [x] **Final handoff F1–F20 merged** → [`VowelDesignFinalAnswers.md`](VowelDesignFinalAnswers.md) — **ARCHITECTURE CLOSED**  
- [x] Host API / binary / 9D ONNX / plate stub / acceptance floors frozen  
- [ ] Phase-2 latent facial space / learned `c_t` — not Phase-1 block  
- [ ] **D35 / F12** Veo landmark GO/NO-GO (before full shoot)  
- [ ] Teacher Package v1 + full shoot (if D35 pass)  
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
| 2026-08-05 | Doc created — skeleton only; awaiting Amin decisions in §5–§8 |
| 2026-08-05 | Locked GA-16, 6 articulators, 6 emotions; §5.5 targets + transfer @ 16.7 ms |
| 2026-08-05 | Dataset B = whole set (attack + hold emotion to end + release), not ramp-only |
| 2026-08-05 | PulseChunk named: one statement, ordered 60 Hz samples for cells |
| 2026-08-05 | PulseChunk sample = all 6 groups each tick (eyes…jaws), zeros allowed |
| 2026-08-05 | Group = NWR object; sample includes all known cells per group |
| 2026-08-05 | §5.6 word → PulseChunk layers W1–W5 (table / compose / host clock) |
| 2026-08-05 | §5.7 video teacher measures A+B; possible without 576 separate clips |
| 2026-08-05 | §5.8 teacher video trains ML; runtime new words → model → PulseChunk |
| 2026-08-05 | §5.9 Model A = state 1 targets; Model B = transfer trajectories |
| 2026-08-05 | §6.1 teacher shoot: Veo prompt; 1×8s or 3×8s parts |
| 2026-08-05 | Focus: no TTS; LLM/API sync; vectors; CHORUS Fabric |
| 2026-08-05 | §6.1 five emotion videos; §6.2 locks seven gaps; VowelTeacherPrompts.md |
| 2026-08-05 | Merged Gemini+GPT+Claude → VowelDesignReviewMerge.md; §6.1/6.2/5.2 updated |
| 2026-08-05 | §3.2 NWR scoped: rigid geometry + vectorized motion; anti jaw-pump |
| 2026-08-05 | Round-2 Gemini D1–D35 + GPT D36–D42 → DetailAnswers; §6.2/6.3; morphological D6 prompts |
| 2026-08-05 | Claude Round-2 merged: 9D D7, LE header, D6 AY/ER, REST nuance, D25 REST-hold; DetailAnswers rewrite |
| 2026-08-05 | Final handoff packet F1–F20 → VowelDesignFinalHandoff.md (close remaining gaps to 100%) |
| 2026-08-05 | FinalAnswers three-way merge (GPT·Gemini·Claude) — ARCHITECTURE CLOSED; next = D35 |
