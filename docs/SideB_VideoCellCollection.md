# Side B — Collect whole-face cell data from avatar video

**Status:** design only (not implemented).  
**Master:** [`TickFeedDesign.md`](TickFeedDesign.md).  
**Pair with:** [`CellFeedBandwidth.md`](CellFeedBandwidth.md) (Side A) ·
[`MultiLayerTickML.md`](MultiLayerTickML.md) · [`AvatarCapture.md`](AvatarCapture.md)

Side A asks how to pass ~241 MB/s.  
Side B asks: **how do we obtain that information from the video avatar?**

Sparse landmarks / 8 group controls are **not enough**. This doc is the collection design.

---

## 0. Side B design at a glance

```text
Scripted calibration video (ordered beats, ~8s)
        │  labels known by time (smile / hi / angry / …)
        ▼
Split every frame → register to .bds face box
        ▼
Dense whole-face track → rasterize to cells @ 60 Hz
        ▼
Align audio/words on same ticks
        ▼
FaceCellTimeline (teacher) + LOOK plates
        ▼
Multi-layer ML (L1–L5) trains independently on packets
        ▼
Live: TickPackage[t] → CHORUS push → NWR master
```

**Principle:** do not guess when smile / “hi” / angry happened.  
**Contract:** a calibration script makes those moments known by design; then
digest frames at full accuracy.

---

## 0b. Avatar Calibration Script (~8s) — in design

Use a **scripted sample** so collection has ground-truth order.  
Lab path: generate an 8s avatar (e.g. Gemini) that performs the beats.  
Production path: **real user** (or their video) performs the **same script** —
Gemini is not identity albedo.

| Time | Beat ID | Actor does | Teaches / labels |
| --- | --- | --- | --- |
| 0.0–1.0s | `REST` | Neutral, mouth closed | rest reference, identity |
| 1.0–2.0s | `SMILE` | Closed-lip smile | smile look + face motion |
| 2.0–3.0s | `OPEN` | Jaw “ah”, teeth visible | open look + jaw cells |
| 3.0–4.0s | `SAY_HI` | Clearly say **“hi”** | speech clock + visemes |
| 4.0–5.0s | `SURPRISE` | Brows up, eyes wide | upper-face motion + plate |
| 5.0–6.0s | `ANGRY` | Frown / angry face | emotion + brow/mouth cells |
| 6.0–7.5s | `TALK` | One short sentence | continuous speech timeline |
| 7.5–8.0s | `REST` | Back to neutral | return-to-rest |

**Talk line (TALK beat):**  
> “Hello there. How are you today?”

Artifact beside the world:

```text
calibration_script.json
  version, duration_s=8, tick_rate=60
  beats: [{id, t0, t1, label, speech?}, …]
```

Digest uses `beats` as **time truth**: e.g. smile ∈ [1.0, 2.0).  
ASR may refine words inside `SAY_HI` / `TALK`; beat windows stay authoritative
for look/emotion sections.

Extends today’s capture kit ([`AvatarCapture.md`](AvatarCapture.md)) with
explicit `SAY_HI` + `ANGRY` and a fixed 8s lab protocol.

---

## 1. What video can and cannot give

| From RGB (+ audio) video | Not in the pixels |
| --- | --- |
| Dense **appearance** each frame | Master Lock / authority channels |
| Dense **2D (or 3D) deformation** of the face surface | Invented identity albedo |
| **When** mouth/brows move | True depth without stereo/prior |
| Audio → **what was said** (ASR / viseme align) | Full 32-ch semantics by magic |

So we do **not** “read 32 floats out of the MP4.”  
We **collect dense motion (+ labels)** and **map** them into the NWR channels that mean kinematics (and keep material/lock from digest).

**Target product of Side B:** a measured **Face Cell Timeline** that can reconstruct whole-face field values at **60 Hz** (the ~241 MB/s truth budget for this avatar’s face box).

**Transport lock (with Side A):** ROI = **full face**; after first keyframe, push **deltas only**.

---

## 2. Collection pipeline (the real system)

```text
Scripted avatar video (RGB + audio) + calibration_script.json
        │
        ▼
[0] Beat clock           map each frame to REST/SMILE/OPEN/SAY_HI/…
        │
        ▼
[1] Frame clock          every frame (native fps); resample / interpolate → 60 Hz
        │
        ▼
[2] Same registration    face box + UV as .bds / source_face (one coordinate system)
        │
        ▼
[3] Dense face track     per-frame dense correspondence (NOT 478 points alone)
        │
        ▼
[4] Rasterize to grid    displacement / velocity into 256² face cells
        │
        ▼
[5] Channel pack         write kinematics (ch 0/1 + …); lock/material stay from digest
        │
        ▼
[6] Speech clock         audio → words/visemes; constrained by SAY_HI/TALK beats
        │
        ▼
FaceCellTimeline  (+ LOOK plates still from keyframes)
        │
        ▼
Side A: compact / CHORUS / apply @ 60 Hz
```

---

## 3. Step detail — how we collect “all of it”

### [1] Time — enough samples

- Read **every** decoded frame (e.g. 30 fps), not 12 fps landmarks.
- Build a **60 Hz** timeline: hold or interpolate between frames so each NWR tick has a sample.
- Store `t_tick`, `source_frame_index`, `time_seconds`.

Without this, we never know smile/speech **at 16.7 ms**.

### [2] Space — same grid as NWR

- Every frame cropped/warped into the **same face box** as digest (`avatar_profile` / seed).
- Cell `(x,y)` in the timeline = cell `(x,y)` in `.bds`.
- Whole-face ROI ≈ face box (~31k cells here), not mouth-only.

### [3] Dense track — the missing engine

Landmarks alone cannot fill the face. Side B needs **one dense tracker** (design choice of method, not optional sparsity):

| Method | What it collects | Fit |
| --- | --- | --- |
| **Dense UV↔image flow** (e.g. FlowFace-style) | Per-UV / per-pixel face correspondence each frame | Strong for monocular video |
| **Non-rigid template track** | Per-vertex deformation of a face mesh/template | Strong if template = our UV |
| **3DMM + per-vertex residual** (FLAME-class) | Expression + dense residual | Good prior; residual must stay measured |
| Sparse 478 landmarks | Curves only | **Rejected as sole source** |

Output of [3] each frame: dense **position or displacement field** on the face UV — brows, cheeks, lips together.

### [4] Rasterize → cells

- Sample / splat the dense field into **256×256** (face cells only).
- \(\Delta\) from rest frame → displacement; \(\Delta\) from previous tick → **velocity**.
- That is the per-cell motion truth for that millisecond.

### [5] Map into NWR channels (honest)

| Collect from video | Write into field |
| --- | --- |
| vx, vy (and optional higher-order) | kinematics (e.g. ch 0/1, …) |
| Confidence / occlusion | intent or mask side-car |
| Rest identity, tissue, lock | **from digest** — not re-estimated as RGB |

Do **not** bake generative face RGB into cells. LOOK plates remain plates.

### [6] “Where are they smiling / saying what”

Parallel tracks on the **same clock**:

| Track | Source | Answers |
| --- | --- | --- |
| `FaceCellTimeline` | dense track [3–5] | **Where** on the face motion is |
| LOOK amounts / plate role | plate similarity or catalog drives | **Which look** (smile/open/…) |
| `SpeechAlign` | ASR + force-align / lipsync | **What word/viseme when** |
| Emotion labels | script / classifier (optional) | mood tags |

Then for any tick:  
`cells(t) + look(t) + viseme(t)` = smile/speech **with place and time**.

---

## 4. How we **prepare** that much data every tick

Preparation is **offline at digest / train time** (not improvised live).  
Live Side A only **reads the next ready package** and pushes to the NWR master.

### 4.1 Goal per tick package

For each `tick = 0 … T-1` at 60 Hz (~16.7 ms):

```text
TickPackage[t]  ≈  whole-face cell truth for that instant
  face_patch[H,W,C] or sparse list of face cells
  C = active channels (phase-1: vx, vy; later more)
  + look_drive (smile/open/surprise amounts)
  + speech (viseme / word)
  + confidence mask
```

Whole-face × full 32 ch ≈ **~3.3–4 MB per tick** (~241–251 MB/s).  
We **prepare** at that quality; we may **store/push** compressed equivalents (delta / f16 / \(c_t\)) as long as expand == this truth.

### 4.2 Prepare recipe (batch over the video)

```text
PREPARE (once per upload / retrain)
────────────────────────────────────
0. Inputs
   video, audio, existing .bds, face_box, source_face (rest), plates

1. Build rest reference
   rest displacement = 0 on face grid (identity pose from digest)

2. For each source frame f = 0 … F-1  (native video fps, every frame)
   a. Register frame → face_box UV (same as .bds)
   b. Dense track vs rest (and vs prev) → dense displacement field D_f
   c. Rasterize D_f → grid G_f[256,256,2+] for face cells only
   d. Record source time t_f

3. Resample to master clock (60 Hz)
   for tick t in 0 … floor(duration*60)-1:
       t_sec = t / 60
       G_t = interpolate(G_f, G_{f+1}) at t_sec     # linear or cubic
       V_t = (G_t - G_{t-1}) / dt                    # velocity for ch 0/1
       Look_t / Viseme_t from aligned tracks
       write TickPackage[t]

4. Speech / look alignment (same ticks)
   force-align words → viseme_t
   plate drives from catalog peaks + similarity or width/open curves

5. Pack for disk
   preferred: delta from rest, face ROI only, float16, chunked by second
   keep ability to reconstruct full TickPackage[t] for QA

6. Emit index
   meta: tick_rate=60, n_ticks, face_box, channel_mask, video hash
────────────────────────────────────
LIVE: producer loads TickPackage[t] (or c_t) → CHORUS push → NWR
```

### 4.3 What “ready every tick” means

| Stage | Work | Latency |
| --- | --- | --- |
| **Prepare** | Dense track all frames → 60 Hz packages | Minutes (offline OK) |
| **Play** | `package = timeline[tick]` | &lt; 1 ms read + push |

We do **not** run full dense tracking inside the 16.7 ms realtime budget.  
We **precompute** so every tick already has its data.

### 4.4 Per-tick contents (minimum accurate set)

| Field | Purpose |
| --- | --- |
| `tick`, `time_seconds` | Master clock index |
| `face_vx`, `face_vy` | Whole-face cell velocity (or displacement) |
| `conf` | Track confidence / occluded cells |
| `viseme` / `word` | What they are saying |
| `smile_amt`, `open_amt`, … | LOOK drives (plates stay resident) |

Optional later: more of the 32 channels when we have measured meaning for them.

### 4.5 Artifact layout

```text
face_cell_timeline/
  meta.json              tick_rate=60, face_box, channels, video
  rest_ref.npz           rest face patch / zeros
  ticks_XXXX.npz         batched packages (e.g. 60 ticks = 1 second)
  speech_align.json      tick → viseme / word
  look_drive.json        tick → smile/open/surprise
  qa_report.json         endpoint error / coverage
```

Raw prepare bandwidth ≈ **241–251 MB/s** of truth; on disk typically much less (delta + f16 + ROI).

---

## 4b. Artifact summary (design)

Same as §4.5 — chunked tick packages + speech/look side tracks.

---

## 5. Relation to what we already have

| Existing | Role after Side B |
| --- | --- |
| `smile.png` / atlas | LOOK evidence (keep) |
| `expression_catalog` | keyframe anchors (keep) |
| group `cell_transition_track` | weak prior / fallback — **not** the main timeline |
| `.bds` | rest + lock + material (keep) |

Side B **replaces** “guess from 8 controls” with **measured dense face ticks**.

---

## 6. Open choice (method only)

Which dense engine for [3]?

1. Dense UV–image flow → rasterize to our grid  
2. Mesh/template non-rigid track → rasterize  
3. 3DMM + dense residual → rasterize  

All three can feed the same `FaceCellTimeline` contract.  
**Rejected:** landmarks-only collection.

---

## 7. Success test (design)

Replay timeline into NWR (no ML invent):

- Mouth + brows move with the source video timing  
- Per-viseme landmark/SSIM or flow endpoint error within budget  
- Smile moment lights the **same face region** as in the take  
- Beat windows match: smile motion peaks inside `SMILE`, “hi” inside `SAY_HI`, etc.

If that fails, collection is still wrong — do not paper over with Side A.

---

## 8. Open choices (Side B)

1. Dense engine: UV-flow vs mesh vs 3DMM+residual  
2. Lab sample source: Gemini vs recorded actor vs user self-take first  
3. Minimum channels in TickPackage phase-1: velocity only vs + more kinematics
