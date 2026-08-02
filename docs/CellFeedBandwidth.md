# Cell feed bandwidth — design (480 MB/s target)

**Status:** implemented on `tickfeedmaster` (CHORUS push + c_t + KEY/Δ f16).  
**Session:** AminIntheLoop — realtime full-cell control.

**Master:** [`TickFeedDesign.md`](TickFeedDesign.md).  
Related: [`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) ·
[`TickPackageHandshake.md`](TickPackageHandshake.md) · [`NWRDataDesign.md`](NWRDataDesign.md)

---

## 1. Exact target (why we design)

| Quantity | Value |
| --- | --- |
| Tick rate | **60 Hz** (`TICK_RATE_HZ`) |
| Tick period | **≈ 16.67 ms** |
| Grid | **256 × 256** cells |
| Channels / cell | **32** float32 |
| One cell / second (all 32) | **1,920** floats (**7.5 KB/s**) |
| Full field / tick | **≈ 8 MB** |
| Full field / second | **≈ 480 MB/s** (~3.8 Gbit/s) |

We now know **exactly how much information NWR can accept per millisecond** for full cell character. The product problem is: supply that *quality* without naively shipping 480 MB/s on every path.

---

## 2. Decided transport: CHORUS Fabric (push, not chat)

**Decision:** stream cell/drive vectors over **CHORUS Fabric** (`chorus-fabric`).

### Master clock + one-way feed (better than send/receive)

NWR (field runtime) is the **master of time**: ticks every **≈ 16.67 ms** (60 Hz).

We do **not** design a request/response every second (or every tick) that both
sends **and** receives ~241–251 MB. That doubles work and adds RTT latency.

```text
BETTER:
  Producer (timeline / AI)  --PUSH only-->  NWR master
       one float32 frame per tick (or compact c_t)
       no per-tick reply payload required

AVOID:
  every 1 s (or every 16 ms): send 251 MB  AND  receive 251 MB
```

| Role | Who | Job |
| --- | --- | --- |
| **Master** | NWR @ 60 Hz | Owns tick; applies latest push into field |
| **Producer** | Side B timeline and/or AI | Pushes next face package before/at tick |
| **Ack** | optional tiny watermark/ack | Integrity only — not a full field echo |

So the realtime question is: **can we push enough accurate data each 16.7 ms?**  
Not: can we round-trip 251 MB/s both ways.

| Property | Role in this design |
| --- | --- |
| Wire format | Raw **float32** frames over gRPC (CHORUS) |
| Direction | **Producer → NWR push** (master does not re-send field each tick) |
| vs JSON/REST | ~4–5× less waste; ~raw binary speed |
| Cipher | \(V_{\mathrm{enc}} = V @ K\) (size-preserving; secure, not a compressor) |
| What it solves | Low-overhead vector push into the master |
| What it does **not** solve | Still need Side B accuracy + content compression for ~251 MB/s truth |

```text
Side B collect (offline or live) → Producer
        --CHORUS PUSH float32-->  NWR master @ 16.7 ms
                                      apply into .bds field
```

Package (design intent): `chorus-fabric` on PyPI / InsightIts chorus_fabric.

---

## 3. Focus problem: making 480 MB/s tractable

CHORUS carries vectors fast. **Content design** must reduce what we put on the wire while preserving per-cell accuracy at 16.7 ms.

### Stack (agreed direction)

```text
Layer 0  Identity resident     .bds + Master Lock stay on GPU (not re-streamed)
Layer 1  Compact cell code     AI emits small float vector c_t each tick
Layer 2  Expand on NWR         GPU/CPU expands c_t → per-cell channels
Layer 3  Delta + ROI           only unlocked / changed cells & channels
Layer 4  Optional quantize     float16 on wire for drives when quality allows
Layer 5  CHORUS Fabric         binary float32 (or f16) stream AI ↔ NWR
```

### Bandwidth ladders (design targets, not yet measured)

| Mode | What is sent each tick | Rough rate |
| --- | --- | --- |
| **Raw full field** | 65,536 × 32 floats | **~480 MB/s** (ceiling / truth budget) |
| **Whole face box** (this avatar) | ~158×199 ≈ **31,442** cells × 32 | **~241 MB/s** (~**200 MB/s** if tighter face mask) |
| Whole face × velocity only | 31,442 × 2 | **~15 MB/s** |
| Velocity-only full grid | 65,536 × 2 | **~30 MB/s** |
| Mouth ROI full-32 | ~4,408 × 32 | **~32 MB/s** |
| Mouth ROI velocity | ~4,408 × 2 | **~2 MB/s** |
| Face/mouth Δ + sparse | changed cells only | **≪ peak** typical speech |
| Compact code \(c_t\) | e.g. 64–512 floats | **~15–120 KB/s** @ 60 Hz |
| Today’s ±4 budget | ≤256 commands | sparse; **not** full control |

**Quality rule:** whatever we send must be expandable (or directly writable) so that at 60 Hz each relevant cell has the correct channels — equivalent in effect to the 480 MB/s truth budget where motion matters.

---

## 4. One-cell contract (building block)

For **1 cell**, full character for 1 second = **1,920** floats.

Feed packet design (conceptual):

```text
CellTick {
  x, y            // address
  mask            // which of 32 channels are present
  values[]        // len = popcount(mask)
  t_tick          // 60 Hz index
}
```

CHORUS frame = batch of `CellTick` or one dense tensor `[N_cells, C_active]` + index list.

---

## 5. Region packages (practical full control)

Prefer **one mouth (or face-ROI) tensor per tick** over 256 random ±4 ops:

```text
MouthFeedTick @ 60 Hz
  cell_ids[N] or bitmask
  channels[N, C]     // C ≤ 32, often 2 (vx, vy) first
  authority / source
```

Ship that tensor as a CHORUS float32 frame. NWR applies under Master Lock.

---

## 6. Compact code path (make 480 MB/s “easy”)

When ROI tensors are still too big for remote AI:

```text
c_t  (small vector from AI / ML)
  → CHORUS Fabric
  → NWR expander (basis / codebook / group→cell learned from measured truth)
  → field write equivalent to dense per-cell update
```

Expander training data must come from **measured** cell timelines (design backlog: dense mouth deltas), not invented RGB.

---

## 7. What stays out of the 480 MB/s pipe

| Asset | Plane | Not streamed every tick |
| --- | --- | --- |
| `source_face.png` | identity | resident |
| `smile.png` / `open.png` / atlas | LOOK plates | resident; only **amounts** in drive |
| Master Lock / material | field | resident unless digest changes |

Plates are **look evidence**. The 480 MB/s problem is **field cell character over time**, not re-sending PNGs.

---

## 8. Current vs target

| | Current | Target design |
| --- | --- | --- |
| Pipe | PaintCommand SSBO ≤256/tick | CHORUS Fabric + region/code tensors |
| Control | sparse ±4 | full unlocked-cell accuracy @ 60 Hz |
| Ceiling | n/a | **480 MB/s** full-field equivalent |
| Working rate | ≪ 1 MB/s effective | ROI/Δ/code toward **KB–few MB/s** |

---

## 9. Options → chosen techniques (quality + realtime)

### Option scorecard

| Technique | Quality | Realtime | Cuts 480 MB/s | Notes |
| --- | --- | --- | --- | --- |
| A. Raw full field every tick | ★★★★★ | ★☆☆☆☆ | no | Truth ceiling only; too heavy for AI API path |
| B. Skip locked cells (ROI) | ★★★★★ | ★★★★☆ | ~10–15× | Face mostly Master-Locked; mouth is the live set |
| C. Channel mask (not always 32) | ★★★★☆ | ★★★★★ | up to 16× | Speech first needs vx,vy; add channels when needed |
| D. Delta from rest / prev tick | ★★★★★ | ★★★★★ | large when calm | Same quality; sparse when idle |
| E. Compact code \(c_t\) + expand | ★★★★☆ | ★★★★★ | 100–1000× | Needs measured training so expand ≠ fake |
| F. float16 on wire | ★★★★☆ | ★★★★★ | 2× | Fine for motion drives; keep f32 for authority |
| G. Predict + residual | ★★★★☆ | ★★★★☆ | medium | Extra complexity; phase-2 |
| H. CHORUS Fabric transport | ★★★★★ | ★★★★★ | vs JSON only | Chosen wire; not a content compressor |
| I. Today ±4 budget 256 | ★★☆☆☆ | ★★★★★ | yes but | **Loses** full-cell quality (demo pain) |

### Chosen stack (best for us)

**Decision: hybrid B + C + D + E + F + H.** Do **not** ship raw A as the product path; keep A as the **accuracy budget** we must match in effect.

**Delta-first:** Side B prepares per-tick cell **values**; transport prefers
**Δ(t) = values(t) − values(t−1)** (or vs rest). Full snapshots stay available
for QA / keyframes. Deltas cut wire size when the face changes little between
16.7 ms ticks — same quality at apply time after integrate/add on the master.

### First load + delta ticks (new design rates)

Numbers for this avatar **face box ≈ 31,442 cells**. MB = 10⁶ bytes.

| Phase | Payload | Size / rate |
| --- | --- | --- |
| **First load** (keyframe) | face × 32 ch snapshot | **~3.8 MB once** |
| **First load** (phase-1 vel) | face × vx,vy | **~0.25 MB once** |
| **Every tick if full resend** | face × 32 @ 60 Hz | **~241 MB/s** (old pain) |
| **Every tick if full vel** | face × 2 @ 60 Hz | **~15 MB/s** |
| **Delta tick** (worst: all cells change) | ≈ same as full | up to ~15 MB/s (vel) or ~241 (32 ch) |
| **Delta tick** (~10% face active, vel, dense) | sparse change | **~1.5 MB/s** |
| **Delta tick** (~10% active, vel, f16) | | **~0.7 MB/s** |
| **Delta tick** (~2% active, vel) | calm / micro | **~0.3 MB/s** |
| **+ compact `c_t`** (64–256 floats @ 60) | codec path | **~15–60 KB/s** |

```text
t = 0:     PUSH full keyframe     (~0.25–3.8 MB once)
t = 1…N:   PUSH Δ only            (typically ≪ full; peaks on smile/speech onsets)
periodic:  optional keyframe      (e.g. every 1–2 s) to limit drift
```

**Steady-state design target (phase-1, whole face, velocity, deltas):**  
about **0.3–2 MB/s** typical speech, not 241 MB/s.  
**Quality ceiling** remains the full face tick truth; deltas reconstruct it on the master.

```text
CHOSEN FEED PATH (realtime, quality-preserving)

1. Resident identity     .bds + lock + LOOK plates stay local (never in 480 pipe)
2. ROI (B)               unlocked mouth (and active face regions only)
3. Channel mask (C)      phase-1: velocity ch0/1; phase-2: more channels
4. Deltas (D)            Δ vs rest or previous tick
5. Compact code (E)      AI emits c_t; NWR expands → per-cell (learned from measured)
6. float16 wire (F)      CHORUS frames for drives; expand to f32 in field
7. CHORUS Fabric (H)     AI API ↔ NWR binary vector stream @ 60 Hz
```

**Locked scope:** ROI = **full face** (not mouth-only). Transport after keyframe = **deltas only**.

**Working bandwidth target (speech):**  
full-face × velocity × deltas × f16 → typically **~0.3–2 MB/s**, often **tens–hundreds KB/s** with \(c_t\).  
**Quality target:** expandable result matches full-face cell truth each **16.7 ms** (equivalent to the ~241 MB/s face ceiling).

### Rejected as primary path

| Reject | Why |
| --- | --- |
| Raw 480 MB/s always | Breaks AI API / network realtime |
| Sparse ±4 only (I) | Realtime but **not** full cell quality |
| CHORUS alone | Transport only; still need B–F |

### Phase plan (design)

| Phase | Ship | Approx rate |
| --- | --- | --- |
| **P1** | CHORUS + mouth ROI + velocity + delta + f16 | ~0.5–2 MB/s peak |
| **P2** | Compact \(c_t\) expander trained on measured mouth cell timelines | ~15–120 KB/s |
| **P3** | More channels / predict+residual if still needed | lower residual |

### Still open (implementation detail, not technique choice)

1. Expander on **AIFace CPU** vs **NWR compute** (or both)?  
2. Plate **amounts** in same CHORUS frame as cell tensor, or side channel?  

**No implementation until you green-light build.**
