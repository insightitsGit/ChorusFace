# TickPackage handshake — full-face ROI

**Status:** contract + local runtime **implemented**; CHORUS lane A+B push
**implemented** for lab (framed inline / TPK_REF); multi-host HELLO_ACK still
operator-owned (see TickFeedDesign §6.2 / §16).  
**Master:** [`TickFeedDesign.md`](TickFeedDesign.md).  
**Pair with:** [`CellFeedBandwidth.md`](CellFeedBandwidth.md) ·
[`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) ·
[`MultiLayerTickML.md`](MultiLayerTickML.md)

This is the **exact field list** for one tick: bytes, dtypes, keyframe vs delta.

---

## 1. Handshake (producer ↔ NWR master)

```text
PRODUCER                              NWR MASTER (60 Hz)
   │                                       │
   │  1) HELLO                             │
   │     world_id, face_box, channel_mask, │
   │     tick_rate=60, codec=delta_v1      │
   │──────────────────────────────────────►│
   │  2) HELLO_ACK                         │
   │     ok, master_tick_rate, apply_mode  │
   │◄──────────────────────────────────────│
   │  3) KEYFRAME  (TickPackage kind=KEY)  │
   │     full-face values @ t0             │
   │──────────────────────────────────────►│  store as state S
   │  4) KEY_ACK (optional tiny)           │
   │◄──────────────────────────────────────│
   │  5) DELTA × N  (kind=DELTA)           │
   │     Δ each ~16.7 ms                   │
   │──────────────────────────────────────►│  S ← S + Δ
   │  (optional KEY every 1–2 s)           │
```

**Rules**

- ROI = **full face** (`face_box` only — not mouth-only, not full 256² unless box is).  
- After KEY, only **DELTA** (plus rare KEY refresh).  
- Master does **not** send face payloads back.  
- LOOK plates / identity photo stay **resident** (not in TickPackage body).

---

## 2. Full-face ROI geometry (what we have vs don’t)

### Have (defined)

| Field | Value (this avatar example) | Notes |
| --- | --- | --- |
| Grid | 256 × 256 | NWR world |
| `face_box` | x,y,w,h e.g. **49, 20, 158, 199** | From profile / seed |
| Face cell count | **w × h = 31,442** | Dense patch = w×h, row-major |
| Cell address | `(bx + i, by + j)` or linear `i + j*w` in patch | Patch-local |
| Tick rate | **60** | `dt = 1/60` s |
| Phase-1 channels | **ch 0 = vx, ch 1 = vy** | Kinematics write set |
| Rest reference | zeros (or digest rest) | Δ vs rest optional |

### Produced vs remaining

| Item | State |
| --- | --- |
| Actual `vx/vy` arrays per tick from video | **Yes** (Farneback → 60 Hz; per-tick `source` provenance) |
| Packed KEY/DELTA bytes (codec) | **Yes** |
| KEY/Δ on remote CHORUS | **Partial** — framed lane B / TPK_REF |
| Confidence map per tick | **Yes** |
| Beat/word/look/emotion @ 60 Hz | **Yes** (script + energy force-align) |
| Master apply full-face patch | **Yes** (`tick_ingest.comp`) |

---

## 3. TickPackage binary layout (`tick_package.v1`)

All multi-byte integers **little-endian**. Floats IEEE.

### 3.1 Header (fixed, every package) — **64 bytes**

| Offset | Type | Name | Meaning |
| --- | --- | --- | --- |
| 0 | `u32` | `magic` | `'TPK1'` = `0x31504B54` |
| 4 | `u16` | `version` | `1` |
| 6 | `u16` | `kind` | `1=KEYFRAME`, `2=DELTA`, `3=HELLO` |
| 8 | `u32` | `tick` | Master tick index |
| 12 | `f32` | `time_seconds` | `tick / 60` |
| 16 | `u16` | `face_x` | Box origin x (grid) |
| 18 | `u16` | `face_y` | Box origin y |
| 20 | `u16` | `face_w` | Box width |
| 22 | `u16` | `face_h` | Box height |
| 24 | `u32` | `channel_mask` | Bit i set → channel i present (phase-1: bits 0+1 = `0x3`) |
| 28 | `u8` | `value_dtype` | `1=f32`, `2=f16` |
| 29 | `u8` | `delta_encoding` | see §4 |
| 30 | `u8` | `flags` | bit0=has_labels, bit1=has_conf; **bit2 `FLAG_VS_REST` reserved / unused in v1** |
| 31 | `u8` | `reserved0` | 0 |
| 32 | `u32` | `payload_bytes` | Size of body after header |
| 36 | `u32` | `crc32` | CRC of header[0..35] + body (optional 0 in v1 lab) |
| 40 | `u64` | `world_hash` | Identity of world / timeline |
| 48 | `u8[16]` | `reserved1` | 0 |

**Header size = 64 bytes** every message.

**Phase-1 channel count:** `channel_mask` must include `0x3` (vx, vy). v1
decoders **may hardcode `C = 2`** (`PHASE1_CHANNELS`) and reject packages whose
mask does not include those bits. Wider masks are reserved for later phases.

### 3.2 Labels block (optional, if `flags.has_labels`) — **48 bytes**

| Offset | Type | Name |
| --- | --- | --- |
| +0 | `u8` | `beat_id` enum: 0=REST … 7=TALK (see calibration script) |
| +1 | `u8` | `emotion_id` enum: 0=NEUTRAL,1=HAPPY,2=SURPRISED,3=ANGRY,… |
| +2 | `u8` | `viseme_id` | index into canonical viseme table |
| +3 | `u8` | `label_conf` | 0–255 |
| +4 | `f32` | `smile_amt` | 0..1 LOOK drive |
| +8 | `f32` | `open_amt` | 0..1 |
| +12 | `f32` | `surprise_amt` | 0..1 |
| +16 | `u8[16]` | `word_utf8` | truncated word / empty |
| +32 | `u8[16]` | `reserved` |

Labels are **small**; they ride with every KEY/DELTA when known.

### 3.3 Face value body (required)

`N = face_w * face_h`  
`C = popcount(channel_mask)`  (phase-1: **C = 2**)  
`E = 4` if f32, `2` if f16  

#### KEYFRAME body (`delta_encoding` ignored / 0)

```text
values[N * C]   row-major patch, interleaved channels (vx,vy,vx,vy,…)
conf[N]         optional u8  (if flags.has_conf) — 0..255
```

| Phase-1 KEY size (no conf) | Bytes |
| --- | --- |
| f32 | 64 + 48 + **251,536** ≈ **251.6 KiB** (~0.25 MB) |
| f16 | 64 + 48 + **125,768** ≈ **125.9 KiB** |

#### DELTA body — encoding modes (`delta_encoding`)

| Code | Name | Body layout | When |
| --- | --- | --- | --- |
| `1` | `DENSE_DELTA` | `delta[N*C]` same layout as KEY | Many cells change |
| `2` | `SPARSE_DELTA` | see below | Typical speech |
| `3` | `EMPTY` | no value body (labels only) | No change above ε |

**`SPARSE_DELTA` (code 2):**

```text
u32  count                 # number of changed cells
u16  idx[count]            # linear index in face patch 0..N-1
f16  or f32  delta[count * C]   # same dtype as value_dtype
u8   conf[count]           # optional if has_conf
```

Master: for each `idx[k]`, `S[idx] += delta[k]`.

| Example DELTA (phase-1, f16, ~10% cells) | Bytes |
| --- | --- |
| count + idx + Δ | ~18.4 KiB / tick → **~1.1 MB/s** @ 60 Hz |
| ~5% active | ~9.2 KiB / tick → **~0.55 MB/s** |

ε (design default): cell omitted if `|Δvx| < 1e-4` and `|Δvy| < 1e-4` (grid units).

---

## 4. Delta encoding rules (handshake semantics)

```text
KEYFRAME:
  S_face := values
  last_tick := tick

DELTA:
  require tick == last_tick + 1  (or allow gaps with KEY refresh)
  S_face[i] += delta[i]   // per present channel
  last_tick := tick

KEY refresh (optional every 60–120 ticks):
  S_face := values   // kill drift
```

- Δ is **additive in channel space** (velocity or displacement — pick one; phase-1 = **velocity**).  
- If `flags.vs_rest`: KEY stores values vs rest; DELTA still consecutive.  
- Channel mask must match HELLO; changing mask requires new HELLO + KEY.

---

## 5. HELLO payload (kind=3) — contract negotiate

| Field | Type | |
| --- | --- | --- |
| `face_x,y,w,h` | u16×4 | ROI |
| `channel_mask` | u32 | e.g. `0x3` |
| `value_dtype` | u8 | prefer f16 on wire |
| `delta_encoding_caps` | u8 bitmask | bit1=DENSE, bit2=SPARSE, bit3=EMPTY |
| `tick_rate` | u16 | 60 |
| `labels` | u8 | 1 if labels always present |
| `world_id` | utf8 / hash | |

ACK returns: `apply_mode` (`velocity_write` | `displacement_write`), max payload, ok/fail.

---

## 6. What rides outside TickPackage (resident)

| Asset | In TickPackage? |
| --- | --- |
| `source_face.png` / Master Lock / `.bds` material | **No** — resident |
| `smile.png` / `open.png` / atlas | **No** — resident; only **amts** in labels |
| Full 32-ch rewrite every tick | **No** in phase-1 |

---

## 7. Handshake checklist — have / don’t

| Item | Contract | Local runtime | Remote CHORUS |
| --- | --- | --- | --- |
| Full-face ROI box | **Yes** | Yes (profile) | n/a |
| Header + kind KEY/DELTA/HELLO | **Yes** | Yes | framed / TPK_REF |
| Phase-1 vx,vy mask `0x3` | **Yes** | Yes | via package / `c_t` |
| f16/f32 + sparse/dense/empty | **Yes** | Yes | via package body |
| Labels 48 B | **Yes** | Yes | in package |
| CRC = header[0..35] + body | **Yes** | Yes | n/a |
| Side B fills values | Spec | Yes + provenance | n/a |
| CHORUS lane A (`c_t`) | Spec | Yes | Yes when plane up |
| CHORUS lane B (TickPackage bytes) | Spec | Yes (`push_package_bytes`) | Yes when plane up (inline / TPK_REF) |
| Master S←KEY / S←S+Δ apply | Spec | Yes (GPU) | consumer host |
| HELLO remote ACK | Spec | Lab self-ACK + lane B push | multi-host ACK TBD |

---

## 8. Phase-1 locked defaults

```text
ROI            = full face_box
channels       = vx, vy  (mask 0x3)
meaning        = VELOCITY  (locked for phase-1; not displacement)
                 KEY/DELTA values are velocity in grid units / second
                 (same family as NWR ch 0/1). Displacement = integrate later if needed.
wire dtype     = f16
KEY            = dense once (~126 KiB f16)
DELTA          = SPARSE_DELTA (fallback DENSE if count > 0.35 * N)
EMPTY          = if count == 0
labels         = on
conf           = on when Side B provides it
refresh KEY    = every 120 ticks (2 s) optional
```

**Locked for now:** velocity. Revisit displacement-only encoding only if build QA says so.

**Implementation branch:** `tickfeedmaster`  
Python codec: `aiface.tickfeed` · GPU ingest shader: `shaders/tick_ingest.comp`  
Bridges B1–B4 adopted in [`TickFeedDesign.md`](TickFeedDesign.md) §6.5.
