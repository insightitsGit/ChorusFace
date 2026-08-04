# NWR data design — what feeds the field

Master design: [`AMIN_DESIGN.md`](AMIN_DESIGN.md) · Index: [`README.md`](README.md)

This is the **full dataset contract** for ChorusFace on NWR: what we extract from a
user upload, what we store beside the world, and what the GPU actually
consumes each tick.

**Rule:** one dense field + compact side-cars. Never per-frame full grids.
Never generative face RGB.

---

## 1. Product contract

```text
Upload video/stills
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  WORLD DIR  (adoptable avatar)                            │
│                                                           │
│  L1 Identity     .bds + source_face + Master Lock         │
│  L2 Regions      region_catalog + tissue/parts maps       │
│  L3 Looks        open/smile/surprise + plate atlas        │
│  L3b Observe     avatar_observations (smile/open vectors) │
│  L4 Maps         condition_maps + gpu_display_recipe      │
│  L5 Trajectories live_vector_* + cell_transition_track_*  │
│  L6 Models       live_vector_model + behavior_model       │
│  L0 Contract     avatar_profile + store/report manifests  │
└───────────────────────────────────────────────────────────┘
        │
        ▼
Runtime tick → PaintCommand ±4 → constraint.comp → avatar.frag (L00–L11)
```

NWR’s native “dataset” for a face is **not** a second neural renderer. It is:

1. **Field truth** — one `256×256×32` `.bds` (identity + unlock mask)  
2. **Look bank** — real capture plates  
3. **Observations** — measured smile/open GPU vectors + rest→look deltas  
4. **Drive tables** — viseme/emotion → jaw/open/width  
5. **Trajectories** — timed controls from the take  
6. **Models** — ML cover (unknown sounds) + ML fill **between** observations  
7. **Ephemeral command stream** — ±4 velocity discs each tick (not persisted yet)

---

## 2. Size budget (current `output/worlds/avatar/`)

| Layer | Artifacts | ~Size |
| --- | --- | --- |
| Identity | `avatar_face.bds` + `source_face.png` | **~9.2 MB** |
| Tissue/parts | `face_tissue.npy`, `face_parts.npy` | **~2 MB** |
| Looks | open/smile/surprise + atlas plates | **~4–6 MB** |
| Observations | `avatar_observations.json` / `.npz` | **~20 KB** |
| Regions/maps | region + condition + recipe + profile | **~30 KB** |
| Trajectories | live + behavior tracks/datasets | **~80 KB** |
| Models | live_vector + behavior joblibs | **~130 KB** |

Dense field math:

```text
256 × 256 × 32 × 4 bytes  ≈  8.4 MB   ← the NWR cell store
```

---

## 3. Layer inventory

### L1 — Identity (feeds NWR field)

| File | Schema | Producer | Consumer | Feeds |
| --- | --- | --- | --- | --- |
| `avatar_face.bds` | `bds-1.0` / 256×256×32 | `capture` → `seed` → `digest` | `FieldRuntime`, shaders | field, lock gate |
| `source_face.png` | pixels (≤1024²) | capture / seed | `avatar.frag` photo | look base |
| BDS `avatar_seed` meta | `avatar-seed-1.1` | seed + capture merge | face_box, adoption | registration |

**Channel groups in `.bds`:** kinematics 0–7 · material 8–15 · intent 16–23 · rules 24–31 (`human_lock` = 31).

### L2 — Regions & tissue (feeds cell address + warp gates)

| File | Schema | Producer | Consumer | Feeds |
| --- | --- | --- | --- | --- |
| `region_catalog.json` | `amin_loop.regions.v1` | `amin_loop.regions` | mouth centroid, profile | object address |
| `face_tissue.npy` | `face-tissue-1.0` HxWx4 | skinning | warp / slit / lids | shader gates |
| `face_parts.npy` | `face-parts-1.0` HxWx4 | parts atlas | part ids | shader / muscles |

Runtime (not a file): `CellClusterIndex` + `mouth_groups` rebuild full
`mouth_unlocked` membership from `.bds` each launch.

### L3 — Looks (feeds GPU plates)

| File | Schema | Producer | Consumer | Feeds |
| --- | --- | --- | --- | --- |
| `open.png` / `smile.png` / `surprise.png` | capture roles | capture | L04/L05/L08 | plates |
| `plate_atlas.json` + `plates/` | `plate-atlas-1.1` | plates | L07 atlas | plates |
| `expression_catalog.json` | `expression-catalog-1.0` | capture | brow/eye/expr | plates |

### L3b — Observations (the data ML fills **from**)

Gap fill needs measured avatar truth first. That package is
`avatar_observations.*` — see [`AvatarObservations.md`](AvatarObservations.md).

| File | Schema | Producer | Consumer | Role |
| --- | --- | --- | --- | --- |
| `avatar_observations.json` | `chorusface.avatar_observations.v1` | `observation.extract` | `BehaviorDriver` | look anchors + GPU contract |
| `avatar_observations.npz` | same | extract | train / QA | numeric pack |

Per look (`rest` / `smile` / `open` / `surprise` + talk samples):

| Field | Source | What it is |
| --- | --- | --- |
| `landmarks` | `expression_catalog` / talk_series | mouth_open, smile_width, teeth, brow |
| `gpu` / `gpu_uniforms` | role → shader contract | what GPU **reads** for that look |
| `plate` | PNG mouth-ROI vs rest | measured pixel delta (not invented RGB) |
| `controls` | landmarks → 8-group vector | same space as behavior track |
| `delta_from_rest` | smile/open − rest | **`smile_vector` / `open_vector`** |
| `cells` | `.bds` mouth groups | geometry counts (identity field) |

**Smile on GPU (not a second `.bds` state):**

```text
smile look → avatar_mouth_pose.w ≈ 1.0  → samples smile.png
             smile_vector = controls(smile) − controls(rest)
             (this avatar: width_n / corner_dx ≈ +0.65)
```

ML may only interpolate **between** these observations.

### L4 — Maps & recipe (feeds uniforms + tables)

| File | Schema | Producer | Consumer | Feeds |
| --- | --- | --- | --- | --- |
| `condition_maps.json` | `amin_loop.mapping.v1` | mapping | jaw table | speech clock |
| `gpu_display_recipe.json` | `chorusface.gpu_display_recipe.v3` | gpu_recipe | knobs / L-path | shader |

Global (code, not per-world): `display_layers` L00–L11, Master Lock policy, ±4 op.

### L5 — Trajectories (the NWR **time dataset**)

This is the missing piece people mean by “dataset for feeding NWR”: **timed
drives learned from the upload**, not a second face.

| File | Shape | Producer | Consumer | Role |
| --- | --- | --- | --- | --- |
| `live_vector_trajectory.json` | t → `[open,jaw,width]` | live_vector.extract | train/QA | speech cover truth |
| `live_vector_dataset.npz` | `X[N,8]→y[N,3]` | extract | fit live model | train set |
| `cell_transition_track.npz` | t, controls[N,8], deltas[N,8], features[N,8] | behavior.track | BehaviorDriver | **measured** group motion |
| `cell_transition_track.json` | human-readable frames | track | QA | inspect deltas |
| `behavior_dataset.npz` | `X→y` group controls | track save | fit behavior | train set |
| `capture_meta.json` `talk_series` | ≤200 landmark rows | capture | priors | weak time series |

**Group control vector (8):**

```text
openness_n, jaw_n, width_n,
upper_lip_dy, lower_lip_dy, corner_dx,
teeth_reveal, cavity_n
```

`deltas[i] = controls[i] - controls[i-1]` — the transformation between samples.

### L6 — Models (fill holes; never rewrite albedo)

| File | Version | Train target | Runtime |
| --- | --- | --- | --- |
| `live_vector_model.joblib` | `live-vector-1.0` | audio features → open/jaw/width | `LiveVectorDriver` |
| `behavior_model.joblib` | `behavior-1.0` | audio features → 8 group controls | `BehaviorDriver` |

Authority:

```text
observed smile/open vectors  →  measured track @ t  →  ML fill  →  viseme table
live-vector ML covers unknown sounds for plate/jaw cover
```

Retrain on new upload (replace L3b+L5+L6):

```powershell
python scripts/retrain_behavior.py --video NEW.mp4 --world-dir output/worlds/avatar
```

### L0 — Adoption contract

| File | Schema | Role |
| --- | --- | --- |
| `avatar_profile.json` | `chorusface.avatar_profile.v1` | Points at all layers; validation gates |
| `amin_data_store.json` | `amin_loop.store.v1` | On-disk inventory |
| `amin_loop_report.json` | `amin_loop.run_all_steps.v1` | Last train audit |

---

## 4. What NWR eats at runtime (command stream)

Not stored today (ephemeral):

```text
each 60 Hz tick
  MouthLayerTimeline  → plate amounts (L04–L07)
  BehaviorDriver      → group flow (observed smile/open / measured / ml_fill / table)
  MouthCellPlan       → cell→neighbor VelocityImpulse list
  LiveVectorDriver    → width/open cover
        │
        ▼
  PaintCommand ±4  (SSBO, budget ~256)
        │
        ▼
  constraint.comp  (Master Lock rejects locked cells)
        │
        ▼
  world ch0/1 velocity  →  avatar.frag field_displacement × field_warp_gain
```

Future (gap): persist session as NWR `.bdl` for replay — see §6.

---

## 5. Train → world → play

```text
amin_train / retrain_behavior
  digest (6 fps)     → L1 + L2 + L3 + capture_meta
  mapping            → L4 condition_maps
  gpu_recipe         → L4 recipe
  live_vectors       → L5 live_* + L6 live_vector_model
  behavior           → L3b observations + L5 transition_* + L6 behavior_model
  profile + store    → L0

play: open_avatar(world)
  → FieldRuntime(.bds) + recipe + plates
  → observations (smile/open vectors) + LiveVectorDriver + BehaviorDriver
  → CellClusterIndex → ±4 + L00–L11 composite
```

---

## 6. Realtime cell feed (480 MB/s design)

Full per-cell character @ 60 Hz is **≈ 8 MB/tick ≈ 480 MB/s**.  
**Current session design:** full-face ROI, keyframe + deltas, CHORUS push,
Side B collect from 8s calibration script — see master
[`TickFeedDesign.md`](TickFeedDesign.md).

Detail: [`CellFeedBandwidth.md`](CellFeedBandwidth.md) ·
[`TickPackageHandshake.md`](TickPackageHandshake.md) — design only, not implemented.

## 7. Gap backlog — denser NWR datasets (honest)

| Gap | Why | Ideal artifact |
| --- | --- | --- |
| Dense mouth-ROI cell deltas | Group landmarks ≠ cell truth | `mouth_cell_deltas.npz` (sparse x,y,Δv @ fps) |
| Full region bitmasks | Catalog samples ≤64 | `region_masks.npy` RLE |
| Session `.bdl` log | NWR replay / determinism | tick-stamped ±4 beside world |
| Phoneme-aligned spans | Better than energy holds | ASR force-align JSON |
| Landmark track @ video fps | 12 fps subsample loses ms | `landmarks.npz` 478×T |
| Multi-take registry | Retrain overwrites | `takes/<id>/` + active pointer |
| Playback score | Step 14 half-open | landmark/SSIM report |

**Still forbidden:** generative face RGB, invented optical flow sold as measured,
per-frame full `.bds` dumps, Path A ownership seals.

---

## 8. Related docs

| Doc | Piece |
| --- | --- |
| [`CellFeedBandwidth.md`](CellFeedBandwidth.md) | **480 MB/s target + CHORUS Fabric feed design** |
| [`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) | Video→cell collect + **8s calibration script** |
| [`MultiLayerTickML.md`](MultiLayerTickML.md) | Layered tick ML + abstract packets |
| [`AMIN_DATA_STORE.md`](AMIN_DATA_STORE.md) | Size philosophy |
| [`AvatarBehavior.md`](AvatarBehavior.md) | Measured + ML fill |
| [`AvatarAdoption.md`](AvatarAdoption.md) | Portable world dir |
| [`DisplayLayers.md`](DisplayLayers.md) | L00–L11 composite |
| [`MouthCellGroups.md`](MouthCellGroups.md) | L03 groups |
| [`BDSMotionMap.md`](BDSMotionMap.md) | 32 channels |
| [`LiveControlVectors.md`](LiveControlVectors.md) | Live-vector cover |
