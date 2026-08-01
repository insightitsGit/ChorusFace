# AminIntheLoop — steps cheat sheet

Full design: [`AMIN_DESIGN.md`](AMIN_DESIGN.md) · Doc index: [`README.md`](README.md)

| # | Step | Design idea | Implementation |
| --- | --- | --- | --- |
| 1 | What is NWR | Propose → validate → GPU field | `vendor/nwr`, `aiface.runtime` |
| 2 | 32 floats / cell | Kinematics / material / intent / rules | `amin_loop.cells`, `runtime.bds` |
| 3 | Control + neighbors | ±4 impulses; Moore clusters | `amin_loop.control` |
| 4 | Digest → objects | Video/photo → `.bds` + plates | `amin_loop.digest`, `aiface.capture` |
| 5 | x,y,(z),t | 2D grid; Z = channel; t = video/speech | `amin_loop.regions` |
| 6 | Looks + maps | Real plates; word/sound/emotion tables | `amin_loop.mapping`, plates |
| 7 | Props per region | mean[32] + lock + samples | `region_catalog.json` |
| 8 | GPU recipe | Same display path at play (L00–L11) | `gpu_recipe`, `display_layers`, `avatar.frag` |
| 9 | Live vectors | Video → controls → model | `aiface.live_vector` |
| 10 | Train + play | One pipeline | `scripts/amin_train.py`, `aiface` |

See also: [`DisplayLayers.md`](DisplayLayers.md) · [`MouthCellGroups.md`](MouthCellGroups.md) ·
[`AvatarAdoption.md`](AvatarAdoption.md) · [`AvatarBehavior.md`](AvatarBehavior.md)

## Realism track (steps 11–14 — why the mouth blurs today)

Field stays 256×256×32 (authority/physics). Display fidelity is a separate
axis — nothing in steps 1–10 fixes it, so these steps do.

| # | Step | Design idea | Status |
| --- | --- | --- | --- |
| 11 | Native-res display plane | Capture photo + plates at 1024², registered to the same face box; grid stays 256. Selected frames are re-cut from the source at display res (`capture.resample_frames_hires`); textures upload at native size, photo mipmapped, part ids split into their own grid-res NEAREST map | **Built** — `DISPLAY_SIZE=1024`, app logs `Avatar base texture: 1024x1024 display res` |
| 12 | Plate snap, not ghost | Sharpen open/smile drive; at `plate_sharpness≥0.75` speech uses a **single** plate (mix=0) so mid-blends cannot ghost | **Built** — default sharpness **0.90** |
| 13 | Denser viseme bank | Landmark-match one real video keyframe per canonical viseme → `viseme_to_plate` in `plate_atlas.json` (up to 16 unique plates) | **Built** — `select_viseme_atlas_frames` |
| 14 | Ground-truth check | Score the pipeline against the source: data half checks capture selection, plates, regions, dataset, model (`scripts/verify_world.py` — 12 checks); playback half (re-render + landmark/SSIM score) still open | **Data half built** |

## Adoption + cell plan + behavior (steps 15–17)

| # | Step | Design idea | Status |
| --- | --- | --- | --- |
| 15 | Mouth groups | Retargetable lip/teeth/cavity cell plan (L03) | **Built** — `mouth_groups`, `mouth_cell_plan`, `cell_cluster` |
| 16 | Avatar adoption | Any qualifying world dir → same GPU path | **Built** — `avatar_profile` / `open_avatar` |
| 17 | Behavior ML | Measured transitions + retrainable fill for gaps | **Built** — `behavior/`, `scripts/retrain_behavior.py` |

## Mouth-blur root causes (fixed after steps 11–13)

The residual "blurry mouth" was compositing, not resolution:

- **Cavity box on closed lips**: `mouth_gap` predicts lip travel with the
  forward displacement while the 3-iteration inverse warp undershoots big jaw
  drops — the predicted hole painted a hard dark rectangle over chin skin.
  Fixed: 6 warp iterations, anatomical span cap, radial mouth falloff,
  span-proportional feather, and the cavity **bows out** as the open plate
  takes over (`dark_cavity: Never` owns the mouth).
- **Washed veil at mid-blend**: plates come from different video frames with
  slightly different exposure, so a half-driven plate discolored the whole
  matte. Fixed: `capture.match_plate_to_reference` color-transfers every plate
  (open/smile/surprise/atlas) onto the rest frame's lighting over the matte's
  feather ring at digest time.
- **Sticky HAPPY smile + soft amount** (still looked blurry after the above):
  `smile_happy_floor` mid-park + soft amount. Fixed: mute smile under open,
  `smile_open_overlap=1.0`, hard-snap plate/smile amount, hold speaking plate
  on REST only, atlas strength → 1.0 when speaking, plates LINEAR no mips.
- **Open layer lingers on lip-tighten (hmm/PP)**: hold kept OH after CLOSED,
  and jaw lag kept `open.png` full. Fixed: PP/MM/CLOSED snap open amount → 0,
  cancel open hold, smile off while tight; hard-snap open.png follows plate
  amount only (not jaw lag).
- **Dark soft rectangle over closed lips**: HAPPY `smile_happy_floor` parked
  `smile.png` (wide soft matte) at full strength while REST; cavity could still
  tint a slit. Fixed: floor default **0**, smile from width only; cavity gated
  off when jaw/open ≈ 0.
- **Wrong appear/disappear timing**: open/smile/atlas followed ease + hold +
  jaw lag, not the viseme schedule. Fixed: `MouthLayerTimeline` snaps GPU
  layers to each span (`due_at`→`duration`); muscle hold may lag, plates do not.
  Polish: **min speech dwell** + **bridge gaps** so AH→REST→OH flashes stop.
  Realtime **Hold** scrollbar in the chat panel (and Mouth Slow/Normal/Fast)
  retunes dwell/bridge live — drag right to keep teeth/plates on longer.
  Hold is **visual plate time only** (not TTS audio).
- **Wide soft open.png under atlas** (residual blur after hard-snap amounts):
  GPU log showed open/smile/mix already 0 or 1 — no ghost drive. Blur came
  from stacking `open.png` (soft ellipse over ~11% of the frame, mean α≈0.36)
  under the tighter atlas, plus warping the identity photo under a static
  plate (double-image lips). Fixed in `avatar.frag`: mute capture open/smile
  when atlas owns speech, harden matte alphas; damp field warp while a speech
  plate is active. Do **not** hard-rest-align the photo under an open jaw
  (that reopened a synthetic mouth gap); cavity suppress follows layer/atlas
  amount, not the muted open.png drive.
- **Eyebrows stuck**: procedural brow lift was multiplied by Master-Lock
  `unlocked` (brows live in locked skull → always 0); HAPPY catalog brow=0;
  telemetry `NEUTRAL` blocked conversation emotion. Fixed: display brow no
  longer lock-gated; HAPPY/speech brow floors; `_active_emotion()` fallback.
- **No per-cell object control**: speakers collapsed Oris/Jaw to one mouth disc
  and constraint had no neighbor exchange. Fixed: `CellClusterIndex` loads every
  unlocked soft cell; speech spreads ±4 across nearby cluster cells; GPU
  velocity-only Moore blend; bridge `GET /cells` + `POST /cells/drive`.
- **Word-timed cell plan**: `MouthCellPlan` detects lip roles on the mouth
  cluster, maps each viseme to an open/width/round flow, and each tick aims
  cells toward their next neighbor using the same phoneme/`active_until` clock
  as the layer timeline (`GET /cells` → `mouth_cell_plan`).
- **Mouth groups (lips / teeth / cavity)**: named cell sets with per-viseme
  recipes and `retarget_group` so teeth vs lips can change later — see
  [`MouthCellGroups.md`](MouthCellGroups.md).
- **Display layer hierarchy (L00–L11)**: one ordered stack
  field → look plates → presentation, coded in `aiface.display_layers`,
  mirrored in `avatar.frag` comments and `gpu_display_recipe.json`. Per-tick
  `FrameLayerState` skips idle L03 cell work and marks atlas-owned capture /
  cavity inactive for consistent realtime — see [`DisplayLayers.md`](DisplayLayers.md).
- **Avatar adoption**: `aiface.avatar_profile` abstracts any world directory that
  meets requirements (`.bds` + identity + open/smile + `mouth_unlocked` cells).
  Train writes `avatar_profile.json`; runtime `open_avatar()` loads the bundle —
  see [`AvatarAdoption.md`](AvatarAdoption.md).
- **Behavior transitions + ML fill**: measured `cell_transition_track` from the
  upload (group controls + deltas); `behavior_model` trained on that track fills
  gaps and live speech — see [`AvatarBehavior.md`](AvatarBehavior.md).

## Runtime addressing + CPU discipline (same audit)

- **Object address**: digestion clusters the seeded high-permeability mouth
  flesh into its own `mouth_unlocked` region (lock state alone merged the
  mouth with unlocked background — the address was lost). The app now reads
  that object's centroid from `region_catalog.json` first
  (`Mouth object address from region catalog: N cells @ (x, y)`).
- **BDS/NWR stay on the GPU**: periodic mouth telemetry reads back only the
  mouth row band (~1.1 MB) instead of the whole 8 MB world every 4 frames
  (`FieldRuntime._read_world_rows`).

## One command

```powershell
python scripts/amin_train.py --video assets/avatar_video_inputs/Generate_a_single_continuous_.mp4
aiface --demo --tts --gpu-log --world output/worlds/avatar/avatar_face.bds
```

`--gpu-log` writes 60 Hz GPU object drives to `output/previews/gpu_tick.log` (stdout every 12 ticks).

## Data save

[`AMIN_DATA_STORE.md`](AMIN_DATA_STORE.md) — one ~8 MB `.bds` + KB side-cars.

## Authority

- **Words/visemes** → jaw  
- **Capture plates** → open/smile looks (must paint, not gap-only)  
- **ML** → unknown cover only  
- **Master Lock** → identity  
