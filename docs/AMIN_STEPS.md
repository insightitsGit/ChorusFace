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
| 8 | GPU recipe | Same display path at play | `gpu_recipe`, `avatar.frag` |
| 9 | Live vectors | Video → controls → model | `aiface.live_vector` |
| 10 | Train + play | One pipeline | `scripts/amin_train.py`, `aiface` |

## Realism track (steps 11–14 — why the mouth blurs today)

Field stays 256×256×32 (authority/physics). Display fidelity is a separate
axis — nothing in steps 1–10 fixes it, so these steps do.

| # | Step | Design idea | Status |
| --- | --- | --- | --- |
| 11 | Native-res display plane | Capture photo + plates at 1024², registered to the same face box; grid stays 256. Selected frames are re-cut from the source at display res (`capture.resample_frames_hires`); textures upload at native size, photo mipmapped, part ids split into their own grid-res NEAREST map | **Built** — `DISPLAY_SIZE=1024`, app logs `Avatar base texture: 1024x1024 display res` |
| 12 | Plate snap, not ghost | Sharpen open/smile drive curve and bias the atlas pair toward the nearest real mouth shape — a 50/50 blend of two photos reads as motion blur | **Built** — `plate_sharpness` recipe knob → `avatar_plate_sharpness` uniform |
| 13 | Denser viseme bank | One real video keyframe per canonical viseme (landmark-matched), not just 8 openness bins — each sound shows the actual video mouth | **To build** |
| 14 | Ground-truth check | Score the pipeline against the source: data half checks capture selection, plates, regions, dataset, model (`scripts/verify_world.py` — 12 checks); playback half (re-render + landmark/SSIM score) still open | **Data half built** |

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
aiface --demo --tts --world output/worlds/avatar/avatar_face.bds
```

## Data save

[`AMIN_DATA_STORE.md`](AMIN_DATA_STORE.md) — one ~8 MB `.bds` + KB side-cars.

## Authority

- **Words/visemes** → jaw  
- **Capture plates** → open/smile looks (must paint, not gap-only)  
- **ML** → unknown cover only  
- **Master Lock** → identity  
