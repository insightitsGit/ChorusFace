# How we save Amin-step data (without drowning)

Master design: [`AMIN_DESIGN.md`](AMIN_DESIGN.md) · Index: [`README.md`](README.md)

You asked: digest cells → regions → 32 props → GPU recipe → video vectors —
**how do we store that much?**

We do **not** dump every float of every cell at every time into JSON.
We store **one dense field + compact side-cars**.

## Size today (`output/worlds/avatar/`)

| Artifact | ~Size | What it is |
| --- | --- | --- |
| `avatar_face.bds` | **~8 MB** | Full world: 256×256×**32** float32 + header |
| `face_tissue.npy` / `face_parts.npy` | ~1 MB each | Per-cell GPU maps (mobility, slit, parts) |
| Plates PNGs (`open`/`smile`/atlas) | ~0.1–0.5 MB each | Condition **looks** (real frames, not invented RGB) |
| `region_catalog.json` | ~17 KB | Region **objects**: mean of 32 channels + cell counts + **sample** coords |
| `condition_maps.json` | ~3 KB | Word/sound/emotion → openness / impulse tables |
| `gpu_display_recipe.json` | ~1 KB | How the GPU shows a look (same path at runtime) |
| `live_vector_dataset.npz` | ~6 KB | Audio features → control vectors (train set) |
| `live_vector_trajectory.json` | ~19 KB | Time series of live vectors from the take |
| `live_vector_model.joblib` | ~50 KB | Small regressor covering **unknown** sounds |
| `cell_transition_track.npz` | ~tens of KB | Measured mouth-group transitions + deltas from the take |
| `avatar_observations.json` | ~20 KB | Measured smile/open GPU vectors + rest deltas (gap-fill anchors) |
| `behavior_model.joblib` | ~50–100 KB | ML fill for **missing** transitions / live speech |

Identity truth lives once in `.bds` + `source_face.png`. Everything else is indexes, tables, or short vectors.

See [`AvatarBehavior.md`](AvatarBehavior.md) for measured → ML authority.  
**Full layer inventory:** [`NWRDataDesign.md`](NWRDataDesign.md).

## Math of the big piece

```text
256 × 256 × 32 × 4 bytes  ≈  8.4 MB   ← one .bds world
```

That **is** the 32-float cell store. We do not replicate it per region or per frame.

## What we intentionally do **not** save

| Temptation | Why we skip it |
| --- | --- |
| Full cell lists for every region in JSON | Millions of coords → use `.bds` + masks / samples |
| Per-frame full grids from video | Video stays the source; we extract **live vectors** (3 floats/time) |
| Generative face RGB | Forbidden — identity albedo is locked |
| True 3D voxel grid | Z is a **channel**, not a third axis |

## Region objects (steps 5–7)

```text
region = {
  name, cell_count,
  mean_channels[32],   ← learned summary of the 32 props
  locked_frac, z_signal_mean,
  cells_sample[≤64]    ← enough to debug, not the whole cluster
}
```

Full membership is recoverable by re-segmenting the `.bds` (lock / opacity), or by a future packed mask `.npy` if we need exact replay.

## Condition maps (step 6)

Known **words / visemes / emotions** → tiny tables (`condition_maps.json`).
Unknowns → `live_vector_model.joblib` (cover), not a second face.

## Live vectors (step 9)

Video truth → `[openness, jaw, width]` over time → train audio→vector model.
Runtime drives the **same GPU recipe** — no second renderer.

## One train command writes the set

```powershell
python scripts/amin_train.py --video YOUR.mp4 --world-dir output/worlds/avatar
```

See also: [`AMIN_STEPS.md`](AMIN_STEPS.md), [`AvatarCellDataflow.md`](AvatarCellDataflow.md).
