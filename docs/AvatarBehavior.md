# Avatar behavior — measured transitions + ML fill

Between capture seconds, per-cell millisecond paths used to be **lost**. Now the
upload teaches two layers:

1. **Measured track** — mouth-group controls + frame deltas from the video  
2. **ML behavior model** — trained on that track to **fill missing** data (gaps
   and live speech)

Related: [`AvatarObservations.md`](AvatarObservations.md) ·
[`AvatarAdoption.md`](AvatarAdoption.md) · [`AMIN_DATA_STORE.md`](AMIN_DATA_STORE.md) ·
[`MouthCellGroups.md`](MouthCellGroups.md)

## Authority

```text
upload video
     │
     ├─► avatar_observations.json         (smile/open GPU + delta vectors)
     ├─► cell_transition_track.npz/json   (measured, honest)
     │         deltas = transform from previous sample
     │
     └─► behavior_model.joblib            (ML fill between observations)
              trained: audio features → group controls

runtime resolve():
  observed smile/open  →  measured @ t  →  ML fill (gaps / live)  →  viseme table
```

We do **not** invent optical flow or generative face RGB. Groups
(`upper_lip`, `lower_lip`, `lip_corners`, `teeth`, `cavity`) are the unit.

## Artifacts (per world dir)

| File | Role |
| --- | --- |
| `avatar_observations.json` | Measured smile/open GPU vectors + rest deltas |
| `cell_transition_track.npz` | times, controls[8], deltas, audio features |
| `cell_transition_track.json` | Readable samples for QA |
| `behavior_dataset.npz` | Train set |
| `behavior_model.joblib` | MLP fill model |
| `behavior_model.meta.json` | MAE / beats_baseline |

Controls: `openness_n`, `jaw_n`, `width_n`, `upper_lip_dy`, `lower_lip_dy`,
`corner_dx`, `teeth_reveal`, `cavity_n`.

## Train / retrain (new upload)

The model is **retrainable**: a new user video replaces
`cell_transition_track.*` + `behavior_model.joblib` in that world dir. Runtime
always loads the latest files from the world (no stale cache).

```powershell
# Full digest + maps + live vectors + behavior (first time / new face)
python scripts/amin_train.py --video YOUR.mp4 --world-dir output/worlds/my_face

# Fast retrain when the user uploads a new take for the same face
python scripts/retrain_behavior.py --video NEW_TAKE.mp4 --world-dir output/worlds/my_face
# same as:
python scripts/amin_train.py --behavior-only --video NEW_TAKE.mp4 --world-dir output/worlds/my_face
```

Current default avatar:

```powershell
python scripts/retrain_behavior.py
```

## Runtime

`BehaviorDriver.try_load(world)`:

- Speech RMS history → ML fill when there is no capture clock
- `MouthCellPlan.apply_behavior_flow(...)` overlays group motion on L03
- Bridge `GET /status` → `behavior` snapshot

## Limits (honest)

- Track is **group-level** from landmarks (~12 fps), not every cell every ms
- Densifying further means sampling the take faster — still measured, not invented
- ML only predicts controls the GPU already understands — never new albedo
