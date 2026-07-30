# Live-vector package (AminIntheLoop)

Master design: [`AMIN_DESIGN.md`](AMIN_DESIGN.md) Step 9.

```text
capture video
  → extract live control vectors (open / jaw / width over time)
  → train audio → vector model
  → runtime LiveVectorDriver → same GPU display recipe
```

Identity stays digested photo + Master Lock. No new face RGB.

## Prefer the full Amin pipeline

```powershell
python scripts/amin_train.py --video assets/avatar_video_inputs/YOUR.mp4
```

That runs digest + maps + recipe + live vectors together.

Vectors-only (world already exists):

```powershell
python scripts/train_avatar_from_video.py --video … --world-dir output/worlds/avatar
# or
python scripts/amin_train.py --skip-digest
```

## Artifacts

| File | Purpose |
| --- | --- |
| `live_vector_trajectory.json` | Video truth as time series |
| `live_vector_dataset.npz` | Train features + labels |
| `live_vector_model.joblib` | Runtime model |
| `live_vector_model.meta.json` | Metrics |

## Runtime authority

- **Visemes / words** own jaw timing  
- **Live vectors** cover unknowns and help plate/smile width  
- GPU path: `avatar.frag` (capture plates + warp + atlas)

See [`LiveControlVectors.md`](LiveControlVectors.md).
