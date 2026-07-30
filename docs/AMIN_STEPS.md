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
