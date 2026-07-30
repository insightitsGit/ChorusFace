# AminIntheLoop

Talking avatar rebuilt **from NWR** — digest cells, learn looks + GPU recipe,
drive with words/tables and live vectors. No Path A mouth seals. No invented face RGB.

Pinned substrate: `vendor/nwr/` (see `vendor/nwr/NWR_REVISION.txt`).

## Documentation (start here)

| Doc | What |
| --- | --- |
| **[`docs/AMIN_DESIGN.md`](docs/AMIN_DESIGN.md)** | **Master design + all 10 steps → code** |
| [`docs/README.md`](docs/README.md) | Full doc index |
| [`docs/AMIN_STEPS.md`](docs/AMIN_STEPS.md) | Step cheat sheet |
| [`docs/AMIN_DATA_STORE.md`](docs/AMIN_DATA_STORE.md) | How we save the data |
| [`docs/AvatarCellDataflow.md`](docs/AvatarCellDataflow.md) | Digest → regions → maps |
| [`docs/LiveControlVectors.md`](docs/LiveControlVectors.md) | Video → vectors → GPU |

## Walkthrough → code

| Step | What | Where |
| --- | --- | --- |
| 1 | What is NWR | `vendor/nwr` |
| 2 | 32 floats / cell | `amin_loop.cells` |
| 3 | Control + neighbors | `amin_loop.control` |
| 4 | Digest → regions | `amin_loop.digest` |
| 5–7 | Regions + props | `amin_loop.regions` |
| 6 | Word/sound/emotion maps | `amin_loop.mapping` |
| 8 | GPU display recipe | `amin_loop.gpu_recipe` + `avatar.frag` |
| 9 | Live vectors | `aiface.live_vector` |
| 10 | Train + play | `scripts/amin_train.py` |

## One command

```powershell
pip install -e ".[ml,voice]"
python scripts/amin_train.py `
  --video assets/avatar_video_inputs/Generate_a_single_continuous_.mp4 `
  --world-dir output/worlds/avatar

aiface --demo --tts --world output/worlds/avatar/avatar_face.bds
```

## Rules we keep

1. World = GPU field of 32-float cells (`.bds`)  
2. AI proposes → runtime validates → GPU executes  
3. Channel 31 Master Lock = identity boundary  
4. No invented face RGB / teeth  
5. No Path A ownership seals  
6. Capture open/smile plates must **paint** on the mouth (same GPU recipe)  

## Layout

```text
vendor/nwr/           NWR libs
src/amin_loop/        Walkthrough implementation
src/aiface/           GPU runtime + capture + live_vector
scripts/amin_train.py
docs/                 Design + implementation docs
```
