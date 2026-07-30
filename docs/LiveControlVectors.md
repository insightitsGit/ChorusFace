# Live control vectors — process

Master design: [`AMIN_DESIGN.md`](AMIN_DESIGN.md)

## One-line contract

```text
Video already has the truth.
Convert it to live control vectors.
Play those vectors through the same GPU display path.
Do not invent face RGB. Do not paint a new identity into albedo.
```

```text
Digest teaches cells and looks.
Mapping teaches words / sounds / emotions.
Runtime validates (Master Lock).
ML covers holes in the map.
Identity albedo stays locked.
```

## The catch (GPU display recipe)

Learning from digestion is **not only** cell colors and locks.

We must also learn **how the GPU is called** to produce that look:

| Recipe piece | Role |
| --- | --- |
| `source_face` | Immutable identity |
| Tissue / mobility maps | Where warp is allowed |
| Muscle + jaw uniforms | Continuous lip/jaw motion |
| `open.png` / `smile.png` | Real capture looks (direct overlay) |
| Plate atlas | Finer viseme shapes |
| Master Lock (ch 31) | Identity cells refuse AI paint |

Playback = **same recipe**, driven by tables + live vectors.

## Vector shape

| Field | Meaning | Consumer |
| --- | --- | --- |
| `openness_n` | Mouth open [0,1] | `open.png` + plate gate |
| `jaw_n` | Jaw drop [0,1] | Viseme table owns this at runtime |
| `width_n` | Mouth width / smile [0,1] | `smile.png` drive |
| `plate_gate` | Blend amount | Shader `avatar_plate_blend.y` |

## Phases

| Phase | Status | Scope |
| --- | --- | --- |
| 1 Mouth vectors | **Implemented** | extract / train / driver + open/smile overlay |
| 2 Region graph | Partial | `region_catalog.json` + lock split |
| 3 Word/sound/emotion maps | Partial | `condition_maps.json` + speech tables |
| 4 NWR observation loop | Future | External filmstrip / schema handoff |

## Non-goals

- Neural face RGB rewriting identity  
- Invented enamel  
- Path A mouth ownership seals  
- Per-frame full grid dumps  

## Related

- [`AvatarCellDataflow.md`](AvatarCellDataflow.md)  
- [`AMIN_DATA_STORE.md`](AMIN_DATA_STORE.md)  
- [`AvatarCapture.md`](AvatarCapture.md)  
- `src/aiface/live_vector/` · `scripts/amin_train.py`  
