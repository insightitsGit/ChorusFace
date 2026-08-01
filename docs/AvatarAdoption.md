# Avatar adoption — any qualifying face

Digestion learns **how each cell couples to the GPU** from the user's upload
(identity photo + open/smile plates + unlocked mouth cluster). Runtime must
not hard-code one face under `output/worlds/avatar/`.

`aiface.avatar_profile` is the abstraction layer: one world directory in →
same display stack (L00–L11) + cell plan out.

Related: [`AvatarCapture.md`](AvatarCapture.md) · [`DisplayLayers.md`](DisplayLayers.md) ·
[`MouthCellGroups.md`](MouthCellGroups.md)

## Contract

```text
upload (video / stills)
        │
        ▼
digest → world_dir/
   avatar_face.bds          dense 256²×32 field
   source_face.png          identity
   open.png / smile.png     capture looks
   region_catalog.json      mouth_unlocked object
   gpu_display_recipe.json  knobs
   avatar_profile.json      ← adoption side-car
        │
        ▼
open_avatar(world_dir)  →  AvatarBundle
        │
        ▼
AvatarFaceApp  (same GPU path for every ok bundle)
```

## Requirements (hard)

| Need | Why |
| --- | --- |
| `avatar_face.bds` (or any `*.bds`) | Cell field + Master Lock |
| `source_face.png` | Immutable identity photo |
| `open.png`, `smile.png` | Look plates (L04/L05) |
| `mouth_unlocked` ≥ 32 cells | Per-cell GPU coupling (L01/L03) |
| `face_box` in BDS seed | Muscle UV registration |

Soft (warn): surprise plate, atlas, expression catalog, live-vector model,
`condition_maps.json`.

Capture hard gates still apply — see [`AvatarCapture.md`](AvatarCapture.md).

## API

```python
from aiface.avatar_profile import open_avatar, list_avatars, meets_requirements

bundle = open_avatar("output/worlds/ava")          # dir or .bds
bundle.require()                                  # raise if incomplete
print(bundle.profile.geometry.mouth_cell_count)

for profile in list_avatars():                    # scan output/worlds/*
    print(profile.id, profile.validation.ok)
```

Train writes the profile at the end of `run_all_steps`. Verify refreshes it:

```powershell
python scripts/verify_world.py --world output/worlds/my_face
aiface --demo --world output/worlds/my_face/avatar_face.bds
```

## Behavior from the upload

Each adopted world also learns **how this face moves** between seconds:

- measured track → `cell_transition_track.*`
- ML fill for missing in-betweens → `behavior_model.joblib`

See [`AvatarBehavior.md`](AvatarBehavior.md).

## What stays global (not per avatar)

- Display layer order L00–L11 (`display_layers`)
- Viseme → group recipes (`mouth_groups`)
- Master Lock / ±4 command policy
- Shader composite order in `avatar.frag`

## Multi-avatar layout

```text
output/worlds/
  avatar/          # default
  friend_a/
  friend_b/
```

Each directory is one adoptable face. Switch with `--world`.
