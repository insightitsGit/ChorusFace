# Mouth cell groups — vision and plan

Related: [`BDSMotionMap.md`](BDSMotionMap.md) · [`AMIN_STEPS.md`](AMIN_STEPS.md) ·
[`DisplayLayers.md`](DisplayLayers.md) (groups are **L03**) ·
`aiface.mouth_groups` · `aiface.mouth_cell_plan`

## Vision

Mouth motion is **not one disc**. It is five named **cell groups** the word
clock can drive differently — and that you can **retarget** when capture
improves teeth vs lips.

```text
words / viseme timing
        │
        ▼
  MouthLayerTimeline  (plates look)
  MouthCellPlan       (field cells)
        │
        ▼
  ┌─────────────┬─────────────┬──────────────┐
  │ upper_lip   │ lower_lip   │ lip_corners  │
  ├─────────────┼─────────────┼──────────────┤
  │ teeth       │ cavity      │              │
  └─────────────┴─────────────┴──────────────┘
        │
        ▼
  each cell → next neighbor (±4)
```

| Group | Role | Why separate |
| --- | --- | --- |
| `upper_lip` | Outer upper lip flesh | Opens up / presses closed |
| `lower_lip` | Outer lower lip flesh | Opens down with jaw |
| `lip_corners` | Commissures | EE widens, OU rounds |
| `teeth` | Dental band (often cavity rim) | Reveal on AH/SS; **hide on PP** |
| `cavity` | Deep oral interior | Soft follow only |

Teeth and lips will change membership after the next capture pass. Membership is
a table (`retarget_group`) — not hardcoded forever.

## Per-viseme recipes (examples)

| Viseme | upper/lower | corners | teeth | cavity |
| --- | --- | --- | --- | --- |
| AH / AA | strong open | mild | reveal | soft |
| EE | mild open | **wide** | mild | soft |
| OU / OH | open + round | **round in** | mild | soft |
| PP / CLOSED | **close press** | press | **off** | off |
| REST | idle | idle | off | off |

Full tables live in `DEFAULT_VISEME_GROUP_RECIPES`
(`src/aiface/mouth_groups.py`).

## Detect → assign → drive

1. **Detect** every `mouth_unlocked` cell (side / lip / radial).
2. **Assign** to a group (geometry today; part-atlas labels when present).
3. **Time** with the same phoneme / `active_until` as plates.
4. **Drive** only cells in **active** groups for that viseme toward neighbors.

## Retarget lips / teeth later

Bridge (with `--bridge`):

```http
POST /cells/drive
{
  "mode": "retarget",
  "group": "teeth",
  "cells": [[128, 82], [129, 82], [130, 82]]
}
```

Or in code:

```python
plan.retarget_group("teeth", [(128, 82), (129, 82)])
plan.groups.set_recipe("AH", "teeth", GroupMotion(open_scale=0.8))
```

## Observe

`GET /cells` → `mouth_cell_plan.mouth_groups` shows counts + vision blurb +
sample steps tagged with `group`.
