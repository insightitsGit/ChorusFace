# Display layers — hierarchy for realtime

Single coding for how AIFace gets from field → pixels. CPU
(`display_layers.py`, timeline, cell plan) and GPU (`avatar.frag`) share the
same **L00–L11** order. Do not reorder Plane B without a deliberate shader
change — mid-stack swaps caused blur / gap regressions.

Related: [`AMIN_STEPS.md`](AMIN_STEPS.md) · [`MouthCellGroups.md`](MouthCellGroups.md) ·
[`AvatarAdoption.md`](AvatarAdoption.md) · `aiface.display_layers` · `aiface.runtime.recipe`

## Planes (authority)

```text
FIELD (moves tissue; never rewrites albedo)
  L00 identity_lock
  L01 field_velocity
  L02 muscle_jaw_warp
  L03 cell_groups          ← MouthCellPlan / mouth_groups

LOOK (photographed plates on UV)
  L04 capture_smile
  L05 capture_open         ← muted when atlas owns speech
  L06 cavity_fill          ← suppressed under atlas
  L07 atlas_viseme         ← speech look authority (hard snap)
  L08 expr_upper

PRESENTATION
  L09 eyes_lids
  L10 brow_procedural
  L11 hud_overlay
```

## Realtime rule

Each sim tick, `evaluate_frame_layers()` builds a `FrameLayerState`:

- **Active flags** — which layers work this frame
- **Amounts** — drive magnitudes for probes / gpu-log
- **Skip** — e.g. L03 cell enqueue is skipped on REST; under hard snap +
  atlas ownership, capture open/smile + cavity are marked inactive (shader
  already mutes; CPU stays consistent)

Inspect live stack:

- `GET /cells` → `display_layers`
- `GET /probe` → `display_layers`
- `--gpu-log` → `layers=identity_lock+muscle_jaw_warp+…`

## Invariants

1. Identity photo + Master Lock stay under every look plate.
2. Capture plates paint **before** cavity; atlas paints **after** cavity.
3. When atlas owns speech (hard snap), do not let muted open.png zero cavity
   suppress — cavity follows **layer/atlas amount**.
4. Brow (L10) is display-only; never gate by Master Lock `unlocked`.
5. Mouth groups are L03 only; they write velocity into L01, not albedo.

## Recipe contract

`gpu_display_recipe.json` `display_path` is generated from `DISPLAY_PATH`
(`aiface.gpu_display_recipe.v3`). Train and play share that list.
