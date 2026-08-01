# Avatar digest → cell regions → drive mapping

Master design: [`AMIN_DESIGN.md`](AMIN_DESIGN.md)

## Agreement

An **object** is not a separate mesh. It is a **connected cluster of cells**
with shared roles (lock, soft tissue, plate look) and relations in
**x, y, (z signal), and time**.

```text
Digest teaches cells and looks.
Mapping teaches words / sounds / emotions.
Video teaches live vectors + measured group transitions.
Runtime validates (Master Lock) on L00–L11.
ML covers holes (live-vector) and fills transition gaps (behavior).
Identity albedo stays locked.
Capture open/smile plates must actually paint (not gap-gated only).
Any qualifying world dir adopts through avatar_profile.
```

## End-to-end data flow

```mermaid
flowchart TB
  src[Photo_or_video_take]
  digest[Digest_landmarks_and_regions]
  cells[Cell_field_bds_256x256x32]
  regions[Region_objects_cell_clusters]
  groups[Mouth_groups_lips_teeth_cavity]
  looks[Condition_looks_rest_smile_open]
  recipe[GPU_display_recipe_L00_L11]
  profile[avatar_profile]
  map[Drive_mapping_words_sounds_emotions]
  cover[Live_vector_ML_cover]
  track[cell_transition_track]
  behavior[behavior_model_ML_fill]
  runtime[Runtime_MasterLock_same_GPU_path]

  src --> digest
  digest --> cells
  digest --> regions
  regions --> groups
  digest --> looks
  digest --> recipe
  digest --> profile
  src --> track
  track --> behavior
  cells --> map
  regions --> map
  looks --> map
  map --> runtime
  cover --> runtime
  behavior --> runtime
  groups --> runtime
  recipe --> runtime
  profile --> runtime
```

## Stage A — Digest

**Input:** frontal capture (rest · smile · open · talk · surprise)

**Output:** `.bds`, `source_face.png`, plates, tissue/parts, `region_catalog.json`

## Stage B — 32 properties / cell

See [`BDSMotionMap.md`](BDSMotionMap.md) and [`AMIN_DESIGN.md`](AMIN_DESIGN.md) Step 2.

## Stage C — Region objects

| Region | Relation | Looks |
| --- | --- | --- |
| Identity (locked) | Master Lock ≥ 0.5 | `source_face` only |
| Mouth unlocked | Soft cluster | `open.png` / atlas when open |
| Smile band | Width / HAPPY | `smile.png` |
| Brows / lids | Upper tissue | `surprise.png` |

## Stage D — Drive mapping

```text
word / sound / viseme / emotion
        ↓
  region drives + plate amounts + group cell plan
        ↓
  same GPU recipe (L00–L11)
```

- **Tables** for known visemes/emotions  
- **Live-vector ML** for unknown sounds  
- **Behavior ML** fills missing transition samples / live group motion  
- **Jaw** follows words; plates follow open/smile drives  

## Stage E — Mouth groups (L03)

See [`MouthCellGroups.md`](MouthCellGroups.md). Membership is retargetable;
recipes per viseme scale open / width / round / close per group.

## Stage F — Adoption + behavior

| Piece | Doc |
| --- | --- |
| Portable world dir | [`AvatarAdoption.md`](AvatarAdoption.md) |
| Measured transitions + retrain | [`AvatarBehavior.md`](AvatarBehavior.md) |
| Display order | [`DisplayLayers.md`](DisplayLayers.md) |

## Storage

[`AMIN_DATA_STORE.md`](AMIN_DATA_STORE.md)
