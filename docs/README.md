# AIFace / AminIntheLoop — documentation index

**Start here for the walkthrough:** [`AMIN_DESIGN.md`](AMIN_DESIGN.md)

## Design (what we agreed)

| Doc | Purpose |
| --- | --- |
| [`AMIN_DESIGN.md`](AMIN_DESIGN.md) | **Master design + step-by-step implementation map** |
| [`AMIN_STEPS.md`](AMIN_STEPS.md) | Short step → code table + recent fixes |
| [`AMIN_DATA_STORE.md`](AMIN_DATA_STORE.md) | How we save data without drowning |
| [`AvatarCellDataflow.md`](AvatarCellDataflow.md) | Digest → regions → mapping → behavior flow |
| [`LiveControlVectors.md`](LiveControlVectors.md) | Video → live vectors → same GPU recipe |
| [`FROM_SCRATCH_LIVE_VECTOR.md`](FROM_SCRATCH_LIVE_VECTOR.md) | Clean live-vector package notes |

## New design contracts (layers / adopt / behavior)

| Doc | Purpose |
| --- | --- |
| [`DisplayLayers.md`](DisplayLayers.md) | **L00–L11** stack: field → look plates → presentation |
| [`MouthCellGroups.md`](MouthCellGroups.md) | Named lip / teeth / cavity groups (L03) + retarget |
| [`AvatarAdoption.md`](AvatarAdoption.md) | Any qualifying world dir → same GPU path |
| [`AvatarBehavior.md`](AvatarBehavior.md) | Measured transitions + **retrainable** ML fill for gaps |

## Substrate / product (NWR + avatar)

| Doc | Purpose |
| --- | --- |
| [`Architecture.md`](Architecture.md) | System architecture |
| [`BDSMotionMap.md`](BDSMotionMap.md) | 32 channels + motion map |
| [`AvatarCapture.md`](AvatarCapture.md) | Video/stills → seed + plates |
| [`Path1Portrait.md`](Path1Portrait.md) | Locked portrait path |
| [`PhoneticFidelity.md`](PhoneticFidelity.md) | Viseme / speech fidelity |
| [`VoiceSync.md`](VoiceSync.md) | Audio ↔ face sync |
| [`AvatarChat.md`](AvatarChat.md) | Chat / bridge UX |
| [`HANDOFF.md`](HANDOFF.md) | Historical handoff notes |

## Rules (permanent)

1. Identity = photo + Master Lock (ch 31) — no generative face RGB  
2. Objects = connected **cell clusters**, not separate meshes  
3. Z is a **channel signal**, not a 3D voxel world  
4. Known words/sounds → tables; unknowns → ML cover  
5. Same GPU display recipe at train and play  
6. **No Path A mouth ownership seals**  
7. Display layer order (L00–L11) is shared CPU/GPU — do not reorder Plane B casually  
8. Behavior authority: **measured track → ML fill → viseme table**  
9. New upload → retrain behavior model in that world dir (replace artifacts)

## Train + play + retrain

```powershell
# Full digest + maps + live vectors + behavior
python scripts/amin_train.py --video assets/avatar_video_inputs/YOUR.mp4

# Fast retrain when the user uploads a new take (same face)
python scripts/retrain_behavior.py --video NEW_TAKE.mp4 --world-dir output/worlds/avatar

# Play
aiface --demo --tts --gpu-log --world output/worlds/avatar/avatar_face.bds
```
