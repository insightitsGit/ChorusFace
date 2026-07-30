# AIFace / AminIntheLoop — documentation index

**Start here for the walkthrough:** [`AMIN_DESIGN.md`](AMIN_DESIGN.md)

## Design (what we agreed)

| Doc | Purpose |
| --- | --- |
| [`AMIN_DESIGN.md`](AMIN_DESIGN.md) | **Master design + step-by-step implementation map** |
| [`AMIN_STEPS.md`](AMIN_STEPS.md) | Short step → code table |
| [`AMIN_DATA_STORE.md`](AMIN_DATA_STORE.md) | How we save data without drowning |
| [`AvatarCellDataflow.md`](AvatarCellDataflow.md) | Digest → regions → mapping flow |
| [`LiveControlVectors.md`](LiveControlVectors.md) | Video → live vectors → same GPU recipe |
| [`FROM_SCRATCH_LIVE_VECTOR.md`](FROM_SCRATCH_LIVE_VECTOR.md) | Clean live-vector package notes |

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

## Train + play

```powershell
python scripts/amin_train.py --video assets/avatar_video_inputs/YOUR.mp4
aiface --demo --tts --world output/worlds/avatar/avatar_face.bds
```
