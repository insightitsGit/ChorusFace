# AminIntheLoop

**Null branch.** No AIFace avatar stack. No Path A mouth ownership. No sealed-mouth patches.

This branch only vendors **Neural World Runtime (NWR)** libraries so we can rebuild
from the substrate — the way Amin asked: start from NWR, not from broken face locks.

## What is here

```text
vendor/nwr/     ← NWR core libs + shaders + design docs (pinned revision)
README.md       ← this file
```

Pinned NWR revision: see `vendor/nwr/NWR_REVISION.txt`.

## NWR rules we keep

1. World = GPU field of 32-float cells (`.bds`)
2. AI proposes commands → runtime validates → GPU executes
3. Channel 31 Master Lock = identity / human boundary
4. No inventing face RGB in this branch yet — substrate first

## What is intentionally absent

- `aiface` avatar app / mouth_owner / live_vector patches
- Path A ownership seals
- Capture / TTS / chat face product code

Those live on other branches (`main`, `live-vector-from-video`). This branch is clean.

## Next (with Amin in the loop)

Rebuild display + control **on top of** `vendor/nwr` only, step by step — not by
re-importing the locked-mouth AIFace path.

## Run NWR sandbox (optional)

From a machine that has NWR deps installed:

```powershell
cd vendor/nwr
# requires moderngl stack; prefer upstream C:/code/NWR .venv for a full run
python main.py
```

Full NWR product docs: `vendor/nwr/docs/FinalDesign.md`, `AI_API.md`.
