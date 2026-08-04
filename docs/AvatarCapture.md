# Avatar Capture Converter

Short high-quality face video (or stills) → Path 1 identity seed **plus** real
smile/open expression plates and lip/jaw travel priors.

This is **not** the NWR `video2game` terrain converter. It does not invent teeth
or cheeks; interiors come from frames of your own face.

## Preflight checklist

Before you hit record:

- [ ] Phone/camera at eye level, lens ~0.5–1 m away
- [ ] Face fills most of the frame (forehead to chin; no extreme crop)
- [ ] Even light on both sides of the face (window + fill, or soft lamp)
- [ ] No sunglasses, hats that hide brows, or strong backlight
- [ ] Head **frontal** — look into the lens; avoid profile / strong yaw
- [ ] Hold still; slight natural motion is fine, walking is not

## Capture kit (5–15 seconds)

One continuous take. Speak the beats out loud if it helps you pace:

| Beat | Duration | What to do |
| --- | --- | --- |
| **1. Rest** | ~1 s | Neutral face, mouth closed, lips relaxed |
| **2. Smile** | ~1 s | Closed-lip smile (teeth optional but lips stay mostly closed) |
| **3. Open** | ~1–2 s | Drop the jaw on “ah” so **teeth are clearly visible** |
| **4. Surprise** | ~1 s | Raise brows, widen eyes (soft “oh!” face) — brows must move |
| **5. Talk** | ~5–10 s | Read the talk line below at a natural pace |

For **Side B dense collection / lab samples**, prefer the fixed **~8s calibration
script** (adds explicit **Say “hi”** + **Angry** + timed windows) in
[`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) §0b.

**Talk line (open vowels + bilabials):**

> “Hello there. My name is Ava. Please buy me a puppy. How are you today?”

Optional stills if you have no video: three photos with the **same framing** —
`rest.png`, `smile.png`, `open.png` — matching beats 1–3.

### Hard reject (converter will fail or drop frames)

| Reason | What went wrong |
| --- | --- |
| `multi_face` | More than one face in frame |
| `blur` | Motion blur / soft focus |
| `yaw` | Profile or strong head turn |
| `eyes` | Sunglasses / eyes not readable |
| `crop` | Face box too small or cut off |
| `landmarks` | Could not lock eyes/mouth reliably |
| `selection` | Rest/smile/open not distinct enough — retake Open/Smile |

## CLI

```bash
# Video take (preferred)
chorusface-capture --video face_take.mp4 --output output/worlds/avatar/

# Stills fallback (same hard gates; use --allow-soft only for slight blur)
chorusface-capture --rest r.png --smile s.png --open o.png --output output/worlds/avatar/

# Play only after QA looks good
chorusface --world output/worlds/avatar/avatar_face.bds

# Any other world dir that meets adoption requirements works the same way
chorusface --world output/worlds/my_face/avatar_face.bds
```

Adoption contract (portable world dirs): [`AvatarAdoption.md`](AvatarAdoption.md).

The CLI prints a **reject report** (kept vs dropped by reason). Exit code **0**
means the take passed selection safety; **1** means retake.

### How to read `capture_qa.png`

Three panels left → right: **REST | SMILE | OPEN**, each with landmark dots.

- REST: mouth closed; open score low
- SMILE: wider corners than REST; still mostly closed
- OPEN: clear jaw drop / teeth; open score clearly higher than REST

If OPEN looks like REST, retake beat 3. If SMILE looks identical to REST, retake
beat 2.

Also check `capture_meta.json` for chosen frame indices, phase scores, and
travel priors.

## Bundle additions

Alongside Path 1 files (`avatar_face.bds`, `source_face.png`, `face_tissue.npy`,
`face_parts.npy`):

| File | Role |
| --- | --- |
| `smile.png` | Real smile plate (RGB + mouth-interior alpha) |
| `open.png` | Real open-mouth plate (RGB + mouth-interior alpha) |
| `surprise.png` | Real upper-face plate (brows + wider eyes, soft alpha) |
| `expression_catalog.json` | Fast DB: emotion → role plate + brow/eye params |
| `capture_meta.json` | Frame indices, phases, talk series, priors, rejects |
| `capture_qa.png` | Contact-sheet QA (REST\|SMILE\|OPEN\|SURPRISE) |

Priors also land in `.bds` under `application_metadata.avatar_seed.capture`.

## Expression catalog (fast DB)

`expression_catalog.json` sits next to the BDS. Runtime looks up the current
emotion label (`SURPRISED`, `HAPPY`, …) → role (`surprise`, `smile`, …) and
reads learned `brow_raise` / `eye_widen` plus the plate path. Capture scores
brows and lid openness from the video so surprise is not invented by the shader.

## Runtime

1. Rest plate = immutable identity warp target (`source_face.png`).
2. Muscles + jaw drive lip silhouette on the rest photo.
3. Inside `mouth_gap`, the shader blends **open** / **smile** plate pixels,
   multiplied by the plate’s mouth alpha (oral-only — no cheek leak).
4. Emotion drives Frontalis / lid widen and composites the **surprise** plate
   over brows/eyes when the catalog says so.
5. Talk-segment priors scale jaw/lip travel — not a neural morph.

## Explicit non-goals (v1)

- NWR optical-flow terrain path for faces
- Per-user ML morph training
- DeepFake / live mesh tracking as the identity path
- Long video — short HQ beats are the design
