# VowelDesign — teacher Veo prompts (5 emotions + NEUTRAL state-0)

**Status:** ready to shoot — paste clips into `output/teacher/teacher_package_v1/videos/`.  
**Duration:** **8–10s OK** (Veo often lands ~10s). Prefer 30 fps (24 OK).  
**Parent:** [`VowelDesign.md`](VowelDesign.md) §6.1 · answers: [`VowelDesignDetailAnswers.md`](VowelDesignDetailAnswers.md) (D6 morphological split)  
**Ingest:** `python scripts/ingest_vowel_teachers.py --limit 3`

## Shoot plan (after review merge)

| Emotion | Clips | Notes |
| --- | --- | --- |
| **NEUTRAL** | **1×8–10s** | State-0 baseline + **natural blinks** (no vowels) |
| HAPPY | 1×8s (or 2×8s if dense) | Faster OK |
| SAD | **2×8s** (morphological Part1 / Part2) | Slower face — default split |
| SURPRISED | 1×8s (or 2×8s if dense) | |
| ANGRY | **1×8–10s one-shot** (VERY angry) or morphological 2×8s | Prefer one-shot if readable |
| THINKING | 1×8s (or 2×8s if dense) | |

**REST-under-emotion (critical):** between vowels, **mouth / lips / teeth / jaws
return to state 0** (closed), but **eyes and eyebrows keep the clip emotion**.
Do not drop to a blank neutral face — Dataset A needs emotion×rest separable.

Phase-2: up to **3 performances** of each prompt for ML variation.

**Canonical GA-16 tag order** (inventory / Dataset A indexing — unchanged):

```text
1 EE  2 IH  3 EY  4 EH  5 AE  6 AA  7 AO  8 OH
9 UH 10 OU 11 AH 12 AX 13 ER 14 AY 15 AW 16 OY
```

**Veo Part1 / Part2 split (D6 merge — morphological, not inventory order):**

```text
Part 1 — Spread / Open / Central (8s):
  EE  IH  EY  EH  AE  AA  AX  ER

Part 2 — Round / Back / Diphthongs (8s):
  AO  OH  UH  OU  AH  AY  AW  OY
```

Reason: fewer extreme muscle flips per clip; diphthongs clustered in Part 2.
(Gemini had AY∈P1 / ER∈P2 — three-way merge took Claude’s AY↔ER placement.)

---

## Shared framing (prepend to every prompt)

```text
Create one continuous video, exactly 8.0 seconds long, 24 or 30 fps.

Subject: one adult woman, blonde, beautiful, natural makeup, clear skin,
frontal face close-up, looking at the camera. Soft natural beauty-light lighting.
Eyes clearly visible. No sunglasses. No cuts, no scene changes.
Head mostly still (tiny natural motion OK). Face fills most of the frame. Stable crop.
Photorealistic, elegant, not cartoon, not stylized anime.
Match typical beauty-light frontal calibration framing (same crop style as product avatar takes).

Silent except for the required vowel sounds below (no full English sentences).

Between every vowel: MOUTH REST (state 0) — mouth fully closed, lips flat, teeth not
visible — BUT keep the clip's emotion visible in the EYES and EYEBROWS (do not
relax the emotional face to blank neutral between vowels).
Hold each vowel long enough that lip shape is clearly readable.
```

---

## Prompt — NEUTRAL (1×8–10s · state 0 + blinks · no vowels)

**Purpose:** capture true face state 0 (mouth closed, relaxed brows/eyes) and natural
blink timing for `EyeSystem` / Dataset A NEUTRAL×REST. No vowel walk in this clip.

```text
Create one continuous video, 8.0 to 10.0 seconds long, 24 or 30 fps.

Subject: one adult woman, blonde, beautiful, natural makeup, clear skin,
frontal face close-up, looking straight at the camera. Soft natural beauty-light.
Eyes clearly visible. No sunglasses. No cuts, no scene changes.
Head mostly still (tiny natural motion OK). Face fills most of the frame. Stable crop.
Photorealistic, elegant, not cartoon, not stylized anime.

Emotion for the WHOLE clip: NEUTRAL / calm resting face.
Completely relaxed expression. Soft natural gaze at camera.
Brows relaxed and level (not raised, not furrowed, not sad-oblique).
Eyes open at a normal resting aperture — not wide, not squinted.
Cheeks soft. No smile. No frown. No thinking crease. No surprise. No anger.

MOUTH STATE 0 for the entire clip:
Mouth fully closed, lips gently sealed and flat, teeth not visible, jaw closed.
Do not speak. No vowel sounds. No whispering. Silent.

BLINKS (required):
Perform 3 to 5 natural full blinks spaced across the clip (about every 2 seconds).
Each blink: both upper lids close fully over the eyes, then reopen to the same
relaxed open state. Natural human blink speed — not slow dramatic, not a wink,
not a squint. Keep the face NEUTRAL before and after every blink.
Between blinks, hold the calm open-eyed resting face.

One continuous take. Same crop and light throughout.
```

Save as: `VowelTeacher_NEUTRAL.mp4`

---

## Prompt — HAPPY (1×8s; vowels 1–16 inventory order)

```text
(Use shared framing.)

Emotion for the whole clip: HAPPY — smile energy in eyes/cheeks even during mouth REST.

Walk all 16 vowels in inventory order EE IH EY EH AE AA AO OH UH OU AH AX ER AY AW OY
across exactly 8.0s with mouth-REST between each.
Vowel sounds only. Prefer clear holds over rushing.
If holds are unclear, reshoot as morphological Part1/Part2 like SAD.
```

---

## Prompt — SAD Part 1 (spread/open) and Part 2 (round/back)

Part 1 (already shot OK) used milder sadness. Prefer **VERY SAD** for Part 2 below.

```text
(Use shared framing.)

Emotion for the whole clip: SAD — down brows/mouth corners; keep sadness in eyes/brows
during mouth REST between vowels.

PART 1 (8.0s): EE IH EY EH AE AA AX ER only, mouth-REST between each.
PART 2 (8.0s): AO OH UH OU AH AY AW OY only, same rules.
Same subject, crop, light as Part 1. Vowel sounds only.
```

### Prompt — SAD Part 2 only (VERY SAD · round/back · paste this)

```text
Create one continuous video, 8.0 to 10.0 seconds long, 24 or 30 fps.

Subject: one adult woman, blonde, beautiful, natural makeup, clear skin,
frontal face close-up, looking straight at the camera. Soft natural beauty-light.
Eyes clearly visible. No sunglasses. No cuts, no scene changes.
Head mostly still (tiny natural motion OK). Face fills most of the frame. Stable crop.
Photorealistic, elegant face identity — but strongly sad expression (not cartoon).

Emotion for the WHOLE clip: VERY SAD / deeply sorrowful / heartbroken.
Inner eyebrows pulled UP and inward (classic sad oblique brows), forehead soft
vertical tension between brows, heavy watery eyes, soft glassy gaze at camera,
mouth corners gently down when closed, cheeks soft and heavy.
Sadness must stay LOUD in eyes and eyebrows even when the mouth closes between vowels.
Do NOT relax into mild concern, polite neutral, or blank face. Not angry. Not scared.
This is clear, readable sorrow — expressive enough for lip/face teaching.

Silent except for the required vowel sounds (no English words or sentences).

PART 2 vowels ONLY — walk these 8 across the clip, with clear MOUTH REST between each:
AO OH UH OU AH AY AW OY

Between every vowel: mouth fully closed, lips flat, teeth not visible (mouth state 0),
BUT keep the very-sad brows and watery eyes — never drop the emotional face.
Hold each vowel long enough that lip shape is clearly readable (round/back shapes).
Prefer clear holds over rushing. One continuous take. Same crop and light throughout.
```

Save as: `VowelTeacher_SAD_Part2.mp4`

---

## Prompt — SURPRISED (1×8s)

```text
(Use shared framing.)

Emotion for the whole clip: SURPRISED — raised brows, wider eyes; keep surprise in
eyes/brows during mouth REST between vowels.

Walk all 16 vowels EE…OY (inventory order) across exactly 8.0s with mouth-REST between each.
Vowel sounds only. If holds are unclear, reshoot as morphological Part1/Part2 like SAD.
```

---

## Prompt — ANGRY (1×8–10s · one shot · VERY angry)

**Use this for Veo when the mild angry take fails.** One continuous clip; do not split.

```text
Create one continuous video, 8.0 to 10.0 seconds long, 24 or 30 fps.

Subject: one adult woman, blonde, beautiful, natural makeup, clear skin,
frontal face close-up, looking straight at the camera. Soft natural beauty-light.
Eyes clearly visible. No sunglasses. No cuts, no scene changes.
Head mostly still (tiny natural motion OK). Face fills most of the frame. Stable crop.
Photorealistic, elegant face identity — but extreme angry expression (not cartoon).

Emotion for the WHOLE clip: VERY ANGRY / furious / enraged.
Hard furrowed brows pulled down and inward (glare), narrowed intense eyes,
tense jaw muscles, nostrils slightly flared, cheeks tight, fierce stare at camera.
Anger must stay LOUD in eyes and eyebrows even when the mouth closes between vowels.
Do NOT relax into mild irritation, polite firmness, or blank neutral. Not sad. Not scared.
This is rage held under control — sharp, hot, confrontational.

Silent except for the required vowel sounds (no English words or sentences).

Walk all 16 vowels in this order across the clip, with a clear MOUTH REST between each:
EE IH EY EH AE AA AO OH UH OU AH AX ER AY AW OY

Between every vowel: mouth fully closed, lips flat, teeth not visible (mouth state 0),
BUT keep the furious brows and angry eyes — never drop the emotional face.
Hold each vowel long enough that lip shape is clearly readable.
Vowel attacks may be sharper / more forceful than a calm take.
Prefer clear holds over rushing. One take, one continuous shot.
```

Optional fallback if one-shot is unreadable: morphological Part1/Part2 with the same
VERY ANGRY emotion language above.

---

## Prompt — THINKING (1×8s)

```text
(Use shared framing.)

Emotion for the whole clip: THINKING — soft focused brows / attentive eyes; keep
thinking look in eyes/brows during mouth REST between vowels.

Walk all 16 vowels EE…OY (inventory order) across exactly 8.0s with mouth-REST between each.
Vowel sounds only. Split to morphological 2×8s if needed.
```

---

## Collect checklist

- [ ] Generate clips per table (SAD/ANGRY = morphological 2 parts)
- [ ] Verify mouth REST ≠ dropped emotional eyes/brows
- [ ] Optional phase-2: 2 more performances per prompt
- [ ] Side B collect → 60 Hz → Dataset A/B → train Model A then B
- [ ] Before full shoot: D35 1-day HAPPY Part1 MediaPipe+RAFT + Savitzky–Golay smoke
