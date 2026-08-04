# 8-second avatar calibration prompt (Gemini / user)

**This is the standard Side B teacher take (dense kit v3 — mouth + blink).**  
Do not skip it. Without this video, TickFeed has code paths but **not** the agreed measured accuracy for smile / hi / angry / talk / tongue TH / blink lids.

`calibration_script.json` locks the same beats in code (`chorusface.tickfeed.calibration`).

---

## Paste into Gemini (exact) — blonde woman dense take + blink

```text
Create one continuous video, exactly 8.0 seconds long, 24 or 30 fps.

Subject: one adult woman, blonde, beautiful, natural makeup, clear skin,
frontal face close-up, looking at the camera. Soft natural beauty-light lighting.
Eyes clearly visible. No sunglasses. No cuts, no scene changes.
Head mostly still (tiny natural motion OK). Face fills most of the frame. Stable crop.
Photorealistic, elegant, not cartoon, not stylized anime.

Speak clearly in English with a natural female voice. Follow this timeline exactly — one beat after another, no skipping:

0.0–0.7s REST
True neutral face: mouth fully closed, flat lips, NO resting smile, teeth not visible,
eyes OPEN and relaxed, silent.

0.7–1.4s SMILE
Closed-lip smile, mouth corners wide as possible, no jaw drop, teeth not shown, silent.

1.4–2.3s OPEN
Say the sound “ah” once with jaw clearly wide open so UPPER and LOWER teeth both show.

2.3–3.0s SAY_HI
Clearly say the single word “hi” once.

3.0–3.7s TONGUE_TH
Clearly say the single word “think” once. Tip of the tongue must be visible between the teeth on the “th”.

3.7–4.4s SURPRISE
Raise eyebrows, widen eyes, soft “oh” expression (short soft “oh” OK).

4.4–5.1s ANGRY
Clear frown / angry expression, brows down, silent or low effort. Eyes stay OPEN.

5.1–5.7s BLINK
Keep mouth closed and neutral. Fully close BOTH eyelids (complete blink), hold closed
for most of this window so closed lids are clearly visible, then open again by 5.7s.
Silent. No head turn. No smile.

5.7–7.5s TALK
Eyes open again. Say exactly this sentence, clearly:
“Hello there. How are you today?”

7.5–8.0s REST
Return to true neutral closed mouth (flat lips, no smile), eyes open, silent.

Output: a single MP4 file only. No text overlay, no logo, no second person.
```

---

## Dense capture checklist (mouth + lids)

| Beat / pose | Requirement |
| --- | --- |
| True neutral REST | Flat lips, **no** resting smile; teeth not visible; eyes open |
| Jaw extreme OPEN | Wide “ah” with clear upper + lower teeth |
| Smile extreme | Closed-lip max corners (no jaw drop) |
| Tongue TH | Tip of tongue visible between teeth on “think” |
| **BLINK** | Full bilateral lid close, held ~0.4–0.6s, mouth closed |
| Talk line | Clear bilabials and “there” TH |

After drop-in:

```powershell
python scripts/build_tickfeed_demo.py --clean --video path\to\new_take.mp4
```

Do **not** invent enamel, tongue, or closed-eye RGB. Plates must come from this take.

---

## After Gemini finishes — give it to me

**Drop folder:**

`assets/avatar_video_inputs/calibration_takes/`

Save exactly:

| File | Subject |
| --- | --- |
| `blonde_woman_8s.mp4` | Blonde woman 8s dense+blink script take |
| `male_8s.mp4` | Male 8s dense+blink script take |

Both should follow the **same** 8s v3 script. Then tell me in chat.

Then I will:

1. Validate against the 8s scaffolding lock  
2. Run Side B collect (flow → 60 Hz FaceCellTimeline + lid curve)  
3. Write `speech_align.json` / `look_drive.json` / `qa_report.json`  
4. Bake `eyes_closed.png` from the BLINK window when frames allow  
5. Retrain L1–L5 on **this** teacher  

```powershell
python scripts/train_tickfeed_ml.py --world-dir output/worlds/avatar --video output/worlds/avatar/calibration_take.mp4 --prepare --validate
```

---

## Accuracy note (honest)

| Piece | Status |
| --- | --- |
| Side A / Side B **code** (KEY/Δ, ring, GPU, labels→LOOK, ML layers) | Implemented |
| Dense kit script (`TONGUE_TH`, true REST, wide OPEN, **BLINK**) | **In code v3** — awaiting matching MP4 |
| Current drop without deliberate blink | Good for mouth/TH only — **regenerate with BLINK** before full rebuild |
| Atlas-only / occlusion / jaw-residual render experiments | **Blocked** until denser oral α from this take |
