# 8-second avatar calibration prompt (Gemini / user)

**This is the standard Side B teacher take.**  
Do not skip it. Without this video, TickFeed has code paths but **not** the agreed measured accuracy for smile / hi / angry / talk.

`calibration_script.json` locks the same beats in code (`aiface.tickfeed.calibration`).

---

## Paste into Gemini (exact)

```text
Create one continuous video, exactly 8.0 seconds long, 24 or 30 fps.

Subject: one adult human face, frontal close-up, looking at the camera.
Neutral indoor lighting. Eyes clearly visible. No sunglasses. No cuts, no scene changes.
Head mostly still (tiny natural motion OK). Face fills most of the frame. Stable crop.

Speak clearly in English. Follow this timeline exactly — one beat after another, no skipping:

0.0–1.0s REST
Neutral face, mouth closed, relaxed, silent.

1.0–2.0s SMILE
Closed-lip smile, mouth corners wide, no big jaw drop, silent.

2.0–3.0s OPEN
Say the sound “ah” once with jaw clearly open so teeth show.

3.0–4.0s SAY_HI
Clearly say the single word “hi” once.

4.0–5.0s SURPRISE
Raise eyebrows, widen eyes, soft “oh” expression (short soft “oh” OK).

5.0–6.0s ANGRY
Clear frown / angry expression, brows down, silent or low effort.

6.0–7.5s TALK
Say exactly this sentence, clearly:
“Hello there. How are you today?”

7.5–8.0s REST
Return to neutral closed mouth, silent.

Output: a single MP4 file only. No text overlay, no logo, no second person.
```

---

## After Gemini finishes — give it to me

Save the file as either:

1. `output/worlds/avatar/calibration_take.mp4` (preferred), or  
2. Drop the MP4 anywhere and tell me the path in chat.

Then I will:

1. Validate against the 8s scaffolding lock  
2. Run Side B collect (every-frame flow → 60 Hz FaceCellTimeline)  
3. Write `speech_align.json` / `look_drive.json` / `qa_report.json`  
4. Retrain L1–L5 on **this** teacher  
5. Confirm beat QA (smile motion in SMILE, “hi” in SAY_HI, talk in TALK)

```powershell
python scripts/train_tickfeed_ml.py --world-dir output/worlds/avatar --video output/worlds/avatar/calibration_take.mp4 --prepare --validate
```

---

## Accuracy note (honest)

| Piece | Status |
| --- | --- |
| Side A / Side B **code** (KEY/Δ, ring, GPU, labels→LOOK, ML layers) | Implemented |
| Teacher video matching **this** script | **Not yet** — current collect used lab file `Generate_a_single_continuous_.mp4`, not a verified Gemini 8s script take |
| Trustworthy smile/hi/angry/talk accuracy | **Blocked until you provide the MP4 above** |

Gemini is **not** identity albedo for a user world. It is the lab/script teacher so Side B labels are known by time.
