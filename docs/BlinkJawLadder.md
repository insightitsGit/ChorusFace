# Blink + jaw ownership ladder — one change, then QA

**Baseline:** dense-kit v3 world (`second_avatar_calibration.mp4`) +
render-fidelity Steps 1–4 kept. Blink plate `eyes_closed.png` baked.

**Rule:** Land **one** step, auto QA, then visual QA. Do not stack.

## Do not re-land yet

| Change | Why wait |
| --- | --- |
| Muscle jaw residual | Extra jaw without matching plate (failed before) |
| mouth_muscles under TickFeed | Muddies FIELD vs plates |
| Occlusion teeth/tongue | Separate oral-interior experiment |

## Ladder

| Step | Change | Auto QA | Visual QA |
| --- | --- | --- | --- |
| **BJ1** | Latch lid teacher + open deadzone + ease from tick 0 | unit + source guard | ✅ kept (lash expand BJ1b reverted) |
| **BJ2** | Pack EyeSystem → `lid_amt` on live/zero/hearing | unit | ⏭ skipped — blink park |
| **BJ3** | Hard-zero `avatar_jaw` under TickFeed LOOK | blink_jaw ladder + HUD `jaw=` | ✅ kept |

Demo:

```powershell
python scripts/run_tickfeed_demo.py --no-chorus --fidelity-hud
```
