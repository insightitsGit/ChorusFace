# Render fidelity ladder — one change, then QA

**Rule:** Land **one** step, run automated QA, then **visual** QA on the demo.
Do not stack the next step until the current one is accepted.

Baseline (good): commit `2ea6fa3` + prior §14 mouth path
(`open.png` dual-owner intact, original `rest_mix`).

## Do not re-land (failed as a bundle)

| Change | Why it failed on this take |
| --- | --- |
| Kill `open.png` under fidelity | Atlas oral α ~1% vs `open.png` ~12%; AA plate openness≈0.29 |
| Lower `rest_mix` under plates | FIELD smeared under weak LOOK |
| Muscle jaw residual | Extra jaw without matching plate |
| Occlusion teeth mask on atlas α | Noise / flicker on thin mattes |

## Ladder (safe → riskier)

| Step | Change | Auto QA | Visual QA |
| --- | --- | --- | --- |
| **1** | L5 never α-blends into `SOURCE_MEASURED` FIELD | `tests/test_render_ladder.py` | ✅ kept (no visible change / cleaner) |
| **2** | Fidelity HUD only (`--fidelity-hud` / **F**) — no shader change | `tests/test_render_ladder.py` Step 2 | ✅ kept |
| **3** | `resolve_mouth_ownership(..., hard_snap=True)` matches GPU commit | `test_mouth_owner` + ladder Step 3 | ✅ kept |
| **4** | Bind `plate_b = plate_a` when mix already 0 | ladder Step 4 | ✅ kept (no visible change) |
| **stop** | New capture take before atlas-only / geometry-only experiments | — | `AvatarCalibrationPrompt.md` dense kit |

After a denser take (strong atlas α + true AA open), revisit atlas-primary and
geometry-led motion as **new** steps — not before.

## Demo QA command

```powershell
python scripts/run_tickfeed_demo.py --no-chorus
```

Watch: measured 8s pass, open “ah”, smile, talk, then zero-mood. Compare to
memory of baseline. If worse → `git checkout` that step’s files and stop.
