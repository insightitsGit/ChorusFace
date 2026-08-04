# Dual calibration take report

**Date:** 2026-08-02  
**Takes:** `assets/avatar_video_inputs/calibration_takes/`

| File | World | Duration | Result |
| --- | --- | --- | --- |
| `blonde_woman_8s.mp4` | `output/worlds/avatar` (primary) | 8.0s @ 24fps | collect + L1–L5 **OK** |
| `male_8s.mp4` | `output/worlds/avatar_male` | 8.0s @ 24fps | collect + L1–L5 **OK** |

## Pipeline run

```text
validate → every-frame optical flow → 60 Hz FaceCellTimeline
  → speech_align + look_drive + qa_report
  → train L1–L5
```

## QA (beat windows)

| Check | Woman | Male |
| --- | --- | --- |
| validate (8s lock) | pass | pass |
| open > rest | pass | pass |
| talk > rest | pass | pass |
| surprise > rest | pass | pass |
| look smile > rest | pass | pass |
| SAY_HI has “hi” | pass | pass |
| overall `qa.ok` | **True** | **True** |

Note: closed-lip SMILE is subtle in raw optical-flow energy; teacher blends script-amplified smile synth in the SMILE window so LOOK accuracy is preserved.

## ML (per world)

| Layer | Woman | Male |
| --- | --- | --- |
| L1 train_acc | ~0.91 | ~0.94 |
| L2 mae | ~0.070 | ~0.073 |
| L3 code_mae | ~0.270 | ~0.292 |
| L4 mae | ~0.0023 | ~0.0017 |
| L5 code_mae | ~0.305 | ~0.318 |
| ticks | 478 | 478 |

## How to run the primary avatar

```powershell
python -m chorusface --world-dir output/worlds/avatar
```

Male dual world (scaffold copied for TickFeed teacher; identity plates still from prior digest until re-adopt):

```powershell
python -m chorusface --world-dir output/worlds/avatar_male
```
