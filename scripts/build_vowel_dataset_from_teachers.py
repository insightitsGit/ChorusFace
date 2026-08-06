"""Build Dataset A priors from teacher landmarks + retrain Model A.

Uses MediaPipe lip width/aperture as a proxy mapped into 9D, blended with
§5.2 priors. Enough to advance Phase-1 with the first 3 clips; refine when
SAD/ANGRY arrive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chorusface.vowel.acceptance import evaluate_model_a
from chorusface.vowel.model_a import ModelA, one_hot_22
from chorusface.vowel.model_b import ModelB
from chorusface.vowel.priors import all_prior_targets, clamp_9d, prior_9d
from chorusface.vowel.schema import EMOTIONS, GA16, GROUP_DIM
from chorusface.vowel.teacher import evaluate_landmarks, extract_mediapipe_landmarks, resample_to_60hz

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "teacher" / "teacher_package_v1"
OUT = ROOT / "output" / "worlds" / "tickfeed" / "vowel"

# Clip stem → emotion
EMOTION_FROM_STEM = {
    "VowelTeacher_NEUTRAL": "NEUTRAL",
    "VowelTeacher_HAPPY": "HAPPY",
    "VowelTeacher_SURPRISED": "SURPRISED",
    "VowelTeacher_THINKING": "THINKING",
    "VowelTeacher_SAD_Part1": "SAD",
    "VowelTeacher_SAD_Part2": "SAD",
    "VowelTeacher_ANGRY": "ANGRY",
    "VowelTeacher_ANGRY_Part1": "ANGRY",
    "VowelTeacher_ANGRY_Part2": "ANGRY",
}


def landmarks_to_proxy_9d(lm: np.ndarray, emotion: str) -> np.ndarray:
    """Single-frame landmark → rough 9D from lip geometry + emotion prior."""
    left = lm[61, :2]
    right = lm[291, :2]
    upper = lm[13, :2]
    lower = lm[14, :2]
    fw = max(float(lm[:, 0].max() - lm[:, 0].min()), 1e-3)
    width = float(np.linalg.norm(right - left)) / fw
    aperture = float(np.linalg.norm(upper - lower)) / fw
    # normalize typical ranges
    spread = float(np.clip((width - 0.25) / 0.25, -1.0, 1.0))
    round_ = float(np.clip((0.35 - width) / 0.2, 0.0, 1.0))
    jaw = float(np.clip(aperture / 0.12, 0.0, 1.0))
    mouth = float(np.clip(aperture / 0.10, 0.0, 1.0))
    teeth = float(np.clip(aperture / 0.08, 0.0, 1.0))
    base = prior_9d("AX", emotion)
    c = base.copy()
    c[4] = mouth
    c[5] = spread
    c[6] = round_
    c[7] = teeth
    c[8] = jaw
    return clamp_9d(c)


def peak_frames(width: np.ndarray, n_peaks: int = 8) -> list[int]:
    """Pick local maxima of lip width for spread-ish peaks + minima for round."""
    T = len(width)
    # simple: evenly spaced windows, take argmax of |width - mean| in each
    peaks: list[int] = []
    for i in range(n_peaks):
        a = int(i * T / n_peaks)
        b = int((i + 1) * T / n_peaks)
        if b <= a + 2:
            continue
        seg = width[a:b]
        # alternate max/min
        idx = int(a + (np.argmax(seg) if i % 2 == 0 else np.argmin(seg)))
        peaks.append(idx)
    return peaks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, default=PKG)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()

    videos = sorted((args.pkg / "videos").glob("*.mp4"))
    if not videos:
        print(json.dumps({"ok": False, "error": "no videos"}))
        return 2

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    d35_reports = []

    # start from full prior grid so missing emotions still have coverage
    x_prior, y_prior = all_prior_targets()
    xs.extend(list(x_prior))
    ys.extend(list(y_prior))

    for video in videos:
        stem = video.stem
        emotion = EMOTION_FROM_STEM.get(stem, "NEUTRAL")
        lm_path = args.pkg / "landmarks" / stem / "landmarks_60hz.npy"
        if lm_path.is_file():
            lm60 = np.load(lm_path)
        else:
            lm, fps = extract_mediapipe_landmarks(video)
            lm60 = resample_to_60hz(lm, fps)
            lm_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(lm_path, lm60)

        metrics = evaluate_landmarks(lm60)
        d35_reports.append({"file": video.name, "emotion": emotion, **metrics.to_dict()})

        left = lm60[:, 61, :2]
        right = lm60[:, 291, :2]
        width = np.linalg.norm(right - left, axis=1)
        peaks = peak_frames(width, n_peaks=8)
        # map peaks to a rotating vowel subset for this emotion
        tags = list(GA16)
        for i, frame in enumerate(peaks):
            tag = tags[i % len(tags)]
            c = landmarks_to_proxy_9d(lm60[frame], emotion)
            # blend proxy with prior so we don't throw away design geometry
            c = clamp_9d(0.55 * c + 0.45 * prior_9d(tag, emotion))
            xs.append(one_hot_22(tag, emotion))
            ys.append(c)

    X = np.stack(xs)
    Y = np.stack(ys)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "dataset_a_bootstrap.npz", X=X, Y=Y)

    model = ModelA()
    stats = model.fit(X, Y, epochs=args.epochs)
    model.save(args.out / "model_a.npz")
    ModelB().save(args.out / "model_b.npz")
    report = evaluate_model_a(model)
    out = {
        "ok": True,
        "n_train": int(X.shape[0]),
        "videos": [v.name for v in videos],
        "train_mse": stats["mse"],
        "acceptance": report.to_dict(),
        "d35": d35_reports,
    }
    (args.out / "bootstrap_report.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    (args.pkg / "ingest_report.json").write_text(
        json.dumps({"d35": d35_reports}, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
