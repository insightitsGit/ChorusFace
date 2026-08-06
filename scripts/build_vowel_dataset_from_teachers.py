"""Build Dataset A/B from teacher videos (eyes+brows+mouth) and retrain Model A/B.

Writes:
  teacher_package_v1/pose_targets.npz   — Dataset A (vowel×emotion → 9D)
  teacher_package_v1/transfer.npz       — Dataset B (tick paths @ 60 Hz)
  output/worlds/tickfeed/vowel/model_a.npz
  output/worlds/tickfeed/vowel/model_b.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chorusface.vowel.acceptance import evaluate_model_a
from chorusface.vowel.landmarks_9d import landmarks_series_to_9d
from chorusface.vowel.model_a import ModelA, one_hot_22
from chorusface.vowel.model_b import ModelB
from chorusface.vowel.priors import all_prior_targets, clamp_9d, prior_9d, rest_9d
from chorusface.vowel.schema import EMOTIONS, GA16, GROUP_DIM, TICK_HZ
from chorusface.vowel.teacher import evaluate_landmarks, extract_mediapipe_landmarks, resample_to_60hz

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "teacher" / "teacher_package_v1"
OUT = ROOT / "output" / "worlds" / "tickfeed" / "vowel"

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


def _load_lm60(pkg: Path, video: Path) -> np.ndarray:
    stem = video.stem
    lm_path = pkg / "landmarks" / stem / "landmarks_60hz.npy"
    if lm_path.is_file():
        return np.load(lm_path)
    lm, fps = extract_mediapipe_landmarks(video)
    lm60 = resample_to_60hz(lm, fps)
    lm_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(lm_path, lm60)
    return lm60


def _mouth_width(lm60: np.ndarray) -> np.ndarray:
    left = lm60[:, 61, :2]
    right = lm60[:, 291, :2]
    fw = np.maximum(lm60[:, :, 0].max(axis=1) - lm60[:, :, 0].min(axis=1), 1e-3)
    return np.linalg.norm(right - left, axis=1) / fw


def _peak_frames(signal: np.ndarray, n_peaks: int = 12) -> list[int]:
    t = len(signal)
    peaks: list[int] = []
    for i in range(n_peaks):
        a = int(i * t / n_peaks)
        b = max(a + 3, int((i + 1) * t / n_peaks))
        seg = signal[a:b]
        if len(seg) < 2:
            continue
        idx = int(a + (np.argmax(seg) if i % 2 == 0 else np.argmin(seg)))
        peaks.append(idx)
    return peaks


def _hold_window(series: np.ndarray, center: int, half: int = 4) -> np.ndarray:
    a = max(0, center - half)
    b = min(len(series), center + half + 1)
    return np.mean(series[a:b], axis=0)


def build_datasets(
    pkg: Path,
) -> tuple[np.ndarray, np.ndarray, dict, dict, dict]:
    """Return X,Y for Model A plus pose_targets / transfer / meta dicts."""
    videos = sorted((pkg / "videos").glob("*.mp4"))
    if not videos:
        raise SystemExit("no teacher videos")

    # Prior grid keeps full GA-16×emotion coverage.
    xs, ys = list(all_prior_targets()[0]), list(all_prior_targets()[1])

    pose_keys: list[str] = []
    pose_vals: list[np.ndarray] = []
    transfer_controls: list[np.ndarray] = []
    transfer_emotions: list[str] = []
    transfer_stems: list[str] = []
    d35_reports: list[dict] = []

    # Per-emotion accumulator for REST-under-emotion (mouth closed, eyes/brows live).
    rest_buckets: dict[str, list[np.ndarray]] = {e: [] for e in EMOTIONS}

    for video in videos:
        emotion = EMOTION_FROM_STEM.get(video.stem, "NEUTRAL")
        lm60 = _load_lm60(pkg, video)
        metrics = evaluate_landmarks(lm60)
        d35_reports.append({"file": video.name, "emotion": emotion, **metrics.to_dict()})

        controls = landmarks_series_to_9d(lm60, emotion)
        transfer_controls.append(controls.astype(np.float32))
        transfer_emotions.append(emotion)
        transfer_stems.append(video.stem)

        width = _mouth_width(lm60)
        # REST-under-emotion: mouth near-closed AND eyes open (blinks are overlay).
        closed = (width < float(np.percentile(width, 25))) & (controls[:, 0] < 0.25)
        if np.any(closed):
            rest_c = np.mean(controls[closed], axis=0)
            rest_c[0] = min(float(rest_c[0]), 0.12)  # rest lids stay open
            rest_buckets[emotion].append(clamp_9d(rest_c))

        # Vowel-ish peaks from lip width extrema; keep measured eyes/brows.
        tags = list(GA16)
        for i, frame in enumerate(_peak_frames(width, n_peaks=12)):
            tag = tags[i % len(tags)]
            measured = _hold_window(controls, frame, half=3)
            # Blend mouth geometry toward vowel prior; keep measured upper face.
            prior = prior_9d(tag, emotion)
            c = measured.copy()
            c[4:] = 0.55 * measured[4:] + 0.45 * prior[4:]
            # Eyes/brows: measured wins (that was the missing half).
            c[0:4] = 0.85 * measured[0:4] + 0.15 * prior[0:4]
            # Never teach vowel holds as blinks — blinks are compose-time overlay.
            if float(c[0]) > 0.35:
                c[0] = 0.15 * float(c[0]) + 0.85 * float(prior[0])
            c = clamp_9d(c)
            xs.append(one_hot_22(tag, emotion))
            ys.append(c)
            pose_keys.append(f"{tag}|{emotion}|{video.stem}|{frame}")
            pose_vals.append(c)

        # Blink peaks stay in Dataset B (transfer) only — not Model A vowel targets.

    for emotion, rows in rest_buckets.items():
        if not rows:
            continue
        rest_c = clamp_9d(np.mean(np.stack(rows), axis=0))
        # Mouth forced toward state-0; eyes/brows keep emotion.
        r = rest_9d(emotion)
        rest_c[4:] = 0.7 * r[4:] + 0.3 * rest_c[4:]
        rest_c = clamp_9d(rest_c)
        for tag in ("AX", "AH"):
            xs.append(one_hot_22(tag, emotion))
            ys.append(rest_c)
            pose_keys.append(f"REST|{emotion}|mean")
            pose_vals.append(rest_c)

    X = np.stack(xs).astype(np.float64)
    Y = np.stack(ys).astype(np.float64)
    pose = {
        "keys": np.asarray(pose_keys),
        "Y": np.stack(pose_vals).astype(np.float32),
        "channel_names": np.asarray(
            [
                "eye_aperture",
                "eye_gaze_or_blink",
                "brow_raise",
                "brow_knit",
                "mouth_cavity_gap",
                "lip_spread",
                "lip_round",
                "teeth_visibility",
                "jaw_drop",
            ]
        ),
    }
    # Ragged transfer: store as object list via parallel arrays + lengths.
    lengths = np.asarray([c.shape[0] for c in transfer_controls], dtype=np.int32)
    flat = np.concatenate(transfer_controls, axis=0) if transfer_controls else np.zeros((0, GROUP_DIM), np.float32)
    transfer = {
        "controls": flat,
        "lengths": lengths,
        "emotions": np.asarray(transfer_emotions),
        "stems": np.asarray(transfer_stems),
        "tick_hz": np.asarray([TICK_HZ]),
    }
    meta = {"d35": d35_reports, "n_pose": len(pose_keys), "n_transfer_clips": len(transfer_stems)}
    return X, Y, pose, transfer, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkg", type=Path, default=PKG)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--b-epochs", type=int, default=200)
    args = ap.parse_args()

    X, Y, pose, transfer, meta = build_datasets(args.pkg)
    args.pkg.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(args.pkg / "pose_targets.npz", **pose)
    np.savez_compressed(args.pkg / "transfer.npz", **transfer)
    np.savez_compressed(args.out / "dataset_a_from_teachers.npz", X=X, Y=Y)

    model_a = ModelA()
    stats_a = model_a.fit(X, Y, epochs=args.epochs)
    model_a.save(args.out / "model_a.npz")

    model_b = ModelB()
    # Fit residual corrector from measured transfer clips.
    stats_b = model_b.fit_from_transfers(
        transfer["controls"],
        transfer["lengths"],
        transfer["emotions"],
        epochs=args.b_epochs,
    )
    model_b.save(args.out / "model_b.npz")

    report = evaluate_model_a(model_a)
    # Upper-face separability: ANGRY vs HAPPY on brows/eyes.
    angry = model_a.predict("AA", "ANGRY")
    happy = model_a.predict("AA", "HAPPY")
    upper_l2 = float(np.linalg.norm(angry[:4] - happy[:4]))

    out = {
        "ok": bool(report.passed),
        "n_train": int(X.shape[0]),
        "pose_targets": str(args.pkg / "pose_targets.npz"),
        "transfer": str(args.pkg / "transfer.npz"),
        "train_mse": stats_a["mse"],
        "model_b": stats_b,
        "upper_face_l2_angry_vs_happy": upper_l2,
        "acceptance": report.to_dict(),
        "d35": meta["d35"],
        "sample_targets": {
            "EE|HAPPY": model_a.predict("EE", "HAPPY").tolist(),
            "OU|HAPPY": model_a.predict("OU", "HAPPY").tolist(),
            "AA|ANGRY": angry.tolist(),
            "AX|NEUTRAL": model_a.predict("AX", "NEUTRAL").tolist(),
        },
    }
    (args.out / "teacher_train_report.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    (args.pkg / "ingest_report.json").write_text(
        json.dumps({"d35": meta["d35"], "pose_n": meta["n_pose"]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))
    return 0 if report.passed and upper_l2 >= 0.12 else 2


if __name__ == "__main__":
    raise SystemExit(main())
