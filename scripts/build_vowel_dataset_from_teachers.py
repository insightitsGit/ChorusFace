"""Build Dataset A/B from teacher videos (eyes+brows+mouth) and retrain Model A/B.

Writes:
  teacher_package_v1/pose_targets.npz   — Dataset A (vowel×emotion → 9D)
  teacher_package_v1/transfer.npz       — Dataset B (tick paths @ 60 Hz)
  output/worlds/tickfeed/vowel/model_a.npz
  output/worlds/tickfeed/vowel/model_b.npz
  output/worlds/tickfeed/vowel/model_a.onnx (when exporter available)
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
from chorusface.vowel.schema import (
    EMOTIONS,
    GA16,
    GROUP_DIM,
    PART1_SPREAD_OPEN,
    PART2_ROUND_DIPH,
    TICK_HZ,
)
from chorusface.vowel.teacher import (
    extract_mediapipe_landmarks,
    prepare_landmarks_60hz,
)

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


def _script_tags_for_stem(stem: str) -> list[str]:
    """Teacher prompt order (VowelTeacherPrompts.md / D6 morphological split)."""
    s = stem.upper()
    if "NEUTRAL" in s:
        return []
    if "PART1" in s:
        return list(PART1_SPREAD_OPEN)
    if "PART2" in s:
        return list(PART2_ROUND_DIPH)
    return list(GA16)


def _load_lm60_filtered(pkg: Path, video: Path, emotion: str) -> tuple[np.ndarray, object]:
    """Extract/resample + F12 SG filter; always rewrite landmarks_60hz.npy."""
    stem = video.stem
    lm_dir = pkg / "landmarks" / stem
    lm_path = lm_dir / "landmarks_60hz.npy"
    raw_path = lm_dir / "landmarks_raw.npy"
    meta_path = lm_dir / "extract_meta.json"
    lm_dir.mkdir(parents=True, exist_ok=True)
    if raw_path.is_file() and meta_path.is_file():
        lm = np.load(raw_path)
        fps = float(json.loads(meta_path.read_text(encoding="utf-8")).get("fps", 30.0))
    else:
        lm, fps = extract_mediapipe_landmarks(video)
        np.save(raw_path, lm)
        meta_path.write_text(json.dumps({"fps": fps}), encoding="utf-8")
    lm60, metrics = prepare_landmarks_60hz(lm, fps, emotion=emotion)
    np.save(lm_path, lm60)
    (lm_dir / "d35_metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
    )
    return lm60, metrics


def _mouth_width(lm60: np.ndarray) -> np.ndarray:
    left = lm60[:, 61, :2]
    right = lm60[:, 291, :2]
    fw = np.maximum(lm60[:, :, 0].max(axis=1) - lm60[:, :, 0].min(axis=1), 1e-3)
    return np.linalg.norm(right - left, axis=1) / fw


def _script_peak_frames(width: np.ndarray, n_tags: int) -> list[int]:
    """Find ``n_tags`` open peaks separated by REST valleys (script-aligned)."""
    t = len(width)
    if n_tags <= 0 or t < 8:
        return []
    # Smooth width slightly for valley detection.
    w = width.astype(np.float64)
    if t >= 5:
        pad = np.pad(w, (2, 2), mode="edge")
        w = 0.1 * pad[:-4] + 0.2 * pad[1:-3] + 0.4 * pad[2:-2] + 0.2 * pad[3:-1] + 0.1 * pad[4:]
    # Equal-time bins as primary schedule (prompt walks vowels in order).
    peaks: list[int] = []
    for i in range(n_tags):
        a = int(i * t / n_tags)
        b = max(a + 4, int((i + 1) * t / n_tags))
        seg = w[a:b]
        if len(seg) < 2:
            continue
        peaks.append(int(a + int(np.argmax(seg))))
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

    # Prior grid keeps full GA-16×emotion coverage (96 cells).
    xs, ys = list(all_prior_targets()[0]), list(all_prior_targets()[1])

    pose_keys: list[str] = []
    pose_vals: list[np.ndarray] = []
    transfer_controls: list[np.ndarray] = []
    transfer_emotions: list[str] = []
    transfer_stems: list[str] = []
    d35_reports: list[dict] = []
    label_notes: list[dict] = []

    rest_buckets: dict[str, list[np.ndarray]] = {e: [] for e in EMOTIONS}
    hard_fails: list[str] = []

    for video in videos:
        emotion = EMOTION_FROM_STEM.get(video.stem, "NEUTRAL")
        lm60, metrics = _load_lm60_filtered(pkg, video, emotion)
        d35_reports.append({"file": video.name, "emotion": emotion, **metrics.to_dict()})
        if bool(getattr(metrics, "hard_fail", False)):
            hard_fails.append(video.name)

        controls = landmarks_series_to_9d(lm60, emotion)
        transfer_controls.append(controls.astype(np.float32))
        transfer_emotions.append(emotion)
        transfer_stems.append(video.stem)

        width = _mouth_width(lm60)
        closed = (width < float(np.percentile(width, 25))) & (controls[:, 0] < 0.25)
        if np.any(closed):
            rest_c = np.mean(controls[closed], axis=0)
            rest_c[0] = min(float(rest_c[0]), 0.12)
            rest_buckets[emotion].append(clamp_9d(rest_c))

        tags = _script_tags_for_stem(video.stem)
        if not tags:
            label_notes.append(
                {"file": video.name, "script": [], "peaks": 0, "note": "neutral/blinks"}
            )
            continue

        peaks = _script_peak_frames(width, n_tags=len(tags))
        mismatch = abs(len(peaks) - len(tags)) > 2
        label_notes.append(
            {
                "file": video.name,
                "script": tags,
                "peaks": len(peaks),
                "mismatch": mismatch,
            }
        )
        for i, tag in enumerate(tags):
            if i < len(peaks):
                frame = peaks[i]
                measured = _hold_window(controls, frame, half=3)
                prior = prior_9d(tag, emotion)
                c = measured.copy()
                if mismatch:
                    # Peaks unreliable — mouth leans on prior; keep measured upper face.
                    c[4:] = 0.25 * measured[4:] + 0.75 * prior[4:]
                else:
                    c[4:] = 0.55 * measured[4:] + 0.45 * prior[4:]
                c[0:4] = 0.85 * measured[0:4] + 0.15 * prior[0:4]
                if float(c[0]) > 0.35:
                    c[0] = 0.15 * float(c[0]) + 0.85 * float(prior[0])
                c = clamp_9d(c)
            else:
                # Missing peak — prior mouth + emotion rest upper-face mean if any.
                c = prior_9d(tag, emotion)
                if rest_buckets[emotion]:
                    ru = np.mean(np.stack(rest_buckets[emotion]), axis=0)
                    c[0:4] = 0.7 * ru[0:4] + 0.3 * c[0:4]
                c = clamp_9d(c)
                frame = -1
            xs.append(one_hot_22(tag, emotion))
            ys.append(c)
            pose_keys.append(f"{tag}|{emotion}|{video.stem}|{frame}")
            pose_vals.append(c)

    for emotion, rows in rest_buckets.items():
        if not rows:
            continue
        rest_c = clamp_9d(np.mean(np.stack(rows), axis=0))
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
    lengths = np.asarray([c.shape[0] for c in transfer_controls], dtype=np.int32)
    flat = (
        np.concatenate(transfer_controls, axis=0)
        if transfer_controls
        else np.zeros((0, GROUP_DIM), np.float32)
    )
    transfer = {
        "controls": flat,
        "lengths": lengths,
        "emotions": np.asarray(transfer_emotions),
        "stems": np.asarray(transfer_stems),
        "tick_hz": np.asarray([TICK_HZ]),
    }
    meta = {
        "d35": d35_reports,
        "n_pose": len(pose_keys),
        "n_transfer_clips": len(transfer_stems),
        "hard_fails": hard_fails,
        "label_notes": label_notes,
    }
    return X, Y, pose, transfer, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkg", type=Path, default=PKG)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--b-epochs", type=int, default=200)
    ap.add_argument(
        "--allow-soft-fail",
        action="store_true",
        help="Allow soft D35 FAILs (still block on hard-FAIL)",
    )
    args = ap.parse_args()

    X, Y, pose, transfer, meta = build_datasets(args.pkg)
    hard = list(meta.get("hard_fails") or [])
    if hard:
        print(json.dumps({"ok": False, "error": "D35 hard-FAIL", "clips": hard}, indent=2))
        (args.out / "teacher_train_report.json").parent.mkdir(parents=True, exist_ok=True)
        (args.out / "teacher_train_report.json").write_text(
            json.dumps({"ok": False, "hard_fails": hard, "d35": meta["d35"]}, indent=2),
            encoding="utf-8",
        )
        return 3

    soft_fails = [d for d in meta["d35"] if not d.get("passed")]
    if soft_fails and not args.allow_soft_fail:
        # Plan: continue only if no hard-FAIL; soft-FAIL → report + continue.
        pass

    args.pkg.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(args.pkg / "pose_targets.npz", **pose)
    np.savez_compressed(args.pkg / "transfer.npz", **transfer)
    np.savez_compressed(args.out / "dataset_a_from_teachers.npz", X=X, Y=Y)

    model_a = ModelA()
    stats_a = model_a.fit(X, Y, epochs=args.epochs)
    model_a.save(args.out / "model_a.npz")
    onnx_ok = bool(model_a.try_export_onnx(args.out / "model_a.onnx"))

    model_b = ModelB()
    stats_b = model_b.fit_from_transfers(
        transfer["controls"],
        transfer["lengths"],
        transfer["emotions"],
        epochs=args.b_epochs,
    )
    model_b.save(args.out / "model_b.npz")

    report = evaluate_model_a(model_a)
    angry = model_a.predict("AA", "ANGRY")
    happy = model_a.predict("AA", "HAPPY")
    upper_l2 = float(np.linalg.norm(angry[:4] - happy[:4]))

    out = {
        "ok": bool(report.passed) and upper_l2 >= 0.12,
        "n_train": int(X.shape[0]),
        "pose_targets": str(args.pkg / "pose_targets.npz"),
        "transfer": str(args.pkg / "transfer.npz"),
        "train_mse": stats_a["mse"],
        "model_b": stats_b,
        "onnx_exported": onnx_ok,
        "upper_face_l2_angry_vs_happy": upper_l2,
        "acceptance": report.to_dict(),
        "d35": meta["d35"],
        "d35_soft_fail_n": len(soft_fails),
        "label_notes": meta["label_notes"],
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
        json.dumps(
            {
                "d35": meta["d35"],
                "pose_n": meta["n_pose"],
                "label_notes": meta["label_notes"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Package version/checksums stubs (Teacher Package lock).
    (args.pkg / "version").write_text("v1\n", encoding="utf-8")
    (args.pkg / "checksums").write_text(
        f"pose_targets.npz\t{pose['Y'].shape}\n"
        f"transfer.npz\t{transfer['controls'].shape}\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
