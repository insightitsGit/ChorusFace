"""CLI entrypoints for VowelDesign Phase-1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chorusface-vowel")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train Model A (and save Model B stub)")
    p_train.add_argument("--out", type=Path, default=Path("output/worlds/tickfeed/vowel"))
    p_train.add_argument("--epochs", type=int, default=400)

    p_compose = sub.add_parser("compose", help="Compose utterance to PulseChunk")
    p_compose.add_argument("--json", type=Path, required=True)
    p_compose.add_argument("--out", type=Path, default=Path("pulsechunk.pls1"))
    p_compose.add_argument("--models", type=Path, default=None)

    p_accept = sub.add_parser("accept", help="Run F15 acceptance on Model A")
    p_accept.add_argument("--models", type=Path, required=True)

    p_d35 = sub.add_parser("d35", help="Run D35 landmark GO/NO-GO on one video")
    p_d35.add_argument("--video", type=Path, required=True)
    p_d35.add_argument("--out", type=Path, default=Path("output/d35"))

    p_w = sub.add_parser("author-w", help="Author expand matrix from region_catalog")
    p_w.add_argument("--catalog", type=Path, default=None)
    p_w.add_argument("--out", type=Path, default=Path("expand_matrix_v1.wexpand"))

    p_pkg = sub.add_parser("teacher-skeleton", help="Create Teacher Package dirs")
    p_pkg.add_argument("--root", type=Path, default=Path("output/teacher"))

    args = parser.parse_args(argv)

    if args.cmd == "train":
        from chorusface.vowel.acceptance import evaluate_model_a
        from chorusface.vowel.model_a import ModelA
        from chorusface.vowel.model_b import ModelB

        args.out.mkdir(parents=True, exist_ok=True)
        a = ModelA()
        stats = a.fit(epochs=args.epochs)
        a.save(args.out / "model_a.npz")
        try:
            a.try_export_onnx(args.out / "model_a.onnx")
        except Exception:
            pass
        ModelB().save(args.out / "model_b.npz")
        report = evaluate_model_a(a)
        (args.out / "acceptance.json").write_text(
            json.dumps({"train": stats, "acceptance": report.to_dict()}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"ok": True, "mse": stats["mse"], "passed": report.passed}))
        return 0 if report.passed else 2

    if args.cmd == "compose":
        from chorusface.vowel.pipeline import compose_utterance
        from chorusface.vowel.pulsechunk import encode_pulsechunk

        payload = json.loads(args.json.read_text(encoding="utf-8-sig"))
        result = compose_utterance(payload, model_dir=args.models)
        raw = encode_pulsechunk(result.chunk)
        args.out.write_bytes(raw)
        print(
            json.dumps(
                {
                    "ok": True,
                    "bytes": len(raw),
                    "n_ticks": result.chunk.n_ticks,
                    "n_words": len(result.chunk.word_slices),
                    "out": str(args.out),
                }
            )
        )
        return 0

    if args.cmd == "accept":
        from chorusface.vowel.acceptance import evaluate_model_a
        from chorusface.vowel.model_a import ModelA

        a = ModelA.load(Path(args.models) / "model_a.npz")
        report = evaluate_model_a(a)
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.passed else 2

    if args.cmd == "d35":
        from chorusface.vowel.teacher import run_d35

        metrics = run_d35(args.video, args.out)
        print(json.dumps(metrics.to_dict(), indent=2))
        return 0 if metrics.passed else 3

    if args.cmd == "author-w":
        from chorusface.vowel.expand import author_w_from_catalog, load_catalog, save_wexpand

        catalog = load_catalog(args.catalog) if args.catalog else {}
        W, cells = author_w_from_catalog(catalog)
        save_wexpand(args.out, W, cells)
        print(json.dumps({"ok": True, "cells": len(cells), "out": str(args.out)}))
        return 0

    if args.cmd == "teacher-skeleton":
        from chorusface.vowel.teacher import write_teacher_package_skeleton

        root = write_teacher_package_skeleton(args.root)
        print(json.dumps({"ok": True, "root": str(root)}))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
