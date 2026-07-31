"""Model-data verification for a digested avatar world (AMIN step 14, data half).

Checks every artifact the pipeline wrote — capture selection (smile / open
detection), display plates, region objects, live-vector dataset + model,
condition maps, and the GPU display recipe — and scores each PASS / WARN /
FAIL so "did digestion work?" is a number, not a feeling.

Usage:
    python scripts/verify_world.py [--world output/worlds/avatar]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str) -> None:
        self.rows.append((status, name, detail))
        line = f"[{status}] {name}: {detail}"
        # Windows consoles often run cp1252 — never die on a symbol.
        print(line.encode("ascii", errors="replace").decode("ascii"))

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status, _, _ in self.rows)

    def summary(self) -> str:
        counts = {status: 0 for status in (PASS, WARN, FAIL)}
        for status, _, _ in self.rows:
            counts[status] += 1
        return (
            f"{counts[PASS]} pass / {counts[WARN]} warn / {counts[FAIL]} fail "
            f"({len(self.rows)} checks)"
        )


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def check_artifacts(report: Report, world_dir: Path) -> None:
    required = (
        "avatar_face.bds",
        "source_face.png",
        "open.png",
        "smile.png",
        "plate_atlas.json",
        "expression_catalog.json",
        "capture_meta.json",
        "region_catalog.json",
        "condition_maps.json",
        "gpu_display_recipe.json",
        "live_vector_dataset.npz",
        "live_vector_model.joblib",
        "live_vector_model.meta.json",
    )
    missing = [name for name in required if not (world_dir / name).is_file()]
    if missing:
        report.add(FAIL, "artifacts", f"missing {', '.join(missing)}")
    else:
        report.add(PASS, "artifacts", f"all {len(required)} pipeline files present")


def check_display_resolution(report: Report, world_dir: Path) -> None:
    """AMIN step 11 — plates must carry more pixels than the 256² field."""
    try:
        from PIL import Image
    except ImportError:
        report.add(WARN, "display-res", "PIL unavailable — skipped")
        return
    names = ["source_face.png", "open.png", "smile.png", "surprise.png"]
    names += [f"plates/{p.name}" for p in sorted((world_dir / "plates").glob("plate_*.png"))]
    sizes: dict[str, tuple[int, int]] = {}
    for name in names:
        path = world_dir / name
        if not path.is_file():
            continue
        with Image.open(path) as image:
            sizes[name] = image.size
    if not sizes:
        report.add(FAIL, "display-res", "no display assets found")
        return
    low = {name: size for name, size in sizes.items() if max(size) < 512}
    smallest = min(max(size) for size in sizes.values())
    largest = max(max(size) for size in sizes.values())
    if not low:
        report.add(
            PASS,
            "display-res",
            f"{len(sizes)} display assets at {smallest}–{largest}px (≥512, step 11 active)",
        )
    elif len(low) == len(sizes):
        report.add(
            WARN,
            "display-res",
            f"all {len(sizes)} assets ≤{largest}px — re-digest to get step 11 hi-res plates",
        )
    else:
        report.add(
            WARN,
            "display-res",
            f"{len(low)}/{len(sizes)} assets under 512px: {', '.join(sorted(low))}",
        )


def check_capture_selection(report: Report, world_dir: Path) -> None:
    """Did smile/open detection pick genuinely different expressions?"""
    catalog = load_json(world_dir / "expression_catalog.json")
    if not catalog or "roles" not in catalog:
        report.add(FAIL, "capture-selection", "expression_catalog.json unreadable")
        return
    roles = catalog["roles"]

    def metric(role: str, key: str) -> float:
        return float((roles.get(role) or {}).get(key, 0.0))

    rest_open = metric("rest", "mouth_open")
    rest_smile = metric("rest", "smile_width")
    smile_delta = metric("smile", "smile_width") - rest_smile
    open_delta = metric("open", "mouth_open") - rest_open
    frames = {name: (roles.get(name) or {}).get("frame_index") for name in roles}
    distinct = len({v for v in frames.values() if v is not None})

    detail = (
        f"smile Δwidth={smile_delta:+.3f}, open Δmouth={open_delta:+.3f}, "
        f"rest mouth_open={rest_open:.3f}, {distinct} distinct role frames"
    )
    if smile_delta <= 0.0 or open_delta <= 0.0:
        report.add(FAIL, "capture-selection", f"role ordering broken — {detail}")
    elif smile_delta < 0.015 or open_delta < 0.035 or distinct < 3:
        report.add(WARN, "capture-selection", f"weak expression contrast — {detail}")
    else:
        report.add(PASS, "capture-selection", detail)

    meta = load_json(world_dir / "capture_meta.json") or {}
    teeth = float((meta.get("selection") or {}).get("rest_teeth", 0.0))
    if teeth > 0.12:
        report.add(
            WARN,
            "rest-frame",
            f"rest frame shows teeth ({teeth:.2f}) — identity is not a closed mouth",
        )
    else:
        report.add(PASS, "rest-frame", f"rest teeth score {teeth:.2f} (closed mouth)")


def check_plate_atlas(report: Report, world_dir: Path) -> None:
    atlas = load_json(world_dir / "plate_atlas.json")
    if not atlas or not atlas.get("plates"):
        report.add(FAIL, "plate-atlas", "plate_atlas.json missing or empty")
        return
    openness = sorted(float(p.get("openness", 0.0)) for p in atlas["plates"])
    span = openness[-1] - openness[0]
    mapping = atlas.get("viseme_to_plate") or {}
    detail = (
        f"{len(openness)} plates, openness {openness[0]:.2f}–{openness[-1]:.2f}, "
        f"{len(mapping)} viseme→plate"
    )
    if not mapping:
        report.add(
            WARN,
            "plate-atlas",
            f"missing viseme_to_plate (step 13) — {detail}",
        )
    elif len(openness) < 4:
        report.add(WARN, "plate-atlas", f"sparse plate bank — {detail}")
    else:
        report.add(PASS, "plate-atlas", detail)

    recipe = load_json(world_dir / "gpu_display_recipe.json") or {}
    knobs = recipe.get("knobs") or {}
    sharpness = float(knobs.get("plate_sharpness", 0.0))
    if sharpness >= 0.75:
        report.add(
            PASS,
            "plate-hard-snap",
            f"plate_sharpness={sharpness:.2f} (≥0.75 hard snap)",
        )
    else:
        report.add(
            WARN,
            "plate-hard-snap",
            f"plate_sharpness={sharpness:.2f} — mid-blend ghosts likely",
        )


def check_world_and_regions(report: Report, world_dir: Path) -> None:
    """Master Lock statistics + the digested mouth object address."""
    from aiface.runtime.bds import HUMAN_LOCK_CHANNEL, load_bds

    try:
        header, grid = load_bds(world_dir / "avatar_face.bds")
    except Exception as exc:  # noqa: BLE001 — verification must report, not crash
        report.add(FAIL, "world", f"avatar_face.bds unreadable ({exc})")
        return
    lock = np.asarray(grid[..., HUMAN_LOCK_CHANNEL])
    locked_frac = float((lock >= 0.5).mean())
    unlocked_cells = int((lock < 0.5).sum())
    if 0.05 <= locked_frac <= 0.98 and unlocked_cells >= 64:
        report.add(
            PASS,
            "master-lock",
            f"{locked_frac:.1%} cells locked, {unlocked_cells} unlocked (mouth can move)",
        )
    else:
        report.add(
            FAIL,
            "master-lock",
            f"lock layout degenerate — {locked_frac:.1%} locked, {unlocked_cells} unlocked",
        )

    # Seed's mouth address (image y-down → grid y-up).
    seed_meta = (header.get("application_metadata") or {}).get("avatar_seed") or {}
    mouth = seed_meta.get("mouth_center_image")
    grid_h = grid.shape[0]
    seed_xy = None
    if mouth:
        seed_xy = (float(mouth["x"]), grid_h - float(mouth["y"]))

    catalog = load_json(world_dir / "region_catalog.json")
    if not catalog:
        report.add(FAIL, "region-objects", "region_catalog.json unreadable")
        return
    regions = catalog.get("regions") or []
    mouth_regions = [r for r in regions if r.get("name") == "mouth_unlocked"]
    if not mouth_regions:
        report.add(
            FAIL,
            "region-objects",
            f"{len(regions)} regions but no mouth_unlocked object — runtime has no address",
        )
        return
    best = max(mouth_regions, key=lambda r: int(r.get("cell_count", 0)))
    cells = best.get("cells") or best.get("cells_sample") or []
    count = int(best.get("cell_count", len(cells)))
    if not cells:
        report.add(WARN, "region-objects", "mouth object has no cell samples")
        return
    cx = float(np.mean([c[0] for c in cells]))
    cy = float(np.mean([c[1] for c in cells]))
    detail = f"{len(regions)} regions; mouth object {count} cells @ ({cx:.1f}, {cy:.1f})"
    if seed_xy is not None:
        distance = float(np.hypot(cx - seed_xy[0], cy - seed_xy[1]))
        detail += f", {distance:.1f} cells from seed mouth {seed_xy[0]:.0f},{seed_xy[1]:.0f}"
        if distance > 40.0:
            report.add(FAIL, "region-objects", f"address drift — {detail}")
            return
    report.add(PASS, "region-objects", detail)


def check_live_vector_data(report: Report, world_dir: Path) -> None:
    """Dataset sanity: do the labels move, and do they follow the audio?"""
    try:
        data = np.load(world_dir / "live_vector_dataset.npz")
    except (OSError, ValueError) as exc:
        report.add(FAIL, "live-dataset", f"unreadable ({exc})")
        return
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    names = [str(n) for n in data.get("control_names", ["openness_n", "jaw_n", "width_n"])]
    if X.shape[0] != y.shape[0] or X.shape[0] < 40:
        report.add(FAIL, "live-dataset", f"too small / misaligned: X{X.shape} y{y.shape}")
        return
    out_of_range = float(((y < -0.001) | (y > 1.001)).mean())
    spans = y.max(axis=0) - y.min(axis=0)
    span_txt = ", ".join(f"{n}={s:.2f}" for n, s in zip(names, spans))
    # Best |correlation| between any audio feature and openness — the video's
    # mouth must actually follow its own soundtrack or the source is suspect.
    openness = y[:, 0]
    corr = 0.0
    if openness.std() > 1e-6:
        for column in range(X.shape[1]):
            feature = X[:, column]
            if feature.std() <= 1e-9:
                continue
            corr = max(corr, abs(float(np.corrcoef(feature, openness)[0, 1])))
    detail = (
        f"{X.shape[0]} samples, {X.shape[1]} features; label spans {span_txt}; "
        f"out-of-range {out_of_range:.1%}; best |audio↔openness corr| {corr:.2f}"
    )
    if out_of_range > 0.01 or spans[0] < 0.10:
        report.add(FAIL, "live-dataset", detail)
    elif corr < 0.25 or spans.min() < 0.05:
        report.add(WARN, "live-dataset", f"weak signal — {detail}")
    else:
        report.add(PASS, "live-dataset", detail)


def check_model(report: Report, world_dir: Path) -> None:
    meta = load_json(world_dir / "live_vector_model.meta.json")
    if not meta:
        report.add(FAIL, "live-model", "live_vector_model.meta.json unreadable")
        return
    val = float(meta.get("val_mae", 1.0))
    baseline = float(meta.get("baseline_mean_mae", 0.0))
    beats = bool(meta.get("beats_baseline", False))
    detail = (
        f"val MAE {val:.4f} vs baseline {baseline:.4f} "
        f"({meta.get('n_train', '?')} train / {meta.get('n_val', '?')} val)"
    )
    if beats and val < baseline:
        report.add(PASS, "live-model", detail)
    else:
        report.add(FAIL, "live-model", f"model does not beat baseline — {detail}")
    try:
        import joblib

        payload = joblib.load(world_dir / "live_vector_model.joblib")
        model = payload.get("pipeline") if isinstance(payload, dict) else payload
        n_features = int(getattr(model, "n_features_in_", 0))
        if n_features <= 0:
            n_features = int(np.load(world_dir / "live_vector_dataset.npz")["X"].shape[1])
        probe = model.predict(np.zeros((1, n_features)))
        report.add(
            PASS,
            "live-model-load",
            f"pipeline loads and predicts shape {tuple(np.shape(probe))}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add(FAIL, "live-model-load", f"cannot load/predict ({exc})")


def check_condition_maps_and_recipe(report: Report, world_dir: Path) -> None:
    maps = load_json(world_dir / "condition_maps.json")
    table = (maps or {}).get("viseme_table") or {}
    with_jaw = sum(1 for entry in table.values() if isinstance(entry, dict) and "jaw" in entry)
    if with_jaw >= 8:
        report.add(PASS, "condition-maps", f"{len(table)} visemes, {with_jaw} with jaw targets")
    else:
        report.add(FAIL, "condition-maps", f"jaw table incomplete ({with_jaw} entries)")

    recipe = load_json(world_dir / "gpu_display_recipe.json")
    knobs = (recipe or {}).get("knobs") or {}
    needed = {"open_jaw_full", "atlas_strength", "field_warp_gain", "plate_sharpness"}
    missing = sorted(needed - set(knobs))
    if not recipe or str(recipe.get("schema", "")) != "aiface.gpu_display_recipe.v2":
        report.add(FAIL, "display-recipe", "wrong/missing schema — re-run amin_train")
    elif missing:
        report.add(WARN, "display-recipe", f"knobs missing {missing} — re-run amin_train")
    else:
        report.add(
            PASS,
            "display-recipe",
            f"{len(knobs)} knobs incl. field_warp_gain={knobs['field_warp_gain']}, "
            f"plate_sharpness={knobs['plate_sharpness']}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--world",
        type=Path,
        default=REPO / "output" / "worlds" / "avatar",
        help="World directory (contains avatar_face.bds)",
    )
    options = parser.parse_args()
    world_dir = options.world if options.world.is_dir() else options.world.parent
    print(f"Verifying world data: {world_dir}\n")

    report = Report()
    check_artifacts(report, world_dir)
    check_display_resolution(report, world_dir)
    check_capture_selection(report, world_dir)
    check_plate_atlas(report, world_dir)
    check_world_and_regions(report, world_dir)
    check_live_vector_data(report, world_dir)
    check_model(report, world_dir)
    check_condition_maps_and_recipe(report, world_dir)

    print(f"\nResult: {report.summary()}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
