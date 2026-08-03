"""Re-apply photographic blink onto HEAD mouth baseline (post-stash restore)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_frag() -> None:
    frag_path = ROOT / "src/aiface/shaders/avatar.frag"
    stash_path = ROOT / ".cursor_tmp/avatar_stash.frag"
    frag = frag_path.read_text(encoding="utf-8")
    stash = stash_path.read_text(encoding="utf-8")

    if "avatar_eye_closed_plate" not in frag:
        frag = frag.replace(
            "uniform vec4 avatar_expr_state;\n",
            "uniform vec4 avatar_expr_state;\n"
            "// Photographed nearly-closed eyes (LOOK region — mouth open.png analogue).\n"
            "uniform sampler2D avatar_eye_closed_plate;\n"
            "uniform int avatar_eye_closed_ready;\n",
            1,
        )

    m = re.search(
        r"(        // L09: eye LOOK region[\s\S]*?"
        r"        // L10: brow raise[\s\S]*?)"
        r"(    \} else \{\n)",
        stash,
    )
    if not m:
        raise SystemExit("could not find stash L09 block")
    new_l09 = m.group(1)

    m2 = re.search(
        r"        // L09: eyes[\s\S]*?"
        r"(        // L10: brow raise[\s\S]*?)"
        r"(    \} else \{\n)",
        frag,
    )
    if not m2:
        raise SystemExit("could not find current L09 block")
    frag = frag[: m2.start()] + new_l09 + m2.group(2) + frag[m2.end() :]
    frag_path.write_text(frag, encoding="utf-8")
    print("avatar.frag: photographic L09 applied")


def patch_capture() -> None:
    path = ROOT / "src/aiface/capture.py"
    text = path.read_text(encoding="utf-8")
    stash = (ROOT / ".cursor_tmp/capture_stash.py").read_text(encoding="utf-8")

    if "EYES_CLOSED_PLATE_NAME" not in text:
        text = text.replace(
            'OPEN_PLATE_NAME: Final = "open.png"\n',
            'OPEN_PLATE_NAME: Final = "open.png"\n'
            'EYES_CLOSED_PLATE_NAME: Final = "eyes_closed.png"\n',
            1,
        )

    if "def default_eyes_closed_plate_path" not in text:
        insert = '''

def default_eyes_closed_plate_path(world: str | Path) -> Path:
    """Photographed nearly-closed eyes LOOK plate (blink ownership)."""
    return Path(world).with_name(EYES_CLOSED_PLATE_NAME)

'''
        text = text.replace(
            "def default_capture_meta_path(world: str | Path) -> Path:\n",
            insert + "def default_capture_meta_path(world: str | Path) -> Path:\n",
            1,
        )

    if "def build_eye_lid_matte" not in text:
        m = re.search(
            r"\ndef build_eye_lid_matte\([\s\S]*?\n    return alpha\.astype\(np\.float32\)\n"
            r"\n\nEYE_ANCHORS_NAME: Final = \"eye_anchors\.json\"\n"
            r"\n\ndef default_eye_anchors_path\([\s\S]*?\n    return root / EYE_ANCHORS_NAME\n"
            r"\n\ndef write_eye_anchors\([\s\S]*?\n    return destination\n",
            stash,
        )
        if not m:
            raise SystemExit("could not extract eye lid helpers from stash capture")
        # Insert before match_plate_to_reference
        anchor = "\ndef match_plate_to_reference(\n"
        if anchor not in text:
            raise SystemExit("match_plate_to_reference missing in capture.py")
        text = text.replace(anchor, m.group(0) + "\n" + anchor, 1)

    # allow_closed_eyes on analyze_frame
    if "allow_closed_eyes" not in text:
        text = text.replace(
            "    min_sharpness: float = MIN_SHARPNESS,\n"
            "    report: RejectReport | None = None,\n"
            ") -> FrameSample | None:\n"
            '    """Normalize one frame and score expression; return ``None`` if rejected."""\n',
            "    min_sharpness: float = MIN_SHARPNESS,\n"
            "    allow_closed_eyes: bool = False,\n"
            "    report: RejectReport | None = None,\n"
            ") -> FrameSample | None:\n"
            '    """Normalize one frame and score expression; return ``None`` if rejected.\n'
            "\n"
            "    ``allow_closed_eyes`` keeps blink frames for ``eyes_closed.png`` (normal\n"
            "    capture rejects low eye aperture).\n"
            '    """\n',
            1,
        )
        text = text.replace(
            "    if (\n"
            "        aperture < MIN_EYE_APERTURE\n"
            '        and not landmarks.method.startswith("canonical")\n'
            "    ):\n"
            '        drop("eyes")\n',
            "    if (\n"
            "        aperture < MIN_EYE_APERTURE\n"
            "        and not allow_closed_eyes\n"
            '        and not landmarks.method.startswith("canonical")\n'
            "    ):\n"
            '        drop("eyes")\n',
            1,
        )

    # __all__ exports
    for name in (
        "EYES_CLOSED_PLATE_NAME",
        "default_eyes_closed_plate_path",
        "build_eye_lid_matte",
        "default_eye_anchors_path",
        "write_eye_anchors",
    ):
        if f'"{name}"' not in text and f"'{name}'" not in text:
            text = text.replace(
                '    "OPEN_PLATE_NAME",\n',
                f'    "OPEN_PLATE_NAME",\n    "{name}",\n',
                1,
            )

    path.write_text(text, encoding="utf-8")
    print("capture.py: blink helpers applied")


def patch_app() -> None:
    path = ROOT / "src/aiface/app.py"
    text = path.read_text(encoding="utf-8")
    stash = (ROOT / ".cursor_tmp/app_stash.py").read_text(encoding="utf-8")

    if "_tickfeed_lid_amt" not in text:
        # Find a good insertion near other tickfeed state
        needle = "        self._tickfeed_look_authority = False\n"
        if needle in text:
            text = text.replace(
                needle,
                needle
                + "        self._tickfeed_lid_amt = 1.0\n"
                + "        self._tickfeed_lid_teacher = False\n",
                1,
            )
        else:
            # fallback after plate atlas init block
            text = text.replace(
                "        self._load_plate_atlas_textures()\n",
                "        self._tickfeed_lid_amt = 1.0\n"
                "        self._tickfeed_lid_teacher = False\n"
                "        self._load_plate_atlas_textures()\n",
                1,
            )

    if "_create_eyes_closed_plate_texture" not in text:
        m = re.search(
            r"\n    def _create_eyes_closed_plate_texture\(self\) -> tuple\[moderngl\.Texture, bool\]:\n"
            r"(?:.*\n)*?"
            r"        return self\._upload_rgba_texture\(rgba\), True\n",
            stash,
        )
        if not m:
            raise SystemExit("missing eyes closed texture loader in stash app")
        text = text.replace(
            "        self._avatar_expr_plate_texture = self._create_expression_catalog_texture()\n",
            "        self._avatar_expr_plate_texture = self._create_expression_catalog_texture()\n"
            "        (\n"
            "            self._avatar_eye_closed_texture,\n"
            "            self._use_eye_closed_plate,\n"
            "        ) = self._create_eyes_closed_plate_texture()\n",
            1,
        )
        # Insert method before _active_emotion or after expression catalog texture
        insert_at = text.find("\n    def _active_emotion(self) -> str:\n")
        if insert_at < 0:
            insert_at = text.find("\n    def _sync_expression_from_emotion(self) -> None:\n")
        if insert_at < 0:
            raise SystemExit("no insertion point for eyes closed loader")
        text = text[:insert_at] + m.group(0) + text[insert_at:]

    if "_eyes_from_anchors_file" not in text:
        m = re.search(
            r"\n    def _eyes_from_anchors_file\(\n"
            r"(?:.*\n)*?"
            r"        return to_grid\(lx, ly\), to_grid\(rx, ry\), \(half_w, half_h\)\n"
            r"\n    def _eyes_from_tissue_aperture\(\n"
            r"(?:.*\n)*?"
            r"        return \(lcx, lcy\), \(rcx, rcy\), \(half_w, half_h\)\n",
            stash,
        )
        if not m:
            raise SystemExit("missing eye anchor helpers in stash app")
        insert_at = text.find("\n    def _derive_face_geometry(self) -> None:\n")
        if insert_at < 0:
            raise SystemExit("no _derive_face_geometry")
        text = text[:insert_at] + m.group(0) + text[insert_at:]

        # Replace eye centre resolution block to prefer anchors
        old = (
            "        # Prefer seed-measured eyes (Path 1). Part anchors are diagnostic only —\n"
            "        # they used to fight the definition UV and jumble the lids.\n"
            "        measured = self._seed_eye_centers_grid()\n"
            "        if measured is not None:\n"
            "            left, right = measured\n"
            "        else:\n"
            '            left = to_grid(eye_positions.get("left", [0.30, 0.472]))\n'
            '            right = to_grid(eye_positions.get("right", [0.70, 0.472]))\n'
        )
        new = (
            "        # 1) eye_anchors.json from blink bake (landmarker sockets)\n"
            "        # 2) tissue.a only when L/R share a horizontal band (rejects cheek/nose)\n"
            "        # 3) seed landmarks 4) definition UV\n"
            "        anchor_eyes = self._eyes_from_anchors_file()\n"
            "        tissue_eyes = self._eyes_from_tissue_aperture()\n"
            "        measured = self._seed_eye_centers_grid()\n"
            "        if anchor_eyes is not None:\n"
            "            left, right, (half_w, half_h) = anchor_eyes\n"
            "            self._eye_shape = (\n"
            "                float(half_w),\n"
            "                float(half_h),\n"
            '                float(eye_config.get("gaze_travel_cells", 2.6)),\n'
            "                0.0,\n"
            "            )\n"
            "            print(\n"
            '                f"Eye sockets from anchors: "\n'
            '                f"L=({left[0]:.1f},{left[1]:.1f}) R=({right[0]:.1f},{right[1]:.1f}) "\n'
            '                f"half=({half_w:.1f},{half_h:.1f})"\n'
            "            )\n"
            "        elif tissue_eyes is not None:\n"
            "            left, right, (half_w, half_h) = tissue_eyes\n"
            "            self._eye_shape = (\n"
            "                float(half_w),\n"
            "                float(half_h),\n"
            '                float(eye_config.get("gaze_travel_cells", 2.6)),\n'
            "                0.0,\n"
            "            )\n"
            "            print(\n"
            '                f"Eye aperture from tissue: "\n'
            '                f"L=({left[0]:.1f},{left[1]:.1f}) R=({right[0]:.1f},{right[1]:.1f}) "\n'
            '                f"half=({half_w:.1f},{half_h:.1f})"\n'
            "            )\n"
            "        elif measured is not None:\n"
            "            left, right = measured\n"
            "        else:\n"
            '            left = to_grid(eye_positions.get("left", [0.30, 0.472]))\n'
            '            right = to_grid(eye_positions.get("right", [0.70, 0.472]))\n'
        )
        if old not in text:
            raise SystemExit("eye centre block not found for patch")
        text = text.replace(old, new, 1)

    # Bind eye closed plate + lid teacher blink in uniforms
    if "avatar_eye_closed_plate" not in text or "avatar_eye_closed_ready" not in text:
        old_bind = (
            "        self._avatar_expr_plate_texture.use(location=8)\n"
            "        program[\"avatar_expr_plate\"].value = 8\n"
            "        # Blink owns the aperture — do not upload widen/brow fight during close.\n"
            "        blink_close = float(\n"
            "            1.0 - min(float(state.lid_left), float(state.lid_right))\n"
            "        )\n"
        )
        new_bind = (
            "        self._avatar_expr_plate_texture.use(location=8)\n"
            "        program[\"avatar_expr_plate\"].value = 8\n"
            "        self._avatar_eye_closed_texture.use(location=10)\n"
            '        program["avatar_eye_closed_plate"].value = 10\n'
            '        program["avatar_eye_closed_ready"].value = (\n'
            "            1 if self._use_eye_closed_plate else 0\n"
            "        )\n"
            "        # Blink owns the aperture — do not upload widen/brow fight during close.\n"
            "        if bool(getattr(self, \"_tickfeed_lid_teacher\", False)):\n"
            "            blink_close = float(\n"
            "                1.0 - max(0.0, min(1.0, float(self._tickfeed_lid_amt)))\n"
            "            )\n"
            "        else:\n"
            "            blink_close = float(\n"
            "                1.0 - min(float(state.lid_left), float(state.lid_right))\n"
            "            )\n"
        )
        if old_bind not in text:
            raise SystemExit("expr plate bind block missing")
        text = text.replace(old_bind, new_bind, 1)

    # eye_state blink from lid teacher
    old_eye = (
        "        program[\"avatar_eye_state\"].value = (\n"
        "            float(state.gaze_x),\n"
        "            float(state.gaze_y),\n"
        "            float(state.pupil),\n"
        "            float(1.0 - min(state.lid_left, state.lid_right)),\n"
        "        )\n"
    )
    new_eye = (
        "        # LOOK lid_amt (1=open) owns blink when teacher present; else EyeSystem.\n"
        "        lid_local = float(min(state.lid_left, state.lid_right))\n"
        "        if self._tickfeed_look_authority and bool(\n"
        '            getattr(self, "_tickfeed_lid_teacher", False)\n'
        "        ):\n"
        '            lid_amt = float(getattr(self, "_tickfeed_lid_amt", 1.0) or 1.0)\n'
        "        else:\n"
        "            lid_amt = lid_local\n"
        "        blink_amt = float(1.0 - max(0.0, min(1.0, lid_amt)))\n"
        '        program["avatar_eye_state"].value = (\n'
        "            float(state.gaze_x),\n"
        "            float(state.gaze_y),\n"
        "            float(state.pupil),\n"
        "            blink_amt,\n"
        "        )\n"
    )
    if old_eye not in text:
        raise SystemExit("avatar_eye_state block missing")
    text = text.replace(old_eye, new_eye, 1)

    # labels → lid teacher
    if "self._tickfeed_lid_amt = lid" not in text:
        old_lab = (
            "        brow = float(getattr(labels, \"brow_amt\", 0.0) or 0.0)\n"
            "        # field_only: keep label openness for debug/status, but force plates closed\n"
        )
        new_lab = (
            "        brow = float(getattr(labels, \"brow_amt\", 0.0) or 0.0)\n"
            "        lid = float(getattr(labels, \"lid_amt\", 1.0) or 1.0)\n"
            "        self._tickfeed_lid_amt = lid\n"
            "        # Teacher present when take actually varies lids (not always 1.0).\n"
            "        self._tickfeed_lid_teacher = lid < 0.98\n"
            "        # field_only: keep label openness for debug/status, but force plates closed\n"
        )
        if old_lab not in text:
            raise SystemExit("label brow block missing")
        text = text.replace(old_lab, new_lab, 1)

    if "self._avatar_eye_closed_texture.release()" not in text:
        text = text.replace(
            "        self._avatar_expr_plate_texture.release()\n",
            "        self._avatar_expr_plate_texture.release()\n"
            "        self._avatar_eye_closed_texture.release()\n",
            1,
        )

    path.write_text(text, encoding="utf-8")
    print("app.py: blink plate + lid teacher applied (mouth left at HEAD)")


def main() -> int:
    patch_frag()
    patch_capture()
    patch_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
