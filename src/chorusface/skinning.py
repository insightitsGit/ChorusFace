"""Continuous skin deformation: tissue maps, muscle packing, CPU reference.

The renderer does not move discrete cut-out pieces. It evaluates one smooth
displacement field over the whole portrait and inverse-warps the photograph
through it, so neighbouring skin stays neighbouring and no piece boundary can
tear. Each muscle contributes a compactly-supported radial basis function
centred on its anchor:

.. math::

    D(x) = \\sum_m a_m \\, t_m \\, \\hat{f}_m \\, w\\!\\left(\\frac{|x - p_m|}{r_m}\\right)

with :math:`a_m` the activation, :math:`t_m` the travel in grid cells,
:math:`\\hat{f}_m` the unit force direction, and :math:`w` a Wendland C² kernel.
Because :math:`w` is twice differentiable and vanishes at its support boundary,
the sum is C² everywhere — smoothness is a property of the representation
rather than something that has to be tuned.

Three things modulate that sum, all baked into :class:`TissueMaps`:

``mobility``
    How freely skin slides over what is underneath it. Bone-backed tissue (nose
    bridge, brow ridge, the silhouette) approaches zero; lips and cheeks
    approach one. This is what replaces a hard lock boundary with a gradient,
    and it is why tissue near the skull damps out instead of shearing.

``mouth_side`` / ``mouth_slit``
    The lip parting is the one place on a face where tissue genuinely separates,
    so muscles may be gated to one side of it. ``mouth_slit`` fades the gate out
    past the mouth corners, where the skin is continuous again and gating would
    only introduce the seam this design exists to avoid.

``eye_aperture``
    Where the eyeball shows through, so the lid wipe and the iris have baked
    geometry to work against instead of hard-coded pixel radii.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import numpy.typing as npt

from chorusface.biomechanics.muscles import Muscle, MuscleRegistry

# Uniform array capacity in avatar.frag. Raising it means editing the shader
# constant too; test_shader_contract keeps the two in step.
MAX_ACTIVE_MUSCLES: Final = 48
# Below this an activation moves skin by well under a tenth of a cell.
ACTIVATION_EPSILON: Final = 0.004

TISSUE_MOBILITY: Final = 0
TISSUE_MOUTH_SIDE: Final = 1
TISSUE_MOUTH_SLIT: Final = 2
TISSUE_EYE_APERTURE: Final = 3

# Soft anatomical zones: (centre_u, centre_v, radius_u, radius_v). Each is a
# plateau with a smooth skirt, and mobility is the strongest zone covering a
# cell. Composing overlapping bumps rather than painting bands is what keeps
# the map free of the steps that would read as creases in otherwise smooth skin.
MOBILITY_ZONES: Final[dict[str, tuple[float, float, float, float]]] = {
    "forehead": (0.50, 0.29, 0.40, 0.13),
    "brow": (0.50, 0.375, 0.38, 0.075),
    # Stopping short of the zygomatic arch on purpose: the skin over it is
    # thin and pinned to bone, and calling it mobile only invites the warp to
    # slide the ear line around.
    "cheek_left": (0.325, 0.665, 0.185, 0.135),
    "cheek_right": (0.675, 0.665, 0.185, 0.135),
    "perioral": (0.50, 0.780, 0.32, 0.115),
    "jaw_left": (0.31, 0.845, 0.19, 0.115),
    "jaw_right": (0.69, 0.845, 0.19, 0.115),
    "chin": (0.50, 0.895, 0.21, 0.105),
    "neck": (0.50, 1.020, 0.30, 0.110),
}

# Peak mobility inside each zone, plus the structures that suppress it.
DEFAULT_MOBILITY: Final[dict[str, float]] = {
    "skull": 0.06,
    "forehead": 0.52,
    "brow": 0.62,
    "eye": 0.22,
    "nose": 0.10,
    "cheek_left": 0.74,
    "cheek_right": 0.74,
    "perioral": 1.0,
    "jaw_left": 0.62,
    "jaw_right": 0.62,
    "chin": 0.80,
    "neck": 0.38,
}
# Bone the mandible does not carry: the temples, zygomatic arch, and hairline
# fall back to this because no zone reaches them.
BONE_FLOOR: Final = 0.05


@dataclass(frozen=True, slots=True)
class TissueMaps:
    """Per-cell deformation properties, y-up to match the world grid."""

    rgba: npt.NDArray[np.float32]

    @property
    def mobility(self) -> npt.NDArray[np.float32]:
        return self.rgba[..., TISSUE_MOBILITY]

    @property
    def mouth_side(self) -> npt.NDArray[np.float32]:
        return self.rgba[..., TISSUE_MOUTH_SIDE]

    @property
    def mouth_slit(self) -> npt.NDArray[np.float32]:
        return self.rgba[..., TISSUE_MOUTH_SLIT]

    @property
    def eye_aperture(self) -> npt.NDArray[np.float32]:
        return self.rgba[..., TISSUE_EYE_APERTURE]


def _smoothstep(
    edge0: float,
    edge1: float,
    value: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    span = max(edge1 - edge0, 1e-9)
    t = np.clip((value - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _box_blur(
    field: npt.NDArray[np.float64],
    radius: int,
    passes: int = 3,
) -> npt.NDArray[np.float64]:
    """Separable box blur; three passes approximate a Gaussian closely enough.

    Hand-rolled so the tissue bake does not drag OpenCV or SciPy into the
    runtime import path.
    """
    if radius < 1:
        return field
    result = np.asarray(field, dtype=np.float64)
    window = 2 * radius + 1
    for _ in range(max(passes, 1)):
        for axis in (0, 1):
            padded = np.pad(result, [(radius + 1, radius) if a == axis else (0, 0)
                                     for a in (0, 1)], mode="edge")
            cumulative = np.cumsum(padded, axis=axis)
            upper = np.take(cumulative, np.arange(window, cumulative.shape[axis]),
                            axis=axis)
            lower = np.take(cumulative, np.arange(0, cumulative.shape[axis] - window),
                            axis=axis)
            result = (upper - lower) / float(window)
    return result


def build_tissue_maps(
    height: int,
    width: int,
    face: Any,
    definition: Mapping[str, Any] | None = None,
    *,
    landmarks: Any | None = None,
) -> TissueMaps:
    """Bake mobility, lip-parting, and eye geometry for one portrait.

    ``face`` is an :class:`chorusface.seed.FaceBox`. When ``landmarks`` are provided
    (from :mod:`chorusface.landmarks`), eyes and mouth follow the measured centres
    instead of the definition's default UV — that is what keeps a real photo
    registered after Path 1 seeding.
    """
    payload: Mapping[str, Any] = definition or {}
    tissue_config = payload.get("tissue_mobility", {})
    regions = {**DEFAULT_MOBILITY, **dict(tissue_config.get("regions", {}))}
    default_mobility = float(tissue_config.get("default", 0.55))
    background = float(tissue_config.get("background", 0.0))
    smoothing = float(tissue_config.get("smoothing_cells", 2.5))

    mouth_center = list(payload.get("mouth_center", [0.50, 0.78]))
    mouth_line = payload.get("mouth_line", {})
    half_width = float(mouth_line.get("half_width", 0.20))
    softness_cells = float(mouth_line.get("softness_cells", 1.6))
    corner_taper = float(mouth_line.get("corner_taper", 0.35))
    eye_positions = payload.get("eye_positions", {})
    eyes_uv = (
        tuple(eye_positions.get("left", [0.30, 0.472])),
        tuple(eye_positions.get("right", [0.70, 0.472])),
    )
    if landmarks is not None:
        eyes_uv = landmarks.eye_uv()
        mouth_center = list(landmarks.mouth_uv())
    eye_shape = payload.get("eye_shape", {})
    aperture_u = float(eye_shape.get("aperture_width", 0.13))
    aperture_v = float(eye_shape.get("aperture_height", 0.075))
    globe_u = float(eye_shape.get("half_width", 0.10))
    globe_v = float(eye_shape.get("half_height", 0.040))

    yy, xx = np.mgrid[0:height, 0:width]
    fx0, fy0 = float(face.x), float(face.y)
    fw, fh = float(max(face.width, 1)), float(max(face.height, 1))
    u = (xx.astype(np.float64) - fx0) / fw
    v = (yy.astype(np.float64) - fy0) / fh

    # Same silhouette the seed uses for the Master Lock, so the two agree.
    radial = np.sqrt(((u - 0.5) / 0.48) ** 2 + ((v - 0.50) / 0.56) ** 2)
    inside = radial <= 1.0

    zones = {**MOBILITY_ZONES, **dict(tissue_config.get("zones", {}))}
    floor = float(tissue_config.get("bone_floor", BONE_FLOOR))

    # Strongest zone wins. Each contributes a plateau with a smooth skirt, so
    # the composite has no steps for the warp to turn into creases.
    mobility = np.full((height, width), floor, dtype=np.float64)
    for name, (zone_u, zone_v, radius_u, radius_v) in zones.items():
        peak = float(regions.get(name, default_mobility))
        reach = np.sqrt(((u - zone_u) / radius_u) ** 2 + ((v - zone_v) / radius_v) ** 2)
        mobility = np.maximum(mobility, peak * (1.0 - _smoothstep(0.55, 1.0, reach)))

    def suppress(reach: npt.NDArray[np.float64], value: float) -> None:
        """Blend mobility down toward a stiffer structure, never up."""
        nonlocal mobility
        weight = 1.0 - _smoothstep(0.60, 1.0, reach)
        mobility = np.minimum(mobility, mobility * (1.0 - weight) + value * weight)

    # Cartilage and bone under the mid-face barely register.
    suppress(
        np.sqrt(((u - 0.50) / 0.105) ** 2 + ((v - 0.505) / 0.155) ** 2),
        regions["nose"],
    )
    suppress(
        np.sqrt(((u - 0.50) / 0.075) ** 2 + ((v - 0.415) / 0.095) ** 2),
        regions["skull"] * 1.5,
    )
    for eye_u, eye_v in eyes_uv:
        suppress(
            np.sqrt(
                ((u - eye_u) / (globe_u * 1.15)) ** 2
                + ((v - eye_v) / (globe_v * 1.3)) ** 2
            ),
            regions["eye"],
        )

    # Skin is pinned wherever it wraps onto the skull edge, so fade the whole
    # map toward the bone value as the silhouette approaches.
    rim = _smoothstep(0.78, 1.0, radial)
    mobility = mobility * (1.0 - rim) + regions["skull"] * rim
    mobility = np.where(inside, mobility, background)

    mobility = _box_blur(mobility, int(round(max(smoothing, 0.0))))
    mobility = np.clip(mobility, 0.0, 1.0)

    # Lip parting: which side of the line, and whether a line exists here.
    line_y = fy0 + float(mouth_center[1]) * fh
    signed = np.clip((line_y - yy.astype(np.float64)) / max(softness_cells, 1e-3),
                     -1.0, 1.0)
    mouth_side = 0.5 + 0.5 * signed
    corner = np.abs(u - float(mouth_center[0])) / max(half_width, 1e-3)
    mouth_slit = 1.0 - _smoothstep(
        1.0 - corner_taper * 0.5, 1.0 + corner_taper, corner
    )
    mouth_slit = np.where(inside, mouth_slit, 0.0)

    eye_aperture = np.zeros((height, width), dtype=np.float64)
    for eye_u, eye_v in eyes_uv:
        distance = np.sqrt(
            ((u - eye_u) / aperture_u) ** 2 + ((v - eye_v) / aperture_v) ** 2
        )
        eye_aperture = np.maximum(eye_aperture, 1.0 - _smoothstep(0.70, 1.0, distance))
    eye_aperture = np.where(inside, eye_aperture, 0.0)

    stack = np.stack(
        [mobility, mouth_side, mouth_slit, eye_aperture], axis=-1
    ).astype(np.float32)
    # World grid y points up; these were built in image space.
    return TissueMaps(rgba=np.ascontiguousarray(np.flipud(stack), dtype=np.float32))


# --------------------------------------------------------------------- muscles


@dataclass(frozen=True, slots=True)
class MuscleUniforms:
    """Fixed-size uniform payload for the shader's displacement loop.

    ``geometry`` rows are ``(anchor_x, anchor_y, radius, gate_code)`` in grid
    space; ``drive`` rows are ``(dx, dy, activation, 0)`` where ``(dx, dy)`` is
    the peak displacement in grid cells this muscle currently asks for.
    """

    geometry: npt.NDArray[np.float32]
    drive: npt.NDArray[np.float32]
    count: int


def muscle_anchor_grid(
    muscle: Muscle,
    face_box: Mapping[str, float],
    grid_height: int,
) -> tuple[float, float]:
    """Face-box UV (v down) to grid coordinates (y up)."""
    image_x = float(face_box["x"]) + muscle.anchor[0] * float(face_box["width"])
    image_y = float(face_box["y"]) + muscle.anchor[1] * float(face_box["height"])
    return image_x, float(grid_height) - image_y


def pack_muscle_uniforms(
    registry: MuscleRegistry,
    activations: Mapping[str, float],
    *,
    face_box: Mapping[str, float],
    grid_height: int,
    capacity: int = MAX_ACTIVE_MUSCLES,
) -> MuscleUniforms:
    """Select the contracting muscles and lay them out for the fragment shader.

    Only active muscles are uploaded, which keeps the shader's inner loop short
    and — because every fragment sees the same list — keeps the branch uniform
    across the warp. Ordering is by descending activation so that a rig larger
    than ``capacity`` degrades by dropping its quietest muscles.
    """
    geometry = np.zeros((capacity, 4), dtype=np.float32)
    drive = np.zeros((capacity, 4), dtype=np.float32)

    ranked = sorted(
        (
            (muscle, float(activations.get(muscle.name, 0.0)))
            for muscle in registry
            if float(activations.get(muscle.name, 0.0)) >= ACTIVATION_EPSILON
            and muscle.travel > 0.0
        ),
        key=lambda item: (-item[1], item[0].name),
    )

    count = 0
    for muscle, activation in ranked:
        if count >= capacity:
            break
        anchor_x, anchor_y = muscle_anchor_grid(muscle, face_box, grid_height)
        magnitude = float(np.hypot(*muscle.force))
        if magnitude <= 1e-6:
            continue
        scale = muscle.travel * activation / magnitude
        geometry[count] = (
            anchor_x,
            anchor_y,
            max(muscle.influence_radius, 1.0),
            muscle.gate_code,
        )
        drive[count] = (
            muscle.force[0] * scale,
            muscle.force[1] * scale,
            activation,
            0.0,
        )
        count += 1

    return MuscleUniforms(geometry=geometry, drive=drive, count=count)


def wendland(distance: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Wendland C² kernel on the unit interval: ``(1-q)^4 (4q + 1)``.

    Chosen over a Gaussian because it reaches exactly zero at ``q = 1`` with a
    zero first derivative, so a muscle's influence ends without a step in the
    field or in its gradient.
    """
    q = np.clip(distance, 0.0, 1.0)
    one_minus = 1.0 - q
    return (one_minus**4) * (4.0 * q + 1.0)


@dataclass(frozen=True, slots=True)
class JawPose:
    """Mandible state in grid space, mirroring the ``avatar_jaw`` uniform."""

    pivot: tuple[float, float]
    angle: float
    reach: float
    chin: float
    span: float
    feather: float

    def profile(self, depth: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Carried travel per radian of opening, by depth below the pivot.

        Linear down to the chin, where bone really does carry skin, then a
        smooth relaxation to nothing through the submental tissue and neck.
        See ``jaw_profile`` in ``avatar.frag``.
        """
        rise = np.clip(depth, 0.0, self.chin)
        return rise * (1.0 - _smoothstep(self.chin, self.reach, depth))

    def lateral(self, position_x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Sideways envelope; see ``jaw_displacement`` in ``avatar.frag``."""
        offset = np.abs(position_x - self.pivot[0])
        return 1.0 - _smoothstep(self.span - self.feather, self.span + self.feather, offset)

    @property
    def uniform(self) -> tuple[float, float, float, float]:
        """The ``avatar_jaw`` payload; the pivot's x rides in ``avatar_jaw_span``."""
        return (self.chin, float(self.pivot[1]), self.angle, self.reach)

    @property
    def span_uniform(self) -> tuple[float, float, float, float]:
        """The ``avatar_jaw_span`` payload."""
        return (float(self.pivot[0]), self.span, self.feather, 0.0)


def jaw_pose_from_definition(
    definition: Mapping[str, Any],
    face_box: Mapping[str, float],
    grid_height: int,
    angle: float = 0.0,
) -> JawPose:
    """Read the mandible landmarks out of the character definition.

    ``jaw_pivot``, ``jaw_chin`` and ``jaw_reach`` are authored as face-box v,
    and everything downstream wants depths below the pivot in grid cells.
    """
    height = float(face_box["height"])
    pivot_uv = definition.get("jaw_pivot", [0.50, 0.62])
    pivot_x = float(face_box["x"]) + float(pivot_uv[0]) * float(face_box["width"])
    pivot_v = float(pivot_uv[1])
    pivot_y = float(grid_height) - (float(face_box["y"]) + pivot_v * height)

    chin = max((float(definition.get("jaw_chin", 0.90)) - pivot_v) * height, 1.0)
    reach = (float(definition.get("jaw_reach", 1.18)) - pivot_v) * height
    width = float(face_box["width"])
    return JawPose(
        pivot=(pivot_x, pivot_y),
        angle=float(angle),
        # The fade has to be long enough to absorb the chin's travel: a
        # smoothstep's steepest slope is 1.5 / length, and the warp inverts the
        # moment that times the travel exceeds one cell per cell.
        reach=max(reach, chin + 1.5 * chin + 1.0),
        chin=chin,
        span=float(definition.get("jaw_half_width", 0.42)) * width,
        feather=float(definition.get("jaw_feather", 0.16)) * width,
    )


def _lip_gate(
    gate_code: float,
    side: npt.NDArray[np.float64],
    slit: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | float:
    if gate_code == 0.0:
        return 1.0
    same = side if gate_code > 0.0 else 1.0 - side
    return 1.0 + slit * (same - 1.0)


def displacement_field(
    uniforms: MuscleUniforms,
    tissue: TissueMaps,
    grid_width: int,
    grid_height: int,
    jaw: JawPose | None = None,
) -> npt.NDArray[np.float64]:
    """CPU reference for the shader's total displacement, in grid cells.

    Mirrors ``total_displacement`` in ``avatar.frag`` so tests can assert
    properties of the warp — smoothness, no folding, lock authority — without a
    GPU. Muscles slide skin and so are scaled by mobility; the jaw moves the
    bone the skin sits on and so is not.
    """
    yy, xx = np.mgrid[0:grid_height, 0:grid_width]
    position_x = xx.astype(np.float64) + 0.5
    position_y = yy.astype(np.float64) + 0.5
    mobility = tissue.mobility.astype(np.float64)
    side = tissue.mouth_side.astype(np.float64)
    slit = tissue.mouth_slit.astype(np.float64)

    total = np.zeros((grid_height, grid_width, 2), dtype=np.float64)
    for index in range(uniforms.count):
        anchor_x, anchor_y, radius, gate_code = uniforms.geometry[index]
        delta_x, delta_y = uniforms.drive[index][:2]
        distance = np.hypot(position_x - anchor_x, position_y - anchor_y) / radius
        weight = wendland(distance) * _lip_gate(float(gate_code), side, slit)
        total[..., 0] += delta_x * weight
        total[..., 1] += delta_y * weight

    total *= mobility[..., None]

    if jaw is not None and jaw.angle > 1e-4:
        travel = jaw.profile(jaw.pivot[1] - position_y)
        total[..., 1] -= (
            travel
            * np.sin(jaw.angle)
            * _lip_gate(-1.0, side, slit)
            * jaw.lateral(position_x)
        )

    return total


# ------------------------------------------------------------------------- io


def default_tissue_path(world_path: str | Path) -> Path:
    return Path(world_path).with_name("face_tissue.npy")


def save_tissue_maps(
    path: str | Path,
    tissue: TissueMaps,
    *,
    face: Any = None,
) -> Path:
    """Write the float RGBA maps plus a channel legend and the source face box.

    Recording the box matters: muscle anchors are authored in face-box UV, so
    packing uniforms against a different rectangle than the one these maps were
    baked from would slide every muscle off its tissue.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, tissue.rgba)
    payload: dict[str, Any] = {
        "version": "face-tissue-1.0",
        "channels": {
            "r": "mobility",
            "g": "mouth_side",
            "b": "mouth_slit",
            "a": "eye_aperture",
        },
        "mobility_mean": float(tissue.mobility.mean()),
        "mobility_max": float(tissue.mobility.max()),
    }
    if face is not None:
        payload["face_box"] = {
            "x": float(face.x),
            "y": float(face.y),
            "width": float(face.width),
            "height": float(face.height),
        }
    destination.with_suffix(".json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return destination


def load_tissue_maps(path: str | Path) -> npt.NDArray[np.float32]:
    array = np.load(path)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"tissue maps must be HxWx4, got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


__all__ = [
    "ACTIVATION_EPSILON",
    "DEFAULT_MOBILITY",
    "MAX_ACTIVE_MUSCLES",
    "TISSUE_EYE_APERTURE",
    "TISSUE_MOBILITY",
    "TISSUE_MOUTH_SIDE",
    "TISSUE_MOUTH_SLIT",
    "JawPose",
    "MuscleUniforms",
    "TissueMaps",
    "build_tissue_maps",
    "default_tissue_path",
    "displacement_field",
    "jaw_pose_from_definition",
    "load_tissue_maps",
    "muscle_anchor_grid",
    "pack_muscle_uniforms",
    "save_tissue_maps",
    "wendland",
]
