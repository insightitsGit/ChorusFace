#version 430

//#prelude

// Continuous facial deformation.
//
// The portrait is never cut into pieces that slide over each other. Every
// muscle contributes a compactly-supported radial displacement, the sum is a
// C2 field over the whole face, and the photograph is inverse-warped through
// it. Smoothness is structural: there is no piece boundary that can tear,
// because there are no pieces.
//
// Three things a displacement field genuinely cannot express are handled as
// explicit occlusion instead, because they are changes in visibility rather
// than deformations of a continuous sheet:
//
//   * the mandible dropping, which is a bone moving under the skin,
//   * the mouth parting, which opens a hole no warp can create,
//   * the eyelid closing, which slides one surface across another.

layout(std430, binding = 0) readonly restrict buffer WorldState {
    Cell cells[];
} world;

uniform uvec2 grid_size;
uniform vec2 viewport_size;
// Window-space rectangle the portrait occupies: xy = origin, zw = size, all in
// [0,1] UV. Defaults to the whole window; the app shrinks it to letterbox the
// face above the chat panel without stretching the photograph.
uniform vec4 avatar_frame;

// RGB = photograph at display resolution (mipmapped; AMIN step 11).
uniform sampler2D avatar_base_color;
// R = part-id / PART_SCALE at grid resolution (NEAREST — discrete labels).
uniform sampler2D avatar_part_ids;
// R = mobility, G = mouth-line side, B = mouth slit, A = eye aperture
uniform sampler2D avatar_tissue;
// Real expression plates from aiface-capture (same UV as source_face).
uniform sampler2D avatar_open_plate;
uniform sampler2D avatar_smile_plate;
// Viseme plate memory: blend two atlas plates over the mouth region.
uniform sampler2D avatar_plate_a;
uniform sampler2D avatar_plate_b;
uniform int avatar_plates_ready;
// x = mix A→B, y = replace amount (1 = full plate memory), z/w unused
uniform vec4 avatar_plate_blend;
// Upper-face expression plate from expression_catalog (surprise / brows).
uniform sampler2D avatar_expr_plate;
// x = eye_widen, y = brow_raise, z = plate blend amount, w = plate ready
uniform vec4 avatar_expr_state;
// Photographed nearly-closed eyes (LOOK region — mouth open.png analogue).
uniform sampler2D avatar_eye_closed_plate;
uniform int avatar_eye_closed_ready;

// Keep in step with aiface.skinning.MAX_ACTIVE_MUSCLES.
const int MAX_MUSCLES = 48;
// xy = anchor in grid cells, z = influence radius, w = lip gate (-1, 0, +1)
uniform vec4 muscle_geometry[MAX_MUSCLES];
// xy = peak displacement in grid cells, z = activation
uniform vec4 muscle_drive[MAX_MUSCLES];
uniform int muscle_count;

// x = mouth centre x, y = rest lip line y, z = half width, w = softness (cells)
uniform vec4 avatar_mouth_line;
// x = chin depth below the pivot, y = condyle pivot y (grid cells),
// z = opening angle (rad), w = depth at which the carry has died out.
uniform vec4 avatar_jaw;
// x = mandible centre x, y = half width, z = feather, all in grid cells
uniform vec4 avatar_jaw_span;
// x = width, y = openness, z = roundness, w = smile/frown
uniform vec4 avatar_mouth_pose;
uniform vec4 avatar_eye_state;   // gaze_x, gaze_y, pupil, blink
uniform vec4 avatar_eye_centers; // left.xy, right.xy in grid cells
uniform vec4 avatar_eye_shape;   // half_width, half_height, gaze_travel, unused
uniform float avatar_lock_overlay;
uniform float avatar_breath_phase;
uniform float avatar_muscle_heat;
uniform int avatar_debug_view;
// 1 once the tissue maps are resident; without them there is nothing to warp
// against and the portrait is shown as-is.
uniform int avatar_deform;
// GPU display recipe knobs (gpu_display_recipe.json — same at train and play):
// x = jaw angle at full open.png drive, y = smile suppression under open plate,
// z = atlas plate strength, w = cavity fill strength.
uniform vec4 avatar_recipe;
// Grid cells of tissue travel per unit of NWR field velocity (channels 0/1).
uniform float avatar_field_gain;
// AMIN step 12 — plate snap. 0 keeps linear cross-fades; 1 commits each drive
// to the nearest captured mouth shape so mid-blends stop reading as blur.
uniform float avatar_plate_sharpness;
// Cosmetic grade on top of locked identity (scaffolding prefs; not new RGB identity).
uniform vec3 avatar_skin_tint;
uniform vec3 avatar_eye_tint;
uniform float avatar_makeup_strength;

in vec2 uv;
layout(location = 0) out vec4 fragment_color;

const float PART_SCALE = 10.0;
const int PART_NONE = 0;
const int PART_FACE = 1;
const int PART_NOSE = 2;
const int PART_LEFT_BROW = 3;
const int PART_RIGHT_BROW = 4;
const int PART_LEFT_EYE = 5;
const int PART_RIGHT_EYE = 6;
const int PART_UPPER_LIP = 7;
const int PART_LOWER_LIP = 8;
const int PART_MOUTH_CAVITY = 9;

// Fixed-point iterations for inverting the forward displacement. Two suffice
// for small warps; a wide-open jaw needs more or the achieved warp undershoots
// the forward prediction and the cavity paints skin below the visible lips.
const int WARP_ITERATIONS = 6;
// Below this mobility a fragment is background rather than skin.
const float FACE_PRESENCE_EDGE = 0.08;

uint cell_index(ivec2 position) {
    ivec2 bounded = clamp(position, ivec2(0), ivec2(grid_size) - ivec2(1));
    return uint(bounded.y) * grid_size.x + uint(bounded.x);
}

float state_at(ivec2 position, int channel) {
    return world.cells[cell_index(position)].value[channel];
}

float lock_boundary(ivec2 position) {
    float centre = state_at(position, HUMAN_LOCK_CHANNEL);
    float delta = 0.0;
    delta = max(delta, abs(centre - state_at(position + ivec2(1, 0), HUMAN_LOCK_CHANNEL)));
    delta = max(delta, abs(centre - state_at(position - ivec2(1, 0), HUMAN_LOCK_CHANNEL)));
    delta = max(delta, abs(centre - state_at(position + ivec2(0, 1), HUMAN_LOCK_CHANNEL)));
    delta = max(delta, abs(centre - state_at(position - ivec2(0, 1), HUMAN_LOCK_CHANNEL)));
    return clamp(delta, 0.0, 1.0);
}

vec3 heat_color(float value) {
    float t = clamp(value, 0.0, 1.0);
    return mix(vec3(0.02, 0.05, 0.18), mix(vec3(0.95, 0.45, 0.05), vec3(1.0, 0.95, 0.55), t), t);
}

vec2 to_uv(vec2 grid_position) {
    return clamp(grid_position / vec2(grid_size), 0.0, 1.0);
}

vec4 tissue_at(vec2 grid_position) {
    return texture(avatar_tissue, to_uv(grid_position));
}

vec3 photo_at(vec2 grid_position) {
    return texture(avatar_base_color, to_uv(grid_position)).rgb;
}

int part_at(vec2 grid_position) {
    return int(round(texture(avatar_part_ids, to_uv(grid_position)).r * PART_SCALE));
}

// Wendland C2 kernel. Unlike a Gaussian it reaches exactly zero at q = 1 with a
// zero derivative, so a muscle's reach ends without a step in the field or in
// its gradient.
float wendland(float q) {
    float clamped = clamp(q, 0.0, 1.0);
    float tail = 1.0 - clamped;
    float tail2 = tail * tail;
    return tail2 * tail2 * (4.0 * clamped + 1.0);
}

// The lip parting is the only genuine discontinuity in facial tissue, so it is
// the only place a muscle is forbidden to pull across. Past the mouth corners
// the slit closes and the gate relaxes to 1, which is what stops the gate from
// introducing the very seam this design avoids.
float lip_gate(float code, float side, float slit) {
    if (code == 0.0) {
        return 1.0;
    }
    float same = code > 0.0 ? side : 1.0 - side;
    return mix(1.0, same, slit);
}

vec2 muscle_displacement(vec2 grid_position, float side, float slit) {
    vec2 total = vec2(0.0);
    for (int index = 0; index < muscle_count; ++index) {
        vec4 geometry = muscle_geometry[index];
        float q = distance(grid_position, geometry.xy) / geometry.z;
        if (q >= 1.0) {
            continue;
        }
        float weight = wendland(q) * lip_gate(geometry.w, side, slit);
        total += muscle_drive[index].xy * weight;
    }
    return total;
}

// Jaw opening is a bone rotating about the condyles, whose axis is horizontal.
// Projected to a frontal portrait that reads as a downward translation growing
// with depth below the pivot; an in-plane rotation would swing the chin
// sideways.
//
// The travel grows linearly down to the chin, because that stretch really is
// rigid bone under skin, and then relaxes back to nothing through the
// submental soft tissue and neck, which stretch instead of translating. That
// second stretch is what keeps the warp invertible: a hard cut-off at the chin
// would put an unbounded gradient right where the travel is largest, and the
// face would fold under itself.
float jaw_profile(float depth) {
    float rise = clamp(depth, 0.0, avatar_jaw.x);
    return rise * (1.0 - smoothstep(avatar_jaw.x, avatar_jaw.w, depth));
}

// Skin over the mandible travels with the bone rather than sliding across it,
// so — unlike muscle pull — this term is not scaled by mobility at all. Doing
// so would be actively wrong: mobility fades to nothing at the silhouette to
// pin sliding skin there, but an opening jaw genuinely moves the silhouette,
// and forcing tens of cells of travel to die inside that narrow rim is exactly
// what folds the chin under itself. The travel is bounded vertically by the
// profile and horizontally by a wide feather instead. The horizontal one is
// free: the jaw displaces along y only, so its x-gradient never reaches the
// Jacobian's diagonal.
vec2 jaw_displacement(vec2 grid_position, float side, float slit) {
    float angle = avatar_jaw.z;
    if (angle < 1e-4) {
        return vec2(0.0);
    }
    float travel = jaw_profile(avatar_jaw.y - grid_position.y);
    float lateral = 1.0 - smoothstep(
        avatar_jaw_span.y - avatar_jaw_span.z,
        avatar_jaw_span.y + avatar_jaw_span.z,
        abs(grid_position.x - avatar_jaw_span.x)
    );
    return vec2(0.0, -travel * sin(angle) * lip_gate(-1.0, side, slit) * lateral);
}

// NWR field warp (AMIN steps 1–3): speech proposes ±4 velocity impulses, the
// constraint pass validates them and integrates the result into channels 0/1
// of every unlocked cell. Sampling that "no rules" 2D vector back into the
// tissue warp closes the loop — the GPU field itself moves pixels instead of
// being telemetry-only next to the analytic muscle/jaw model.
vec2 field_velocity(ivec2 cell) {
    float locked = step(0.5, state_at(cell, HUMAN_LOCK_CHANNEL));
    return vec2(state_at(cell, 0), state_at(cell, 1)) * (1.0 - locked);
}

// Bilinear so the per-cell vector stays smooth enough for the fixed-point
// inverse warp to converge.
vec2 field_displacement(vec2 grid_position) {
    if (avatar_field_gain <= 0.0) {
        return vec2(0.0);
    }
    vec2 base = grid_position - vec2(0.5);
    ivec2 cell = ivec2(floor(base));
    vec2 f = fract(base);
    vec2 lower = mix(field_velocity(cell), field_velocity(cell + ivec2(1, 0)), f.x);
    vec2 upper = mix(
        field_velocity(cell + ivec2(0, 1)),
        field_velocity(cell + ivec2(1, 1)),
        f.x
    );
    return mix(lower, upper, f.y) * avatar_field_gain;
}

// Articulation is muscles + jaw + the NWR field. MouthPose lip bias was
// removed: it opened a billboard hole and fought the photograph warp.
vec2 total_displacement(vec2 grid_position) {
    vec4 tissue = tissue_at(grid_position);
    vec2 muscles = muscle_displacement(grid_position, tissue.g, tissue.b) * tissue.r;
    vec2 jaw = jaw_displacement(grid_position, tissue.g, tissue.b);
    vec2 field = field_displacement(grid_position) * tissue.r;
    // BUGFIX: never damp jaw. An older path multiplied the whole mouth
    // displacement by ~0.18 whenever plate memory was on, which left the lips
    // looking zipped while cheeks still twitched.
    // Only soften lateral smile-muscle tear when a smile plate is dominant and
    // the jaw is nearly closed — speech open must keep full warp.
    float smile = clamp(avatar_mouth_pose.w, 0.0, 1.0);
    float jaw_n = clamp(avatar_jaw.z / 0.55, 0.0, 1.0);
    float plate_amt = avatar_plates_ready == 1
        ? clamp(avatar_plate_blend.y, 0.0, 1.0) : 0.0;
    float smile_damp = smile * plate_amt * (1.0 - jaw_n);
    float mouth_prox = 1.0 - smoothstep(
        avatar_mouth_line.z * 0.9,
        avatar_mouth_line.z * 2.4,
        distance(grid_position, avatar_mouth_line.xy)
    );
    muscles *= mix(1.0, 0.55, smile_damp * mouth_prox);
    // Region gate (B4-safe): plate owns the oral disk — mute FIELD *inside*
    // that disk so warped identity doesn't ghost under open.png. Cheeks /
    // outside the disk keep FIELD. Never globally kill open.png.
    float plate_disk = plate_amt * (1.0 - smoothstep(
        avatar_mouth_line.z * 0.45,
        avatar_mouth_line.z * 1.05,
        distance(grid_position, avatar_mouth_line.xy)
    ));
    field *= (1.0 - plate_disk);
    // Single-owner eye disk during blink (mouth-band analogue): mute FIELD
    // under closing lids so brow/cheek teacher motion cannot smear L09.
    float blink_amt = clamp(avatar_eye_state.w, 0.0, 1.0);
    if (blink_amt > 0.04) {
        float eye_r = max(avatar_eye_shape.x, avatar_eye_shape.y) * 2.35;
        float d_l = distance(grid_position, avatar_eye_centers.xy);
        float d_r = distance(grid_position, avatar_eye_centers.zw);
        float near_eye = 1.0 - smoothstep(eye_r * 0.65, eye_r * 1.35, min(d_l, d_r));
        field *= (1.0 - blink_amt * near_eye);
        // Soften muscle twitches in the same disk (TickFeed usually has none).
        muscles *= (1.0 - blink_amt * near_eye * 0.85);
    }
    return muscles + jaw + field;
}

// Sampling the tissue maps at the candidate source rather than at the fragment
// keeps the warp self-consistent: a pixel that came from the upper lip is
// gated as upper lip, whichever side of the parting it has since moved to.
vec2 inverse_warp(vec2 grid_position) {
    vec2 source = grid_position;
    for (int step_index = 0; step_index < WARP_ITERATIONS; ++step_index) {
        source = grid_position - total_displacement(source);
    }
    return source;
}

// Where the two lip edges have travelled to in this column. Anything between
// them is a hole the warp cannot fill, so it is painted as mouth cavity.
vec2 mouth_gap(float column_x) {
    vec2 at_line = vec2(column_x, avatar_mouth_line.y);
    vec4 tissue = tissue_at(at_line);
    float mobility = tissue.r;
    float slit = tissue.b;
    // Field warp shifts both lip edges together so the cavity tracks the
    // NWR-driven tissue instead of lagging behind it.
    float field_y = field_displacement(at_line).y * mobility;
    float upper = muscle_displacement(at_line, 1.0, slit).y * mobility + field_y;
    float lower = muscle_displacement(at_line, 0.0, slit).y * mobility
        + jaw_displacement(at_line, 0.0, slit).y + field_y;
    // Anatomical cap: a hole taller than ~0.85× the mouth half-width is warp
    // prediction overshoot, not a real opening — don't paint chin skin dark.
    lower = max(lower, upper - avatar_mouth_line.z * 0.85);
    return vec2(avatar_mouth_line.y + lower, avatar_mouth_line.y + upper);
}

vec3 cavity_color(float grid_y, vec2 gap, float column_x) {
    // Soft photo shadow by default. When capture plates exist, show the user's
    // real open/smile interior — never invented enamel.
    float span = max(gap.y - gap.x, 1e-3);
    float depth = clamp((grid_y - gap.x) / span, 0.0, 1.0);
    float throat = 1.0 - abs(depth * 2.0 - 1.0);
    vec3 upper_lip = photo_at(vec2(column_x, gap.y + 1.2));
    vec3 lower_lip = photo_at(vec2(column_x, gap.x - 1.2));
    vec3 lip_mix = mix(lower_lip, upper_lip, depth);
    vec3 shade = lip_mix * mix(0.50, 0.18, throat);
    shade = mix(shade, vec3(0.06, 0.025, 0.03), throat * 0.65);

    if (avatar_plates_ready == 1) {
        // Sample plates at the gap sample point. Plate alpha is a mouth-interior
        // matte from aiface-capture so cheeks never leak into the hole.
        vec2 plate_uv = to_uv(vec2(column_x, grid_y));
        vec4 open_s = texture(avatar_open_plate, plate_uv);
        vec4 smile_s = texture(avatar_smile_plate, plate_uv);
        float jaw = clamp(avatar_jaw.z / 0.55, 0.0, 1.0);
        float openness = clamp(avatar_mouth_pose.y / 14.0, 0.0, 1.0);
        float smile = clamp(avatar_mouth_pose.w, 0.0, 1.0);
        // Open plate fills the cavity when the jaw parts. Smile plate only for
        // happy expression — never from speech openness (that hid a smile).
        float open_w = clamp(jaw * 0.80 + openness * 0.45, 0.0, 1.0)
            * max(throat, 0.20) * open_s.a;
        float smile_w = clamp(smile, 0.0, 1.0)
            * (1.0 - jaw) * (1.0 - open_w)
            * throat * smile_s.a;
        shade = mix(shade, smile_s.rgb, smile_w * 0.70);
        shade = mix(shade, open_s.rgb, open_w * 0.90);
    }
    return shade;
}

vec3 part_debug_color(int part) {
    if (part == PART_LEFT_EYE || part == PART_RIGHT_EYE) return vec3(0.15, 0.7, 1.0);
    if (part == PART_LEFT_BROW || part == PART_RIGHT_BROW) return vec3(1.0, 0.65, 0.15);
    if (part == PART_UPPER_LIP) return vec3(1.0, 0.35, 0.55);
    if (part == PART_LOWER_LIP) return vec3(1.0, 0.15, 0.35);
    if (part == PART_MOUTH_CAVITY) return vec3(0.08, 0.08, 0.1);
    if (part == PART_NOSE) return vec3(0.55, 1.0, 0.4);
    if (part == PART_FACE) return vec3(0.55, 0.45, 0.4);
    return vec3(0.05);
}

// Capture mattes are soft ellipses (mean alpha ~0.3 over a wide lower-face
// region). At full drive that still reads as a washed veil / motion blur.
// Steepen the alpha so only the real oral interior commits.
float harden_matte(float alpha, float snap) {
    float a = clamp(alpha, 0.0, 1.0);
    return mix(a, smoothstep(0.22, 0.78, a), clamp(snap, 0.0, 1.0));
}

// Hybrid core+edge matte (Nuke Primatte / Resolve edge-extension idea):
// opaque oral CORE + thin soft EDGE. Keeps open.png readable while killing
// the wide half-transparent veil that reads as motion blur.
// Core commits earlier so teeth/lips don't stay 50% transparent over warped
// identity (double-teeth ghost). Edge stays thin — no cheek stamp.
float hybrid_matte(float alpha, float snap) {
    float a = clamp(alpha, 0.0, 1.0);
    float s = clamp(snap, 0.0, 1.0);
    // Under hard snap: opaque oral core, almost no half-α cheek veil
    // (that veil was reading as mouth motion blur / oval stamp).
    float core = smoothstep(mix(0.22, 0.32, s), mix(0.48, 0.55, s), a);
    float edge = smoothstep(mix(0.08, 0.16, s), mix(0.22, 0.32, s), a)
        * (1.0 - core);
    float edge_w = mix(0.22, 0.05, s);
    float hybrid = clamp(core + edge * edge_w, 0.0, 1.0);
    // Blend toward hybrid with plate_sharpness; never invent opacity from zero.
    return mix(harden_matte(a, s), hybrid, s);
}

// Atlas speech ownership in [0,1]. When high under hard snap, open.png must
// not also stamp — both share avatar_plate_blend.y today.
float atlas_own_amount() {
    if (avatar_plates_ready != 1) {
        return 0.0;
    }
    return clamp(avatar_plate_blend.y, 0.0, 1.0) * clamp(avatar_recipe.z, 0.0, 1.0);
}

void main() {
    // Everything below works in portrait UV, so map the window into the frame
    // first and hand the surrounding matte back immediately. Keeping the frame
    // here rather than shrinking the viewport means the HUD and chat overlay
    // still address the whole window.
    vec2 framed_uv = (uv - avatar_frame.xy) / max(avatar_frame.zw, vec2(1e-6));
    if (any(lessThan(framed_uv, vec2(0.0))) || any(greaterThan(framed_uv, vec2(1.0)))) {
        // Matches the runtime's clear colour so the letterbox reads as one
        // continuous surround rather than a visible band on two sides.
        fragment_color = vec4(0.002, 0.004, 0.012, 1.0);
        return;
    }

    vec2 grid_position = framed_uv * vec2(grid_size);
    ivec2 cell_position = clamp(
        ivec2(floor(grid_position)),
        ivec2(0),
        ivec2(grid_size) - ivec2(1)
    );
    uint index = cell_index(cell_position);

    float density = abs(world.cells[index].value[3]);
    float velocity = length(vec2(world.cells[index].value[0], world.cells[index].value[1]));
    float locked = step(0.5, world.cells[index].value[HUMAN_LOCK_CHANNEL]);
    float unlocked = 1.0 - locked;
    float priority = world.cells[index].value[PRIORITY_CHANNEL];

    if (avatar_debug_view == 1) {
        fragment_color = vec4(vec3(density), 1.0);
        return;
    }
    if (avatar_debug_view == 2) {
        fragment_color = vec4(heat_color(velocity / 1.5), 1.0);
        return;
    }
    if (avatar_debug_view == 3) {
        fragment_color = vec4(heat_color(unlocked * avatar_muscle_heat), 1.0);
        return;
    }
    if (avatar_debug_view == 4) {
        float emotion = 0.5 + 0.5 * avatar_mouth_pose.w;
        fragment_color = vec4(emotion, 0.25, 1.0 - emotion, 1.0);
        return;
    }
    if (avatar_debug_view == 5) {
        float jaw = clamp(avatar_jaw.z / 0.55, 0.0, 1.0);
        float band = 1.0 - smoothstep(8.0, 28.0, distance(grid_position, avatar_mouth_line.xy));
        fragment_color = vec4(vec3(0.1) + vec3(0.9, 0.4, 0.1) * jaw * band, 1.0);
        return;
    }
    if (avatar_debug_view == 6) {
        fragment_color = vec4(locked, priority, unlocked, 1.0);
        return;
    }
    if (avatar_debug_view == 7) {
        float heat = unlocked * clamp(velocity * 0.7 + avatar_muscle_heat, 0.0, 1.0);
        fragment_color = vec4(heat_color(heat), 1.0);
        return;
    }
    if (avatar_debug_view == 8) {
        int part = part_at(grid_position);
        fragment_color = vec4(mix(photo_at(grid_position), part_debug_color(part), 0.72), 1.0);
        return;
    }
    if (avatar_debug_view == 9) {
        // Displacement: red right, green up, brightness = magnitude in cells.
        vec2 displacement = total_displacement(grid_position);
        float magnitude = length(displacement);
        vec3 direction = vec3(0.5 + 0.5 * normalize(displacement + vec2(1e-6)), 0.0);
        fragment_color = vec4(direction * clamp(magnitude / 6.0, 0.0, 1.0), 1.0);
        return;
    }
    if (avatar_debug_view == 10) {
        fragment_color = vec4(heat_color(tissue_at(grid_position).r), 1.0);
        return;
    }
    if (avatar_debug_view == 11) {
        vec4 tissue = tissue_at(grid_position);
        fragment_color = vec4(tissue.g, tissue.b, tissue.a, 1.0);
        return;
    }

    vec3 color;
    float face_alpha;

    float mouth_inside = 0.0;
    if (avatar_deform == 1) {
        // Display layer stack (aiface.display_layers) — DO NOT REORDER:
        // L02 muscle_jaw_warp → L04 smile → L05 open → L06 cavity →
        // L07 atlas → L08 expr → L09 eyes → L10 brow. Field L01 is sampled
        // inside inverse_warp; cell groups L03 write ch0/1 before this pass.
        float snap = clamp(avatar_plate_sharpness, 0.0, 1.0);
        float atlas_own = atlas_own_amount();
        float layer_open = clamp(avatar_plate_blend.y, 0.0, 1.0);

        // Resolve LOOK plate ownership FIRST (unwarped UVs), then warp only
        // where plates do not own the pixel. Warping under open.png was the
        // remaining ghost/blur (live mouth sharp≈1.97 vs open.png asset≈2.40).
        float open_w = 0.0;
        float smile_w = 0.0;
        float open_drive_g = layer_open;
        vec4 open_s = vec4(0.0);
        vec4 smile_s = vec4(0.0);
        vec4 atlas_a_s = vec4(0.0);
        vec4 atlas_b_s = vec4(0.0);
        float atlas_a_ev = 0.0;
        float open_primary_g = 0.0;
        float atlas_primary_g = 0.0;
        if (avatar_plates_ready == 1) {
            vec2 capture_uv = to_uv(grid_position);
            open_s = texture(avatar_open_plate, capture_uv);
            smile_s = texture(avatar_smile_plate, capture_uv);
            // Sample atlas early for ownership evidence — never mute open.png
            // from drive amount alone (9s plates have weak oral α → dead zone).
            atlas_a_s = texture(avatar_plate_a, capture_uv);
            atlas_b_s = texture(avatar_plate_b, capture_uv);
            atlas_a_ev = hybrid_matte(
                max(atlas_a_s.a, atlas_b_s.a), max(snap, 0.55)
            );
            float open_from_plate = layer_open;
            float open_from_jaw = clamp(
                avatar_jaw.z / max(avatar_recipe.x, 1e-3), 0.0, 1.0
            );
            float smile_drive = clamp(avatar_mouth_pose.w, 0.0, 1.0);
            float open_drive = mix(
                max(open_from_plate, open_from_jaw),
                open_from_plate,
                snap
            );
            open_drive = mix(open_drive, smoothstep(0.12, 0.72, open_drive), max(snap, 0.55));
            smile_drive = mix(smile_drive, smoothstep(0.18, 0.82, smile_drive), snap);
            smile_drive *= (1.0 - smoothstep(0.12, 0.40, open_drive));
            // Single LOOK owner only when atlas can paint. Otherwise open.png
            // keeps mid-band (blur_audit: mute-without-evidence → stuck smile).
            open_primary_g = smoothstep(0.42, 0.72, layer_open);
            float atlas_want = (1.0 - open_primary_g) * mix(
                0.0, smoothstep(0.20, 0.55, atlas_own), max(snap, 0.55)
            );
            atlas_primary_g = atlas_want * smoothstep(0.14, 0.42, atlas_a_ev);
            open_drive *= (1.0 - atlas_primary_g);
            smile_drive *= (1.0 - max(atlas_primary_g, open_drive));
            open_w = open_drive * hybrid_matte(open_s.a, max(snap, 0.55));
            // Kill the soft ellipse cheek veil when open.png is primary —
            // that half-α ring is the visible oval stamp / "blur".
            open_w *= mix(1.0, smoothstep(0.30, 0.52, open_s.a), open_primary_g);
            smile_w = smile_drive * hybrid_matte(smile_s.a, snap)
                * (1.0 - open_w * avatar_recipe.y);
            open_drive_g = open_drive;
        }
        float plate_own = clamp(max(open_w, smile_w * 0.85), 0.0, 1.0);
        float plate_commit = atlas_primary_g;
        // Rest-align under the active LOOK owner only (muted drive / atlas).
        if (avatar_plates_ready == 1) {
            // Rest-align on oral core only — wide soft alpha made a pasted chin.
            plate_own = max(
                plate_own,
                open_drive_g * smoothstep(0.22, 0.48, open_s.a)
            );
            plate_own = max(plate_own, plate_commit * atlas_own);
        }

        // L02: warp identity only outside plate-owned pixels.
        // During plate ownership, rest-align harder so FIELD travel cannot
        // smear under a committed LOOK plate (transition single-owner).
        // Mandible lockstep (§14.3): same LOOK open as plates. Undo jaw in
        // the inverse warp (1st order), rest-align FIELD under the matte,
        // then re-apply jaw with a soft fade at the oral matte so the chin
        // supports open.png without a hard seam tear at the plate edge.
        vec4 tj = tissue_at(grid_position);
        vec2 jaw_d = jaw_displacement(grid_position, tj.g, tj.b);
        vec2 source = inverse_warp(grid_position) + jaw_d;
        float rest_mix = clamp(plate_own * 0.99 + plate_commit * 0.55, 0.0, 0.99);
        source = mix(source, grid_position, rest_mix);
        // Oral matte for jaw assist tracks capture ownership only — when atlas
        // muted open.png, don't keep jaw-fading across the wide open ellipse.
        float oral_matte = avatar_plates_ready == 1
            ? hybrid_matte(open_s.a, max(snap, 0.55)) * open_drive_g
            : 0.0;
        oral_matte = max(oral_matte, plate_commit * 0.65);
        // Narrower chin seam: plate owns oral core; jaw only assists outside.
        float jaw_fade = 1.0 - smoothstep(0.14, 0.58, oral_matte);
        source -= jaw_d * jaw_fade;
        color = photo_at(source);
        face_alpha = smoothstep(0.02, 0.12, max(color.r, max(color.g, color.b)));

        // L04/L05 CAPTURE LOOKS — plate RGB on top of rest-aligned identity.
        float open_plate_w = 0.0;
        if (avatar_plates_ready == 1) {
            // Cover resting smile-corner creases that sit outside the open O
            // (full-cycle QA: dark horizontal \"scars\" beside the mouth).
            float wing_y = 1.0 - smoothstep(
                avatar_mouth_line.z * 0.45,
                avatar_mouth_line.z * 1.05,
                abs(grid_position.y - avatar_mouth_line.y)
            );
            float wing_x = smoothstep(
                avatar_mouth_line.z * 0.75,
                avatar_mouth_line.z * 2.10,
                abs(grid_position.x - avatar_mouth_line.x)
            );
            // Corner cover when open.png owns (incl. mid-band fallback).
            float corner_cover = open_drive_g * wing_y * wing_x
                * smoothstep(0.12, 0.28, open_s.a)
                * (1.0 - open_w)
                * max(open_primary_g, 1.0 - atlas_primary_g)
                * 0.35;
            float open_commit = max(open_w, corner_cover);
            color = mix(color, smile_s.rgb, smile_w);
            color = mix(color, open_s.rgb, open_commit);
            face_alpha = max(face_alpha, max(open_commit, smile_w));
            open_plate_w = open_commit;
        }

        // L06: optional cavity fill when the jaw actually parts.
        vec2 gap = mouth_gap(grid_position.x);
        float slit = tissue_at(vec2(grid_position.x, avatar_mouth_line.y)).b;
        float span = gap.y - gap.x;
        // A real hole, not lip contact: painting sub-cell separations dark put
        // a translucent gray film on lips that were visibly still closed.
        float hole = smoothstep(1.5, 3.2, span);
        // Radial falloff around the mouth centre. mouth_gap predicts lip travel
        // with the FORWARD displacement while the fixed-point inverse warp
        // undershoots big jaw drops, so the predicted hole can reach into chin
        // skin — it stamped a hard dark rectangle below the visible lips.
        float near_mouth = 1.0 - smoothstep(
            avatar_mouth_line.z * 1.0,
            avatar_mouth_line.z * 1.8,
            distance(grid_position, avatar_mouth_line.xy)
        );
        // Feather scales with the opening so edges never read as 2-px lines.
        float feather = max(span * 0.22, 0.45);
        mouth_inside = slit * hole * near_mouth
            * smoothstep(gap.x, gap.x + feather, grid_position.y)
            * (1.0 - smoothstep(gap.y - feather, gap.y, grid_position.y));
        // Real pixels beat synthetic shadow (`dark_cavity: Never` owns the
        // mouth). Atlas/open layer amount suppresses cavity — never the muted
        // capture drive alone (that reopened a dark gap under speech plates).
        float plate_takeover = smoothstep(
            0.15, 0.55, max(open_drive_g, atlas_own)
        );
        // Closed / lip-tighten: never paint cavity — a residual slit+span still
        // stamped a dark soft rectangle over visibly closed lips ("the blur").
        float cavity_gate = smoothstep(
            0.04, 0.14, max(clamp(avatar_jaw.z, 0.0, 1.0), open_drive_g)
        );
        // Atlas billboard owns the oral interior — kill synthetic gap fill.
        cavity_gate *= (1.0 - mix(0.0, smoothstep(0.30, 0.70, atlas_own), snap));
        color = mix(
            color,
            cavity_color(grid_position.y, gap, grid_position.x),
            clamp(mouth_inside, 0.0, 1.0) * avatar_recipe.w
                * (1.0 - plate_takeover)
                * (1.0 - open_plate_w)
                * cavity_gate
        );

        // L07: atlas plate memory (finer viseme shapes) on top when amount > 0.
        // Reuse early samples — ownership already gated on atlas_a_ev.
        if (avatar_plates_ready == 1 && avatar_plate_blend.y > 0.001) {
            float mix_ab = clamp(avatar_plate_blend.x, 0.0, 1.0);
            // Step 12: bias toward the nearest real captured shape instead of
            // an even cross-fade of two different mouths. Stronger when snap
            // is high (transition owner) so mid-band A/B ghosts die early.
            mix_ab = mix(
                mix_ab,
                smoothstep(0.42, 0.58, mix_ab),
                clamp(snap * 1.15, 0.0, 1.0)
            );
            vec3 plate_rgb = mix(atlas_a_s.rgb, atlas_b_s.rgb, mix_ab);
            // Atlas primary when evidence is real; else light detail only.
            float atlas_ceil = mix(0.28, 0.95, atlas_primary_g);
            float plate_a = atlas_a_ev
                * clamp(avatar_plate_blend.y, 0.0, 1.0)
                * avatar_recipe.z
                * atlas_ceil
                * (1.0 - open_primary_g * 0.85);
            color = mix(color, plate_rgb, plate_a);
            face_alpha = max(face_alpha, plate_a);
        }

        // L08: upper-face expression plate (surprise brows / wider lids).
        // Bow out under a blink so L09 lids stay single-owner.
        if (avatar_expr_state.w > 0.5 && avatar_expr_state.z > 0.001) {
            vec4 expr = texture(avatar_expr_plate, to_uv(grid_position));
            float expr_a = expr.a * clamp(avatar_expr_state.z, 0.0, 1.0);
            expr_a *= (1.0 - smoothstep(0.08, 0.40, clamp(avatar_eye_state.w, 0.0, 1.0)));
            color = mix(color, expr.rgb, expr_a);
            face_alpha = max(face_alpha, expr_a);
        }

        // L09: eye LOOK region (§14.7) — same playbook as mouth plates.
        // Photographed eyes_closed.png owns the aperture while blinking.
        // Without that plate: do NOT invent lids and do NOT paint skin disks
        // (Alt C looked worse than an open-eye hold on this take).
        float aperture = max(tissue_at(source).a, tissue_at(grid_position).a);
        float blink = clamp(avatar_eye_state.w, 0.0, 1.0);
        float widen = clamp(avatar_expr_state.x, 0.0, 1.0);
        widen *= (1.0 - smoothstep(0.04, 0.22, blink));
        float half_height = max(avatar_eye_shape.y, 1.0) * (1.0 + widen * 0.45);
        float half_width = max(avatar_eye_shape.x, 1.0) * (1.0 + widen * 0.18);
        float eye_mid_x = 0.5 * (avatar_eye_centers.x + avatar_eye_centers.z);
        vec2 centre = grid_position.x < eye_mid_x
            ? avatar_eye_centers.xy
            : avatar_eye_centers.zw;

        if (widen > 0.02 && blink < 0.05 && aperture > 0.004) {
            vec2 gaze = vec2(avatar_eye_state.x, avatar_eye_state.y)
                * avatar_eye_shape.z;
            vec3 globe = photo_at(
                source + vec2(0.0, -widen * half_height * 0.22) - gaze
            );
            color = mix(color, globe, clamp(aperture * widen * 0.55, 0.0, 1.0));
        }

        // Plate path: photographed matte owns the socket (not tissue.a).
        if (blink > 0.04 && avatar_eye_closed_ready == 1) {
            vec2 axis = max(vec2(half_width, half_height), vec2(1.0));
            vec2 eye_d = (grid_position - centre) / (axis * vec2(1.12, 1.28));
            float eye_disk = 1.0 - smoothstep(0.86, 1.10, length(eye_d));
            vec4 ec = texture(avatar_eye_closed_plate, to_uv(grid_position));
            // Flatten plate alpha so iris cannot ghost through soft feather.
            float plate_a = clamp(ec.a * 1.25, 0.0, 1.0);
            plate_a = mix(plate_a, step(0.12, plate_a), smoothstep(0.50, 0.82, blink));
            float ownership = clamp(max(eye_disk * plate_a, plate_a), 0.0, 1.0);
            float lid_w = ownership * smoothstep(0.04, 0.38, blink);
            lid_w = mix(lid_w, ownership, smoothstep(0.40, 0.70, blink));
            color = mix(color, ec.rgb, clamp(lid_w, 0.0, 1.0));
        }

        // L10: brow raise without a plate — display-only, not Master-Lock gated.
        // Mute brow lift inside a blink so lids stay the single owner.
        float brow = clamp(avatar_expr_state.y, 0.0, 1.0);
        brow *= (1.0 - smoothstep(0.08, 0.35, clamp(avatar_eye_state.w, 0.0, 1.0)));
        if (brow > 0.04 && avatar_expr_state.z < 0.15) {
            int part = part_at(grid_position);
            if (part == PART_LEFT_BROW || part == PART_RIGHT_BROW || part == PART_FACE) {
                float brow_band = smoothstep(
                    avatar_eye_centers.y + avatar_eye_shape.y * 1.05,
                    avatar_eye_centers.y + avatar_eye_shape.y * 3.4,
                    grid_position.y
                );
                float lift = brow * 5.2;
                vec3 lifted = photo_at(source - vec2(0.0, lift));
                color = mix(color, lifted, brow_band * brow * 0.72);
            }
        }
    } else {
        // No atlas or tissue maps: show the portrait undeformed rather than
        // inventing motion the biomechanics cannot register against.
        color = photo_at(grid_position);
        face_alpha = smoothstep(0.02, 0.12, max(color.r, max(color.g, color.b)));
    }

    vec3 background = vec3(0.01, 0.01, 0.012);
    color = mix(background, color, max(face_alpha, 0.15));
    color.y += (avatar_breath_phase - 0.5) * 0.0; // keep uniform live

    // Cosmetic grade (prefs) — multiplies unlocked look; does not invent identity RGB.
    color *= clamp(avatar_skin_tint, vec3(0.0), vec3(2.0));
    float eye_r = max(avatar_eye_shape.x, avatar_eye_shape.y) * 1.35;
    float d_l = length(grid_position - avatar_eye_centers.xy);
    float d_r = length(grid_position - avatar_eye_centers.zw);
    float eye_m = max(
        1.0 - smoothstep(eye_r * 0.35, eye_r, d_l),
        1.0 - smoothstep(eye_r * 0.35, eye_r, d_r)
    );
    // Do not tint a closing lid — reads as a flat colored disk.
    eye_m *= (1.0 - smoothstep(0.08, 0.35, clamp(avatar_eye_state.w, 0.0, 1.0)));
    color = mix(color, color * clamp(avatar_eye_tint, vec3(0.0), vec3(2.0)), eye_m * 0.85);
    float makeup = clamp(avatar_makeup_strength, 0.0, 1.0);
    if (makeup > 0.001) {
        float lip = clamp(tissue_at(grid_position).b, 0.0, 1.0);
        color = mix(color, color * vec3(1.08, 0.92, 0.94), lip * makeup * 0.55);
    }

    float locked_edge = lock_boundary(cell_position) * avatar_lock_overlay;
    color = mix(color, vec3(1.0, 0.08, 0.72), locked_edge * 0.85);

    // Keeps viewport_size linked: the runtime sets it every frame, and an
    // optimised-out uniform would make that assignment fail.
    color += viewport_size.x * 0.0;

    fragment_color = vec4(pow(max(color, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
}
