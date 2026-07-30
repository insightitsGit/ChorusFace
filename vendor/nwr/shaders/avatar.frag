#version 430

//#prelude

layout(std430, binding = 0) readonly restrict buffer WorldState {
    Cell cells[];
} world;

uniform uvec2 grid_size;
uniform vec2 viewport_size;
uniform int use_neural_material;
uniform sampler2D material_weights;
uniform int weight_rows;

// RGB = photograph, A = part-id / 10
uniform sampler2D avatar_base_color;
uniform vec2 avatar_mouth_center;
// x = width, y = openness, z = roundness, w = smile/frown.
uniform vec4 avatar_mouth_pose;
uniform float avatar_lock_overlay;
uniform float avatar_jaw_angle;
uniform vec4 avatar_eye_state;      // gaze_x, gaze_y, pupil, blink
uniform vec2 avatar_brow_state;     // raise, knit
uniform float avatar_breath_phase;
uniform int avatar_debug_view;
uniform float avatar_muscle_heat;
// Piece transforms: upper_dy, lower_dy, width_pull, reserved
uniform vec4 avatar_lip_parts;
// Eye piece centres in grid space (left.xy, right.xy)
uniform vec4 avatar_eye_centers;
uniform vec4 avatar_brow_centers;
uniform int avatar_use_parts;

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

int part_at(vec2 sample_uv) {
    float encoded = texture(avatar_base_color, clamp(sample_uv, 0.0, 1.0)).a;
    return int(round(encoded * PART_SCALE));
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

void main() {
    vec2 grid_position = uv * vec2(grid_size);
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
        float jaw = clamp(avatar_jaw_angle / 0.55, 0.0, 1.0);
        float band = 1.0 - smoothstep(8.0, 28.0, distance(grid_position, avatar_mouth_center));
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
        int part = part_at(uv);
        vec3 photo = texture(avatar_base_color, uv).rgb;
        fragment_color = vec4(mix(photo, part_debug_color(part), 0.72), 1.0);
        return;
    }

    float width = max(avatar_mouth_pose.x, 3.0);
    float openness = max(avatar_mouth_pose.y, 0.7);
    float roundness = clamp(avatar_mouth_pose.z, 0.0, 1.0);
    float expression = clamp(avatar_mouth_pose.w, -1.0, 1.0);
    float open_amount = clamp((openness - 1.0) / 11.0, 0.0, 1.0);
    open_amount = max(open_amount, clamp(avatar_jaw_angle / 0.55, 0.0, 1.0) * 0.85);

    float upper_dy = avatar_lip_parts.x; // grid cells, positive = up
    float lower_dy = avatar_lip_parts.y; // grid cells, positive = down
    float width_pull = avatar_lip_parts.z;
    float smile = expression * 1.8;

    vec2 inv_grid = 1.0 / vec2(grid_size);
    vec4 base = texture(avatar_base_color, uv);
    vec3 color = base.rgb;
    float face_alpha = smoothstep(0.02, 0.12, max(base.r, max(base.g, base.b)));

    if (avatar_use_parts == 1) {
        // Inverse-map each movable piece so lips and eyes are real photo cutouts.
        float nx = clamp((grid_position.x - avatar_mouth_center.x) / max(width, 1.0), -1.0, 1.0);
        vec2 upper_rest = uv - vec2(smile * nx * inv_grid.x, upper_dy * inv_grid.y);
        vec2 lower_rest = uv - vec2(-smile * 0.25 * nx * inv_grid.x, -lower_dy * inv_grid.y);
        // Horizontal stretch for EE / OO around the mouth centre.
        vec2 stretch_rest = avatar_mouth_center * inv_grid;
        upper_rest.x = mix(upper_rest.x, stretch_rest.x + (upper_rest.x - stretch_rest.x) / max(1.0 - width_pull, 0.55), open_amount);
        lower_rest.x = mix(lower_rest.x, stretch_rest.x + (lower_rest.x - stretch_rest.x) / max(1.0 - width_pull, 0.55), open_amount);

        int upper_part = part_at(upper_rest);
        int lower_part = part_at(lower_rest);
        int here = part_at(uv);

        vec2 mouth_local = grid_position - avatar_mouth_center;
        float mouth_r = length(mouth_local / vec2(max(width, 1.0), max(3.5 + open_amount * 7.0, 1.0)));
        bool near_mouth = mouth_r < 1.45;
        if (near_mouth) {
            if (upper_part == PART_UPPER_LIP) {
                color = texture(avatar_base_color, clamp(upper_rest, 0.0, 1.0)).rgb;
            } else if (lower_part == PART_LOWER_LIP) {
                color = texture(avatar_base_color, clamp(lower_rest, 0.0, 1.0)).rgb;
            } else if (
                here == PART_UPPER_LIP
                || here == PART_LOWER_LIP
                || here == PART_MOUTH_CAVITY
            ) {
                // Only darken cells that were lips/cavity at rest — the gap the
                // pieces left behind. Do not paint a big dark disc over skin.
                float gap = clamp(open_amount * 1.05, 0.0, 1.0);
                color = mix(color, vec3(0.025, 0.01, 0.012), gap * (1.0 - smoothstep(0.7, 1.35, mouth_r)));
            }
        }

        // Eyes: sample the eye piece with a gaze offset; blink shrinks lids.
        vec2 gaze = vec2(avatar_eye_state.x, avatar_eye_state.y) * 2.8 * inv_grid;
        float blink = clamp(avatar_eye_state.w, 0.0, 1.0);
        for (int eye = 0; eye < 2; ++eye) {
            vec2 center = eye == 0 ? avatar_eye_centers.xy : avatar_eye_centers.zw;
            float dist = distance(grid_position, center);
            if (dist < 16.0) {
                vec2 eye_rest = uv - gaze;
                // Blink pulls samples from above the eye (lid tissue).
                eye_rest.y -= blink * 5.0 * inv_grid.y;
                int eye_part = part_at(eye_rest);
                int want = eye == 0 ? PART_LEFT_EYE : PART_RIGHT_EYE;
                if (eye_part == want) {
                    vec3 eye_color = texture(avatar_base_color, clamp(eye_rest, 0.0, 1.0)).rgb;
                    float cover = 1.0 - smoothstep(11.0, 16.0, dist);
                    color = mix(color, eye_color, cover);
                    // Pupil emphasis near iris centre.
                    float pupil = clamp(avatar_eye_state.z, 0.15, 0.95);
                    float iris = 1.0 - smoothstep(2.0, 5.5 * pupil, distance(grid_position, center + gaze * vec2(grid_size)));
                    color = mix(color, color * 0.35, iris * 0.35 * (1.0 - blink));
                } else if (blink > 0.15 && dist < 12.0) {
                    color *= 1.0 - blink * 0.55 * (1.0 - smoothstep(8.0, 12.0, dist));
                }
            }
        }

        // Brows rise / knit as rigid pieces.
        float raise = avatar_brow_state.x * 4.5;
        float knit = avatar_brow_state.y * 3.0;
        for (int brow = 0; brow < 2; ++brow) {
            vec2 center = brow == 0 ? avatar_brow_centers.xy : avatar_brow_centers.zw;
            float dist = distance(grid_position, center);
            if (dist < 18.0) {
                float side = brow == 0 ? 1.0 : -1.0;
                vec2 brow_rest = uv - vec2(side * knit * inv_grid.x, raise * inv_grid.y);
                int brow_part = part_at(brow_rest);
                int want = brow == 0 ? PART_LEFT_BROW : PART_RIGHT_BROW;
                if (brow_part == want) {
                    vec3 brow_color = texture(avatar_base_color, clamp(brow_rest, 0.0, 1.0)).rgb;
                    float cover = 1.0 - smoothstep(12.0, 18.0, dist);
                    color = mix(color, brow_color, cover);
                }
            }
        }
    } else {
        // Legacy whole-face warp fallback when no part atlas is available.
        vec2 local = grid_position - avatar_mouth_center;
        float mouth_influence = unlocked * (1.0 - smoothstep(width, width * 1.4, length(local)));
        float lip_side = smoothstep(-0.45, 0.45, local.y) * 2.0 - 1.0;
        vec2 warped = local;
        warped.y -= lip_side * open_amount * 3.6 * mouth_influence;
        vec2 sample_uv = clamp((avatar_mouth_center + warped) / vec2(grid_size), 0.0, 1.0);
        color = texture(avatar_base_color, mix(uv, sample_uv, mouth_influence)).rgb;
    }

    vec3 background = vec3(0.01, 0.01, 0.012);
    color = mix(background, color, max(face_alpha, 0.15));
    color.y += (avatar_breath_phase - 0.5) * 0.0; // keep uniform live

    float locked_edge = lock_boundary(cell_position) * avatar_lock_overlay;
    color = mix(color, vec3(1.0, 0.08, 0.72), locked_edge * 0.85);

    if (use_neural_material == 2 && weight_rows > 0) {
        color += texelFetch(material_weights, ivec2(0), 0).rrr * 0.000001;
    }
    color += viewport_size.x * 0.0;

    fragment_color = vec4(pow(max(color, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
}
