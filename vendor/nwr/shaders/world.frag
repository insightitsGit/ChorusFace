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

in vec2 uv;
layout(location = 0) out vec4 fragment_color;

uint cell_index(ivec2 position) {
    ivec2 bounded = clamp(position, ivec2(0), ivec2(grid_size) - ivec2(1));
    return uint(bounded.y) * grid_size.x + uint(bounded.x);
}

float boundary_signal(ivec2 position) {
    uint index = cell_index(position);
    float density = abs(world.cells[index].value[3]);
    float hard_surface = abs(world.cells[index].value[24]);
    float lock = world.cells[index].value[HUMAN_LOCK_CHANNEL];
    return clamp(density * 0.65 + hard_surface * 0.8 + lock, 0.0, 2.0);
}

float sobel_strength(ivec2 position) {
    float top_left = boundary_signal(position + ivec2(-1, 1));
    float top = boundary_signal(position + ivec2(0, 1));
    float top_right = boundary_signal(position + ivec2(1, 1));
    float left = boundary_signal(position + ivec2(-1, 0));
    float right = boundary_signal(position + ivec2(1, 0));
    float bottom_left = boundary_signal(position + ivec2(-1, -1));
    float bottom = boundary_signal(position + ivec2(0, -1));
    float bottom_right = boundary_signal(position + ivec2(1, -1));
    float gradient_x =
        -top_left + top_right
        - 2.0 * left + 2.0 * right
        - bottom_left + bottom_right;
    float gradient_y =
        top_left + 2.0 * top + top_right
        - bottom_left - 2.0 * bottom - bottom_right;
    return length(vec2(gradient_x, gradient_y));
}

float weight_at(int row, int column) {
    return texelFetch(material_weights, ivec2(column, row), 0).r;
}

vec3 neural_material(float state[CHANNELS]) {
    float hidden[16];
    for (int unit = 0; unit < 16; ++unit) {
        float total = weight_at(unit, CHANNELS);
        for (int channel = 0; channel < CHANNELS; ++channel) {
            total += weight_at(unit, channel) * state[channel];
        }
        hidden[unit] = max(total, 0.0);
    }
    vec3 output_color = vec3(0.0);
    for (int component = 0; component < 3; ++component) {
        int row = 16 + component;
        float total = weight_at(row, 16);
        for (int unit = 0; unit < 16; ++unit) {
            total += weight_at(row, unit) * hidden[unit];
        }
        output_color[component] = total;
    }
    return max(output_color, vec3(0.0));
}

vec3 procedural_material(float state[CHANNELS]) {
    vec3 albedo = max(vec3(state[8], state[9], state[10]), vec3(0.0));
    float opacity = clamp(state[11], 0.0, 1.0);
    float emission = max(state[14], 0.0);
    float energy = max(state[7], 0.0);
    float intent = group_norm(state, 2);
    float glow = emission * 1.8 + energy * 0.65 + intent * 0.35;
    vec3 tint = mix(albedo, vec3(0.12, 0.72, 1.0), 0.25);
    return albedo * (0.2 + opacity * 0.8) + tint * glow;
}

void main() {
    vec2 grid_position = uv * vec2(grid_size);
    ivec2 cell_position = clamp(
        ivec2(floor(grid_position)),
        ivec2(0),
        ivec2(grid_size) - ivec2(1)
    );
    uint index = cell_index(cell_position);

    float state[CHANNELS];
    for (int channel = 0; channel < CHANNELS; ++channel) {
        state[channel] = world.cells[index].value[channel];
    }

    vec3 material_layer =
        use_neural_material == 1 && weight_rows == 19
            ? neural_material(state)
            : procedural_material(state);
    material_layer = vec3(1.0) - exp(-material_layer * 1.35);
    material_layer += vec3(0.002, 0.004, 0.012);

    float gradient = sobel_strength(cell_position);
    vec2 within_cell = fract(grid_position);
    vec2 pixels_per_cell = viewport_size / vec2(grid_size);
    float cell_edge_distance = min(
        min(within_cell.x, 1.0 - within_cell.x) * pixels_per_cell.x,
        min(within_cell.y, 1.0 - within_cell.y) * pixels_per_cell.y
    );
    float one_pixel_line = 1.0 - smoothstep(0.65, 1.35, cell_edge_distance);
    float edge = smoothstep(0.12, 0.9, gradient) * one_pixel_line;
    vec3 edge_color = mix(
        vec3(0.02, 0.95, 1.0),
        vec3(1.0, 0.08, 0.78),
        state[HUMAN_LOCK_CHANNEL]
    );
    vec3 composited = mix(material_layer, edge_color, edge);
    composited += edge_color * edge * 0.35;

    fragment_color = vec4(pow(max(composited, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
}
