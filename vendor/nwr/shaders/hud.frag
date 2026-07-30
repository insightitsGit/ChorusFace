#version 430

//#prelude

uniform sampler2D hud_texture;

in vec2 uv;
layout(location = 0) out vec4 fragment_color;

void main() {
    // HUD texture is top-down; OpenGL UV origin is bottom-left.
    vec4 texel = texture(hud_texture, vec2(uv.x, 1.0 - uv.y));
    if (texel.a < 0.04) {
        discard;
    }
    fragment_color = texel;
}
