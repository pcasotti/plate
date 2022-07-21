#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.model * vec4(inPosition, 0.0, 1.0);
    fragTexCoord = uv;
}
