#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 proj;
    mat4 view;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.proj * ubo.view * vec4(inPosition, 1.0);
    fragTexCoord = uv;
}
