#pragma once
#include "sumi.h"

namespace sumi {

// A stateless hash function: Input (x,y) -> Output (Random Float 0..1)
// This is "gold" for shaders because it requires no texture lookups.
SUMI_CTX inline float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031f);
    p3 += dot(p3, p3.yzx() + 33.33f);
    return fract((p3.x + p3.y) * p3.z);
}

// 2D Noise
SUMI_CTX inline float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    // Four corners in 2D of a tile
    float a = hash(i);
    float b = hash(i + vec2(1.0f, 0.0f));
    float c = hash(i + vec2(0.0f, 1.0f));
    float d = hash(i + vec2(1.0f, 1.0f));

    // Cubic Hermite Interpolation (same as smoothstep)
    vec2 u = f * f * (3.0f - 2.0f * f);

    // Mix 4 corners percentages
    return mix(a, b, u.x) + 
           (c - a)* u.y * (1.0f - u.x) + 
           (d - b) * u.x * u.y;
}

// Fractal Brownian Motion (FBM) - The "Cloudy" look
SUMI_CTX inline float fbm(vec2 p, int octaves) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 0.0f;
    
    // Unrolling loops is safer for older GPU compilers, 
    // but for "sumi-ese" we can trust the loop.
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise(p);
        p *= 2.0f;
        amplitude *= 0.5f;
    }
    return value;
}

} // namespace sumi
