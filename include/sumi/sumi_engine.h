#pragma once
#include "sumi.h"

namespace sumi {

struct Camera {
    vec3 position;
    vec3 target;
    vec3 up;
    float fov;         // In radians
    float aspectRatio;
    float nearClip;
    float farClip;

    SUMI_CTX Camera() 
        : position(0, 0, 5), target(0, 0, 0), up(0, 1, 0), 
          fov(radians(45.0f)), aspectRatio(16.0f/9.0f), 
          nearClip(0.1f), farClip(100.0f) {}

    // 1. The View Matrix (World Space -> Camera Space)
    SUMI_CTX mat4 getViewMatrix() const {
        // Recalculate basis vectors
        vec3 f = normalize(target - position); // Forward
        vec3 r = normalize(cross(f, up));      // Right
        vec3 u = cross(r, f);                  // True Up

        mat4 view(1.0f);
        view[0][0] = r.x; view[1][0] = r.y; view[2][0] = r.z;
        view[0][1] = u.x; view[1][1] = u.y; view[2][1] = u.z;
        view[0][2] =-f.x; view[1][2] =-f.y; view[2][2] =-f.z;
        view[3][0] =-dot(r, position);
        view[3][1] =-dot(u, position);
        view[3][2] = dot(f, position);
        return view;
    }

    // 2. The Projection Matrix (Camera Space -> Clip Space)
    SUMI_CTX mat4 getProjectionMatrix() const {
        float tanHalfFovy = tan(fov / 2.0f);
        mat4 proj(0.0f);
        
        // Basic Perspective Setup
        proj[0][0] = 1.0f / (aspectRatio * tanHalfFovy);
        proj[1][1] = 1.0f / (tanHalfFovy);
        proj[2][2] = -(farClip + nearClip) / (farClip - nearClip);
        proj[2][3] = -1.0f;
        proj[3][2] = -(2.0f * farClip * nearClip) / (farClip - nearClip);
        
        return proj;
    }
};

} // namespace sumi
