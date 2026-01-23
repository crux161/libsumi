#pragma once

// =============================================================================
// 🍎 APPLE METAL (MSL)
// =============================================================================
#if defined(__METAL_VERSION__)
    #define SUMI_PLATFORM_METAL 1
    
    #include <metal_stdlib>
    
    // In Metal, functions are inline by default in the standard library context
    #define SUMI_CTX inline
    #define SUMI_DEVICE 
    #define SUMI_HOST 
    #define SUMI_CONSTANT constant
    #define SUMI_GLOBAL device
    
    // Enable "Native Mode" to skip custom struct definitions in sumi.h
    #define SUMI_USE_NATIVE_TYPES 1

    namespace sumi {
        // Alias Metal's optimized SIMD types directly
        using vec2 = metal::float2;
        using vec3 = metal::float3;
        using vec4 = metal::float4;
        
        using ivec2 = metal::int2;
        using ivec3 = metal::int3;
        using ivec4 = metal::int4;
        
        using uvec2 = metal::uint2;
        using uvec3 = metal::uint3;
        using uvec4 = metal::uint4;
        
        using mat2 = metal::float2x2;
        using mat3 = metal::float3x3;
        using mat4 = metal::float4x4;
        
        // Map GLSL-style math intrinsics to Metal equivalents
        // Metal usually matches GLSL (e.g., mix, smoothstep), 
        // but explicit mapping ensures consistency.
        using metal::mix;
        using metal::smoothstep;
        using metal::dot;
        using metal::cross;
        using metal::normalize;
        using metal::length;
        using metal::sin;
        using metal::cos;
        using metal::pow;
        using metal::abs;
        using metal::fract;
        using metal::floor;
        using metal::ceil;
        using metal::min;
        using metal::max;
        using metal::clamp;
    }

// =============================================================================
// 🟢 NVIDIA CUDA
// =============================================================================
#elif defined(__CUDACC__)
    #define SUMI_PLATFORM_CUDA 1
    
    #include <cuda_runtime.h>
    #include <math.h>
    
    // Qualifiers to generate code for both Host (CPU) and Device (GPU)
    #define SUMI_CTX __host__ __device__ inline
    #define SUMI_DEVICE __device__
    #define SUMI_HOST __host__
    #define SUMI_CONSTANT __constant__
    #define SUMI_GLOBAL 
    
    // CUDA's built-in float3/float4 are simple data structs without math operators.
    // We do NOT use native types here; libsumi must provide the struct definitions.
    #define SUMI_USE_NATIVE_TYPES 0

// =============================================================================
// 💻 STANDARD C++ (CPU Host)
// =============================================================================
#else
    #define SUMI_PLATFORM_CPU 1
    
    #include <cmath>
    #include <algorithm>
    #include <cstdint>
    
    #define SUMI_CTX inline
    #define SUMI_DEVICE 
    #define SUMI_HOST 
    #define SUMI_CONSTANT const
    #define SUMI_GLOBAL 
    
    #define SUMI_USE_NATIVE_TYPES 0
    
#endif

// =============================================================================
// 🌐 CONSTANTS & COMMON UTILS
// =============================================================================
namespace sumi {
    #ifndef M_PI
        #define M_PI 3.14159265358979323846
    #endif
    
    static const float PI = 3.1415926535f;
    static const float TAU = 6.2831853071f;
    
    // Forward declarations for Non-Native platforms (CPU/CUDA)
    #if !SUMI_USE_NATIVE_TYPES
        struct vec2;
        struct vec3;
        struct vec4;
        struct mat3;
        // mat4, etc. to be added
    #endif
}
