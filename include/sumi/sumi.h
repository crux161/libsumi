/**
 * @file sumi.h
 * @brief Sumi (墨) - A lightweight, CUDA-compatible graphics math library.
 * @version 1.0.0
 * @namespace sumi
 */

#pragma once

#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <limits>

// =================================================================================================
// Platform / Compiler Definitions
// =================================================================================================

#ifdef __CUDACC__
    #define SUMI_CTX __host__ __device__
    #define SUMI_INLINE __forceinline__
#else
    #define SUMI_CTX
    #define SUMI_INLINE inline
#endif

// Constant Definitions
#ifndef SUMI_PI
#define SUMI_PI 3.14159265358979323846f
#endif

#ifndef SUMI_EPSILON
#define SUMI_EPSILON 1.192092896e-07F
#endif

namespace sumi {

// =================================================================================================
// Forward Declarations
// =================================================================================================
struct vec2;
struct vec3;
struct vec4;
struct mat3;
struct mat4;
struct Sampler2D;

// =================================================================================================
// Type Definitions
// =================================================================================================

struct vec2 {
    union { float x, r, s; };
    union { float y, g, t; };

    SUMI_CTX vec2() : x(0.0f), y(0.0f) {}
    SUMI_CTX explicit vec2(float s) : x(s), y(s) {}
    SUMI_CTX vec2(float _x, float _y) : x(_x), y(_y) {}
    
    // Accessors
    SUMI_CTX float& operator[](int i) { return (&x)[i]; }
    SUMI_CTX const float& operator[](int i) const { return (&x)[i]; }

    // Unary
    SUMI_CTX vec2 operator-() const { return vec2(-x, -y); }

    // Operators
    SUMI_CTX vec2 operator+(const vec2& v) const { return vec2(x+v.x, y+v.y); }
    SUMI_CTX vec2 operator-(const vec2& v) const { return vec2(x-v.x, y-v.y); }
    SUMI_CTX vec2 operator*(const vec2& v) const { return vec2(x*v.x, y*v.y); }
    SUMI_CTX vec2 operator/(const vec2& v) const { return vec2(x/v.x, y/v.y); }
    SUMI_CTX vec2 operator+(float s) const { return vec2(x+s, y+s); }
    SUMI_CTX vec2 operator-(float s) const { return vec2(x-s, y-s); }
    SUMI_CTX vec2 operator*(float s) const { return vec2(x*s, y*s); }
    SUMI_CTX vec2 operator/(float s) const { return vec2(x/s, y/s); }

    // Assignment
    SUMI_CTX vec2& operator+=(const vec2& v) { x+=v.x; y+=v.y; return *this; }
    SUMI_CTX vec2& operator-=(const vec2& v) { x-=v.x; y-=v.y; return *this; }
    SUMI_CTX vec2& operator*=(const vec2& v) { x*=v.x; y*=v.y; return *this; }
    SUMI_CTX vec2& operator/=(const vec2& v) { x/=v.x; y/=v.y; return *this; }
    SUMI_CTX vec2& operator+=(float s) { x+=s; y+=s; return *this; }
    SUMI_CTX vec2& operator-=(float s) { x-=s; y-=s; return *this; }
    SUMI_CTX vec2& operator*=(float s) { x*=s; y*=s; return *this; }
    SUMI_CTX vec2& operator/=(float s) { x/=s; y/=s; return *this; }

    // Swizzles (Basic)
    SUMI_CTX vec2 xy() const { return vec2(x,y); }
    SUMI_CTX vec2 yx() const { return vec2(y,x); }
};

struct vec3 {
    union { float x, r, s; };
    union { float y, g, t; };
    union { float z, b, p; };

    SUMI_CTX vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    SUMI_CTX explicit vec3(float s) : x(s), y(s), z(s) {}
    SUMI_CTX vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    SUMI_CTX vec3(const vec2& v, float _z) : x(v.x), y(v.y), z(_z) {}
    SUMI_CTX vec3(float _x, const vec2& v) : x(_x), y(v.x), z(v.y) {}

    SUMI_CTX float& operator[](int i) { return (&x)[i]; }
    SUMI_CTX const float& operator[](int i) const { return (&x)[i]; }
    SUMI_CTX vec3 operator-() const { return vec3(-x, -y, -z); }

    SUMI_CTX vec3 operator+(const vec3& v) const { return vec3(x+v.x, y+v.y, z+v.z); }
    SUMI_CTX vec3 operator-(const vec3& v) const { return vec3(x-v.x, y-v.y, z-v.z); }
    SUMI_CTX vec3 operator*(const vec3& v) const { return vec3(x*v.x, y*v.y, z*v.z); }
    SUMI_CTX vec3 operator/(const vec3& v) const { return vec3(x/v.x, y/v.y, z/v.z); }
    SUMI_CTX vec3 operator+(float s) const { return vec3(x+s, y+s, z+s); }
    SUMI_CTX vec3 operator-(float s) const { return vec3(x-s, y-s, z-s); }
    SUMI_CTX vec3 operator*(float s) const { return vec3(x*s, y*s, z*s); }
    SUMI_CTX vec3 operator/(float s) const { return vec3(x/s, y/s, z/s); }

    SUMI_CTX vec3& operator+=(const vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    SUMI_CTX vec3& operator-=(const vec3& v) { x-=v.x; y-=v.y; z-=v.z; return *this; }
    SUMI_CTX vec3& operator*=(const vec3& v) { x*=v.x; y*=v.y; z*=v.z; return *this; }
    SUMI_CTX vec3& operator/=(const vec3& v) { x/=v.x; y/=v.y; z/=v.z; return *this; }
    SUMI_CTX vec3& operator+=(float s) { x+=s; y+=s; z+=s; return *this; }
    SUMI_CTX vec3& operator-=(float s) { x-=s; y-=s; z-=s; return *this; }
    SUMI_CTX vec3& operator*=(float s) { x*=s; y*=s; z*=s; return *this; }
    SUMI_CTX vec3& operator/=(float s) { x/=s; y/=s; z/=s; return *this; }

    // Swizzles
    SUMI_CTX vec2 xy() const { return vec2(x, y); }
    SUMI_CTX vec2 xz() const { return vec2(x, z); }
    SUMI_CTX vec2 yz() const { return vec2(y, z); }
};

struct vec4 {
    union { float x, r, s; };
    union { float y, g, t; };
    union { float z, b, p; };
    union { float w, a, q; };

    SUMI_CTX vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    SUMI_CTX explicit vec4(float s) : x(s), y(s), z(s), w(s) {}
    SUMI_CTX vec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
    SUMI_CTX vec4(const vec3& v, float _w) : x(v.x), y(v.y), z(v.z), w(_w) {}
    SUMI_CTX vec4(const vec2& v, float _z, float _w) : x(v.x), y(v.y), z(_z), w(_w) {}

    SUMI_CTX float& operator[](int i) { return (&x)[i]; }
    SUMI_CTX const float& operator[](int i) const { return (&x)[i]; }
    SUMI_CTX vec4 operator-() const { return vec4(-x, -y, -z, -w); }

    SUMI_CTX vec4 operator+(const vec4& v) const { return vec4(x+v.x, y+v.y, z+v.z, w+v.w); }
    SUMI_CTX vec4 operator-(const vec4& v) const { return vec4(x-v.x, y-v.y, z-v.z, w-v.w); }
    SUMI_CTX vec4 operator*(const vec4& v) const { return vec4(x*v.x, y*v.y, z*v.z, w*v.w); }
    SUMI_CTX vec4 operator/(const vec4& v) const { return vec4(x/v.x, y/v.y, z/v.z, w/v.w); }
    SUMI_CTX vec4 operator+(float s) const { return vec4(x+s, y+s, z+s, w+s); }
    SUMI_CTX vec4 operator-(float s) const { return vec4(x-s, y-s, z-s, w-s); }
    SUMI_CTX vec4 operator*(float s) const { return vec4(x*s, y*s, z*s, w*s); }
    SUMI_CTX vec4 operator/(float s) const { return vec4(x/s, y/s, z/s, w/s); }

    SUMI_CTX vec4& operator+=(const vec4& v) { x+=v.x; y+=v.y; z+=v.z; w+=v.w; return *this; }
    SUMI_CTX vec4& operator-=(const vec4& v) { x-=v.x; y-=v.y; z-=v.z; w-=v.w; return *this; }
    SUMI_CTX vec4& operator*=(const vec4& v) { x*=v.x; y*=v.y; z*=v.z; w*=v.w; return *this; }
    SUMI_CTX vec4& operator/=(const vec4& v) { x/=v.x; y/=v.y; z/=v.z; w/=v.w; return *this; }
    
    SUMI_CTX vec3 xyz() const { return vec3(x,y,z); }
    SUMI_CTX vec3 rgb() const { return vec3(x,y,z); }
    SUMI_CTX vec2 xy() const { return vec2(x,y); }
    SUMI_CTX vec2 zw() const { return vec2(z,w); }
};

// Global Scalar Operators (s * v, s / v, etc.)
inline SUMI_CTX vec2 operator*(float s, const vec2& v) { return vec2(v.x*s, v.y*s); }
inline SUMI_CTX vec3 operator*(float s, const vec3& v) { return vec3(v.x*s, v.y*s, v.z*s); }
inline SUMI_CTX vec4 operator*(float s, const vec4& v) { return vec4(v.x*s, v.y*s, v.z*s, v.w*s); }
inline SUMI_CTX vec2 operator/(float s, const vec2& v) { return vec2(s/v.x, s/v.y); }
inline SUMI_CTX vec3 operator/(float s, const vec3& v) { return vec3(s/v.x, s/v.y, s/v.z); }
inline SUMI_CTX vec4 operator/(float s, const vec4& v) { return vec4(s/v.x, s/v.y, s/v.z, s/v.w); }
inline SUMI_CTX vec2 operator+(float s, const vec2& v) { return vec2(s+v.x, s+v.y); }
inline SUMI_CTX vec3 operator+(float s, const vec3& v) { return vec3(s+v.x, s+v.y, s+v.z); }
inline SUMI_CTX vec4 operator+(float s, const vec4& v) { return vec4(s+v.x, s+v.y, s+v.z, s+v.w); }
inline SUMI_CTX vec2 operator-(float s, const vec2& v) { return vec2(s-v.x, s-v.y); }
inline SUMI_CTX vec3 operator-(float s, const vec3& v) { return vec3(s-v.x, s-v.y, s-v.z); }
inline SUMI_CTX vec4 operator-(float s, const vec4& v) { return vec4(s-v.x, s-v.y, s-v.z, s-v.w); }

struct mat3 {
    vec3 cols[3];
    SUMI_CTX mat3() {
        cols[0] = vec3(1,0,0); cols[1] = vec3(0,1,0); cols[2] = vec3(0,0,1);
    }
    SUMI_CTX explicit mat3(float s) {
        cols[0] = vec3(s,0,0); cols[1] = vec3(0,s,0); cols[2] = vec3(0,0,s);
    }
    SUMI_CTX mat3(const vec3& c0, const vec3& c1, const vec3& c2) {
        cols[0] = c0; cols[1] = c1; cols[2] = c2;
    }
    // Column-major construction
    SUMI_CTX mat3(float x0, float y0, float z0,
                  float x1, float y1, float z1,
                  float x2, float y2, float z2) {
        cols[0] = vec3(x0,y0,z0); cols[1] = vec3(x1,y1,z1); cols[2] = vec3(x2,y2,z2);
    }
    SUMI_CTX vec3& operator[](int i) { return cols[i]; }
    SUMI_CTX const vec3& operator[](int i) const { return cols[i]; }

    SUMI_CTX mat3 operator*(const mat3& m) const {
        mat3 res(0.0f);
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) {
                res[i][j] = cols[0][j]*m[i][0] + cols[1][j]*m[i][1] + cols[2][j]*m[i][2];
            }
        }
        return res;
    }
    SUMI_CTX vec3 operator*(const vec3& v) const {
        return vec3(cols[0]*v.x + cols[1]*v.y + cols[2]*v.z);
    }
};

struct mat4 {
    vec4 cols[4];
    SUMI_CTX mat4() {
        cols[0]=vec4(1,0,0,0); cols[1]=vec4(0,1,0,0); cols[2]=vec4(0,0,1,0); cols[3]=vec4(0,0,0,1);
    }
    SUMI_CTX explicit mat4(float s) {
        cols[0]=vec4(s,0,0,0); cols[1]=vec4(0,s,0,0); cols[2]=vec4(0,0,s,0); cols[3]=vec4(0,0,0,s);
    }
    SUMI_CTX mat4(const vec4& c0, const vec4& c1, const vec4& c2, const vec4& c3) {
        cols[0]=c0; cols[1]=c1; cols[2]=c2; cols[3]=c3;
    }
    SUMI_CTX mat4(float x0, float y0, float z0, float w0,
                  float x1, float y1, float z1, float w1,
                  float x2, float y2, float z2, float w2,
                  float x3, float y3, float z3, float w3) {
        cols[0] = vec4(x0,y0,z0,w0); cols[1] = vec4(x1,y1,z1,w1);
        cols[2] = vec4(x2,y2,z2,w2); cols[3] = vec4(x3,y3,z3,w3);
    }
    
    SUMI_CTX vec4& operator[](int i) { return cols[i]; }
    SUMI_CTX const vec4& operator[](int i) const { return cols[i]; }

    SUMI_CTX mat4 operator*(const mat4& m) const {
        mat4 res(0.0f);
        for(int i=0; i<4; ++i) { // Column of result
            for(int j=0; j<4; ++j) { // Row of result
                res[i][j] = cols[0][j]*m[i][0] + cols[1][j]*m[i][1] + 
                            cols[2][j]*m[i][2] + cols[3][j]*m[i][3];
            }
        }
        return res;
    }
    SUMI_CTX vec4 operator*(const vec4& v) const {
        return vec4(cols[0]*v.x + cols[1]*v.y + cols[2]*v.z + cols[3]*v.w);
    }
};

struct Sampler2D {
    int w, h;
    vec4* data;
};

// =================================================================================================
// Standard Math Functions
// =================================================================================================

// Scalar Wrappers
SUMI_CTX inline float radians(float d) { return d * (SUMI_PI / 180.0f); }
SUMI_CTX inline float degrees(float r) { return r * (180.0f / SUMI_PI); }
SUMI_CTX inline float min(float a, float b) { return (a < b) ? a : b; }
SUMI_CTX inline float max(float a, float b) { return (a > b) ? a : b; }
SUMI_CTX inline float clamp(float x, float minVal, float maxVal) { return min(max(x, minVal), maxVal); }
SUMI_CTX inline float mix(float x, float y, float a) { return x * (1.0f - a) + y * a; }
SUMI_CTX inline float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
SUMI_CTX inline float smoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
SUMI_CTX inline float floor(float x) { return ::floorf(x); }
SUMI_CTX inline float ceil(float x) { return ::ceilf(x); }
SUMI_CTX inline float fract(float x) { return x - floor(x); }
SUMI_CTX inline float mod(float x, float y) { return x - y * floor(x / y); }
SUMI_CTX inline float abs(float x) { return ::fabsf(x); }
SUMI_CTX inline float sign(float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); }
SUMI_CTX inline float sin(float x) { return ::sinf(x); }
SUMI_CTX inline float cos(float x) { return ::cosf(x); }
SUMI_CTX inline float tan(float x) { return ::tanf(x); }
SUMI_CTX inline float asin(float x) { return ::asinf(x); }
SUMI_CTX inline float acos(float x) { return ::acosf(x); }
SUMI_CTX inline float atan(float y, float x) { return ::atan2f(y, x); }
SUMI_CTX inline float sqrt(float x) { return ::sqrtf(x); }
SUMI_CTX inline float exp(float x) { return ::expf(x); }
SUMI_CTX inline float pow(float x, float y) { return ::powf(x, y); }
SUMI_CTX inline float log(float x) { return ::logf(x); }

// Vector Functions (Generics)
#define DEFINE_VEC_OP1(func_name, op_call) \
    SUMI_CTX inline vec2 func_name(const vec2& v) { return vec2(op_call(v.x), op_call(v.y)); } \
    SUMI_CTX inline vec3 func_name(const vec3& v) { return vec3(op_call(v.x), op_call(v.y), op_call(v.z)); } \
    SUMI_CTX inline vec4 func_name(const vec4& v) { return vec4(op_call(v.x), op_call(v.y), op_call(v.z), op_call(v.w)); }

#define DEFINE_VEC_OP2(func_name, op_call) \
    SUMI_CTX inline vec2 func_name(const vec2& a, const vec2& b) { return vec2(op_call(a.x,b.x), op_call(a.y,b.y)); } \
    SUMI_CTX inline vec3 func_name(const vec3& a, const vec3& b) { return vec3(op_call(a.x,b.x), op_call(a.y,b.y), op_call(a.z,b.z)); } \
    SUMI_CTX inline vec4 func_name(const vec4& a, const vec4& b) { return vec4(op_call(a.x,b.x), op_call(a.y,b.y), op_call(a.z,b.z), op_call(a.w,b.w)); }

#define DEFINE_VEC_OP2_SCALAR(func_name, op_call) \
    SUMI_CTX inline vec2 func_name(const vec2& a, float b) { return vec2(op_call(a.x,b), op_call(a.y,b)); } \
    SUMI_CTX inline vec3 func_name(const vec3& a, float b) { return vec3(op_call(a.x,b), op_call(a.y,b), op_call(a.z,b)); } \
    SUMI_CTX inline vec4 func_name(const vec4& a, float b) { return vec4(op_call(a.x,b), op_call(a.y,b), op_call(a.z,b), op_call(a.w,b)); }

DEFINE_VEC_OP1(floor, floor)
DEFINE_VEC_OP1(ceil, ceil)
DEFINE_VEC_OP1(fract, fract)
DEFINE_VEC_OP1(abs, abs)
DEFINE_VEC_OP1(sign, sign)
DEFINE_VEC_OP1(sin, sin)
DEFINE_VEC_OP1(cos, cos)
DEFINE_VEC_OP1(tan, tan)
DEFINE_VEC_OP1(asin, asin)
DEFINE_VEC_OP1(acos, acos)
DEFINE_VEC_OP1(sqrt, sqrt)
DEFINE_VEC_OP1(exp, exp)
DEFINE_VEC_OP1(log, log)

DEFINE_VEC_OP2(min, min)
DEFINE_VEC_OP2(max, max)
DEFINE_VEC_OP2(mod, mod)
DEFINE_VEC_OP2(pow, pow)

DEFINE_VEC_OP2_SCALAR(min, min)
DEFINE_VEC_OP2_SCALAR(max, max)
DEFINE_VEC_OP2_SCALAR(mod, mod)
DEFINE_VEC_OP2_SCALAR(pow, pow)

// Clamp specializations
SUMI_CTX inline vec2 clamp(const vec2& v, float minVal, float maxVal) { return vec2(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal)); }
SUMI_CTX inline vec3 clamp(const vec3& v, float minVal, float maxVal) { return vec3(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal), clamp(v.z, minVal, maxVal)); }
SUMI_CTX inline vec4 clamp(const vec4& v, float minVal, float maxVal) { return vec4(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal), clamp(v.z, minVal, maxVal), clamp(v.w, minVal, maxVal)); }

// Mix specializations
SUMI_CTX inline vec2 mix(const vec2& x, const vec2& y, float a) { return x * (1.0f - a) + y * a; }
SUMI_CTX inline vec3 mix(const vec3& x, const vec3& y, float a) { return x * (1.0f - a) + y * a; }
SUMI_CTX inline vec4 mix(const vec4& x, const vec4& y, float a) { return x * (1.0f - a) + y * a; }

// Geometric Functions
SUMI_CTX inline float dot(const vec2& a, const vec2& b) { return a.x*b.x + a.y*b.y; }
SUMI_CTX inline float dot(const vec3& a, const vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
SUMI_CTX inline float dot(const vec4& a, const vec4& b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

SUMI_CTX inline vec3 cross(const vec3& a, const vec3& b) {
    return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

SUMI_CTX inline float length(const vec2& v) { return sqrt(dot(v, v)); }
SUMI_CTX inline float length(const vec3& v) { return sqrt(dot(v, v)); }
SUMI_CTX inline float length(const vec4& v) { return sqrt(dot(v, v)); }

SUMI_CTX inline float distance(const vec3& a, const vec3& b) { return length(a - b); }
SUMI_CTX inline float distance(const vec2& a, const vec2& b) { return length(a - b); }

SUMI_CTX inline vec2 normalize(const vec2& v) { float l = length(v); return l > SUMI_EPSILON ? v / l : vec2(0.0f); }
SUMI_CTX inline vec3 normalize(const vec3& v) { float l = length(v); return l > SUMI_EPSILON ? v / l : vec3(0.0f); }
SUMI_CTX inline vec4 normalize(const vec4& v) { float l = length(v); return l > SUMI_EPSILON ? v / l : vec4(0.0f); }

SUMI_CTX inline vec3 reflect(const vec3& I, const vec3& N) {
    return I - 2.0f * dot(N, I) * N;
}

SUMI_CTX inline vec3 refract(const vec3& I, const vec3& N, float eta) {
    float k = 1.0f - eta * eta * (1.0f - dot(N, I) * dot(N, I));
    if (k < 0.0f) return vec3(0.0f);
    else return eta * I - (eta * dot(N, I) + sqrt(k)) * N;
}

// Matrix Transforms
SUMI_CTX inline mat4 translate(const mat4& m, const vec3& v) {
    mat4 r = m;
    r.cols[3] = m.cols[0]*v.x + m.cols[1]*v.y + m.cols[2]*v.z + m.cols[3];
    return r;
}

SUMI_CTX inline mat4 scale(const mat4& m, const vec3& v) {
    mat4 r;
    r.cols[0] = m.cols[0] * v.x;
    r.cols[1] = m.cols[1] * v.y;
    r.cols[2] = m.cols[2] * v.z;
    r.cols[3] = m.cols[3];
    return r;
}

SUMI_CTX inline mat4 rotate(const mat4& m, float angle, const vec3& axis) {
    float c = cos(angle);
    float s = sin(angle);
    vec3 a = normalize(axis);
    vec3 temp = a * (1.0f - c);

    mat4 R; // Identity by default
    R[0][0] = c + temp.x * a.x;
    R[0][1] = temp.x * a.y + s * a.z;
    R[0][2] = temp.x * a.z - s * a.y;

    R[1][0] = temp.y * a.x - s * a.z;
    R[1][1] = c + temp.y * a.y;
    R[1][2] = temp.y * a.z + s * a.x;

    R[2][0] = temp.z * a.x + s * a.y;
    R[2][1] = temp.z * a.y - s * a.x;
    R[2][2] = c + temp.z * a.z;

    // Last row/col remains identity for rotation matrix R
    // R[3][3] is 1.0 from constructor
    return m * R;
}

SUMI_CTX inline mat4 perspective(float fovy, float aspect, float zNear, float zFar) {
    float tanHalfFovy = tan(fovy / 2.0f);
    mat4 r(0.0f);
    r[0][0] = 1.0f / (aspect * tanHalfFovy);
    r[1][1] = 1.0f / (tanHalfFovy);
    r[2][2] = -(zFar + zNear) / (zFar - zNear);
    r[2][3] = -1.0f;
    r[3][2] = -(2.0f * zFar * zNear) / (zFar - zNear);
    return r;
}

SUMI_CTX inline mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up) {
    vec3 f(normalize(center - eye));
    vec3 s(normalize(cross(f, up)));
    vec3 u(cross(s, f));

    mat4 r(1.0f);
    r[0][0] = s.x; r[1][0] = s.y; r[2][0] = s.z;
    r[0][1] = u.x; r[1][1] = u.y; r[2][1] = u.z;
    r[0][2] =-f.x; r[1][2] =-f.y; r[2][2] =-f.z;
    r[3][0] =-dot(s, eye);
    r[3][1] =-dot(u, eye);
    r[3][2] = dot(f, eye);
    return r;
}

// Matrix Operations
SUMI_CTX inline mat3 transpose(const mat3& m) {
    return mat3(
        m[0][0], m[1][0], m[2][0],
        m[0][1], m[1][1], m[2][1],
        m[0][2], m[1][2], m[2][2]
    );
}

SUMI_CTX inline mat4 transpose(const mat4& m) {
    return mat4(
        m[0][0], m[1][0], m[2][0], m[3][0],
        m[0][1], m[1][1], m[2][1], m[3][1],
        m[0][2], m[1][2], m[2][2], m[3][2],
        m[0][3], m[1][3], m[2][3], m[3][3]
    );
}

// Texture Sampling (Basic CPU Implementation)
SUMI_CTX inline vec4 texture(const Sampler2D& sampler, const vec2& uv) {
    if (!sampler.data) return vec4(0.0f, 0.0f, 0.0f, 1.0f);
    
    float fx = uv.x - floor(uv.x);
    float fy = uv.y - floor(uv.y);
    if(fx < 0) fx += 1.0f;
    if(fy < 0) fy += 1.0f;

    int tx = (int)(fx * sampler.w);
    int ty = (int)(fy * sampler.h);

    if(tx < 0) tx = 0;
    if(tx >= sampler.w) tx = sampler.w - 1;
    
    if(ty < 0) ty = 0; 
    if(ty >= sampler.h) ty = sampler.h - 1;

    return sampler.data[ty * sampler.w + tx];
}

// IO Operators (Host only)
inline std::ostream& operator<<(std::ostream& os, const vec2& v) {
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const vec3& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const vec4& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    return os;
}

} // namespace sumi

// Alias for ease of migration/use
namespace glsl = sumi;
