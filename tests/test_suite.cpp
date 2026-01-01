#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

// Include the library
#include "sumi/sumi.h"

// ============================================================================
// Mini Unit-Test Framework
// ============================================================================
int g_tests_run = 0;
int g_tests_passed = 0;
int g_tests_failed = 0;

#define ASSERT_TRUE(condition, msg) \
    do { \
        g_tests_run++; \
        if((condition)) { \
            g_tests_passed++; \
        } else { \
            g_tests_failed++; \
            std::cerr << "[\033[1;31mFAIL\033[0m] " << __FUNCTION__ << ": " << msg << std::endl; \
        } \
    } while(0)

#define ASSERT_NEAR(a, b, epsilon, msg) \
    ASSERT_TRUE(std::abs((a) - (b)) < (epsilon), msg)

// FIXED: Renamed parameters x,y,z to val_x, val_y, val_z to avoid collision with v.x, v.y, v.z
#define ASSERT_VEC3_EQ(v, val_x, val_y, val_z, msg) \
    ASSERT_TRUE(std::abs(v.x - val_x) < 0.001f && std::abs(v.y - val_y) < 0.001f && std::abs(v.z - val_z) < 0.001f, msg)

void print_summary() {
    std::cout << "========================================" << std::endl;
    std::cout << "Tests Run:    " << g_tests_run << std::endl;
    std::cout << "Tests Passed: \033[1;32m" << g_tests_passed << "\033[0m" << std::endl;
    if (g_tests_failed > 0) {
        std::cout << "Tests Failed: \033[1;31m" << g_tests_failed << "\033[0m" << std::endl;
        exit(1);
    } else {
        std::cout << "Status:       \033[1;32mALL CLEAR\033[0m" << std::endl;
    }
    std::cout << "========================================" << std::endl;
}

// ============================================================================
// Test Cases
// ============================================================================

void test_vec_constructors() {
    using namespace sumi;
    vec3 v1(1.0f);
    ASSERT_TRUE(v1.x == 1.0f && v1.y == 1.0f && v1.z == 1.0f, "vec3(s) constructor");

    vec3 v2(1.0f, 2.0f, 3.0f);
    ASSERT_TRUE(v2.r == 1.0f && v2.g == 2.0f && v2.b == 3.0f, "vec3 union access (r,g,b)");
    
    vec2 v2d(5.0f, 6.0f);
    vec3 v3_from_2(v2d, 7.0f);
    ASSERT_TRUE(v3_from_2.x == 5.0f && v3_from_2.z == 7.0f, "vec3(vec2, float) constructor");
}

void test_vec_arithmetic() {
    using namespace sumi;
    vec3 a(1, 2, 3);
    vec3 b(4, 5, 6);

    vec3 c = a + b;
    ASSERT_VEC3_EQ(c, 5.0f, 7.0f, 9.0f, "vec3 addition");

    vec3 d = b - a;
    ASSERT_VEC3_EQ(d, 3.0f, 3.0f, 3.0f, "vec3 subtraction");

    vec3 e = a * 2.0f;
    ASSERT_VEC3_EQ(e, 2.0f, 4.0f, 6.0f, "vec3 scalar mult");

    // Test operator precedence/scalar order
    vec3 f = 2.0f * a; 
    ASSERT_VEC3_EQ(f, 2.0f, 4.0f, 6.0f, "scalar * vec3");
}

void test_glsl_math() {
    using namespace sumi;
    
    // Clamp
    float c1 = clamp(10.0f, 0.0f, 1.0f);
    ASSERT_NEAR(c1, 1.0f, 0.0001f, "clamp upper");
    float c2 = clamp(-5.0f, 0.0f, 1.0f);
    ASSERT_NEAR(c2, 0.0f, 0.0001f, "clamp lower");

    // Mix
    float m = mix(0.0f, 10.0f, 0.5f);
    ASSERT_NEAR(m, 5.0f, 0.0001f, "mix scalar linear interpolation");

    // Step
    float s1 = step(0.5f, 0.4f); // 0.4 < 0.5 -> 0
    ASSERT_NEAR(s1, 0.0f, 0.0001f, "step returns 0");
    float s2 = step(0.5f, 0.6f); // 0.6 >= 0.5 -> 1
    ASSERT_NEAR(s2, 1.0f, 0.0001f, "step returns 1");

    // Length / Dot
    vec3 up(0, 1, 0);
    vec3 right(1, 0, 0);
    ASSERT_NEAR(dot(up, right), 0.0f, 0.0001f, "dot product orthogonal");
    ASSERT_NEAR(length(vec3(3, 4, 0)), 5.0f, 0.0001f, "length 3-4-5 triangle");
    
    // Cross
    vec3 fwd = cross(right, up); // Right handed system: X cross Y = Z
    ASSERT_VEC3_EQ(fwd, 0.0f, 0.0f, 1.0f, "cross product");
}

void test_matrix_transforms() {
    using namespace sumi;

    mat4 identity(1.0f);
    vec4 p(1.0f, 0.0f, 0.0f, 1.0f); // Point at X=1

    // Translation
    mat4 T = translate(identity, vec3(2.0f, 0.0f, 0.0f));
    vec4 p_trans = T * p; 
    // New pos should be 1+2 = 3
    ASSERT_TRUE(std::abs(p_trans.x - 3.0f) < 0.001f, "mat4 translate");

    // Scale
    mat4 S = scale(identity, vec3(2.0f, 2.0f, 2.0f));
    vec4 p_scale = S * p;
    ASSERT_TRUE(std::abs(p_scale.x - 2.0f) < 0.001f, "mat4 scale");

    // Rotation (Rotate 90 deg around Z)
    // (1,0,0) -> (0,1,0)
    mat4 R = rotate(identity, radians(90.0f), vec3(0, 0, 1));
    vec4 p_rot = R * p;
    
    ASSERT_NEAR(p_rot.x, 0.0f, 0.001f, "mat4 rotate 90deg (x comp)");
    ASSERT_NEAR(p_rot.y, 1.0f, 0.001f, "mat4 rotate 90deg (y comp)");
}

void test_camera_math() {
    using namespace sumi;
    
    // LookAt check
    vec3 eye(0, 0, 10);
    vec3 center(0, 0, 0);
    vec3 up(0, 1, 0);
    
    mat4 view = lookAt(eye, center, up);
    vec4 p_world(0, 0, 0, 1);
    vec4 p_view = view * p_world;
    
    // In view space, if camera is at (0,0,10) looking at (0,0,0), 
    // the object at (0,0,0) should be at Z = -10 (standard OpenGL convention, camera looks down -Z)
    ASSERT_NEAR(p_view.z, -10.0f, 0.001f, "lookAt transformation Z-depth");
}

int main() {
    std::cout << "Sumi (墨) v1.0 - Verification Suite" << std::endl;
    
    test_vec_constructors();
    test_vec_arithmetic();
    test_glsl_math();
    test_matrix_transforms();
    test_camera_math();

    print_summary();
    return 0;
}
