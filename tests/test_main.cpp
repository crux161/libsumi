#include <iostream>
#include <cassert>
#include "../include/sumi/sumi.h"

int main() {
    using namespace sumi;

    // Vector Construction & Arithmetic
    vec3 v1(1.0f, 2.0f, 3.0f);
    vec3 v2(4.0f, 5.0f, 6.0f);
    vec3 v3 = v1 + v2;
    assert(v3.x == 5.0f && v3.y == 7.0f && v3.z == 9.0f);
    std::cout << "Vector arithmetic passed.\n";

    // Matrix Multiplication
    mat4 id(1.0f);
    vec4 v4(1.0f, 2.0f, 3.0f, 1.0f);
    vec4 v4_trans = id * v4;
    assert(v4_trans.x == 1.0f && v4_trans.y == 2.0f);
    std::cout << "Identity transform passed.\n";

    // Translation
    mat4 trans = translate(id, vec3(10.0f, 0.0f, 0.0f));
    vec4 v_moved = trans * v4;
    assert(v_moved.x == 11.0f);
    std::cout << "Translation passed.\n";

    // Cross Product
    vec3 up(0,1,0);
    vec3 right(1,0,0);
    vec3 fwd = cross(right, up);
    assert(fwd.z == 1.0f);
    std::cout << "Cross product passed.\n";

    std::cout << "All Sumi tests passed." << std::endl;
    return 0;
}
