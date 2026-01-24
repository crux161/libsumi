cmake_minimum_required(VERSION 3.14)
project(sumi VERSION 1.0.0 LANGUAGES CXX)

# ==============================================================================
# 1. Library Definition (The Upstream Target)
# ==============================================================================
# Define 'sumi' as an Interface library (Header-only)
add_library(sumi INTERFACE)

# Define where headers are located
target_include_directories(sumi INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Enforce C++17 Standard for any project using Sumi
target_compile_features(sumi INTERFACE cxx_std_17)

# Optional: Add compiler warnings if this is the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    if(MSVC)
        target_compile_options(sumi INTERFACE /W4)
    else()
        target_compile_options(sumi INTERFACE -Wall -Wextra -Wpedantic)
    endif()
endif()

# ==============================================================================
# 2. Testing Configuration (Only builds if this is the main project)
# ==============================================================================
option(SUMI_BUILD_TESTS "Build the test suite" ON)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND SUMI_BUILD_TESTS)
    enable_testing()

    # Link the test against the library (pulls in include paths and C++17 req)
    target_link_libraries(sumi_test PRIVATE sumi)

    # Register the test with CTest
    add_test(NAME CoreVerification COMMAND sumi_test)
endif()

# ==============================================================================
# 3. Installation Logic (For system-wide or vcpkg use)
# ==============================================================================
include(GNUInstallDirs)

install(TARGETS sumi
    EXPORT sumiTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/sumi DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT sumiTargets
    FILE sumiConfig.cmake
    NAMESPACE sumi::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sumi
)
