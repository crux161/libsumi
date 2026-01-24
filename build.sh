#!/bin/bash

# 1. Setup - Adjust these paths to your actual source files
SRC_FILES="sumi.cpp"  # Add all your .cpp files here
INCLUDE_PATH="include/sumi"                  # Where your .h files are
OUTPUT_NAME="libsumi"

# Cleanup previous builds
rm -f *.o *.a

echo "Building for Intel (x86_64)..."
# -c means "compile to object code only", don't link yet
clang++ -c $SRC_FILES \
    -target x86_64-apple-macos11 \
    -I$INCLUDE_PATH \
    -std=c++17 \
    -o sumi_x86.o

# Create the Intel archive (.a)
ar rcs libsumi_x86.a sumi_x86.o

echo "Building for Apple Silicon (arm64)..."
clang++ -c $SRC_FILES \
    -target arm64-apple-macos11 \
    -I$INCLUDE_PATH \
    -std=c++17 \
    -o sumi_arm.o

# Create the ARM archive (.a)
ar rcs libsumi_arm.a sumi_arm.o

echo "Creating Universal Binary..."
# Stitch the two archives together
lipo -create -output libsumi.a libsumi_x86.a libsumi_arm.a

# Cleanup intermediate files
rm *_x86.o *_arm.o *_x86.a *_arm.a

echo "✅ Success! 'libsumi.a' is now a universal static library."
file libsumi.a
