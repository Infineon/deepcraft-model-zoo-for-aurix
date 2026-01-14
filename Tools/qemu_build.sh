#!/bin/bash

# Copyright (c) 2025, Infineon Technologies AG, or an affiliate of Infineon Technologies AG. All rights reserved.
#
# This software, associated documentation and materials ("Software") is owned by Infineon Technologies AG or one 
# of its affiliates ("Infineon") and is protected by and subject to worldwide patent protection, worldwide copyright laws, 
# and international treaty provisions. Therefore, you may use this Software only as provided in the license agreement accompanying 
# the software package from which you obtained this Software. If no license agreement applies, then any use, reproduction, modification, 
# translation, or compilation of this Software is prohibited without the express written permission of Infineon.

# Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF THIRD-PARTY RIGHTS AND IMPLIED WARRANTIES 
# SUCH AS WARRANTIES OF FITNESS FOR A SPECIFIC USE/PURPOSE OR MERCHANTABILITY. Infineon reserves the right to make changes to the Software 
# without notice. You are responsible for properly designing, programming, and testing the functionality and safety of your intended application 
# of the Software, as well as complying with any legal requirements related to its use. Infineon does not guarantee that the Software will be 
# free from intrusion, data theft or loss, or other breaches ("Security Breaches"), and Infineon shall have no liability arising out of any 
# Security Breaches. Unless otherwise explicitly approved by Infineon, the Software may not be used in any application where a failure of the 
# Product or any consequences of the use thereof can reasonably be expected to result in personal injury.

# QEMU Build Script for Docker Environment
# This script clones and builds QEMU TriCore emulator with optimizations

set -e  # Exit on any error

echo "🚀 QEMU TriCore Build Script"
echo "📋 Building optimized QEMU for TriCore emulation..."
echo ""

# Get number of CPU cores for parallel compilation
NPROC=$(nproc)
echo "🔧 Detected $NPROC CPU cores for parallel compilation"

# Setup build optimizations
echo "💾 Setting up build optimizations..."

# Optimize compiler flags for faster compilation (not runtime performance)
export CFLAGS="-O1 -pipe"  # -O1 faster than -O2 for compilation, -pipe uses pipes instead of temp files
export CXXFLAGS="-O1 -pipe"
echo "   ✅ Compiler flags optimized for fast compilation"

# Clone QEMU repository
echo ""
echo "📥 Cloning QEMU TriCore repository..."
QEMU_DIR="/tmp/qemu_6250_tricore"
cd /tmp
git clone https://github.com/volumit/qemu_6250_tricore.git
cd "$QEMU_DIR"

# Initialize Git submodules (required for QEMU build)
echo "📥 Initializing Git submodules..."
git submodule update --init --recursive
echo "   ✅ Git submodules initialized"
echo "   ✅ QEMU repository cloned"

# Build QEMU
echo ""
echo "⚡ Building QEMU with optimizations..."

# Create build directory
mkdir -p build && cd build

# Configure with minimal features (only what we need for TriCore)
echo "🔧 Configuring QEMU build..."
../configure \
    --target-list=tricore-softmmu \
    --disable-werror \
    --disable-debug-info \
    --disable-docs \
    --disable-gtk \
    --disable-sdl \
    --disable-vnc \
    --disable-curses
echo "   ✅ QEMU configuration completed"

# Reconfigure meson with parallel jobs
echo "🔧 Setting up meson build system..."
meson setup --reconfigure
echo "   ✅ Meson build system configured"

# Compile with all available cores
echo "⚡ Compiling QEMU (using $NPROC parallel jobs)..."
START_TIME=$(date +%s)
meson compile -j "$NPROC"
END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))

echo "   ✅ QEMU compilation completed in ${BUILD_TIME} seconds"

# Verify the binary was built successfully
QEMU_BIN="$QEMU_DIR/build/qemu-system-tricore"
if [ -f "$QEMU_BIN" ]; then
    # shellcheck disable=SC2012
    echo "   ✅ QEMU binary created: $(ls -lh $QEMU_BIN | awk '{print $5}')"
else
    echo "   ❌ QEMU binary not found!"
    exit 1
fi

echo ""
echo "✅ QEMU TriCore build completed successfully!"
echo "📍 Binary location: $QEMU_BIN"