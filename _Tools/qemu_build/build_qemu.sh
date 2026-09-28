#!/usr/bin/env bash
# Copyright (c) 2025, Infineon Technologies AG, or an affiliate of Infineon Technologies AG. All rights reserved.
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

# QEMU Build Script — builds TriCore and/or ARM emulators from source
#
# Usage:
#   ./build_qemu.sh                  # Build all targets
#   ./build_qemu.sh --tricore-only   # Build TriCore only
#   ./build_qemu.sh --arm-only       # Build ARM only
#
# Output binaries are placed in the parent Tools/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_ROOT="/tmp/nn2ifx_qemu_build"

# --- Configuration ---
TRICORE_REPO="https://github.com/volumit/qemu_6250_tricore.git"
TRICORE_COMMIT="fb04318fddecfdc0958804d90fd4919134f2ee0f"
ARM_REPO="https://gitlab.com/qemu-project/qemu.git"
ARM_TAG="v9.2.0"  # Stable release with plugin support
ARM_DEVICE_CONFIG="$SCRIPT_DIR/nn2ifx-arm-m4.mak"

TRICORE_OUTPUT="$TOOLS_DIR/qemu-system-tricore"
ARM_OUTPUT="$TOOLS_DIR/qemu-system-arm"

# --- Parse arguments ---
BUILD_TRICORE=true
BUILD_ARM=true

for arg in "$@"; do
    case "$arg" in
        --tricore-only) BUILD_ARM=false ;;
        --arm-only)     BUILD_TRICORE=false ;;
        --help|-h)
            echo "Usage: $0 [--tricore-only|--arm-only]"
            exit 0 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- Helpers ---
info()  { echo -e "\033[0;32m[INFO]\033[0m $1"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $1"; }
error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; exit 1; }

NPROC=$(nproc)

check_dependencies() {
    local missing=()
    for cmd in git ninja meson pkg-config gcc python3; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done
    # Check dev libraries via pkg-config
    for lib in glib-2.0 pixman-1; do
        if ! pkg-config --exists "$lib" 2>/dev/null; then
            missing+=("lib${lib}-dev")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing build dependencies: ${missing[*]}
Install with: sudo apt install -y ninja-build meson pkg-config libglib2.0-dev libpixman-1-dev git gcc python3"
    fi
    info "All build dependencies found"
}

# Common QEMU configure options (minimal, fast build)
COMMON_OPTS=(
    --enable-plugins
    --disable-werror
    --disable-debug-info
    --disable-docs
    --disable-gtk
    --disable-sdl
    --disable-vnc
    --disable-curses
    --disable-opengl
    --disable-virglrenderer
    --disable-xen
    --disable-spice
)

build_target() {
    local name="$1"
    local repo="$2"
    local revision="${3:-}"
    local target_list="$4"
    local output_bin="$5"
    local device_config="${6:-}"
    local src_dir="$BUILD_ROOT/$name"

    info "=== Building QEMU ($name) ==="

    # Clone if not already present
    if [[ ! -d "$src_dir" ]]; then
        info "Cloning $repo ..."
        git clone --depth 1 "$repo" "$src_dir"
        cd "$src_dir"
        git fetch --depth 1 origin "$revision"
        git checkout --detach FETCH_HEAD
        # Skip submodules — they contain ROMs/firmware we don't need for bare-metal emulation
    else
        info "Source already cloned at $src_dir"
        cd "$src_dir"
        git fetch --depth 1 origin "$revision"
        git checkout --detach FETCH_HEAD
    fi

    # Configure
    local build_dir="$src_dir/build"
    mkdir -p "$build_dir" && cd "$build_dir"

    local device_opts=()
    if [[ -n "$device_config" ]]; then
        local device_config_name
        device_config_name="$(basename "$device_config" .mak)"
        cp "$device_config" "$src_dir/configs/devices/$target_list/$device_config_name.mak"
        device_opts=(
            --without-default-devices
            "--with-devices-${target_list%-softmmu}=$device_config_name"
        )
    fi

    info "Configuring ($target_list, plugins enabled)..."
    export CFLAGS="-O2 -pipe"
    export CXXFLAGS="-O2 -pipe"
    ../configure \
        --target-list="$target_list" \
        "${device_opts[@]}" \
        "${COMMON_OPTS[@]}"

    # Build
    info "Compiling with $NPROC parallel jobs..."
    local start_time
    start_time=$(date +%s)
    ninja -j "$NPROC"
    local build_time=$(( $(date +%s) - start_time ))
    info "Compilation completed in ${build_time}s"

    # Install binary
    local built_bin="$build_dir/qemu-system-${target_list%-softmmu}"
    if [[ ! -f "$built_bin" ]]; then
        # Try alternative path
        built_bin=$(find "$build_dir" -name "qemu-system-*" -type f | head -1)
    fi

    if [[ -f "$built_bin" ]]; then
        cp "$built_bin" "$output_bin"
        chmod +x "$output_bin"
        info "Installed: $output_bin ($(du -h "$output_bin" | cut -f1))"
    else
        error "Build succeeded but binary not found in $build_dir"
    fi
}

# --- Main ---
info "QEMU Build Script for nn2ifx"
info "Using $NPROC CPU cores for compilation"
echo ""

check_dependencies
mkdir -p "$BUILD_ROOT"

if [[ "$BUILD_TRICORE" == "true" ]]; then
    build_target "tricore" "$TRICORE_REPO" "$TRICORE_COMMIT" "tricore-softmmu" "$TRICORE_OUTPUT"
    echo ""
fi

if [[ "$BUILD_ARM" == "true" ]]; then
    build_target "arm" "$ARM_REPO" "$ARM_TAG" "arm-softmmu" "$ARM_OUTPUT" "$ARM_DEVICE_CONFIG"
    echo ""
fi

info "Done! Built QEMU binaries:"
[[ "$BUILD_TRICORE" == "true" ]] && info "  TriCore: $TRICORE_OUTPUT"
[[ "$BUILD_ARM" == "true" ]]     && info "  ARM:     $ARM_OUTPUT"
