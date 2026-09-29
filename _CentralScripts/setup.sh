#!/bin/bash

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
sed -i 's/\r$//' "$0" 2>/dev/null || true
set -euo pipefail

echo "🚀 Setting up AI Model Zoo Python Environment..."
echo "📋 This will install dependencies, download target tools, and build the Docker environment"
echo ""

PYTHON_VERSION="3.11"
HADOLINT_VERSION="2.12.0"
UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "unknown")
echo "📋 Using Python ${PYTHON_VERSION} (detected Ubuntu ${UBUNTU_VERSION})"

if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. Please install Docker Engine before running setup."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "⚠️  Docker is unavailable to the current user."
    echo "   Ensure the Docker daemon is running and configure non-root Docker access."
    echo "   Then verify access with: docker info"
    exit 1
fi

echo "✅ Docker available to the current user: $(docker --version)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
echo "📍 Working from repository root: $REPO_ROOT"

ONNX2C_VERSION=$(<"$REPO_ROOT/_Tools/onnx2c_version.txt")
TOOL_LOADER_DIR="$REPO_ROOT/_CentralScripts/tool_loader"
echo "📋 onnx2c version: $ONNX2C_VERSION"

if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    NC=''
fi

show_progress() {
    local pid=$1
    local message="$2"

    if [ -t 1 ]; then
        while kill -0 "$pid" 2>/dev/null; do
            for i in {1..3}; do
                if kill -0 "$pid" 2>/dev/null; then
                    echo -ne "\r$message$(printf "%${i}s" "" | tr ' ' '.')$(printf "%$((3-i))s" "" | tr ' ' ' ')"
                    sleep 0.5
                fi
            done
        done
    fi
    wait "$pid"
    echo -ne "\r${GREEN}$message... ✓${NC}\n"
}

manifest_value() {
    python - "$TOOL_LOADER_DIR/tools.csv" "$1" "$2" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        if row["tool_id"] == sys.argv[2]:
            print(row[sys.argv[3]])
            raise SystemExit(0)
raise SystemExit(f"Tool not found in manifest: {sys.argv[2]}")
PY
}

# Step 1: Install system dependencies
echo "[1/6] 📦 Installing system dependencies..."
(sudo apt update && sudo apt install -y \
    build-essential software-properties-common curl shellcheck \
    libsecret-1-0 libdbus-1-3 gnome-keyring) > /dev/null 2>&1 &
show_progress $! "[1/6] 📦 Installing system dependencies"

echo "[1/6] 📦 Installing hadolint ${HADOLINT_VERSION}..."
sudo curl -fsSL \
    -o /usr/local/bin/hadolint \
    "https://github.com/hadolint/hadolint/releases/download/v${HADOLINT_VERSION}/hadolint-Linux-x86_64"
sudo chmod +x /usr/local/bin/hadolint
hadolint --version
echo ""

# Step 2: Create the Python environment and install downloader dependencies
if ! command -v "python${PYTHON_VERSION}" &> /dev/null; then
    (sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update) > /dev/null 2>&1 &
    show_progress $! "[2/6] 🐍 Setting up Python repository"

    (sudo apt install -y \
        "python${PYTHON_VERSION}" \
        "python${PYTHON_VERSION}-dev" \
        "python${PYTHON_VERSION}-venv") > /dev/null 2>&1 &
    show_progress $! "[2/6] 🐍 Installing Python ${PYTHON_VERSION}"
else
    echo "[2/6] 🐍 Python ${PYTHON_VERSION} already available ✓"
fi

"python${PYTHON_VERSION}" -m venv "$REPO_ROOT/venv" &
show_progress $! "[2/6] 📁 Creating Python virtual environment"

# shellcheck disable=SC1091
source "$REPO_ROOT/venv/bin/activate"

pip install --upgrade -r "$REPO_ROOT/_CentralScripts/bootstrap_tool_versions.txt" > /dev/null 2>&1 &
show_progress $! "[2/6] ⬆️ Installing pinned bootstrap tools"

pip install -r "$TOOL_LOADER_DIR/requirements.txt" > /dev/null 2>&1 &
show_progress $! "[2/6] 🌐 Installing tool downloader dependencies"

playwright install chromium > /dev/null 2>&1 &
show_progress $! "[2/6] 🌐 Installing Chromium for IDC authentication"

sudo "$REPO_ROOT/venv/bin/playwright" install-deps chromium > /dev/null 2>&1 &
show_progress $! "[2/6] 🌐 Installing Chromium system dependencies"
echo ""

# Step 3: Download the manifest-pinned target tools
echo "[3/6] ⬇️ Downloading target tools from Infineon Developer Center..."
echo "   A browser may open for Infineon login when no valid session is cached."

"$TOOL_LOADER_DIR/tool_loader.sh" --auto-cookies

AURIX_GCC_ARCHIVE=$(manifest_value aurixgcc filename)
ACS_ARCHIVE=$(manifest_value ACS-Edge-AI-Package filename)
ACS_VERSION=$(manifest_value ACS-Edge-AI-Package version)
TSIM_ARCHIVE=$(manifest_value tsim filename)
TSIM_VERSION=$(manifest_value tsim version)
echo "[3/6] ✅ Manifest-pinned target tools are downloaded and verified"
echo ""

# Step 4: Build the Docker image with the downloaded tools and embedded QEMU
echo "[4/6] 🐳 Building Docker image with AI tools and QEMU..."
echo "   📋 This may take 10-15 minutes on first build (or seconds if cached)..."

DOCKER_BUILDKIT=1 docker build \
    --build-arg "AURIX_GCC_ARCHIVE=${AURIX_GCC_ARCHIVE}" \
    --build-arg "ACS_ARCHIVE=${ACS_ARCHIVE}" \
    --build-arg "ACS_VERSION=${ACS_VERSION}" \
    --build-arg "TSIM_ARCHIVE=${TSIM_ARCHIVE}" \
    --build-arg "TSIM_VERSION=${TSIM_VERSION}" \
    -f "$REPO_ROOT/_Tools/dockerfile" \
    -t "ai_model_zoo_tools:${ONNX2C_VERSION}" \
    "$REPO_ROOT"

if docker image inspect "ai_model_zoo_tools:${ONNX2C_VERSION}" > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Docker image built and tagged successfully${NC}"
else
    echo -e "${RED}   ❌ Docker image not found after build!${NC}"
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}" | head -5
    exit 1
fi

echo "   🔬 Verifying conversion service startup..."
SMOKE_CONTAINER_ID=$(docker run --rm -d -p 127.0.0.1::8080 "ai_model_zoo_tools:${ONNX2C_VERSION}")
cleanup_smoke_container() {
    docker rm -f "$SMOKE_CONTAINER_ID" > /dev/null 2>&1 || true
}
trap cleanup_smoke_container EXIT
SMOKE_PORT=$(docker port "$SMOKE_CONTAINER_ID" 8080/tcp | awk -F: 'NR == 1 {print $NF}')
if ! curl --fail --silent --retry 10 --retry-delay 1 \
    --retry-connrefused --retry-all-errors \
    "http://127.0.0.1:${SMOKE_PORT}/convert" > /dev/null; then
    echo -e "${RED}   ❌ Conversion service failed to start${NC}"
    docker logs "$SMOKE_CONTAINER_ID"
    exit 1
fi
cleanup_smoke_container
trap - EXIT
echo -e "${GREEN}   ✅ Conversion service is reachable${NC}"
echo ""

# Step 5: Install model-zoo Python dependencies
echo "[5/6] 📚 Installing ML and AI dependencies..."
cd "$REPO_ROOT/_CentralScripts"
pip install -r requirements.txt > /dev/null 2>&1 &
show_progress $! "[5/6] 📚 Installing ML and AI dependencies"
echo ""

# Step 6: Run tests and verify installation
pip install pytest > /dev/null 2>&1 &
show_progress $! "[6/6] 📦 Installing test framework"

PYTHONWARNINGS="ignore" python -m pytest test_requirements.py -q --disable-warnings --tb=no > /dev/null 2>&1 &
show_progress $! "[6/6] 🧪 Verifying installation"

python -c "from helper_functions import load_onnx_model; print('✅ Helper functions imported successfully')" 2>/dev/null &
show_progress $! "[6/6] 🔬 Testing helper functions"
cd "$REPO_ROOT"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Summary:"
echo "   ✅ System dependencies installed"
echo "   ✅ IDC tools downloaded and checksum-verified"
echo "   ✅ Docker image built with QEMU"
echo "   ✅ Conversion service startup verified"
echo "   ✅ Python environment ready"
echo "   ✅ ML dependencies installed"
echo "   ✅ All tests passed"
echo ""
echo "To activate the environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "To deactivate when finished:"
echo "    deactivate"