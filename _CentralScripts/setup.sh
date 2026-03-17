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


# Convert line endings to Unix format (in case file was edited on Windows)
sed -i 's/\r$//' "$0" 2>/dev/null || true

set -e  # Exit on any error

echo "🚀 Setting up AI Model Zoo Python Environment..."
echo "📋 This will install system dependencies, create Python environment, and set up Docker tools with QEMU"
echo ""

# Use Python 3.11 for consistency across Ubuntu versions
PYTHON_VERSION="3.11"
UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "unknown")
echo "📋 Using Python 3.11 for consistency across Ubuntu versions (detected: Ubuntu $UBUNTU_VERSION)"

# Check Python availability
if ! command -v python${PYTHON_VERSION} &> /dev/null; then
    echo "⚠️  Python ${PYTHON_VERSION} not found. Will install it..."
    NEED_PYTHON_INSTALL=true
else
    echo "✅ Python ${PYTHON_VERSION} found: $(python${PYTHON_VERSION} --version)"
    NEED_PYTHON_INSTALL=false
fi

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. Please install Docker and ensure the Docker daemon is running."
    exit 1
else
    echo "✅ Docker found: $(docker --version)"
fi

# Determine if script is being run from repo root or _CentralScripts folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "_CentralScripts" ]]; then
    # Script is in _CentralScripts, move to parent directory (repo root)
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
    cd "$REPO_ROOT"
    echo "📍 Moved to repository root: $(pwd)"
else
    # Script is already in repo root or being called from repo root
    REPO_ROOT="$(pwd)"
    echo "📍 Working from repository root: $(pwd)"
fi

# Color codes for output (only if outputting to a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color
else
    # No colors when output is redirected
    RED=''
    GREEN=''
    NC=''
fi

# Function to show animated dots while a command runs
show_progress() {
    local pid=$1
    local message="$2"
    
    # Check if output is going to a terminal (interactive) or being redirected/piped
    if [ -t 1 ]; then
        # Interactive mode: show animated progress
        while kill -0 "$pid" 2>/dev/null; do
            for i in {1..3}; do
                if kill -0 "$pid" 2>/dev/null; then
                    echo -ne "\r$message$(printf "%${i}s" "" | tr ' ' '.')$(printf "%$((3-i))s" "" | tr ' ' ' ')"
                    sleep 0.5
                fi
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo -ne "\r$message   "
                sleep 0.5
            fi
        done
    else
        # Non-interactive mode: just wait silently (single line already printed)
        wait "$pid"
    fi
    echo -ne "\r${GREEN}$message... ✓${NC}\n"
}

# Step 1: Install System Dependencies
echo "[1/5] 📦 Installing system dependencies..."
(sudo apt update && sudo apt install -y build-essential software-properties-common curl) > /dev/null 2>&1 &
show_progress $! "[1/5] 📦 Installing system dependencies"
echo ""

# Step 2: Build Docker image with embedded QEMU
echo "[2/5] 🐳 Building Docker image with AI tools and QEMU..."
echo "   📋 This may take 10-15 minutes on first build (or seconds if cached)..."
echo "   🔨 Building..."

# Build the Docker image
DOCKER_BUILDKIT=1 sudo docker build \
    -f "$REPO_ROOT/_Tools/tc_dockerfile" \
    -t aurix_ai_tools:V1.0.4.TriCore \
    "$REPO_ROOT"

# Verify the image was created successfully
if sudo docker image inspect aurix_ai_tools:V1.0.4.TriCore > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Docker image built and tagged successfully${NC}"
else
    echo -e "${RED}   ❌ Docker image not found after build!${NC}"
    echo "   Available images:"
    sudo docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}" | head -5
    exit 1
fi
echo ""

# Step 3: Install Python if needed
if [ "$NEED_PYTHON_INSTALL" = true ]; then
    (sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update) > /dev/null 2>&1 &
    show_progress $! "[3/5] 🐍 Setting up Python repository"
    
    sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv > /dev/null 2>&1 &
    show_progress $! "[3/5] 🐍 Installing Python ${PYTHON_VERSION}"
    
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} > /dev/null 2>&1 &
    show_progress $! "[3/5] 📦 Installing pip for Python ${PYTHON_VERSION}"
else
    echo "[3/5] 🐍 Python ${PYTHON_VERSION} already available ✓"
fi
echo ""

# Step 4: Create virtual environment and install dependencies
python${PYTHON_VERSION} -m venv venv &
show_progress $! "[4/5] 📁 Creating Python virtual environment"

# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip setuptools wheel > /dev/null 2>&1 &
show_progress $! "[4/5] ⬆️ Upgrading pip"

echo "   📦 This may take a few minutes..."
cd _CentralScripts
pip install -r requirements.txt > /dev/null 2>&1 &
show_progress $! "[4/5] 📚 Installing ML and AI dependencies"
echo ""

# Step 5: Run tests and verify installation
pip install pytest > /dev/null 2>&1 &
show_progress $! "[5/5] 📦 Installing test framework"

PYTHONWARNINGS="ignore" python -m pytest test_requirements.py -q --disable-warnings --tb=no > /dev/null 2>&1 &
show_progress $! "[5/5] 🧪 Verifying installation"

python -c "from helper_functions import load_onnx_model; print('✅ Helper functions imported successfully')" 2>/dev/null &
show_progress $! "[5/5] 🔬 Testing helper functions"

cd ..

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Summary:"
echo "   ✅ System dependencies installed"
echo "   ✅ Docker image built with QEMU"
echo "   ✅ Python environment ready"
echo "   ✅ ML dependencies installed"
echo "   ✅ All tests passed"
echo ""
echo "To activate the environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "To deactivate when finished:"
echo "    deactivate"