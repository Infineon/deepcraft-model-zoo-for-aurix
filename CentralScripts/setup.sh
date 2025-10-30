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

# Determine if script is being run from repo root or CentralScripts folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "CentralScripts" ]]; then
    # Script is in CentralScripts, move to parent directory (repo root)
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
    cd "$REPO_ROOT"
    echo "📍 Moved to repository root: $(pwd)"
else
    # Script is already in repo root or being called from repo root
    REPO_ROOT="$(pwd)"
    echo "📍 Working from repository root: $(pwd)"
fi

# 1. Install System Dependencies
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y build-essential software-properties-common curl

# install QEMU dependencies
echo "📦 Installing QEMU"
sudo apt-get install -y git libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev ninja-build meson

# clone QEMU if not already present
QEMU_DIR="$REPO_ROOT/Tools/qemu/qemu_6250_tricore"
if [ ! -d "$QEMU_DIR" ]; then
    mkdir -p $REPO_ROOT/Tools/qemu
    cd $REPO_ROOT/Tools/qemu
    git clone https://github.com/volumit/qemu_6250_tricore.git
    cd qemu_6250_tricore
else
    echo "✅ QEMU repo already cloned at $QEMU_DIR"
    cd "$QEMU_DIR"
fi

# build QEMU if artifact not present
QEMU_BIN="$QEMU_DIR/build/qemu-system-tricore"
if [ ! -f "$QEMU_BIN" ]; then
    echo "📦 Building QEMU"
    mkdir -p build && cd build
    ../configure
    meson setup --reconfigure
    meson compile
    cd "$REPO_ROOT"
else
    echo "✅ QEMU build artifact found at $QEMU_BIN, skipping build"
    cd "$REPO_ROOT"
fi


# build docker image
echo "🐳 Building Docker image..."
sudo docker build -f $REPO_ROOT/Tools/tc_dockerfile -t aurix_ai_tools:V1.0.1.TriCore $REPO_ROOT

# Install Python if needed
if [ "$NEED_PYTHON_INSTALL" = true ]; then
    echo "🐍 Installing Python ${PYTHON_VERSION}..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    
    # Install Python packages (distutils not needed for 3.11+)
    if [[ "$PYTHON_VERSION" == "3.11" ]]; then
        sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
    else
        sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
    fi
    
    # Install pip for the specific Python version
    echo "📦 Installing pip for Python ${PYTHON_VERSION}..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}
else
    echo "✅ Python ${PYTHON_VERSION} already available, skipping installation"
fi

# 2. Create virtual environment in repo root
echo "📁 Creating virtual environment in repo root..."
echo "🐍 Creating virtual environment with Python ${PYTHON_VERSION}: venv"
python${PYTHON_VERSION} -m venv venv

echo "✅ Activating virtual environment..."
source venv/bin/activate

# 5. Upgrade pip to latest version
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 6. Install dependencies
echo "📚 Installing dependencies from requirements.txt..."
cd CentralScripts
pip install -r requirements.txt

# 7. Run tests
echo "🧪 Running tests to verify installation..."
pip install pytest
PYTHONWARNINGS="ignore" python -m pytest test_requirements.py -q --disable-warnings --tb=no

# 8. Test helper functions
echo "🔬 Testing helper functions..."
python -c "from helper_functions import load_onnx_model; print('✅ Helper functions imported successfully')"

# 9. Return to repo root
cd ..

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "To deactivate when finished:"
echo "    deactivate"