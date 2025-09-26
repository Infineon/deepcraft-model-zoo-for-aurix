# Copyright (C) Infineon Technologies AG 2025
# 
# Use of this file is subject to the terms of use agreed between (i) you or the company in which ordinary course of 
# business you are acting and (ii) Infineon Technologies AG or its licensees. If and as long as no such terms of use
# are agreed, use of this file is subject to following:
# 
# This file is licensed under the terms of the Boost Software License. See the LICENSE file in the root of this repository
# for complete details.

#!/bin/bash

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

# 6. Install validation tools first
echo "🔧 Installing validation tools..."
pip install pip-tools packaging safety

# 7. Validate requirements (need to navigate to CentralScripts for requirements.txt)
echo "🔍 Validating requirements..."
cd CentralScripts
python validate_requirements.py

# 8. Install dependencies
echo "📚 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 9. Run tests
echo "🧪 Running tests to verify installation..."
pip install pytest
PYTHONWARNINGS="ignore" python -m pytest test_requirements.py -q --disable-warnings --tb=no

# 10. Test helper functions
echo "🔬 Testing helper functions..."
python -c "from helper_functions import load_onnx_model; print('✅ Helper functions imported successfully')"

# 11. Return to repo root
cd ..

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "To deactivate when finished:"
echo "    deactivate"