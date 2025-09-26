# Central Scripts

A centralized repository of shared utility scripts and helper functions for the AI Model Zoo ecosystem. This repository serves as a git submodule to avoid code duplication across multiple model repositories.

## Purpose

This repository contains common functionality that is needed across various AI model projects in the model zoo, including:

- **Helper Functions**: Shared utilities for model conversion, validation, and analysis
- **Requirements Management**: Centralized dependency specifications and validation
- **Testing Utilities**: Common test functions and validation scripts

By maintaining these scripts centrally, we ensure consistency, reduce duplication, and simplify maintenance across all model zoo projects.

## Key Components

### Core Files

- **`helper_functions.py`**: Main utility library containing functions for:
  - ONNX model conversion and export (TensorFlow ↔ PyTorch ↔ ONNX)
  - Model analysis and parameter counting
  - Data preprocessing and tensor operations
  - Performance metrics calculation
  - Docker container management for model compilation

- **`requirements.txt`**: Centralized Python dependencies for the entire model zoo ecosystem

- **`validate_requirements.py`**: Dependency validation script that checks:
  - Dependency resolution without installation
  - Security vulnerabilities using `safety`
  - Requirements syntax validation

- **`test_requirements.py`**: Comprehensive test suite using pytest to verify:
  - Package installation and compatibility
  - Critical package imports
  - Version compatibility checks
  - Package conflict detection

## Installation

### Setting up and testing a virutal python environment

For standalone development and testing, you can either use the automated setup script or follow the manual steps below.

#### Quick Setup (Recommended)

Use the provided setup script to automatically configure everything:

```bash
# Make the setup script executable and run it
chmod +x setup.sh
./setup.sh
```

The script will automatically handle all the setup steps including system dependencies, virtual python environment creation, validation, and testing.

#### Manual Setup

For advanced users who prefer to understand each step, follow these detailed instructions that mirror the automated setup.sh script:

#### 1. Navigate to Repository Root

Ensure you're working from the repository root directory:

```bash
# If you're in `CentralScripts/` folder, move to parent directory
cd ..
# Verify you're in the repository root
pwd
```

#### 2. Install System Dependencies

Install the necessary build tools and development packages:

```bash
# Update package list and install build essentials
sudo apt update
sudo apt install -y build-essential python3-dev python3-venv python3-pip
```

#### 3. Create Virtual Environment in Repository Root

Create the virtual environment in the repository root (not in a global location):

```bash
# Create virtual environment named 'venv' in repository root
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

#### 4. Upgrade pip and Core Tools

Upgrade pip and install essential build tools:

```bash
# Upgrade pip to latest version along with setuptools and wheel
pip install --upgrade pip setuptools wheel
```

#### 5. Install Validation Tools

Install tools needed for dependency validation:

```bash
# Install validation and security tools
pip install pip-tools packaging safety
```

#### 6. Navigate to central_scripts and Validate Requirements

Move to the central_scripts directory and validate dependencies before installation:

```bash
# Navigate to central_scripts folder
cd CentralScripts/

# Run validation script to check for conflicts and security issues
python validate_requirements.py
```

#### 7. Install Dependencies

Install all required packages from requirements.txt:

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

#### 8. Run Tests

Verify the installation by running the test suite:

```bash
# Install pytest testing framework
pip install pytest

# Run tests with warnings suppressed for cleaner output
PYTHONWARNINGS="ignore" python -m pytest test_requirements.py -q --disable-warnings --tb=no
```

#### 9. Test Helper Functions

Verify that the helper functions can be imported correctly:

```bash
# Test core helper function imports
python -c "from helper_functions import load_onnx_model; print('✅ Helper functions imported successfully')"
```

#### 10. Return to Repository Root

Navigate back to the repository root for project work:

```bash
# Return to repository root
cd ..
```

#### Environment Management

When finished working:

```bash
deactivate
```

To reactivate the environment later:

```bash
# From repository root
source venv/bin/activate
```

## Supported Frameworks

- **TensorFlow** (≥2.8.0, <2.17)
- **PyTorch** (latest)
- **ONNX** (≥1.8.0)
- **ONNX Runtime**
- **tf2onnx** (1.16.1)

## Dependencies

Core dependencies include:
- Deep learning frameworks (TensorFlow, PyTorch)
- Model conversion tools (ONNX, tf2onnx)
- Data science stack (NumPy, Pandas, Matplotlib, Seaborn)
- Development tools (Jupyter, Flask)
- Testing and validation tools (pytest, safety)

See `requirements.txt` for the complete list with version specifications.

## Development

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-utility`)
3. Add your changes and tests
4. Validate requirements: `python validate_requirements.py`
5. Run tests: `python -m pytest test_requirements.py`
6. Submit a merge request

### Adding New Helper Functions

When adding new utilities to `helper_functions.py`:

1. Follow existing naming conventions
2. Add comprehensive docstrings
3. Include type hints where appropriate
4. Add corresponding tests if needed
5. Update this README if the function provides new major functionality

### Testing

```bash
# Validate requirements without installation
python validate_requirements.py

# Test all requirements
python -m pytest test_requirements.py -v

# Test specific functionality
python -c "from helper_functions import test_onnx_pb; test_onnx_pb('test_model')"
```

## Support

For questions, issues, or feature requests:

- Create an issue in this repository for general utilities
- Contact the AI/ML team for integration support
- Check existing issues before creating new ones

## Versioning

This repository follows semantic versioning. Major version changes may introduce breaking changes to the API.

## License

Please see our [LICENSE](../LICENSE) for copyright and license information.
