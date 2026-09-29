# DEEPCRAFT&trade; - A Model Zoo for AURIX&trade; Microcontroller Units

This model zoo is a collection of machine learning models and tools for developing, training, and deploying AI solutions on AURIX&trade; microcontroller units.

## Repository Structure

This repository contains the following components:

### `AIenhancedPID/`
This example demonstrates a proportional-integral-derivative (PID) controller enhanced by a simple multi-layer perceptron (MLP) network that tunes the controller parameters in real-time. This approach improves ride comfort and adaptability to unseen scenarios compared to conventional controllers with constant parameters.

### `AnomalyDetection/`
The AnomalyDetection module showcases the use of autoencoder neural networks trained on the Controlled Anomalies Time Series (CATS) dataset. It demonstrates how to build, train, and deploy MLP-based autoencoders to detect anomalies in multivariate time series.

### `DriverMonitoringSystem/`
This example demonstrates in-cabin driver monitoring using the MiniDMS model to classify driver behavior from camera input.

### `Quantization_CNNClassificationRainDrops/`
This example demonstrates post-training and quantization-aware training of a CNN for weather classification (presence of rain drops).

### `KeywordDetection/`
KeywordDetection is a neural network implementation for detecting English words in microphone recordings, suitable for voice commands in automotive applications. It was trained on the Google Speech Commands dataset (35 classes).

### `MNISTimageClassification/`
MNISTimageClassification is a well-known example of handwritten digit classification using deep learning techniques.

### `MobileNetV3ClassificationRainDrops/`
This example is based on a MobileNet architecture designed for efficient on-device vision applications. It is adapted for weather classification (rainy and clear conditions). 

### `MobileNetV3ClassificationTrafficObjects/`
This example is based on a MobileNet architecture designed for efficient on-device vision applications. It is adapted for traffic object classification (vehicles, pedestrians, cyclists, ...). 

### `RemainingUsefulLifePrediction/`
RemainingUsefulLifePrediction is a complete implementation for predicting remaining useful life of complex systems using deep learning. It demonstrates machine learning techniques for predictive maintenance using the NASA Turbofan Engine dataset.

### `VehicleTracker/`

This example demonstrates online multi-target vehicle tracking using an RNN motion model and an LSTM data-association model trained on CARLA simulator data.

### `_CentralScripts/`
This directory contains shared utility scripts and helper functions for model conversion, validation, testing, and deployment. See its [detailed documentation](_CentralScripts/README.md) for the conversion REST API and `CallTools` client.

### `_ModelTemplate/`
This is a template for adding new AI models to the model zoo. It provides a standardized structure and workflow for implementing new machine learning models. 

### `_Tools/`
This directory contains the Docker build definition, cross-toolchain support files, linker scripts, and QEMU sources used by the conversion environment.

### `_LICENSES/`
This directory contains license information for third-party tools and components distributed with or used by the model zoo.

## Getting Started

### Prerequisites

- Python 3.11
- Docker Engine with the daemon running
- Permission to run Docker as your normal user
- Ubuntu 22.04, 24.04, or 26.04, native or via WSL
- An [Infineon Developer Center](https://softwaretools.infineon.com/home) account for tool download. You will be prompted to login on the Infineon website during the setup process.

The repository has been tested on Ubuntu 22.04, 24.04, and 26.04. The Docker
image uses Ubuntu 24.04 because the downloaded PPU toolchain requires the
newer glibc provided by that image; this does not limit the supported host
Ubuntu versions.

Verify that Docker is available without `sudo` before running setup:

```bash
docker info
```

If this reports a permission error, configure
[rootless Docker](https://docs.docker.com/engine/security/rootless/) or follow
Docker's [Linux post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/)
and then log out and back in. Membership in the `docker` group grants
root-level privileges; do not make the Docker socket world-writable.

### Cloning the Repository

This repository uses Git LFS (Large File Storage) for datasets, model
checkpoints, and other large artifacts. Install and initialize Git LFS before
cloning so these files are downloaded instead of being left as pointer files.

On Ubuntu, install Git LFS with:

```bash
sudo apt install git-lfs
git lfs install
```

```bash
git clone https://github.com/Infineon/deepcraft-model-zoo-for-aurix.git
cd deepcraft-model-zoo-for-aurix/
```

If Git LFS was installed after cloning, or if the files were not downloaded
during the clone, fetch them afterwards:

```bash
git lfs pull
```

Verify that the LFS files are properly downloaded. Each listed file should have
an asterisk (`*`) next to it:

```bash
git lfs ls-files
```

### Setting Up the Environment

The repository includes an optimized setup script that downloads the required
target tools from Infineon Developer Center (IDC), creates a Docker image, and
creates the Python virtual environment. The versions, download URLs, and
SHA-256 checksums in `_CentralScripts/tool_loader/tools.csv` are authoritative;
the repository does not bundle fallback copies of these tools.

The setup process includes the following main steps:
- Installing system dependencies
- Downloading from Infineon website and checksum-verifying ACS Edge AI, AURIX&trade; GCC, and TSIM. Download requires to login.
- Building QEMU emulators (TriCore&trade; and Arm&reg;) with optimized compilation
- Building the Docker image with all AI tools and toolchains
- Creating the Python virtual environment
- Installing ML/AI packages (TensorFlow, PyTorch, ONNX, etc.)
- Validating Docker access, conversion-service startup, and the Python installation

```bash
# Make setup script executable
chmod +x _CentralScripts/setup.sh

# Run the script from the repository root
./_CentralScripts/setup.sh
```
The script uses `sudo` for individual system-level operations and may prompt for
your Linux password. Run the script as your normal user so that the virtual
environment and its files remain owned by your user account.

The first run may open a browser for Infineon authentication. Complete the IDC
login and allow the setup to continue. Downloaded packages and authentication
state are stored under `_CentralScripts/tool_loader/`; these generated files are
ignored by Git. Subsequent runs reuse cached packages only after validating
their manifest checksums.

#### Manual Tool Download Fallback

If the automated IDC login or download fails, download the three archives in a
regular browser while logged in to Infineon Developer Center. Use the exact
versions and filenames below; the values in
[`tools.csv`](_CentralScripts/tool_loader/tools.csv) remain authoritative.

| Tool | Version | IDC download | Required filename |
| --- | --- | --- | --- |
| AURIX&trade; GCC for Linux x86-64 | 03-2026 | [Download from IDC](https://softwaretools-hosting.infineon.com/packages/com.ifx.tb.tool.aurixgcc/versions/03-2026/artifacts/aurixgcc_03-2026_Linux_x86-x64.zip/download) | `aurixgcc_03-2026_Linux_x86-x64.zip` |
| TSIM TriCore&trade; instruction-set simulator for Linux x86-64 | 1.18.196 | [Download from IDC](https://softwaretools-hosting.infineon.com/packages/com.ifx.tb.tool.tsimtricoreinstructionsetsimulator/versions/1.18.196/artifacts/tsimtricoreinstructionsetsimulator_1.18.196_Linux_x86-x64.deb/download) | `tsimtricoreinstructionsetsimulator_1.18.196_Linux_x86-x64.deb` |
| ACS Edge AI Package | 1.0.0 | [Download from IDC](https://softwaretools-hosting.infineon.com/packages/com.ifx.tb.tool.acsedgeaipackage/versions/1.0.0/artifacts/ACS-Edge-AI-Package-1.0.0.zip/download) | `ACS-Edge-AI-Package-1.0.0.zip` |

The ACS package is marked as a Windows package in the IDC manifest, but its ZIP
also contains the Linux tools required by the Docker image. Accept any licenses
presented by IDC, then place all three files, unchanged, in:

```text
_CentralScripts/tool_loader/downloads/
```

Create the directory if necessary. Ensure that the browser has not added a
suffix such as `(1)` to a filename. The archives are proprietary and this
directory is ignored by Git; do not commit or redistribute them.

Rerun setup from the repository root after placing the files:

```bash
./_CentralScripts/setup.sh
```

The loader verifies each cached archive against the SHA-256 checksum in the
manifest. Valid files are reused without another login; missing, renamed, or
invalid files still trigger the normal download flow or an integrity error.

The script shows progress with animated indicators for each step and completes in approximately:
- **Fresh installation**: 10-60 minutes (depending on system specs and internet speed)
- **Subsequent runs**: Much faster due to build caching and optimization

**Note:** The setup is optimized for parallel compilation using available CPU cores and includes build caching for faster rebuilds. You'll see progress indicators with animated feedback during longer operations.

For authentication, download, or checksum errors, see the
[tool-loader documentation](_CentralScripts/tool_loader/README.md).

### Activating Environment and Starting JupyterLab

After setting up, activate the Python virtual environment.

```bash
source venv/bin/activate
```
Start JupyterLab. Open the URL printed in the terminal to access JupyterLab and run the template notebook (new_model_template.ipynb).
```bash
# Navigate to the project you want to work on, for example:
cd _ModelTemplate

# Start JupyterLab
jupyter lab
```
Once this is finished, close the JupyterLab browser tab and feel free to deactivate the virtual environment.

```bash
deactivate
```

## Dependencies

Core components include:
- **onnx2c** (onnx2c-ifx): a tool that converts Open Neural Network Exchange Format (ONNX) models to C code; provided by the ACS Edge AI package
- **AURIX&trade; GCC**: a cross-compiler for AURIX&trade; TriCore&trade; targets (TC3x, TC4x)
- **ARC LLVM/clang**: a cross-compiler for the AURIX&trade; TC4x Parallel Processing Unit (PPU); provided by the ACS Edge AI package
- **Arm&reg; GCC** (`gcc-arm-none-eabi`): a cross-compiler for Arm&reg; Cortex&reg;-M4 targets
- **QEMU**: machine emulators for TriCore&trade; and Arm&reg;, with a CPI plugin for cycle estimation
- **nSIM**: the Synopsys ARC instruction-set simulator used for the PPU; provided by the ACS Edge AI package
- **TSIM**: the default cycle-accurate TriCore&trade; instruction-set simulator (the QEMU CPI model remains available as an alternative)
- **Flask**: a REST API framework

### Conversion service

The Docker image provides a REST service that converts ONNX models to C,
compiles them, and benchmarks the selected hardware target. The generated
`model.c`, benchmark results, and pipeline log can be downloaded after each
conversion. See the [_CentralScripts conversion-service documentation](_CentralScripts/README.md#conversion-service-rest-api)
for supported targets, request fields, returned artifacts, and `CallTools` usage.
When a notebook performs model conversion, it automatically starts the required
Docker-based conversion service. No Docker commands are needed for the notebook
workflow.
The local helper binds the service to localhost by default. Uploads are limited
to 512 MiB by default; set `ZOO_MAX_CONTENT_LENGTH` to change that limit.


## License

Please see the [LICENSE](LICENSE), [EULA](EULA.txt), and [_LICENSES/README.md](_LICENSES/README.md) for copyright, usage, and third-party license information.