# DEEPCRAFT&trade; - A Model Zoo for AURIX&trade; Microcontroller Units

This model zoo is a collection of machine learning models and tools for developing, training, and deploying AI solutions on AURIX&trade; microcontroller units.

## Repository Structure

This repository contains the following components:

### `AIenhancedPID/`
This example demonstrates a proportional-integral-derivative (PID) controller enhanced by a simple multi-layer perceptron (MLP) network that tunes the controller parameters in real-time. This approach improves ride comfort and adaptability to unseen scenarios compared to conventional controllers with constant parameters.

### `AnomalyDetection/`
The AnomalyDetection module showcases the use of autoencoder neural networks trained on the Controlled Anomalies Time Series (CATS) dataset. It demonstrates how to build, train, and deploy MLP-based autoencoders to detect anomalies in multivariate time series.

### `CentralScripts/`
This directory contains shared utility scripts and helper functions for model conversion, validation, testing, and deployment.

### `ClassificationMobilenet/`
This example is based on a MobileNet architecture designed for efficient on-device vision applications. It is adapted for two tasks: traffic object classification (vehicles, pedestrians, cyclists, ...) and weather classification (rainy and clear conditions). 

### `KeywordDetection/`
KeywordDetection is a neural network implementation for detecting English words in microphone recordings, suitable for voice commands in automotive applications. It was trained on the Google Speech Commands dataset (35 classes).

### `MNISTimageClassification/`
MNISTimageClassification is a well-known example of handwritten digit classification using deep learning techniques.

### `ModelTemplate/`
This is a template for adding new AI models to the model zoo. It provides a standardized structure and workflow for implementing new machine learning models. 

### `RemainingUsefulLifePrediction/`
RemainingUsefulLifePrediction is a complete implementation for predicting remaining useful life of complex systems using deep learning. It demonstrates machine learning techniques for predictive maintenance using the NASA Turbofan Engine dataset.

## Getting Started

### Prerequisites

- Python 3.11
- Docker
- Ubuntu 22.04, native or via WSL

### Cloning the Repository

```bash
git clone https://github.com/Infineon/deepcraft-model-zoo-for-aurix.git
cd deepcraft-model-zoo-for-aurix/
```

**Note:** This repository uses Git LFS (Large File Storage) for managing large data files (CSV datasets, model checkpoints, etc.). The clone command above will automatically download LFS files if Git LFS is installed on your system.

If you don't have Git LFS installed, run the following command. 
```bash
sudo apt install git-lfs
```

If LFS files weren't downloaded during clone, fetch them afterwards.
```bash
git lfs pull
```

Verify that the LFS files are properly downloaded.
```bash
git lfs ls-files
```

### Setting Up the Environment

The repository includes an optimized setup script that creates a Docker image and Python virtual environment. The script provides progress indicators and is optimized for fast execution with parallel compilation and build caching.

The setup process includes 8 main steps:
- Installing system dependencies
- Setting up QEMU (TriCore emulator) with optimized compilation
- Building Docker image with AI tools
- Creating Python virtual environment
- Installing ML/AI packages (TensorFlow, PyTorch, ONNX, etc.)
- Validating the installation

```bash
# Make setup script executable
chmod +x CentralScripts/setup.sh

# Run the script from the repository root
sudo CentralScripts/setup.sh
```
The ```sudo``` command requires your Linux password.

The script shows progress with animated indicators for each step and completes in approximately:
- **Fresh installation**: 10-60 minutes (depending on system specs and internet speed)
- **Subsequent runs**: Much faster due to build caching and optimization

**Note:** The setup is optimized for parallel compilation using available CPU cores and includes build caching for faster rebuilds. You'll see progress indicators with animated feedback during longer operations.

### Activating Environment and Starting JupyterLab

After setting up, activate the Python virtual environment.

```bash
source venv/bin/activate
```
Start JupyterLab. Open the URL printed in the terminal to access JupyterLab and run the template notebook (new_model_template.ipynb).
```bash
# Navigate to the project you want to work on, for example:
cd ModelTemplate

# Start JupyterLab
jupyter lab
```
Once this is finished, close the JupyterLab browser tab and feel free to deactivate the virtual environment.

```bash
deactivate
```

## Dependencies

Core components include:
- **ONNX2C**: a tool that converts Open Neural Network Exchange Format (ONNX) models to C code
- **AURIX&trade; GCC**: a cross-compiler for AURIX&trade; targets
- **QEMU**: a machine emulator
- **Flask**: a REST API framework


## License

Please see the [LICENSE](LICENSE) for any details regarding copyright and license.