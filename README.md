# AI Model Zoo

A collection of machine learning models and tools for developing, training, and deploying AI solutions on embedded systems. This repository consolidates various model examples and shared utilities to support AI development workflows from research to production deployment.

## Repository Structure

This repository contains the following components:

### `AIenhancedPID/`
Demonstrates a Proportional-Integral-Derivative (PID) controller enhanced by a simple MLP network that tunes the controller parameters in real time. This approach enables higher driving comfort and adaptability to unseen scenarios compared to conventional controllers with constant parameters. Uses Simulink FMU models and FMPy library for model execution.

### `AnomalyDetection/`
Demonstrates anomaly detection using autoencoder neural networks trained on the Controlled Anomalies Time Series (CATS) dataset. Shows how to build, train, and deploy MLP-based autoencoders for detecting anomalies in multivariate time series data with precisely labeled anomalies for algorithm development and evaluation.

### `CentralScripts/`
Shared utility scripts and helper functions used across the model zoo ecosystem. Contains common functionality for model conversion, validation, testing, and deployment. Includes tools for converting models between different frameworks (TensorFlow, PyTorch, ONNX) and managing dependencies.

### `ClassificationMobilenet/`
Lightweight deep learning models using the MobileNet architecture designed for efficient on-device vision applications. Includes implementations for traffic object detection (vehicles, pedestrians, cyclists) and weather classification (rainy vs clear conditions). Optimized for speed and low power consumption, making them suitable for embedded hardware deployment.

### `KeywordDetection/`
Neural network implementation for detecting English words in microphone recordings, suitable for wake word detection or voice commands in automotive applications. Uses the Google Speech Commands dataset with 35 classes of English words and includes noise recordings for realistic training scenarios.

### `MNISTimageClassification/`
A machine learning reference implementation for handwritten digit classification using deep learning techniques. Provides a complete workflow for computer vision solutions with applications in ADAS, autonomous driving, quality control, and safety systems. Includes an end-to-end deployment pipeline for embedded automotive systems and AURIX&trade; microcontroller deployment.

### `ModelTemplate/`
A template for adding new AI models to the zoo. Provides a standardized structure and workflow for implementing new machine learning models. Includes example code for model definition, training, and deployment pipeline setup.

### `RemainingUsefulLifePrediction/`
A complete implementation for predicting remaining useful life of complex systems using deep learning. Demonstrates machine learning techniques for predictive maintenance applications using the NASA Turbofan Engine dataset. Includes both LSTM and MLP model architectures with deployment to embedded systems.

## Getting Started

### Prerequisites

- Python 3.11
- Docker

### Getting the tool docker image

```bash
# Pull the tool docker image
docker pull ghcr.io/infineon/docker-image-infineon:latest
```

### Clone the Repository

```bash
git clone https://github.com/Infineon/deepcraft-model-zoo-for-aurix.git
cd deepcraft-model-zoo-for-aurix/
```

**Note:** This repository uses Git LFS (Large File Storage) for managing large data files (CSV datasets, model checkpoints, etc.). The clone command above will automatically download LFS files if Git LFS is installed on your system.

**If you don't have Git LFS installed:**
```bash
# Install Git LFS (if not already installed)
# On Ubuntu/Debian:
sudo apt install git-lfs
```

**If LFS files weren't downloaded during clone:**
```bash
# After cloning, ensure LFS files are downloaded
git lfs pull
```

**To verify LFS files are properly downloaded:**
```bash
# Check LFS file status
git lfs ls-files
```

### Set Up Python Environment

The repository includes an automated setup script that creates a Python virtual environment, installs all required dependencies, validates the installation, and tests the setup. Installation will take several minutes depending on your local setup and internet connectivity.

**Quick Setup:**

Quick setup is intended for use under Linux. 

```bash
# Make setup script executable
chmod +x CentralScripts/setup.sh

# Run the script from the repository root
bash CentralScripts/setup.sh
```

Running the setup script will take several minutes. This script will:
- Create a Python virtual environment
- Install all required packages from `requirements.txt`
- Validate the package installation
- Run setup tests to ensure everything is working correctly

For detailed setup instructions and troubleshooting, refer to the [CentralScripts README](CentralScripts/README.md).

### Activate Environment and Start Jupyter Lab

After setup, activate the Python virtual environment and start Jupyter Lab:

```bash
# Activate the virtual environment (from repository root)
source venv/bin/activate

# Navigate to the project you want to work on, for example:
cd remaining-useful-life-prediction

# Start Jupyter Lab
jupyter lab
```

Copying the url from the command shell output into a new browser tab will open Jupyter Lab where you can run the RUL prediction notebooks (`RUL_MLP_prediction.ipynb` or `RUL_LSTM_prediction.ipynb`). When finsihed simply close the jupyter lab browser tab and deactivate the virtual environment running:

```bash
deactivate
```

## Support

For questions, issues, or feature requests:

- Create an issue in this repository for general utilities
- Contact the AI/ML team for integration support
- Check existing issues before creating new ones

## License

Please see our [LICENSE](LICENSE) for copyright and license information.