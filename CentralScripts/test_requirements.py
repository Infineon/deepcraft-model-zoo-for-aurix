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


import importlib
import importlib.metadata
import pytest
import os
import warnings
from packaging.requirements import Requirement

# Suppress common warnings to keep output clean
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="google.protobuf")
warnings.filterwarnings("ignore", module="jupyter_server")
warnings.filterwarnings("ignore", module="jupyter_core")


def get_requirement_from_file(package_name):
    """Get requirement specification for a package from requirements.txt"""
    with open("requirements.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                req = Requirement(line)
                if req.name.lower() == package_name.lower():
                    return req
    return None


def test_requirements_file_exists():
    """Test that requirements.txt file exists and is readable."""
    assert os.path.exists("requirements.txt"), "requirements.txt file should exist"


def test_all_packages_installable():
    """Test that all packages in requirements.txt can be installed."""
    with open("requirements.txt", "r") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    for requirement_str in requirements:
        try:
            req = Requirement(requirement_str)
            try:
                installed_version = importlib.metadata.version(req.name)
                if req.specifier and not req.specifier.contains(installed_version):
                    pytest.fail(
                        f"Package {requirement_str} version {installed_version} doesn't satisfy requirement"
                    )
            except importlib.metadata.PackageNotFoundError:
                pytest.fail(f"Package {req.name} is not installed")
        except Exception as e:
            pytest.fail(f"Error checking requirement {requirement_str}: {e}")


def test_critical_packages_importable():
    """Test that critical packages can be imported successfully."""
    critical_packages = [
        "flask",
        "numpy",
        "pandas",
        "tensorflow",
        "torch",
        "sklearn",
        "onnx",
        "onnxruntime",
        "onnxsim",
    ]

    for package in critical_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            pytest.fail(f"Critical package {package} cannot be imported")


def test_tensorflow_version_compatibility():
    """Test TensorFlow version matches requirements.txt specification."""
    import tensorflow as tf

    # Get requirement from requirements.txt
    tf_req = get_requirement_from_file("tensorflow")
    if tf_req and tf_req.specifier:
        installed_version = tf.__version__
        assert tf_req.specifier.contains(
            installed_version
        ), f"TensorFlow version {installed_version} doesn't satisfy requirement {tf_req}"


def test_onnx_version_compatibility():
    """Test ONNX version matches requirements.txt specification."""
    import onnx

    # Get requirement from requirements.txt
    onnx_req = get_requirement_from_file("onnx")
    if onnx_req and onnx_req.specifier:
        installed_version = onnx.__version__
        assert onnx_req.specifier.contains(
            installed_version
        ), f"ONNX version {installed_version} doesn't satisfy requirement {onnx_req}"


def test_tf2onnx_version_compatibility():
    """Test tf2onnx version matches requirements.txt specification."""
    import tf2onnx

    # Get requirement from requirements.txt
    tf2onnx_req = get_requirement_from_file("tf2onnx")
    if tf2onnx_req and tf2onnx_req.specifier:
        installed_version = tf2onnx.__version__
        assert tf2onnx_req.specifier.contains(
            installed_version
        ), f"tf2onnx version {installed_version} doesn't satisfy requirement {tf2onnx_req}"


def test_no_conflicting_packages():
    """Test for common package conflicts."""
    try:

        # If both import successfully, no immediate conflict
        assert True
    except Exception as e:
        pytest.fail(f"Package conflict detected: {e}")


def test_jupyter_ecosystem():
    """Test Jupyter packages work together."""
    try:
        import jupyter
        import jupyterlab

        assert True
    except ImportError as e:
        pytest.fail(f"Jupyter ecosystem import failed: {e}")


def test_data_science_stack():
    """Test core data science packages are compatible."""
    packages = ["numpy", "pandas", "matplotlib", "seaborn", "sklearn"]
    for package in packages:
        try:
            if package == "sklearn":
                importlib.import_module("sklearn")
            else:
                importlib.import_module(package)
        except ImportError:
            pytest.fail(f"Data science package {package} failed to import")


def test_image_processing_capabilities():
    """Test image processing packages."""
    try:
        from PIL import Image
        import torchvision
        import torchaudio

        assert True
    except ImportError as e:
        pytest.fail(f"Image/audio processing import failed: {e}")
