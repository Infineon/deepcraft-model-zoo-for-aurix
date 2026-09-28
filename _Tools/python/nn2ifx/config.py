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

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


def _get_required_path(env_var: str, description: str) -> str:
    """Get a required path from environment variable and verify it exists."""
    path = os.getenv(env_var)
    if not path:
        raise ValueError(
            f"{env_var} not set. Please configure it in your .env file.\n"
            f"Copy .env.example to .env and set the path to {description}"
        )

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"{env_var}={path} does not exist.\n"
            f"Please verify the path to {description} in your .env file"
        )

    return path


def _get_optional_path(env_var: str, description: str) -> str:
    """Get an optional path from environment variable. Returns empty string if not set."""
    path = os.getenv(env_var)
    if not path:
        return ""

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"{env_var}={path} does not exist.\n"
            f"Please verify the path to {description} in your .env file"
        )

    return path


# ONNX2C paths
ONNX2C_CMD = _get_required_path("ONNX2C_CMD", "onnx2c executable")
ONNX2C_INCLUDE = _get_required_path("ONNX2C_INCLUDE", "onnx2c include directory")

# TriCore toolchain
TRICORE_GCC_CMD = _get_required_path("TRICORE_GCC_CMD", "tricore-elf-gcc compiler")
TRICORE_EMULATOR_CMD = _get_required_path(
    "TRICORE_EMULATOR_CMD", "qemu-system-tricore emulator"
)
TRICORE_LINKER_SCRIPT = _get_required_path(
    "TRICORE_LINKER_SCRIPT", "TriCore linker script"
)
TRICORE_CRT = _get_required_path("TRICORE_CRT", "TriCore startup object (crttsim.o)")

# NSIM (Emulator) paths - optional, needed for PPU target only
NSIM_CMD = _get_optional_path("NSIM_CMD", "nSIM emulator")
NSIM_TCF = _get_optional_path("NSIM_TCF", "nSIM TCF configuration file")

# TSIM (TriCore ISS) paths - optional, needed for cycle-accurate simulation
TSIM_CMD = _get_optional_path("TSIM_CMD", "TSIM TriCore ISS executable")
TSIM_MCONFIG = _get_optional_path("TSIM_MCONFIG", "TSIM memory configuration file")

# ARM toolchain
ARM_GCC_CMD = _get_optional_path("ARM_GCC_CMD", "arm-none-eabi-gcc compiler")
ARM_EMULATOR_CMD = _get_optional_path("ARM_EMULATOR_CMD", "qemu-system-arm emulator")
ARM_LINKER_SCRIPT = _get_optional_path(
    "ARM_LINKER_SCRIPT", "ARM Cortex-M linker script"
)
ARM_CRT = _get_optional_path("ARM_CRT", "ARM Cortex-M startup assembly (crt_arm.s)")

# ARC PPU toolchain (LLVM/clang) - optional, needed for PPU target
ARC_CLANG_CMD = _get_optional_path("ARC_CLANG_CMD", "LLVM/clang for ARC compiler")
ARC_SYSROOT = _get_optional_path("ARC_SYSROOT", "ARC newlib sysroot directory")
# ARC_LINKER_SCRIPT removed - clang driver handles linker script selection
