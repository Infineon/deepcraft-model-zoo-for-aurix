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
from pathlib import Path
import subprocess
import logging
from ..config import ARM_GCC_CMD

logger = logging.getLogger("nn2ifx.arm_gcc")


def compile(srcs: list, out: Path, opts: list, log=None) -> Path:
    """
    Compile C source files into an ELF executable using arm-none-eabi-gcc.

    Args:
        srcs: List of source file paths (.c files and .s assembly)
        out: Output directory for the ELF file
        opts: Compiler options list
        log: Optional PipelineLog instance for structured logging

    Returns:
        Path to the generated ELF file
    """
    if not ARM_GCC_CMD:
        raise RuntimeError(
            "ARM toolchain not configured. Set ARM_GCC_CMD in .env or run setup.sh."
        )
    out.mkdir(parents=True, exist_ok=True)

    elf_path = out / "model.elf"

    # Library flags (-l...) must come *after* the source/object files on the
    # GNU ld command line, otherwise their symbols are dropped before the
    # objects that reference them are seen (e.g. libm's expf -> undefined ref).
    lib_opts = [o for o in opts if o.startswith("-l")]
    non_lib_opts = [o for o in opts if not o.startswith("-l")]

    cmd = (
        [ARM_GCC_CMD]
        + non_lib_opts
        + [str(Path(s).resolve()) for s in srcs]
        + lib_opts
        + ["-o", str(elf_path.resolve())]
    )

    logger.info(f"Compiling: {' '.join(cmd)}")
    if log:
        log.begin_step("ARM Compilation")
        log.log_command(cmd)

    try:
        result = subprocess.run(
            cmd, cwd=out, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("arm-none-eabi-gcc timed out after 600s")

    if result.returncode != 0:
        logger.error(f"Compilation stderr: {result.stderr}")
        logger.error(f"Compilation stdout: {result.stdout}")
        if log:
            log.log_error(
                f"arm-none-eabi-gcc failed with return code {result.returncode}"
            )
            log.log_output(result.stderr, "stderr")
            log.log_output(result.stdout, "stdout")
        raise RuntimeError(
            f"arm-none-eabi-gcc failed with return code {result.returncode}\n{result.stderr}"
        )

    if not elf_path.exists() or elf_path.stat().st_size == 0:
        raise RuntimeError(f"Compilation produced no output: {elf_path}")

    logger.info(f"Compilation successful: {elf_path}")
    if log:
        log.log_output(result.stderr, "Compiler warnings")
        log.log_info(f"ELF output: {elf_path} ({elf_path.stat().st_size} bytes)")

    return elf_path
