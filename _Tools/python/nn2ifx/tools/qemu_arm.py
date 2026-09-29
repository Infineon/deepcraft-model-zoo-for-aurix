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
from ..config import ARM_EMULATOR_CMD
from ..emu_parsing import validate_emulator_completion

logger = logging.getLogger("nn2ifx.qemu_arm")


def run(elf: Path, opts: list, timeout: int = 200, plugin: str = None, log=None) -> str:
    """
    Run an ELF binary on qemu-system-arm and capture output.

    Uses semihosting for I/O — the emulator runs to completion and
    stdout is captured as the result.

    Args:
        elf: Path to the ELF executable
        opts: Emulator options list (e.g., ["-machine", "mps2-an386"])
        timeout: Maximum execution time in seconds
        plugin: Optional path to QEMU plugin (.so file)
        log: Optional PipelineLog instance for structured logging

    Returns:
        Emulator stdout as a string (with stderr appended if plugin used)
    """
    if not ARM_EMULATOR_CMD:
        raise RuntimeError(
            "ARM_EMULATOR_CMD not configured. Set it in .env or run setup.sh.\n"
            "The ARM target requires qemu-system-arm."
        )
    cmd = [ARM_EMULATOR_CMD] + opts
    if plugin:
        cmd += ["-plugin", plugin]
    cmd += ["-kernel", str(elf.resolve())]

    logger.info(f"Running emulator: {' '.join(cmd)}")
    if log:
        log.begin_step("Emulation (QEMU ARM)")
        log.log_command(cmd)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"Emulator timed out after {timeout}s")
        if log:
            log.log_error(f"qemu-system-arm timed out after {timeout}s")
        raise RuntimeError(f"qemu-system-arm timed out after {timeout}s")

    if log:
        log.log_output(result.stderr, "Emulator log")
        log.log_output(result.stdout, "Emulator output")

    # QEMU ARM exits with code 1 for semihosting SYS_EXIT. The completion
    # marker distinguishes that expected exit from an emulator failure.
    validate_emulator_completion(
        backend="QEMU ARM",
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        accepted_returncodes=(0, 1),
    )
    logger.info(f"Emulator finished (exit code {result.returncode})")
    logger.debug(f"Emulator stdout:\n{result.stdout}")

    # Always return combined stdout+stderr so caller can parse plugin output
    return result.stdout + "\n" + result.stderr
