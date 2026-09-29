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
import os
import subprocess
import logging
from ..config import NSIM_CMD, ARC_CLANG_CMD
from ..emu_parsing import validate_emulator_completion

logger = logging.getLogger("nn2ifx.nsim")


def run(elf: Path, opts: list, timeout: int = 1200, log=None, trace_path=None) -> str:
    """
    Run an ELF binary on nSIM ARC emulator and capture output.

    Uses hostlink I/O — the emulator runs to completion and stdout
    is captured as the result.

    Args:
        elf: Path to the ELF executable
        opts: Emulator options list (e.g., ["--tcf=ev71_base.tcf"])
        timeout: Maximum execution time in seconds
        log: Optional PipelineLog instance for structured logging
        trace_path: If set, enable the nSIM instruction trace and write it to
            this file (used to isolate the inference region by PC). Roughly
            doubles run time, so callers pass it only when needed.

    Returns:
        Emulator stdout as a string (with stderr appended)
    """
    if not NSIM_CMD:
        raise RuntimeError(
            "NSIM_CMD not configured. Set it in .env or run setup.sh.\n"
            "The PPU target requires the nSIM tool from the ACS Edge AI package."
        )
    trace_opts = []
    if trace_path is not None:
        trace_path = Path(trace_path).resolve()
        trace_path.unlink(missing_ok=True)
        trace_opts = [
            "-prop=nsim_trace=1",
            f"-prop=nsim_trace-output={trace_path}",
        ]
    cmd = [NSIM_CMD] + opts + trace_opts + [str(elf.resolve())]

    env = os.environ.copy()
    # Older nSIM distributions shipped compatibility libraries beside bin/.
    # The ACS 0.0.2 build does not need or include this directory.
    nsim_lib_dir = Path(NSIM_CMD).parent.parent / "lib"
    if nsim_lib_dir.is_dir():
        env["LD_LIBRARY_PATH"] = (
            str(nsim_lib_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
        )

    logger.info(f"Running emulator: {' '.join(cmd)}")
    if log:
        log.begin_step("Emulation (nSIM ARC PPU)")
        log.log_command(cmd)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"nSIM timed out after {timeout}s")
        if log:
            log.log_error(f"nSIM timed out after {timeout}s")
        raise RuntimeError(f"nSIM timed out after {timeout}s")

    if log:
        log.log_output(result.stderr, "Emulator log")
        log.log_output(result.stdout, "Emulator output")

    validate_emulator_completion(
        backend="nSIM",
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        accepted_returncodes=(0,),
    )
    logger.info(f"Emulator finished (exit code {result.returncode})")

    # Combine stdout and stderr so caller can parse both
    return result.stdout + "\n" + result.stderr


def _arc_nm_cmd() -> str | None:
    """Locate ``llvm-nm`` next to the ARC clang driver, or None."""
    if not ARC_CLANG_CMD:
        return None
    nm = Path(ARC_CLANG_CMD).parent / "llvm-nm"
    return str(nm) if nm.exists() else None


def resolve_symbol_ranges(elf: Path, names: set) -> dict:
    """Resolve ``{name: (start, end)}`` address ranges for the given symbols.

    Returns an empty dict if the tool or symbols are unavailable (callers
    should fall back to whole-program analysis).
    """
    from .elf_symbols import resolve_symbol_ranges as _resolve

    return _resolve(elf, names, _arc_nm_cmd())


def resolve_node_ranges(elf: Path, prefix: str = "node_") -> dict:
    """Resolve ``{name: (start, end)}`` ranges for every ONNX node function.

    onnx2c emits one C function per graph node, named ``node_<...>``. At ``-O2``
    the ARC clang build keeps these as distinct symbols (they are called as
    sequential siblings from ``entry()``), so their PC ranges can be used to
    attribute traced instructions to individual nodes. Returns an empty dict if
    no such symbols are present (e.g. an aggressively-inlined build).
    """
    from .elf_symbols import resolve_node_ranges as _resolve

    return _resolve(elf, _arc_nm_cmd(), prefix)
