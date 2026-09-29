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
"""Parsers for emulator / plugin / simulator output.

These are pure, stateless ``str -> data`` helpers used by the pipeline to pull
model outputs, instruction counts, CPI results and per-function profiles out of
the various emulator and plugin text streams. They contain no pipeline logic so
they can be unit-tested in isolation.
"""

import re
import logging
from pathlib import Path

import numpy as np

from .tools.elf_symbols import nm_entries

logger = logging.getLogger(__name__)

RUN_COMPLETION_MARKER = "NN2IFX_RUN_COMPLETE"


class EmulationError(RuntimeError):
    """Raised when an emulator does not complete the generated harness."""


def validate_emulator_completion(
    *,
    backend: str,
    returncode: int,
    stdout: str,
    stderr: str,
    accepted_returncodes: tuple[int, ...],
) -> None:
    """Require an accepted exit status and the generated harness marker."""
    problems = []
    if returncode not in accepted_returncodes:
        problems.append(f"unexpected exit code {returncode}")
    if RUN_COMPLETION_MARKER not in stdout:
        problems.append(f"missing completion marker {RUN_COMPLETION_MARKER!r}")
    if not problems:
        return

    diagnostic = stderr.strip() or stdout.strip()
    if len(diagnostic) > 2000:
        diagnostic = diagnostic[-2000:]
    detail = f"{backend} failed: {', '.join(problems)}"
    if diagnostic:
        detail += f"\nLast emulator output:\n{diagnostic}"
    raise EmulationError(detail)


def parse_output(stdout: str) -> tuple[np.ndarray, int]:
    """Parse model output values and instruction count from emulator stdout."""
    values = []
    insn_count = 0
    for line in stdout.strip().splitlines():
        line = line.strip()
        # Match "out[N]: actual=<value>" format
        m = re.match(r"out\[\d+\]:\s*actual=([\d.eE+\-]+)", line)
        if m:
            values.append(float(m.group(1)))
            continue
        # Match instruction count
        m = re.match(r"Inference instructions:\s*(\d+)", line)
        if m:
            insn_count = int(m.group(1))
            continue
    return np.array(values, dtype=np.float32), insn_count


def parse_profile(stdout: str, elf_path: Path, gcc_cmd: str) -> list[dict]:
    """Parse PROFILE lines from stdout and map addresses to function names.

    Supports two formats:
      - TriCore: "PROFILE: 0x<addr> <count>" (count from ICNT register)
      - ARM:     "PROFILE: 0x<addr>" + "PROFILE_INSN: <count>" from CPI plugin

    Returns list of dicts with 'name', 'address', 'instructions' keys.
    """
    # Derive nm from the toolchain's gcc and resolve the symbol table via the
    # shared elf_symbols helper (nm --print-size -n), matching the trace profilers.
    nm_cmd = gcc_cmd.replace("-gcc", "-nm")
    entries = nm_entries(Path(elf_path), nm_cmd)
    if not entries:
        logger.warning("nm (%s) produced no symbols for %s", nm_cmd, elf_path)
        return []
    symbols = {}
    for addr, _size, name in entries:
        # Skip _end markers (linker symbols sharing address with next function)
        if name.endswith("_end"):
            continue
        # Keep the first symbol seen at an address; later aliases (or
        # zero-size markers) sharing the same address must not clobber it.
        symbols.setdefault(addr, name)

    def lookup_symbol(addr):
        """Look up symbol by address, handling ARM Thumb bit (LSB)."""
        name = symbols.get(addr)
        if name:
            return name
        # ARM Thumb function pointers have bit 0 set; try without it
        name = symbols.get(addr & ~1)
        if name:
            return name
        return f"0x{addr:08x}"

    # Collect marker deltas (from CPI plugin, ARM marker-based profiling).
    plugin_results = parse_profile_results(stdout)
    plugin_insn_counts = []
    for line in stdout.splitlines():
        m = re.match(r"PROFILE_INSN:\s+(\d+)", line)
        if m:
            plugin_insn_counts.append(int(m.group(1)))

    # Parse PROFILE lines
    profile_data = []
    plugin_idx = 0
    for line in stdout.splitlines():
        # Try full format first: "PROFILE: 0x<addr> <count>" (TriCore)
        m = re.match(r"PROFILE:\s*0x([0-9a-fA-F]+)\s+(\d+)", line)
        if m:
            addr = int(m.group(1), 16)
            insn = int(m.group(2))
            # If count is 0 and we have plugin data, use that instead
            if insn == 0 and plugin_idx < len(plugin_insn_counts):
                insn = plugin_insn_counts[plugin_idx]
                plugin_idx += 1
            name = lookup_symbol(addr)
            item = {"name": name, "address": addr, "instructions": insn}
            if plugin_idx and plugin_idx <= len(plugin_results):
                item["estimated_cycles"] = plugin_results[plugin_idx - 1]["cycles"]
            profile_data.append(item)
            continue
        # Try address-only format: "PROFILE: 0x<addr>" (ARM with plugin markers)
        m = re.match(r"PROFILE:\s*0x([0-9a-fA-F]+)\s*$", line)
        if m:
            addr = int(m.group(1), 16)
            insn = (
                plugin_insn_counts[plugin_idx]
                if plugin_idx < len(plugin_insn_counts)
                else 0
            )
            plugin_idx += 1
            name = lookup_symbol(addr)
            item = {"name": name, "address": addr, "instructions": insn}
            if plugin_idx <= len(plugin_results):
                item["estimated_cycles"] = plugin_results[plugin_idx - 1]["cycles"]
            profile_data.append(item)

    return profile_data


def parse_cpi_result(output: str) -> dict | None:
    """Parse CPI_RESULT line from QEMU plugin output.

    Returns dict with 'cycles', 'instructions', 'avg_cpi' or None.
    """
    for line in output.splitlines():
        m = re.match(r"CPI_RESULT:\s+(\d+)\s+(\d+)\s+([\d.]+)", line)
        if m:
            return {
                "cycles": int(m.group(1)),
                "instructions": int(m.group(2)),
                "avg_cpi": float(m.group(3)),
            }
    return None


def parse_inference_marker(output: str) -> int | None:
    """Parse the inference-region instruction count from a PROFILE_INSN line.

    Emitted by the ARM CPI plugin when the guest brackets entry() with writes
    to the marker addresses (non-profile ARM runs). Returns the first count.
    """
    for line in output.splitlines():
        m = re.match(r"PROFILE_INSN:\s+(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def parse_profile_results(output: str) -> list[dict]:
    """Parse ordered QEMU marker deltas.

    Each result contains ``instructions`` and CPI-model ``cycles`` for one
    enter/exit region. Nested regions are emitted in exit order, matching the
    guest's PROFILE address lines.
    """
    results = []
    for line in output.splitlines():
        match = re.match(r"PROFILE_RESULT:\s+(\d+)\s+(\d+)", line)
        if match:
            results.append(
                {
                    "instructions": int(match.group(1)),
                    "cycles": int(match.group(2)),
                }
            )
    return results


def parse_inference_profile_result(output: str) -> dict | None:
    """Return the first structured marker result for a clean inference run."""
    results = parse_profile_results(output)
    return results[0] if results else None


def parse_nsim_stats(output: str) -> dict | None:
    """Parse nSIM simulation statistics from output.

    The bundled nSIM 2025.12 (005_FREE) is license-limited to functional
    simulation: it rejects NCAM (`-prop=cycles=1`) with "Cycle Approximate mode
    is not supported in this version of nSIM". An NCAM-enabled nSIM would expose
    near-cycle-accurate `crun` counters; the cycle-accurate Synopsys model is
    xCAM. So here only the instruction count is available. With
    nsim_print_stats_on_exit=1, nSIM prints:
      Instruction Count =   20814 [# Total]

    Returns dict with 'instructions' (and 'cycles'/'avg_cpi' when available) or None.
    """
    cycles = None
    instructions = None
    for line in output.splitlines():
        m = re.search(r"[Tt]otal\s+[Cc]ycles\s*[=:]\s*(\d+)", line)
        if m:
            cycles = int(m.group(1))
        m = re.search(r"[Ii]nstruction\s+[Cc]ount\s*=\s*(\d+)", line)
        if m:
            instructions = int(m.group(1))
        else:
            m = re.search(
                r"[Tt]otal\s+[Ii]nstructions?\s+[Ee]xecuted\s*[=:]\s*(\d+)", line
            )
            if m:
                instructions = int(m.group(1))
    if instructions is None:
        return None
    return {
        "cycles": cycles,
        "instructions": instructions,
        "avg_cpi": (cycles / instructions if cycles and instructions > 0 else None),
    }
