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
"""TSIM instruction-trace profiler for TriCore targets.

TSIM does not emulate the ICNT register, so the inference region cannot be
isolated from inside the guest (as the QEMU path does). Instead this module
runs TSIM with its extended instruction trace (``-e``) and attributes every
traced instruction to an ONNX node by program counter — mirroring the PPU
(nSIM) trace attribution, but using TSIM's cycle-accurate per-instruction
counter so per-node figures are **measured cycles**, not a CPI estimate.

The ``-e`` trace line format (verified against tsim16p_e 1.18.196) is::

    PRS0 <instr_idx>(<cum_cycles>,<other>)   <pc_hex> (<sym> + 0x<off>) <disasm> ...

where the leading index is the cumulative instruction count and the first value
in the parentheses is the cumulative cycle count (both match TSIM's final
``Total number of instructions/cycles`` totals). Per-node cycles are therefore
just the delta of the cumulative cycle counter across each node's instructions.

The trace is enormous (hundreds of MB for a small model, GBs for a MobileNet),
so it is streamed through a FIFO and parsed line-by-line — nothing large is ever
written to disk.
"""

from pathlib import Path
import os
import re
import subprocess
import threading
import tempfile
import logging

from ..config import TSIM_CMD, TRICORE_GCC_CMD
from ..emu_parsing import validate_emulator_completion
from .elf_symbols import resolve_symbol_ranges, resolve_node_ranges
from .trace_attrib import TraceAttributor

logger = logging.getLogger("nn2ifx.tsim_profile")

# Trace line (verified for both TSIM models):
#   TC3xx (tc162p):    "PRS0 <idx>(<cyc>,<n>)   <pc> (sym + 0xoff) <disasm...>"
#   TC4Dx (tc18_bstep):"VM0 PRS0 <idx>(<cyc>,<n>)   <pc> ..."  (hypervisor model
#                       prepends a "VM<n> " virtual-machine prefix)
_TRACE_RE = re.compile(r"^(?:VM\d+\s+)?PRS\d+\s+\d+\((\d+),\d+\)\s+([0-9a-fA-F]+)\s")

_TOTAL_INSN_RE = re.compile(r"Total number of instructions executed\s*=\s*(\d+)")
_TOTAL_CYC_RE = re.compile(r"Total number of cycles run\s*=\s*(\d+)")


def _tricore_nm() -> str | None:
    """Derive ``tricore-elf-nm`` from the configured gcc, or None."""
    if not TRICORE_GCC_CMD:
        return None
    return TRICORE_GCC_CMD.replace("-gcc", "-nm")


def _make_tsim_parser():
    """A TSIM ``-e`` trace line parser for :class:`TraceAttributor`.

    Returns ``(pc, cycle_delta, None)`` per matched line, where the cost is the
    delta of TSIM's cumulative cycle counter (the first parenthesised value).
    The previous cumulative value is kept in a closure and updated on *every*
    matched line — including lines before the inference region — so the first
    in-region delta reflects the cycles spent reaching it.
    """
    prev_cum = None

    def parse(line: str):
        nonlocal prev_cum
        m = _TRACE_RE.match(line)
        if not m:
            return None
        cum = int(m.group(1))
        pc = int(m.group(2), 16)
        delta = 0 if prev_cum is None else cum - prev_cum
        prev_cum = cum
        return pc, delta, None

    return parse


def run_profile(
    elf: Path, opts: list, *, profile_nodes: bool, timeout: int = 6000, log=None
) -> dict:
    """Run TSIM with tracing and return measured per-node / region figures.

    Args:
        elf: the TriCore ELF to simulate.
        opts: base TSIM options (``-MConfig``, ``-tc162p`` / ``-tc18_bstep``, ...).
        profile_nodes: also split the inference region per ONNX node (requires
            the ``node_*`` symbols to be out-of-line in the build).
        timeout: max wall-clock seconds for the TSIM run.
        log: optional PipelineLog.

    Returns a dict with keys: ``stdout``, ``program_cycles``,
    ``program_instructions``, ``region_cycles``, ``region_instructions``,
    ``region_found``, ``per_node`` (``{name: {"cycles", "instructions"}}`` or
    ``None``).
    """
    elf = Path(elf).resolve()
    nm = _tricore_nm()

    ranges = resolve_symbol_ranges(elf, {"entry", "main"}, nm)
    if "entry" not in ranges or "main" not in ranges:
        raise RuntimeError(
            "TSIM profiler could not resolve entry()/main() symbols "
            f"(nm={nm}). Cannot isolate the inference region."
        )
    entry_lo, entry_hi = ranges["entry"]
    main_lo, main_hi = ranges["main"]

    node_starts = {}
    if profile_nodes:
        node_ranges = resolve_node_ranges(elf, nm)
        node_starts = {start: name for name, (start, _e) in node_ranges.items()}
        if not node_starts:
            logger.warning(
                "No node_* symbols found; per-node table unavailable "
                "(build must force nodes out-of-line)."
            )

    walker = TraceAttributor(
        entry_lo, entry_hi, main_lo, main_hi, node_starts, _make_tsim_parser()
    )

    work_dir = elf.parent
    log_file = elf.with_suffix(".tsim_prof.log")
    fifo = Path(tempfile.mkdtemp(prefix="tsim_trace_", dir=work_dir)) / "trace.fifo"
    os.mkfifo(fifo)

    cmd = (
        [TSIM_CMD]
        + list(opts)
        + [
            "-H",
            "-S",
            "0x80000020",
            "-x",
            "0",
            "-e",
            "-trace-instr-file",
            str(fifo),
            "-o",
            str(elf),
            "-log-file",
            str(log_file),
        ]
    )

    logger.info(f"Running TSIM (trace profile): {' '.join(cmd)}")
    if log:
        log.begin_step("Emulation (TSIM trace profiler)")
        log.log_command(cmd)

    # Reader thread drains the FIFO while TSIM writes it (avoids huge files and
    # pipe-buffer deadlock). Started before TSIM so the FIFO has a reader.
    reader_err = []

    def _read():
        try:
            with open(fifo, "r") as f:
                for line in f:
                    walker.feed(line)
        except Exception as exc:  # noqa: BLE001 - surfaced to caller below
            reader_err.append(exc)

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()

    stdout_path = elf.with_suffix(".tsim_prof.out")
    stderr_path = elf.with_suffix(".tsim_prof.err")
    try:
        with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
            proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise RuntimeError(f"TSIM trace profile timed out after {timeout}s")

        # Normal case: TSIM closed its trace write end → reader hit EOF.
        reader.join(timeout=60)
        if reader.is_alive():
            # TSIM died before opening the FIFO; unblock the reader's open().
            with open(fifo, "w"):
                pass
            reader.join(timeout=30)
    finally:
        try:
            os.unlink(fifo)
            os.rmdir(fifo.parent)
        except OSError:
            pass

    if reader_err:
        raise RuntimeError(f"TSIM trace reader failed: {reader_err[0]}")

    stdout = stdout_path.read_text() if stdout_path.exists() else ""
    stderr = stderr_path.read_text() if stderr_path.exists() else ""
    validate_emulator_completion(
        backend="TSIM",
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        accepted_returncodes=(0, 2),
    )

    program_cycles = program_instructions = 0
    if log_file.exists():
        log_text = log_file.read_text()
        m = _TOTAL_CYC_RE.search(log_text)
        if m:
            program_cycles = int(m.group(1))
        m = _TOTAL_INSN_RE.search(log_text)
        if m:
            program_instructions = int(m.group(1))

    per_node = None
    if profile_nodes and node_starts and walker.node_instructions:
        per_node = {
            name: {
                "cycles": walker.node_cost.get(name, 0),
                "instructions": walker.node_instructions.get(name, 0),
            }
            for name in walker.node_instructions
        }

    result = {
        "stdout": stdout,
        "program_cycles": program_cycles,
        "program_instructions": program_instructions,
        "region_cycles": walker.region_cost,
        "region_instructions": walker.region_instructions,
        "region_found": walker.region_found,
        "per_node": per_node,
    }

    logger.info(
        f"TSIM trace profile: region {walker.region_instructions} insn / "
        f"{walker.region_cost} cyc; program {program_instructions} insn / "
        f"{program_cycles} cyc"
    )
    if log:
        log.log_output(stdout, "Program output")
        log.log_result(
            "TSIM region",
            f"{walker.region_instructions} insn, {walker.region_cost} cycles "
            f"(inference); program {program_instructions} insn, "
            f"{program_cycles} cycles",
        )

    return result
