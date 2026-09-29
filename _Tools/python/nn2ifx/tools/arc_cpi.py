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
"""CPI-weighted cycle estimation for the Aurix PPU (ARC VPX/HS vector DSP).

The bundled nSIM 2025.12 (005_FREE) is a functional simulator with no cycle
counter: this FREE-flavor build is license-limited and rejects Near-Cycle-Accurate Mode
(``-prop=cycles=1``) with "Cycle Approximate mode is not supported in this
version of nSIM". It only reports an instruction histogram (with
``nsim_emt=1``). To put the PPU on the same footing as the QEMU-based targets
-- which estimate cycles from an instruction-class CPI table -- we apply an
equivalent CPI model to nSIM's histogram. (An NCAM-enabled nSIM would provide
near-cycle-accurate ``crun`` counters; xCAM is the cycle-accurate model.)

The CPI values below are *approximate throughput* figures for the ARC vector
DSP; there is no cycle-accurate PPU reference to calibrate against, so the
resulting runtime is explicitly an estimate (``cycle_source = "estimated"``).
"""

import re

from .trace_attrib import TraceAttributor


# Instruction classes (mirrors the QEMU CPI plugin, plus vector classes).
CLASS_CPI = {
    "ALU": 1,
    "Load": 2,
    "Store": 1,
    "Branch": 2,
    "Mul": 2,
    "FPU_simple": 2,
    "FPU_MAC": 3,
    "FPU_double": 4,
    "FPU_DIV": 20,
    "FPU_SQRT": 20,
    "Vec_simple": 1,  # vector ALU / move / select — throughput ~1
    "Vec_MAC": 1,  # vector fused multiply-add — throughput ~1
    "Vec_Load": 2,
    "Vec_Store": 1,
    "Vec_special": 10,  # vector divide / sqrt / exp — multi-cycle
    "Nop": 1,
}


def classify_arc(mnemonic: str) -> str:
    """Classify an ARC PPU instruction mnemonic into a CPI class."""
    m = mnemonic.strip()

    # No-ops
    if m in ("nop", "vnop"):
        return "Nop"

    # Vector special (divide / sqrt / exp / reciprocal) — check before generic vec
    if m.startswith(("vfsqrt", "vfrdiv", "vfexp", "vfdiv", "vfrsqrt", "vflog")):
        return "Vec_special"
    # Vector fused multiply-add
    if m.startswith(("vfmadd", "vfmsub", "vfmac")):
        return "Vec_MAC"
    # Vector loads / stores
    if m.startswith("vld") or m.startswith("vst"):
        return "Vec_Load" if m.startswith("vld") else "Vec_Store"
    # Generic vector ops (move, add, sub, max, select, shuffle, ...)
    if m.startswith("v"):
        return "Vec_simple"

    # Scalar double-precision FP (fd*)
    if m.startswith("fd"):
        if m.startswith(("fdmadd", "fdmsub")):
            return "FPU_MAC"
        return "FPU_double"

    # Scalar single-precision FP MAC
    if m.startswith(("fsmadd", "fsmsub")):
        return "FPU_MAC"
    # Scalar single-precision FP divide / sqrt
    if m.startswith(("fsdiv", "fsrdiv")):
        return "FPU_DIV"
    if m.startswith("fssqrt"):
        return "FPU_SQRT"
    # Scalar single-precision FP simple (fsadd, fsmul, fssub, fscmpf, ...)
    if m.startswith("fs"):
        return "FPU_simple"
    # FP conversions and *_f / mov_f scalar float ops
    if m.startswith("fcvt") or m.endswith("_f"):
        return "FPU_simple"

    # Integer multiply
    if m.startswith("mpy"):
        return "Mul"

    # Loads (ld, ldd, ldw, ldb, ldh) and lr (aux read)
    if m.startswith("ld") or m == "lr":
        return "Load"
    # Stores (st, std, stw, stb, sth) and sr (aux write)
    if m.startswith("st") or m == "sr":
        return "Store"

    # Branches / jumps / loops
    if m.startswith(("b", "j", "lp", "dbnz")) and not m.startswith(
        ("bmsk", "bclr", "bset", "btst", "bic", "bbit", "bxor")
    ):
        return "Branch"
    if m.startswith("bbit"):
        return "Branch"

    # Everything else: scalar ALU
    return "ALU"


def parse_histogram(output: str) -> dict:
    """Parse nSIM's <Histogram-Instructions> block into {mnemonic: count}."""
    counts = {}
    in_hist = False
    for line in output.splitlines():
        if "<Histogram-Instructions>" in line:
            in_hist = True
            continue
        if "</Histogram-Instructions>" in line:
            break
        if not in_hist:
            continue
        # Rows look like:  "   vmov1_to             |   140464|   46.42"
        parts = line.split("|")
        if len(parts) < 2:
            continue
        mnemonic = parts[0].strip()
        m = re.match(r"\s*(\d+)", parts[1])
        if not mnemonic or not m:
            continue
        # Skip separators / summary rows
        if mnemonic in ("Instruction", "Total", "Delay Slot"):
            continue
        counts[mnemonic] = int(m.group(1))
    return counts


def estimate_from_counts(counts: dict) -> dict | None:
    """Apply the ARC CPI table to a {mnemonic: count} dict.

    Returns a dict with:
        - instructions: total instruction count
        - cycles: CPI-weighted cycle estimate
        - avg_cpi: cycles / instructions
        - class_counts: {class: (count, cycles)}
    or None if the dict is empty.
    """
    if not counts:
        return None

    class_insn = {}
    class_cyc = {}
    total_insn = 0
    total_cyc = 0
    for mnemonic, n in counts.items():
        cls = classify_arc(mnemonic)
        cpi = CLASS_CPI.get(cls, 1)
        class_insn[cls] = class_insn.get(cls, 0) + n
        class_cyc[cls] = class_cyc.get(cls, 0) + n * cpi
        total_insn += n
        total_cyc += n * cpi

    return {
        "instructions": total_insn,
        "cycles": total_cyc,
        "avg_cpi": (total_cyc / total_insn) if total_insn else 0.0,
        "class_counts": {c: (class_insn[c], class_cyc[c]) for c in class_insn},
    }


# Backwards-compatible private alias (kept for existing callers/tests).
_estimate_from_counts = estimate_from_counts


def estimate_cycles(output: str) -> dict | None:
    """Estimate whole-program PPU cycles from an nSIM instruction histogram.

    Returns the same dict shape as :func:`estimate_from_counts`, or None if
    no histogram is present in `output`.
    """
    return estimate_from_counts(parse_histogram(output))


# --------------------------------------------------------------------------- #
# Change #1: memory-hierarchy-aware load cost (spilled-weight Flash penalty).
#
# A vector-load from Flash (a spilled weight) costs far more than one from VCCM,
# but the nSIM histogram cannot see the tier. This adds an external penalty for
# each node's Flash vector-loads that are not hidden behind compute. Calibrated
# on RUL_MLP (the pure Flash-weight matmul); see _CentralScripts/ppu_cpi_model.md.
# --------------------------------------------------------------------------- #
FLASH_EXTRA_CPI = 36  # extra cyc per 32-B Flash vector-load over the VCCM baseline
FLASH_INTENSITY_THRESHOLD = 2.0  # MACs/spilled-byte above which Flash is compute-hidden


def flash_load_penalty(model_md, onnx_path) -> int:
    """Extra PPU cycles for spilled-weight (Flash) loads not hidden by compute.

    Returns 0 when the residency inputs are missing or nothing spills to Flash.
    """
    try:
        from nn2ifx.tools.ppu_residency import gated_flash_vloads

        vloads = gated_flash_vloads(model_md, onnx_path, FLASH_INTENSITY_THRESHOLD)
    except Exception:
        return 0
    return int(vloads) * FLASH_EXTRA_CPI


# --------------------------------------------------------------------------- #
# Change #2: spilled-activation (CSM_RW) streaming cost.
#
# CSM_RW is ~2 cyc for the low-traffic INT8 control (qat), but FP32 CNNs spill
# 4x the activation bytes and stream them (with conv reuse) faster than the flat
# model assumes. The error tracks CSM-spill *volume*, not compute dtype — an
# FP32 vfmadd is ~1 cyc (MNIST_CNN, all-resident, is accurate). So this charges
# per spilled-activation byte; FP32's 4x bytes give it 4x the charge for free.
# Fit on cnn_weather (regular FP32 CNN); see _CentralScripts/ppu_cpi_model.md.
# --------------------------------------------------------------------------- #
CSM_STREAM_CYC_PER_BYTE = 5.3  # extra cyc per spilled-activation (CSM_RW) byte


def csm_stream_penalty(model_md) -> int:
    """Extra PPU cycles for spilled-activation (CSM_RW) streaming.

    Returns 0 when nothing spills to CSM_RW or model.md is unavailable.
    """
    try:
        from nn2ifx.tools.ppu_residency import csm_spill_bytes

        return int(round(csm_spill_bytes(model_md) * CSM_STREAM_CYC_PER_BYTE))
    except Exception:
        return 0


# An nSIM instruction-trace line (property nsim_trace=1) looks like:
#   [0x00003a10] 0x1cc8b348            [VM:0]    AD K  Z    st.a   r13,[...] : ...
# We capture the PC and everything after the [VM:n] field (flags + mnemonic).
_TRACE_RE = re.compile(r"^\[0x([0-9a-fA-F]+)\]\s+.*?\[VM:\d+\]\s+(.*)$")


def _mnemonic_from_trace_tail(tail: str) -> str | None:
    """Extract the instruction mnemonic from a trace line's post-[VM:n] text.

    The text starts with zero or more all-uppercase status/mode flags (A, D, K,
    Z, ...) followed by the (lower-case) mnemonic and its operands. The first
    token that is not entirely uppercase is the mnemonic.
    """
    for tok in tail.split():
        if tok.isupper():
            continue
        return tok
    return None


def parse_trace_region_and_nodes(
    trace_path,
    entry_lo: int,
    entry_hi: int,
    main_lo: int,
    main_hi: int,
    node_starts: dict,
):
    """Single pass over an nSIM trace: region and per-node histograms.

    The nSIM instruction trace can be several gigabytes, so this walks it once
    and returns both pieces of information needed for the PPU profile:

    * ``region_counts`` -- a ``{mnemonic: count}`` histogram of every
      instruction executed in the inference region (entering ``entry()`` until
      control returns into ``main()``), for the whole-region CPI cycle estimate.
    * ``node_counts`` -- a ``{node_name: count}`` instruction tally per ONNX
      node, in execution order. Because onnx2c calls the ``node_*`` functions as
      sequential siblings from ``entry()``, the *currently executing* node is the
      most recent ``node_*`` entry point seen; instructions in shared
      vector-runtime helpers called from that node are attributed to it
      (inclusive, matching the QEMU per-node basis). Instructions executed in
      ``entry()`` glue (argument setup between calls, pre/post the node loop) are
      collected under the ``"__entry__"`` key.
        * ``node_histograms`` -- ``{node_name: {mnemonic: count}}`` for applying
            the same ARC CPI model to each node without walking the trace again.

    Args:
        trace_path: path to the nSIM trace file (``nsim_trace=1`` output).
        entry_lo, entry_hi: ``[start, end)`` PC range of ``entry()``. The region
            begins the first time the PC equals ``entry_lo``.
        main_lo, main_hi: ``[start, end)`` PC range of ``main()``; the region
            ends when the PC first re-enters it.
        node_starts: ``{start_pc: node_name}`` for every ONNX node function.

    Returns ``(region_counts, node_counts, node_histograms)``, or three ``None``
    values if ``entry()`` is never reached in the trace.
    """

    def parse(line: str):
        m = _TRACE_RE.match(line)
        if not m:
            return None
        pc = int(m.group(1), 16)
        return pc, None, _mnemonic_from_trace_tail(m.group(2))

    attr = TraceAttributor(entry_lo, entry_hi, main_lo, main_hi, node_starts, parse)
    with open(trace_path) as f:
        for line in f:
            attr.feed(line)
            if attr.done:
                break

    if not attr.region_found:
        return None, None, None
    return attr.region_mnemonics, attr.node_instructions, attr.node_mnemonics
