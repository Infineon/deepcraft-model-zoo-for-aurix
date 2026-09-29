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
"""Pre-compile memory-fit check.

Before invoking the cross-compiler we can already tell whether a model has any
chance of fitting on the target: onnx2c writes a ``model.md`` report listing the
total size of the network's *constants* (weights/biases -> read-only/flash) and
*variables* (activation buffers -> RAM), and every target's linker script
declares the physical size of those memory regions in its ``MEMORY { ... }``
block.

Comparing the two lets us fail fast with a clear message instead of waiting for
the linker to emit a cryptic ``region 'FLASH' overflowed`` error (or, worse,
producing an ELF that silently won't run).
"""

from pathlib import Path
import re
import logging

logger = logging.getLogger("nn2ifx.memory_check")


# --- model.md parsing -------------------------------------------------------


def parse_model_report(md_path: Path) -> dict:
    """Extract the total constant/variable sizes (bytes) from a model.md report.

    Returns a dict with keys ``constants`` and ``variables``. Missing entries
    are reported as ``None``.
    """
    md_path = Path(md_path)
    text = md_path.read_text()

    def _grab(pattern):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    return {
        "constants": _grab(r"Total Size Constants:\*\*\s*([0-9]+)\s*bytes"),
        "variables": _grab(
            r"Total Size Variables \(Layouted\):\*\*\s*([0-9]+)\s*bytes"
        ),
    }


# --- linker script parsing --------------------------------------------------


def _parse_size_expr(expr: str) -> int:
    """Evaluate a GNU ld size expression like ``4M``, ``16M`` or ``2M-16K-1K``."""
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3}
    total = 0
    sign = 1
    for tok in re.findall(r"[+\-]|0x[0-9a-fA-F]+[KMG]?|\d+[KMG]?", expr):
        if tok == "+":
            sign = 1
        elif tok == "-":
            sign = -1
        else:
            unit = 1
            if tok[-1] in mult:
                unit = mult[tok[-1]]
                tok = tok[:-1]
            total += sign * int(tok, 0) * unit
            sign = 1
    return total


def parse_ld_regions(ld_path: Path) -> dict:
    """Parse the ``MEMORY { ... }`` block of a GNU ld script.

    Returns ``{region_name: length_in_bytes}``. Supports both the
    ``ORIGIN = ..., LENGTH = ...`` and ``org = ..., len = ...`` spellings.
    """
    ld_path = Path(ld_path)
    text = ld_path.read_text()

    block = re.search(r"MEMORY\s*\{(.*?)\}", text, re.DOTALL)
    if not block:
        return {}

    regions = {}
    line_re = re.compile(
        r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*"
        r"(?:ORIGIN|org)\s*=\s*[^,]+,\s*"
        r"(?:LENGTH|len)\s*=\s*([^\n/]+)",
        re.IGNORECASE,
    )
    for line in block.group(1).splitlines():
        m = line_re.match(line)
        if m:
            regions[m.group(1)] = _parse_size_expr(m.group(2))
    return regions


# --- fit check --------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MiB ({n:,} bytes)"


def check_model_fits(device, report_path: Path, log=None) -> tuple[bool, list[str]]:
    """Check whether the generated model fits the target's memory regions.

    Uses the device's ``memory_regions`` descriptor (mapping the onnx2c report
    categories ``constants``/``variables`` to ``(linker_region_name,
    linker_script_path)``). Devices without that attribute are not checked and
    always report a fit.

    Returns ``(fits, messages)`` where ``messages`` describes the per-region
    usage (and any overflow).
    """
    memory_regions = getattr(device, "memory_regions", None)
    if not memory_regions:
        return True, []

    report = parse_model_report(report_path)
    messages = []
    fits = True

    # Cache parsed linker scripts so we only read each once.
    ld_cache: dict = {}

    for category, (region_name, ld_path) in memory_regions.items():
        used = report.get(category)
        if used is None:
            continue
        if ld_path is None:
            continue
        if ld_path not in ld_cache:
            try:
                ld_cache[ld_path] = parse_ld_regions(ld_path)
            except OSError as exc:
                logger.warning(f"Could not read linker script {ld_path}: {exc}")
                ld_cache[ld_path] = {}
        capacity = ld_cache[ld_path].get(region_name)
        if capacity is None:
            logger.warning(
                f"Region '{region_name}' not found in {ld_path}; skipping {category} check"
            )
            continue

        if used > capacity:
            fits = False
            overflow = used - capacity
            messages.append(
                f"{category} ({region_name}): {_fmt_bytes(used)} exceeds "
                f"capacity {_fmt_bytes(capacity)} by {_fmt_bytes(overflow)}"
            )
        else:
            messages.append(
                f"{category} ({region_name}): {_fmt_bytes(used)} of "
                f"{_fmt_bytes(capacity)} ({100 * used / capacity:.1f}%)"
            )

    return fits, messages
