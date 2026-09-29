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
"""Shared ELF symbol-range resolution (toolchain-agnostic).

Both the PPU (nSIM) and the TriCore (TSIM) per-node profilers need to map
program-counter values from an instruction trace back to the function they
belong to. This module resolves ``{name: (start, end)}`` address ranges from an
ELF using an ``nm`` utility, parametrized by the ``nm`` command so it works for
any toolchain (``llvm-nm`` for ARC, ``tricore-elf-nm`` for TriCore, ...).
"""

from pathlib import Path
import subprocess
import logging

logger = logging.getLogger("nn2ifx.elf_symbols")


def nm_entries(elf: Path, nm_cmd: str | None) -> list:
    """Return ``[(addr, size_or_None, name), ...]`` sorted by address.

    Uses ``<nm_cmd> --print-size -n`` on the ELF. Returns an empty list if the
    tool is unavailable or fails.
    """
    if not nm_cmd or not Path(nm_cmd).exists():
        return []
    try:
        proc = subprocess.run(
            [str(nm_cmd), "--print-size", "-n", str(Path(elf).resolve())],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(f"nm symbol resolution failed: {exc}")
        return []

    # Lines look like:  "00003a10 00000b98 T entry"  (with size)
    #              or:  "000045a8 T main"            (no size)
    entries = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4:
            entries.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
        elif len(parts) == 3:
            entries.append((int(parts[0], 16), None, parts[2]))
    return entries


def ranges_from_entries(entries: list, keep) -> dict:
    """Build ``{name: (start, end)}`` for entries whose name satisfies ``keep``.

    ``keep`` is a predicate ``name -> bool``. ``end`` is ``start + size`` when
    the symbol size is known, otherwise the address of the next symbol.
    """
    ranges = {}
    for i, (addr, size, sym) in enumerate(entries):
        if not keep(sym):
            continue
        if size:
            end = addr + size
        else:
            end = entries[i + 1][0] if i + 1 < len(entries) else addr + 4
        ranges[sym] = (addr, end)
    return ranges


def resolve_symbol_ranges(elf: Path, names: set, nm_cmd: str | None) -> dict:
    """Resolve ``{name: (start, end)}`` ranges for the given symbol names."""
    return ranges_from_entries(nm_entries(elf, nm_cmd), lambda sym: sym in names)


def resolve_node_ranges(elf: Path, nm_cmd: str | None, prefix: str = "node_") -> dict:
    """Resolve ``{name: (start, end)}`` ranges for every ONNX node function.

    onnx2c emits one C function per graph node, named ``node_<...>``. When kept
    out-of-line (the profiling build forces this), their PC ranges attribute
    traced instructions to individual nodes. Returns an empty dict if no such
    symbols are present.
    """
    return ranges_from_entries(
        nm_entries(elf, nm_cmd),
        lambda sym: (sym.startswith(prefix) and not sym.endswith("_end")),
    )
