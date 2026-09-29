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
"""Per-node memory-tier residency for the PPU (changes #1/#2 of the CPI model).

Derives, per ONNX node, how many spilled-weight (Flash) vector-loads it issues,
by joining the onnx2c ``model.md`` tensor tables (size + tier) with the ONNX
graph (each node's constant inputs). Flash loads hidden behind compute (high
MACs per spilled byte) are excluded by ``gated_flash_vloads``; ``csm_spill_bytes``
totals the spilled activations (change #2).

Single source of truth: imported in-container by ``arc_cpi`` to charge the
penalties, and runnable as a host CLI for fitting/validation. See
``testing/cpi_model_implementation.md`` (internal).

Usage (from the repo root):
    python _Tools/python/nn2ifx/tools/ppu_residency.py               # summary
    python _Tools/python/nn2ifx/tools/ppu_residency.py --per-node    # + Flash nodes
    python _Tools/python/nn2ifx/tools/ppu_residency.py --model RUL_MLP
    python _Tools/python/nn2ifx/tools/ppu_residency.py --json tiers.json
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import onnx
from onnx import shape_inference

BYTES_PER_VLOAD = 32  # 256-bit vector load

# model.md section headers that hold tensor rows.
_TENSOR_SECTIONS = {
    "Constant Tensors": "const",
    "Variable Tensors": "var",
    "I/O Tensors": "io",
}


def _tier(row: str, kind: str) -> str:
    """VCCM if resident, else Flash for a spilled constant and CSM_RW for a
    spilled variable/IO. onnx2c does not always print the ``.rodata`` section,
    so the const-vs-var split comes from the table, not the row text.
    """
    if "__vccm" in row:
        return "vccm"
    return "flash" if kind == "const" else "csm"


def parse_model_md(md_path: Path):
    """Return (tensors, sections) from a model.md tensor tables."""
    tensors = {}
    sections = {"const": [], "var": [], "io": []}
    current = None
    for line in Path(md_path).read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current = _TENSOR_SECTIONS.get(stripped[3:].strip())
            continue
        if current is None or not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 7 or cells[0] in ("Tensor ID", "") or not cells[0].isdigit():
            continue
        size = int(re.sub(r"[^\d]", "", cells[3]) or 0)
        name = cells[-1]
        tensors[name] = {
            "size": size,
            "tier": _tier(stripped, current),
            "kind": current,
        }
        sections[current].append(name)
    return tensors, sections


def _node_macs(model):
    """MACs per Conv/Gemm/MatMul node, for arithmetic intensity (0 otherwise)."""
    inferred = shape_inference.infer_shapes(model)
    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    init = {i.name: list(i.dims) for i in model.graph.initializer}

    macs = {}
    for node in model.graph.node:
        weight = next(
            (init[i] for i in node.input if i in init and len(init[i]) >= 2), None
        )
        out = shapes.get(node.output[0]) if node.output else None
        if not weight or not out:
            continue
        out_elems = math.prod(d for d in out if d)
        if node.op_type == "Conv":
            macs[node.name] = out_elems * math.prod(weight[1:])
        elif node.op_type in ("Gemm", "MatMul"):
            k = weight[1] if len(weight) == 2 else weight[0]
            macs[node.name] = out_elems * k
    return macs


def per_node_const_tiers(md_path: Path, onnx_path: Path):
    """Per-node constant-load bytes by tier, joined via the ONNX graph.

    Returns {node_name: {"op", "vccm", "csm", "flash", "flash_vloads", "macs"}}
    for every node that reads at least one constant.
    """
    tensors, _ = parse_model_md(md_path)
    model = onnx.load(str(onnx_path))
    initializers = {i.name for i in model.graph.initializer}
    macs = _node_macs(model)

    per_node = {}
    for node in model.graph.node:
        const_inputs = [i for i in node.input if i in initializers]
        if not const_inputs:
            continue
        tiers = {"vccm": 0, "csm": 0, "flash": 0}
        for name in const_inputs:
            info = tensors.get(name)
            if info is None:  # onnx2c may rename; a miss is not a crash
                continue
            tiers[info["tier"]] += info["size"]
        per_node[node.name or f"<{node.op_type}>"] = {
            "op": node.op_type,
            **tiers,
            "flash_vloads": tiers["flash"] // BYTES_PER_VLOAD,
            "macs": macs.get(node.name, 0),
        }
    return per_node


def gated_flash_vloads(md_path, onnx_path, intensity_threshold: float = 2.0) -> int:
    """Total Flash vector-loads from memory-bound nodes only.

    A node's Flash latency is exposed when MACs/spilled-byte < threshold and
    hidden behind compute otherwise (the arithmetic-intensity gate of change #1).
    Returns 0 if the inputs are missing or nothing spills to Flash.
    """
    md_path, onnx_path = Path(md_path), Path(onnx_path)
    if not md_path.exists() or not onnx_path.exists():
        return 0
    total = 0
    for d in per_node_const_tiers(md_path, onnx_path).values():
        if not d["flash"]:
            continue
        if (
            intensity_threshold is not None
            and d["macs"] / d["flash"] >= intensity_threshold
        ):
            continue  # compute-bound node: Flash latency hidden
        total += d["flash_vloads"]
    return total


def csm_spill_bytes(md_path) -> int:
    """Total spilled-activation (CSM_RW) bytes for a model (change #2 lever).

    Only needs ``model.md`` — the CSM tier is read directly from the Variable
    Tensors table. FP32 activations spill ~4x the bytes of INT8, so this scales
    the streaming penalty by dtype without any compute-CPI change.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return 0
    tensors, sections = parse_model_md(md_path)
    return sum(
        tensors[n]["size"] for n in sections["var"] if tensors[n]["tier"] == "csm"
    )


def model_summary(md_path):
    """Per-model residency totals for cross-checking the residency table."""
    tensors, sections = parse_model_md(md_path)
    spilled_consts = sum(
        t["size"] for n in sections["const"] if (t := tensors[n])["tier"] == "flash"
    )
    spilled_vars = sum(
        t["size"] for n in sections["var"] if (t := tensors[n])["tier"] == "csm"
    )
    spilled = [
        (n, tensors[n]["size"])
        for n in tensors
        if tensors[n]["tier"] != "vccm" and tensors[n]["kind"] != "io"
    ]
    largest = max(spilled, key=lambda kv: kv[1], default=(None, 0))
    weights_resident = all(tensors[n]["tier"] == "vccm" for n in sections["const"])
    return {
        "spilled_consts": spilled_consts,
        "spilled_vars": spilled_vars,
        "largest_spilled": largest,
        "weights_resident": weights_resident,
    }


def discover_reports(root: Path, model_filter=None):
    """Yield (model_name, model.md, model.onnx) for each PPU report under out/."""
    for md in sorted(root.glob("*/out/**/tc4dx_ppu/model.md")):
        onnx_path = md.parent.parent / "model.onnx"
        if not onnx_path.exists():
            continue
        model_name = md.parent.parent.parent.name  # out/<model>/test_<model>/..
        if model_filter and model_filter.lower() not in model_name.lower():
            continue
        yield model_name, md, onnx_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", default=".", help="Repository root to scan (default: cwd)."
    )
    parser.add_argument("--model", default=None, help="Filter models by substring.")
    parser.add_argument(
        "--per-node", action="store_true", help="List nodes with Flash loads."
    )
    parser.add_argument(
        "--json", default=None, help="Write per-node tier bytes to this file."
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    reports = list(discover_reports(root, args.model))
    if not reports:
        print(f"No PPU model.md reports found under {root}.")
        return 1

    all_tiers = {}
    header = (
        f"{'Model':<28} {'spill consts':>13} {'spill vars':>11} {'largest spilled':>28}"
    )
    print(header)
    print("-" * len(header))
    for model_name, md, onnx_path in reports:
        summ = model_summary(md)
        per_node = per_node_const_tiers(md, onnx_path)
        all_tiers[model_name] = per_node
        name, size = summ["largest_spilled"]
        largest = f"{size:,} ({name})" if name else "—"
        print(
            f"{model_name:<28} {summ['spilled_consts']:>13,} "
            f"{summ['spilled_vars']:>11,} {largest:>28}"
        )
        if args.per_node:
            flash_nodes = sorted(
                ((n, d) for n, d in per_node.items() if d["flash"]),
                key=lambda kv: kv[1]["flash"],
                reverse=True,
            )
            for node_name, d in flash_nodes:
                print(
                    f"    {node_name:<44} {d['op']:<10} "
                    f"flash={d['flash']:,} B ({d['flash_vloads']:,} vloads)"
                )

    if args.json:
        Path(args.json).write_text(json.dumps(all_tiers, indent=2))
        print(f"\nWrote per-node tiers for {len(all_tiers)} models to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
