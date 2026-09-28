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

"""ONNX → C → binary → emulate pipeline for the model-zoo conversion service.

Wraps the bundled ``nn2ifx`` benchmarking package with two model-zoo adaptations:
  * ``_plugin_path`` honors the ``QEMU_PLUGIN_DIR`` set in the Docker image;
  * ``run_compare`` aggregates several targets for the Flask ``compare`` endpoint.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np

from nn2ifx.devices import TARGETS
from nn2ifx.pipeline_log import PipelineLog
from nn2ifx.results import (
    AccuracyOutput,
    AccuracyResult,
    BenchmarkResult,
    ProfileNode,
    ProfileResult,
    SingleBenchmarkResult,
    TensorIdentity,
    TimingResult,
    WorkloadIdentity,
    format_single_benchmark,
    sha256_file,
    sha256_json,
    write_single_result,
)
from nn2ifx.emu_parsing import (
    parse_output,
    parse_profile,
    parse_cpi_result,
    parse_inference_profile_result,
    parse_inference_marker,
    parse_nsim_stats,
)
from nn2ifx.comparison import build_comparison, format_comparison


def get_tools(toolchain: str):
    """Return (compile_fn, emulate_fn) for the given toolchain."""
    if toolchain == "tricore":
        from nn2ifx.tools.tricore_gcc import compile
        from nn2ifx.tools.qemu_tricore import run as emulate

        return compile, emulate
    elif toolchain == "arm":
        from nn2ifx.tools.arm_gcc import compile
        from nn2ifx.tools.qemu_arm import run as emulate

        return compile, emulate
    elif toolchain == "ppu":
        from nn2ifx.tools.arc_clang import compile
        from nn2ifx.tools.nsim import run as emulate

        return compile, emulate
    else:
        raise ValueError(f"Unknown toolchain: {toolchain}")


def main():
    parser = argparse.ArgumentParser(
        description="nn2ifx pipeline: ONNX → C → binary → emulate"
    )
    parser.add_argument(
        "-t",
        "--target",
        choices=list(TARGETS.keys()),
        default="tc3xx",
        help="Target hardware (default: tc3xx)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "model.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path(__file__).parent / "test_data_set",
        help="Path to test data directory (input_0.pb, output_0.pb)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(".out"),
        help="Output directory (default: .out)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-node instruction profiling (TriCore/ARM/PPU)",
    )
    parser.add_argument(
        "--profiler",
        choices=["qemu", "tsim"],
        default="tsim",
        help="Timing/profile backend for TriCore targets (default: tsim). "
        "'tsim' isolates the inference region from TSIM's instruction "
        "trace and reports measured, memory-aware cycles per node.",
    )
    args = parser.parse_args()

    # Early validation of input paths
    if not args.model.exists():
        parser.error(f"Model file not found: {args.model}")
    if not args.test_data.exists():
        parser.error(f"Test data directory not found: {args.test_data}")

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
    )

    result = run_target(
        target=args.target,
        model=args.model,
        test_data=args.test_data,
        out=args.output,
        profile=args.profile,
        profiler=args.profiler,
    )

    print("\n" + format_single_benchmark(result))
    print("\nArtifacts")
    print(f"  Results:  {args.output / 'results.json'}")
    print(f"  Log:      {args.output / 'pipeline.log'}")


def run_target(
    target,
    model,
    test_data,
    out,
    profile=False,
    ppu_isolate=False,
    profiler="tsim",
    run_id=None,
) -> SingleBenchmarkResult:
    """Run one clean benchmark and optionally a separate profiling build.

    Writes results.json (and pipeline.log) into `out`. Safe to call directly
    (e.g. from compare.py) without going through argparse.
    """
    out = Path(out)
    device = TARGETS[target]
    compile_fn, emulate_fn = get_tools(device.toolchain)

    _clean_run_artifacts(out)
    log = PipelineLog(out / "pipeline.log")
    try:
        log.begin_step("Clean benchmark pass")
        result = _run_pipeline(
            target,
            device,
            compile_fn,
            emulate_fn,
            out,
            model,
            test_data,
            False,
            log,
            ppu_isolate=(ppu_isolate or device.toolchain == "ppu"),
            profiler=profiler,
        )
        single_result = _to_single_result(
            result, target, device, Path(model), Path(test_data), profiler, run_id
        )

        if profile and result.status == "ok" and result.cycles is not None:
            profile_out = out / "profile"
            log.begin_step("Profiling pass (separate instrumented build)")
            try:
                profile_run = _run_pipeline(
                    target,
                    device,
                    compile_fn,
                    emulate_fn,
                    profile_out,
                    model,
                    test_data,
                    True,
                    log,
                    ppu_isolate=True,
                    profiler=profiler,
                    announce=False,
                )
                single_result.profile = _profile_result(profile_run)
            except (
                Exception
            ) as exc:  # Optional diagnostics must not erase a clean result.
                single_result.profile = ProfileResult(
                    enabled=True,
                    status="failed",
                    build="instrumented",
                    limitations=[f"Profiling pass failed: {exc}"],
                )
        elif profile:
            single_result.profile = ProfileResult(
                enabled=True,
                status="unavailable",
                limitations=[
                    "Clean inference timing was unavailable; profiling was skipped."
                ],
            )

        write_single_result(out, single_result)
        log.begin_step("Final benchmark report")
        log.log_output(format_single_benchmark(single_result), "Report")
    except Exception as exc:
        log.begin_step("Pipeline failed")
        log.log_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        log.close()

    return single_result


_RUN_ARTIFACT_FILES = {
    "results.json",
    "results.json.tmp",
    "pipeline.log",
    "model.c",
    "main.c",
    "model.elf",
    "model.md",
    "ppu_trace.log",
    "model.tsim_prof.log",
    "model.tsim_prof.out",
    "model.tsim_prof.err",
}


def _clean_run_artifacts(out: Path) -> None:
    """Remove artifacts owned by a previous run while preserving user files."""
    out = Path(out)
    for name in _RUN_ARTIFACT_FILES:
        (out / name).unlink(missing_ok=True)
    for path in out.glob("model.elf-*"):
        if path.is_file() or path.is_symlink():
            path.unlink()
    shutil.rmtree(out / "profile", ignore_errors=True)


def _tensor_shape(value_info) -> list[int]:
    return [
        dimension.dim_value if dimension.HasField("dim_value") else -1
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def _workload_identity(model: Path, test_data: Path) -> WorkloadIdentity:
    import onnx

    graph = onnx.load(str(model)).graph
    input_files = sorted(test_data.glob("input_*.pb"))
    output_files = sorted(test_data.glob("output_*.pb"))
    inputs = [
        TensorIdentity(value.name, sha256_file(path), _tensor_shape(value))
        for value, path in zip(graph.input, input_files)
    ]
    outputs = [
        TensorIdentity(value.name, sha256_file(path), _tensor_shape(value))
        for value, path in zip(graph.output, output_files)
    ]
    if len(inputs) != len(graph.input) or len(outputs) != len(graph.output):
        raise ValueError("test data files do not match the model input/output count")
    return WorkloadIdentity(sha256_file(model), 1, inputs, outputs)


def _timing_method(device, profiler: str) -> tuple[str, str, str, str]:
    if device.toolchain == "tricore" and profiler == "tsim":
        return (
            "tsim_trace_cycles",
            "TSIM inference trace timing model",
            "TSIM model",
            "TSIM 1.18.196",
        )
    if device.toolchain == "tricore":
        return (
            "qemu_tricore_cpi",
            "QEMU TriCore instruction-class CPI model",
            "QEMU + TriCore CPI",
            "QEMU TriCore",
        )
    if device.toolchain == "arm":
        return (
            "qemu_arm_cpi",
            "QEMU ARM instruction-class CPI model",
            "QEMU + ARM CPI",
            "QEMU ARM",
        )
    return (
        "nsim_arc_cpi",
        "nSIM instruction trace + ARC CPI model",
        "nSIM + ARC CPI",
        "nSIM 2025.12 FREE",
    )


def _to_single_result(legacy, target, device, model, test_data, profiler, run_id):
    method_id, label, short_label, simulator = _timing_method(device, profiler)
    simulator_options = (
        list(device.default_tsim_opts)
        if method_id == "tsim_trace_cycles"
        else list(device.default_emulator_opts)
    )
    repository_commit, dirty_worktree = _repository_state()
    converter_options = list(device.default_onnx2c_opts)
    compiler_options = list(device.default_compiler_opts)
    timing_available = legacy.basis == "inference" and legacy.cycles is not None
    timing = TimingResult(
        status="available" if timing_available else "unavailable",
        scope="inference" if timing_available else legacy.basis,
        region="entry" if timing_available else "program",
        build="optimized",
        method_id=method_id,
        method_label=label,
        method_short_label=short_label,
        simulator=simulator,
        cycles_kind=(
            "timing_model" if legacy.cycle_source == "measured" else "estimated"
        ),
        cycles=legacy.cycles if timing_available else None,
        instructions=legacy.instructions if timing_available else None,
        average_cpi=legacy.avg_cpi if timing_available else None,
        limitations=list(legacy.notes),
        diagnostics={"program_instructions": legacy.program_instructions},
    )
    accuracy = AccuracyResult()
    if legacy.max_abs_err is not None and legacy.rmse is not None:
        output_elements = _expected_output_elements(test_data)
        accuracy = AccuracyResult(
            status="computed",
            output_count=1,
            total_output_elements=output_elements,
            aggregate_max_abs_error=legacy.max_abs_err,
            aggregate_rmse=legacy.rmse,
            outputs=[
                AccuracyOutput(
                    name="output_0",
                    element_count=output_elements,
                    max_abs_error=legacy.max_abs_err,
                    rmse=legacy.rmse,
                )
            ],
        )
    return SingleBenchmarkResult(
        target=target,
        toolchain=device.toolchain,
        model=str(model),
        nominal_clock_mhz=device.clock_freq_hz / 1e6,
        workload=_workload_identity(model, test_data),
        timing=timing,
        accuracy=accuracy,
        status=legacy.status,
        status_detail=legacy.status_detail,
        reproducibility={
            "repository_commit": repository_commit,
            "dirty_worktree": dirty_worktree,
            "timing_method_id": method_id,
            "clean_build": True,
            "tool_versions": {
                "converter": "ifx-onnx2c 1.1.0",
                "compiler": {
                    "tricore": "AURIX GCC 11.3.1",
                    "arm": "arm-none-eabi-gcc",
                    "ppu": "ARC clang 21.1.8",
                }[device.toolchain],
                "simulator": simulator,
            },
            "converter_options": converter_options,
            "compiler_options": compiler_options,
            "simulator_options": simulator_options,
            "option_hashes": {
                "converter": sha256_json(converter_options),
                "compiler": sha256_json(compiler_options),
                "simulator": sha256_json(simulator_options),
                "timing_configuration": sha256_json(
                    {
                        "method_id": method_id,
                        "nominal_clock_mhz": device.clock_freq_hz / 1e6,
                        "simulator_options": simulator_options,
                    }
                ),
            },
        },
        **({"run_id": run_id} if run_id else {}),
    )


def _repository_state() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _expected_output_elements(test_data: Path) -> int:
    import onnx
    from onnx import numpy_helper

    tensor = onnx.TensorProto.FromString((test_data / "output_0.pb").read_bytes())
    return int(numpy_helper.to_array(tensor).size)


def _node_profile_items(profile_data):
    """Return only generated ONNX node functions from CPU instrumentation."""
    return [item for item in profile_data if item["name"].startswith("node_")]


def _profile_result(profile_run) -> ProfileResult:
    payload = profile_run.profile_payload
    if not payload:
        return ProfileResult(enabled=True, status="unavailable", build="instrumented")

    internal_metric = payload["primary_metric"]
    estimated = payload.get("estimated", False)
    primary_metric = (
        "estimated_cycles"
        if internal_metric == "cycles" and estimated
        else internal_metric
    )
    total = payload["inference_total"]
    nodes = []
    for order, (name, values) in enumerate(payload["nodes"].items()):
        metric_value = values.get(internal_metric, 0)
        nodes.append(
            ProfileNode(
                name=name,
                execution_order=order,
                instructions=values.get("instructions"),
                estimated_cycles=(values.get("cycles") if estimated else None),
                cycles=(values.get("cycles") if not estimated else None),
                share_pct=(100.0 * metric_value / total if total else 0.0),
            )
        )
    node_total = sum(
        values.get(internal_metric, 0) for values in payload["nodes"].values()
    )
    node_instructions = sum(
        values.get("instructions", 0) for values in payload["nodes"].values()
    )
    entry = payload.get("entry") or {}
    overhead_value = entry.get(internal_metric, max(total - node_total, 0))
    overhead_instructions = entry.get("instructions")
    inference_instructions = payload.get("inference_instructions")
    if overhead_instructions is None and inference_instructions is not None:
        overhead_instructions = max(inference_instructions - node_instructions, 0)
    overhead = {
        "name": "entry() glue",
        primary_metric: overhead_value,
        "instructions": overhead_instructions,
        "share_pct": (100.0 * overhead_value / total if total else 0.0),
    }
    attributed = node_total + overhead_value
    unattributed_value = max(total - attributed, 0)
    unattributed_instructions = None
    if inference_instructions is not None and overhead_instructions is not None:
        unattributed_instructions = max(
            inference_instructions - node_instructions - overhead_instructions, 0
        )
    unattributed = {
        primary_metric: unattributed_value,
        "instructions": unattributed_instructions,
        "share_pct": (100.0 * unattributed_value / total if total else 0.0),
    }
    return ProfileResult(
        enabled=True,
        status="available",
        build="instrumented",
        primary_metric=primary_metric,
        inference_total=total,
        attributed_total=attributed,
        coverage_pct=(100.0 * attributed / total if total else None),
        nodes=nodes,
        overhead=overhead,
        unattributed=unattributed,
        limitations=[
            "Node functions are forced out-of-line in the profiling build.",
            "Use node shares to find bottlenecks; this build is not used for headline latency.",
        ],
    )


def _run_pipeline(
    target,
    device,
    compile_fn,
    emulate_fn,
    out,
    model,
    test_data,
    profile,
    log,
    ppu_isolate=False,
    profiler="qemu",
    announce=True,
) -> BenchmarkResult:
    """Execute the full pipeline and return a populated BenchmarkResult.

    ``profile`` enables per-node instruction profiling (a slightly different,
    instrumented build) for every toolchain. ``ppu_isolate`` is a lighter PPU-only
    option that enables nSIM tracing just to isolate the inference region for the
    harmonized basis, without the per-node breakdown or the out-of-line build;
    it is ignored when ``profile`` is set (profiling already traces).

    ``profiler`` selects the TriCore profiling backend: ``"qemu"`` (default,
    CPI-estimated) or ``"tsim"`` (TSIM instruction trace → measured, memory-aware
    cycles per node). Ignored for non-TriCore toolchains.
    """
    out = Path(out)
    model = Path(model)
    test_data = Path(test_data) if test_data else None
    if announce:
        print(f"Target: {target} (toolchain: {device.toolchain})")

    tsim_profiler = profiler == "tsim" and device.toolchain == "tricore"

    result = BenchmarkResult(
        target=target,
        toolchain=device.toolchain,
        clock_mhz=device.clock_freq_hz / 1e6,
        model=str(model),
    )

    # Step 1: ONNX → C (+ main.c with embedded test data)
    cpath = out / "model.c"
    mainpath = out / "main.c"
    if not _generate_c(device, model, cpath, mainpath, test_data, profile, log, result):
        return result

    # Step 1a: Memory-fit check (fail fast before invoking the compiler).
    if not _check_memory_fit(device, cpath.with_suffix(".md"), result, log):
        return result

    # Step 2: Compile
    elf = _compile(
        device,
        compile_fn,
        cpath,
        mainpath,
        profile,
        out,
        log,
        tsim_profiler=tsim_profiler,
    )

    # TriCore TSIM profiler: a dedicated, QEMU-free path that isolates the
    # inference region and reports cycles from the configured timing model.
    if tsim_profiler:
        return _run_tricore_tsim_profile(result, device, elf, test_data, profile, log)

    # Step 3: Emulate
    plugin_path = _plugin_path(device)
    # On PPU, an nSIM instruction trace lets us isolate the inference region
    # (entry()) by PC, matching the other targets' basis. Enabled by --profile
    # (which also attributes instructions per node) or the lighter ppu_isolate
    # path (inference basis only, no per-node / out-of-line build).
    ppu_trace_path = (
        out / "ppu_trace.log"
        if device.toolchain == "ppu" and (profile or ppu_isolate)
        else None
    )
    emu_stdout = _emulate(emulate_fn, device, elf, plugin_path, ppu_trace_path, log)
    emu_result, insn_count = parse_output(emu_stdout)

    # Per-function profile (CPU --profile) parsed up front: reused both to
    # populate the harmonized ARM inference count and to print the per-node
    # table below.
    profile_data, arm_profile_entry_insn = _collect_cpu_profile(
        device, emu_stdout, elf, profile
    )
    if profile_data:
        metric = "cycles" if device.toolchain == "arm" else "instructions"
        entry = next((item for item in profile_data if item["name"] == "entry"), None)
        nodes = {
            item["name"]: {
                "instructions": item["instructions"],
                **(
                    {"cycles": item.get("estimated_cycles", 0)}
                    if metric == "cycles"
                    else {}
                ),
            }
            for item in _node_profile_items(profile_data)
        }
        if entry:
            result.profile_payload = {
                "primary_metric": metric,
                "inference_total": (
                    entry.get("estimated_cycles", 0)
                    if metric == "cycles"
                    else entry["instructions"]
                ),
                "inference_instructions": entry["instructions"],
                "nodes": nodes,
                "estimated": True,
            }

    # Step 3a: harmonized performance metrics (+ PPU per-node tally, if traced)
    ppu_per_node = _populate_performance(
        result,
        device,
        emu_stdout,
        insn_count,
        plugin_path,
        profile,
        elf=elf,
        ppu_trace_path=ppu_trace_path,
        arm_profile_entry_insn=arm_profile_entry_insn,
    )
    # Changes #1/#2: charge spilled-weight (Flash) + spilled-activation (CSM) memory.
    if device.toolchain == "ppu" and result.cycles is not None:
        _apply_ppu_memory_penalty(result, cpath.with_suffix(".md"), model, log)
    result.finalize()
    _report_runtime_estimate(result, device, log)

    # Step 3b: per-node instruction profile (optional)
    _report_profile(device, profile, profile_data, ppu_per_node, insn_count, log)

    # Step 4: Numerical agreement with the supplied expected output
    _compute_accuracy(result, emu_result, test_data)
    return result


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #


def _generate_c(device, model, cpath, mainpath, test_data, profile, log, result):
    """Step 1: convert the ONNX model to C and emit main.c.

    Returns True on success. When onnx2c aborts because the model does not fit
    the target's memory, records a ``model_does_not_fit`` outcome on ``result``
    and returns False so the caller can skip compilation without a traceback.
    """
    from nn2ifx.tools.onnx2c import generate, Onnx2cMemoryError

    try:
        generate(
            onnxpath=model,
            cpath=cpath,
            mainpath=mainpath,
            opts=device.default_onnx2c_opts,
            test_data_dir=test_data,
            profile=profile,
            log=log,
            toolchain=device.toolchain,
        )
    except Onnx2cMemoryError as exc:
        print("\n=== ONNX to C: FAILED (model does not fit target memory) ===")
        print(f"  {exc.summary}")
        print(
            "onnx2c could not place all tensors in the target's memory; "
            "skipping compilation."
        )
        log.log_error(
            "Model exceeds target memory; onnx2c aborted, compilation skipped."
        )
        result.cycle_source = "none"
        result.status = "model_does_not_fit"
        result.status_detail = exc.summary
        result.notes.append(
            "onnx2c could not fit the model to the target's memory; "
            "compilation skipped."
        )
        result.notes.append(exc.summary)
        return False
    return True


def _check_memory_fit(device, reportpath, result, log) -> bool:
    """Step 1a: compare onnx2c's reported sizes against the linker regions.

    Returns True if the model fits (or the target declares no regions). On a
    miss, records the failure on ``result`` and returns False so the caller can
    skip compilation.
    """
    from nn2ifx.memory_check import check_model_fits

    fits, mem_messages = check_model_fits(device, reportpath, log=log)
    if mem_messages:
        log.begin_step("Memory Fit Check")
        for msg in mem_messages:
            log.log_info(msg)
    if fits:
        return True

    print("\n=== Memory Fit Check: FAILED ===")
    for msg in mem_messages:
        print(f"  {msg}")
    print("Model does not fit the target's memory; skipping compilation.")
    for msg in mem_messages:
        log.log_error(f"Memory fit: {msg}")
    log.log_error("Model exceeds target memory; compilation skipped.")
    result.cycle_source = "none"
    result.status = "model_does_not_fit"
    result.status_detail = "; ".join(mem_messages)
    result.notes.append("Model exceeds target memory (won't fit); compilation skipped.")
    result.notes.extend(mem_messages)
    return False


def _compile(
    device, compile_fn, cpath, mainpath, profile, out, log, tsim_profiler=False
) -> Path:
    """Step 2: cross-compile, applying the toolchain's profiling instrumentation.

    ``tsim_profiler`` selects the TriCore TSIM per-node build: node functions are
    forced out-of-line (so their PC ranges survive for trace attribution) but no
    ``-finstrument-functions`` hooks are added (TSIM attributes by PC, not hooks).
    """
    compiler_opts = list(device.default_compiler_opts)
    if tsim_profiler and device.toolchain == "tricore":
        # Per-node TSIM profiling forces nodes out-of-line so their symbols/PC
        # ranges exist in the trace; only meaningful with --profile, but the
        # flag is harmless otherwise (entry()/main isolation still works).
        if profile:
            compiler_opts.append("-DFUNC_PREFIX=__attribute__((noinline))")
    elif profile and device.toolchain in ("tricore", "arm"):
        compiler_opts.append("-finstrument-functions")
    elif profile and device.toolchain == "ppu":
        # No compiler instrumentation on the PPU; per-node profiling attributes
        # nSIM-traced instructions to each node_* function by PC. onnx2c prefixes
        # every node function with the FUNC_PREFIX macro (guarded by #ifndef), so
        # force them out-of-line by defining it to noinline — otherwise clang
        # inlines most nodes into entry() and their cost can't be attributed.
        compiler_opts.append("-DFUNC_PREFIX=__attribute__((noinline))")
        print(
            "Note: --profile on PPU isolates the inference region via nSIM "
            "instruction tracing and attributes instructions per node "
            "(node functions forced out-of-line; slower run)."
        )
    elif profile:
        print(
            f"Warning: --profile not supported for toolchain '{device.toolchain}', ignoring"
        )
    return compile_fn([cpath, mainpath], out, compiler_opts, log=log)


def _plugin_path(device) -> str | None:
    """Resolve the CPI plugin .so for QEMU targets, or None."""
    if device.toolchain not in ("tricore", "arm"):
        return None
    plugin_name = (
        "libcpi_counter_arm.so" if device.toolchain == "arm" else "libcpi_counter.so"
    )
    # The Docker image sets QEMU_PLUGIN_DIR; fall back to the in-repo layout.
    plugin_dir = os.environ.get("QEMU_PLUGIN_DIR")
    if plugin_dir:
        plugin = Path(plugin_dir) / plugin_name
    else:
        plugin = Path(__file__).parent.parent / "Tools" / "qemu_plugin" / plugin_name
    return str(plugin.resolve()) if plugin.exists() else None


def _emulate(emulate_fn, device, elf, plugin_path, ppu_trace_path, log) -> str:
    """Step 3: run the ELF on the target emulator and return its combined output."""
    kwargs = {"log": log}
    if plugin_path:
        kwargs["plugin"] = plugin_path
    elif device.toolchain == "ppu":
        kwargs["trace_path"] = ppu_trace_path
    return emulate_fn(elf, device.default_emulator_opts, **kwargs)


def _collect_cpu_profile(device, emu_stdout, elf, profile):
    """Parse the per-function profile for a CPU (tricore/arm) --profile run.

    Returns ``(profile_data, arm_entry_insn)``; both are None when not
    applicable. ``arm_entry_insn`` is entry()'s inclusive hook-exit delta (the
    inference-region total), used to populate the ARM performance metrics in
    --profile mode where the dedicated marker bracket is skipped (otherwise
    parse_inference_marker's "first line" trick would pick an arbitrary node).
    """
    if not (profile and device.toolchain in ("tricore", "arm")):
        return None, None
    if device.toolchain == "tricore":
        from nn2ifx.config import TRICORE_GCC_CMD as gcc_cmd
    else:
        from nn2ifx.config import ARM_GCC_CMD as gcc_cmd
    profile_data = parse_profile(emu_stdout, elf, gcc_cmd)
    arm_entry_insn = None
    if profile_data and device.toolchain == "arm":
        entry_data = [p for p in profile_data if p["name"] == "entry"]
        if entry_data:
            arm_entry_insn = entry_data[0]["instructions"]
    return profile_data, arm_entry_insn


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _report_runtime_estimate(result: BenchmarkResult, device, log):
    """Print + log the harmonized runtime estimate (when cycles are available)."""
    if result.cycles is None:
        return
    source = {
        "tricore": "CPI plugin",
        "arm": "CPI plugin",
        "ppu": (
            "nSIM trace + ARC CPI table"
            if result.basis == "inference"
            else "nSIM histogram + ARC CPI table"
        ),
    }.get(device.toolchain, "estimate")

    log.begin_step(f"Runtime Estimation ({source})")
    log.log_info(f"Basis: {result.basis}  (cycle source: {result.cycle_source})")
    log.log_info(f"Instructions ({result.basis}): {result.instructions}")
    if result.program_instructions:
        log.log_info(f"Program total instructions: {result.program_instructions}")
    log.log_info(f"Average CPI: {result.avg_cpi:.4f}")
    log.log_info(f"Est. cycles: {result.cycles}")
    log.log_info(f"Clock frequency: {result.clock_mhz:.0f} MHz")
    log.log_info(f"Est. runtime: {result.runtime_us:.1f} µs")
    log.log_info(f"Throughput: {result.throughput_per_s:.0f} inferences/s")


def _report_profile(device, profile, profile_data, ppu_per_node, insn_count, log):
    """Print + log the per-node instruction profile (TriCore/ARM or PPU)."""
    if profile and device.toolchain in ("tricore", "arm"):
        if not profile_data:
            return
        nodes = [
            (p["name"], p["instructions"]) for p in _node_profile_items(profile_data)
        ]
        entry_data = [p for p in profile_data if p["name"] == "entry"]
        # entry()'s inclusive count is the total inference insn (with hooks).
        inference_insn = entry_data[0]["instructions"] if entry_data else insn_count
        node_sum = sum(n for _name, n in nodes)
        overhead = (inference_insn - node_sum) if inference_insn else None
        _report_node_profile(
            log,
            title="Per-Node Instruction Profile",
            rows=nodes,
            node_sum=node_sum,
            overhead_print_label="Profiling overhead (hooks+printf)",
            overhead_value=overhead,
            preamble=[
                "Inference total (clean, without hooks): see non-profile run",
                f"Inference total (with hooks): {inference_insn} insn",
            ],
            sum_log_line=f"Sum (nodes):        {node_sum} insn",
            overhead_log_line=(
                f"Profiling overhead: {overhead} insn  (hooks + printf)"
                if overhead is not None
                else None
            ),
        )

    elif profile and device.toolchain == "ppu" and ppu_per_node:
        # Instructions in shared vector-runtime helpers are folded into the
        # calling node (inclusive basis); entry() glue is reported as overhead.
        profile_counts = dict(ppu_per_node)
        entry_overhead = profile_counts.pop("__entry__", 0)
        nodes = list(profile_counts.items())
        node_sum = sum(n for _name, n in nodes)
        _report_node_profile(
            log,
            title="Per-Node Instruction Profile (nSIM trace)",
            rows=nodes,
            node_sum=node_sum,
            overhead_print_label="entry() glue (overhead)",
            overhead_value=entry_overhead,
            preamble=[
                "Inference instructions attributed per ONNX node "
                "(inclusive of nested vector-runtime helpers).",
            ],
            sum_log_line=f"Sum (nodes):           {node_sum} insn",
            overhead_log_line=f"entry() glue overhead: {entry_overhead} insn",
        )


def _report_node_profile(
    log,
    *,
    title,
    rows,
    node_sum,
    overhead_print_label,
    overhead_value,
    preamble,
    sum_log_line,
    overhead_log_line,
):
    """Render a per-node ``Function | Instructions`` table to console + log."""
    log.begin_step(title)
    for line in preamble:
        log.log_info(line)
    log.log_info("")
    for name, n in rows:
        log.log_result(name, f"{n} insn")
    log.log_info("")
    log.log_info(sum_log_line)
    if overhead_log_line is not None:
        log.log_info(overhead_log_line)


def _run_tricore_tsim_profile(
    result: BenchmarkResult, device, elf, test_data, profile, log
) -> BenchmarkResult:
    """QEMU-free TriCore path: TSIM trace to timing-model cycles.

    Isolates the inference region (``entry()``) from TSIM's ``-e`` trace by PC
    and reports modeled memory-aware cycles (cache + flash wait-states per the
    MConfig). With ``profile`` it also attributes those cycles per ONNX node.
    """
    import os

    from nn2ifx.config import TSIM_CMD
    from nn2ifx.tools.tsim_profile import run_profile

    tsim_executable = bool(
        TSIM_CMD and Path(TSIM_CMD).is_file() and os.access(TSIM_CMD, os.X_OK)
    )
    if not tsim_executable or not getattr(device, "default_tsim_opts", None):
        raise RuntimeError(
            "--profiler tsim requested but TSIM is not configured for this "
            "target (a runnable TSIM_CMD and TSIM_MCONFIG are required). "
            "Run setup.sh or use --profiler qemu."
        )

    prof = run_profile(elf, device.default_tsim_opts, profile_nodes=profile, log=log)
    emu_result, _ = parse_output(prof["stdout"])

    result.cycle_source = "measured"
    result.program_instructions = prof["program_instructions"] or None
    if prof["region_found"] and prof["region_instructions"]:
        result.basis = "inference"
        result.instructions = prof["region_instructions"]
        result.cycles = prof["region_cycles"]
        result.notes.append(
            "TSIM timing-model cycles cover entry() only and include modeled "
            "cache and flash latency from representative MConfig values; "
            "they are not silicon-calibrated measurements."
        )
    else:
        result.basis = "program"
        result.instructions = prof["program_instructions"] or None
        result.cycles = prof["program_cycles"] or None
        result.notes.append(
            "TSIM inference-region isolation unavailable (entry()/main() symbols "
            "missing?); figures cover the whole program (startup + printf)."
        )
    if result.instructions and result.cycles:
        result.avg_cpi = result.cycles / result.instructions
    result.finalize()

    # Detailed timing remains in the pipeline log; the console gets one final report.
    if result.cycles is not None:
        log.begin_step("Runtime (TSIM timing model)")
        log.log_info(f"Basis: {result.basis}  (cycle source: timing model)")
        log.log_info(f"Instructions ({result.basis}): {result.instructions}")
        if result.program_instructions:
            log.log_info(f"Program total instructions: {result.program_instructions}")
        log.log_info(f"Average CPI: {result.avg_cpi:.4f}")
        log.log_info(f"TSIM timing-model cycles: {result.cycles}")
        log.log_info(f"Estimated runtime: {result.runtime_us:.1f} µs")
        log.log_info(f"Throughput: {result.throughput_per_s:.0f} inferences/s")

    # Per-node measured-cycle breakdown.
    if profile and prof["per_node"]:
        per_node = dict(prof["per_node"])
        entry = per_node.pop("__entry__", {"cycles": 0, "instructions": 0})
        result.profile_payload = {
            "primary_metric": "cycles",
            "inference_total": prof["region_cycles"],
            "inference_instructions": prof["region_instructions"],
            "nodes": per_node,
            "estimated": False,
            "entry": entry,
        }
        _report_tsim_node_profile(log, prof["per_node"], result.cycles)
    elif profile:
        print("\nWarning: no node_* symbols found; per-node table unavailable.")

    _compute_accuracy(result, emu_result, test_data)
    return result


def _report_tsim_node_profile(log, per_node: dict, region_cycles):
    """Render the TSIM per-node table (cycles primary, instructions secondary)."""
    # entry() glue is overhead; nodes are reported in execution order.
    entry = per_node.pop("__entry__", None)
    rows = list(per_node.items())
    cyc_sum = sum(v["cycles"] for _n, v in rows)
    insn_sum = sum(v["instructions"] for _n, v in rows)

    log.begin_step("Per-Node Profile (TSIM timing model)")
    log.log_info(
        "TSIM timing-model cycles per ONNX node (inclusive of nested "
        "vector-runtime/helper calls); memory latency is modeled."
    )
    log.log_info("")
    for name, v in rows:
        log.log_result(name, f"{v['cycles']} cyc, {v['instructions']} insn")
    log.log_info("")
    log.log_info(f"Sum (nodes): {cyc_sum} cyc, {insn_sum} insn")
    if entry is not None:
        log.log_info(
            f"entry() glue overhead: {entry['cycles']} cyc, "
            f"{entry['instructions']} insn"
        )


def _compute_accuracy(result: BenchmarkResult, emu_result, test_data):
    """Step 4: compare emulator output against the supplied expected output."""
    if not (test_data and test_data.exists()):
        return
    output_pb = test_data / "output_0.pb"
    if not output_pb.exists():
        return

    import onnx
    from onnx import numpy_helper

    ref_output = numpy_helper.to_array(
        onnx.TensorProto.FromString(output_pb.read_bytes())
    ).flatten()

    if len(emu_result) != len(ref_output):
        print(
            f"WARNING: Output length mismatch: got {len(emu_result)}, expected {len(ref_output)}"
        )
        return
    diff = np.abs(emu_result - ref_output)
    result.max_abs_err = float(diff.max())
    result.rmse = float(np.sqrt(np.mean(diff**2)))


def _populate_performance(
    result: BenchmarkResult,
    device,
    emu_stdout,
    insn_count,
    plugin_path,
    profile,
    elf=None,
    ppu_trace_path=None,
    arm_profile_entry_insn=None,
):
    """Fill in the harmonized cycle/runtime fields for the given target.

    Returns a ``{node_name: instructions}`` per-node tally for the PPU when an
    instruction trace is available (used to print the per-node profile), else
    ``None``.
    """
    if device.toolchain == "ppu":
        return _populate_ppu_perf(result, emu_stdout, profile, elf, ppu_trace_path)
    _populate_cpu_perf(
        result,
        device,
        emu_stdout,
        insn_count,
        plugin_path,
        profile,
        arm_profile_entry_insn,
    )
    return None


def _populate_cpu_perf(
    result, device, emu_stdout, insn_count, plugin_path, profile, arm_profile_entry_insn
):
    """CPI-plugin performance for the QEMU (tricore/arm) targets."""
    cpi_result = parse_cpi_result(emu_stdout) if plugin_path else None
    result.cycle_source = "estimated"
    if cpi_result:
        result.avg_cpi = cpi_result["avg_cpi"]
        result.program_instructions = cpi_result["instructions"]

    arm_hooked_inference = False
    if device.toolchain == "tricore":
        # ICNT register gives the inference-only instruction count.
        inf_insn = insn_count or None
    elif not profile:
        # ARM: DWT CYCCNT isn't emulated; the inference count comes from the
        # plugin's ENTER/EXIT markers around entry() (non-profile runs only).
        inf_insn = parse_inference_marker(emu_stdout)
        marker_result = parse_inference_profile_result(emu_stdout)
        if marker_result:
            inf_insn = marker_result["instructions"]
            result.cycles = marker_result["cycles"]
            result.avg_cpi = result.cycles / inf_insn if inf_insn else None
    else:
        # ARM + --profile: the dedicated marker bracket is skipped (it would
        # corrupt the shared enter/exit stack that -finstrument-functions also
        # uses), but entry()'s own hook exit delta is inclusive of every node it
        # calls, i.e. the same inference-region total.
        inf_insn = arm_profile_entry_insn
        arm_hooked_inference = inf_insn is not None

    if inf_insn:
        result.basis = "inference"
        result.instructions = inf_insn
        if arm_hooked_inference:
            result.notes.append(
                "Inference instruction count includes --profile "
                "instrumentation overhead (function hooks + per-node "
                "printf); slightly higher than a clean (non-profile) run."
            )
    elif cpi_result:
        # Fall back to whole-program total.
        result.basis = "program"
        result.instructions = cpi_result["instructions"]
        result.notes.append(
            "No inference-region counter available; figures cover the whole "
            "program (includes startup/printf)."
        )

    if result.cycles is None and result.instructions and result.avg_cpi:
        result.cycles = int(result.instructions * result.avg_cpi)


def _populate_ppu_perf(result, emu_stdout, profile, elf, ppu_trace_path):
    """ARC-CPI-table performance for the PPU (nSIM) target.

    The bundled nSIM 2025.12 FREE build is functional-only; cycles are estimated
    from an ARC CPI table applied to the whole-program histogram. An optional
    instruction trace isolates the inference region (entry()) and, when the
    node_* symbols survive, attributes instructions to each ONNX node. Returns
    the per-node tally or None.
    """
    from nn2ifx.tools.arc_cpi import (
        estimate_cycles,
        parse_trace_region_and_nodes,
        estimate_from_counts,
    )
    from nn2ifx.tools.nsim import resolve_symbol_ranges, resolve_node_ranges

    per_node = None
    est = estimate_cycles(emu_stdout)  # whole-program histogram
    if est:
        result.cycle_source = "estimated"
        result.program_instructions = est["instructions"]

    # Isolate the inference region from the instruction trace, and (when the
    # node functions are present) split it per ONNX node in the same single pass
    # — the trace can be multiple GB, so walk it once.
    inf = None
    if ppu_trace_path and Path(ppu_trace_path).exists() and elf is not None:
        ranges = resolve_symbol_ranges(Path(elf), {"entry", "main"})
        # Node attribution is only needed (and only meaningful — the build is
        # forced out-of-line) when profiling. The lighter isolate path walks the
        # trace just for the inference-region histogram.
        node_starts = {}
        if profile:
            node_ranges = resolve_node_ranges(Path(elf))
            node_starts = {start: name for name, (start, _e) in node_ranges.items()}
        if "entry" in ranges and "main" in ranges:
            entry_lo, entry_hi = ranges["entry"]
            main_lo, main_hi = ranges["main"]
            region_counts, node_counts, node_histograms = parse_trace_region_and_nodes(
                ppu_trace_path, entry_lo, entry_hi, main_lo, main_hi, node_starts
            )
            if region_counts:
                inf = estimate_from_counts(region_counts)
            if profile and node_counts and node_histograms and inf:
                per_node = node_counts
                entry_histogram = node_histograms.pop("__entry__", {})
                node_values = {}
                for name, histogram in node_histograms.items():
                    estimate = estimate_from_counts(histogram)
                    if estimate is None:
                        continue
                    node_values[name] = {
                        "instructions": estimate["instructions"],
                        "cycles": estimate["cycles"],
                    }
                result.profile_payload = {
                    "primary_metric": "cycles",
                    "inference_total": inf["cycles"],
                    "inference_instructions": inf["instructions"],
                    "nodes": node_values,
                    "estimated": True,
                    "entry": estimate_from_counts(entry_histogram),
                }

    if inf:
        result.basis = "inference"
        result.cycle_source = "estimated"
        result.instructions = inf["instructions"]
        result.avg_cpi = inf["avg_cpi"]
        result.cycles = inf["cycles"]
        result.notes.append(
            "PPU inference region isolated via nSIM instruction trace "
            "(PC-bracketed around entry(), includes nested runtime calls); "
            "cycles estimated from an ARC CPI table. Bundled nSIM 2025.12 "
            "(005_FREE) rejects NCAM (cycles=1); NCAM-enabled nSIM or xCAM "
            "is required for timing counters."
        )
    elif est:
        result.basis = "program"
        result.instructions = est["instructions"]
        result.avg_cpi = est["avg_cpi"]
        result.cycles = est["cycles"]
        result.notes.append(
            "PPU runtime is estimated from nSIM's instruction histogram via "
            "an ARC CPI table; covers the whole program incl. startup/printf "
            "(run with --profile to isolate the inference region via tracing). "
            "Bundled nSIM 2025.12 (005_FREE) rejects NCAM (cycles=1); "
            "NCAM-enabled nSIM or xCAM is required for timing counters."
        )
    else:
        result.basis = "program"
        result.cycle_source = "none"
        stats = parse_nsim_stats(emu_stdout)
        if stats:
            result.instructions = stats["instructions"]
            result.program_instructions = stats["instructions"]
            result.notes.append(
                "nSIM produced no instruction histogram; only the raw "
                "instruction count is available (no cycle/runtime estimate)."
            )

    if est or inf:
        result.notes.append(
            "PPU output differs numerically from the FPU targets due to "
            "vectorized fast-math reductions (expect ~1e-3 abs error)."
        )

    return per_node


def _apply_ppu_memory_penalty(result, model_md, onnx_path, log):
    """Add memory-hierarchy load costs to the PPU estimate: #1 spilled-weight
    (Flash) loads and #2 spilled-activation (CSM_RW) streaming.

    No-op when nothing spills or the residency inputs are unavailable.
    """
    from nn2ifx.tools.arc_cpi import flash_load_penalty, csm_stream_penalty

    flash = flash_load_penalty(model_md, onnx_path)
    csm = csm_stream_penalty(model_md)
    if flash + csm <= 0:
        return
    result.cycles += flash + csm
    if result.instructions:
        result.avg_cpi = result.cycles / result.instructions
    parts = []
    if flash:
        parts.append(f"+{flash:,} spilled-weight Flash loads (change #1)")
    if csm:
        parts.append(f"+{csm:,} spilled-activation CSM streaming (change #2)")
    result.notes.append(
        "Includes memory-hierarchy load cost: " + "; ".join(parts) + "."
    )
    log.log_info(f"PPU memory penalty: Flash +{flash:,}, CSM +{csm:,} cyc")


def run_compare(targets, model, test_data, out, profile=False, profiler="tsim") -> dict:
    """Benchmark a model across several targets and aggregate a comparison.

    Mirrors tests/compare.py: each target's results.json/pipeline.log are written
    to ``<out>/<target>/``, and the aggregated table is written to
    ``<out>/comparison.txt`` and ``<out>/comparison.json``. A single run id ties
    the per-target artifacts together. Failed targets stay represented by a
    status row rather than aborting the whole comparison. Returns the aggregated
    comparison as a dict.
    """
    out = Path(out)
    results: dict = {}
    run_id = str(uuid.uuid4())
    for target in targets:
        outdir = out / target
        try:
            result = run_target(
                target=target,
                model=model,
                test_data=test_data,
                out=outdir,
                profile=profile,
                profiler=profiler,
                run_id=run_id,
            )
            results[target] = result.to_dict()
        except (
            Exception
        ) as exc:  # noqa: BLE001 - report and continue with other targets
            print(f"[{target}] FAILED: {exc}")
            results[target] = {"_comparison_status": "run_failed"}

    comparison = build_comparison(results, list(targets))
    table = format_comparison(comparison)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.txt").write_text(table + "\n")
    (out / "comparison.json").write_text(
        json.dumps(comparison.to_dict(), indent=2) + "\n"
    )
    return comparison.to_dict()


if __name__ == "__main__":
    main()
