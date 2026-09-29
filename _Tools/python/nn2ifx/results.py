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
"""Harmonized benchmark result schema and cross-platform comparison.

Every pipeline run produces a `BenchmarkResult` (written as `results.json` in
the run's output directory) using the *same* fields regardless of target, so
results from different hardware platforms can be compared directly.

Key idea: the only truly cross-comparable performance numbers are
**estimated runtime** (cycles / clock), **throughput**, and **accuracy**.
Raw instruction counts are architecture-specific (a TriCore scalar add vs. a
PPU 256-bit vector MAC are not the same unit of work), so they are reported
for transparency but are not the basis of comparison.

Each result carries provenance tags so the user knows how trustworthy a number
is:
  - ``cycle_source``: how cycles were obtained
        "measured"   — cycle-accurate simulator (TSIM)
        "estimated"  — CPI model (QEMU plugin or ARC CPI table)
        "none"       — no cycle figure available
  - ``basis``: which region the figures cover
        "inference"  — the model's entry() only (startup/printf excluded)
        "program"    — whole program (includes startup + printf overhead)
"""

import json
import hashlib
import math
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SINGLE_RESULT_SCHEMA_VERSION = 2


@dataclass
class BenchmarkResult:
    target: str
    toolchain: str
    clock_mhz: float
    model: str
    status: str = "ok"
    status_detail: str | None = None

    # Numerical agreement with the supplied expected output
    max_abs_err: float | None = None
    rmse: float | None = None

    # Performance (harmonized)
    basis: str = "inference"  # "inference" | "program"
    cycle_source: str = "estimated"  # "measured" | "estimated" | "none"
    instructions: int | None = None  # count for `basis` region
    program_instructions: int | None = None  # whole-program count (context)
    avg_cpi: float | None = None
    cycles: int | None = None
    runtime_us: float | None = None
    throughput_per_s: float | None = None

    # Free-form notes/caveats surfaced in the comparison
    notes: list = field(default_factory=list)
    profile_payload: dict[str, Any] | None = field(default=None, repr=False)

    def finalize(self):
        """Derive runtime/throughput from cycles + clock when possible."""
        if self.cycles is not None and self.clock_mhz:
            self.runtime_us = self.cycles / (self.clock_mhz * 1e6) * 1e6
            if self.runtime_us > 0:
                self.throughput_per_s = 1e6 / self.runtime_us
        return self


def load_result(path) -> dict:
    return json.loads(Path(path).read_text())


# Schema v2. Kept beside the legacy result during migration so existing pipeline
# entry points remain usable while their producers move to structured reporting.


@dataclass
class TensorIdentity:
    name: str
    sha256: str
    shape: list[int]


@dataclass
class WorkloadIdentity:
    model_sha256: str
    sample_count: int
    inputs: list[TensorIdentity]
    expected_outputs: list[TensorIdentity]

    def workload_id(self) -> str:
        identity = {
            "model_sha256": self.model_sha256,
            "sample_count": self.sample_count,
            "inputs": [
                {"name": item.name, "sha256": item.sha256} for item in self.inputs
            ],
            "expected_outputs": [
                {"name": item.name, "sha256": item.sha256}
                for item in self.expected_outputs
            ],
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass
class AccuracyOutput:
    name: str
    element_count: int
    max_abs_error: float
    rmse: float


@dataclass
class AccuracyResult:
    status: str = "unavailable"
    sample_count: int = 1
    output_count: int = 0
    total_output_elements: int = 0
    aggregate_max_abs_error: float | None = None
    aggregate_rmse: float | None = None
    outputs: list[AccuracyOutput] = field(default_factory=list)


@dataclass
class TimingResult:
    status: str
    scope: str
    region: str
    build: str
    method_id: str
    method_label: str
    method_short_label: str
    simulator: str
    cycles_kind: str = "estimated"
    cycles: int | None = None
    instructions: int | None = None
    average_cpi: float | None = None
    runtime_us: float | None = None
    throughput_per_s: float | None = None
    limitations: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def finalize(self, nominal_clock_mhz: float) -> None:
        if self.status != "available" or self.cycles is None:
            self.runtime_us = None
            self.throughput_per_s = None
            return
        if nominal_clock_mhz <= 0:
            raise ValueError("nominal clock must be positive")
        self.runtime_us = self.cycles / nominal_clock_mhz
        self.throughput_per_s = (
            1_000_000.0 / self.runtime_us if self.runtime_us > 0 else None
        )


@dataclass
class ProfileNode:
    name: str
    execution_order: int
    instructions: int | None = None
    estimated_cycles: int | None = None
    cycles: int | None = None
    share_pct: float = 0.0
    onnx_node_id: str | None = None


@dataclass
class ProfileResult:
    enabled: bool = False
    status: str = "not_requested"
    build: str | None = None
    primary_metric: str | None = None
    inference_total: int | None = None
    attributed_total: int | None = None
    coverage_pct: float | None = None
    nodes: list[ProfileNode] = field(default_factory=list)
    overhead: dict[str, Any] | None = None
    unattributed: dict[str, Any] | None = None
    limitations: list[str] = field(default_factory=list)


@dataclass
class SingleBenchmarkResult:
    target: str
    toolchain: str
    model: str
    nominal_clock_mhz: float
    workload: WorkloadIdentity
    timing: TimingResult
    accuracy: AccuracyResult = field(default_factory=AccuracyResult)
    profile: ProfileResult = field(default_factory=ProfileResult)
    status: str = "ok"
    status_detail: str | None = None
    failure: dict[str, Any] | None = None
    reproducibility: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    artifact_type: str = "single_target_benchmark"
    schema_version: int = SINGLE_RESULT_SCHEMA_VERSION

    def finalize(self) -> "SingleBenchmarkResult":
        self.timing.finalize(self.nominal_clock_mhz)
        self.validate()
        return self

    def validate(self) -> None:
        if self.artifact_type != "single_target_benchmark":
            raise ValueError("invalid single-target artifact type")
        if self.schema_version != SINGLE_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported single-target schema version")
        if self.nominal_clock_mhz <= 0:
            raise ValueError("nominal clock must be positive")
        if self.timing.status == "available":
            if self.timing.scope != "inference" or self.timing.region != "entry":
                raise ValueError("available timing must cover entry() inference")
            if self.timing.build != "optimized":
                raise ValueError("headline timing must use the optimized build")
            if self.timing.cycles is None or self.timing.runtime_us is None:
                raise ValueError("available timing requires cycles and runtime")
            if (
                not math.isfinite(self.timing.cycles)
                or not math.isfinite(self.timing.runtime_us)
                or self.timing.cycles <= 0
                or self.timing.runtime_us <= 0
            ):
                raise ValueError(
                    "available timing requires positive finite cycles and runtime"
                )
            if self.timing.throughput_per_s is not None and (
                not math.isfinite(self.timing.throughput_per_s)
                or self.timing.throughput_per_s <= 0
            ):
                raise ValueError("available timing requires positive finite throughput")
        if self.accuracy.status == "computed":
            values = (
                self.accuracy.aggregate_max_abs_error,
                self.accuracy.aggregate_rmse,
            )
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError("computed accuracy requires finite aggregates")
        if self.profile.enabled and self.profile.status == "available":
            if self.profile.inference_total is None:
                raise ValueError("available profile requires inference total")
            attributed = self.profile.attributed_total or 0
            unattributed = 0
            if self.profile.unattributed:
                metric = self.profile.primary_metric or "cycles"
                unattributed = int(self.profile.unattributed.get(metric, 0))
            if attributed + unattributed != self.profile.inference_total:
                raise ValueError("profile attribution does not match total")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["workload_id"] = self.workload.workload_id()
        return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def write_single_result(out_dir: Path, result: SingleBenchmarkResult) -> Path:
    result.finalize()
    path = Path(out_dir) / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    temporary_path.replace(path)
    return path


def _format_runtime(runtime_us: float | None) -> str:
    if runtime_us is None:
        return "unavailable"
    if runtime_us >= 1000:
        return f"{runtime_us / 1000:.3g} ms"
    return f"{runtime_us:.3g} us"


def format_single_benchmark(result: SingleBenchmarkResult) -> str:
    result.finalize()
    accuracy = result.accuracy
    timing = result.timing
    lines = [
        "=" * 72,
        f"Benchmark: {result.model} on {result.target}",
        "=" * 72,
        "",
        f"Accuracy ({accuracy.sample_count} test sample)",
    ]
    if accuracy.status == "computed":
        lines.extend(
            [
                f"  Max absolute error:  {accuracy.aggregate_max_abs_error:.2e}",
                f"  RMSE:                {accuracy.aggregate_rmse:.2e}",
                f"  Output elements:     {accuracy.total_output_elements:,}",
            ]
        )
    else:
        lines.append("  Status:              unavailable")

    lines.extend(["", "Inference timing"])
    if timing.status == "available":
        cycle_line = (
            f"  TSIM timing-model cycles:  {timing.cycles:,}"
            if timing.cycles_kind == "timing_model"
            else f"  Estimated cycles:    {timing.cycles:,}"
        )
        lines.extend(
            [
                "  Scope:               entry() only",
                f"  Method:              {timing.method_label}",
                cycle_line,
                f"  Target clock:        {result.nominal_clock_mhz:g} MHz",
                f"  Estimated runtime:   {_format_runtime(timing.runtime_us)}",
                f"  Est. throughput:     {timing.throughput_per_s:,.0f} inference/s",
            ]
        )
        if timing.instructions is not None:
            lines.append(f"  Instructions:        {timing.instructions:,}")
        if timing.average_cpi is not None:
            lines.append(f"  Average modeled CPI: {timing.average_cpi:.2f}")
    else:
        lines.append("  Status:              unavailable")

    if timing.limitations:
        lines.extend(["", "Limitations"])
        lines.extend(f"  {limitation}" for limitation in timing.limitations)

    if result.profile.enabled:
        lines.extend(
            ["", "Per-node profile (profiling build; not used for headline runtime)"]
        )
        if result.profile.status != "available":
            lines.append(f"  Status: {result.profile.status}")
        else:
            primary_metric = result.profile.primary_metric or "cycles"
            metric_label = {
                "estimated_cycles": "Est. cycles",
                "cycles": "TSIM cycles",
                "instructions": "Instructions",
            }.get(primary_metric, "Primary metric")
            lines.append(
                f"  {'Node':<44} {metric_label:>12} {'Share':>8} {'Instructions':>14}"
            )
            for node in sorted(
                result.profile.nodes,
                key=lambda item: item.estimated_cycles or item.cycles or 0,
                reverse=True,
            ):
                cycles = (
                    node.estimated_cycles
                    if node.estimated_cycles is not None
                    else node.cycles
                )
                cycle_text = f"{cycles:,}" if cycles is not None else "-"
                insn_text = (
                    f"{node.instructions:,}" if node.instructions is not None else "-"
                )
                lines.append(
                    f"  {node.name:<44} {cycle_text:>12} "
                    f"{node.share_pct:>7.1f}% {insn_text:>14}"
                )
            if result.profile.overhead:
                overhead = result.profile.overhead
                value = overhead.get(primary_metric)
                value_text = f"{value:,}" if value is not None else "-"
                instructions = overhead.get("instructions")
                instructions_text = (
                    f"{instructions:,}" if instructions is not None else "-"
                )
                lines.append(
                    f"  {overhead.get('name', 'Overhead'):<44} {value_text:>12} "
                    f"{overhead.get('share_pct', 0.0):>7.1f}% {instructions_text:>14}"
                )
            unattributed = result.profile.unattributed or {}
            unattributed_value = unattributed.get(primary_metric, 0)
            if unattributed_value:
                instructions = unattributed.get("instructions")
                instructions_text = (
                    f"{instructions:,}" if instructions is not None else "-"
                )
                lines.append(
                    f"  {'Unattributed':<44} {unattributed_value:>12,} "
                    f"{unattributed.get('share_pct', 0.0):>7.1f}% {instructions_text:>14}"
                )
            total_instructions = sum(
                node.instructions or 0 for node in result.profile.nodes
            )
            if result.profile.overhead:
                total_instructions += result.profile.overhead.get("instructions") or 0
            if result.profile.unattributed:
                total_instructions += (
                    result.profile.unattributed.get("instructions") or 0
                )
            lines.append(
                f"  {'Total':<44} {result.profile.inference_total:>12,} "
                f"{100.0:>7.1f}% {total_instructions:>14,}"
            )
    return "\n".join(lines)
