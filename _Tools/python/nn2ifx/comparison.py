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

"""Compatibility checks and rendering for schema-v2 benchmark results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass
class ComparisonRow:
    target: str
    status: str
    detail: str | None
    source_status: str | None
    log_path: str | None = None
    eligible: bool = False
    rank: int | None = None
    method_id: str | None = None
    method_label: str | None = None
    nominal_clock_mhz: float | None = None
    runtime_us: float | None = None
    throughput_per_s: float | None = None
    estimated_speedup: float | None = None
    max_abs_error: float | None = None
    rmse: float | None = None


@dataclass
class ComparisonResult:
    selected_targets: list[str]
    workload_id: str | None
    baseline: str | None
    rows: list[ComparisonRow]
    status: str
    limitations: list[str] = field(default_factory=list)
    artifact_type: str = "cross_platform_comparison"
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _target_log_path(target: str) -> str:
    return f"{target}/pipeline.log"


def _failure_log_path(target: str, status: str) -> str | None:
    if status in {"model_does_not_fit", "run_failed"}:
        return _target_log_path(target)
    return None


def _eligibility(payload: dict, workload_id: str | None) -> tuple[str, str | None]:
    if (
        payload.get("artifact_type") != "single_target_benchmark"
        or payload.get("schema_version") != 2
    ):
        return "unsupported_schema", "schema-v2 single-target result required"
    if workload_id is not None and payload.get("workload_id") != workload_id:
        return "incompatible_workload", "workload identity differs from the comparison"
    source_status = payload.get("status")
    if source_status != "ok":
        if source_status == "model_does_not_fit":
            return "model_does_not_fit", "Model exceeds target memory."
        return "run_failed", "Target execution failed."
    accuracy = payload.get("accuracy", {})
    if accuracy.get("status") != "computed":
        return "accuracy_unavailable", "numerical agreement was not computed"
    timing = payload.get("timing", {})
    if timing.get("scope") != "inference" or timing.get("region") != "entry":
        return "incompatible_scope", "timing must cover one entry() inference"
    if timing.get("build") != "optimized":
        return "profile_build_only", "headline timing came from a profiling build"
    required = (
        "method_id",
        "method_short_label",
        "simulator",
        "cycles",
        "runtime_us",
        "throughput_per_s",
    )
    if timing.get("status") != "available" or any(
        timing.get(key) is None for key in required
    ):
        return (
            "timing_unavailable",
            "complete simulator timing provenance is unavailable",
        )
    if not all(
        _is_positive_finite(timing.get(key))
        for key in ("cycles", "runtime_us", "throughput_per_s")
    ):
        return "invalid_timing", "timing values must be positive and finite"
    if not _is_positive_finite(payload.get("nominal_clock_mhz")):
        return "invalid_timing", "nominal target clock must be positive and finite"
    return "ok", None


def build_comparison(
    results: dict[str, dict | None],
    selected_targets: list[str],
    baseline: str | None = None,
) -> ComparisonResult:
    """Validate results, rank eligible rows, and calculate baseline speedups."""
    workload_id = next(
        (
            payload.get("workload_id")
            for payload in results.values()
            if payload
            and payload.get("artifact_type") == "single_target_benchmark"
            and payload.get("schema_version") == 2
        ),
        None,
    )
    rows = []
    for target in selected_targets:
        payload = results.get(target)
        if payload is None:
            rows.append(
                ComparisonRow(
                    target,
                    "missing_result",
                    "No benchmark result was produced.",
                    None,
                )
            )
            continue
        if payload.get("_comparison_status") == "run_failed":
            rows.append(
                ComparisonRow(
                    target,
                    "run_failed",
                    "Target execution failed.",
                    "run_failed",
                    log_path=_target_log_path(target),
                )
            )
            continue
        status, detail = _eligibility(payload, workload_id)
        timing = payload.get("timing", {})
        accuracy = payload.get("accuracy", {})
        rows.append(
            ComparisonRow(
                target=target,
                status=status,
                detail=detail,
                source_status=payload.get("status"),
                log_path=_failure_log_path(target, status),
                eligible=status == "ok",
                method_id=timing.get("method_id"),
                method_label=timing.get("method_short_label"),
                nominal_clock_mhz=payload.get("nominal_clock_mhz"),
                runtime_us=timing.get("runtime_us"),
                throughput_per_s=timing.get("throughput_per_s"),
                max_abs_error=accuracy.get("aggregate_max_abs_error"),
                rmse=accuracy.get("aggregate_rmse"),
            )
        )

    eligible = [row for row in rows if row.eligible]
    if baseline is not None and not any(row.target == baseline for row in eligible):
        raise ValueError(f"baseline '{baseline}' is not an eligible selected target")
    if baseline is None and eligible:
        baseline = (
            "tc4dx"
            if any(row.target == "tc4dx" for row in eligible)
            else eligible[0].target
        )

    eligible.sort(key=lambda row: row.runtime_us)
    for rank, row in enumerate(eligible, 1):
        row.rank = rank
    if baseline is not None:
        baseline_runtime = next(
            row.runtime_us for row in eligible if row.target == baseline
        )
        for row in eligible:
            row.estimated_speedup = baseline_runtime / row.runtime_us

    ordered_rows = eligible + [row for row in rows if not row.eligible]
    comparison_status = (
        "ok"
        if len(eligible) == len(selected_targets)
        else "partial" if eligible else "no_eligible_results"
    )
    return ComparisonResult(
        selected_targets=list(selected_targets),
        workload_id=workload_id,
        baseline=baseline,
        rows=ordered_rows,
        status=comparison_status,
        limitations=[
            "All latency, throughput, and speedup values are simulator-derived estimates.",
            "Architecture-specific cycles, instructions, and CPI are not cross-platform metrics.",
            "Transfer, scheduling, batching, and system integration overhead are excluded.",
        ],
    )


def _time(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.3g} ms"
    return f"{value:.3g} us"


def _speedup(value: float) -> str:
    return "<0.1x" if 0 < value < 0.05 else f"{value:.1f}x"


def _unavailable_detail(row: ComparisonRow) -> str:
    detail = row.detail or "Unavailable."
    if row.log_path:
        return f"{detail} See {row.log_path}."
    return detail


def format_comparison(result: ComparisonResult) -> str:
    """Render a compact public comparison report."""
    lines = [
        "=" * 72,
        "Cross-platform benchmark",
        f"Workload: {result.workload_id or 'unavailable'}",
        "Scope: one optimized entry() inference",
        f"Baseline: {result.baseline or 'none'}",
        "=" * 72,
        "",
        "Performance estimates",
        f"  {'Rank':>4}  {'Target':<12} {'Method':<22} {'Clock':>9} "
        f"{'Est. latency':>13} {'Est. throughput':>16} {'Est. speedup':>13}",
    ]
    for row in (item for item in result.rows if item.eligible):
        assert row.nominal_clock_mhz is not None
        assert row.runtime_us is not None
        assert row.throughput_per_s is not None
        assert row.estimated_speedup is not None
        lines.append(
            f"  {row.rank:>4}  {row.target:<12} {row.method_label:<22} "
            f"{row.nominal_clock_mhz:>6g} MHz {_time(row.runtime_us):>13} "
            f"{row.throughput_per_s:>13,.0f}/s {_speedup(row.estimated_speedup):>13}"
        )

    lines.extend(
        [
            "",
            "Numerical agreement (1 test sample)",
            f"  {'Target':<12} {'Max abs. error':>15} {'RMSE':>12} {'Status':>12}",
        ]
    )
    for row in result.rows:
        max_error = f"{row.max_abs_error:.2e}" if row.max_abs_error is not None else "-"
        rmse = f"{row.rmse:.2e}" if row.rmse is not None else "-"
        accuracy_status = "computed" if row.max_abs_error is not None else "unavailable"
        lines.append(
            f"  {row.target:<12} {max_error:>15} {rmse:>12} {accuracy_status:>12}"
        )

    unavailable = [row for row in result.rows if not row.eligible]
    if unavailable:
        lines.extend(
            ["", "Unavailable targets", f"  {'Target':<12} {'Status':<24} Detail"]
        )
        lines.extend(
            f"  {row.target:<12} {row.status:<24} {_unavailable_detail(row)}"
            for row in unavailable
        )

    lines.extend(["", "Interpretation"])
    lines.extend(f"  {limitation}" for limitation in result.limitations)
    lines.append(
        f"  {len(result.rows) - len(unavailable)} of {len(result.selected_targets)} "
        "selected targets are performance-comparable."
    )
    return "\n".join(lines)
