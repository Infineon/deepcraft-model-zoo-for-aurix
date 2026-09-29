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


# helper_functions should contain functions that will be shared with different Model Zoo Animals / Models.

import os

# TensorFlow warning suppression - must be set BEFORE any TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress ALL TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Turn off oneDNN custom operations

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import onnx
from onnx import TensorProto
import onnx.numpy_helper as nh
from onnx.numpy_helper import to_array
import onnxruntime
import onnxsim

import pandas as pd
import requests
import seaborn as sns

from IPython.display import display, HTML

import tensorflow as tf

import tf2onnx
from matplotlib import pyplot as plt
from matplotlib import gridspec

import re

_ONNX2C_VERSION = (
    (Path(__file__).parent.parent / "_Tools" / "onnx2c_version.txt").read_text().strip()
)

COLORS = {
    "BLACK": "#1D1D1D",
    "WHITE": "#FFFFFF",
    "OCEAN": "#0A8276",
    "OCEAN_1": "#3B9B91",
    "OCEAN_2": "#6CB4AD",
    "OCEAN_3": "#B8DEDA",
    "LAWN_MAIN": "#9BBA43",
    "BERRY_MAIN": "#9C216E",
    "ENGINEERING_MAIN": "#575352",
    "SUN_MAIN": "#F97414",
    "SAND_MAIN": "#FCD442",
}


def save_data(model_folder, data, is_input=False):

    if is_input:
        data = np.expand_dims(data, axis=0)  # Ensure data is 4D for ONNX input

    onnx_input_tensor = nh.from_array(data)
    data_path = f"{model_folder}/test_data_set"
    dataset_dir = Path(data_path)

    if is_input:
        file = dataset_dir / "input_0.pb"
    else:
        file = dataset_dir / "output_0.pb"

    if not dataset_dir.exists():
        os.makedirs(dataset_dir)

    with open(file, "wb") as f:
        f.write(onnx_input_tensor.SerializeToString())


def update_batch_size(model, new_batch_size):
    for input_tensor in model.graph.input:
        shape = input_tensor.type.tensor_type.shape
        dim = shape.dim[0]  # Batch size is usually the first dimension
        dim.dim_value = new_batch_size  # Set the new batch size
    return model


def onnx_export(model, input, path, origin, opset_version=15):

    if origin == "tf":
        input_tensor = numpy_to_tensor(input, origin)
        model.output_names = ["output"]

        # Check if input already has batch dimension
        if len(input.shape) >= 2 and input.shape[0] == 1:
            spec = (tf.TensorSpec(input.shape, tf.float32, name="input"),)
        else:
            spec = (tf.TensorSpec((1, *input_tensor.shape), tf.float32, name="input"),)

        onnx_model, _ = tf2onnx.convert.from_keras(
            model, input_signature=spec, opset=opset_version
        )
        onnx.save(onnx_model, path)

    elif origin == "torch":
        model = model.to("cpu")
        input_tensor = numpy_to_tensor(input, origin)

        torch.onnx.export(
            model,
            input_tensor,
            path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
        )

        model = load_onnx_model(path)
        model = update_batch_size(model, 1)
        model, _ = onnxsim.simplify(model)
        onnx.save(model, path)
    else:
        raise ValueError("Origin is only defined as 'tf' or 'torch'!")


def clean_dir(model_name):
    model_folder, _ = get_output_paths(model_name)
    dataset_dir = Path(model_folder)
    if not dataset_dir.exists():
        os.makedirs(dataset_dir)
        print(f"Directory created: {dataset_dir}")
    else:
        print(f"Directory already exists: {dataset_dir}")


def get_output_paths(model_name):
    model_folder = os.path.join("out", model_name, f"test_{model_name}")
    onnx_model_file = os.path.join(model_folder, "model.onnx")

    return model_folder, onnx_model_file


def numpy_to_tensor(array, origin):
    if origin == "torch":
        return torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)
    elif origin == "tf":
        return tf.convert_to_tensor(array, dtype=tf.float32)


def get_predictions(origin, model, input):
    if origin == "torch":
        device = torch.device("cpu")
        with torch.no_grad():
            model.to(device)
            input_tensor = numpy_to_tensor(input, origin)
            input_tensor = input_tensor.to(device)
            predictions = model(input_tensor)
            return predictions.numpy()

    elif origin == "tf":
        with tf.device("/cpu:0"):
            # Check if input already has batch dimension
            if len(input.shape) >= 2 and input.shape[0] == 1:
                # Input already has batch dimension (e.g., (1, 28, 28))
                input_tf = tf.convert_to_tensor(input, tf.float32)
            else:
                # Input needs batch dimension (e.g., (28, 28))
                input_tf = tf.convert_to_tensor(np.expand_dims(input, 0), np.float32)
            output = model.predict(input_tf)
        return output


def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        print(f"File does not exist: {model_path}")
        return None
    if not model_path.endswith(".onnx"):
        print(f"File is not an ONNX model: {model_path}")
        return None
    print(f"Model loaded from {model_path}")
    return onnx.load(model_path)


def save_all(model_name, input_target, output_target, model, origin, opset=15) -> None:
    model_folder, onnx_model_file = get_output_paths(model_name)
    clean_dir(model_name)
    save_data(model_folder, input_target, is_input=True)
    save_data(model_folder, output_target, is_input=False)
    onnx_export(model, input_target, onnx_model_file, origin, opset_version=opset)


def analyse_onnx(model):
    total_params = 0
    params_per_layer = {}
    num_layers = 0

    for node in model.graph.node:
        layer_name = node.name

        layer_params = 0

        for tensor in list(node.input) + list(node.output):
            for initializer in model.graph.initializer:
                if initializer.name == tensor:
                    tensor_shape = initializer.dims

                    tensor_params = 1
                    for dim in tensor_shape:
                        tensor_params *= dim

                    total_params += tensor_params
                    layer_params += tensor_params

        params_per_layer[layer_name] = layer_params

        if layer_params > 0:
            num_layers += 1
    return num_layers, total_params, params_per_layer


# Canonical hardware targets and their display names (used in charts/tables).
TARGET_DISPLAY = {
    "tc3xx": "AURIX\u2122 TC3x",
    "tc4dx": "AURIX\u2122 TC4x",
    "tc4dx_ppu": "AURIX\u2122 TC4x PPU",
    "arm_m4": "TRAVEO\u2122 T2G Cortex\u00ae-M4",
    # Legacy folder names kept for backwards compatibility
    "TC3": "AURIX\u2122 TC3x",
    "TC4": "AURIX\u2122 TC4x",
}

DEFAULT_TARGETS = ["tc3xx", "tc4dx", "tc4dx_ppu", "arm_m4", "TC3", "TC4"]


def load_results_json(results_path):
    """Load a harmonized results.json produced by the conversion pipeline."""
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def extract_node_df_from_results(results_path):
    """Extract the per-node instruction profile from a results.json.

    The pipeline writes a ``per_node`` list (``{"node", "instructions"}`` per
    ONNX node) into results.json when run with ``profile=True``, for every
    hardware target. Returns a DataFrame with columns ``node`` and ``clk``
    (instruction count), or an empty DataFrame when no profile is present.
    """
    per_node = load_results_json(results_path).get("per_node") or []
    data = [
        (e.get("node"), int(e.get("instructions", 0)))
        for e in per_node
        if e.get("node") is not None
    ]
    return pd.DataFrame(data, columns=["node", "clk"])


def _node_df_for_entry(entry):
    """Per-node instruction DataFrame for a target, preferring results.json.

    Falls back to parsing the pipeline.log profile section for older runs that
    predate the ``per_node`` field in results.json.
    """
    if "results" in entry:
        df = extract_node_df_from_results(entry["results"])
        if not df.empty:
            return df
    if "log" in entry:
        return extract_node_clk_to_df(entry["log"])
    return pd.DataFrame(columns=["node", "clk"])


def _target_total_instructions(entry):
    """Total instruction count for a target.

    Prefers the harmonized results.json; falls back to summing the per-node
    profile from pipeline.log.
    """
    if "results" in entry:
        instr = load_results_json(entry["results"]).get("instructions")
        if instr is not None:
            return instr
    if "log" in entry:
        df = extract_node_clk_to_df(entry["log"])
        if not df.empty:
            return int(df["clk"].sum())
    return np.nan


def display_inference_benchmark(model_folder, target):
    """Render the timing & accuracy overview from a target's results.json."""
    results_path = os.path.join(model_folder, target, "results.json")
    results = load_results_json(results_path)

    timing = results.get("timing", {})
    accuracy = results.get("accuracy", {})

    def fmt(value, spec=None):
        if value is None:
            return "\u2014"
        return format(value, spec) if spec else str(value)

    # Group shared context under "General", keep the metrics specific to each section
    rows = [
        ("General", "Target", TARGET_DISPLAY.get(target, target)),
        ("General", "Method", fmt(timing.get("method_label"))),
        ("General", "Simulator", fmt(timing.get("simulator"))),
        (
            "General",
            "Nominal clock [MHz]",
            fmt(results.get("nominal_clock_mhz"), ".0f"),
        ),
        ("Timing", "Status", fmt(timing.get("status"))),
        (
            "Timing",
            f"Cycles ({fmt(timing.get('cycles_kind'))})",
            fmt(timing.get("cycles"), ",d"),
        ),
        ("Timing", "Instructions", fmt(timing.get("instructions"), ",d")),
        ("Timing", "Average CPI", fmt(timing.get("average_cpi"), ".3f")),
        ("Timing", "Latency [\u00b5s]", fmt(timing.get("runtime_us"), ".2f")),
        ("Timing", "Throughput [1/s]", fmt(timing.get("throughput_per_s"), ",.0f")),
        ("Accuracy", "Status", fmt(accuracy.get("status"))),
        ("Accuracy", "Samples", fmt(accuracy.get("sample_count"), ",d")),
        (
            "Accuracy",
            "Output elements",
            fmt(accuracy.get("total_output_elements"), ",d"),
        ),
        (
            "Accuracy",
            "Max abs. error",
            fmt(accuracy.get("aggregate_max_abs_error"), ".2e"),
        ),
        ("Accuracy", "RMSE", fmt(accuracy.get("aggregate_rmse"), ".2e")),
    ]

    df = pd.DataFrame(rows, columns=["Section", "Metric", "Value"])

    # Show each section label only on its first row so it sits at the top of the group
    df["Section"] = df["Section"].mask(df["Section"].duplicated(), "")

    display(
        df.style.hide(axis="index")
        .set_caption(f"Inference benchmark \u2014 {TARGET_DISPLAY.get(target, target)}")
        .set_properties(subset=["Value"], **{"text-align": "right"})
        .set_properties(subset=["Section"], **{"font-weight": "bold"})
    )

    # Show the estimation method's limitations as a wrapping list (no horizontal scroll)
    limitations = timing.get("limitations", [])
    if limitations:
        items = "".join(
            f"<li style='margin-bottom:4px'>{note}</li>" for note in limitations
        )
        display(
            HTML(
                "<div style='max-width:720px; white-space:normal'>"
                "<b>Limitations</b>"
                f"<ul style='margin-top:4px; padding-left:20px'>{items}</ul>"
                "</div>"
            )
        )


def plot_profile_shares(model_folder, target):
    """Plot each node's share of the compute load from a target's results.json."""
    results = load_results_json(os.path.join(model_folder, target, "results.json"))

    if results.get("status") != "ok":
        message = f"No results for {TARGET_DISPLAY.get(target, target)}: {results.get('status', 'unknown')}"
        detail = results.get("status_detail")
        if detail:
            message += f" ({detail})"
        print(message)
        return

    profile = results.get("profile", {})
    if profile.get("status") != "available":
        print(
            f"Profiling data unavailable for {TARGET_DISPLAY.get(target, target)}: "
            f"{profile.get('status', 'unknown')}"
        )
        return

    node_df = pd.DataFrame(
        [
            {"Node": n["name"], "Share of compute load": n["share_pct"]}
            for n in profile["nodes"]
        ]
    )

    _, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(node_df))))
    sns.barplot(
        data=node_df,
        x="Share of compute load",
        y="Node",
        color=COLORS["OCEAN"],
        ax=ax,
    )
    ax.set_xlabel("Share of compute load")
    ax.set_ylabel("")
    ax.set_title(f"Per-node profiling \u2014 {TARGET_DISPLAY.get(target, target)}")
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
    ax.margins(x=0.12)
    plt.tight_layout()
    plt.show()


UNAVAILABLE_STATUS_DISPLAY = {
    "model_does_not_fit": "Memory limit exceeded",
    "run_failed": "Execution failed",
    "missing_result": "Missing result",
}


def _unavailable_target_row(row, model_folder):
    target = row["target"]
    log_path = os.path.join(target, "pipeline.log")
    detail = (
        f"See {model_folder}/{log_path}, comparison.txt, and comparison.json "
        "for details."
    )

    return {
        "Target": TARGET_DISPLAY.get(target, target),
        "Status": UNAVAILABLE_STATUS_DISPLAY.get(row["status"], row["status"]),
        "Detail": detail,
    }


def display_platform_comparison(model_folder, display_output=True):
    """Build the cross-platform benchmark table from a model's comparison.json."""
    with open(os.path.join(model_folder, "comparison.json")) as f:
        comparison = json.load(f)

    comparison_columns = [
        "Rank",
        "Target",
        "Method",
        "Clock [MHz]",
        "Est. latency [ms]",
        "Est. throughput [1/s]",
        "Est. speedup",
        "Max abs. error",
        "RMSE",
    ]
    comparison_df = pd.DataFrame(
        [
            {
                "Rank": row["rank"],
                "Target": TARGET_DISPLAY.get(row["target"], row["target"]),
                "Method": row["method_label"],
                "Clock [MHz]": row["nominal_clock_mhz"],
                "Est. latency [ms]": row["runtime_us"] / 1000,
                "Est. throughput [1/s]": row["throughput_per_s"],
                "Est. speedup": row["estimated_speedup"],
                "Max abs. error": row["max_abs_error"],
                "RMSE": row["rmse"],
            }
            # Only eligible rows carry performance data; others have None fields.
            for row in comparison["rows"]
            if row.get("eligible")
        ],
        columns=comparison_columns,
    ).sort_values("Rank")

    if display_output:
        print(
            f"Baseline: {TARGET_DISPLAY.get(comparison['baseline'], comparison['baseline'])}"
        )

        display(
            comparison_df.style.hide(axis="index")
            .format(
                {
                    "Clock [MHz]": "{:.0f}",
                    "Est. latency [ms]": "{:.2f}",
                    "Est. throughput [1/s]": "{:.0f}",
                    "Est. speedup": "{:.1f}x",
                    "Max abs. error": "{:.2e}",
                    "RMSE": "{:.2e}",
                }
            )
            .background_gradient(subset=["Est. speedup"], cmap="Greens")
            .set_caption("Cross-platform benchmark \u2014 one optimized inference")
        )

    unavailable_rows = [row for row in comparison["rows"] if not row.get("eligible")]
    if unavailable_rows and display_output:
        unavailable_df = pd.DataFrame(
            [_unavailable_target_row(row, model_folder) for row in unavailable_rows]
        )
        display(
            unavailable_df.style.hide(axis="index").set_caption("Unavailable targets")
        )

    limitations = comparison.get("limitations", [])
    if limitations and display_output:
        items = "".join(
            f"<li style='margin-bottom:4px'>{note}</li>" for note in limitations
        )
        display(
            HTML(
                "<div style='max-width:720px; white-space:normal'>"
                "<b>Interpretation</b>"
                f"<ul style='margin-top:4px; padding-left:20px'>{items}</ul>"
                "</div>"
            )
        )

    return comparison_df


def search_model_folder(folder, targets=None):
    """Locate per-target artifacts (results.json + pipeline.log) under `folder`.

    Returns a dict mapping each found target to a dict with keys ``"results"``
    (path to results.json) and ``"log"`` (path to pipeline.log).
    """
    if targets is None:
        targets = DEFAULT_TARGETS

    found = {}
    for root, dirs, _ in os.walk(folder):
        for target in targets:
            if target in dirs and target not in found:
                target_folder = os.path.join(root, target)
                entry = {}
                results_path = os.path.join(target_folder, "results.json")
                if os.path.exists(results_path):
                    entry["results"] = results_path
                log_path = os.path.join(target_folder, "pipeline.log")
                if os.path.exists(log_path):
                    entry["log"] = log_path
                else:
                    logs = [f for f in os.listdir(target_folder) if f.endswith(".log")]
                    if logs:
                        entry["log"] = os.path.join(target_folder, logs[0])
                if entry:
                    found[target] = entry
    return found


def extract_node_clk_to_df(log_path):
    """Extract per-node instruction counts from a pipeline.log profile section.

    The pipeline writes a "Per-Node Instruction Profile" block where each node
    is logged as ``  <node_name>: <N> insn``. Returns a DataFrame with columns
    ``node`` and ``clk`` (instruction count).
    """
    data = []
    pattern = re.compile(r"^\s*([A-Za-z0-9_./]+):\s*(\d+)\s*insn\s*$")
    in_profile = False

    with open(log_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if "Per-Node Instruction Profile" in stripped:
                in_profile = True
                continue
            if not in_profile:
                continue
            if stripped.startswith("Sum (nodes)") or stripped.startswith(
                "Profiling overhead"
            ):
                continue
            match = pattern.match(line.rstrip("\n"))
            if match:
                node = match.group(1)
                if node in ("entry", "Inference total"):
                    continue
                data.append((node, int(match.group(2))))
    return pd.DataFrame(data, columns=["node", "clk"])


def _render_instruction_counts_chart(df, hue_col, title, palette, is_small_font=False):
    """Shared rendering logic for instruction-count bar charts."""
    num_nodes = len(df["node"].unique())

    if num_nodes <= 10:
        figsize = (7, 6)
        fontsize = 11
    elif num_nodes <= 25:
        figsize = (7, max(10, num_nodes * 0.5))
        fontsize = 11
    elif num_nodes <= 50:
        figsize = (7, max(12, num_nodes * 0.4))
        fontsize = 9
    else:
        figsize = (7, max(14, num_nodes * 0.3))
        fontsize = 9

    _, ax = plt.subplots(1, 1, figsize=figsize)

    n_hues = df[hue_col].nunique()
    palette = palette[:n_hues]
    sns.barplot(data=df, y="node", x="clk", hue=hue_col, palette=palette)
    ax.set_title(title)
    ax.set_xlabel("Instruction counts")
    ax.set_xscale("log")
    ax.set_ylabel("")
    if is_small_font:
        for label in ax.get_xticklabels():
            label.set_fontsize(7)

    ax.grid(True, axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="y", labelsize=fontsize)

    if num_nodes > 30:
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    plt.tight_layout()
    plt.show()


def plot_instruction_counts(model_name, is_small_font=False):
    model_folder, _ = get_output_paths(model_name)
    found = search_model_folder(model_folder)
    df_list = []
    totals = {}

    for target, entry in found.items():
        df_t = _node_df_for_entry(entry)
        if df_t.empty:
            continue
        display = TARGET_DISPLAY.get(target, target)
        df_t["Target"] = display
        totals[display] = int(df_t["clk"].sum())
        df_list.append(df_t)

    if not df_list:
        print(
            "No per-node profile data found. Run the conversion with profile=True "
            "(supported on all targets: TriCore, Arm and PPU)."
        )
        return

    df = pd.concat(df_list, axis=0)

    if df.empty:
        print(
            "No data to plot in the logfiles, probably a compilation error. Check the log manually!"
        )
        return

    title = "Total instruction counts " + ", ".join(
        f"{name}: {value}" for name, value in totals.items()
    )
    _render_instruction_counts_chart(
        df,
        hue_col="Target",
        title=title,
        palette=[
            COLORS["BERRY_MAIN"],
            COLORS["OCEAN_3"],
            COLORS["SUN_MAIN"],
            COLORS["LAWN_MAIN"],
            COLORS["OCEAN"],
        ],
        is_small_font=is_small_font,
    )


# Post-training quantization variants produced by the notebook, in report order.
# Each entry is (display label, folder suffix); the FP32 baseline has no suffix.
QUANT_VARIANTS = [
    ("FP32", None),
    ("INT8 (static)", "static_ptq_int8"),
    ("Dynamic", "dynamic_ptq"),
]


def _variant_model_name(model_name, suffix):
    return model_name if suffix is None else f"{model_name}_{suffix}"


def parse_model_size(model_md_path):
    """Read compiled constant and total variable sizes from an onnx2c model.md.

    Returns a dict with ``constants`` (weights) and ``variables`` (activation
    buffers, including layouted and non-layout storage) in bytes, or ``None``
    for a field that is absent.
    """
    sizes = {"constants": None, "variables": None}
    try:
        with open(model_md_path, "r") as f:
            text = f.read()
    except OSError:
        return sizes

    const = re.search(r"Total Size Constants:\*\*\s*([\d,]+)\s*bytes", text)
    var_layouted = re.search(
        r"Total Size Variables \(Layouted\):\*\*\s*([\d,]+)\s*bytes", text
    )
    var_no_layout = re.search(
        r"Total Size Variables \(No-Layout\):\*\*\s*([\d,]+)\s*bytes", text
    )
    if const:
        sizes["constants"] = int(const.group(1).replace(",", ""))
    if var_layouted or var_no_layout:
        sizes["variables"] = sum(
            int(match.group(1).replace(",", ""))
            for match in (var_layouted, var_no_layout)
            if match
        )
    return sizes


def _normalize_node_name(name):
    """Align per-node names across FP32 and quantized profiles.

    Strips the ``node_`` prefix and folds the dynamic-quant ``_Gemm_MatMul``
    rename back onto ``_Gemm`` so the same layer lines up across variants.
    """
    display = name[len("node_") :] if name.startswith("node_") else name
    display = display.lstrip("_")
    if display.endswith("_Gemm_MatMul"):
        display = display[: -len("_MatMul")]
    return display


def _is_quant_node(name):
    return name.endswith("QuantizeLinear") or name.endswith("DequantizeLinear")


def plot_node_profile_compare(
    model_name, target="tc4dx_ppu", metric="estimated_cycles"
):
    """Grouped horizontal bar chart of per-node cost across quantization variants.

    Reads ``profile.nodes`` from each variant's results.json, normalizes node
    names so the same layer aligns across FP32/INT8/dynamic, and plots the
    chosen ``metric`` (``estimated_cycles`` or ``instructions``) on a log x-axis
    to keep nodes comparable despite large per-variant totals. Quantize/
    Dequantize nodes (present only in quantized variants) are grouped at the
    bottom as quantization overhead.
    """
    palette = [
        COLORS["OCEAN"],
        COLORS["BERRY_MAIN"],
        COLORS["SUN_MAIN"],
        COLORS["LAWN_MAIN"],
    ]

    rows = []
    order = []  # preserve first-seen node order, quant nodes pushed to the end
    seen = set()
    quant_nodes = []
    labels_present = []

    for (label, suffix), color in zip(QUANT_VARIANTS, palette):
        variant = _variant_model_name(model_name, suffix)
        model_folder, _ = get_output_paths(variant)
        results_path = os.path.join(model_folder, target, "results.json")
        nodes = load_results_json(results_path).get("profile", {}).get("nodes") or []
        if not nodes:
            continue
        labels_present.append(label)
        for node in nodes:
            value = node.get(metric)
            if value is None:
                continue
            display = _normalize_node_name(node["name"])
            if display not in seen:
                seen.add(display)
                (quant_nodes if _is_quant_node(node["name"]) else order).append(display)
            rows.append({"node": display, "Variant": label, "value": value})

    if not rows:
        print(
            f"No per-node profile data for '{model_name}' on "
            f"{TARGET_DISPLAY.get(target, target)}. Run the conversion with profile=True."
        )
        return

    node_order = order + quant_nodes
    df = pd.DataFrame(rows)

    num_nodes = len(node_order)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * num_nodes)))
    sns.barplot(
        data=df,
        y="node",
        x="value",
        hue="Variant",
        order=node_order,
        hue_order=labels_present,
        palette=palette[: len(labels_present)],
        ax=ax,
    )
    metric_label = (
        "Estimated cycles" if metric == "estimated_cycles" else "Instruction counts"
    )
    ax.set_xscale("log")
    ax.set_xlabel(f"{metric_label} (log scale)")
    ax.set_ylabel("")
    ax.set_title(f"Per-node profiling \u2014 {TARGET_DISPLAY.get(target, target)}")
    ax.grid(True, axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(title="", loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()


def compare_quantization_overview(
    model_name,
    target="tc4dx_ppu",
    accuracy=None,
    output_root=None,
    display_output=True,
):
    """Overview table comparing FP32 and quantized variants on one target.

    Columns: latency and estimated cycles (from results.json ``timing``),
    compiled weight and activation sizes (from model.md), weight-size reduction
    versus FP32, output fidelity (max abs error / RMSE from results.json
    ``accuracy``) and, when supplied, classification accuracy.

    ``accuracy`` is the dict returned by
    ``quantization_helper.validate_quantization_options`` (keyed by the ONNX
    model path); pass it through to populate the accuracy column.
    """
    accuracy = accuracy or {}
    rows = []
    baseline_weights = None

    for label, suffix in QUANT_VARIANTS:
        variant = _variant_model_name(model_name, suffix)
        if output_root is None:
            model_folder, onnx_path = get_output_paths(variant)
        else:
            model_folder = os.path.join(output_root, variant, f"test_{variant}")
            onnx_path = os.path.join(model_folder, "model.onnx")
        target_folder = os.path.join(model_folder, target)
        results = load_results_json(os.path.join(target_folder, "results.json"))
        if not results:
            continue
        timing = results.get("timing", {})
        acc = results.get("accuracy", {})
        sizes = parse_model_size(os.path.join(target_folder, "model.md"))

        weights = sizes["constants"]
        if suffix is None:
            baseline_weights = weights

        rows.append(
            {
                "Variant": label,
                "Latency [\u00b5s]": timing.get("runtime_us"),
                "Est. cycles": timing.get("cycles"),
                "Weights [KB]": weights / 1024 if weights is not None else None,
                "Activations [KB]": (
                    sizes["variables"] / 1024
                    if sizes["variables"] is not None
                    else None
                ),
                "Weight reduction": (
                    1 - weights / baseline_weights
                    if weights is not None and baseline_weights
                    else None
                ),
                "Accuracy [%]": accuracy.get(onnx_path),
                "Max abs. error": acc.get("aggregate_max_abs_error"),
                "RMSE": acc.get("aggregate_rmse"),
            }
        )

    if not rows:
        print(f"No results for '{model_name}' on {TARGET_DISPLAY.get(target, target)}.")
        return

    df = pd.DataFrame(rows)
    styler = (
        df.style.hide(axis="index")
        .format(
            {
                "Latency [\u00b5s]": "{:,.1f}",
                "Est. cycles": "{:,.0f}",
                "Weights [KB]": "{:,.1f}",
                "Activations [KB]": "{:,.1f}",
                "Weight reduction": "{:.0%}",
                "Accuracy [%]": "{:.2f}",
                "Max abs. error": "{:.2e}",
                "RMSE": "{:.2e}",
            },
            na_rep="\u2014",
        )
        .set_caption(
            f"Quantization overview \u2014 {TARGET_DISPLAY.get(target, target)}"
        )
    )
    if display_output:
        display(styler)
    return df


def ensure_docker_container(
    url="http://localhost:8080/convert",
    docker_image=f"ai_model_zoo_tools:{_ONNX2C_VERSION}",
):
    try:
        response = requests.get(url, timeout=100)
        if response.status_code == 200:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"ancestor={docker_image}",
                    "--format",
                    "{{.Names}}",
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
            container_name = result.stdout.strip()
            print(
                f"Docker container '{container_name}' (from image '{docker_image}') is running at {url}"
            )
            return
    except Exception:
        print("Container not reachable. Starting container...")

    subprocess.run(
        ["docker", "run", "-p", "127.0.0.1:8080:8080", "-d", f"{docker_image}"],
        check=True,
    )
    time.sleep(5)  # Wait for the container to start

    # Check if the container is running
    result = subprocess.run(
        [
            "docker",
            "ps",
            "--filter",
            f"ancestor={docker_image}",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
    )
    container_name = result.stdout.strip()
    if container_name:
        print(f"Container is running. Name: {container_name}")
    else:
        print("Container is not running.")


def get_numpy_array(file_path):
    with open(file_path, "rb") as f:
        serialized_tensor = f.read()

    onnx_tensor = TensorProto()
    onnx_tensor.ParseFromString(serialized_tensor)
    return to_array(onnx_tensor)


def get_onnx_tensor(model_path):
    return onnxruntime.InferenceSession(model_path)


def get_pb(folder, name):
    return get_numpy_array(os.path.join(folder, name))


def get_onnx_pb(model_name):
    model_folder, onnx_model_file = get_output_paths(model_name)
    input_array = get_pb(model_folder, os.path.join("test_data_set", "input_0.pb"))
    output_array = get_pb(model_folder, os.path.join("test_data_set", "output_0.pb"))
    ort_session = get_onnx_tensor(onnx_model_file)
    return ort_session, input_array, output_array


def get_onnx_predictions(ort_session, input_array):
    ort_inputs = {ort_session.get_inputs()[0].name: input_array}
    return ort_session.run(None, ort_inputs)[0]


def test_onnx_pb(model_name):

    ort_session, input_array, output_array = get_onnx_pb(model_name)
    ort_outs = get_onnx_predictions(ort_session, input_array)
    diff = np.max(np.abs(np.array(output_array) - np.array(ort_outs)))

    if diff < 1e-4:
        print("Output matches expected output within tolerance.")
    else:
        print("Output does not match expected output. Max difference:", diff)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_training_history(history, model_name="Model"):
    fig = plt.figure(figsize=(8, 5))
    gs = gridspec.GridSpec(3, 1)

    # Get colors excluding BLACK and WHITE
    available_colors = [
        color
        for color_name, color in COLORS.items()
        if color_name not in ["BLACK", "WHITE"]
    ]

    # Plot loss and accuracy
    ax1 = fig.add_subplot(gs[0:2, :])
    color_index = 0
    for key in history.history.keys():
        if key != "learning_rate":
            color = available_colors[color_index % len(available_colors)]
            ax1.plot(
                history.epoch, history.history[key], label=key, linewidth=2, color=color
            )
            color_index += 1

    ax1.set_title(f"{model_name} Training History")
    ax1.legend(loc="best")
    ax1.set_xticklabels([])
    ax1.set_ylabel("Loss/Accuracy")
    ax1.set_yscale("log")

    # Plot learning rate
    ax2 = fig.add_subplot(gs[2])
    if "learning_rate" in history.history:
        ax2.plot(
            history.epoch,
            history.history["learning_rate"],
            label="learning rate",
            color=COLORS["OCEAN"],
        )
        ax2.set_yscale("log")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="best")
    ax2.set_ylabel("Learning rate")

    plt.tight_layout()
    plt.show()


def predict_class(model, input, threshold=0.5):
    output = get_onnx_predictions(model, input)
    scores = np.asarray(output)
    if scores.size == 1:
        prediction = np.asarray(scores).reshape(-1)[0] > threshold
        return np.asarray(prediction, dtype="int32"), output
    return np.argmax(scores).astype("int32"), output
