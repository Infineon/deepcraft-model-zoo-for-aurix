# Copyright (C) Infineon Technologies AG 2025
#
# Use of this file is subject to the terms of use agreed between (i) you or the company in which ordinary course of
# business you are acting and (ii) Infineon Technologies AG or its licensees. If and as long as no such terms of use
# are agreed, use of this file is subject to following:
#
# This file is licensed under the terms of the Boost Software License. See the LICENSE file in the root of this repository
# for complete details.

# helper_functions should contain functions that will be shared with different Model Zoo Animals / Models.

import os

# TensorFlow warning suppression - must be set BEFORE any TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress ALL TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Turn off oneDNN custom operations

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import subprocess
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

import onnx
from onnx import TensorProto
import onnx.numpy_helper as nh
from onnx.numpy_helper import to_array
import onnxruntime
from onnx_opcounter import calculate_params, calculate_macs
import onnxsim

import pandas as pd
import requests
import seaborn as sns

import tensorflow as tf

import tf2onnx
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import gridspec
from torchvision import transforms

import re

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
        file = dataset_dir / f"input_0.pb"
    else:
        file = dataset_dir / f"output_0.pb"

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


def transform_to_tensor(image):
    transform = transforms.Compose([transforms.ToTensor()])  # convert image to tensor
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)


def get_image_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform_to_tensor(image)

    return image_tensor


def get_image(image_path):
    image = import_image_tensor(image_path)
    image = downsample_image_tensor(image)
    return image


def import_image_tensor(image_path):
    with Image.open(image_path) as img:
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
    return img


def downsample_image_tensor(image):
    return nn.functional.interpolate(
        image, size=(224, 224), mode="bilinear", align_corners=False
    )


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


def calculate_metrics(model_name):
    model_folder, onnx_model_file = get_output_paths(model_name)
    onnx_model = load_onnx_model(onnx_model_file)
    num_layers, total_params, _ = analyse_onnx(onnx_model)

    log_files = search_model_folder(model_folder)
    clk_tc3 = clk_tc4 = np.nan

    if "TC3" in log_files:
        df_tc3 = extract_node_clk_to_df(log_path=log_files["TC3"])
        if df_tc3.empty:
            print("TC3 log file found but no data extracted.")
        else:
            clk_tc3 = df_tc3["clk"].sum()
    if "TC4" in log_files:
        df_tc4 = extract_node_clk_to_df(log_path=log_files["TC4"])
        if df_tc4.empty:
            print("TC4 log file found but no data extracted.")
        else:
            clk_tc4 = df_tc4["clk"].sum()

    res = {
        "Model type": model_name.split("_")[0],
        "No. Paramters": total_params,
        "No. Layers": num_layers,
        "TC3 clock cycles": clk_tc3,
        "TC4 clock cycles": clk_tc4,
    }

    return pd.DataFrame(res, index=[1])


def search_model_folder(folder):
    targets = ["TC3", "TC4"]
    log_files = {}

    for root, dirs, _ in os.walk(folder):
        for target in targets:
            if target in dirs and target not in log_files:
                target_folder = os.path.join(root, target)
                log_file = [f for f in os.listdir(target_folder) if f.endswith(".log")]
                if log_file:
                    print(f"Found .log file in {target} folder: {log_file[0]}")
                    log_files[target] = os.path.join(target_folder, log_file[0])
                else:
                    print(f"No .log file found in {target} folder.")

        if len(log_files) == len(targets):
            break

    for target in targets:
        if target not in log_files:
            print(f"{target} folder does not exist in {folder} or its subfolders.")

    return log_files


def extract_node_clk_to_df(log_path):
    data = []
    pattern = re.compile(r"\b(node_[\w_]+)\b.*?(\d+)\s+0\s*$")

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            match = pattern.search(line)
            if match:
                node = match.group(1).replace("node_", "", 1)
                clk = int(match.group(2))
                data.append((node, clk))
    df = pd.DataFrame(data, columns=["node", "clk"])
    return df


def plot_execution_timing(model_name, is_small_font=False):
    model_folder, _ = get_output_paths(model_name)
    log_files = search_model_folder(model_folder)
    df_list = []
    clk_tc3 = clk_tc4 = 0

    if "TC3" in log_files:
        df_tc3 = extract_node_clk_to_df(log_path=log_files["TC3"])
        clk_tc3 = df_tc3["clk"].sum()
        df_tc3["Target"] = "AURIX TC3x"
        df_list.append(df_tc3)
    if "TC4" in log_files:
        df_tc4 = extract_node_clk_to_df(log_path=log_files["TC4"])
        clk_tc4 = df_tc4["clk"].sum()
        df_tc4["Target"] = "AURIX TC4x"
        df_list.append(df_tc4)

    if df_list:
        df = pd.concat(df_list, axis=0)
    else:
        print("No TC3 or TC4 log files found.")
        return

    if df.empty:
        print(
            "No data to plot in the logfiles, probably a compilation error. Check the log manually!"
        )
        return

    # Adjust figure size and font based on number of nodes
    num_nodes = len(df["node"].unique())

    # Scale figure height to accommodate all node labels
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

    sns.barplot(
        data=df,
        y="node",
        x="clk",
        hue="Target",
        palette=[COLORS["BERRY_MAIN"], COLORS["OCEAN_3"]],
    )
    ax.set_title(f"Total clock cycles AURIX TC3x: {clk_tc3}, AURIX TC4x: {clk_tc4}")
    ax.set_xlabel("Clock cycles")
    ax.set_xscale("log")
    ax.set_ylabel("")
    if is_small_font:
        for label in ax.get_xticklabels():
            label.set_fontsize(7)

    plt.tight_layout()

    # Add gridlines for better readability
    ax.grid(True, axis="x", alpha=0.3, linestyle="-", linewidth=0.5)

    # Ensure all tick labels are visible and readable
    ax.tick_params(axis="y", labelsize=fontsize)

    # Rotate labels if there are many nodes to improve readability
    if num_nodes > 30:
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()


def ensure_docker_container(
    url="http://localhost:8080/convert",
    docker_image="ghcr.io/infineon/docker-image-infineon:latest",
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
        ["docker", "run", "-p", "8080:8080", "-d", f"{docker_image}"], check=True
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


def extract_clock_cycles(path):
    clocks = 0
    with open(path, "r") as file:
        lines = file.readlines()

        for line in lines:
            if "clocks" in line:
                # split the line before and after the word clocks
                parts = line.split("clocks")
                clocks = int(parts[1])
    return clocks


def count_onnx_model(model):
    params = calculate_params(model)
    macs = calculate_macs(model)
    return params, macs


def get_processing_frequency(target="TC4"):

    match target:
        case "TC4":
            return 5e6
        case _:
            print("target note defined")
            return None


def get_idealized_runtime_bound(model, target="TC4"):
    processing_frq = get_processing_frequency(target)
    _, number_macs = count_onnx_model(model)

    if target == "TC4":
        ideal_clock_count = number_macs
        ideal_runtime = number_macs / processing_frq  # assume 1 fused mac
    else:
        print("target not defined")
        ideal_clock_count = 0
        ideal_runtime = 0

    return ideal_clock_count, ideal_runtime


def get_clock_counts(model_folder, target):
    path = f"{model_folder}/{target}/model_conversion.log"
    number_clocks = extract_clock_cycles(path)
    return number_clocks


def get_clock_counts_from_file(model_name):
    model_folder, onnx_model_file = get_output_paths(model_name)
    model_onnx = onnx.load_model(onnx_model_file)
    ideal_clock_count, ideal_runtime = get_idealized_runtime_bound(model_onnx)
    return ideal_clock_count


def analyze_onnx(model_name, target):
    model_folder, onnx_model_file = get_output_paths(model_name)
    number_clocks = get_clock_counts(model_folder, target)
    ideal_clock_count = get_clock_counts_from_file(model_name)
    utilization = ideal_clock_count / number_clocks

    print(f"Ideal clock count: {ideal_clock_count}")
    print(f"Emulated clock count: {number_clocks}")
    print(f"Utilization: {utilization*100:.2f} %")
    return number_clocks, ideal_clock_count, utilization


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_training_history(history, model_name="Model"):
    """
    Plot training history including loss, accuracy, and learning rate.

    Parameters:
    history: Training history from model.fit()
    model_name: Name of the model for plot title
    save_path: Optional path to save the plot
    """
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
