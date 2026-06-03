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


import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import onnx

from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import (
    quant_pre_process,
)


parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import _CentralScripts.helper_functions as cs
import _CentralScripts.mobilenet_helper as mo

plt.rcParams.update({"font.size": 13})


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightRaindropCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 2
            DepthwiseSeparableConv(16, 32),
            nn.MaxPool2d(2),
            # block 3
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2),
            # block 4
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2),
            # block 5
            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def get_model(num_classes):
    model = LightRaindropCNN(num_classes)
    return model


def train_model(
    model,
    dataloader,
    dataset_sizes,
    criterion,
    optimizer,
    is_train_entire_model=True,
    num_epochs=10,
):

    if is_train_entire_model:
        model.train()
    else:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier")
    max_accuracy = 0.0
    device = cs.get_device()
    model = model.to(device)

    for epoch in range(num_epochs):

        print(f"Epoch {epoch + 1 }/{num_epochs}")
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            optimizer.step()

        epoch_loss = running_loss / dataset_sizes
        epoch_accuracy = running_corrects.double() / dataset_sizes

        if epoch_accuracy > max_accuracy:
            max_accuracy = epoch_accuracy

        print(f"Loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.4f}")

    print(f"Maximim accuracy: {max_accuracy:.4f}")

    model.eval()


def onnx_preprocessing(input_file, output_file):
    quant_pre_process(input_file, output_file)


def get_nodes_exclude(
    model_path_preprocessed, indices_exclude, is_exclude_activation=False
):

    model_preprocessed = onnx.load(model_path_preprocessed)
    nodes = model_preprocessed.graph.node

    nodes_exclude = []

    for index in indices_exclude:
        nodes_exclude.append(nodes[index].name)

    if is_exclude_activation:
        for node in nodes:
            if "Hard" in node.name:
                nodes_exclude.append(node.name)
            elif "Relu" in node.name:
                nodes_exclude.append(node.name)

    return nodes_exclude


class NumpyDataReader(CalibrationDataReader):
    def __init__(self, input_name, data_list):
        self.input_name = input_name
        self.data_list = data_list
        self._iter = iter(self.data_list)

    def get_next(self):
        try:
            return {self.input_name: next(self._iter)}
        except StopIteration:
            return None

    def rewind(self):
        self._iter = iter(self.data_list)


def get_calib_data(folder_path, resolution=224, num_samples=100):

    calib_samples = []
    labels = []

    dataloader, _, _ = mo.get_dataloader(
        folder_path, batch_size=1, mode="train", resolution=resolution
    )

    for data, label in dataloader:
        calib_samples.append(data.numpy())
        labels.append(label.item())
        if len(calib_samples) >= num_samples:
            break

    return calib_samples, labels


def get_input_name(model_path_preprocessed):
    model = onnx.load(model_path_preprocessed)
    input_name = model.graph.input[0].name
    return input_name


def create_folders_quantization(model_name):

    list_options = ["preprocessed", "quantized"]
    file_list = []

    for option in list_options:
        path, file = cs.get_output_paths(f"{model_name}_{option}")
        os.makedirs(path, exist_ok=True)
        file_list.append(file)

    return file_list


def predict_class(model, input, threshold=0.5):
    output = cs.get_onnx_predictions(model, input)
    return np.argmax(np.array(output) > threshold).astype("int32"), output


def validate_quantization(qdq_model_path, onnx_model_file, loader):

    qdq_model = cs.get_onnx_tensor(qdq_model_path)
    original_model = cs.get_onnx_tensor(onnx_model_file)

    num_samples = 0
    correctly_predicted_q = 0
    correctly_predicted_o = 0
    num_same_prediciton = 0

    for input_array, label in loader:

        input_array = input_array.numpy()
        label = label.item()

        predicted_class_q, _ = predict_class(qdq_model, input_array)
        predicted_class_o, _ = predict_class(original_model, input_array)

        if predicted_class_q == label:
            correctly_predicted_q += 1

        if predicted_class_o == label:
            correctly_predicted_o += 1

        if predicted_class_q == predicted_class_o:
            num_same_prediciton += 1

        num_samples += 1

        if num_samples == 1000:
            break

    print(
        f"Quantized: percentage of correctly predicted {correctly_predicted_q/num_samples*100:.2f} %: {correctly_predicted_q} out of {num_samples} samples."
    )
    print(
        f"Original: percentage of correclty predicted {correctly_predicted_o/num_samples*100:.2f} %: {correctly_predicted_o} out of {num_samples} samples."
    )
    print(
        f"Number of same predictions: {num_same_prediciton} out of {num_samples} samples."
    )
