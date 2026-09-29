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


import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torchsummary import summary

import os
import matplotlib.pyplot as plt
import sys
import onnx
import warnings

parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import _CentralScripts.helper_functions as cs
import _CentralScripts.mobilenet_helper as mo

plt.rcParams.update({"font.size": 13})

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="torch.ao.quantization is deprecated"
)
warnings.filterwarnings("ignore", message="Please use quant_min and quant_max")


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
            nn.Conv2d(3, 10, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 2
            DepthwiseSeparableConv(10, 20),
            nn.MaxPool2d(2),
            # block 3
            DepthwiseSeparableConv(20, 40),
            nn.MaxPool2d(2),
            # block 4
            DepthwiseSeparableConv(40, 80),
            nn.MaxPool2d(2),
            # block 5
            DepthwiseSeparableConv(80, 160),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 128),
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


def train_model_qat(
    model,
    dataloader,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=1,
    observer_epochs=3,
    freeze_bn_epoch=5,
    device="cpu",
):

    model.train()
    model.to(device)

    for epoch in range(num_epochs):

        if epoch == 0:
            model.apply(torch.ao.quantization.disable_fake_quant)
            model.apply(torch.ao.quantization.enable_observer)
            print("Phase 1: Observer enabled (fake quantization disabled)")

        if epoch == observer_epochs:
            model.apply(torch.ao.quantization.enable_fake_quant)
            model.apply(torch.ao.quantization.disable_observer)
            print("Phase 2: Enabling Fake quantization, observers disabled")

        if epoch == freeze_bn_epoch:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            print("Phase 3: BatchNorm stats frozen")

        print(f"Epoch {epoch+1:>2}/{num_epochs}")

        running_loss = 0.0
        running_correct = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_correct += (preds == labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_correct / dataset_sizes

        print(
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_acc:.4f} | "
            f"Learning rate: {scheduler.get_last_lr()[0]:.2e}"
        )

    print("Training complete.")
    return model


def create_model(input_size):

    random_input = torch.randn(input_size)

    model = LightRaindropCNN(num_classes=2)
    model.eval()

    summary(model, input_size, device="cpu")

    qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm")
    model_prepared = prepare_qat_fx(
        model, qconfig_mapping, example_inputs=(random_input,)
    )
    return model_prepared


def train_quantize_model(
    model,
    dataloader,
    dataset_sizes,
    num_epochs=10,
    observer_epochs=3,
    freeze_bn_epoch=6,
):
    device = "cpu"
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    model = train_model_qat(
        model,
        dataloader,
        dataset_sizes,
        criterion,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        observer_epochs=observer_epochs,
        freeze_bn_epoch=freeze_bn_epoch,
        device=device,
    )

    model.eval()
    model_quantized = convert_fx(model)
    model_quantized.eval()

    return model_quantized


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

    options = ["preprocessed", "static_ptq_int8", "dynamic_ptq"]
    file_list = []

    for option in options:
        path, file = cs.get_output_paths(f"{model_name}_{option}")
        os.makedirs(path, exist_ok=True)
        file_list.append(file)

    return file_list, options


def prepare_code_example():
    in_files = [
        "out/cnn_weather/test_cnn_weather/model.onnx",
        "out/cnn_weather_static_ptq_int8/test_cnn_weather_static_ptq_int8/model.onnx",
        "out/cnn_weather_dynamic_ptq/test_cnn_weather_dynamic_ptq/model.onnx",
    ]

    out_files = [
        "out/model_zoo2code_example/model_fp32.onnx",
        "out/model_zoo2code_example/model_int8_static.onnx",
        "out/model_zoo2code_example/model_int8_dynamic.onnx",
    ]

    for in_file, out_file in zip(in_files, out_files):
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
        shutil.copy2(in_file, out_file)
