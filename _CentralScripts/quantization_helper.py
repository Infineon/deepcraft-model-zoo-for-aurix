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

import contextlib
import io
import numpy as np
import _CentralScripts.helper_functions as cs
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_dynamic,
    quantize_static,
    quant_pre_process,
)
import onnxslim
import onnx
from onnx import version_converter


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


def static_quantization(
    model_path_preprocessed,
    model_path_ptq,
    calib_samples,
    input_name,
    type,
    input_target,
):

    calibration_data = NumpyDataReader(input_name, calib_samples)

    quantize_static(
        model_input=model_path_preprocessed,
        model_output=model_path_ptq,
        calibration_data_reader=calibration_data,
        per_channel=False,  # per-channel shall be False
        activation_type=type,  # typical for activations
        weight_type=type,  # typical for weights
    )

    save_quantized_model(model_path_ptq, input_target)
    print(f"Saved QDQ model to: {model_path_ptq}")


def dynamic_quantization(
    model_path_preprocessed, model_path_dynamic, type, input_target
):

    quantize_dynamic(
        model_path_preprocessed,
        model_path_dynamic,
        weight_type=type,
        nodes_to_quantize=["Conv", "Gemm", "MatMul"],
    )

    save_quantized_model(model_path_dynamic, input_target)
    print(f"Saved QDQ model to: {model_path_dynamic}")


def save_quantized_model(model_path_ptq, input_target):

    with contextlib.redirect_stdout(io.StringIO()):
        onnxslim.slim(model_path_ptq, model_path_ptq)

    quantized_model_onnx = cs.get_onnx_tensor(model_path_ptq)
    _, output_target = cs.predict_class(
        quantized_model_onnx, np.expand_dims(input_target, axis=0)
    )

    cs.save_data(model_path_ptq.replace("model.onnx", ""), input_target, is_input=True)
    cs.save_data(
        model_path_ptq.replace("model.onnx", ""), output_target, is_input=False
    )


def onnx_preprocessing(input_file, output_file):
    quant_pre_process(input_file, output_file)


def update_optset(model_path, opset=21):
    model = onnx.load(model_path)

    upgraded_model = version_converter.convert_version(model, opset)

    # Save the upgraded model
    onnx.save(upgraded_model, model_path)


def validate_quantization_options(model_paths, loader):

    models = []
    for model_path in model_paths:
        models.append(cs.get_onnx_tensor(model_path))

    num_samples = 0

    correctly_predicted = np.zeros(len(models))

    for input_array, label in loader:

        input_array = input_array.numpy()
        label = label.item()

        for i, model in enumerate(models):
            predicted_class, _ = cs.predict_class(model, input_array)

            if predicted_class == label:
                correctly_predicted[i] += 1

        num_samples += 1

        if num_samples == 1000:
            break

    accuracies = {}
    for i, model_path in enumerate(model_paths):
        accuracy = correctly_predicted[i] / num_samples * 100
        accuracies[model_path] = accuracy
        print(
            f"Model {model_path}, \n percentage of correctly predicted: {accuracy:.2f} % \n {correctly_predicted[i]} out of {num_samples} samples. \n"
        )

    return accuracies
