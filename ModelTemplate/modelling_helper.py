# Copyright (C) Infineon Technologies AG 2025
#
# Use of this file is subject to the terms of use agreed between (i) you or the company in which ordinary course of
# business you are acting and (ii) Infineon Technologies AG or its licensees. If and as long as no such terms of use
# are agreed, use of this file is subject to following:
#
# This file is licensed under the terms of the Boost Software License. See the LICENSE file in the root of this repository
# for complete details.

import torch.nn as nn
import torch
from keras.models import Model
from keras.layers import Dense


class TorchNet(nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # 10 input features and 64 hidden neurons
        self.fc3 = nn.Linear(64, 2)  # 2 output features

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x


class KerasModel(Model):
    def __init__(self):
        super(KerasModel, self).__init__()
        self.dense1 = Dense(64, activation="relu", input_shape=(10, 1))
        self.dense2 = Dense(2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x


def get_model(origin):

    if origin == "torch":
        model = TorchNet()
        model.eval()
        return model

    elif origin == "tf":
        model = KerasModel()
        return model
