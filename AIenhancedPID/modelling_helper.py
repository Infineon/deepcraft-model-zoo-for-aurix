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


from tensorflow.keras.models import Model
from typing import TypeVar, Tuple, Any
import numpy as np
import numpy.typing as npt
from scipy.io import loadmat, savemat
import logging as log
import scipy
from fmpy import *
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from fmpy.util import plot_result, download_test_file
import shutil
import os
import sys
from matplotlib import pyplot as plt
import plotly.subplots as sp
import plotly.graph_objs as go

from keras.models import Model
from keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from CentralScripts.helper_functions import COLORS


T = TypeVar("T", bound=np.float32)


def train_model(model, data_x, data_y):
    """
    Function training the model
    """
    callback = EarlyStopping()
    history = model.fit(
        x=data_x,
        y=data_y,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[callback],
    )

    return model, history


def preprocess_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min)

    return data


def load_data():
    try:
        data_x = np.load("./data/saved_data.npy")
        log.basicConfig(level=log.INFO)
        log.info("DATA LOADED SUCCESSFULLY")

    except (FileNotFoundError, IOError):
        log.basicConfig(level=log.ERROR)
        log.error("DATA FILE DOES NOT EXIST")
        sys.exit()

    try:
        data_y = np.load("./data/labels.npy")
        # data_y = loadmat("./data/labels.mat")
        # data_y = data_y["gains"]
        log.basicConfig(level=log.INFO)
        log.info("LABELS LOADED SUCCESSFULLY")
    except (FileNotFoundError, IOError):
        log.basicConfig(level=log.ERROR)
        log.error("LABEL FILE DOES NOT EXIST")
        sys.exit()

    data_x = preprocess_data(data_x)
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.float32)

    return data_x[:110, :], data_y[:110, :], data_x[110:, :], data_y[110:, :]


# Reshape/flatten input data


def load_data1(input_data_paths):
    # Load input data
    p_coeff = scipy.io.loadmat(input_data_paths["p_coeff"])
    i_coeff = scipy.io.loadmat(input_data_paths["i_coeff"])
    d_coeff = scipy.io.loadmat(input_data_paths["d_coeff"])
    dummy_cte = scipy.io.loadmat(input_data_paths["dummy_cte"])
    data_inference = scipy.io.loadmat(input_data_paths["data_inference"])

    return {
        "p_": p_coeff,
        "i_": i_coeff,
        "d_": d_coeff,
        "cte_": dummy_cte,
        "data_i_": data_inference,
    }


def preprocess_data1(input_data):
    # Reshape/flatten input data
    input_data["p_"] = input_data["p_"]["p"].reshape(
        -1,
    )

    input_data["i_"] = input_data["i_"]["i"].reshape(
        -1,
    )

    input_data["d_"] = input_data["d_"]["d"].reshape(
        -1,
    )

    input_data["cte_"] = input_data["cte_"]["dummy_cte"].reshape(
        -1,
    )

    data_inference = np.transpose(input_data["data_i_"]["data_inference"], axes=(1, 0))
    data_inference = data_inference[:, 1:]
    input_data["data_i_"] = data_inference

    return input_data


batch_prediction = True
input_data_paths = {
    "p_coeff": "./data/p_.mat",
    "i_coeff": "./data/i_.mat",
    "d_coeff": "./data/d_.mat",
    "dummy_cte": "./data/dummy_cte_.mat",
    "data_inference": "./data/data_inference_.mat",
}


def simulate_custom_input(
    input_paths=None, model=None, show_plot=True, batch_prediction=False, ai_mode=True
):
    """
    Function performing a simulation on fmu model

    Function loads an FMU model, loads input data, load neural network model
    In the for loop NN predicts PID coefficient for each simulation step.
    In each simulation step the simulation result is saved in the list.

    Inputs:
        show_plot: bool
            if true then result of the simulation is plotted else not plotted
        batch_prediction: bool
            if batch_prediction is True then NN predict in batches and all predicted samples are an input to the FMU model
            if batch_prediction is False then simulation is performed in steps
    """

    # define the model name and simulation parameters
    fmu_filename = "exported_model/nn_tuner3a.fmu"
    start_time = 0.0
    threshold = 2.0
    stop_time = 30.0
    step_size = 1e-2

    # read the model description (info about model)
    model_description = read_model_description(fmu_filename)

    # collect the value references
    vrs = {}
    for variable in model_description.modelVariables:
        vrs[variable.name] = variable.valueReference

    # get the value references for the variables we want to get/set
    p_coeff_id = vrs["p_coeff"]  # p coefficient
    i_coeff_id = vrs["i_coeff"]
    d_coeff_id = vrs["d_coeff"]
    cte_id = vrs["cte"]
    steering_id = vrs["steering"]

    # extract the FMU
    unzipdir = extract(fmu_filename)

    fmu = FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzipdir,
        modelIdentifier=model_description.coSimulation.modelIdentifier,
        instanceName="instance1",
    )

    # initialize
    fmu.instantiate()
    fmu.setupExperiment(startTime=start_time)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    time = start_time

    rows = []  # list to record the results

    data_inference = load_data1(input_paths)
    data_inference = preprocess_data1(data_inference)

    if batch_prediction:
        data_inference["data_i_"] = model.predict(data_inference["data_i_"])

    # simulation loop
    for cte, data_in in zip(data_inference["cte_"], data_inference["data_i_"]):

        # NOTE: the FMU.get*() and FMU.set*() functions take lists of
        # value references as arguments and return lists of values

        if batch_prediction:
            predicted = data_in
        else:
            data_in = data_in.reshape(1, 10)

            predicted = model.predict([data_in])
            predicted = predicted[0, :]

        if ai_mode:
            p = predicted[0]
            i = predicted[1]
            d = predicted[2]
        else:
            p = 9
            i = 10
            d = 14

        # set the input
        fmu.setReal([p_coeff_id, i_coeff_id, d_coeff_id, cte_id], [p, i, d, cte])

        # perform one step
        fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

        # advance the time
        time += step_size

        # get the values for 'inputs' and 'outputs[4]'
        p_in, i_in, d_in, cte_in, steering_out = fmu.getReal(
            [p_coeff_id, i_coeff_id, d_coeff_id, cte_id, steering_id]
        )

        # append the results
        rows.append((time, p_in, i_in, d_in, cte_in, steering_out))

        if time > stop_time:
            break

    fmu.terminate()
    fmu.freeInstance()

    # clean up
    shutil.rmtree(unzipdir, ignore_errors=True)

    # convert the results to a structured NumPy array
    result = np.array(
        rows,
        dtype=np.dtype(
            [
                ("time", np.float64),
                ("p_in", np.float64),
                ("i_in", np.float64),
                ("d_in", np.float64),
                ("cte_in", np.float64),
                ("steering_out", np.float64),
            ]
        ),
    )

    # plot the results
    if show_plot:
        plot_result(result)

    return result


def plot_stacked_results(res_ai, res_non_ai):
    """
    time, p, i, d, cte, steering_angle
    """
    res_ai_ = res_ai.view(np.float64).reshape(res_ai.shape + (-1,))
    res_non_ai_ = res_non_ai.view(np.float64).reshape(res_non_ai.shape + (-1,))
    fig = sp.make_subplots(rows=5, cols=1, shared_xaxes=True)

    fig.add_trace(
        go.Scatter(
            x=res_ai_[:, 0],
            y=res_ai_[:, 1],
            name="ai-enhanced controller",
            line=dict(color=COLORS["OCEAN"]),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=res_non_ai_[:, 0],
            y=res_non_ai_[:, 1],
            name="conventional controller",
            line=dict(color=COLORS["LAWN_MAIN"]),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="P", row=1, col=1)
    # plot i
    fig.add_trace(
        go.Scatter(
            x=res_ai_[:, 0],
            y=res_ai_[:, 2],
            name="ai-enhanced controller",
            line=dict(color=COLORS["OCEAN"]),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=res_non_ai_[:, 0],
            y=res_non_ai_[:, 2],
            name="conventional controller",
            line=dict(color=COLORS["LAWN_MAIN"]),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="I", row=2, col=1)
    # plot d
    fig.add_trace(
        go.Scatter(
            x=res_ai_[:, 0],
            y=res_ai_[:, 3],
            name="ai-enhanced controller",
            line=dict(color=COLORS["OCEAN"]),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=res_non_ai_[:, 0],
            y=res_non_ai_[:, 3],
            name="conventional controller",
            line=dict(color=COLORS["LAWN_MAIN"]),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="D", row=3, col=1)
    # plot cte
    fig.add_trace(
        go.Scatter(
            x=res_ai_[:, 0],
            y=res_ai_[:, 4],
            name="cte",
            line=dict(color=COLORS["SUN_MAIN"]),
        ),
        row=4,
        col=1,
    )
    fig.update_yaxes(title_text="CTE [m]", row=4, col=1)
    # steering angle
    fig.add_trace(
        go.Scatter(
            x=res_ai_[:, 0],
            y=res_ai_[:, 5],
            name="ai-enhanced controller",
            line=dict(color=COLORS["OCEAN"]),
            showlegend=False,
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=res_non_ai_[:, 0],
            y=res_non_ai_[:, 5],
            name="conventional controller",
            line=dict(color=COLORS["LAWN_MAIN"]),
            showlegend=False,
        ),
        row=5,
        col=1,
    )
    fig.update_yaxes(title_text="Steering Angle [degree]", row=5, col=1)
    fig.update_layout(height=1600)
    fig.show()


def plot_stacked_results_(res_ai, res_non_ai):
    """
    time, p, i, d, cte, steering_angle
    """
    res_ai_ = res_ai.view(np.float64).reshape(res_ai.shape + (-1,))
    res_non_ai_ = res_non_ai.view(np.float64).reshape(res_non_ai.shape + (-1,))
    # fig = sp.make_subplots(rows=10, cols=1)
    # fig = go.Figure()
    # plot p
    # fig1 = go.Figure()
    # fig1 = go.Figure(data=go.Scatter(x=[1,2,3], y=[4,5,6]))
    # fig1 = go.Figure(data=go.Bar(x=[1,2,3], y=[6,5,4]))

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(x=res_ai_[:, 0], y=res_ai_[:, 1], name="p - ai-enhanced controller")
    )
    fig1.add_trace(
        go.Scatter(
            x=res_non_ai_[:, 0], y=res_non_ai_[:, 1], name="p - conventional controller"
        )
    )
    # plot i
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=res_ai_[:, 0], y=res_ai_[:, 2], name="i - ai-enhanced controller")
    )
    fig2.add_trace(
        go.Scatter(
            x=res_non_ai_[:, 0], y=res_non_ai_[:, 2], name="i - conventional controller"
        )
    )
    # plot d
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(x=res_ai_[:, 0], y=res_ai_[:, 3], name="d - ai-enhanced controller")
    )
    fig3.add_trace(
        go.Scatter(
            x=res_non_ai_[:, 0], y=res_non_ai_[:, 3], name="d - conventional controller"
        )
    )
    # plot cte
    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(x=res_ai_[:, 0], y=res_ai_[:, 4], name="cross-track error")
    )
    # fig4.add_trace(go.Scatter(x=res_non_ai_[:,0], y=res_non_ai_[:,4], name="cte"))
    # steering angle
    fig5 = go.Figure()
    fig5.add_trace(
        go.Scatter(
            x=res_ai_[:, 0],
            y=res_ai_[:, 5],
            name="steering angle - ai-enhanced controller",
        )
    )
    fig5.add_trace(
        go.Scatter(
            x=res_non_ai_[:, 0],
            y=res_non_ai_[:, 5],
            name="steering angle - conventional controller",
        )
    )
    # fig.show()
    fig1.update_layout(
        width=800,
        xaxis=dict(title=dict(text="Time [s]")),
        yaxis=dict(title=dict(text="P")),
    )
    fig1.show()
    fig2.update_layout(
        width=800,
        xaxis=dict(title=dict(text="Time [s]")),
        yaxis=dict(title=dict(text="I")),
    )
    fig2.show()
    fig3.update_layout(
        width=800,
        xaxis=dict(title=dict(text="Time [s]")),
        yaxis=dict(title=dict(text="D")),
    )
    fig3.show()
    fig4.update_layout(
        width=800,
        xaxis=dict(title=dict(text="Time [s]")),
        yaxis=dict(title=dict(text="CTE [m]")),
        showlegend=True,
    )
    fig4.show()
    fig5.update_layout(
        width=1000,
        xaxis=dict(title=dict(text="Time [s]")),
        yaxis=dict(title=dict(text="Steering Angle [rad]")),
    )
    fig5.show()


class KerasModel(Model):
    def __init__(self):
        super(KerasModel, self).__init__()
        self.dense1 = Dense(32, activation="relu", input_shape=(10,))
        self.dropout1 = Dropout(0.25)
        self.dense2 = Dense(16, activation="relu")
        self.dropout2 = Dropout(0.25)
        self.dense3 = Dense(3)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        return x


def get_model(origin):

    if origin == "torch":
        model = TorchNet()
        model.eval()
        return model

    elif origin == "tf":
        model = KerasModel()
        model.compile(loss=MeanSquaredError(), optimizer=Adam())
        return model
