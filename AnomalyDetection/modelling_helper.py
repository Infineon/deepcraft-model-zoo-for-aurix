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

import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data_cols = [
    "aimp",
    "amud",
    "arnd",
    "asin1",
    "asin2",
    "adbr",
    "adfl",
    "bed1",
    "bed2",
    "bfo1",
    "bfo2",
    "bso1",
    "bso2",
    "bso3",
    "ced1",
    "cfo1",
    "cso1",
]


def load_data(data_path="data/data.csv", meta_path="data/metadata.csv"):
    try:
        print(f"Loading file: {data_path}")
        full_data = pd.read_csv(data_path)
        full_data["Train_Test"] = "Test"
        ind = np.zeros(full_data.shape[0], dtype=bool)
        ind[0:1000000] = True
        full_data.loc[ind, "Train_Test"] = "Train"
        train = full_data.loc[full_data["Train_Test"] == "Train", :].copy()
        test = full_data.loc[full_data["Train_Test"] == "Test", :].copy()
    except Exception as e:
        print(f"Could not load {data_path}: {e}")
        train = None
        test = None

    try:
        print(f"Loading file: {meta_path}")
        meta_data = pd.read_csv(meta_path)
    except Exception as e:
        print(f"Could not load {meta_path}: {e}")
        meta_data = None

    return train, test, meta_data


def normalize_data(df, cols=data_cols, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[cols])
    else:
        if not isinstance(scaler, StandardScaler):
            raise TypeError("scaler must be a StandardScaler instance from sklearn.")
        df_scaled = scaler.transform(df[cols])

    df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df.index)
    return df_scaled, scaler


def sample_sequences(df, sequence_length, n_samples, cols=data_cols, random_state=42):
    data = df[cols].to_numpy()
    num_samples = data.shape[0]
    max_start = num_samples - sequence_length
    rng = np.random.default_rng(random_state)
    start_indices = rng.choice(max_start + 1, size=n_samples, replace=False)
    sequences = np.stack(
        [data[i : i + sequence_length, :] for i in start_indices], axis=0
    )  # shape: (n_samples, sequence_length, len(data_cols))
    return sequences


def sequential_sequences(df, sequence_length, cols=data_cols):
    """
    Generate sequential sequences from a dataframe with stride of 1.

    Parameters:
    df: DataFrame with time series data
    sequence_length: Length of each sequence
    cols: List of columns to include (default: mh.data_cols)

    Returns:
    numpy array of shape (n_sequences, sequence_length, n_features)
    """
    data = df[cols].to_numpy()
    n_samples, n_features = data.shape

    # Calculate number of sequences we can create
    n_sequences = n_samples - sequence_length + 1

    # Create sequences with stride 1
    sequences = np.array([data[i : i + sequence_length] for i in range(n_sequences)])

    return sequences


def plot_time_steps(df, n_steps, cols=data_cols):
    _, axes = plt.subplots(len(cols), 1, figsize=(8, 10), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        axes[i].plot(df[col].iloc[:n_steps])
        axes[i].set_ylabel(col)
    axes[-1].set_xlabel("Time step @ 1 Hz")
    plt.tight_layout()
    plt.show()


def mlp_autoencoder(input_dim, bottleneck, layers=3, p_drop=0.2):
    """
    Build an MLP-based autoencoder with symmetric encoder-decoder architecture.

    Parameters:
    input_dim: Number of input features
    bottleneck: Size of the bottleneck layer (compressed representation)
    layers: Number of layers in encoder (decoder will be symmetric)
    p_drop: Dropout probability

    Returns:
    Compiled Keras model
    """
    regularized_dense = partial(
        tf.keras.layers.Dense,
        activation="relu",
        kernel_initializer="he_uniform",
        bias_initializer=tf.keras.initializers.Constant(0),
    )

    inputs = tf.keras.Input(shape=(input_dim,), name="input_layer")

    # Calculate layer sizes for gradual compression
    if layers == 1:
        # Special case: direct compression to bottleneck
        encoder_sizes = [bottleneck]
        decoder_sizes = [input_dim]
    else:
        # Calculate intermediate layer sizes using geometric progression
        # This ensures smooth compression from input_dim to bottleneck
        ratio = (bottleneck / input_dim) ** (1 / layers)
        encoder_sizes = []
        for i in range(layers):
            size = int(input_dim * (ratio ** (i + 1)))
            # Ensure we don't go below bottleneck size
            size = max(size, bottleneck)
            encoder_sizes.append(size)

        # Decoder sizes are symmetric (reverse of encoder, excluding bottleneck)
        decoder_sizes = encoder_sizes[:-1][::-1] + [input_dim]

    # Encoder: gradually compress input to bottleneck
    x = inputs
    for i, units in enumerate(encoder_sizes):
        x = regularized_dense(units=units, name=f"encoder_layer_{i+1}")(x)
        x = tf.keras.layers.Dropout(p_drop, name=f"encoder_dropout_{i+1}")(x)

    # Bottleneck layer
    x = regularized_dense(units=bottleneck, name="bottleneck")(x)
    x = tf.keras.layers.Dropout(p_drop, name="bottleneck_dropout")(x)

    # Decoder: gradually expand back to input size
    for i, units in enumerate(decoder_sizes):
        x = regularized_dense(units=units, name=f"decoder_layer_{i+1}")(x)
        x = tf.keras.layers.Dropout(p_drop, name=f"decoder_dropout_{i+1}")(x)

    # Output layer (no activation for reconstruction)
    outputs = tf.keras.layers.Dense(
        units=input_dim,
        activation="linear",
        kernel_initializer="glorot_uniform",
        name="reconstruction_output",
    )(x)

    model = tf.keras.models.Model(
        inputs=inputs, outputs=outputs, name="MLP_Autoencoder"
    )

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
    )

    print("=== MLP Autoencoder Architecture ===")
    print(f"Input dimension: {input_dim}")
    print(f"Bottleneck size: {bottleneck}")
    print(f"Compression ratio: {bottleneck/input_dim:.3f}")
    print(f"Encoder layers: {layers}")
    if layers > 1:
        print(f"Encoder sizes: {encoder_sizes}")
        print(f"Decoder sizes: {decoder_sizes}")
    print("\n" + "=" * 50)
    print(model.summary())

    return model


def get_callbacks(min_delta=0.0001):
    """
    Get training callbacks for early stopping and learning rate reduction.

    Parameters:
    min_delta: Minimum change in monitored quantity to qualify as improvement

    Returns:
    list: List of Keras callbacks
    """
    callbacks = []

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, min_delta=min_delta, verbose=1
    )
    callbacks.append(early_stopping)

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        verbose=1,
        mode="min",
        cooldown=0,
        min_delta=min_delta,
        min_lr=1e-8,
    )
    callbacks.append(lr_schedule)

    return callbacks


def generate_model_name(model, test_loss):
    """
    Generate a descriptive name for the model based on its parameters and performance.

    Parameters:
    model: Trained Keras model
    test_accuracy: Test accuracy
    test_loss: Test loss

    Returns:
    str: Generated model name
    """
    trainable_layers = sum(1 for layer in model.layers if layer.count_params() > 0)
    total_params = model.count_params()
    model_type = model.name.lower()  # 'mlp' or 'cnn'

    loss_str = str(round(test_loss, 2)).replace(".", "-")

    name = f"{model_type}_model_layers_{trainable_layers}_param_{total_params}_loss_{loss_str}"
    return name


def plot_reconstruction_error(recon_error, threshold):
    _, ax = plt.subplots(1, 1, figsize=(8, 5))

    sns.histplot(
        data=recon_error,
        x="MSE",
        hue="Train_test",
        bins=np.logspace(0, 3, 50),
        stat="density",
        multiple="dodge",
        common_norm=False,
        ax=ax,
        kde=False,
    )
    ax.axvline(threshold, color="red", linestyle="--", label="95% threshold")
    ax.set_xscale("log")
    plt.tight_layout()


def plot_anomaly_detection(anomaly_data, threshold):
    _, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Create a mask for anomalous time points based on MSE threshold
    anomaly_mask = anomaly_data["err"]["MSE"] > threshold

    # Plot the base line in normal color
    sns.lineplot(
        data=anomaly_data["rows_scaled"],
        x=anomaly_data["rows"].index,
        y=anomaly_data["root_cause"],
        ax=ax[0],
        color="blue",
        alpha=0.7,
    )

    # Overlay red points where MSE exceeds threshold
    # Align timestamps: err timestamps start from sequence_length-1 index
    anomaly_timestamps = anomaly_data["err"].loc[anomaly_mask, "timestamp"]
    anomaly_values = anomaly_data["rows_scaled"].loc[
        anomaly_timestamps, anomaly_data["root_cause"]
    ]

    if len(anomaly_values) > 0:
        ax[0].scatter(
            anomaly_timestamps,
            anomaly_values,
            color="red",
            s=10,
            alpha=0.8,
            label="Anomaly detected",
        )
        ax[0].legend()

    ax[0].set_title(f'Root cause channel: {anomaly_data["root_cause"]}')

    sns.lineplot(data=anomaly_data["err"], x="timestamp", y="MSE", ax=ax[1])
    ax[1].axhline(threshold, color="red", linestyle="--", label="Anomaly threshold")
    ax[1].set_yscale("log")
    ax[1].legend(loc="best")
    plt.tight_layout()
