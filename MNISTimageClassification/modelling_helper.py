# Copyright (C) Infineon Technologies AG 2025
#
# Use of this file is subject to the terms of use agreed between (i) you or the company in which ordinary course of
# business you are acting and (ii) Infineon Technologies AG or its licensees. If and as long as no such terms of use
# are agreed, use of this file is subject to following:
#
# This file is licensed under the terms of the Boost Software License. See the LICENSE file in the root of this repository
# for complete details.

import os
import sys

import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist

parent_dir = os.path.dirname(os.getcwd())

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import CentralScripts.helper_functions as cs


def load_mnist_data(val_size=0.1):
    """
    Load and split MNIST dataset into train, validation, and test sets.

    Parameters:
    val_size (float): Proportion of training data to use for validation

    Returns:
    dict: Dictionary containing x_train, y_train, x_val, y_val, x_test, y_test
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Split training data into train and validation
    np.random.seed(42)
    rand = np.random.uniform(size=x_train.shape[0])
    ind_val = rand < val_size
    ind_train = rand >= val_size

    x_train_split = x_train[ind_train]
    y_train_split = y_train[ind_train]
    x_val = x_train[ind_val]
    y_val = y_train[ind_val]

    return {
        "x_train": x_train_split,
        "y_train": y_train_split,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def normalize_data(x_train, x_val, x_test):
    """
    Normalize pixel values to 0-1 range.

    Parameters:
    x_train, x_val, x_test: Image arrays

    Returns:
    tuple: Normalized image arrays
    """
    x_train_norm = x_train.astype("float32") / 255.0
    x_val_norm = x_val.astype("float32") / 255.0
    x_test_norm = x_test.astype("float32") / 255.0

    return x_train_norm, x_val_norm, x_test_norm


def plot_sample_images(x_train, y_train, n_samples=10):
    """
    Plot a random sample of training images with their labels.

    Parameters:
    x_train: Training images
    y_train: Training labels
    n_samples: Number of samples to display
    save_path: Optional path to save the plot
    """
    np.random.seed(random.randint(0, 1024))
    indices = np.random.choice(range(len(x_train)), n_samples, replace=False)

    rows = 2
    cols = n_samples // rows
    _, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(x_train[indices[i]], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Label = {y_train[indices[i]]}", fontsize=14)

    plt.tight_layout()
    plt.show()


def get_input_output_shapes(x_train, y_train, model_type="mlp"):
    """
    Determine input and output shapes from training data.

    Parameters:
    x_train: Training images
    y_train: Training labels
    model_type: 'mlp' or 'cnn' to determine shape format

    Returns:
    tuple: (input_shape, output_shape)
    """
    if model_type == "mlp":
        # For MLP models, use original image shape without channel dimension
        if len(x_train.shape) == 3:  # (samples, height, width)
            height, width = x_train.shape[1], x_train.shape[2]
            input_shape = (height, width)  # No channel dimension for MLP
        elif len(x_train.shape) == 4:  # (samples, height, width, channels)
            input_shape = x_train.shape[1:]
        else:
            raise ValueError("Unexpected input data shape")

    elif model_type == "cnn":
        # For CNN models, add channel dimension if needed
        if len(x_train.shape) == 3:  # (samples, height, width)
            height, width = x_train.shape[1], x_train.shape[2]
            input_shape = (height, width, 1)  # Add channel dimension for CNN
        elif len(x_train.shape) == 4:  # (samples, height, width, channels)
            input_shape = x_train.shape[1:]
        else:
            raise ValueError("Unexpected input data shape")

    else:
        raise ValueError("model_type must be 'mlp' or 'cnn'")

    # Output shape for classification
    n_classes = len(np.unique(y_train))
    output_shape = (n_classes,)

    return input_shape, output_shape


def create_cnn_model(
    input_shape, output_shape, conv_layers=None, dense_units=None, p_drop=0.25
):
    """
    Create and compile a CNN model for MNIST classification.

    Parameters:
    input_shape: Shape of input images (height, width, channels)
    output_shape: Shape of output (number of classes)
    conv_layers: List of tuples (filters, kernel_size) for conv layers
    dense_units: List of units in dense layers
    p_drop: Dropout rate

    Returns:
    tf.keras.Model: Compiled CNN model
    """
    if conv_layers is None:
        conv_layers = [(32, 3), (64, 3)]
    if dense_units is None:
        dense_units = [128]

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Convolutional layers
    for i, (filters, kernel_size) in enumerate(conv_layers):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
            kernel_initializer="he_uniform",
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(p_drop)(x)

    # Flatten and dense layers
    x = tf.keras.layers.Flatten()(x)

    for units in dense_units:
        x = tf.keras.layers.Dense(
            units=units, activation="relu", kernel_initializer="he_uniform"
        )(x)
        x = tf.keras.layers.Dropout(p_drop)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(
        units=output_shape[0], activation="softmax", kernel_initializer="he_uniform"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CNN")
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model


def prepare_cnn_data(x_train, x_val, x_test):
    """
    Prepare data for CNN by adding channel dimension if needed.

    Parameters:
    x_train, x_val, x_test: Normalized image arrays

    Returns:
    tuple: Reshaped image arrays with channel dimension
    """
    # Add channel dimension if not present
    if len(x_train.shape) == 3:  # (samples, height, width)
        x_train = np.expand_dims(x_train, axis=-1)
        x_val = np.expand_dims(x_val, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    return x_train, x_val, x_test


def create_mlp_model(input_shape, output_shape, units, p_drop=0.05):
    """
    Create and compile an MLP model for MNIST classification.

    Parameters:
    input_shape: Shape of input images
    output_shape: Shape of output (number of classes)
    units: List of units in each hidden layer
    p_drop: Dropout rate

    Returns:
    tf.keras.Model: Compiled MLP model
    """
    dense_layer = partial(
        tf.keras.layers.Dense,
        activation="relu",
        kernel_initializer="he_uniform",
        bias_initializer=tf.keras.initializers.Constant(0),
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)

    # Add hidden layers
    for unit_count in units:
        x = dense_layer(unit_count)(x)
        x = tf.keras.layers.Dropout(p_drop)(x)

    # Output layer
    outputs = dense_layer(
        units=output_shape[0],
        activation="softmax",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MLP")
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Nadam(),
        metrics=["accuracy"],
    )

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


def train_model(model, x_train, y_train, x_val, y_val, batch_size=64, epochs=100):
    """
    Train the model with the provided data.

    Parameters:
    model: Compiled Keras model
    x_train, y_train: Training data and labels
    x_val, y_val: Validation data and labels
    batch_size: Batch size for training
    epochs: Maximum number of epochs

    Returns:
    tf.keras.callbacks.History: Training history
    """
    callbacks = get_callbacks()

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    return history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate model on test data.

    Parameters:
    model: Trained Keras model
    x_test, y_test: Test data and labels

    Returns:
    tuple: (test_loss, test_accuracy)
    """
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, test_accuracy


def plot_confusion_matrix(model, x_test, y_test, class_names=None, save_path=None):
    """
    Plot confusion matrix as a heatmap with numbers displayed in each cell.

    Parameters:
    model: Trained Keras model
    x_test: Test data
    y_test: True test labels
    class_names: Optional list of class names for labels
    save_path: Optional path to save the plot
    """
    # Make predictions
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # Calculate adaptive vmax based on maximum off-diagonal entry
    # Create a mask for off-diagonal elements
    off_diagonal_mask = ~np.eye(cm.shape[0], dtype=bool)
    max_off_diagonal = (
        cm[off_diagonal_mask].max() if cm[off_diagonal_mask].size > 0 else cm.max()
    )

    # Set up class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_test)))]

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Create heatmap with numbers
    sns.heatmap(
        cm,
        annot=True,  # Show numbers in cells
        fmt="d",  # Format as integers
        cmap=[cs.COLORS["OCEAN"], cs.COLORS["OCEAN_1"], cs.COLORS["OCEAN_2"]],
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,  # Show colorbar
        vmin=0,
        vmax=max_off_diagonal,
        square=True,
    )

    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def generate_model_name(model, test_accuracy, test_loss):
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

    accuracy_str = str(round(test_accuracy * 100, 2)).replace(".", "-")
    loss_str = str(round(test_loss, 2)).replace(".", "-")

    name = f"{model_type}_model_layers_{trainable_layers}_param_{total_params}_accuracy_{accuracy_str}_loss_{loss_str}"
    return name
