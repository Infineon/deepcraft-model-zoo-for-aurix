# Copyright (C) Infineon Technologies AG 2025
#
# Use of this file is subject to the terms of use agreed between (i) you or the company in which ordinary course of
# business you are acting and (ii) Infineon Technologies AG or its licensees. If and as long as no such terms of use
# are agreed, use of this file is subject to following:
#
# This file is licensed under the terms of the Boost Software License. See the LICENSE file in the root of this repository
# for complete details.

import os
from typing import Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from CentralScripts.helper_functions import COLORS

column_names = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 24)]
)


def load_dataframe(file_path):
    df = pd.read_csv(file_path, sep=" ", header=None, names=column_names)

    # Remove completely empty columns (all NaN)
    # First identify columns that are completely empty
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        print(f"Removing completely empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)

    # Remove any unnamed columns that might have been created by extra spaces
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed_cols:
        print(f"Removing unnamed columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    return df


def clean_data(df):
    removed_columns = []

    # Remove empty columns
    empty_cols = df.columns[df.isna().all()]
    removed_columns.extend(empty_cols)

    # Remove columns with a standard deviation smaller than 0.02
    # Use skipna=True to handle NaN values and check for valid std calculation
    low_std_cols = []
    for col in df.columns:
        if "sensor" in col or "op_setting" in col:
            try:
                # Calculate std with skipna=True to avoid NaN issues
                col_std = df[col].std(skipna=True)
                # Check if std is valid (not NaN) and less than threshold
                if pd.notna(col_std) and col_std < 0.02:
                    low_std_cols.append(col)
            except (ZeroDivisionError, RuntimeWarning):
                # If std calculation fails, consider removing the column
                low_std_cols.append(col)

    removed_columns.extend(low_std_cols)

    print("Removed columns:", removed_columns)
    return removed_columns


# Calculate Remaining Useful Life (RUL)
def add_rul(df):
    max_cycles = df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]
    df = df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


# Normalize sensor data
def normalize_data(df, scaler=None):
    # Extract sensor columns
    sensor_cols = [col for col in df.columns if "sensor" in col or "op_setting" in col]

    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # Handle problematic values before normalization
    for col in sensor_cols:
        # Replace infinite values with NaN
        df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan)

        # If column has all NaN values, fill with zeros
        if df_copy[col].isna().all():
            df_copy[col] = 0.0
        # If column has some NaN values, fill with column mean
        elif df_copy[col].isna().any():
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())

    # Check for columns with zero or very small variance that could cause issues
    problematic_cols = []
    for col in sensor_cols:
        col_std = df_copy[col].std()
        if pd.isna(col_std) or col_std == 0 or np.isinf(col_std):
            problematic_cols.append(col)

    # For problematic columns, set them to a constant value to avoid normalization issues
    for col in problematic_cols:
        df_copy[col] = 0.0

    if scaler is None:
        scaler = StandardScaler()
        # Fit and transform only non-problematic columns
        valid_sensor_cols = [col for col in sensor_cols if col not in problematic_cols]
        if valid_sensor_cols:
            df_copy[valid_sensor_cols] = scaler.fit_transform(
                df_copy[valid_sensor_cols]
            )
        # For problematic columns, keep them as zeros (already normalized)
    else:
        # Transform only columns that were used during fitting
        # Get feature names from the scaler if available
        try:
            if hasattr(scaler, "feature_names_in_"):
                scaler_cols = list(scaler.feature_names_in_)
            else:
                # Fall back to all sensor columns (assuming same order as training)
                scaler_cols = sensor_cols

            valid_cols = [
                col
                for col in scaler_cols
                if col in df_copy.columns and col not in problematic_cols
            ]
            if valid_cols:
                df_copy[valid_cols] = scaler.transform(df_copy[valid_cols])
        except Exception as e:
            print(f"Warning: Could not apply scaler to some columns: {e}")
            # Fall back to transforming all available sensor columns
            available_cols = [col for col in sensor_cols if col not in problematic_cols]
            if available_cols:
                try:
                    df_copy[available_cols] = scaler.transform(df_copy[available_cols])
                except Exception:
                    print("Warning: Scaler transform failed, keeping original values")

    return df_copy, scaler


def prep_data(file_path):
    df = load_dataframe(file_path)
    df = add_rul(df)
    df, scaler = normalize_data(df, scaler=None)
    return df, scaler


# Prepare sequences for LSTM input
def prepare_train_sequences(df, sequence_length):
    sensor_cols = [col for col in df.columns if "sensor" in col or "op_setting" in col]
    X, y = [], []
    for unit in df["unit_number"].unique():
        unit_data = df[df["unit_number"] == unit]
        for i in range(len(unit_data) - sequence_length):
            X.append(unit_data[sensor_cols].iloc[i : i + sequence_length].values)
            y.append(unit_data["RUL"].iloc[i + sequence_length])
    X, y = np.array(X), np.array(y)

    return X, y


def prepare_test_sequences(test_data, test_RUL, sequence_length):
    sensor_cols = [
        col for col in test_data.columns if "sensor" in col or "op_setting" in col
    ]
    X, y = [], []
    for unit in test_data["unit_number"].unique():
        unit_data = test_data[test_data["unit_number"] == unit]
        if unit_data.shape[0] > sequence_length:
            X.append(unit_data[sensor_cols].iloc[-sequence_length:].values)
            y.append(test_RUL["RUL"].iloc[unit - 1])
    X, y = np.array(X), np.array(y)

    return X, y


class MLPmodel(nn.Module):
    def __init__(
        self,
        layer_units: list,
        input_size: int,
        output_size: int,
        dropout_rate: float = 0.5,
    ):
        super(MLPmodel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.model_type = "MLP"

        # Add the first layer
        self.layers.append(nn.Linear(input_size, layer_units[0]))
        self.dropouts.append(nn.Dropout(dropout_rate))

        # Add hidden layers
        for i in range(1, len(layer_units)):
            self.layers.append(nn.Linear(layer_units[i - 1], layer_units[i]))
            self.dropouts.append(nn.Dropout(dropout_rate))

        # Add the output layer (no dropout here)
        self.layers.append(nn.Linear(layer_units[-1], output_size))

        # Initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, nonlinearity="relu"
                )  # Kaiming for ReLU
                nn.init.constant_(layer.bias, 0)  # Bias initialized to 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = nn.ReLU()(x)
            x = self.dropouts[i](x)  # Apply dropout

        # Output layer (no activation or dropout)
        x = self.layers[-1](x)
        return x


class LSTMmodel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5
    ):
        super(LSTMmodel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = "LSTM"

        # Define LSTM layer with dropout (applies dropout between layers if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate
        )

        # Define a fully connected layer to map LSTM output to the target
        self.fc = nn.Linear(hidden_size, output_size)

        # Define a dropout layer before the fully connected layer
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)  # Bias initialized to 0

        # Initialize fully connected layer weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        out = out[:, -1, :]

        # Apply dropout before the fully connected layer
        out = self.dropout(out)

        # Pass through the fully connected layer
        out = self.fc(out)
        return out


class CMAPSSDataset(Dataset):
    def __init__(self, X, y):
        if not isinstance(X, (np.ndarray, torch.Tensor)):
            print("Warning: X is not a numpy array or torch.Tensor.")
        if not isinstance(y, (np.ndarray, torch.Tensor)):
            print("Warning: y is not a numpy array or torch.Tensor.")

        self.X = (
            X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        )
        self.y = (
            y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path="model_checkpoints/best_model.pth"):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: nn.Module) -> None:
        """Save the model when validation loss improves."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)


def train_model(model, X_train, y_train, X_val, y_val, initial_lr=0.01, num_epochs=5):
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create PyTorch Dataset and DataLoader
    train_dataset = CMAPSSDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Validation dataset
    val_dataset = CMAPSSDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=50, delta=0.0001)

    # Initialize lists to store loss values
    train_losses = []
    val_losses = []
    lr = []

    # Training loop with early stopping
    model.train()

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        model.train()  # Set model to training mode
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Compute average training loss for the epoch
        train_losses.append(epoch_train_loss / len(train_loader))

        # Validation loss
        model.eval()  # Set model to evaluation mode
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(
                    device
                )
                val_outputs = model(X_val_batch)
                val_loss = criterion(val_outputs.squeeze(), y_val_batch)
                epoch_val_loss += val_loss.item()

        # Compute average validation loss for the epoch
        val_losses.append(epoch_val_loss / len(val_loader))

        # Print progress with learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        lr.append(current_lr)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, LR: {current_lr:.6f}"
        )

        # Step the scheduler with validation loss
        scheduler.step(val_losses[-1])

        # Check early stopping
        early_stopping(val_losses[-1], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return train_losses, val_losses, lr


def plot_training_progress(train_losses, val_losses, lr, model_name="Model"):
    _, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )

    # Upper plot: Training and Validation Loss
    ax1.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Train Loss",
        color=COLORS["OCEAN"],
    )
    ax1.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Validation Loss",
        color=COLORS["OCEAN_3"],
    )
    ax1.set_yscale("log")  # Set y-axis to log scale
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} Training History")
    ax1.legend()
    ax1.grid()

    # Lower plot: Learning Rate
    ax2.plot(range(1, len(lr) + 1), lr, label="Learning Rate", color=COLORS["OCEAN"])
    ax2.set_yscale("log")  # Set y-axis to log scale
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Learning Rate")
    ax2.grid()

    plt.tight_layout()
    plt.show()


def update_batch_size(model, new_batch_size):
    for input_tensor in model.graph.input:
        shape = input_tensor.type.tensor_type.shape
        dim = shape.dim[0]  # Batch size is usually the first dimension
        dim.dim_value = new_batch_size  # Set the new batch size
    return model


def predict_with_onnx(model_path, X):
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Get the input name for the ONNX model
    input_name = session.get_inputs()[0].name

    # Get the batch size from the ONNX model
    batch_size = session.get_inputs()[0].shape[0]

    # Split the data into batches
    num_samples = X.shape[0]
    predictions = []

    for i in range(0, num_samples, batch_size):
        batch = X[i : i + batch_size]
        # Run the model on the batch
        batch_predictions = session.run(None, {input_name: batch.astype(np.float32)})
        predictions.append(batch_predictions[0])

    # Combine all predicted batches into one output array
    y = np.concatenate(predictions, axis=0)

    return y


def calculate_kpis(y_pred):
    metrics = {}
    for data_split in ["Train", "Test"]:
        ind = y_pred["Train_test"] == data_split
        mse = mean_squared_error(y_pred[ind]["y"], y_pred[ind]["y_pred"])
        r2 = r2_score(y_pred[ind]["y"], y_pred[ind]["y_pred"])
        metrics[f"{data_split.capitalize()}"] = [mse, r2]

    kpi_df = pd.DataFrame(metrics, index=["MSE", "R2"])
    return kpi_df


def plot_true_vs_predicted(y_pred, kpi_df):
    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    sns.scatterplot(
        data=y_pred,
        x="y",
        y="y_pred",
        hue="Train_test",
        palette={"Train": COLORS["OCEAN_3"], "Test": COLORS["BERRY_MAIN"]},
        style="Train_test",
        ax=ax,
        edgecolor=None,
        markers=[".", "s"],
        alpha=0.9,
    )
    ax.plot(
        [y_pred["y"].min(), y_pred["y"].max()],
        [y_pred["y"].min(), y_pred["y"].max()],
        color=COLORS["BLACK"],
        linestyle="-",
        lw=2,
    )
    ax.set_title(
        f"MSE Train: {kpi_df.loc['MSE', 'Train']:.0f}, R2 Train: {kpi_df.loc['R2', 'Train']:.2f}\n"
        f"MSE Test: {kpi_df.loc['MSE', 'Test']:.0f}, R2 Test: {kpi_df.loc['R2', 'Test']:.2f}\n"
    )
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    plt.tight_layout()
    plt.show()
