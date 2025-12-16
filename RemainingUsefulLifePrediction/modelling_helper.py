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


import os
import io
import zipfile
from typing import Tuple
from urllib.request import urlopen
from urllib.error import URLError

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

# NASA C-MAPSS dataset configuration
NASA_DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
NASA_DATASET_FILES = {
    "train": [
        "train_FD001.txt",
        "train_FD002.txt",
        "train_FD003.txt",
        "train_FD004.txt",
    ],
    "test": ["test_FD001.txt", "test_FD002.txt", "test_FD003.txt", "test_FD004.txt"],
    "rul": ["RUL_FD001.txt", "RUL_FD002.txt", "RUL_FD003.txt", "RUL_FD004.txt"],
}


def download_nasa_dataset(data_dir="data"):
    """
    Download and extract the NASA C-MAPSS dataset from the official source.

    Args:
        data_dir (str): Directory where data files should be stored

    Returns:
        bool: True if successful, False otherwise
    """
    os.makedirs(data_dir, exist_ok=True)

    # Check if all required files already exist
    all_files_exist = True
    for file_group in NASA_DATASET_FILES.values():
        for filename in file_group:
            if not os.path.exists(os.path.join(data_dir, filename)):
                all_files_exist = False
                break
        if not all_files_exist:
            break

    if all_files_exist:
        return True

    print("Downloading NASA C-MAPSS dataset from official source...")

    try:
        # Download the zip file
        print(f"Connecting to: {NASA_DATASET_URL}")
        with urlopen(NASA_DATASET_URL) as response:
            if response.status != 200:
                print(f"HTTP Error {response.status}: Failed to download dataset")
                return False

            print("Downloading zip file...")
            zip_data = response.read()
            print(f"Downloaded {len(zip_data):,} bytes")

        # Extract the outer zip file and then the inner zip file
        print("Extracting files...")
        with zipfile.ZipFile(io.BytesIO(zip_data)) as outer_zip:
            # Find the inner CMAPSSData.zip file
            inner_zip_path = None
            for file_path in outer_zip.namelist():
                if file_path.endswith("CMAPSSData.zip"):
                    inner_zip_path = file_path
                    break

            if not inner_zip_path:
                print("Error: CMAPSSData.zip not found in archive")
                return False

            print(f"Found inner archive: {inner_zip_path}")

            # Extract the inner zip file
            with outer_zip.open(inner_zip_path) as inner_zip_file:
                inner_zip_data = inner_zip_file.read()

            # Now extract files from the inner zip
            with zipfile.ZipFile(io.BytesIO(inner_zip_data)) as inner_zip:
                zip_files = inner_zip.namelist()
                print(f"Found {len(zip_files)} files in inner archive")

                extracted_files = []
                for file_group in NASA_DATASET_FILES.values():
                    for filename in file_group:
                        # Find the file in the inner zip
                        matching_files = [f for f in zip_files if f.endswith(filename)]

                        if matching_files:
                            zip_file_path = matching_files[0]
                            print(f"Extracting: {filename}")
                            with inner_zip.open(zip_file_path) as source:
                                target_path = os.path.join(data_dir, filename)
                                with open(target_path, "wb") as target:
                                    target.write(source.read())
                            extracted_files.append(filename)
                        else:
                            print(f"Warning: {filename} not found in inner archive")

                print(f"Successfully extracted {len(extracted_files)} files")
                return len(extracted_files) >= len(NASA_DATASET_FILES["train"])

    except URLError as e:
        print(f"Network error: {e}")
        print("Please check your internet connection and try again.")
        return False
    except zipfile.BadZipFile as e:
        print(f"Archive error: {e}")
        print("Downloaded file appears to be corrupted.")
        return False
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        return False


def load_dataframe(file_path):
    """
    Load NASA C-MAPSS dataset with automatic download fallback.

    Args:
        file_path (str): Path to the data file (e.g., "data/train_FD001.txt")

    Returns:
        pandas.DataFrame: Loaded and cleaned dataframe
    """
    # Try loading local file first
    if os.path.exists(file_path):
        print(f"Loading data from local file: {file_path}")
        df = pd.read_csv(file_path, sep=" ", header=None, names=column_names)
    else:
        # File doesn't exist, try to download
        print(f"Local file not found: {file_path}")
        data_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else "data"

        if download_nasa_dataset(data_dir):
            if os.path.exists(file_path):
                print(f"Downloaded NASA dataset. Loading: {file_path}")
                df = pd.read_csv(file_path, sep=" ", header=None, names=column_names)
            else:
                raise FileNotFoundError(
                    f"File '{file_path}' not found even after download attempt."
                )
        else:
            raise FileNotFoundError(
                f"Cannot load '{file_path}' and download failed. Check internet connection."
            )

    # Clean the dataframe
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        print(f"Removing completely empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)

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
    """
    Prepare data by loading, adding RUL, and normalizing.

    Args:
        file_path (str): Path to the data file

    Returns:
        tuple: (processed_dataframe, scaler)
    """
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
