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
import random
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from tqdm import tqdm
import _CentralScripts.helper_functions as cs

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DATASET_URL = "https://data.mendeley.com/public-api/zip/mzb4b6dff3/download/1"
DATASET_FALLBACK_URL = "https://data.mendeley.com/public-files/datasets/mzb4b6dff3/files/8d63eebb-fdcb-4d8e-b18c-683a9f570c82/file_downloaded"
DATASET_PAGE_URL = "https://data.mendeley.com/datasets/mzb4b6dff3/1"
DATASET_CACHE_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/mzb4b6dff3-1.zip"
DATASET_DISPLAY_NAME = "Multi-Class Driver Behavior Image Dataset"


class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None, class_to_idx=None):
        self.samples = samples
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_name = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label


def _normalize_local_dataset_root(dataset_root: str) -> Path:
    """Convert Windows-style paths to a valid local filesystem path, including WSL paths."""
    path = dataset_root.strip()
    if not path:
        return Path(".").resolve()

    if path.startswith("/"):
        return Path(path).expanduser().resolve()

    normalized = path.replace("\\", "/")
    if ":" in normalized and normalized[1:3] == ":/":
        drive, rest = normalized[0].lower(), normalized[2:]
        if drive == "c":
            return Path("/mnt/c") / rest.lstrip("/")
        return Path(normalized).expanduser().resolve()

    return Path(normalized).expanduser().resolve()


class _DSBlock(nn.Module):
    """Depthwise separable block with optional residual connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw(self.dw(x))
        if self.use_residual:
            out = out + x
        return out


class _MiniDMSNet(nn.Module):
    """Simple wrapper exposing features/classifier for compatibility."""

    def __init__(self, features: nn.Sequential, classifier: nn.Sequential) -> None:
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_minidms(
    width_mult: float = 1.0, num_classes: int = 4, verbose: bool = False
) -> nn.Module:
    """
    MiniDMS-Net: compact CNN designed for lightweight embedded deployment.

    The implementation follows the same structure as the distracted driver classifier
    MiniDMS builder, with a width-scaling multiplier to adjust the model size.
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be greater than 0")

    def c(n: int) -> int:
        return max(1, int(n * width_mult))

    features = nn.Sequential(
        nn.Conv2d(3, c(16), 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c(16)),
        nn.ReLU6(inplace=True),
        _DSBlock(c(16), c(32), stride=2),
        _DSBlock(c(32), c(32), stride=1),
        _DSBlock(c(32), c(64), stride=2),
        _DSBlock(c(64), c(64), stride=1),
        _DSBlock(c(64), c(64), stride=1),
        _DSBlock(c(64), c(128), stride=2),
        _DSBlock(c(128), c(128), stride=1),
        _DSBlock(c(128), c(128), stride=1),
    )

    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(p=0.2),
        nn.Linear(c(128), num_classes),
    )

    model = _MiniDMSNet(features, classifier)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    if verbose:
        try:
            from torchinfo import summary

            summary(
                model,
                input_size=(1, 3, 128, 128),
                col_names=("input_size", "output_size", "num_params"),
                depth=4,
            )
        except ImportError:
            print(model)
            print("\n[install torchinfo: pip install torchinfo]")

    return model


def _download_file_with_progress(
    url: str, destination: str, session: Optional[requests.Session] = None
) -> None:
    """Download a file with a visible progress bar when tqdm is available."""
    session = session or requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/zip,application/octet-stream,*/*",
        "Referer": DATASET_PAGE_URL,
        "Origin": "https://data.mendeley.com",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = session.get(
        url, headers=headers, stream=True, timeout=120, allow_redirects=True
    )
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024

    if tqdm is not None:
        with tqdm(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading dataset",
        ) as pbar:
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def _find_in_cabin_dataset_root(root: Path) -> Optional[Path]:
    """Find the extracted dataset root if it already exists locally."""
    candidates = [
        root / DATASET_DISPLAY_NAME,
        root / "raw" / DATASET_DISPLAY_NAME,
        root / "dataset" / DATASET_DISPLAY_NAME,
        root / "extracted" / DATASET_DISPLAY_NAME,
        *[p for p in root.rglob(DATASET_DISPLAY_NAME) if p.is_dir()],
    ]

    if root.name == DATASET_DISPLAY_NAME:
        candidates.append(root)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def extract_local_in_cabin_zip(
    project_dir: str = "./In-CabinMonitoringSystem",
    zip_name: str = "Multi-Class Driver Behavior Image Dataset.zip",
    extract_to: Optional[str] = None,
    expected_root_name: str = DATASET_DISPLAY_NAME,
    force_extract: bool = False,
) -> str:
    """Extract the local In-Cabin dataset ZIP and return the extracted root path."""
    project_path = Path(project_dir).expanduser().resolve()
    zip_path = project_path / zip_name

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    target_base = (
        Path(extract_to).expanduser().resolve() if extract_to else project_path
    )
    target_base.mkdir(parents=True, exist_ok=True)

    expected_root = target_base / expected_root_name
    if expected_root.exists() and any(expected_root.iterdir()) and not force_extract:
        print(f"Dataset already extracted at {expected_root}. Skipping extraction.")
        return str(expected_root)

    print(f"Extracting {zip_path.name} into {target_base}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_base)
    print(f"Extraction finished. Dataset available at: {expected_root}")

    if expected_root.exists() and expected_root.is_dir():
        return str(expected_root)

    # Fallback: infer the top-level extracted directory from ZIP members.
    with zipfile.ZipFile(zip_path, "r") as zf:
        top_levels = {
            p.split("/")[0]
            for p in zf.namelist()
            if p and not p.startswith("__MACOSX/")
        }
    top_levels = {name for name in top_levels if name.strip()}
    if len(top_levels) == 1:
        inferred_root = target_base / next(iter(top_levels))
        if inferred_root.exists() and inferred_root.is_dir():
            return str(inferred_root)

    return str(target_base)


def download_and_extract_dataset(
    dataset_url: str = DATASET_URL,
    dataset_root: str = "./data",
    archive_name: str = "in_cabin_dataset.zip",
    force_download: bool = False,
) -> str:
    """Download and extract the In-Cabin dataset if it is not already present."""
    root = Path(dataset_root).expanduser().resolve()
    download_dir = root / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / archive_name

    extracted_root = _find_in_cabin_dataset_root(root)
    if (
        extracted_root
        and extracted_root.exists()
        and any(extracted_root.iterdir())
        and not force_download
    ):
        print(f"Dataset already available at {extracted_root}. Skipping download.")
        return str(extracted_root)

    if force_download or not archive_path.exists():
        print(f"Downloading dataset from {dataset_url}...")
        session = requests.Session()
        # Warm up cookies/session on the dataset page before file download endpoints.
        try:
            session.get(DATASET_PAGE_URL, timeout=30)
        except requests.RequestException:
            pass

        candidates = [dataset_url, DATASET_FALLBACK_URL, DATASET_CACHE_URL]
        last_error = None
        for candidate in candidates:
            try:
                if candidate != dataset_url:
                    print(f"Retrying with alternate URL: {candidate}")
                _download_file_with_progress(
                    candidate, str(archive_path), session=session
                )
                last_error = None
                break
            except requests.RequestException as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(
                "Unable to download dataset automatically (all known endpoints failed). "
                f"Please download the ZIP manually from {DATASET_PAGE_URL} and place it at {archive_path}."
            ) from last_error

    print(f"Extracting dataset into {root}...")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(root)

    extracted_root = _find_in_cabin_dataset_root(root)
    if extracted_root is None:
        raise FileNotFoundError(
            f"Dataset was extracted, but '{DATASET_DISPLAY_NAME}' was not found under {root}."
        )

    return str(extracted_root)


def _find_image_root(extracted_root: str) -> str:
    """Find the first image dataset directory inside the extracted archive."""
    root = Path(extracted_root)
    if (root / "train").exists() and (root / "test").exists():
        return str(root)

    candidate_dirs = [
        root,
        *[p for p in root.rglob("*") if p.is_dir()],
    ]

    for folder in candidate_dirs:
        if any((folder / name).is_dir() for name in ("train", "test")):
            return str(folder)
        if any(
            (folder / file).is_file()
            for file in os.listdir(folder)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ):
            return str(folder)
    return str(root)


def prepare_train_test_dirs(
    extracted_root: str,
    train_dir: str | None = None,
    test_dir: str | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[str, str]:
    """Create separate train and test directories if they are not already present."""
    source_root = Path(_find_image_root(extracted_root))
    base_root = (
        source_root.parent if source_root.name in {"raw", "dataset"} else source_root
    )
    train_path = Path(train_dir) if train_dir else base_root / "train"
    test_path = Path(test_dir) if test_dir else base_root / "test"

    if (train_path / "images").exists() and (test_path / "images").exists():
        return str(train_path), str(test_path)

    if (source_root / "train").exists() and (source_root / "test").exists():
        return str(source_root / "train"), str(source_root / "test")

    class_dirs = [p for p in source_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found under {source_root}")

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        class_images = sorted(p for p in class_dir.iterdir() if p.is_file())
        if not class_images:
            continue

        rng = random.Random(seed)
        rng.shuffle(class_images)
        split_index = max(1, int(len(class_images) * train_ratio))

        train_class_dir = train_path / class_dir.name
        test_class_dir = test_path / class_dir.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        for image in class_images[:split_index]:
            dst = train_class_dir / image.name
            dst.write_bytes(image.read_bytes())

        for image in class_images[split_index:]:
            dst = test_class_dir / image.name
            dst.write_bytes(image.read_bytes())

    return str(train_path), str(test_path)


def _normalize_class_name(name: str) -> str:
    """Normalize a class label to a stable folder name."""
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned


def get_local_train_test_loaders(
    dataset_root: str,
    classes: Optional[Iterable[str]] = None,
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/test dataset loaders from a local directory structure.

    Supported layouts:
      - dataset_root/train/... and dataset_root/test/...
      - dataset_root/<class_name>/images
      - dataset_root/<class_name>/*

    Default class names are a compact in-cabin driver behavior set:
      safe_driving, turning, texting_phone, talking_phone
    """
    root = _normalize_local_dataset_root(dataset_root)
    if classes is None:
        classes = [
            "safe_driving",
            "turning",
            "texting_phone",
            "talking_phone",
        ]

    class_list = [_normalize_class_name(c) for c in classes]

    direct_class_dirs = [
        p
        for p in root.iterdir()
        if p.is_dir() and _normalize_class_name(p.name) in set(class_list)
    ]

    if direct_class_dirs:
        train_dir = root
        test_dir = root
    else:
        train_dir = root / "train"
        test_dir = root / "test"

        if not train_dir.exists() or not test_dir.exists():
            split_root = root / "split"
            train_dir = split_root / "train"
            test_dir = split_root / "test"
            train_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)

            for class_name in class_list:
                class_candidates = [
                    root / class_name,
                    root / class_name.replace("_", " "),
                ]
                found_dir = next(
                    (p for p in class_candidates if p.exists() and p.is_dir()), None
                )
                if found_dir is None:
                    for folder in root.iterdir():
                        if (
                            folder.is_dir()
                            and _normalize_class_name(folder.name) == class_name
                        ):
                            found_dir = folder
                            break
                if found_dir is None:
                    continue

                image_files = sorted(
                    p
                    for p in found_dir.rglob("*")
                    if p.is_file()
                    and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                )
                if not image_files:
                    continue

                rng = random.Random(seed)
                rng.shuffle(image_files)
                split_index = max(1, int(len(image_files) * train_ratio))

                train_class_dir = train_dir / class_name
                test_class_dir = test_dir / class_name
                train_class_dir.mkdir(parents=True, exist_ok=True)
                test_class_dir.mkdir(parents=True, exist_ok=True)

                for img in image_files[:split_index]:
                    target = train_class_dir / img.name
                    target.write_bytes(img.read_bytes())
                for img in image_files[split_index:]:
                    target = test_class_dir / img.name
                    target.write_bytes(img.read_bytes())

        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(
                f"Could not find a compatible local dataset structure under {root}. "
                "Expected train/test folders or class folders such as safe_driving, turning, texting_phone, talking_phone."
            )

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageFolder(root=str(train_dir), transform=transform)
    test_dataset = ImageFolder(root=str(test_dir), transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def get_train_test_loaders(
    dataset_root: str = "./data",
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Download the dataset, extract it, split into train/test folders, and return DataLoaders."""
    extracted_root = download_and_extract_dataset(dataset_root=dataset_root)
    train_dir, test_dir = prepare_train_test_dirs(
        extracted_root,
        train_ratio=train_ratio,
        seed=seed,
    )

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    """Run one full pass over loader. Returns (avg_loss, accuracy)."""
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    device: torch.device,
    phase_name: str,
    extra_meta: dict | None = None,
    patience: int = 0,
    ckpt_dir: Path = None,
    img_size: int = 0,
    num_classes: int = 0,
) -> None:
    """Train for `epochs` epochs, saving the best checkpoint by val loss.

    Args:
        patience: early-stopping patience in epochs (0 = disabled).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimiser, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimiser, device, train=False
        )
        scheduler.step()

        print(
            f"[{phase_name}] Epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            ckpt_path = ckpt_dir / f"best_{phase_name}.pth"
            ckpt = {
                "state_dict": model.state_dict(),
                "img_size": img_size,
                "num_classes": num_classes,
            }
            if extra_meta:
                ckpt.update(extra_meta)
            torch.save(ckpt, ckpt_path)
            print(f"  → saved checkpoint: {ckpt_path}")
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(
                    f"  Early stopping after {epoch} epochs "
                    f"({patience} epochs without improvement)."
                )
                break


def plot_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    class_names: Optional[Iterable[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> tuple[float, np.ndarray]:
    """Evaluate a PyTorch classifier and plot its confusion matrix."""
    model.eval()
    device = torch.device(device)

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        if hasattr(loader, "dataset") and hasattr(loader.dataset, "classes"):
            class_names = [
                str(name).replace("_", " ").title() for name in loader.dataset.classes
            ]
        else:
            class_names = [str(i) for i in range(cm.shape[0])]
    else:
        class_names = [str(name) for name in class_names]

    plot_values = cm.astype(int)
    if normalize:
        row_sums = plot_values.sum(axis=1, keepdims=True)
        plot_values = np.divide(
            plot_values,
            row_sums,
            out=np.zeros_like(plot_values, dtype=float),
            where=row_sums != 0,
        )

    off_diagonal_mask = ~np.eye(plot_values.shape[0], dtype=bool)
    max_off_diagonal = (
        plot_values[off_diagonal_mask].max()
        if plot_values[off_diagonal_mask].size > 0
        else plot_values.max()
    )
    vmax = max_off_diagonal if max_off_diagonal > 0 else plot_values.max()

    plt.figure(figsize=(8, 6))
    cmap = [
        cs.COLORS["OCEAN"],
        cs.COLORS["OCEAN_1"],
        cs.COLORS["OCEAN_2"],
        cs.COLORS["OCEAN_3"],
    ]
    sns.heatmap(
        plot_values,
        annot=True,
        # fmt=annot_fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0,
        vmax=vmax,
        square=True,
    )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    print(f"Validation accuracy: {accuracy:.4f}")
    class_accuracy = np.divide(
        cm.diagonal(),
        cm.sum(axis=1),
        out=np.zeros(cm.shape[0], dtype=float),
        where=cm.sum(axis=1) != 0,
    )
    print("Per-class accuracy:")
    print(class_accuracy)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return accuracy, cm
