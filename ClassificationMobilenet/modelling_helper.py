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


from pyexpat import model
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from PIL import Image
import torch
import os
import kagglehub
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import copy


import requests
import os
import shutil
import zipfile
import sys
import json

parent_dir = os.path.dirname(os.getcwd())

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import CentralScripts.helper_functions as cs

plt.rcParams.update({"font.size": 13})


def get_rain_classes():
    return ["Clear", "Rain Drop"]


def get_classes_json(filename):
    json_path = os.path.join("json", filename)
    with open(json_path, "r") as f:
        classes = json.load(f).items()
    return {int(key): value for key, value in classes}


def adapt_model(model, number_classes):
    model = model.to("cpu")
    number_ftrs = model.classifier[-1].in_features
    number_classes = 2
    model.classifier[-1] = nn.Linear(number_ftrs, number_classes)
    return model


def get_model_indices(model, classes):
    indices = get_keys(classes)
    layer = model.classifier.pop(key=-1)
    new_classifier = nn.Linear(layer.in_features, len(indices))
    new_classifier.weight.data = layer.weight.data[indices, :]
    new_classifier.bias.data = layer.bias.data[indices]
    model.classifier.append(new_classifier)
    model.eval()
    return model


def get_data():

    data_folder = os.path.join(os.getcwd(), "data")
    data_set = os.path.join("sautkin", "imagenet1kvalid")
    folder_path = os.path.join(data_folder, "datasets", data_set, "versions", "2")

    if not os.path.isdir(folder_path):
        # set download folder for kaggle
        os.environ["KAGGLEHUB_CACHE"] = data_folder
        kagglehub.dataset_download(data_set)

    return folder_path


def get_keys(dictionary):
    return list(dictionary.keys())


def get_values(dictionary):
    return list(dictionary.values())


def get_random_index(classes):
    classes_indices = get_keys(classes)
    random_class = random.choice(classes_indices)
    class_name = classes[random_class]
    return random_class, class_name


def get_folder_index(folder_path, index):
    folder_list = os.listdir(folder_path)
    folder = list(filter(lambda folder: str(index) in folder, folder_list))[0]
    return folder


def get_path_random_image(folder_path, folder):
    # import random image from folder
    image_list = os.listdir(os.path.join(folder_path, folder))
    random_index = torch.randint(0, len(image_list), (1,)).item()
    image_path = os.path.join(folder_path, folder, image_list[random_index])
    return image_path


def import_image_tensor(image_path):
    with Image.open(image_path) as img:
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
    return img


def downsample_image_tensor(image):
    return nn.functional.interpolate(
        image, size=(224, 224), mode="bilinear", align_corners=False
    )


def get_image(image_path):
    image = import_image_tensor(image_path)
    image = downsample_image_tensor(image)
    return image


def plot_image(image, title):
    image_plot = np.transpose(image, axes=(1, 2, 0))
    image_plot = scale_image(image_plot)
    plt.imshow(image_plot)
    plt.title(title)
    plt.axis("off")  # Optional: Turn off axes for a cleaner look
    plt.show()


def get_input_dataloader(dataloader, is_plot=False):
    inputs, label = next(iter(dataloader))
    input_target = inputs[0].numpy()
    input_size = inputs[0].shape
    classes = get_rain_classes()
    class_name = classes[label[0].item()]

    if is_plot:
        plot_image(input_target, class_name)

    return input_target, input_size


def get_random_input(folder_path, classes, is_plot=False):
    random_class, class_name = get_random_index(classes)
    input_folder = get_folder_index(folder_path, random_class)
    image_path = get_path_random_image(folder_path, input_folder)
    input_target = np.squeeze(get_image(image_path).numpy())
    if is_plot:
        plot_image(input_target, class_name)
    return input_target, np.shape(input_target)


def plot_confusion(confusion_matrix, classes_names):

    plt.figure(figsize=(15, 10))

    df_cm = pd.DataFrame(
        confusion_matrix, index=classes_names, columns=classes_names
    ).astype(int)
    heatmap = sns.heatmap(
        df_cm,
        annot=True,
        fmt="d",
        cmap=[cs.COLORS["OCEAN"], cs.COLORS["OCEAN_1"], cs.COLORS["OCEAN_2"]],
    )

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right"
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.show()


def download_zip(url, output_path):

    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Raise exception if request fails
        total_size = int(r.headers.get("Content-Length", 0))
        downloaded_size = 0

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Calculate and display percentage
                    percentage = (
                        (downloaded_size / total_size) * 100 if total_size else 0
                    )
                    print(f"\rDownloaded: {percentage:.2f}%", end="")


def unzip(zip_path, file_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(file_path)
    print(f"\nFiles extracted to: {file_path}")


def get_old_folder_names():
    return ["Clear", "Drop"]


def get_split_names():
    return ["train", "val", "test"]


def restructure_old_folders(path):
    old_folder_names = get_old_folder_names()
    for folder_name in old_folder_names:
        folder_path = os.path.join(path, folder_name)
        restructure_folder(folder_path)


def restructure_folder(folder_path):
    # Get all subfolders in the main folder
    subfolders = [
        f
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]

    # Iterate through each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)

        # Get all files in the subfolder
        files = os.listdir(subfolder_path)

        # Iterate through each file
        for file in files:
            # Construct new file name
            new_file_name = f"{subfolder}{file}"
            new_file_path = os.path.join(folder_path, new_file_name)

            # Move and rename the file
            old_file_path = os.path.join(subfolder_path, file)
            shutil.move(old_file_path, new_file_path)

        # Remove the now-empty subfolder
        os.rmdir(subfolder_path)


def create_new_folders(path):
    split_names = get_split_names()
    old_folder_names = get_old_folder_names()

    for split_name in split_names:
        split_path = os.path.join(path, split_name)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        for folder in old_folder_names:
            split_folder_path = os.path.join(split_path, folder)
            if not os.path.exists(split_folder_path):
                os.makedirs(split_folder_path)


def move_file(old_path, new_path):
    shutil.move(old_path, new_path)


def delete_folder(path, folder):
    del_path = os.path.join(path, folder)
    if os.path.exists(del_path):
        shutil.rmtree(del_path)


def move_data(path):

    old_folder_names = get_old_folder_names()

    for folder in old_folder_names:

        split_names = get_split_names()

        files = os.listdir(os.path.join(path, folder))
        num_files = len(files)

        # Calculate the number of files for each split
        num_train = int(num_files * 0.8)
        num_val = int(num_files * 0.1)

        # Move files to train, val, and test folders
        for i, file in enumerate(files):

            old_path = os.path.join(path, folder, file)

            if i < num_train:
                new_path = os.path.join(path, split_names[0], folder, file)

            elif i < num_train + num_val:
                new_path = os.path.join(path, split_names[1], folder, file)

            else:
                new_path = os.path.join(path, split_names[2], folder, file)

            move_file(old_path, new_path)

        # Remove the original folders
        delete_folder(path, folder)


def fetch_sort_data():

    # Dropbox direct download URL
    url = "https://www.dropbox.com/scl/fi/qes7r934c10qzb21funoj/DayRainDrop_Train.zip?rlkey=bdqa53wgvmhj9x1yf40q0c1p7&dl=1"
    zip_path = "data/DayRainDrop_Train.zip"
    data_file_path = "data"
    file_path = os.path.join(data_file_path, "DayRainDrop_Train")

    if not os.path.isdir(file_path):
        os.makedirs(data_file_path, exist_ok=True)
        download_zip(url, zip_path)
        unzip(zip_path, data_file_path)
        delete_folder(file_path, "Blur")
        restructure_old_folders(file_path)
        create_new_folders(file_path)
        move_data(file_path)

    print(f"Data available in {file_path}!")

    return file_path


def get_predicted_class(input, model, classes):
    classes_names = get_values(classes)
    output_target = torch.tensor(cs.get_predictions("torch", model, np.array(input)))
    max_index = torch.argmax(output_target).item()
    output_target = np.array(output_target)

    predicted_class = classes_names[max_index]
    print(f"Predicting the class: {predicted_class}\n")
    return output_target, predicted_class


def get_dataloader(path, batch_size=1, mode="train"):

    data_transforms = get_data_transforms()
    image_datasets = datasets.ImageFolder(path, data_transforms[f"{mode}"])

    if mode not in ["train", "val", "test"]:
        print("Mode not found")
        return None, None, None

    dataloader = torch.utils.data.DataLoader(
        image_datasets, batch_size=batch_size, shuffle=True, num_workers=0
    )
    dataset_sizes = len(image_datasets)
    class_names = image_datasets.classes

    return dataloader, class_names, dataset_sizes


def get_data_transforms():
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.18, 0.18, 0.18]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.18, 0.18, 0.18]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.18, 0.18, 0.18]),
            ]
        ),
    }
    return data_transforms


def calculate_confusion_matrix(folder_path, model, device, classes):

    dataloader, _, _ = get_dataloader(folder_path, batch_size=1, mode="val")

    classes_indices = get_keys(classes)
    classes_names = get_values(classes)
    number_classes = len(classes_indices)

    confusion_matrix = np.zeros((number_classes, number_classes))

    model = model.to(device)

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloader):
            if classes.item() in classes_indices:
                inputs = inputs.to(device)
                classes = classes.to(device)
                indices = [classes_indices.index(c.item()) for c in classes]
                outputs = model(inputs)
                ind = torch.argmax(outputs, 1)[0].item()
                confusion_matrix[indices[0], ind] += 1

    plot_confusion(confusion_matrix, classes_names)


def get_sample_images(model, dataloader, classes, device="cpu", num_samples=5):

    if type(classes) is dict:
        classes_keys = get_keys(classes)
        classes_values = get_values(classes)
    elif type(classes) is list:
        classes_keys = list(range(len(classes)))
        classes_values = classes

    # Get a batch from the dataloader
    data_iter = iter(dataloader)
    inputs, true_labels = next(data_iter)
    sample_inputs = []
    true_classes = []
    predicted_classes = []

    for input, true_label in zip(inputs, true_labels):
        if true_label.item() in classes_keys:
            sample_inputs.append(input)
            true_classes.append(classes[true_label.item()])

            sample_input = input.detach().clone().to(device)
            predictions = cs.get_predictions("torch", model, sample_input)
            predicted_class = torch.argmax(torch.tensor(predictions), dim=1)
            predicted_classes.append(classes_values[predicted_class.item()])

        if len(sample_inputs) >= num_samples:
            break

    num_samples = len(sample_inputs)

    return sample_inputs, predicted_classes, true_classes


def scale_image(data):
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return scaled_data


def plot_sample_images(sample_inputs, predicted_classes, true_classes, num_samples=5):

    number_images = min(len(sample_inputs), num_samples)

    fig, axes = plt.subplots(1, number_images, figsize=(15, 3))

    for i in range(min(len(sample_inputs), number_images)):
        img = sample_inputs[i].permute(1, 2, 0).cpu().numpy()
        img = scale_image(img)
        axes[i].imshow(img)
        predicted_class = predicted_classes[i]
        true_class = true_classes[i]
        axes[i].set_title(f"Prediction: {predicted_class}\nGround truth: {true_class}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def calculate_predictions_and_plot(model, folder_path, classes, num_samples=5):

    dataloader, _, _ = get_dataloader(folder_path, batch_size=200, mode="test")
    sample_inputs, predicted_classes, true_labels = get_sample_images(
        model, dataloader, classes
    )
    plot_sample_images(sample_inputs, predicted_classes, true_labels, num_samples)


def train_model(model, dataloader, dataset_sizes, criterion, optimizer):

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    num_epochs = 10
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


def test_model(model, dataloader, dataset_sizes, criterion):

    device = cs.get_device()
    model.to(device)

    running_loss = 0.0
    running_corrects = 0

    classes = get_rain_classes()
    number_classes = len(classes)
    confusion_matrix = np.zeros((number_classes, number_classes))

    model = model.to(device)

    with torch.no_grad():

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            confusion_matrix[np.array(labels.cpu()), np.array(preds.cpu())] += 1
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / dataset_sizes
    acc = running_corrects.double() / dataset_sizes
    print(f"Loss: {loss:.4f}, accuracy: {acc:.4f}")
    plot_confusion(confusion_matrix, classes)
