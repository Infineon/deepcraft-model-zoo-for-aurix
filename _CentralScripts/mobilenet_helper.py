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


import matplotlib.pyplot as plt
import _CentralScripts.helper_functions as cs
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets


def get_keys(dictionary):
    return list(dictionary.keys())


def get_values(dictionary):
    return list(dictionary.values())


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


def scale_image(data):
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return scaled_data


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


def get_dataloader(path, batch_size=1, mode="train", resolution=224):

    data_transforms = get_data_transforms(resolution)
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


def get_data_transforms(resolution=224):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(resolution),
                transforms.ColorJitter(brightness=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.18, 0.18, 0.18]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(int(resolution * 1.2)),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.18, 0.18, 0.18]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(int(resolution * 1.2)),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.18, 0.18, 0.18]),
            ]
        ),
    }
    return data_transforms
