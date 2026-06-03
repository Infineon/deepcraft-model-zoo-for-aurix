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


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import kagglehub
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

parent_dir = os.path.dirname(os.getcwd())

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import _CentralScripts.helper_functions as cs
import _CentralScripts.mobilenet_helper as mo

plt.rcParams.update({"font.size": 13})


def get_classes_json(filename):
    json_path = os.path.join("json", filename)
    with open(json_path, "r") as f:
        classes = json.load(f).items()
    return {int(key): value for key, value in classes}


def get_model_indices(model, classes):
    indices = mo.get_keys(classes)
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


def get_random_index(classes):
    classes_indices = mo.get_keys(classes)
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
    image_plot = mo.scale_image(image_plot)
    plt.imshow(image_plot)
    plt.title(title)
    plt.axis("off")  # Optional: Turn off axes for a cleaner look
    plt.show()


def get_random_input(folder_path, classes, is_plot=False):
    random_class, class_name = get_random_index(classes)
    input_folder = get_folder_index(folder_path, random_class)
    image_path = get_path_random_image(folder_path, input_folder)
    input_target = np.squeeze(get_image(image_path).numpy())
    if is_plot:
        plot_image(input_target, class_name)
    return input_target, np.shape(input_target)


def get_predicted_class(input, model, classes):
    classes_names = mo.get_values(classes)
    output_target = torch.tensor(cs.get_predictions("torch", model, np.array(input)))
    max_index = torch.argmax(output_target).item()
    output_target = np.array(output_target)

    predicted_class = classes_names[max_index]
    print(f"Predicting the class: {predicted_class}\n")
    return output_target, predicted_class


def calculate_confusion_matrix(folder_path, model, device, classes):

    dataloader, _, _ = mo.get_dataloader(folder_path, batch_size=1, mode="val")

    classes_indices = mo.get_keys(classes)
    classes_names = mo.get_values(classes)
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

    mo.plot_confusion(confusion_matrix, classes_names)
