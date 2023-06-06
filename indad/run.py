import click
from data import MVTecDataset, mvtec_classes, backbones, DEFAULT_SIZE, DEFAULT_RESIZE
from models import PatchCore
from utils import print_and_export_results
from typing import List
from pathlib import Path
from torchvision import transforms
import torch
import random
import numpy as np
import cv2



# seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import warnings  # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

# mvtec_classes() defined in "data.py" returns a list of str containing the datasets
ALL_CLASSES = mvtec_classes()

# choose the class of objects 
def select_dataset():
    print("Available Classes:")
    for index, dataset in enumerate(ALL_CLASSES):
        print(f"{index + 1}. {dataset}")

    while True:
        print("--------------------------")
        choice = input("Select a Class by entering its number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(ALL_CLASSES):
            dataset_index = int(choice) - 1
            print("Chosen:", ALL_CLASSES[dataset_index])  # print the chosen dataset
            return ALL_CLASSES[dataset_index]
        else:
            print("Invalid choice. Please enter a valid number.")

def cli_interface(dataset: str):
    print("--------------------------")
    print("Available Backbones:")
    index = 1
    # backbones dictionary imported from "utils.py"
    for key, value in backbones.items():
        for item in value:
          print(f"{index}. {key}: {item}")
          index += 1
    
    while True:
      print("--------------------------")
      choice = int(input("Select a Backbone by entering its number: "))
      if 1 <= choice < index:
        if choice == 1:
          imgnet = True
          backbone_name = backbones["WideResNet"][choice-1]
          print("-----ImageNet Version-----")
        else:
          imgnet = False
          backbone_name = backbones["ResNet"][choice-2]
          print("-------CLIP Version-------")

        print("Chosen:", backbone_name)
        return [backbone_name, imgnet]
      else:
        print("Invalid choice. Please enter a valid number.")


def run_model(class_name: str, backbone_name: str, imgnet: bool):

    """
    Runs a PatchCore model on a given dataset and returns the results.

    Args:
        classes (List): The classes to run the model on.

    Returns:
        dict: A dictionary containing the results of running the model on each class.
    """

    # according to imgnet value assign the image size using the constant used in "data.py"
    # image size CLIP
    #DEFAULT_SIZE = 224    
    # image size ImagNet
    #DEFAULT_RESIZE = 256
    if imgnet:
        size = DEFAULT_RESIZE
    else:
        size = DEFAULT_SIZE

    # create the PatchCore model
    # by default in "model.py" f_coreset is setted to 0.01
    # this means that the 1% of the total training samples is used to create the coreset
    # here it is specified to use the 10%
    model = PatchCore(
        f_coreset=.10,
        image_size = size,
        imgnet = imgnet, # True: ImageNet, False: CLIP 
        backbone_name = backbone_name
    )

    # variable inizialization
    results = {}    # key = class, Value = [image-level ROC AUC, pixel-level ROC AUC]

    print(f"\n█│ Running patchcore on {class_name} dataset.")
    print(f" ╰{'─' * (len(class_name) + 23)}\n")

    print("Load the training and test datasets")
    train_ds, test_ds = MVTecDataset(class_name, size, imgnet).get_dataloaders()

    print("Fit the model on the training dataset")
    print("   Training ...")
    model.fit(train_ds)
    
    # evaluate the model on the test dataset
    print("   Testing ...")
    image_rocauc, pixel_rocauc = model.evaluate(test_ds, class_name, backbone_name, size, imgnet)

    print(f"\n   ╭{'─' * (len(class_name) + 15)}┬{'─' * 20}┬{'─' * 20}╮")
    print(
        f"   │ Test results {class_name} │ image_rocauc: {image_rocauc:.2f} │ pixel_rocauc: {pixel_rocauc:.2f} │")
    print(f"   ╰{'─' * (len(class_name) + 15)}┴{'─' * 20}┴{'─' * 20}╯")
    results[class_name] = [float(image_rocauc), float(pixel_rocauc)]

    # calculate average image and pixel ROC AUC
    image_results = [v[0] for _, v in results.items()]
    average_image_roc_auc = sum(image_results) / len(image_results)
    image_results = [v[1] for _, v in results.items()]
    average_pixel_roc_auc = sum(image_results) / len(image_results)

    total_results = {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "average pixel rocauc": average_pixel_roc_auc,
        "model parameters": model.get_parameters(),
    }
    return total_results

# when you run */run.py, automatically the variable __name__ becomes __main__
if __name__ == "__main__":
    dataset = select_dataset()
    [backbone_name, imgnet] = cli_interface(dataset)
    total_results = run_model(dataset, backbone_name, imgnet)
    print_and_export_results(total_results, backbone_name)