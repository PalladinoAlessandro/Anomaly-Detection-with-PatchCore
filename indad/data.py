import os
from os.path import isdir
import tarfile
import wget
from PIL import Image
import torch
from torch import tensor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
import numpy as np



# main folder path
MAIN_PATH = Path("/content/drive/MyDrive")

# complete path of "datasets" folder
DATASETS_PATH = MAIN_PATH / "Data Analysis" / "code" / "datasets"

# check the existence of folders and create missing ones
os.makedirs(DATASETS_PATH, exist_ok=True)

# image size CLIP
DEFAULT_SIZE = 224    
# image size ImagNet
DEFAULT_RESIZE = 256

IMAGENET_MEAN = tensor([.485, .456, .406])  
IMAGENET_STD = tensor([.229, .224, .225]) 
CLIP_MEAN = tensor([.481, .457, .408])
CLIP_STD = tensor([.268, .261, .275])

def mvtec_classes():
    return [
        "bottle",
        "cable",
        "capsule",
        "carpet_reduced",     # substitue with "carpet" if you have more computational power
        "grid_reduced",       # substitue with "grid" if you have more computational power
        "hazelnut_reduced",   # substitue with "hazelnut" if you have more computational power
        "leather",
        "metal_nut",
        "pill",
        "screw_reduced",      # substitue with "screw" if you have more computational power
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

# backbones definition
backbones = {
    'WideResNet':['WideResNet50'],
    'ResNet':['RN50', 'RN101'],
}

class MVTecDataset:
    def __init__(self, class_name : str, size : int = DEFAULT_SIZE, imgnet: bool = True):
        # definition of the MVTec dataset class
        self.class_name = class_name
        # default size of images setted as DEFAULT_SIZE
        self.size = size
        # if the value of class_name is present in the list of classes, call the function _download
        if class_name in mvtec_classes():
            self._download()

        # defines "train_dl" and "test_dl" equal to the MVTecTrain and Test classes
        self.train_ds = MVTecTrainDataset(class_name, size, imgnet)
        self.test_ds = MVTecTestDataset(class_name, size, imgnet)
    
    def get_datasets(self):
        return self.train_ds, self.test_ds

    # get_dataloaders returns the output of "DataLoader" fcn imported from torch.utils.data
    def get_dataloaders(self):
        return DataLoader(self.train_ds), DataLoader(self.test_ds)

    # _download() is responsible for downloading the compressed file (.tar.xz) and extracting 
    # the files contained within the compressed file into the destination directory if not already present.
    def _download(self):
        # check if the destination directory exists inside the folder defined in DATASETS_PATH
        if not isdir(DATASETS_PATH / self.class_name):
            print(f"   Could not find '{self.class_name}' in '{DATASETS_PATH}/'. Downloading ... ")
            url = self.get_link(self.class_name)
            wget.download(url)
            with tarfile.open(f"{self.class_name}.tar.xz") as tar:
                tar.extractall(DATASETS_PATH)
            # Remove the compressed file (.tar.xz) after the extraction
            os.remove(f"{self.class_name}.tar.xz")
            print("")
        else:
            print(f"   Found '{self.class_name}' in '{DATASETS_PATH}/', so no need to download {self.class_name}.tar.xz\n")

    def get_link(self, class_name: str):
        base_url_MVTec = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/"
        links_MVTec = {
          "bottle": "420937370-1629951468/bottle.tar.xz",
          "cable": "420937413-1629951498/cable.tar.xz",
          "capsule": "420937454-1629951595/capsule.tar.xz",
          # you can use it if you have more computational power instead of "carpet_reduced"
          "carpet": "420937484-1629951672/carpet.tar.xz",
          # you can use it if you have more computational power instead of "grid_reduced"
          "grid": "420937487-1629951814/grid.tar.xz",
          # you can use it if you have more computational power instead of "hazelnut_reduced"
          "hazelnut": "420937545-1629951845/hazelnut.tar.xz",
          "leather": "420937607-1629951964/leather.tar.xz",
          "metal_nut": "420937637-1629952063/metal_nut.tar.xz",
          "pill": "420938129-1629953099/pill.tar.xz",
          # you can use it if you have more computational power instead of "screw_reduced"
          "screw": "420938130-1629953152/screw.tar.xz",
          "tile": "420938133-1629953189/tile.tar.xz",
          "toothbrush": "420938134-1629953256/toothbrush.tar.xz",
          "transistor": "420938166-1629953277/transistor.tar.xz",
          "wood": "420938383-1629953354/wood.tar.xz",
          "zipper": "420938385-1629953449/zipper.tar.xz"
        }

        if class_name in links_MVTec:
            return base_url_MVTec + links_MVTec[class_name]
        else:
            return "Dataset not found."


class MVTecTrainDataset(ImageFolder):
    def __init__(self, class_name : str, size : int, imgnet: bool = True):
        # ImageNet image pre-processing
        if imgnet:
            transform = transforms.Compose([        
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
            ])     
        else: # CLIP image pre-processing
            transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ])

        # parameters
        super().__init__(
            root=DATASETS_PATH / class_name / "train",
            transform=transform)
        self.class_name = class_name
        self.size = size


class MVTecTestDataset(ImageFolder):
    def __init__(self, class_name : str, size : int, imgnet: bool = True):
        # ImageNet image and mask pre-processing
        if imgnet:
            transform = transforms.Compose([         # image transform
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
                transforms.ToTensor(), 
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
            ])
            target_transform = transforms.Compose([  # mask transform
                transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        else: # CLIP image and mask pre-processing
            transform  = transforms.Compose([        # image transform
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ])

            target_transform = transforms.Compose([  # mask transform
                transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
                convert_image_to_rgb, 
                transforms.ToTensor(),
            ])

        # parameters
        super().__init__(
            root=DATASETS_PATH / class_name / "test",
            transform=transform,
            target_transform = target_transform
        )
        self.class_name = class_name
        self.size = size
            
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
      
        if "good" in path: # nominal image
            target = Image.new('L', (self.size, self.size))
            sample_class = 0
        else: # anomalous image
            target_path = path.replace("test", "ground_truth")
            target_path = target_path.replace(".png", "_mask.png")
            target = self.loader(target_path)
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target[:1], sample_class 

# converts the image to a rgb
def convert_image_to_rgb(image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image