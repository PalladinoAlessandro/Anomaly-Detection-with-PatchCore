import os
import sys
import yaml
from tqdm import tqdm
from datetime import datetime
import torch
from torch import tensor
from torchvision import transforms
import PIL
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageOps, ImageEnhance  # needed for CLIP
from sklearn import random_projection
from data import mvtec_classes, DATASETS_PATH, IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD



def get_output_folder(dataset_name, backbone_type):
    '''
    Starting from dataset name and backbone the function returns
    the path to save the images in which are highlighted the anomalous pixels
    '''

    base_folder = DATASETS_PATH / dataset_name
    backbone_folder = base_folder / f"output_{backbone_type}"
    
    if not backbone_folder.exists():
        os.makedirs(backbone_folder)    # create the folder if it does not exist
    
    return backbone_folder


def inverse_transform(img, imgnet=True, size=256):

    if imgnet:
        inverse_transform = transforms.Compose([
            transforms.Normalize((-IMAGENET_MEAN / IMAGENET_STD).tolist(), (1.0 / IMAGENET_STD).tolist()),
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
            transforms.ToPILImage(),
            transforms.Resize(size),
        ])
    else:
        inverse_transform = transforms.Compose([
            transforms.Normalize((-CLIP_MEAN / CLIP_STD).tolist(), (1.0 / CLIP_STD).tolist()),
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
            transforms.ToPILImage(),
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        ])

    # Reshape the input image to remove the batch dimension if present
    img = img.squeeze(0)

    img = inverse_transform(img)

    # Restore saturation
    enhancer = ImageEnhance.Color(img)
    sample_with_saturation = enhancer.enhance(1.2)

    # Restore contrast
    enhancer = ImageEnhance.Contrast(sample_with_saturation)
    sample_with_saturation_contrast = enhancer.enhance(3.5)

    # Restore sharpness
    enhancer = ImageEnhance.Sharpness(sample_with_saturation_contrast)
    sample_with_saturation_contrast_sharpness = enhancer.enhance(3)

    # Restore brightness
    enhancer = ImageEnhance.Brightness(sample_with_saturation_contrast_sharpness)
    sample_with_saturation_contrast_sharpness_brightness = enhancer.enhance(0.5)

    # Convert the resulting image to a numpy array format
    img_result = np.array(sample_with_saturation_contrast_sharpness_brightness)
    
    # Convert the resulting image from numpy array to PIL.Image oboject
    img_result = Image.fromarray(img_result)
    
    return img_result


def get_tqdm_params():
    '''
    Returns a dictionary used as an argument for tqdm in the 
    "model.py" file to display a progress bar.

    The dictionary defines the settings of this bar.
    '''

    TQDM_PARAMS = {
	  "file" : sys.stdout,
	  "bar_format" : "   {l_bar}{bar:10}{r_bar}{bar:-10b}",
    }
    return TQDM_PARAMS


class GaussianBlur:
    def __init__(self, radius : int = 4):
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(
            self.unload(img[0]/map_max).filter(self.blur_kernel)
        )*map_max
        return final_map


# The following section selects a representative subset coreset 
# using a sparse random projection algorithm and Euclidean distance.
def get_coreset_idx_randomp(
    z_lib : tensor, 
    n : int = 1000,
    eps : float = 0.90,
    float16 : bool = True,
    force_cpu : bool = False,
) -> tensor:

    """Returns n coreset idx for given z_lib.

    Args:
        z_lib:      (n, d) tensor of patches.
        n:          Number of patches to select.
        eps:        Agression of the sparse random projection.
        float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
        force_cpu:  Force cpu, useful in case of GPU OOM.

    Returns:
        coreset indices
    """
    print(f"   Fitting random projections. Start dim = {z_lib.shape}.")

    # initialize a sparse random projection transformer with an aggrssivity parameter eps
    try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        # z_lib tensor dimension is reduced according to the random projection 
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
    except ValueError:
        print( "   Error: could not project vectors. Please increase `eps`.")

    select_idx = 0
    last_item = z_lib[select_idx:select_idx+1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True) # min dist between last_item and all the patches in z_lib using Euclidean norm 

    # the line below is not faster than linalg.norm, it is kept for reference.
    # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

    # check if tensors cast to float16 it's needed
    if float16:                                       # if yes converts in half that represents data type with precision reduced to 16 bit
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()

    # check if there is an available GPU and if "force_cpu" is not True
    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")              # moves to GPU using the "cuda" method
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in tqdm(range(n-1), **get_tqdm_params()):
        distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True)         # broadcasting step
        # distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True) # broadcasting step
        min_distances = torch.minimum(distances, min_distances)                      # iterative step
        select_idx = torch.argmax(min_distances)                                     # selection step (max argument in "min_distances" -> max dist)

        # bookkeeping
        last_item = z_lib[select_idx:select_idx+1]                                   # update last_item value by selecting the tensor corresponding to the idx from "z_lib"  
        min_distances[select_idx] = 0                                                # this assure that the selected point will not be considered in the next iterations
        coreset_idx.append(select_idx.to("cpu"))                                     # add idx to the coreset list converted to CPU
        # at the end "coreset_idx" will contain the selected point idx for the coreset
    return torch.stack(coreset_idx)


def print_and_export_results(results : dict, backbone_name : str):

    """Writes results to .yaml and serialized results to .txt."""
    method = "Patchcore"

    # write
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    name = f"{method}_{backbone_name}_{timestamp}"

    results_yaml_path = f"./results/{name}.yml"
    scoreboard_path = f"./results/{name}.txt"

    with open(results_yaml_path, "w") as outfile:
        yaml.safe_dump(results, outfile, default_flow_style=False)
    with open(scoreboard_path, "w") as outfile:
        outfile.write(serialize_results(results["per_class_results"]))
        
    print(f"   Results written to {results_yaml_path}")

def serialize_results(results : dict) -> str:

    """Serialize a results dict into something usable in markdown."""
  
    n_first_col = 20
    ans = []
    for k, v in results.items():
        s = k + " "*(n_first_col-len(k))
        s = s + f"| {v[0]*100:.1f}  | {v[1]*100:.1f}  |"
        ans.append(s)
    return "\n".join(ans)

def display_backbones():
    imgnet = True
    print("ImageNet PatchCore backbone:")
    print(f"- WideResNet50")
    print("CLIP Image Encoder architectures for PatchCore backbone:")
    for k, _ in backbones.items():
        if imgnet:
            imgnet = False
            continue
        print(f"- {k}")
    print()
    
def display_MVTec_classes():
    print(mvtec_classes())