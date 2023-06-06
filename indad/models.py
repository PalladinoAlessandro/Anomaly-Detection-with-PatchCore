from typing import Tuple, List
from tqdm import tqdm
import torch
from torch import tensor
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import clip 
from PIL import Image, ImageDraw, ImageOps, ImageEnhance  # needed for CLIP
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params, get_output_folder, inverse_transform
from data import IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD



class PatchCore(torch.nn.Module):
  def __init__(
    self,
    f_coreset: float = 0.01,   # fraction the number of training samples
    coreset_eps: float = 0.90, # SparseProjector parameter
    image_size: int = 224,
    imgnet: bool = False,      # parameter added from the input
    backbone_name: str = ""    # parameter added from the input
  ):
    super(PatchCore, self).__init__()

    # hook to extract feature maps
    def hook(module, input, output) -> None:
        """This hook saves the extracted feature map on self.featured."""
        self.features.append(output)

    # register hooks
    if imgnet:
        self.model = torch.hub.load('pytorch/vision:v0.13.0', 'wide_resnet50_2', pretrained=True)
        self.model.layer2[-1].register_forward_hook(hook)            
        self.model.layer3[-1].register_forward_hook(hook)            
    else:
        self.model, _ = clip.load(backbone_name, device="cpu", jit=False, download_root=None)
        self.model.visual.layer2[-1].register_forward_hook(hook)
        self.model.visual.layer3[-1].register_forward_hook(hook)

    # disable gradient computation
    self.model.eval()
    for param in self.model.parameters():
        param.requires_grad = False

    # parameters
    self.f_coreset = f_coreset          # fraction rate of training samples
    self.coreset_eps = coreset_eps      # SparseProjector parameter
    self.imgnet = imgnet                # True: imgnet, False: CLIP
    self.backbone_name = backbone_name
    self.image_size = image_size
    self.average = torch.nn.AvgPool2d(3, stride=1)
    self.blur = GaussianBlur(4)
    self.n_reweight = 3

    self.patch_lib = []
    self.resize = None

  def forward(self, sample: tensor):
    """
      Initialize self.features and let the input sample passing
      throught the backbone net self.model.
      The registered hooks will extract the layer 2 and 3 feature maps.
      Return:
        self.feature filled with extracted feature maps
    """
    self.features = []
    if self.imgnet:
      _ = self.model(sample)
    else:
      _ = self.model.visual(sample)  # CLIP

    return self.features

  # "fit" has in input "train_dl" containing the training data and a scale factor set by default as 1 
  # since the backbone we use do not need a different scaling
  def fit(self, train_dl):
    '''
     The code iterates through the training data and calculates
     a patch tensor representing the resized feature maps for each sample.
     A progress bar is displayed using the tqdm module and the get_tqdm_params dictionary

    '''

    for sample, _ in tqdm(train_dl, **get_tqdm_params()):                                                                       # ** needed for the dictionary
      feature_maps = self(sample)                                                                                               # for each sample calculates fmaps representing the activations of different features
      largest_fmap_size = feature_maps[0].shape[-2:]                                                                            # evaluates the largest fmap to get the desired output size
      self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)                                                               # to be recalled in "predict" to resize the map
      resized_maps = [torch.nn.functional.adaptive_avg_pool2d(self.average(fmap), largest_fmap_size) for fmap in feature_maps]  # fcn to resize the fmaps
      patch = torch.cat(resized_maps, 1)                                                                                        # resized fmaps are summed along axis 1 with "torch.cat" to get 1 tensor (patch)
      patch = patch.reshape(patch.shape[1], -1).T                                                                               # resize "patch" to get compatible dimensions with the next iterazion 
      self.patch_lib.append(patch)                                                                                              # adding "patch" at each step to this list
    self.patch_lib = torch.cat(self.patch_lib, 0)                                                                               # sums all the tensor along axis 0 to get a final tensor

    if self.f_coreset < 1:
      self.coreset_idx = get_coreset_idx_randomp(        # returns the patch indexes selected as coreset
        self.patch_lib,
        n=int(self.f_coreset * self.patch_lib.shape[0]), # number of patches to select as coreset
        eps=self.coreset_eps,
      )
      self.patch_lib = self.patch_lib[self.coreset_idx]  # reduces the patches only to the coreset patches


  def evaluate(self, test_dl: DataLoader, database_name: str, backbone_name: str, size: int, imgnet: bool) -> Tuple[float, float]:
    image_preds = []
    image_labels = []
    pixel_preds = []
    pixel_labels = []

    output_folder = get_output_folder(database_name, backbone_name)


    for i, (sample, mask, label) in enumerate(tqdm(test_dl, **get_tqdm_params())):
        z_score, fmap = self.predict(sample)  # anomaly detection

        image_preds.append(z_score.numpy())
        image_labels.append(label)
        pixel_preds.extend(fmap.flatten().numpy())
        pixel_labels.extend(mask.flatten().numpy())

        # inverse the sample to an RGB image 
        sample_img = inverse_transform(sample, imgnet, size)

        # saving the image with anomalous area highlighted
        output_path = os.path.join(output_folder, f'test_image_{i}.png')
        composite_img = self.create_highlighted_image(sample_img, mask)
        composite_img.convert("RGB").save(output_path)


    image_labels = np.stack(image_labels)
    image_preds = np.stack(image_preds)

    # compute ROC AUC for prediction scores
    image_rocauc = roc_auc_score(image_labels, image_preds)
    pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)

    n_samples = len(test_dl.dataset)
    n_pixels = test_dl.dataset[0][1].numel()

    pixel_preds_mat = np.zeros((n_samples, n_pixels))

    for i, (sample, mask, label) in enumerate(test_dl):
        z_score, fmap = self.predict(sample)
        pixel_preds = fmap.flatten().numpy()
        pixel_preds_mat[i, :] = pixel_preds

    return image_rocauc, pixel_rocauc


  def predict(self, sample):		
    feature_maps = self(sample)
    resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
    patch = torch.cat(resized_maps, 1)
    patch = patch.reshape(patch.shape[1], -1).T

    dist = torch.cdist(patch, self.patch_lib)
    min_val, min_idx = torch.min(dist, dim=1)
    s_idx = torch.argmax(min_val)
    s_star = torch.max(min_val)

    # reweighting
    m_test = patch[s_idx].unsqueeze(0)                               # anomalous patch
    m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)             # closest neighbour
    w_dist = torch.cdist(m_star, self.patch_lib)                     # find knn to m_star pt.1
    _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False) # pt.2
    
    # equation 7 from the paper
    m_star_knn = torch.linalg.norm(m_test-self.patch_lib[nn_idx[0,1:]], dim=1)
    
    # softmax normalization trick as in transformers.
    # as the patch vectors grow larger, their norm might differ a lot.
    # exp(norm) can give infinities.
    D = torch.sqrt(torch.tensor(patch.shape[1]))
    w = 1-(torch.exp(s_star/D)/(torch.sum(torch.exp(m_star_knn/D))))
    s = w*s_star

    # segmentation map
    s_map = min_val.view(1,1,*feature_maps[0].shape[-2:])
    s_map = torch.nn.functional.interpolate(
      s_map, size=(self.image_size,self.image_size), mode='bilinear'
    )
    s_map = self.blur(s_map)

    return s, s_map


  # used in fit() to get the loading-bar while running
  def get_parameters(self, extra_params : dict = None) -> dict:
    return {
      "backbone_name": self.backbone_name,
      "f_coreset": self.f_coreset,
      "n_reweight": self.n_reweight,
    }


  def create_highlighted_image(self, sample_img, mask):
    mask = mask.squeeze().numpy()
    sample_arr = np.array(sample_img)

    boundary_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    boundary_mask = (boundary_mask[1:-1, :-2] + boundary_mask[1:-1, 2:] +
                    boundary_mask[:-2, 1:-1] + boundary_mask[2:, 1:-1] - 4 * boundary_mask[1:-1, 1:-1]) > 0

    interior_mask = np.logical_and(mask > 0.5, np.logical_not(boundary_mask))

    highlight_img = Image.new("RGBA", sample_img.size, (0, 0, 0, 0))
    highlight_img.paste((255, 255, 0, 40), mask=Image.fromarray(interior_mask.astype(np.uint8) * 255))

    sample_arr_with_border = np.copy(sample_arr)
    sample_arr_with_border[boundary_mask] = [255, 0, 0]

    sample_img_with_alpha = sample_img.convert("RGBA")
    sample_img_with_alpha.putalpha(0)

    sample_arr_with_border_resized = sample_arr_with_border[:sample_img.size[1], :sample_img.size[0]]
    sample_img_with_alpha_resized = sample_img_with_alpha.resize(sample_arr_with_border_resized.shape[:2][::-1])

    sample_arr_with_border_rgba = np.concatenate(
        (sample_arr_with_border_resized, np.full(sample_arr_with_border_resized.shape[:2] + (1,), 255, dtype=np.uint8)),
        axis=2)

    sample_arr_with_border_image = Image.fromarray(sample_arr_with_border_rgba, 'RGBA')

    composite_img = Image.alpha_composite(sample_img_with_alpha_resized, sample_arr_with_border_image)
    composite_img = Image.alpha_composite(composite_img, highlight_img)

    return composite_img