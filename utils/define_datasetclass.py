import os
import random
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from utils.image_utils import *

def transform_image(image,lst):
    """
    Applies a series of transformations to an image.

    Args:
        image (PIL.Image.Image): The input image to be transformed.
        lst (list of torchvision.transforms): A list of torchvision transform objects.

    Returns:
        Tensor: The transformed image.
        image_lst (list of torchvision.transforms): The list of transformations that were applied to the image.
    """
    image_lst = [T.ToTensor(),T.Normalize(mean=[0.4089, 0.3797, 0.2822], std=[0.1462, 0.1143, 0.1049])]
    for i in range(len(lst)):
        random_bol = random.randint(0, 1)
        if random_bol:
            image_lst.append(lst[i])
    curr_transform = T.Compose(image_lst)
    image = curr_transform(image)
    return image , image_lst   

def transform_mask(mask,transforms_list):
    """
    Applies a series of transformations to a segmentation mask, excluding transformations that are not suitable for masks.

    Args:
        mask (PIL.Image.Image): The input mask to be transformed. 
        transforms_list (list of torchvision.transforms): A list of torchvision transform objects. Transformations that are not suitable for masks
                                                          (e.g., `ColorJitter`, `Normalize`, `ToTensor`) are filtered out.

    Returns:
        torch.Tensor: The transformed mask. If the input mask was a NumPy array.

   """
    types_to_remove = (T.ColorJitter,T.Normalize,T.ToTensor)
    filtered_transforms = [item for item in transforms_list if not isinstance(item, types_to_remove)]
    filtered_transforms.append(T.PILToTensor())
    curr_transform = T.Compose(filtered_transforms)
    mask = curr_transform(mask)
    return mask

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, number_class=7):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)
        self.number_class = number_class
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        # Get image and mask paths
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # class reduction
        if self.number_class != 7:
            mask = class_reduction(mask,self.number_class)

        # Apply transformations
        image,transforms_list = transform_image(image,self.transform)
        mask = transform_mask(mask,transforms_list)
        
        return image, mask,self.image_names[idx]
    


