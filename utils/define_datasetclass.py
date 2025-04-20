import os
import random
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import yaml
from utils.transform_utils import transform_image, transform_mask

# Load config
with open('config.yaml', 'rt') as f:
    cfg = yaml.safe_load(f.read())
data_name = cfg['data']['name']

def class_reduction(mask, new_class):
    """
    Reduces the number of classes in a segmentation mask by combining several classes into fewer classes.
    warning! the combined depended on data set. for custom datac set insert which classes you want to merge
    Args:
        mask (PIL) : array representing the segmentation mask with class labels.
        new_class (int): The number of desired classes after reduction.
                         Determines how classes are combined:
                         - 4: Combine classes based on specific rules (for "landcover" dataset).
                         - 3: Combine classes into 3 classes, with special handling for class 5.
                         - 2: Combine all classes into 2 classes, with all classes except class 1 being set to 0.
        data_name (str): The name of the dataset, used to apply specific class reduction rules if `new_class` is 4.

    Returns:
    PIL: The updated mask with reduced classes.
    """
    mask = np.array(mask)
    if new_class == 4:
        if data_name == "landcover":
             mask[mask == 2] = 0  #agriculture to unknown
             mask[mask == 5] = 0  #water to unknown
             mask[mask == 6] = 3  #rangeland to barren_land
             mask[mask == 4] = 2  #forest is class number 2
    
        if data_name == "MiniFrance":
            mask[mask == 2] = 1     # Class 1: Urban areas
            mask[mask == 3] = 1  
            mask[mask == 0] = 0     # Class 0: Unknown
            mask[mask == 15] = 0    
            mask[mask == 14] = 0  
            mask[mask == 4] = 3     # Class 3: Landscape
            mask[mask == 5] = 3  
            mask[mask == 6] = 3  
            mask[mask == 7] = 3
            mask[mask == 8] = 3
            mask[mask == 9] = 3
            mask[mask == 12] = 3  
            mask[mask == 13] = 3
            mask[mask == 10] = 2    # Class 2: Forest
            mask[mask == 11] = 2          
            
    #back to PIL          
    mask = Image.fromarray(mask)
    return mask

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
    


