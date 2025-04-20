import random
import torchvision.transforms as T

def transform_image(image, lst):
    """
    Applies a series of transformations to an image.

    Args:
        image (PIL.Image.Image): The input image to be transformed.
        lst (list of torchvision.transforms): A list of torchvision transform objects.

    Returns:
        Tensor: The transformed image.
        image_lst (list of torchvision.transforms): The list of transformations that were applied to the image.
    """
    image_lst = [T.ToTensor(), T.Normalize(mean=[0.4089, 0.3797, 0.2822], std=[0.1462, 0.1143, 0.1049])]
    for i in range(len(lst)):
        random_bol = random.randint(0, 1)
        if random_bol:
            image_lst.append(lst[i])
    curr_transform = T.Compose(image_lst)
    image = curr_transform(image)
    return image, image_lst

def transform_mask(mask, transforms_list):
    """
    Applies a series of transformations to a segmentation mask, excluding transformations that are not suitable for masks.

    Args:
        mask (PIL.Image.Image): The input mask to be transformed. 
        transforms_list (list of torchvision.transforms): A list of torchvision transform objects. Transformations that are not suitable for masks
                                                          (e.g., `ColorJitter`, `Normalize`, `ToTensor`) are filtered out.

    Returns:
        torch.Tensor: The transformed mask. If the input mask was a NumPy array.
    """
    types_to_remove = (T.ColorJitter, T.Normalize, T.ToTensor)
    filtered_transforms = [item for item in transforms_list if not isinstance(item, types_to_remove)]
    filtered_transforms.append(T.PILToTensor())
    curr_transform = T.Compose(filtered_transforms)
    mask = curr_transform(mask)
    return mask 