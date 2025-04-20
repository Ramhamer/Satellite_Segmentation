import os
import random
import torch
import math
import numpy as np
from PIL import Image
import datetime
from torchvision import transforms as T  
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import DataLoader
from utils.define_datasetclass import SegmentationDataset
from utils.image_utils import rgb_to_grey

def train_dir(model_name,criterion_name):
    """
    Description: Create a directory to save the training results for a given model.

    Args: model_name (str): Name of the model.

    Returns: None
    """
    date = datetime.datetime.now() + datetime.timedelta(hours=3)
    date = date.strftime("%b-%d-%Y_%H:%M")
    train_name = model_name + "_" + criterion_name + "_" + date
    save_dir = os.path.join('results',train_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,"checkpoints"), exist_ok=True)
    return save_dir

def create_metadata(cfg, save_dir):
    """
    Create a metadata dictionary for the training process.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        txt: A text file containing the metadata.
    """
    # Create a metadata dictionary
    metadata = {
        'Date': datetime.datetime.now().strftime('%d/%m/%Y'),
        'Time': (datetime.datetime.now() + datetime.timedelta(hours=3)).strftime('%H:%M:%S'),
        'Data_directory': cfg['data']['dir'],
        'Desirable_class': cfg['train']['desirable_class'],
        'Model_name': cfg['model']['model_name'],
        'Encoder': cfg['model']['encoder_name'],
        'Encoder_weights': cfg['model']['encoder_weights'],
        'Activation_function': cfg['model']['activation'],
        'Pooling': cfg['model']['pooling'],
        'Dropout': cfg['model']['dropout'],
        'Loss_function': cfg['loss']['name'],
        'Optimizer': cfg['train']['optimizer_name'],
        'Learning_rate': cfg['train']['lr'],
        'Weight_decay': cfg['train']['weight_decay'],
        'Num_epochs': cfg['train']['num_epochs'],
        'Batch_size': cfg['train']['batch_size'],
        'Augmentations': cfg['transformes']['types'],
        'Early_stopping(epsilon)': cfg['loss']['eps'],
    }

    # Save the metadata to a text file
    with open(os.path.join(save_dir, 'metadata.txt'), 'w') as file:
        file.write("Training Metadata:\n")
        file.write("-------------------------------------\n")
        for key, value in metadata.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        file.write("-------------------------------------\n")
        file.write("\n")
        file.write("Class Distribution:\n")
        file.write(f"{cfg['train']['class_distribution']}\n")
    return

def get_directories(base_dir: str ,test_mode=None):
    """
    Construct and return the paths to directories for training and validation images and annotations.

    Args:
        base_dir (str): Base directory where the dataset is located.

    Returns:
        tuple: A tuple containing four paths:
            - Path to the training images directory.
            - Path to the training annotations directory.
            - Path to the validation images directory.
            - Path to the validation annotations directory.
    """
    if test_mode:
        image_train_dir = os.path.join(base_dir, 'test', 'images')
        mask_train_dir = os.path.join(base_dir, 'test', 'masks')
        image_val_dir = os.path.join(base_dir, 'test', 'images')
        mask_val_dir = os.path.join(base_dir, 'test', 'masks')
        return image_train_dir, mask_train_dir, image_val_dir, mask_val_dir
        
    image_train_dir = os.path.join(base_dir, 'train', 'images')
    mask_train_dir = os.path.join(base_dir, 'train', 'masks')
    image_val_dir = os.path.join(base_dir, 'val', 'images')
    mask_val_dir = os.path.join(base_dir, 'val', 'masks')
    return image_train_dir, mask_train_dir, image_val_dir, mask_val_dir

def select_transform(config,test_mode=None):
    """
    Select and return image transformations for training and validation based on the given configuration.

    Args:
        config (dict): Configuration dictionary specifying which transformations to apply.
        test_mode (bool, optional): If True, return an empty list of transformations for validation. Defaults to None.

    Returns:
        tuple: A tuple containing two lists:
            - List of transformations to apply during training.
            - List of transformations to apply during validation (usually empty if test_mode is True).
    """
    if test_mode:
        train_transform = []
        return train_transform
    
    my_dict = {'horizontal_flip' :T.RandomHorizontalFlip(p=1),
               'vertical_flip': T.RandomVerticalFlip(p=1),
               'ColorJitter':T.ColorJitter(brightness=random.uniform(0.9, 1.1),contrast=random.uniform(0.9, 1.1),saturation=random.uniform(0.9, 1.1))}
    
    train_transform = [my_dict[key] for key in my_dict if config['transformes']['types'][key] is True]
    
    return train_transform

def load_data(cfg,desirable_class,batch_size,data_dir,test_mode=None):
    """
    Load the training and validation data using the given configuration.

    Args: 
        cfg (dict): Configuration dictionary.
        desirable_class (int): Number of classes to reduce the masks to.
        batch_size (int): Batch size for the DataLoader.
        data_dir (str): Path to the directory containing the dataset.
        test_mode (bool): If True, load the test data. Defaults to False.

    Returns: 
        tuple: A tuple containing two DataLoader objects:
            - DataLoader for the training data.
            - DataLoader for the validation data.
    """

    #Select the Augmentation
    train_transform  = select_transform(cfg,test_mode)

    #get the data directories
    image_train_dir, mask_train_dir, image_val_dir, mask_val_dir = get_directories(data_dir,test_mode)

    train_dataset = SegmentationDataset(
        image_dir=image_train_dir,
        mask_dir=mask_train_dir, 
        transform = train_transform,
        number_class = desirable_class)

    val_dataset = SegmentationDataset(
        image_dir=image_val_dir, 
        mask_dir=mask_val_dir, 
        transform = [],
        number_class = desirable_class)

    #DataLoader
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4,drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4,drop_last=False, pin_memory=True)

    return train_loader, val_loader



####image handeling#######

def split_image(image, target_size=(1024, 1024)):
    """ 
    Splits an image into patches with padding and saves them in a temporary directory.
    
    Args:
    - image (Tensor): Image to split.
    - target_size (tuple): Target size of the patches (width, height).

    Returns:
    - Tensor: Tensor with shape (num_patches, channels, height, width) containing the image patches.
    """
    
    # Image dimensions
    channels, height, width = image.shape

    # Calculate the number of patches in each dimension
    num_patches_x = math.ceil(width / target_size[0])
    num_patches_y = math.ceil(height / target_size[1])
    num_patches = num_patches_x * num_patches_y
    main_tensor = torch.zeros((num_patches,channels, target_size[1], target_size[0]))


    # Split the image into patches with padding
    i=0
    for y in range(0, height, target_size[0]):
        for x in range(0, width, target_size[1]):
            # Calculate patch boundaries
            h_end = min(y + target_size[0], height)
            w_end = min(x + target_size[1], width)
            
            # Extract patch
            patch = image[:, y:h_end, x:w_end]
            
            # Place patch in output tensor
            main_tensor[i, :, :patch.shape[1], :patch.shape[2]] = patch
            i += 1

    return main_tensor

def rebuild_image(patches, original_size):
    """
    Rebuilds an image from patches with padding.

    Args:
    - image (torch.Tensor): Image tensor containing patches.
    - original_size (tuple): Original size of the image (width, height).

    Returns:
    - torch.Tensor: Reconstructed image tensor.
    """
    #vars
    num_patches = patches.shape[0]
    channels = patches.shape[1]
    patch_height, patch_width = patches.shape[2:]

    # Calculate grid dimensions
    grid_y = math.ceil(original_size[1] / patch_height)
    grid_x = math.ceil(original_size[2] / patch_width)

    # Initialize output tensor
    temp_height = grid_y * patch_height
    temp_width = grid_x * patch_width
    output = torch.zeros((channels, temp_height, temp_width))
    
    # Reconstruct image from patches
    i = 0
    for y in range(0, temp_height, patch_height):
        for x in range(0, temp_width, patch_width):
            if i >= num_patches:
                break
            output[:, y:y+patch_height, x:x+patch_width] = patches[i]
            i += 1
        if i >= num_patches:
            break
            
    # Crop to original size
    output = output[:, :original_size[1], :original_size[2]]
    
    return output


def calculate_one_percent(directory,train_percents=0.8,val_percents=0.1):
    """
    Calculate the number of files for training, validation, and testing based on given percentages.
    Args:
        directory (str): The directory containing the files.
        train_percents (float, optional): The percentage of files to be used for training. Defaults to 0.8.
        val_percents (float, optional): The percentage of files to be used for validation. Defaults to 0.1.
    Returns : None  
    """

    try:
        # Count the number of files
        num_files = len(os.listdir(directory))
        # Calculate percents
        train_num = int(num_files * train_percents)
        val_num = int(num_files * val_percents)
        test_num = num_files - train_num - val_num
        return [train_num, val_num, test_num]
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def divide_images(images_path,new_path,train_percents=0.8,val_percents=0.1):
    """
    Divides images from a given directory into training, validation, and test sets based on specified percentages.
    Args:
        images_path (str): The path to the directory containing the images to be divided.
        new_path (str): The path to the directory where the divided images will be stored.
        train_percents (float, optional): The percentage of images to be used for training. Default is 0.8.
        val_percents (float, optional): The percentage of images to be used for validation. Default is 0.1.
    Returns:
        None
    """
    try:
        # Get the list of files
        files = os.listdir(images_path)

        # Calculate the number of files
        sizes = calculate_one_percent(images_path,train_percents,val_percents)

        #make 3 dir for train, val, test
        os.makedirs(os.path.join(new_path, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'train', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'val', 'images'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'val', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'test', 'masks'), exist_ok=True)

        # Divide the data randomaly into train, val, and test and remove them to the corresponding folders
        for i in range(sizes[0]):
            file = random.choice(files)
            os.rename(os.path.join(images_path, file), os.path.join(new_path, 'train', 'images', file))
            files.remove(file)
        for i in range(sizes[1]):
            file = random.choice(files)
            os.rename(os.path.join(images_path, file), os.path.join(new_path, 'val', 'images', file))
            files.remove(file)
        for file in files:
            os.rename(os.path.join(images_path, file), os.path.join(new_path, 'test', 'images', file))
        
        print("The images has been divided successfully")

    except Exception as e:
        print(f"An error occurred: {e}")

def divide_masks(masks_path,new_path):
    """
    Divide masks from a given directory into training, validation, and test sets based on the images directory.

    Args:masks_path (str): The path to the directory containing the masks to be divided.
        new_path (str): The path to the directory where the divided masks will be stored.

    Returns: None
    """
    try:
        #dirs for images
        train_img_path = os.path.join(new_path, 'train', 'images')
        val_img_path = os.path.join(new_path, 'val', 'images')
        test_img_path = os.path.join(new_path, 'test', 'images')

        # Get the list of files
        masks_files = os.listdir(masks_path)
        train_files = os.listdir(train_img_path)
        val_files = os.listdir(val_img_path)
        test_files = os.listdir(test_img_path)

        # Get the common files between the masks and images
        train_files = list(set(masks_files) & set(train_files))
        val_files = list(set(masks_files) & set(val_files))
        test_files = list(set(masks_files) & set(test_files))

        #move the common files to the train and val dirs
        for file in train_files:
            os.rename(os.path.join(masks_path, file), os.path.join(new_path, 'train', 'masks', file))
            masks_files.remove(file)
        for file in val_files:
            os.rename(os.path.join(masks_path, file), os.path.join(new_path, 'val', 'masks', file))
            masks_files.remove(file)
        for file in test_files:
            os.rename(os.path.join(masks_path, file), os.path.join(new_path, 'test', 'masks', file))
            masks_files.remove(file)
        print("The masks has been divided successfully")

    except Exception as e:
        print(f"An error occurred: {e}")

    return

def divide_data(data_path = None,new_path = None):
    """
    Divide the data into training, validation, and test sets and save them in a new directory.

    Args:
        data_path (str): The path to the directory containing the data to be divided.
        new_path (str): The path to the directory where the divided data will be stored.

    Returns: None
    """
    images_path = os.path.join(data_path, 'images')
    masks_path = os.path.join(data_path, 'masks')
    divide_images(images_path,new_path)
    divide_masks(masks_path,new_path)
    return



#### cvat convertor ####
def parse_color_file(base_dir):
    """
    Parse the labelmap.txt file and return a dictionary mapping class labels to RGB colors.

    Args:
        base_dir (str): The base directory where the labelmap.txt file is located.
    
    Returns:
        dict: A dictionary mapping class labels to RGB colors.
    """
    txt_path = os.path.join(base_dir, 'labelmap.txt')
    label_dict = {}
    with open(txt_path, 'r') as file:
        next(file)  # Skip the first line
        for line in file:
            if line.strip() :  # Ignore empty lines
                label, color, *_ = line.split(':')
                color_rgb = np.array([*map(int, color.split(','))])
                label_dict[label] = color_rgb
    return label_dict

def fetch_masks(base_dir):
    """
    Fetch the masks from the dataset and save them as grayscale images.
    
    Args:
        base_dir (str): The base directory where the dataset is located.
    
    Returns:
        list: A list of the names of the mask files.
    """
    # Get the current date
    date = datetime.now().strftime("%b-%d-%Y_%H:%M")

    # where the masks are located
    masks_path = os.path.join(base_dir, 'SegmentationClass')

    # create a directory to save the annotations
    annotations_dir = os.path.join(date ,'annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    # Get the list of files
    masks_files = os.listdir(masks_path)

    # Get the color dictionary
    label_dict =parse_color_file(base_dir)

    # Convert the masks to grayscale and save them
    for file in masks_files:
        mask = np.array(Image.open(os.path.join(masks_path, file)))
        grey_mask = Image.fromarray(rgb_to_grey(mask, label_dict))
        grey_mask.save(os.path.join(annotations_dir, file))
    return masks_files

def fetch_images(base_dir,masks_files):
    """
    Fetch the images from the dataset and save them in a new directory.

    Args:
        base_dir (str): The base directory where the dataset is located.
        masks_files (list): A list of the names of the mask files.
    
    Returns: None
    """
    # Get the current date
    date = datetime.now().strftime("%b-%d-%Y_%H:%M")
    
    # where the images are located
    images_path = os.path.join(base_dir, 'JPEGImages')

    # create a directory to save the images
    images_dir = os.path.join(date ,'images')
    os.makedirs(images_dir, exist_ok=True)

    # Move the images to the new directory
    for file in masks_files:
        file_path = os.path.join(images_path, file)
        
        #open the image and save it in the new directory
        image = Image.open(file_path)
        image.save(os.path.join(images_dir, file))
    return

def pull_data(base_dir):
    """
    Pull the images and masks from the dataset and save them in a new directory.

    Args:
        base_dir (str): The base directory where the dataset is located.

    Returns: None
    """
    # Fetch the masks and images
    masks_files = fetch_masks(base_dir)
    fetch_images(base_dir, masks_files)
    
    # Print a success message
    print("Data has been pulled successfully")
    return

if __name__ == "__main__":
    image_tensor = Image.open('/workspace/results/606_mask.png')
    image_array = np.array(image_tensor)

    main_tensor= split_image(image_array)
    new_tensor = rebuild_image(main_tensor,image_array.shape) 
    print(image_array.size -(new_tensor==image_array.sum()))