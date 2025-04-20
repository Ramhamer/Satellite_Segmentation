import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import utils.data_utils
import yaml
from PIL import Image
import cv2
import webcolors
from shapely.geometry import Polygon , MultiPolygon
import geopandas as gpd

#handle config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
with open('config.yaml', 'rt') as f:
        cfg = yaml.safe_load(f.read())
data_name = cfg['data']['name']

def crop_image(image):
    """
    Crops an image to a NxN pixel area, centered on the original image.

    Args:
        image (PIL.Image.Image): The input image to be cropped. It must be a valid image object loaded using the Pillow library.

    Returns:
        PIL.Image.Image: The cropped image with a size of 256x256 pixels, centered around the middle of the original image.
    """
    width, height = image.size
    left = (width - 256) // 2
    top = (height - 256) // 2
    right = (width + 256) // 2
    bottom = (height + 256) // 2
    image = image.crop((left, top, right, bottom))
    return image 

def add_mask(original_image,mask,alpha=0.15):
    """
    Overlays a color-encoded mask on an original image by blending the two, with padding if necessary.

    Args: 
        original_image: The original image to overlay the mask on.
        mask: The color mask to overlay on the original image.
        alpha: The blending factor, which determines the transparency of the mask. 
                       A value of 0 results in the original image, while a value of 1 results in the mask image.
    Returns:
        np.ndarray: The blended image, where the mask has been applied on top of the original image.
                    The image is returned as a NumPy array with values in the range [0, 255].

    """

    # Read the original image and the color mask

    original_matrix = np.array(original_image)
    if original_matrix.shape != mask.shape:
        pad = (mask.shape[0]-original_matrix.shape[0])//2
        original_matrix = np.pad(original_matrix, ((pad,pad), (pad, pad), (0, 0)) , mode='constant', constant_values=0)
   
    # Blend the original image and the color mask
    blended_image = (1 - alpha) * original_matrix + alpha *mask
    # Ensure the blended image has valid pixel values in the range [0, 255]
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image

def add_class_label_to_image(image,color_dict):
    '''
    This function takes the image and the class dictionary and add the class label to the image
    
    Args:
        image : np.array : the image array
        class_dict : dict : the class dictionary
    return : np.array : the image array with class label
    '''
    class_dict = class_label()

    # Create a dictionary that the key is from class dict and value is the color_dict_value
    final_dict = {list(class_dict.values())[i] : webcolors.rgb_to_name(color_dict[(list(class_dict.keys()))[i]])for i in range(len(class_dict))}
    final_dict["Urban"] = "green"

    # Create a copy of the image to draw on
    overlay = image.copy()  
    h, w, _ = image.shape
    legend_w, legend_h = int(0.22 * w), int(0.105 * h)  # 25% width, 15% height

    # Define the top-left corner of the legend
    x1, y1 = w - legend_w - 30, h - legend_h - 20  # Bottom-right corner offset
    x2, y2 = w - 30, h - 20  # Bottom-right corner of the box

    # Draw semi-transparent grey shadow
    shadow_color = (100, 100, 100)  # Dark grey
    alpha = 0.3
    cv2.rectangle(overlay, (x1, y1), (x2, y2), shadow_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Adaptive font size and spacing based on image size
    font_scale = 5  # Scale text based on image width
    thickness = 5  # Scale thickness based on image size
    spacing = max(30, int(0.05 * legend_h))  # Scale spacing dynamically

    # Add class colors and names
    y_offset = y1 +170
    for class_name in final_dict.keys():
        cv2.putText(image, f"{class_name}  :  {final_dict[class_name]}", (x1 + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        y_offset += 250
    return image

def class_reduction(mask,new_class):
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

def class_label(data_name=data_name):
    """
    retuen the class label of the data set
    """
    if data_name == "MiniFrance":
        class_labels = {0: "Landscape", 1: "Urban", 2: "Forest", 3: "Unknown"}


    return class_labels
    
def one_hot(masks,desirable_class):
    """
    Converts a segmentation mask with class labels into a one-hot encoded tensor with a specified number of classes.

    Args:
        mask (torch.Tensor): A 3D tensor representing the segmentation mask with class labels.
        new_class (int): The number of classes for the one-hot encoding.

    Returns:
        torch.Tensor: A 4D tensor representing the one-hot encoded mask, with dimensions [batch_size, num_classes, height, width].
    """

    # Ensure the mask is of type Long for one-hot encoding
    masks = masks.long()  # [batch_size, 1, H, W]
    masks = masks.squeeze(1)  # [batch_size, H, W]

    # Apply one-hot encoding
    one_hot = torch.nn.functional.one_hot(masks, num_classes=desirable_class)  # [batch_size, H, W, num_classes]

    # Convert to float and rearrange dimensions to [batch_size, num_classes, H, W]
    one_hot = one_hot.permute(0, 3, 1, 2).float()

    # need to be conginous
    one_hot = one_hot.contiguous()

    if cfg['loss']['name'] == "DiceLoss":
        one_hot = one_hot.long()
    
    return one_hot

def tensor_to_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a normalized tensor back to a PIL Image with pixel values in [0, 255].

    Parameters:
        tensor (torch.Tensor): The normalized tensor to convert.
        mean (list or tuple): Mean values used for normalization.
        std (list or tuple): Standard deviation values used for normalization.

    Returns:
        PIL.Image: The denormalized and converted PIL Image.
    """
    # Ensure the tensor has the same number of channels as the mean and std
    if tensor.size(0) != len(mean):
        raise ValueError("Tensor channels do not match the length of mean and std lists")

    # Reverse normalization
    mean = torch.tensor(mean).reshape(-1, 1, 1)
    std = torch.tensor(std).reshape(-1, 1, 1)
    denormalized_tensor = tensor * std + mean
    
    # Clip tensor values to the [0, 1] range and scale to [0, 255]
    denormalized_tensor = denormalized_tensor.clip(0, 1)  # Ensure values are within [0, 1]
    denormalized_tensor = (denormalized_tensor * 255).byte()  # Scale to [0, 255] and convert to byte

    # Convert tensor to PIL image
    return denormalized_tensor




def grey_to_rgb_mask(output, path = None):
    """
    Converts a grayscale mask to an RGB mask using a color dictionary.

    Args:
        output (torch.Tensor): The grayscale mask tensor to convert to an RGB mask.
        path (str): The path to the color dictionary file. If None, a default color dictionary is used.

    Returns:
        np.ndarray: The RGB mask as a NumPy array with shape (H, W, 3).
    """

    # Get the model output
    # output = utils.train_utlis.predict(model, input)
    output = output.cpu().numpy()

    # Load the color dictionary
    if path:
        color_dict = utils.data_utils.parse_color_file(path)
    else: 
        color_dict = {
        0 : (255, 0, 0),      # Red
        1 : (0, 255, 0),      # Green
        2 : (0, 0, 255),      # Blue
        3: (255, 255, 0),    # Yellow
        4: (0, 255, 255),    # Cyan
        5: (255, 0, 255),    # Magenta
        6: (192, 192, 192),  # Silver
        7: (128, 0, 0),      # Maroon
        8: (0, 128, 0),      # Dark Green
        9: (0, 0, 128),}     # Navy
        color_dict = {k: color_dict[k] for k in range(0, cfg['train']['desirable_class'])}

    # Convert the grayscale mask to an RGB mask
    output = output.squeeze(0)                 
    # Create an empty RGB mask
    rgb_mask = np.zeros((3, output.shape[0], output.shape[1]), dtype=np.uint8)
    
    # Convert the grayscale mask to an RGB mask
    for index,color in enumerate(color_dict.values()):
        rgb_mask[0][output==index] = color[0]  # Red channel
        rgb_mask[1][output==index] = color[1]  # Green channel
        rgb_mask[2][output==index] = color[2]  # Blue channel
    rgb_mask = rgb_mask.transpose(1,2,0)

    return rgb_mask, color_dict

def visualize_comparison(pred, target):
    """
    Visualizes a comparison between a predicted mask and a target mask.

    Args:
        pred (torch.Tensor): The predicted mask tensor.
        target (torch.Tensor): The target mask tensor.

    Returns:
        None
    """
    # Convert the input and target to numpy arrays
    pred = pred.permute(1,2,0).cpu().numpy()

    #plot the input and target side by side
    plt.figure(figsize=(10, 5))

    #first subplot
    plt.subplot(1, 2, 1)
    plt.imshow(pred)
    plt.title('prediction')
    plt.axis('off')

    #second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(target)
    plt.title('ground truth')
    plt.axis('off')

    #show the plot
    plt.tight_layout()
    plt.show()
  
    return 



def rgb_to_grey(mask,label_dict):
    """
    Converts an RGB mask to a grayscale mask using a label dictionary.

    Args:
        mask (np.ndarray): The RGB mask to convert to a grayscale mask.
        label_dict (dict): A dictionary mapping RGB colors to class labels.
    
    Returns:
        np.ndarray: The grayscale mask as a NumPy array with shape (H, W).
    """

    # Create an empty grayscale mask
    grey_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    
    # Convert the RGB mask to a grayscale mask
    for index, color in enumerate(label_dict.values()):
        grey_mask[np.all(mask == color,axis = -1)] = index
        
    return grey_mask


def find_contours(mask):

    """
    Finds contours in a mask for each class value.

    Args:
        mask (np.ndarray): The mask to find contours in.

    Returns:
        dict: A dictionary mapping class values to a list of contours for that class.
    """

    #make the mask numpy array
    mask = np.array(mask.squeeze(0))

    #find class_values in the mask and create a binary mask for each class
    class_values = np.unique(mask)  # For example: [0, 128, 255]
    
    # Dictionary to store contours for each class
    contours_by_class = {}

    # Loop through each class value and find contours
    for value in class_values:
        
        # Create a binary mask for the current class
        value_mask = np.uint8(mask == value)
        
        # Find contours for the current class
        contours, _ = cv2.findContours(value_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        multipoly = []
        #If number of points is high, Reduce the number of points in the contour
        for contour in contours:
            while len(contour) >= 40:
                # Reduce the number of points
                epsilon = 0.02 * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # check if the contour is valid to be polygon
            if len(contour) < 3:
                continue

            #make it polygon
            poly = Polygon(contour.reshape(-1, 2))
            
            # Check if the polygon is valid
            multipoly.append(poly)  

        # Store the contours for the current class
        contours_by_class[value] = MultiPolygon(multipoly)

    return contours_by_class

def contours_to_vectors(dict):
    """
    Converts contours to vector polygons and assigns class labels.

    Args:
        dict (dict): A dictionary mapping class values to a list of contours for that class.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the vector polygons and class labels.
    """
    geometries = []
    classes = []
    for class_label, multi_polygon in dict.items():
        geometries.append(multi_polygon)
        classes.append(class_label)

    # Step 4: Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': geometries, 'class': classes}, crs="EPSG:4326")

    return gdf

def mask_to_vector(grey_mask):
    """
    Converts a grayscale mask to vector polygons and assigns class labels.

    Args:
        grey_mask (np.ndarray): The grayscale mask to convert to vector polygons.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the vector polygons and class labels.
    """

      # Find contours for each class
    contours_by_class = find_contours(grey_mask)

    # Convert contours to polygons and assign classes
    gdf = contours_to_vectors(contours_by_class)

    return gdf
   
def compare(images,masks):
    """
    Compare between images and masks
    Args:
        images (list): list of images
        masks (list): list of masks
    Returns:
        None
    """
    photo_num = input("Enter the number of the photo you want to compare: ")
    photo_num = int(photo_num)
    image = images[photo_num]
    mask = masks[photo_num]
    #convert the mask to rgb 
    mask = grey_to_rgb_mask(mask)
    visualize_comparison(image,mask)