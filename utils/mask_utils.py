import numpy as np
from PIL import Image

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