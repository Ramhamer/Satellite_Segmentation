from colorama import Fore, init
init()
import os
import tifffile as tiff
import torch
import cv2
import numpy as np
import webcolors
from PIL import Image
from torchvision import transforms
from utils.data_utils import split_image , rebuild_image
from utils.image_utils import grey_to_rgb_mask , add_mask , mask_to_vector,class_label
from utils.train_utlis import model_predict
from utils.cfg_utils import load_yaml
from utils.train_utlis import load_model



    

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

    # # Adaptive font size and spacing based on image size
    # font_scale = max(0.3, 0.0003 * w)  # Scale text based on image width
    # thickness = max(1, int(0.001 * w))  # Scale thickness based on image size
    # spacing = max(15, int(0.05 * legend_h))  # Scale spacing dynamically

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



def predict(model, image):
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if np.array(image).shape[2] == 4:
        image = image.convert('RGB')
    ##open the image 
    preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])              
    image_tensor = preprocess(image)

    #split the image if needed
    main_tensor= split_image(image_tensor)
    
    #pred the input image
    pred_tensor = torch.zeros(main_tensor.shape[0],1,main_tensor.shape[2],main_tensor.shape[3]).to(device)
    for i in range(main_tensor.shape[0]):
        input = ((main_tensor[i, :, :, :]).unsqueeze(0)).to(device)
        output = model_predict(model, input)
        pred_tensor[i,:,:, :] = output
        
    # Rebuild the image
    rebuild_tensor = rebuild_image(pred_tensor,image_tensor.shape)
    
    return rebuild_tensor

def segmented_image(pred_tensor,image,pred_image_path):
    '''
    This function takes the predicted tensor and the image array and save the segmented image
    pred_tensor : torch.Tensor : the predicted tensor
    image_array : np.array : the image array
    pred_image_path : str : the path to save the image
    return : None
    '''
    rgb_mask,color_dict = grey_to_rgb_mask(pred_tensor)         
    output = add_mask(image,rgb_mask)
    final_image = add_class_label_to_image(output,color_dict)
    pred_image = Image.fromarray(final_image)
    path, extension = os.path.splitext(pred_image_path)
    pred_image.save((path.split("/")[-1]).split('.')[0]+'_prediction'+extension)
    return

def image_vector_map(pred_tensor,pred_image_path):
    '''
    This function takes the predicted tensor and save the image vector map
    pred_tensor : torch.Tensor : the predicted tensor
    pred_image_path : str : the path to save the image
    return : None
    '''
    map_vector = mask_to_vector(pred_tensor)
    path, extension = os.path.splitext(pred_image_path)
    map_vector.to_file(path + '.geojson', driver='GeoJSON')
    return

def create_mask(image_path,rebuilt_tensor):
    '''
    This function takes the image path and the rebuilt tensor and save the mask
    
    Args:
        image_path : str : the image path that predicted
        rebuilt_tensor : torch.Tensor : the predicted mask
    return : None
    '''
    image_np = rebuilt_tensor.squeeze(0).numpy().astype(np.uint8)
    mask_name = image_path.replace('.png','_mask.png')
    image = Image.fromarray(image_np)
    image.save(mask_name)
    return

def inference(cfg,weight_path,image_path,device):
    model = load_model(cfg)
    model.to(device)
    model.load_state_dict(torch.load(weight_path))
    print(f"{Fore.GREEN}Model loaded successfully.{Fore.RESET}")
    model.eval()
    image_mode = input(f"{Fore.CYAN}Do you want a segmented image or an image vactor map?{Fore.RESET} \n1. Segmented image \n2. Image mask \n3. Image vector map \n")

    if image_path.endswith('.tif'):
        image = tiff.imread(image_path)
    else:   
        image = Image.open(image_path)   #open the image

    # Predict the image
    pred_tensor = predict(model, image)

    # 1. Segmented image
    if image_mode == "1":
        segmented_image(pred_tensor,image,image_path)

    #2. Image mask
    if image_mode == "2":
        create_mask(image_path,pred_tensor)
        print(f"{Fore.GREEN}The prediction has been completed successfully. \nYou can find the prediction in the image path.{Fore.RESET}\n")

    #3. Image vector map
    if image_mode == "3":
        image_vector_map(pred_tensor,image_path)
        print(f"{Fore.GREEN}The prediction has been completed successfully. \nYou can find the prediction in the image path.{Fore.RESET}\n")
    return

if __name__ == "__main__":
    lst = ['49-2013-0375-6685-LA93-0M50-E080.jp2.tif']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_file = '/workspace/config.yaml'
    cfg = load_yaml(yaml_file)
    
    for i in lst:
        image_path = os.path.join('origin_data/test/images/',i)
    
        weight_path = 'models/DeepLabV3Plus_JaccardFocalLoss_bestX512.pth'
        inference(cfg,weight_path,image_path,device)
    