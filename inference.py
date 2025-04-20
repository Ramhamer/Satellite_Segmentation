from colorama import Fore, init
init()
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.data_utils import split_image , rebuild_image
from utils.image_utils import grey_to_rgb_mask , add_mask , mask_to_vector, add_class_label_to_image
from utils.train_utlis import model_predict
from utils.cfg_utils import load_yaml
from utils.train_utlis import load_model

def predict(model, image):
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    pred_image = Image.fromarray(output)
    path, extension = os.path.splitext(pred_image_path)
    pred_image.save(path+'_prediction'+extension)
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
    print(f"{Fore.GREEN}Model loaded successfully.{Fore.RESET}")
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    image_mode = input(f"{Fore.CYAN}Do you want a segmented image or an image vactor map?{Fore.RESET} \n1. Segmented image \n2. Image mask \n3. Image vector map \n")
    image = Image.open(image_path)   #open the image

    # Predict the image
    pred_tensor = predict(model, image)

    # 1. Segmented image
    if image_mode == "1":
        segmented_image(pred_tensor,image,image_path)
        print(f"{Fore.GREEN}The prediction has been completed successfully. \nYou can find the prediction in the image path.{Fore.RESET}\n")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_file = '/workspace/config.yaml'
    cfg = load_yaml(yaml_file)
    image_path = 'inference_images/TargetSize512/29-2012-0135-6845-LA93-0M50-E080.jp2.png'
    weight_path = 'models/DeepLabV3Plus_JaccardFocalLoss_bestX512.pth'
    inference(cfg,weight_path,image_path,device)