import os
import tifffile as tiff
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from colorama import Fore, Style, init

def save_as_png(image, filename):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(filename, format='PNG')

def rename_masks_to_match_images(output_mask_dir):
    print(f"{Fore.RED}Starting dataset rename..{Fore.RESET}")
    for mask_filename in os.listdir(output_mask_dir):
        if "_UA2012" in mask_filename:
            new_name = mask_filename.replace("_UA2012", "")
            old_path = os.path.join(output_mask_dir, mask_filename)
            new_path = os.path.join(output_mask_dir, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {mask_filename} -> {new_name}")

def process_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir, missing_masks_file_path, crop_size=1024, overlap_percent=10,resize =None):
    print(f"{Fore.RED}Starting to process the dataset..{Fore.RESET}")
    stride = int(crop_size * (1 - overlap_percent / 100))
    print(f"Using crop size: {crop_size}x{crop_size} with {overlap_percent}% overlap (stride: {stride}")
    
    # os.makedirs(output_image_dir, exist_ok=True)
    # os.makedirs(output_mask_dir, exist_ok=True)
    
    converted_images = 0
    total_crops = 0
    missing_masks = []
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif") or f.endswith(".tiff")]
    
    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        image_path = os.path.join(image_dir, filename)
        base_name = filename.replace(".jp2.tif", "")
        mask_filename = base_name + "_UA2012.tif"
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            missing_masks.append(mask_path)
            continue
        
        image = tiff.imread(image_path)
        mask = tiff.imread(mask_path)
        
        if len(mask.shape) > 2:
            mask = mask[:, :, 0] if mask.shape[2] > 0 else mask
        
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        img_height, img_width = image.shape[:2]
        image_crops = 0
        
        for y in tqdm(range(0, img_height - crop_size + 1, stride), desc=f"Cropping {filename}", leave=False):
            for x in range(0, img_width - crop_size + 1, stride):
                image_crop = image[y:y+crop_size, x:x+crop_size]
                mask_crop = mask[y:y+crop_size, x:x+crop_size]

                #check if the crop have unkonwn value in the mask 
                if np.any(mask_crop == 0):
                    continue  # Skip this crop if any pixel is unknow
                         
                if image_crop.shape[0] == crop_size and image_crop.shape[1] == crop_size:
                    image_crop_filename = f"{base_name}_y{y}_x{x}.png"
                    mask_crop_filename = f"{base_name}_y{y}_x{x}.png"

                    if resize:
                        image_crop = cv2.resize(image_crop,(crop_size//2,crop_size//2))
                        mask_crop = cv2.resize(mask_crop,(crop_size//2,crop_size//2),interpolation = cv2.INTER_NEAREST)
                    save_as_png(image_crop, os.path.join(output_image_dir, image_crop_filename))
                    save_as_png(mask_crop, os.path.join(output_mask_dir, mask_crop_filename))
                    
                    image_crops += 1
                    total_crops += 1
        
        print(f"Generated {image_crops} crops for {filename}")
        converted_images += 1
    
    


if __name__ == "__main__":
    lst = ["D035","D029","D013"]
    for i in lst:
        image_dir = f"origin_data/{i}"
        mask_dir = f"origin_data/{i}_labels"
        output_image_dir = '/origin_data/1024_crop/512_compress/train/images'
        output_mask_dir = '/origin_data/1024_crop/512_compress/train/masks'
        missing_masks_file_path = "Rachel_Tzuria/Data/origin_data/for_test/D049/missing_masks.txt" # corelated to the image dir
        
        process_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir, missing_masks_file_path,resize = True)
        rename_masks_to_match_images(output_mask_dir)
        print(f"{Fore.GREEN}Dataset rename complete.{Fore.RESET}")