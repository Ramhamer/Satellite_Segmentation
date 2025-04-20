import os
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm

def crop_images_with_masks(image_dir, mask_dir, output_image_dir, output_mask_dir, 
                          crop_size=512, overlap_percent=10, file_extension="*.png"):
    """
    Crop images and their corresponding masks to 512x512 with specified overlap percentage.
    Discards edge crops that can't form a complete 512x512 without padding.
    
    Parameters:
    -----------
    image_dir : str
        Directory containing original images
    mask_dir : str
        Directory containing original masks
    output_image_dir : str
        Directory to save cropped images
    output_mask_dir : str
        Directory to save cropped masks
    crop_size : int
        Size of the square crop (default: 512)
    overlap_percent : int
        Percentage of overlap between crops (default: 20)
    file_extension : str
        File extension pattern to match images (default: "*.png")
    """
    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Calculate stride based on overlap percentage
    stride = int(crop_size * (1 - overlap_percent / 100))
    
    # Get list of image files
    image_files = sorted(glob.glob(os.path.join(image_dir, file_extension)))
    
    print(f"Processing {len(image_files)} images with {overlap_percent}% overlap...")
    
    # For each image and corresponding mask
    for img_path in tqdm(image_files):
        # Get base filename without extension
        base_name = os.path.basename(img_path)
        filename_without_ext = os.path.splitext(base_name)[0]
        
        # Construct mask path with the same name
        mask_path = os.path.join(mask_dir, base_name)
        
        # Check if mask exists
        if not os.path.exists(mask_path):
            print(f"Warning: No corresponding mask found for {img_path}. Skipping.")
            continue
        
        # Open image and mask
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        # Convert to numpy arrays
        img_array = np.array(img)
        mask_array = np.array(mask)
        
        # Get dimensions
        img_height, img_width = img_array.shape[:2]
        
        # Counter for naming crops
        crop_count = 0
        
        # Generate crops
        for y in range(0, img_height - crop_size + 1, stride):
            for x in range(0, img_width - crop_size + 1, stride):
                # Extract crop from image and mask
                img_crop = img_array[y:y+crop_size, x:x+crop_size]
                mask_crop = mask_array[y:y+crop_size, x:x+crop_size]
                
                # Check if crop has the required size (512x512)
                if img_crop.shape[0] == crop_size and img_crop.shape[1] == crop_size:
                    # Create output filenames
                    crop_img_filename = f"{filename_without_ext}_crop_{crop_count}.png"
                    crop_mask_filename = f"{filename_without_ext}_crop_{crop_count}.png"
                    
                    # Save crops
                    Image.fromarray(img_crop).save(os.path.join(output_image_dir, crop_img_filename))
                    Image.fromarray(mask_crop).save(os.path.join(output_mask_dir, crop_mask_filename))
                    
                    crop_count += 1
        
        print(f"Generated {crop_count} crops for {base_name}")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual directories
    image_directory = "path/to/images"
    mask_directory = "path/to/masks"
    output_image_directory = "path/to/output/images"
    output_mask_directory = "path/to/output/masks"
    
    # Set your desired overlap percentage
    overlap = 30  # 30% overlap
    
    crop_images_with_masks(
        image_directory, 
        mask_directory,
        output_image_directory,
        output_mask_directory,
        overlap_percent=overlap,
        file_extension="*.png"  # Change if your images have a different extension
    )