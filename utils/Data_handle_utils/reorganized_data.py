import os
import random
import shutil
from pathlib import Path
import re

def extract_base_name(filename):
    """Extract the base name from the image or mask filename."""
    # For mask files: remove the _UA2012.tif suffix
    if "_UA2012.tif" in filename:
        return filename.replace("_UA2012.tif", "")
    # For image files: remove the .jp2.tif suffix
    elif ".jp2.tif" in filename:
        return filename.replace(".jp2.tif", "")
    return filename

def split_dataset(src_images_dir, src_masks_dir, output_dir, train_ratio=0.5):
    """
    Split the dataset into train and validation sets.
    
    Args:
        src_images_dir: Source directory for images
        src_masks_dir: Source directory for masks
        output_dir: Output directory
        train_ratio: Ratio of data to be used for training (default 0.5)
    
    Returns:
        Number of images without corresponding masks
    """
    # Create output directories
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_masks_dir = os.path.join(output_dir, "train", "masks")
    val_images_dir = os.path.join(output_dir, "val", "images")
    val_masks_dir = os.path.join(output_dir, "val", "masks")
    
    for directory in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get all mask files
    mask_files = [f for f in os.listdir(src_masks_dir) if f.endswith(".tif")]
    
    # Shuffle the mask files to randomize the selection
    random.shuffle(mask_files)
    
    # Calculate split point
    split_idx = int(len(mask_files) * train_ratio)
    
    # Split into train and validation sets
    train_masks = mask_files[:split_idx]
    val_masks = mask_files[split_idx:]
    
    # Process each mask and find its corresponding image
    processed_pairs = 0
    for mask_set, dest_img_dir, dest_mask_dir in [
        (train_masks, train_images_dir, train_masks_dir),
        (val_masks, val_images_dir, val_masks_dir)
    ]:
        for mask_file in mask_set:
            # Get base name without _UA2012.tif
            base_name = extract_base_name(mask_file)
            
            # Find matching image
            matching_image = None
            for img_file in os.listdir(src_images_dir):
                if extract_base_name(img_file) == base_name:
                    matching_image = img_file
                    break
            
            if matching_image:
                # Copy mask and image to respective destinations
                shutil.copy2(
                    os.path.join(src_masks_dir, mask_file),
                    os.path.join(dest_mask_dir, mask_file)
                )
                shutil.copy2(
                    os.path.join(src_images_dir, matching_image),
                    os.path.join(dest_img_dir, matching_image)
                )
                processed_pairs += 1
    
    print(f"Processed {processed_pairs} image-mask pairs")
    print(f"  - {len(train_masks)} pairs in training set")
    print(f"  - {len(val_masks)} pairs in validation set")
    
    # Count images without masks
    images_without_masks = 0
    for img_file in os.listdir(src_images_dir):
        if img_file.endswith(".tif"):
            base_name = extract_base_name(img_file)
            has_mask = False
            for mask_file in mask_files:
                if extract_base_name(mask_file) == base_name:
                    has_mask = True
                    break
            if not has_mask:
                images_without_masks += 1
    
    return images_without_masks

if __name__ == "__main__":
    # =====================================================================
    # MODIFY THE PATHS BELOW TO RUN DIRECTLY FROM VSCode
    # =====================================================================
    
    # Input directories
    src_images_dir = "Rachel_Tzuria/Data/OLD/New_dir_to_check/D049"  # CHANGE THIS: path to your source images directory
    src_masks_dir = "Rachel_Tzuria/Data/OLD/New_dir_to_check/labels/D049"    # CHANGE THIS: path to your source masks directory
    
    # Output directory - subdirectories will be created automatically
    output_dir = "Rachel_Tzuria/Data/OLD/for_validation/D049"      # CHANGE THIS: path to your output directory
    
    # Other settings
    train_ratio = 0.5                   # Ratio of data for training (0.5 means 50% train, 50% val)
    use_move = True                    # Set to True if you want to move files instead of copying
    random_seed = 42                    # Random seed for reproducibility
    
    # =====================================================================
    # END OF CONFIGURATION
    # =====================================================================
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Change copy function to move if specified
    if use_move:
        shutil.copy2 = shutil.move
    
    print(f"Processing dataset:")
    print(f"- Source images: {src_images_dir}")
    print(f"- Source masks: {src_masks_dir}")
    print(f"- Output directory: {output_dir}")
    print(f"- Train ratio: {train_ratio}")
    print(f"- {'Moving' if use_move else 'Copying'} files")
    
    images_without_masks = split_dataset(
        src_images_dir, 
        src_masks_dir, 
        output_dir, 
        train_ratio
    )
    
    print(f"Number of images without corresponding masks: {images_without_masks}")
    