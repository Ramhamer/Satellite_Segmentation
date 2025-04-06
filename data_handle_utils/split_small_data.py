import os
import random
import shutil

# Function to split the data
def create_small_dataset(full_dataset_dir, output_dir, percentage):
    """
    Create a smaller dataset by randomly sampling a percentage of files from each directory.
    
    Args:
        full_dataset_dir: Path to the full dataset with train/val/test directories
        output_dir: Path where the smaller dataset will be created
        percentage: Percentage of files to sample (0-100)
    """
    # Validate percentage
    if percentage <= 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")
        
    # List of subdirectories to process
    subdirs = ['train', 'val', 'test']
    
    # Process each subdirectory
    for subdir in subdirs:
        images_src_dir = os.path.join(full_dataset_dir, subdir, 'images')
        masks_src_dir = os.path.join(full_dataset_dir, subdir, 'masks')
        
        # Skip if source directories don't exist
        if not os.path.exists(images_src_dir) or not os.path.exists(masks_src_dir):
            print(f"Warning: {subdir} directory not found, skipping.")
            continue
            
        # Create destination directories
        images_dst_dir = os.path.join(output_dir, subdir, 'images')
        masks_dst_dir = os.path.join(output_dir, subdir, 'masks')
        
        for dir_path in [images_dst_dir, masks_dst_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Get list of image files
        image_files = [f for f in os.listdir(images_src_dir) if os.path.isfile(os.path.join(images_src_dir, f))]
        
        # Filter for images that have corresponding masks
        valid_pairs = []
        for img_file in image_files:
            mask_file = os.path.join(masks_src_dir, img_file)
            if os.path.exists(mask_file):
                valid_pairs.append(img_file)
            else:
                print(f"Warning: No matching mask found for {img_file} - skipping this pair")
        
        if not valid_pairs:
            print(f"Error: No valid image-mask pairs found in {subdir}")
            continue
            
        # Calculate how many files to sample
        num_files_to_sample = max(1, int(len(valid_pairs) * percentage / 100))
        
        # Randomly sample files
        sampled_files = random.sample(valid_pairs, num_files_to_sample)
        
        print(f"Copying {len(sampled_files)} image-mask pairs from {subdir} (out of {len(valid_pairs)} valid pairs)")
        
        # Copy the sampled files
        for filename in sampled_files:
            # Copy image file
            src_image = os.path.join(images_src_dir, filename)
            dst_image = os.path.join(images_dst_dir, filename)
            shutil.copy2(src_image, dst_image)
            
            # Copy corresponding mask file
            src_mask = os.path.join(masks_src_dir, filename)
            dst_mask = os.path.join(masks_dst_dir, filename)
            shutil.copy2(src_mask, dst_mask)

# Example usage
if __name__ == "__main__":
<<<<<<< HEAD
    full_dataset_dir = '/workspace/origin_data/1024_crop/verified'
    output_dir = '/workspace/origin_data/1024_crop/03_04_25'
    percentage = 10 
=======
<<<<<<< HEAD
    full_dataset_dir = 'Rachel_Tzuria/Data/Dataset/Small_test_dataset'
    output_dir = 'Rachel_Tzuria/Data/Dataset/Debug_dataset'
    percentage = 5  # Take 20% of the original data
=======
    full_dataset_dir = '/workspace/origin_data/1024_crop/verified'
    output_dir = '/workspace/origin_data/1024_crop/03_04_25'
    percentage = 10 
>>>>>>> origin/dan_branch
>>>>>>> origin/main
    
    create_small_dataset(full_dataset_dir, output_dir, percentage)
    print(f"Created smaller dataset with {percentage}% of the original data in {output_dir}")