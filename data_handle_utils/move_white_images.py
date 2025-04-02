import os
import shutil
import re

def process_txt_files(txt_files, destination_dir):
    """
    Process each line in the provided text files, extract image paths, and move (not copy) images and masks
    to appropriate destination directories.
    
    Args:
        txt_files (list): List of paths to the txt files containing image paths
        destination_dir (str): Base destination directory (e.g., 'white_images')
    """
    # Create destination directories if they don't exist
    images_dest = os.path.join(destination_dir, 'images')
    masks_dest = os.path.join(destination_dir, 'masks')
    
    os.makedirs(images_dest, exist_ok=True)
    os.makedirs(masks_dest, exist_ok=True)
    
    # Process each txt file
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            for line in f:
                # Extract image path (everything before the colon)
                match = re.match(r'(.*?):\s*[\d.]+%\s*white\s*pixels', line.strip())
                if match:
                    image_path = match.group(1)
                    
                    # Construct mask path by replacing 'images' with 'masks' in the path
                    mask_path = image_path.replace('/images/', '/masks/')
                    
                    # Get just the filename without the path
                    image_filename = os.path.basename(image_path)
                    
                    # Construct destination paths
                    image_dest_path = os.path.join(images_dest, image_filename)
                    mask_dest_path = os.path.join(masks_dest, image_filename)
                    
                    # Move files
                    try:
                        shutil.move(image_path, image_dest_path)
                        print(f"Moved image: {image_path} -> {image_dest_path}")
                        
                        if os.path.exists(mask_path):
                            shutil.move(mask_path, mask_dest_path)
                            print(f"Moved mask: {mask_path} -> {mask_dest_path}")
                        else:
                            print(f"Warning: Mask not found: {mask_path}")
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    # List of txt files to process
    txt_files = [
        "Rachel_Tzuria/Data/Dataset/Full_Dataset/white_pixel_report_train.txt",
        "Rachel_Tzuria/Data/Dataset/Full_Dataset/white_pixel_report_val.txt",
        "Rachel_Tzuria/Data/Dataset/Full_Dataset/white_pixel_report_test.txt"
    ]
    
    # Destination directory
    destination_dir = "Rachel_Tzuria/Data/Dataset/Full_Dataset/white_pixel_images"
    
    # Process the files
    process_txt_files(txt_files, destination_dir)