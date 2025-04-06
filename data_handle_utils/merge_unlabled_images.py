import os
import shutil
from pathlib import Path

def collect_images(txt_files, base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    
    for txt_file in txt_files:
        txt_path = Path(base_dir) / txt_file  # Construct full path within the container
        
        if not txt_path.exists():
            print(f"Warning: {txt_path} not found.")
            continue
        
        with open(txt_path, 'r') as file:
            for line in file:
                image_path = (Path(base_dir) / line.strip()).resolve()
                
                if image_path.exists():
                    filename = image_path.name
                    dest_path = Path(output_dir) / filename
                    
                    # Avoid overwriting files with the same name
                    counter = 1
                    while dest_path.exists():
                        name, ext = os.path.splitext(filename)
                        dest_path = Path(output_dir) / f"{name}_{counter}{ext}"
                        counter += 1
                    
                    shutil.move(str(image_path), str(dest_path))
                    print(f"Moved: {image_path} -> {dest_path}")
                else:
                    print(f"Warning: {image_path} not found.")

if __name__ == "__main__":
    txt_files = [  # List of relative paths to .txt files
        "Rachel_Tzuria/Data/RAW/for_train/D013/missing_masks.txt", 
        "Rachel_Tzuria/Data/RAW/for_train/D035/missing_masks.txt"
    ]  
    
    base_dir = '/home/oury/Documents/Segmentation_model/'
    output_dir = '/home/oury/Documents/Segmentation_model/Rachel_Tzuria/Data/RAW/Unlabled_Images/'
    
    collect_images(txt_files, base_dir, output_dir)
