import os
import shutil

def check_pairs(path):
    """"
    check which images dosent have a corresponding mask and write the path on a txt file
    """
    counter = 0
    for i, filename in enumerate(os.listdir(path)):
        mask_path = os.path.join(path,filename)
        image_path = mask_path.replace("/masks/", "/images/")
        if os.path.exists(mask_path):
           if os.path.exists(image_path):
                new_path = "/Data/verified"
                #move the mask to the new directory
                new_mask_dir = os.path.join(new_path, "masks")
                new_image_dir = os.path.join(new_path, "images")
                shutil.copy(mask_path, new_mask_dir)
                shutil.copy(image_path, new_image_dir)
                counter += 1
    print(f"found{counter} masks")
    
    
if __name__ == "__main__":
    path = "/Data/train/masks"
    check_pairs(path)