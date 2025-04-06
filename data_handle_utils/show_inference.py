import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import os


# Fix Fontconfig error (disable Matplotlib's font manager cache)
os.environ["MPLCONFIGDIR"] = "/tmp"

def show_inference(masks_path, class_value, max_files=300):
    """
    Displays an overlay of masks on images where the specified class_value is found.
    Allows users to hover over the image to see mask values.
    
    :param masks_path: Path to the directory containing mask images.
    :param class_value: The class value to look for in masks.
    :param max_files: Maximum number of files to process (default: 5 for debugging).
    """
    counter = 0
    
    for i, filename in enumerate(os.listdir(masks_path)):
         # Limit processing to avoid long runtimes
        
        if filename.endswith(".tif"):
            mask_path = os.path.join(masks_path, filename)
            image_path = mask_path.replace("/labels", "", 1).replace("_UA2012.tif", ".jp2.tif")

            if not os.path.exists(image_path):
                print(f"Skipping: Image file not found for {mask_path}")
                continue
            
            print(f"Processing {filename}...")
            
            # Read image and mask
            image = tiff.imread(image_path)
            mask = tiff.imread(mask_path)
        if filename.endswith(".png") :
            mask_path = os.path.join(masks_path, filename)
            image_path = mask_path.replace("/masks", "/images", 1)

            #open image and mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if class_value in np.unique(mask):
            counter += 1
            print(f"Class {class_value} found in {mask_path}")

            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            original_mask = mask.copy()
            
            # Normalize mask for visualization
            mask_norm = ((mask - mask.min()) / (mask.max() - mask.min() + 1e-6) * 255).astype(np.uint8)
            mask_colored = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
            
            # Blend image and mask
            overlay = cv2.addWeighted(image, 1, mask_colored, 0.5, 0)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax.set_title("Hover over the image to see mask values")
            plt.axis("off")

            def on_mouse_move(event):
                if event.inaxes is not None and event.xdata is not None and event.ydata is not None:
                    x, y = int(event.xdata), int(event.ydata)
                    if 0 <= y < original_mask.shape[0] and 0 <= x < original_mask.shape[1]:
                        mask_value = original_mask[y, x]
                        ax.set_title(f"Mask Value (Class): {mask_value}")
                        fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
            plt.show()
            
            #write the image name on a txt file
            with open("Rachel_Tzuria/Data/Dataset/Full_Dataset/masks_with_class_test_0.txt", "a") as f:
               f.write(f"{mask_path}\n")

        # else:
        #     print(f"Class {class_value} not found in {mask_path}")
            
    print(f"Class {class_value} found in {counter} masks")


def delete_zero_class(files_path,txt_file_path):
    """
    Deletes files that are listed in the txt file.
    
    :param files_path: Path to the directory containing files to delete.
    :param txt_file_path: Path to the txt file containing the list of files to delete.
    """
    with open(txt_file_path, "r") as f:
        files_to_delete = f.readlines()
    i=0
    for file in files_to_delete:
        file = file.strip()
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue
        os.remove(file)
        image_path = file.replace("/masks", "/images", 1)
        os.remove(image_path)
        i+=1
        print(f"Deleted {file}")
    print(f"Deleted {i} files")
if __name__ == "__main__":
    masks_path = "Rachel_Tzuria/Data/origin_data/labels/for_test/D049"  # Update path as needed
    class_value = 2  # Class value to visualize
    show_inference(masks_path, class_value)
    txt_file_path = "Rachel_Tzuria/Data/Dataset/Full_Dataset/masks_with_class_test_0.txt"
    # delete_zero_class(masks_path, txt_file_path)