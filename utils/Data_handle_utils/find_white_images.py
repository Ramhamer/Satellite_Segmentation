import os
import cv2
import numpy as np

def check_white_pixels(path, threshold=0.1, output_file="white_pixel_report.txt"):
    # Open the file once in write mode to clear any previous content
    with open(output_file, "w") as file:
        file.write("White Pixel Analysis Report\n")
        file.write("-------------------------\n")
        
    count = 0
    
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        
        # Skip directories, only process files
        if os.path.isdir(image_path):
            continue
            
        # Try to load the image
        try:
            # Load the image in color (RGB) format
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                # Skip files that aren't images
                continue
                
            # Calculate the total number of pixels
            total_pixels = image.shape[0] * image.shape[1]
            
            # Count white pixels (intensity close to [255, 255, 255])
            # Create a mask where all channels are close to 255
            white_mask = (image > 240).all(axis=2)
            white_pixels = np.sum(white_mask)
            
            # Compute white pixel percentage
            white_pixel_ratio = white_pixels / total_pixels
            
            # Write results to the file in append mode
            if white_pixel_ratio > threshold:
                with open(output_file, "a") as file:
                    file.write(f"{image_path}: {white_pixel_ratio:.2%} white pixels\n")
                count += 1
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"Analysis complete. Found {count} images with more than {threshold:.0%} white pixels.")
    print(f"Results saved to {output_file}")
    
if __name__ == "__main__":
    # Define the image path
    path = "Rachel_Tzuria/Data/Dataset/Full_Dataset/test/images"
    
    # Define the threshold for white pixels (10%)
    threshold = 0.1
    
    # Define the output file
    output_file = "Rachel_Tzuria/Data/Dataset/Full_Dataset/white_pixel_report_test.txt"
    
    # Check for white pixels
    check_white_pixels(path, threshold, output_file)