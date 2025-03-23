import wandb
import numpy as np
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as F

# Custom Dataset for Image-Mask Pairs
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))  # Ensure order is consistent

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)  # Assumes masks have the same filename

        image = Image.open(img_path).convert("L")  # Convert to grayscale
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, img_name  # Returning filename for logging

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
])


# Function to calculate IoU
def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 0  # Avoid division by zero


if __name__ == "__main__":
	# Initialize WandB
	wandb.init(project="try")
    
	num_epochs = 3
	model = smp.Unet(
    encoder_name="resnet34",    # Choose encoder: resnet34, efficientnet-b0, etc.
    encoder_weights="imagenet", # Use pre-trained weights
    in_channels=1,              # Change to 3 for RGB images
    classes=1)
		# Dataset Paths
	image_dir = "/workspace/DATA/train/images"
	mask_dir = "/workspace/DATA/train/masks"

	# Create DataLoader Directly
	val_loader = DataLoader(
		SegmentationDataset(image_dir, mask_dir, transform=transform),
		batch_size=4,  # Adjust as needed
		shuffle=False,
		num_workers=2)
      
	table = wandb.Table(columns=["Epoch", "ID", "Prediction", "GT", "IoU"])  # New table each epoch
   
	for epoch in range(num_epochs):
            
		for  images, masks, filenames in val_loader:
			
			outputs = model(images)
			outputs = outputs.detach().numpy() # Forward pass  # Convert to NumPy
			gt_masks = masks.cpu().numpy()
			for j in range(len(filenames)):  # Loop through batch
				iou = calculate_iou(outputs[j], gt_masks[j])  # IoU computation

				# Convert to PIL images for logging
				pred_img = F.to_pil_image(torch.tensor(outputs[j]).float())
				gt_img = F.to_pil_image(torch.tensor(gt_masks[j]).float())

				# Add to WandB Table
				table.add_data(epoch,
					filenames[j],
					wandb.Image(pred_img, caption="Prediction"),
					wandb.Image(gt_img, caption="Ground Truth"),
					iou)

		# Log table for the current epoch
		wandb.log({
            "Epoch IoU": wandb.Chart(
                data=[{"epoch": epoch, "IoU": epoch}],
                columns=["epoch", "IoU"]
            )
        })
            
	wandb.finish()






