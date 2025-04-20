import os
import shutil
import random

def split_dataset(images_dir, masks_dir, output_dir, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, reduce_ratio=1.0, seed=42):
    """
    Splits the dataset in images_dir and masks_dir into train/val/test subsets, with optional size reduction.

    Args:
        images_dir: Path to the images directory
        masks_dir: Path to the masks directory
        output_dir: Path to the output dataset directory
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        reduce_ratio: Fraction of total dataset to keep (e.g., 0.1 for 10%)
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    assert 0 < reduce_ratio <= 1.0, "reduce_ratio must be between 0 and 1"

    # Get list of valid image-mask pairs
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    valid_files = [f for f in image_files if os.path.exists(os.path.join(masks_dir, f))]

    if not valid_files:
        raise ValueError("No valid image-mask pairs found.")

    # Shuffle the full list
    random.seed(seed)
    random.shuffle(valid_files)

    # Reduce dataset size if needed
    reduced_count = int(len(valid_files) * reduce_ratio)
    reduced_files = valid_files[:reduced_count]

    print(f"Using {reduced_count}/{len(valid_files)} total files ({reduce_ratio*100:.1f}%)")

    # Split
    total = len(reduced_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': reduced_files[:train_end],
        'val': reduced_files[train_end:val_end],
        'test': reduced_files[val_end:]
    }

    for split, files in splits.items():
        images_out = os.path.join(output_dir, split, 'images')
        masks_out = os.path.join(output_dir, split, 'masks')
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(masks_out, exist_ok=True)

        for filename in files:
            shutil.copy2(os.path.join(images_dir, filename), os.path.join(images_out, filename))
            shutil.copy2(os.path.join(masks_dir, filename), os.path.join(masks_out, filename))
        print(f"Copied {len(files)} files to {split}")

# Example usage
if __name__ == "__main__":
    images_dir = 'Data/MiniFrance/Dataset/1024_Debug_dataset/D029/images'
    masks_dir = 'Data/MiniFrance/Dataset/1024_Debug_dataset/D029/masks'
    output_dir = 'Data/MiniFrance/Dataset/1024_Debug_dataset'

    # Only take 10% of the dataset, and split it 75/15/10
    split_dataset(images_dir, masks_dir, output_dir,
                  train_ratio=0.75, val_ratio=0.15, test_ratio=0.10,
                  reduce_ratio=0.2)
