import os
import torch
from pathlib import Path
from data import download_extract_data
from helpers import walk_through_dir, display_random_image

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

data_path = Path("data/")
image_path = Path(data_path / "dogs-vs-cats")

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test1"



if __name__ == "__main__":
    if not os.path.exists("data"):
        download_extract_data(data_path=data_path, image_path=image_path)
    # walk_through_dir("data")
    # display_random_image(image_path=train_dir)
