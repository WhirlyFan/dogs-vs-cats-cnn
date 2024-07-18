import os
from typing import List, Tuple
import zipfile
from pathlib import Path
import torch
from torch.utils.data import Dataset
from helpers import find_classes
from PIL import Image

def download_extract_data(data_path=Path("data/"), image_path=Path("data/dogs-vs-cats")):
    # If the image folder doesn't exist, create it
    if image_path.is_dir():
        print(f"{image_path} directory already exists... skipping download")
    else:
        print(f"{image_path} does not exist, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

    # Check if kaggle.json exists
    kaggle_path = Path("~/.kaggle/kaggle.json").expanduser()

    if not kaggle_path.is_file():
        print("kaggle.json not found, please make sure it is in the ~/.kaggle directory.")
        exit()
    else:
        print("kaggle.json found.")

    # Download dogs-vs-cats dataset into the data folder directly
    os.system(f'kaggle competitions download -c dogs-vs-cats -p {data_path}')

    # Unzip dogs-vs-cats.zip data if it hasn't been unzipped already
    if not (data_path / "train.zip").is_file() or not (data_path / "test1.zip").is_file() or not (data_path / "sampleSubmission.csv").is_file():
        with zipfile.ZipFile(data_path / "dogs-vs-cats.zip", "r") as zip_ref:
            print("Unzipping dogs-vs-cats.zip data...")
            zip_ref.extractall(data_path)
    else:
        print("dogs-vs-cats.zip data already unzipped... skipping")

    # Unzip test1.zip into the dogs-vs-cats directory if it hasn't been unzipped already
    if not (image_path / "test1").is_dir():
        with zipfile.ZipFile(data_path / "test1.zip", "r") as zip_ref:
            print("Unzipping test1.zip data...")
            zip_ref.extractall(image_path)
    else:
        print("test1.zip data already unzipped... skipping")

    # Unzip train.zip into the dogs-vs-cats directory if it hasn't been unzipped already
    if not (image_path / "train").is_dir():
        with zipfile.ZipFile(data_path / "train.zip", "r") as zip_ref:
            print("Unzipping train.zip data...")
            zip_ref.extractall(image_path)
    else:
        print("train.zip data already unzipped... skipping")
    print("Done!")



# Write a custom dataset class (inherits from torch.utils.data.Dataset)
# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, paths: List[Path], transform=None) -> None:
        # 3. Create class attributes
        # Get all image paths
        self.paths = paths
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(paths[0].parent)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[index].stem.split(".")[0]
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
