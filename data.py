import os
import zipfile
from pathlib import Path

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
