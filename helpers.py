import os
import random
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def walk_through_dir(dir_path):
  """Walks through dir_path returning its contents."""
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")


def display_random_image(image_path: Path, seed=None) -> None:
    """Display a random image from a specified directory."""
    # Set seed
    random.seed(seed)

    # 1. Get all image paths
    image_path_list = list(image_path.glob("*.jpg"))

    # 2. Pick a random image path
    random_image_path = random.choice(image_path_list)
    print(random_image_path)

    # 3. Get image class from path name (the image class is the name of the directory)
    image_class = random_image_path.stem.split('.')[0]
    print(image_class)

    # 4. Open image
    img = Image.open(random_image_path)

    # 5. Print metadata
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")

    # 6. Display image using matplotlib
    plt.imshow(img)
    plt.title(f"Class: {image_class}")
    plt.axis('off')  # Hide axes
    plt.show()
