import os
import torch
import random
from tqdm.auto import tqdm
from torchinfo import summary
from pathlib import Path
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from data import download_extract_data, ImageFolderCustom
from helpers import find_classes, device, plot_loss_curves
from model import ModelV0, train
from constants import BATCH_SIZE, MODEL_PATH, MODEL_SAVE_PATH, NUM_EPOCHS, NUM_WORKERS

print(f"Device: {device}")
data_path = Path("data/")
image_path = Path(data_path / "dogs-vs-cats")
# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test1"

def main():
    train_list = list(train_dir.glob("*.jpg"))
    test_list = list(test_dir.glob("*.jpg"))
    train_list, validation_list = train_test_split(train_list , test_size=0.2)
    class_names, class_to_idx = find_classes(train_dir)

    # Use smaller subset of data for testing
    # train_list = random.sample(train_list, 1000)
    # validation_list = random.sample(validation_list, 200)
    # test_list = random.sample(test_list, 200)

    print(f"Train list: {len(train_list)} | Validation list: {len(validation_list)} | Test list: {len(test_list)}")

    train_transform = transforms.Compose([
    # Resize our images to 64x64
    transforms.Resize(size=(224, 224)),
    # Flip the images randomly on the horzontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Randomly crop the images
    transforms.RandomResizedCrop(224),
    # Turn the image into a torch.Tensor
    transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        # Resize our images to 64x64
        transforms.Resize(size=(224, 224)),
        # Turn the image into a torch.Tensor
        transforms.ToTensor()
    ])

    train_data = ImageFolderCustom(train_list, transform=train_transform)
    test_data = ImageFolderCustom(test_list, transform=test_transform)
    val_data = ImageFolderCustom(validation_list, transform=test_transform)

    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                shuffle=False)
    validation_dataloader = DataLoader(dataset=val_data,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    print(f"Train data: {len(train_data)} | Train dataloader: {len(train_dataloader)}")
    print(f"Test data: {len(test_data)} | Test dataloader: {len(test_dataloader)}")
    print(f"Validation data: {len(val_data)} | Validation dataloader: {len(validation_dataloader)}")
    print(f"Train data shape: {train_data[0][0].shape} | Train data label: {train_data[0][1]}")

    model_0 = ModelV0().to(device)
    image_batch, label_batch = next(iter(train_dataloader))
    print(f"Image batch shape: {image_batch.shape} | Label batch shape: {label_batch.shape}")
    # print(model_0(image_batch.to(device))) # test matrix fit

    # Get the model summary
    # summary(model_0, input_size=[1, 3, 224, 224])

    # Train the model
    model_0_results = train(model=model_0,
          train_dataloader=train_dataloader,
          test_dataloader=validation_dataloader,
          optimizer=torch.optim.Adam(model_0.parameters(), lr=0.001),
          loss_fn=torch.nn.CrossEntropyLoss(),
          epochs=NUM_EPOCHS,
          device=device)

    # Save the model
    if not MODEL_PATH.is_dir():
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

    plot_loss_curves(model_0_results)
    return

if __name__ == "__main__":
    if not os.path.exists("data"):
        download_extract_data(data_path=data_path, image_path=image_path)
    main()
