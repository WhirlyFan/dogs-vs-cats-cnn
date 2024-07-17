import os
import torch
from tqdm.auto import tqdm
from torch import nn
import pathlib
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data import download_extract_data
from helpers import walk_through_dir, display_random_image
from model import data_transform, Cnn

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

BATCH_SIZE = 100
NUM_WORKERS = os.cpu_count()

data_path = Path("data/")
image_path = Path(data_path / "dogs-vs-cats")

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test1"


class dataset(Dataset):
  def __init__(self,file_list,transform = None):
      self.file_list=file_list
      self.transform=transform

  def __len__(self):
      self.filelength =len(self.file_list)
      return self.filelength

  def __getitem__(self,idx):
      img_path = self.file_list[idx]
      img = Image.open(img_path)
      img_transformed = self.transform(img)

      label = img_path.stem.split('.')[0]
      if label == 'dog':
          label = 1
      elif label == 'cat':
          label = 0

      return img_transformed, label

train_list = list(train_dir.glob("*.jpg"))
test_list = list(test_dir.glob("*.jpg"))
train_list, validation_list = train_test_split(train_list , test_size=0.2)

train_data = dataset(train_list, transform=data_transform)
test_data = dataset(test_list, transform=data_transform)
val_data = dataset(validation_list, transform=data_transform)



train_loader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

test_loader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=False)
val_loader = DataLoader(dataset=val_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
print(f"Train data: {len(train_data)} | Train loader: {len(train_loader)}")
print(f"Test data: {len(test_data)} | Test loader: {len(test_loader)}")
print(f"Validation data: {len(val_data)} | Validation loader: {len(val_loader)}")

print(f"Train data shape: {train_data[0][0].shape} | Train data label: {train_data[0][1]}")

model = Cnn().to(device)
model.train()
optimizer = torch.optim.Adam(params = model.parameters(),lr =0.001)
criterion = nn.CrossEntropyLoss()

epochs = 5

for epoch in tqdm(range(epochs)):
    epoch_loss =0
    epoch_accuracy = 0

    for data,label in train_loader:
        data= data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = ((output.argmax(dim=1)==label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)

    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))

    with torch.no_grad():
        epoch_val_accuracy =0
        epoch_val_loss = 0
        for data,label in  val_loader:
            data= data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(output,label)


            acc = ((output.argmax(dim=1)==label).float().mean())
            epoch_val_accuracy += acc/len(val_loader)
            epoch_val_loss += val_loss/len(val_loader)
        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))

if __name__ == "__main__":
    if not os.path.exists("data"):
        download_extract_data(data_path=data_path, image_path=image_path)
    # walk_through_dir("data")
    # display_random_image(image_path=train_dir)
