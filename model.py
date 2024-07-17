import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Write a transform for image
data_transform = transforms.Compose([
    # Resize our images to 64x64
    transforms.Resize(size=(224, 224)),
    # Flip the images randomly on the horzontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Randomly crop the images
    transforms.RandomResizedCrop(224),
    # Turn the image into a torch.Tensor
    transforms.ToTensor()
])

class dataset(Dataset):
  def __init__(self,file_list,transform = None):
    self.file_list=file_list
    self.transform=transform

  def __len__(self):
    self.filelength =len(self.file_list)
    return self.filelength

  def __getitem__(self,idx):
    img_path =self.file_list[idx]
    img = Image.open(img_path)
    img_transformed = self.transform(img)

    label = img_path.stem.split('.')[0]
    if label == 'dog':
        label = 1
    elif label == 'cat':
        label = 0

    return img_transformed, label


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,padding=0,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )


        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=0,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()

    def forward(self,x):
        out =self.layer1(x)
        out =self.layer2(out)
        out =self.layer3(out)
        out =out.view(out.size(0),-1)
        out =self.relu(self.fc1(out))
        out =self.fc2(out)
        return out
