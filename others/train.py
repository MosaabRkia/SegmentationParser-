import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import models
from PIL import Image
import numpy as np
import random
import glob
from tqdm import tqdm

# Define constants
IMAGE_SIZE =  (512, 512)
NUM_CLASSES = 5
BATCH_SIZE = 10
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICES_IDS = [0, 1]
NUM_GPUS = torch.cuda.device_count()

print('IMAGE_SIZE:', IMAGE_SIZE)
print('NUM_CLASSES:', NUM_CLASSES)
print('BATCH_SIZE:', BATCH_SIZE)
print('NUM_EPOCHS:', NUM_EPOCHS)
print('LEARNING_RATE:', LEARNING_RATE)
print('DEVICE:', DEVICE)
print('NUM_GPUS:', NUM_GPUS)

# Define colors for each class
class_colors = {
    0: [22, 22, 22],   # Left arm
    1: [21, 21, 21],   # Right arm
    2: [5, 5, 5],      # Middle part
    3: [24, 24, 24],   # Collar
    4: [25, 25, 25]    # Body back parts
}

# Dataset class
class ClothDataset(Dataset):
    def __init__(self, root_dir, target_size=(512, 512)):
        self.image_files = glob.glob(os.path.join(root_dir, "input", "*.jpg"))
        self.mask_files = glob.glob(os.path.join(root_dir, "output", "*.png"))
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_size = IMAGE_SIZE

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx]).convert("RGB")

        image = self.transform(image)
        mask = transforms.Resize(self.target_size, interpolation=Image.NEAREST)(mask)
        mask_tensor = self.mask_to_tensor(mask)

        return image, mask_tensor

    def mask_to_tensor(self, mask):
        mask_tensor = torch.zeros((NUM_CLASSES, self.target_size[0], self.target_size[1]), dtype=torch.float)
        for i, color in enumerate(class_colors.values()):
            color_tensor = torch.tensor(color, dtype=torch.uint8).unsqueeze(1).unsqueeze(2)
            mask_single_channel = torch.eq(transforms.ToTensor()(mask), color_tensor).all(dim=0)
            mask_tensor[i] = mask_single_channel.float()

        return mask_tensor


# Define the model architecture
class ClothParserModel(nn.Module):
    def __init__(self):
        super(ClothParserModel, self).__init__()
        self.base_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        # Remove batch normalization layers
        self.base_model.apply(self.remove_batchnorm)

        # Replace the classifier's last layer to adapt to the number of classes
        self.base_model.classifier[-1] = nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))

    def remove_batchnorm(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            # Replace BatchNorm layers with identity
            return nn.Identity()

    def forward(self, x):
        return self.base_model(x)['out']

def main():
    # Create dataset and dataloaders
    dataset = ClothDataset("dataset")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    # model = ClothParserModel().to(DEVICE)
    model = ClothParserModel().to(DEVICE)
    model = nn.DataParallel(model, device_ids=DEVICES_IDS)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Train the model
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

        # Save model checkpoint
        if epoch + 1 % 10 == 0:
            torch.save(model.state_dict(), f"cloth_parser_epoch_{epoch + 1}.pth")
        else: 
            torch.save(model.state_dict(), f"cloth_parser_epoch_master.pth")

if __name__ == "__main__":
    main()
