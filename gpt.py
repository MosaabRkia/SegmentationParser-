import torch
torch.autograd.set_detect_anomaly(True)
print(torch.version.cuda)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
nameofmodel = "garment_segmentation_model.pth"  # Model filename
completeorscratch = False  # True to resume training, False to start from scratch
batch_size = 1  # Batch size for training
epochs = 30  # Number of epochs
learning_rate = 0.0001  # Learning rate
checkpoint_dir = "checkpoints"  # Directory to save checkpoints
sizeToTrainOn = (192, 256)  # Resize images to this size
OneGpu = True  # Set to True to force using only GPU 0

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_log.txt")

# Logging function
def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# Detect number of GPUs
num_gpus = torch.cuda.device_count()

# Color to class mapping
COLOR_MAP = {
    (22, 22, 22): 0,  # Left Arm
    (21, 21, 21): 1,  # Right Arm
    (5, 5, 5): 2,     # Chest/Middle
    (24, 24, 24): 3,  # Collar (Front)
    (25, 25, 25): 4   # Body Back Parts
}

# Convert color mask to class mask
def color_mask_to_class(mask):
    if len(mask.shape) == 2:  # If the mask is grayscale (single channel)
        mask = np.stack([mask] * 3, axis=-1)  # Convert to 3-channel RGB-like format

    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    
    for color, class_idx in COLOR_MAP.items():
        class_mask[np.all(mask == color, axis=-1)] = class_idx
    
    return class_mask

# Custom Dataset class
class GarmentSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_filename = os.path.splitext(self.images[idx])[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_filename)

        image = Image.open(image_path).convert("RGB").resize(sizeToTrainOn)
        mask = Image.open(mask_path)
        if mask.mode != "RGB":
            mask = mask.convert("RGB")  # Ensure masks are RGB
        mask = mask.resize(sizeToTrainOn)
        mask = color_mask_to_class(np.array(mask))

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()

        return image, mask

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = GarmentSegmentationDataset("./input", "./mask", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the DeepLabV3+ model
model = models.segmentation.deeplabv3_resnet101(weights=None, progress=True, num_classes=5)

# Set GPU usage
if OneGpu:
    log_message("🟢 Using only GPU 0")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_gpus > 1:
        log_message(f"🔵 Using {num_gpus} GPUs")
        model = nn.DataParallel(model)

# Move model to GPU
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to save checkpoint
def save_checkpoint(epoch, model, optimizer, loss, path=checkpoint_dir):
    os.makedirs(path, exist_ok=True)
    checkpoint_file = os.path.join(path, f"epoch_{epoch+1}.pth")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_file)
    log_message(f"✅ Checkpoint saved: {checkpoint_file}")

# Function to load latest checkpoint
def load_checkpoint(path=checkpoint_dir):
    if completeorscratch and os.path.exists(path):
        checkpoint_files = sorted([f for f in os.listdir(path) if f.endswith(".pth")], key=lambda x: int(x.split('_')[1].split('.')[0]))
        if checkpoint_files:
            latest_checkpoint = os.path.join(path, checkpoint_files[-1])
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
            log_message(f"🔄 Resuming training from {latest_checkpoint}")
            return start_epoch
    return 0

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=epochs, start_epoch=0):
    model.train()
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        log_message(f"📌 Starting Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader)):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            # Log batch loss every 100 batches
            if batch_idx % 100 == 0:
                log_message(f"📝 Batch {batch_idx}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader.dataset)
        log_message(f"📉 Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        save_checkpoint(epoch, model, optimizer, epoch_loss)

# Load checkpoint if exists
start_epoch = load_checkpoint(checkpoint_dir)

# Train model
train_model(model, train_loader, criterion, optimizer, num_epochs=epochs, start_epoch=start_epoch)

# Save the final trained model
torch.save(model.state_dict(), nameofmodel)
log_message(f"✅ Final trained model saved as {nameofmodel}")
