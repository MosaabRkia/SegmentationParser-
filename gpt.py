# Import necessary libraries
import torch
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
    mask = np.array(mask)
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for color, class_idx in COLOR_MAP.items():
        class_mask[np.all(mask == color, axis=-1)] = class_idx
    return class_mask

# Custom Dataset class for loading the images and masks
class GarmentSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        # Adjusting mask file extension from .jpg to .png
        mask_filename = os.path.splitext(self.images[idx])[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_filename)

        image = Image.open(image_path).convert("RGB").resize((1024, 768))
        mask = Image.open(mask_path).resize((1024, 768))
        mask = color_mask_to_class(mask)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()

        return image, mask

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = GarmentSegmentationDataset("./dataset/input", "./dataset/mask", transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Define the DeepLabV3+ model with ResNet-101 backbone
# model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=5)
model = models.segmentation.deeplabv3_resnet50(weights=None, progress=True, num_classes=5)

model = model.cuda() if torch.cuda.is_available() else model

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in tqdm(dataloader):
            images, masks = images.cuda(), masks.cuda()
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Training the model
train_model(model, train_loader, criterion, optimizer)

# Save the trained model
torch.save(model.state_dict(), "garment_segmentation_model.pth")

# Inference function for predictions
def inference(model, image_path, output_dir="./dataset/output"):
    model.eval()
    image = Image.open(image_path).convert("RGB").resize((1024, 768))
    image = transform(image).unsqueeze(0)
    image = image.cuda() if torch.cuda.is_available() else image

    with torch.no_grad():
        output = model(image)['out']
        pred = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    # Save output as PNG
    output_image = Image.fromarray((pred * 50).astype(np.uint8))
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_segmented.png"
    output_image.save(os.path.join(output_dir, output_filename))

    return pred

# Visualization function
def visualize_segmentation(image_path, prediction):
    image = Image.open(image_path).resize((1024, 768))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    # Define the RGB mapping for each segment
    colors = {
        0: (22, 22, 22),  # Left Arm
        1: (21, 21, 21),  # Right Arm
        2: (5, 5, 5),     # Chest/Middle
        3: (24, 24, 24),  # Collar (Front)
        4: (25, 25, 25)   # Body Back Parts
    }

    # Create a color mask
    segmented_img = np.zeros((768, 1024, 3), dtype=np.uint8)
    for c in range(5):
        segmented_img[prediction == c] = colors[c]

    plt.subplot(1, 2, 2)
    plt.title("Segmented Output")
    plt.imshow(segmented_img)
    plt.axis("off")
    plt.show()

# Example usage for inference and visualization
# pred = inference(model, "./input/sample_image.jpg")
# visualize_segmentation("./input/sample_image.jpg", pred)
