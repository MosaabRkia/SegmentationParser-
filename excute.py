import torch
import os
import random
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Configuration
checkpoint_dir = "checkpoints"
nameofmodel = "epoch_1.pth"  # Change to your latest checkpoint file
input_dir = "./input"
output_dir = "./results"
resize_size = (384, 512)  # Resize to (width, height)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = models.segmentation.deeplabv3_resnet101(num_classes=5)  # Ensure correct number of classes
model = model.to(device)

# Handle multi-GPU models
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Load model checkpoint properly
checkpoint_path = os.path.join(checkpoint_dir, nameofmodel)
if not os.path.exists(checkpoint_path):
    print(f"❌ Model checkpoint not found: {checkpoint_path}")
    exit()

print(f"✅ Loading model from: {checkpoint_path}")

# Use `weights_only=True` to avoid pickle security risks
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

# Ensure model_state_dict is loaded correctly
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)  # Fallback if model was saved without optimizer state

model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Pick a random image from input folder
image_files = [f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    print("❌ No images found in input directory!")
    exit()

random_image = random.choice(image_files)
image_path = os.path.join(input_dir, random_image)

print(f"📷 Selected image: {image_path}")

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(image)["out"]
    pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

# Debugging: Print unique values
print("Unique values in prediction:", np.unique(pred))

# If only background class (0) is detected, warn the user
if np.unique(pred).tolist() == [0]:
    print("⚠️ WARNING: Model predicted only background (0). Check if training was successful.")

# Define colors for each class
COLORS = {
    0: (255, 0, 0),  # Red for Left Arm
    1: (0, 255, 0),  # Green for Right Arm
    2: (0, 0, 255),  # Blue for Chest/Middle
    3: (255, 255, 0),  # Yellow for Collar (Front)
    4: (255, 0, 255)   # Magenta for Body Back Parts
}

# Ensure pred shape matches (H, W) for color assignment
pred_h, pred_w = pred.shape
segmented_img = np.zeros((pred_h, pred_w, 3), dtype=np.uint8)  # (Height, Width, 3)

for class_id, color in COLORS.items():
    mask = pred == class_id  # Find all pixels of this class
    segmented_img[mask] = color  # Assign color

# Convert to Image
output_image = Image.fromarray(segmented_img)
output_filename = os.path.join(output_dir, random_image.replace(".", "_segmented."))

# Save segmented image
output_image.save(output_filename)
print(f"✅ Inference completed. Segmented image saved: {output_filename}")
