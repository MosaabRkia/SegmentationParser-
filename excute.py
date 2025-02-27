import torch
import os
import numpy as np
import random
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
nameofmodel = "garment_segmentation_model.pth"  # Model filename
checkpoint_dir = "checkpoints"  # Where the model checkpoints are stored
input_dir = "./input"  # Directory containing input images
output_dir = "./results"  # Where output images will be saved
sizeToTrainOn = (192, 256)  # Image size used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(log_dir, "inference_log.txt")

# Logging function
def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# Load model
def load_model():
    model = models.segmentation.deeplabv3_resnet101(weights=None, num_classes=5)
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")], key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not checkpoint_files:
        log_message("❌ No checkpoint found. Make sure training has completed.")
        exit(1)

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
    log_message(f"✅ Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model

# Image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Select a random image from input
def get_random_image():
    images = [f for f in os.listdir(input_dir) if f.endswith(".png") or f.endswith(".jpg")]
    if not images:
        log_message("❌ No input images found in the folder!")
        exit(1)
    
    selected_image = random.choice(images)
    image_path = os.path.join(input_dir, selected_image)
    log_message(f"📷 Selected image: {selected_image}")
    return image_path

# Inference function
def run_inference(model, image_path):
    image = Image.open(image_path).convert("RGB").resize(sizeToTrainOn)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)["out"]
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    log_message(f"🧐 Unique values in prediction: {np.unique(pred)}")

    return pred

# Color mapping for visualization
COLOR_PALETTE = {
    0: (22, 22, 22),  # Left Arm
    1: (21, 21, 21),  # Right Arm
    2: (5, 5, 5),     # Chest/Middle
    3: (24, 24, 24),  # Collar (Front)
    4: (25, 25, 25)   # Body Back Parts
}

# Save segmented output
def save_segmented_output(image_path, prediction):
    segmented_img = np.zeros((*sizeToTrainOn, 3), dtype=np.uint8)
    
    for class_id, color in COLOR_PALETTE.items():
        segmented_img[prediction == class_id] = color  # Apply colors

    segmented_pil = Image.fromarray(segmented_img)
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_segmented.png"
    output_filepath = os.path.join(output_dir, output_filename)
    segmented_pil.save(output_filepath)
    
    log_message(f"✅ Segmented image saved: {output_filepath}")

# Main execution
if __name__ == "__main__":
    model = load_model()
    image_path = get_random_image()
    pred = run_inference(model, image_path)
    save_segmented_output(image_path, pred)
