import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision import models

NUM_CLASSES = 5

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


# Load the trained model
model = ClothParserModel()

# Load the state dictionary, ignoring missing keys
state_dict = torch.load("cloth_parser_epoch_master.pth")
model.load_state_dict(state_dict, strict=False)

model.eval()

# Define colors for each class
class_colors = {
    0: [22, 22, 22],   # Left arm
    1: [21, 21, 21],   # Right arm
    2: [5, 5, 5],      # Middle part
    3: [24, 24, 24],   # Collar
    4: [25, 25, 25]    # Body back parts
}

# Define transforms for preprocessing the input image
transform = transforms.Compose([
    transforms.Resize((1024, 768)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the input image
input_image = Image.open("00000_00.jpg").convert("RGB")
input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Post-process the output masks to generate the final image
output_masks = torch.sigmoid(output[0]).cpu().numpy()  # Sigmoid to convert logits to probabilities
output_masks = (output_masks > 0.5).astype(np.uint8)  # Thresholding
output_image = np.zeros((1024, 768, 3), dtype=np.uint8)
for i, color in enumerate(class_colors.values()):
    mask = output_masks[i]
    output_image[mask == 1] = color

# Save the final image
output_image = Image.fromarray(output_image)
output_image.save("output_image.jpg")
