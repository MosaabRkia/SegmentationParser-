import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from architecture.segnet import SegNet
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import Dataset
import os
from torch.nn import DataParallel
from tqdm import tqdm
from torchvision import transforms

generator_name_to_load = "final_model_mosaab.pth"
batch_size = 20
device = "cuda"

class Fashion_Data(Dataset):
    def __init__(self, folder_train, augmentation=None):
        self.folder_train = folder_train
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.filenames = [f for f in os.listdir(folder_train) if os.path.isfile(os.path.join(folder_train, f))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name_train = os.path.join(self.folder_train, self.filenames[idx])
        img_train = Image.open(img_name_train).convert('L')
        img_train = self.transform(img_train)
        return img_train

test_folder = "logs/input/"
test_set = Fashion_Data(test_folder)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Load the pre-trained SegNet model
generator = SegNet(in_channels=1, out_channels=1).to(device)
generator = DataParallel(generator)
generator.load_state_dict(torch.load(f'models/{generator_name_to_load}'))
generator.eval()

# Iterate over the test dataset and generate images
for idx, data in enumerate(test_dataloader):
    input_images = data.to(device, dtype=torch.float32)
    
    # Generate output images from the input
    with torch.no_grad():
        output_images = generator(input_images)
    
    # Save the generated images with the name of the corresponding input image
    for i in range(len(output_images)):
        output_filename = f"generated_{idx}_{i}_{generator_name_to_load}.png"  # Generate the output filename
        save_image(output_images[i], os.path.join("logs", "output", output_filename))
