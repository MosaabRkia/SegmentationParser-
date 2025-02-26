import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from architecture.segnet import SegNet
from torchvision.transforms import ToTensor, Resize
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
WIDTH_EVAL = 768
HEIGHT_EVAL = 576
WIDTH_SAVE = 768
HEIGHT_SAVE = 1024

class Fashion_Data(Dataset):
    def __init__(self, folder_train, resize_shape_eval=(WIDTH_EVAL, HEIGHT_EVAL), resize_shape_save=(HEIGHT_SAVE, WIDTH_SAVE), augmentation=None):
        self.folder_train = folder_train
        # Define two different resize transforms
        self.transform_eval = transforms.Compose([
            Resize(resize_shape_eval),
            transforms.ToTensor()
        ])
        self.transform_save = transforms.Compose([
            Resize(resize_shape_save),
            transforms.ToTensor()
        ])
        self.filenames = [f for f in os.listdir(folder_train) if os.path.isfile(os.path.join(folder_train, f))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name_train = os.path.join(self.folder_train, self.filenames[idx])
        img_train = Image.open(img_name_train).convert('L')
        # Apply the appropriate resize transform based on the purpose
        img_train_eval = self.transform_eval(img_train)
        img_train_save = self.transform_save(img_train)
        return img_train_eval, img_train_save

test_folder = "logs/input/"
# Define custom resize dimensions for evaluation and saving
resize_width_eval = WIDTH_EVAL
resize_height_eval = HEIGHT_EVAL
resize_width_save = WIDTH_SAVE
resize_height_save = HEIGHT_SAVE
test_set = Fashion_Data(test_folder, resize_shape_eval=( resize_height_eval, resize_width_eval), resize_shape_save=( resize_height_save, resize_width_save))
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Load the pre-trained SegNet model
generator = SegNet(in_channels=1, out_channels=1).to(device)
generator = DataParallel(generator)
generator.load_state_dict(torch.load(f'models/{generator_name_to_load}'))

# Iterate over the test dataset and generate images
for idx, data in enumerate(test_dataloader):
    input_images_eval, input_images_save = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.float32)
    
    # Generate output images from the input
    with torch.no_grad():
        output_images = generator(input_images_save)  # Use input_images_save for generating output images
    
    # Save the resized generated images with custom filenames
    for i in range(len(output_images)):
        output_filename = f"generated_{idx}_{i}_{generator_name_to_load}_resized.png"  # Generate the output filename
        save_image(output_images[i], os.path.join("logs", "output", output_filename))
