import os
import gdown

# Function to download file from Google Drive
def download_file(file_id, output_filename):
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Downloading from: {url}")
    print(f"To: {output_filename}")
    gdown.download(url, output_filename, quiet=False)
    print("Download completed.")

# File IDs and output filenames
file_ids = ["1Y_39YXWHtKIHvSe_p3GDdGSO07Q-rpOv"]  # Replace with your file IDs
output_filenames = ["file1.rar"]  # Output filenames for downloaded files

# Download files
for file_id, output_filename in zip(file_ids, output_filenames):
    print(f"Downloading file with ID: {file_id}")
    download_file(file_id, output_filename)
