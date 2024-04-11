import os
import gdown
from rarfile import RarFile, NotRarFile

# Function to download file from Google Drive
def download_file(file_id, output_filename):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_filename, quiet=False)

# Function to extract files from RAR archive
def extract_rar(archive_path, output_dir):
    try:
        with RarFile(archive_path) as rar:
            rar.extractall(output_dir)
    except NotRarFile:
        print(f"{archive_path} is not a RAR file.")

# File IDs and output filenames
file_ids = ["1Y_39YXWHtKIHvSe_p3GDdGSO07Q-rpOv"]  # Replace with your file IDs
output_filenames = ["file1.rar", "file2.rar", "file3.rar"]  # Output filenames for downloaded files

# Download files
for file_id, output_filename in zip(file_ids, output_filenames):
    download_file(file_id, output_filename)

# Extract files from downloaded archives
for filename in output_filenames:
    extract_rar(filename, os.path.splitext(filename)[0])  # Extract to a directory with the same name as the RAR file (without extension)
