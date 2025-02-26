# import os
# import patoolib

# # Function to extract files from RAR archive
# def extract_rar(archive_path, output_dir):
#     try:
#         print('archive_path', archive_path)
#         patoolib.util.executable = "/usr/bin/unrar"  # Set the path to unrar
#         patoolib.extract_archive(archive_path, outdir=output_dir)
#         print(f"Successfully extracted {archive_path} to {output_dir}")
#     except patoolib.util.PatoolError as e:
#         print(f"Failed to extract {archive_path}: {e}")

# # File names and output directory
# file_paths = ["D:\\Downloads\\fresh_projects\SegmentationParser-\\dataset\\file1.rar"]  # Replace with the full paths of your RAR files
# output_dir = "./"  # Output directory for extracted files

# # Extract files from downloaded archives
# for filepath in file_paths:
#     print(filepath)
#     extract_rar(filepath, output_dir)

import os
import patoolib

# Get current folder full path
current_folder = os.getcwd()

# Extract archive into the current folder
patoolib.extract_archive("output.rar", outdir=current_folder)