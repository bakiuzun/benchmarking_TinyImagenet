"""
Author: Uzun Baki
"""
import os
import shutil

"""
this file was used to reorganize the tiny imagenet dataset to get at the end a structure like this:

train/
    classA/
        img1
        img2
        ...
    classB/
        img1
        img2
        ...
val/
    IDEM as Train
"""


folder_path = 'images_dataset/train'

# Get a list of all directories in the folder
directories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]


# Iterate over the directories
for directory in directories:
    # Define the source and destination directories
    source_dir = os.path.join(folder_path+"/"+directory, 'images')
    dest_dir = folder_path+"/"+directory
    
    # Iterate over the files in the source directory and move them to the destination directory
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)

        if filename.endswith(".txt"):
            os.remove(source_path)
        else:
            dest_path = os.path.join(dest_dir, filename)
            shutil.move(source_path, dest_path)
    
    # Remove the source directory
    os.rmdir(source_dir)

folder_path = "images_dataset/val"

with open(f'{folder_path}/val_annotations.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        filename = parts[0]
        label = parts[1]

        dir_path = os.path.join(folder_path, label)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


        shutil.move(f"{folder_path}/images/{filename}", dir_path)

        
