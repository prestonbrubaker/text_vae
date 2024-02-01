from PIL import Image
import os
import random

def rotate_images(source_folder, target_folder):
    # Check if the target folder exists, if not, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all files in the source folder
    file_list = os.listdir(source_folder)

    for file_name in file_list:
        # Construct full file path
        file_path = os.path.join(source_folder, file_name)
        
        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            # Open the image
            with Image.open(file_path) as img:
                # Rotate the image
                rotated_img = img.rotate(random.uniform(0, 360))

                # Save the rotated image to the target folder
                rotated_img.save(os.path.join(target_folder, file_name))

# Define the source and target folders
source_folder = "photos"
target_folder = "photos_2"

# Rotate the images
rotate_images(source_folder, target_folder)

print("Rotation of images complete. Images are saved in 'photos_2' folder.")
