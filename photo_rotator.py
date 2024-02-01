from PIL import Image
import numpy as np
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
            # Open the image and convert it to a numpy array
            with Image.open(file_path) as img:
                img_array = np.array(img)

                # Rotate the image using numpy
                angle = random.uniform(0, 360)
                rotated_array = np.rot90(img_array, k=int(angle / 90))

                # Convert the numpy array back to an image
                rotated_img = Image.fromarray(rotated_array)

                # Save the rotated image to the target folder
                rotated_img.save(os.path.join(target_folder, file_name))

# Define the source and target folders
source_folder = "photos"
target_folder = "photos_2"

# Rotate the images
rotate_images(source_folder, target_folder)

print("Rotation of images complete. Images are saved in 'photos_2' folder.")
