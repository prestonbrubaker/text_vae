from PIL import Image, ImageDraw, ImageFont
import random
import string
import os

FILE_PATH = "test_photos"

def create_image(image_number):
    global FILE_PATH
    # Create a 256x256 greyscale image with white background
    img = Image.new('L', (256, 256), color=255)  # White background

    # Select a random letter
    letter = random.choice(string.ascii_uppercase)

    # Calculate a random font size (10% to 100% of 128 pixels)
    font_size = random.randint(int(0.1 * 128), 128)

    # Load a font
    font = ImageFont.load_default()

    # Create a draw object
    draw = ImageDraw.Draw(img)

    # Calculate the size of the letter
    text_width, text_height = draw.textsize(letter, font=font)

    # Calculate the position to center the letter
    x = (256 - text_width) // 2
    y = (256 - text_height) // 2

    # Draw the letter
    draw.text((x, y), letter, font=font, fill=0)  # Black text

    # Save the image in the 'photos' subfolder
    img.save(f'{FILE_PATH}/{image_number:05d}_random_letter.png')

# Create the 'photos' subfolder if it doesn't exist
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH)

# Generate and save 1,000 images
for i in range(1000):
    create_image(i)
