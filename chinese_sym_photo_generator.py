from PIL import Image, ImageDraw, ImageFont
import random
import os

FILE_PATH = "photos"

# List of 500 unique Chinese characters
# This is just an example list - replace with your desired characters
chinese_chars = [
    '的', '一', '是', '不', '了', '在', '人', '有', '我', '他', 
    # ... (add more characters to reach at least 500 unique ones)
]

def create_image(image_number, char):
    global FILE_PATH
    # Create a 256x256 greyscale image with white background
    img = Image.new('L', (256, 256), color=255)

    # Calculate a random font size (20% to 100% of 128 pixels)
    font_size = random.randint(int(0.2 * 128), 128)

    # Load a font that supports Chinese characters
    font = ImageFont.truetype("/path/to/NotoSansCJK-Regular.ttc", font_size)

    # Create a draw object
    draw = ImageDraw.Draw(img)

    # Calculate the size of the character
    text_width, text_height = draw.textsize(char, font=font)

    # Calculate the position to center the character
    x = (256 - text_width) // 2
    y = (256 - text_height) // 2

    # Draw the character
    draw.text((x, y), char, font=font, fill=0)

    # Save the image in the 'photos' subfolder
    img.save(f'{FILE_PATH}/{image_number:05d}_random_chinese.png')

# Create the 'photos' subfolder if it doesn't exist
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH)

# Generate and save images for each character
for i, char in enumerate(chinese_chars):
    create_image(i, char)
