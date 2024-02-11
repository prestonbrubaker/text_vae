import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import os
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels=1, img_size=256):
        super(Generator, self).__init__()
        self.img_size = img_size
        # Calculate the size of the tensor before the first ConvTranspose2d layer
        self.init_size = img_size // 16
        self.fc = nn.Linear(z_dim, 512 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        # Reshape noise to match the expected input for the first ConvTranspose2d layer
        noise = self.fc(noise)
        noise = noise.view(-1, 512, self.init_size, self.init_size)
        img = self.conv_blocks(noise)
        return img



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

print("Using: " + str(device))


z_dim = 100
img_channels = 1
img_size = 256
generator = Generator(z_dim, img_channels).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

output_folder = "generated_photos"
os.makedirs(output_folder, exist_ok=True)

num_images = 500
with torch.no_grad():
    for i in range(num_images):
        z = torch.randn(1, z_dim, device=device)
        generated_image = generator(z)
        img = generated_image.squeeze(0).cpu().detach()
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(output_folder, f"generated_image_{i+1}.png"))

print(f"Generated {num_images} images in '{output_folder}' folder.")
