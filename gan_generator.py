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
        # Initial size before ConvTranspose layers
        self.init_size = img_size // 16  # Start size (for example, 16x16)
        self.fc = nn.Linear(z_dim, 512 * self.init_size ** 2)  # Prepare input for ConvTranspose

        self.model = nn.Sequential(
            # Input: B x 512*init_size^2 -> B x 512 x init_size x init_size
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # First upsampling
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # -> B x 256 x 2*init_size x 2*init_size
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Second upsampling
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # -> B x 128 x 4*init_size x 4*init_size
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Third upsampling
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> B x 64 x 8*init_size x 8*init_size
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Fourth upsampling to get to 256x256
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),  # -> B x img_channels x 16*init_size x 16*init_size
            nn.Tanh()  # Tanh to get values between -1 and 1
        )

    def forward(self, noise):
        # Transform noise to match ConvTranspose2d input
        noise = self.fc(noise)
        noise = noise.view(noise.size(0), 512, self.init_size, self.init_size)  # Reshape to (batch, channels, H, W)
        img = self.model(noise)
        return img



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu")

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
