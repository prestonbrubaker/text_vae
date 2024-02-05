import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import os

class Generator(nn.Module):
    def __init__(self, z_dim=27, img_channels=1):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: Z_dim x 1 x 1
            self._block(z_dim, 512, 4, 1, 0),  # img: 4x4
            self._block(512, 256, 4, 2, 1),    # img: 8x8
            self._block(256, 128, 4, 2, 1),    # img: 16x16
            self._block(128, 64, 4, 2, 1),     # img: 32x32
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),  # img: 64x64
            nn.ConvTranspose2d(img_channels, img_channels, 16, 4, 6),  # img: 256x256
            nn.Tanh()  # Output: img_channels x 256 x 256
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.gen(x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 27
generator = Generator(z_dim).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

output_folder = "generated_photos"
os.makedirs(output_folder, exist_ok=True)

num_images = 500
with torch.no_grad():
    for i in range(num_images):
        z = torch.randn(1, z_dim, 1, 1, device=device)
        generated_image = generator(z)
        img = generated_image.squeeze(0).cpu().detach()
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(output_folder, f"generated_image_{i+1}.png"))

print(f"Generated {num_images} images in '{output_folder}' folder.")
