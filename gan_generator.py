import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import os
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention layer for the generator."""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.scale = 1.0 / (in_dim ** 0.5)

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query_conv(x).view(batch, -1, height*width).permute(0, 2, 1)  # [batch, seq_len, depth]
        key = self.key_conv(x).view(batch, -1, height*width)  # [batch, depth, seq_len]
        value = self.value_conv(x).view(batch, -1, height*width).permute(0, 2, 1)  # [batch, seq_len, depth]
    
        # Ensure the dimensionality matches for bmm
        attention = torch.bmm(query, key) * self.scale  # [batch, seq_len, seq_len]
        attention = F.softmax(attention, dim=-1)
        
        # Adjust value or the permute operation to match bmm expectations
        value_permuted = value.permute(0, 2, 1)  # Permute to match the expected dimensions for bmm
        out = torch.bmm(value_permuted, attention.permute(0, 2, 1))  # Now [batch, depth, seq_len]
        out = out.permute(0, 2, 1).reshape(batch, channels, height, width)  # Use .reshape() instead of .view()
    
        return out + x  # Skip connection

class ResBlock(nn.Module):
    """Residual block for the generator."""
    def __init__(self, in_channels, out_channels, upsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = upsample
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.upsample:
            out = F.interpolate(out, scale_factor=self.upsample, mode='nearest')
            residual = F.interpolate(residual, scale_factor=self.upsample, mode='nearest')
        if self.adjust_channels:
            residual = self.adjust_channels(residual)
        out += residual
        return F.relu(out)

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels):
        super(Generator, self).__init__()
        self.init_size = 256 // 16  # This will give a starting size of 16x16
        self.l1 = nn.Sequential(nn.Linear(z_dim, 128 * self.init_size ** 2))

        self.model = nn.Sequential(
            ResBlock(128, 128, upsample=2),
            SelfAttention(128),
            ResBlock(128, 64, upsample=2),
            ResBlock(64, 32, upsample=2),
            SelfAttention(32),
            ResBlock(32, img_channels, upsample=2),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        # Adjusted reshaping based on the corrected initial size calculation
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)  # Reshape to (batch_size, 128, 16, 16)
        img = self.model(out)
        return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu")

print("Using: " + str(device))


z_dim = 20
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
