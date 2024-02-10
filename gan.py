import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
from PIL import Image
import os
import random
import torch.nn.functional as F

torch.cuda.empty_cache()


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



class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: img_channels x 256 x 256
            nn.Conv2d(img_channels, 64, 4, 2, 1),  # Output: 64 x 128 x 128
            nn.LeakyReLU(0.01),
            self._block(64, 128, 4, 2, 1),         # Output: 128 x 64 x 64
            self._block(128, 256, 4, 2, 1),        # Output: 256 x 32 x 32
            self._block(256, 512, 4, 2, 1),        # Output: 512 x 16 x 16
            nn.Conv2d(512, 512, 4, 2, 1),          # Output: 512 x 8 x 8
            nn.LeakyReLU(0.01),
            nn.Conv2d(512, 512, 4, 2, 1),          # Output: 512 x 4 x 4
            nn.LeakyReLU(0.01),
            # Ensure the final output is 1x1
            nn.Conv2d(512, 1, 4, 1, 0),            # Output: 1 x 1 x 1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        x = self.disc(x)
        return x.view(x.size(0), -1)



# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: " + str(device))

# Hyperparameters
z_dim = 100
learning_rate_gen = 0.001
learning_rate_disc = 0.00001
batch_size = 5
img_channels = 1
img_size = 256
num_epochs = 5000

# Initialize generator and discriminator
generator = Generator(z_dim=z_dim, img_channels=img_channels).to(device)
discriminator = Discriminator(img_channels=img_channels).to(device)

# Attempt to load existing models
generator_path = 'generator.pth'
discriminator_path = 'discriminator.pth'

if os.path.exists(generator_path):
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    print("Generator model loaded.")
else:
    print("No saved generator model found. Initializing a new one.")

if os.path.exists(discriminator_path):
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    print("Discriminator model loaded.")
else:
    print("No saved discriminator model found. Initializing a new one.")

# Optimizers
opt_gen = optim.Adam(generator.parameters(), lr=learning_rate_gen, betas=(0.5, 0.999))
opt_disc = optim.Adam(discriminator.parameters(), lr=learning_rate_disc, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Move models to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
])

dataset = CustomImageDataset(root_dir='photos_2', transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


lambda_gp = 10  # Gradient penalty lambda hyperparameter

for epoch in range(num_epochs):
    for batch_idx, real in enumerate(loader):
        real = real.to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake = generator(noise)

        # Adjust targets to match discriminator output shape [batch_size, 1]
        real_labels = torch.ones(batch_size, 1, device=device)  # Shape [100, 1] for real images
        fake_labels = torch.zeros(batch_size, 1, device=device)  # Shape [100, 1] for fake images

        ### Train Discriminator with Gradient Penalty
        discriminator.zero_grad()
        real_output = discriminator(real)
        fake_output = discriminator(fake.detach())
        
        real_loss = criterion(real_output, real_labels)
        fake_loss = criterion(fake_output, fake_labels)
        
        # Calculate gradient penalty on interpolated data
        loss_disc = real_loss + fake_loss

        loss_disc.backward()
        opt_disc.step()


        generator.zero_grad()
        # Discriminator output for generated images
        gen_output = discriminator(fake)
        gen_loss = criterion(gen_output, real_labels)
        gen_loss.backward()
        opt_gen.step()
        
    # Logging
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_disc:.4f}, Loss G: {gen_loss:.4f}")
    with open('model_history.txt', 'a') as file:
        file.write(f"Epoch {epoch+1} Loss D: {loss_disc:.4f}, Loss G: {gen_loss:.4f} \n")
    if (epoch + 1) % 1 == 0:
        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(discriminator.state_dict(), 'discriminator.pth')
        

