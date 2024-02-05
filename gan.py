import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3):
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

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: img_channels x 256 x 256
            nn.Conv2d(img_channels, 64, 4, 2, 1),  # img: 128x128
            nn.LeakyReLU(0.2),
            self._block(64, 128, 4, 2, 1),    # img: 64x64
            self._block(128, 256, 4, 2, 1),   # img: 32x32
            self._block(256, 512, 4, 2, 1),   # img: 16x16
            nn.Conv2d(512, 1, 4, 1, 0),  # img: 1x1
            nn.Sigmoid()  # Output: 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x).view(-1)

# Hyperparameters
z_dim = 100
learning_rate = 0.0002
batch_size = 64
img_channels = 3
img_size = 256
num_epochs = 50

# Initialize generator and discriminator
generator = Generator(z_dim, img_channels).to(device)
discriminator = Discriminator(img_channels).to(device)

# Optimizers
opt_gen = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_disc = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Move models to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Assuming your dataset is in a folder called 'dataset_folder'
dataset = datasets.ImageFolder(root='photos_2', transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake = generator(noise)

        ### Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
        disc_real = discriminator(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        discriminator.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = discriminator(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")
