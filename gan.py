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


import torch
import torch.nn as nn

class ConvGenerator(nn.Module):
    def __init__(self, z_dim, img_channels=1, img_size=256):
        super(ConvGenerator, self).__init__()
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
learning_rate_gen = 0.01
learning_rate_disc = 0.0001
batch_size = 10
img_channels = 1
img_size = 256
num_epochs = 5000

# Initialize generator and discriminator
generator = ConvGenerator(z_dim, img_channels).to(device)


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

weight_decay = 1e-5

opt_gen = optim.Adam(generator.parameters(), lr=learning_rate_gen, betas=(0.5, 0.999), weight_decay=weight_decay)
opt_disc = optim.Adam(discriminator.parameters(), lr=learning_rate_disc, betas=(0.5, 0.999), weight_decay=weight_decay)


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

def transfer_model_parameters(source_model, target_model):
    target_model.load_state_dict(source_model.state_dict())

for epoch in range(num_epochs):
    for batch_idx, real in enumerate(loader):
        
        #print(str(batch_idx))
        
        real = real.to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, z_dim)
        noise = noise.to(device)
        fake = generator(noise)

        real_labels = torch.ones(batch_size, 1, device=device)  # Shape [100, 1] for real images
        fake_labels = torch.zeros(batch_size, 1, device=device)  # Shape [100, 1] for fake images
        

        
        # Discriminator Loss
        disc_real = discriminator(real).view(-1)
        loss_disc_real = criterion(disc_real, real_labels.view(-1))

        
        disc_fake = discriminator(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, fake_labels.view(-1))
        

        
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        loss_disc.backward()
        opt_disc.step()
        


        # Generator Loss
        gen_output = discriminator(fake).view(-1)
        gen_loss = criterion(gen_output, real_labels.view(-1))
        

        
        total_gen_loss = gen_loss
        total_gen_loss.backward()
        opt_gen.step()
        

        # Intra-epoch logging
        #print(f"Epoch {epoch+1} Sub: {batch_idx} Loss D: {loss_disc:.4f}, Loss G: {total_gen_loss:.4f}")
        with open('expanded_model_history.txt', 'a') as file:
            file.write(f"Epoch {epoch+1} Sub: {batch_idx} Loss D: {loss_disc:.4f}, Loss G: {total_gen_loss:.4f} \n")





    if (epoch + 1) % 1 == 0:
        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(discriminator.state_dict(), 'discriminator.pth')
        print("Models all saved")
    
    
        
        
    # Logging
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_disc:.4f}, Loss G: {total_gen_loss:.4f}")
    with open('model_history.txt', 'a') as file:
        file.write(f"Epoch {epoch+1} Loss D: {loss_disc:.4f}, Loss G: {total_gen_loss:.4f} \n")
    
        

