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

class Generator_2(nn.Module):
    def __init__(self, z_dim, img_channels):
        super(Generator_2, self).__init__()
        self.init_size = 256 // 16  # This will give a starting size of 16x16
        self.l1 = nn.Sequential(nn.Linear(z_dim_2, 128 * self.init_size ** 2))

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




class Discriminator_2(nn.Module):
    def __init__(self, img_channels=1):
        super(Discriminator_2, self).__init__()
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
z_dim = 200
z_dim_2 = 200
learning_rate_gen = 0.05
learning_rate_gen_2 = 0.00000001
learning_rate_disc = 0.0005
learning_rate_disc_2 = 0.00000001
batch_size = 10
img_channels = 1
img_size = 256
num_epochs = 5000

# Initialize generator and discriminator
generator = Generator(z_dim=z_dim, img_channels=img_channels).to(device)
generator_2 = Generator_2(z_dim=z_dim, img_channels=img_channels).to(device)


discriminator = Discriminator(img_channels=img_channels).to(device)
discriminator_2 = Discriminator_2(img_channels=img_channels).to(device)

# Attempt to load existing models
generator_path = 'generator.pth'
generator_2_path = 'generator_2.pth'
discriminator_path = 'discriminator.pth'
discriminator_2_path = 'discriminator_2.pth'

if os.path.exists(generator_path):
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    print("Generator model loaded.")
else:
    print("No saved generator model found. Initializing a new one.")

if os.path.exists(generator_2_path):
    generator.load_state_dict(torch.load(generator_2_path, map_location=device))
    print("Generator_2 model loaded.")
else:
    print("No saved generator_2 model found. Initializing a new one.")

if os.path.exists(discriminator_path):
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    print("Discriminator model loaded.")
else:
    print("No saved discriminator model found. Initializing a new one.")

if os.path.exists(discriminator_2_path):
    discriminator.load_state_dict(torch.load(discriminator_2_path, map_location=device))
    print("Discriminator_2 model loaded.")
else:
    print("No saved discriminator_2 model found. Initializing a new one.")



# Optimizers

weight_decay = 1e-5

opt_gen = optim.Adam(generator.parameters(), lr=learning_rate_gen, betas=(0.5, 0.999), weight_decay=weight_decay)
opt_gen_2 = optim.Adam(generator_2.parameters(), lr=learning_rate_gen_2, betas=(0.5, 0.999), weight_decay=weight_decay)
opt_disc = optim.Adam(discriminator.parameters(), lr=learning_rate_disc, betas=(0.5, 0.999), weight_decay=weight_decay)
opt_disc_2 = optim.Adam(discriminator_2.parameters(), lr=learning_rate_disc_2, betas=(0.5, 0.999), weight_decay=weight_decay)


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
        noise = torch.randn(batch_size, z_dim, device=device)
        fake = generator(noise)

        noise_2 = torch.randn(batch_size, z_dim_2, device=device)
        fake_2 = generator_2(noise_2)

        # Adjust targets to match discriminator output shape [batch_size, 1]
        real_labels = torch.ones(batch_size, 1, device=device)  # Shape [100, 1] for real images
        fake_labels = torch.zeros(batch_size, 1, device=device)  # Shape [100, 1] for fake images

        # Adjust targets to match discriminator output shape [batch_size, 1]
        real_labels_2 = torch.ones(batch_size, 1, device=device)  # Shape [100, 1] for real images
        fake_labels_2 = torch.zeros(batch_size, 1, device=device)  # Shape [100, 1] for fake images
        
        # Discriminator 1 Loss
        disc_real = discriminator(real).view(-1)
        loss_disc_real = criterion(disc_real, real_labels.view(-1))
        
        disc_fake = discriminator(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, fake_labels.view(-1))
        
        disc_fake_2 = discriminator(fake_2.detach()).view(-1)
        loss_disc_fake_2 = criterion(disc_fake_2, fake_labels.view(-1))

        
        loss_disc = (loss_disc_real + (loss_disc_fake + loss_disc_fake_2) / 2) / 2
        loss_disc.backward()
        opt_disc.step()
        
        # Discriminator 2 Loss
        disc_real_2 = discriminator_2(real).view(-1)
        loss_disc_real_2 = criterion(disc_real_2, real_labels_2.view(-1))
        
        disc_fake = discriminator_2(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, fake_labels_2.view(-1))
        
        disc_fake_2 = discriminator_2(fake_2.detach()).view(-1)
        loss_disc_fake_2 = criterion(disc_fake_2, fake_labels_2.view(-1))
        
        loss_disc_2 = (loss_disc_real_2 + (loss_disc_fake + loss_disc_fake_2) / 2) / 2
        loss_disc_2.backward()
        opt_disc_2.step()


        # Generator 1 Loss
        gen_output = discriminator(fake).view(-1)
        gen_loss = criterion(gen_output, real_labels.view(-1))
        
        gen_output_2 = discriminator_2(fake).view(-1)
        gen_loss_2 = criterion(gen_output_2, real_labels.view(-1))

        
        total_gen_loss = (gen_loss + gen_loss_2) / 2
        total_gen_loss.backward()
        opt_gen.step()
        
        # Generator 2 Loss
        gen_output = discriminator(fake_2).view(-1)
        gen_loss = criterion(gen_output, real_labels_2.view(-1))
        
        gen_output_2 = discriminator_2(fake_2).view(-1)
        gen_loss_2 = criterion(gen_output_2, real_labels_2.view(-1))
        
        total_gen_loss_2 = (gen_loss + gen_loss_2) / 2
        total_gen_loss_2.backward()
        opt_gen_2.step()

        # Intra-epoch logging
        #print(f"Epoch {epoch+1} Sub: {batch_idx} Loss D: {loss_disc:.4f}, Loss G: {total_gen_loss:.4f} Loss D2: {loss_disc_2:.4f}, Loss G2: {total_gen_loss_2:.4f}")
        with open('expanded_model_history.txt', 'a') as file:
            file.write(f"Epoch {epoch+1} Sub: {batch_idx} Loss D: {loss_disc:.4f}, Loss G: {total_gen_loss:.4f} Loss D2: {loss_disc_2:.4f}, Loss G2: {total_gen_loss_2:.4f} \n")

        epsilon = 0.00001
    
        if total_gen_loss <= epsilon:
            transfer_model_parameters(generator_2, generator)
            print("Generator model was killed and revived as Generator_2")
        if total_gen_loss_2 <= epsilon:
            transfer_model_parameters(generator, generator_2)
            print("Generator_2 model was killed and revived as Generator")

        if loss_disc <= epsilon or loss_disc >= 50 - epsilon:
            transfer_model_parameters(discriminator_2, discriminator)
            print("Discriminator model was killed and revived as Discriminator_2")
        if loss_disc_2 <= epsilon or loss_disc_2 >= 50 - epsilon:
            transfer_model_parameters(discriminator, discriminator_2)
            print("Discriminator_2 model was killed and revived as Discriminator")


        
        if total_gen_loss > total_gen_loss_2 and batch_idx % 100 == 0:
            transfer_model_parameters(generator_2, generator)
            print("Generator model was killed and revived as Generator_2")
        if total_gen_loss_2 > total_gen_loss and batch_idx % 100 == 0:
            transfer_model_parameters(generator, generator_2)
            print("Generator_2 model was killed and revived as Generator")

        if loss_disc > loss_disc_2 and batch_idx % 100 == 0:
            transfer_model_parameters(discriminator_2, discriminator)
            print("Discriminator model was killed and revived as Discriminator_2")
        if loss_disc_2 > loss_disc and batch_idx % 100 == 0:
            transfer_model_parameters(discriminator, discriminator_2)
            print("Discriminator_2 model was killed and revived as Discriminator")


    if (epoch + 1) % 1 == 0:
        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(generator_2.state_dict(), 'generator_2.pth')
        torch.save(discriminator.state_dict(), 'discriminator.pth')
        torch.save(discriminator_2.state_dict(), 'discriminator_2.pth')
        print("Models all saved")
    
    
        
        
    # Logging
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_disc:.4f}, Loss G: {total_gen_loss:.4f} Loss D2: {loss_disc_2:.4f}, Loss G2: {total_gen_loss_2:.4f}")
    with open('model_history.txt', 'a') as file:
        file.write(f"Epoch {epoch+1} Loss D: {loss_disc:.4f}, Loss G: {total_gen_loss:.4f} Loss D2: {loss_disc_2:.4f}, Loss G2: {total_gen_loss_2:.4f} \n")
    
        

