import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
from PIL import Image
import os
import random
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights




class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1):
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
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: img_channels x 256 x 256
            nn.Conv2d(img_channels, 64, 4, 2, 1),  # Output: 64 x 128 x 128
            nn.LeakyReLU(0.2),
            self._block(64, 128, 4, 2, 1),         # Output: 128 x 64 x 64
            self._block(128, 256, 4, 2, 1),        # Output: 256 x 32 x 32
            self._block(256, 512, 4, 2, 1),        # Output: 512 x 16 x 16
            nn.Conv2d(512, 512, 4, 2, 1),          # Output: 512 x 8 x 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, 2, 1),          # Output: 512 x 4 x 4
            nn.LeakyReLU(0.2),
            # Ensure the final output is 1x1
            nn.Conv2d(512, 1, 4, 1, 0),            # Output: 1 x 1 x 1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.disc(x)
        return x.view(x.size(0), -1)



class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load the pre-trained VGG16 model with the recommended approach
        vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16_model.features

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x)


def diversity_penalty(features, eps=1e-8):
    """
    Calculate diversity penalty based on cosine similarity of features.
    :param features: Tensor of shape (batch_size, feature_size) containing features of generated images.
    :param eps: Small value to avoid division by zero.
    :return: Diversity penalty loss.
    """
    normalized_features = features / (features.norm(dim=1, keepdim=True) + eps)
    similarity_matrix = torch.mm(normalized_features, normalized_features.t())
    # Zero out diagonal elements (self-similarity) and take upper triangle to avoid double counting
    similarity_matrix.fill_diagonal_(0)
    upper_triangle_indices = torch.triu_indices(similarity_matrix.size(0), similarity_matrix.size(1), offset=1)
    average_similarity = similarity_matrix[upper_triangle_indices[0], upper_triangle_indices[1]].mean()
    # Penalty for high similarity (low diversity)
    diversity_loss = average_similarity
    return diversity_loss




def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty for WGAN-GP."""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.to(device).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2, 3]) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty




# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: " + str(device))

# Hyperparameters
z_dim = 100
learning_rate_gen = 0.001
learning_rate_disc = 0.00001
batch_size = 100
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
lambda_div = 0.5  # Example value, adjust based on your needs

# Initialize feature extractor once before the training loop
feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()

for epoch in range(num_epochs):
    for batch_idx, real in enumerate(loader):
        real = real.to(device)
        batch_size = real.size(0)

        # Prepare noise input for generator
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake = generator(noise)

        ### Update Discriminator: maximize log(D(x)) + log(1 - D(G(z))) + gradient_penalty
        discriminator.zero_grad()

        real_pred = discriminator(real)
        fake_pred = discriminator(fake.detach())

        real_loss = criterion(real_pred, torch.ones_like(real_pred))
        fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))

        # Compute Gradient Penalty on real and fake data
        gp = compute_gradient_penalty(discriminator, real.data, fake.data, device)

        d_loss = real_loss + fake_loss + lambda_gp * gp
        if(d_loss < 0.000001):
            break
        d_loss.backward()
        opt_disc.step()

        ### Update Generator: minimize log(1 - D(G(z))) or maximize log(D(G(z))) + diversity_penalty
        generator.zero_grad()

        # Generate a batch of images
        gen_pred = discriminator(fake)

        # Original adversarial loss
        g_loss_adv = criterion(gen_pred, torch.ones_like(gen_pred))

        # Compute features of generated images to apply diversity penalty
        gen_features = feature_extractor(fake).view(batch_size, -1)  # Adjust shape as needed
        div_penalty = diversity_penalty(gen_features)

        # Combine adversarial loss with diversity penalty
        g_loss = g_loss_adv - lambda_div * div_penalty  # Note: Subtracting to minimize penalty
        if(g_loss < 0.000001):
            break
        g_loss.backward()
        opt_gen.step()
        
    # Logging
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {d_loss:.4f}, Loss G: {g_loss:.4f}")
    with open('model_history.txt', 'a') as file:
        file.write(f"Epoch {epoch+1} Loss D: {d_loss:.4f}, Loss G: {g_loss:.4f} \n")
    if (epoch + 1) % 1 == 0:
        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(discriminator.state_dict(), 'discriminator.pth')
    if(d_loss < 0.000001 or g_loss < 0.000001):
            torch.save(generator.state_dict(), 'generator_final.pth')
            torch.save(discriminator.state_dict(), 'discriminator_final.pth')
            break
        
