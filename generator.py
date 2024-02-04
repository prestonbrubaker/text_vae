import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Model Parameters
latent_dim = 27  # Example latent space dimension
LATENT_DIM = latent_dim



class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

       # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # Output: 16x128x128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Output: 64x64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 128x32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 256x16x16
            nn.ReLU(),
            nn.Flatten(),  # Flatten for linear layer input
            nn.Linear(128*16*16, 1024),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 1024)

        self.decoder = nn.Sequential(
            nn.Linear(1024, 128*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),  # Unflatten to 256x16x16 for conv transpose input
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 128x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 64x64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: 32x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Output: 1x256x256
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var * 10)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# Function to load the model
def load_model(path, device):
    model = VariationalAutoencoder(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(path))
    return model

def load_mean_latents(file_path, latent_dim):
    with open(file_path, 'r') as file:
        last_line = file.readline().strip()  # Read the first (and presumably only) line
        values = [float(value) for value in last_line.split(' ')]
        
        # Assuming the first half is mu and the second half is log_var
        mean_mu = values[:latent_dim]
        mean_log_var = values[latent_dim:]
        
        return mean_mu, mean_log_var

def generate_images_with_random_latents(model, num_images, folder_path, mean_mu, mean_log_var):
    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists

    for i in range(num_images):
        with torch.no_grad():
            # Sample a new latent_vector for each image
            mu = torch.tensor(mean_mu).unsqueeze(0).to(device)
            log_var = torch.tensor(mean_log_var).unsqueeze(0).to(device)
            latent_vector = model.reparameterize(mu, log_var)

            # Decode the sampled latent vector
            generated_image = model.decode(latent_vector).cpu()

            # Convert to PIL image and save
            generated_image = generated_image.squeeze(0)
            generated_image = transforms.ToPILImage()(generated_image)
            generated_image.save(os.path.join(folder_path, f"generated_image_{i+1}.png"))

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_path = 'variational_autoencoder.pth'
vae_model = load_model(model_path, device)
vae_model.eval()  # Set the model to evaluation mode

# Specify the path to the mean latents file
mean_latents_file = 'latent_logs/mean_latents.txt'

# Correctly read the mean latents from the file, unpacking them into mean_mu and mean_log_var
mean_mu, mean_log_var = load_mean_latents(mean_latents_file, LATENT_DIM)

num_generated_images = 500

# Correctly pass mean_mu and mean_log_var to the function
generate_images_with_random_latents(vae_model, num_generated_images, 'generated_photos', mean_mu, mean_log_var)

print(f"Generated {num_generated_images} images in 'generated_photos' folder.")
