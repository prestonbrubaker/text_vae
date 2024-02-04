import torch
from torchvision import transforms
from PIL import Image
import os

# Assuming the VariationalAutoencoder class is defined in vae_model.py
from vae_model import VariationalAutoencoder

# Model Parameters
latent_dim = 27  # Example latent space dimension

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, latent_dim, device):
    """Load the trained VAE model from a given path."""
    model = VariationalAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def load_mean_latents(file_path, latent_dim):
    """Load mean latent variables (mu and log_var) from a file."""
    with open(file_path, 'r') as file:
        last_line = file.readline().strip()
        values = [float(value) for value in last_line.split(' ')]
        mean_mu = values[:latent_dim]
        mean_log_var = values[latent_dim:]
        return mean_mu, mean_log_var

def reparameterize(mu, log_var, variability_scale=1.0):
    """Reparameterize with an additional scale for variability."""
    std = torch.exp(0.5 * log_var) * variability_scale
    eps = torch.randn_like(std)
    return mu + eps * std

def generate_images_with_random_latents(model, num_images, folder_path, mean_mu, mean_log_var, variability_scale=1.0):
    """Generate and save images using sampled latent variables with controlled variability."""
    os.makedirs(folder_path, exist_ok=True)

    for i in range(num_images):
        with torch.no_grad():
            mu = torch.tensor(mean_mu, device=device).unsqueeze(0)
            log_var = torch.tensor(mean_log_var, device=device).unsqueeze(0)
            # Use the reparameterize function with the variability_scale
            latent_vector = reparameterize(mu, log_var, variability_scale)
            generated_image = model.decode(latent_vector)

            # Convert the generated image to PIL format and save
            generated_image = generated_image.squeeze(0).cpu()
            img = transforms.ToPILImage()(generated_image)
            img.save(os.path.join(folder_path, f"generated_image_{i+1}.png"))

if __name__ == "__main__":
    # Load the trained model
    model_path = 'variational_autoencoder.pth'
    vae_model = load_model(model_path, latent_dim, device)

    # Specify the path to the mean latents file
    mean_latents_file = 'latent_logs/mean_latents.txt'
    mean_mu, mean_log_var = load_mean_latents(mean_latents_file, latent_dim)

    # Generate images using the mean latents with a custom variability scale
    num_generated_images = 500
    output_folder = 'generated_photos'
    variability_scale = 1.5  # Adjust this value to increase or decrease variability

    generate_images_with_random_latents(vae_model, num_generated_images, output_folder, mean_mu, mean_log_var, variability_scale)

    print(f"Generated {num_generated_images} images in '{output_folder}' folder with variability scale {variability_scale}.")
