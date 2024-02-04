import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy

# Model Parameters
latent_dim = 27 
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
        std = torch.exp(0.5 * log_var)
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

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = BCE + KLD
    return BCE, KLD, total_loss



# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_list[idx])
        image = Image.open(img_path).convert('L')  # Convert to greyscale
        if self.transform:
            image = self.transform(image)
        return image

def test_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    mse_loss = nn.MSELoss(reduction='mean')
    total_mse = 0.0
    with torch.no_grad():  # No need to track gradients
        for data in dataloader:
            img = data.to(device)
            recon, _, _ = model(img)
            loss = mse_loss(recon, img)
            total_mse += loss.item()

    avg_mse = total_mse / len(dataloader)
    return avg_mse

def load_pretrained_model(path, latent_dim, device):
    model = VariationalAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
    


# Load dataset


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# Instantiate the dataset
dataset = CustomDataset(folder_path='photos_2', transform=transform)

# Dataset and Dataloader
dataloader = DataLoader(dataset, batch_size=500, shuffle=True)


# Dataset and Dataloader for test data
test_dataset = CustomDataset(folder_path='test_photos', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) 


# Instantiate VAE model with latent_dim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: " + str(device))

saved_model_path = 'variational_autoencoder.pth'
if os.path.exists(saved_model_path):
    model = load_pretrained_model(saved_model_path, LATENT_DIM, device)
    print("Loaded pretrained model.")
else:
    model = VariationalAutoencoder(latent_dim=LATENT_DIM).to(device)
    print("No pretrained model found. Starting from scratch.")


# Loss and optimizer
optimizer = optim.Adadelta(model.parameters(), lr=0.0005, eps=1e-8, weight_decay=0.00)
#optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)


# Train the model
num_epochs = 100000
for epoch in range(num_epochs):
    total_bce_loss = 0
    total_kld_loss = 0
    total_loss = 0

    for data in dataloader:
        img = data.to(device)

        # Forward pass
        recon_batch, mu, log_var = model(img)
        
        # Calculate loss
        BCE_loss, KLD_loss, loss = loss_function(recon_batch, img, mu, log_var)

        # Record information
        with open("latent_mapping.txt", "a") as file:
            mu_np = mu.detach().cpu().numpy()
            log_var_np = log_var.detach().cpu().numpy()
            file.write("mu: " + str(mu_np) + " log_var: " + str(log_var_np) + " BCE_loss: " + str(BCE_loss.item()) + " KLD_loss: " + str(KLD_loss.item()) + "\n")


        # Accumulate losses for averaging
        total_bce_loss += BCE_loss.item()
        total_kld_loss += KLD_loss.item()
        total_loss += loss.item()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        optimizer.step()

    # Average losses over the dataset
    avg_bce_loss = total_bce_loss / len(dataloader.dataset)
    avg_kld_loss = total_kld_loss / len(dataloader.dataset)
    avg_total_loss = total_loss / len(dataloader.dataset)

    avg_mse_test = test_model(model, test_dataloader, device)


    print(f'Epoch [{epoch+1}/{num_epochs}], Avg Total Loss: {avg_total_loss:.6f}, Avg BCE Loss: {avg_bce_loss:.6f}, Avg KLD Loss: {avg_kld_loss:.6f}, Test MSE Loss: {avg_mse_test:.6f}')
    with open('model_history.txt', 'a') as file:
        file.write(f'Epoch: {epoch}, Avg_Total_Loss: {avg_total_loss:.6f}, Avg_BCE_Loss: {avg_bce_loss:.6f}, Avg_KLD_Loss: {avg_kld_loss:.6f}, Test_MSE_Loss: {avg_mse_test:.6f} \n')
    
    if (epoch % 25 == 0 and epoch > 24):
        torch.save(model.state_dict(), f'variational_autoencoder.pth')
        print("Model Saved at Epoch: ", epoch)

# Save the final model
torch.save(model.state_dict(), 'variational_autoencoder_final.pth')
