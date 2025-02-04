import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import torch.nn.functional as F

# Updated VAE architecture.
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        # Encoder layers.
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.fc21 = nn.Linear(512, latent_dim)  # Mean for latent space
        self.fc22 = nn.Linear(512, latent_dim)  # Log variance for latent space

        # Decoder layers.
        self.fc3 = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.fc3(z)
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction help.
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum')

        # KL divergence
        KL_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1. - logvar)
        
        return BCE + beta * KL_div, BCE, KL_div

# Dataset for loading the data.
class PCGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)

# Helper functions.
def save_model(model, epoch, path="vae_model.pth"):
    torch.save({"epoch": epoch, "model_state_dict": model_state_dict()}, path)
    print(f"Model saved at epoch {epoch} to {path}.")

def log_loss(losses, filename="training_loss.txt"):
    with open(filename, "a") as log_file:
        for loss in losses:
            log_file.write(f"{loss}\n")
    print(f"Loss values saved to {filename}")

def train_vae():
    # Load the preprocessed dataset.
    with open("pcg_preprocessed_global_normalized.json", "r") as file:
        dataset = json.load(file)
    
    # Flatten dataset for training
    flat_data = [item for env in dataset.values() for category in env.values() for item in category if item]

    # Prepare the DataLoader.
    pcg_data = PCGDataset(flat_data)
    dataloader = DataLoader(pcg_data, batch_size=32, shuffle=True)

    # Input dimension in the size of each feature vector.
    input_dim = len(flat_data[0])
    latent_dim = 10

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=5e-4, weight_decay=1e-5)

    num_epochs = 100
    losses = []

    for epoch in range(num_epochs):
        train_loss = 0
        for data in dataloader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss, recon_loss, kl_loss = vae.loss_function(recon_batch, data, mu, logvar, beta=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=0.1)
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(dataloader.dataset)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Total loss: {avg_loss:.4f}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            save_model(vae, epoch)
        
    log_loss(losses)

if __name__ == "__main__":
    train_vae()