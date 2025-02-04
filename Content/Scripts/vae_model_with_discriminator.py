import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import torch.nn.functional as F

def get_input_dim(dataset_path="pcg_preprocessing_global_normalized.json"):
    """Determine the input dimension (length of a feature vector) from the dataset."""
    with open(dataset_path, "r") as file:
        dataset = json.load(file)
    
    for category, envs in dataset.items():
        for env_name, subcategories in envs.items():
            for subcategory_name, vectors in subcategories.items():
                if isinstance(vectors, list) and len(vectors) > 0:
                    # Return the length of the first valid feature vector.
                    return len(vectors[0])

    raise ValueError("No valid feature vectors found in the dataset.")

INPUT_DIM = get_input_dim()
print(f"Input Dimension: {INPUT_DIM}")

LATENT_DIM = 10

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

# Import the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        return self.model(x)

class PCGDataset(Dataset):
    def __init__(self, dataset_path):
        # Load the dataset.
        with open(dataset_path, "r") as file:
            dataset = json.load(file)

        self.data = []
        for category, envs in dataset.items():
            for env_name, subcategories in envs.items():
                for subcategory_name, vectors in subcategories.items():
                    if isinstance(vectors, list): # Ensure valid data.
                        self.data.extend(vectors)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Load the discriminator model
def load_discriminator(input_dim, model_path="discriminator.pth"):
    discriminator = Discriminator(input_dim=input_dim)
    discriminator.load_state_dict(torch.load(model_path))
    discriminator.eval()  # Set to evaluation mode
    return discriminator

def prepare_dataloader(dataset_path="pcg_preprocessed_global_normalized.json", batch_size=32):
    """Prepare DataLoader for the dataset."""
    pcg_dataset = PCGDataset(dataset_path)
    return DataLoader(pcg_dataset, batch_size=batch_size, shuffle=True)

# Updated VAE training loop with discriminator integration
def train_vae_with_discriminator():
    # Load your dataset as before
    dataloader = prepare_dataloader()

    # Initialize the VAE
    vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-4)

    # Load the trained discriminator
    discriminator = load_discriminator(input_dim=INPUT_DIM)

    # Training parameters
    num_epochs = 100
    adversarial_weight = 0.1  # Weight of the adversarial loss in the overall VAE loss

    for epoch in range(num_epochs):
        total_vae_loss = 0.0
        total_adv_loss = 0.0

        vae.train()
        for batch in dataloader:
            inputs = batch[0]  # Assuming inputs are the first item in the batch
            vae_optimizer.zero_grad()

            # Forward pass through the VAE
            reconstructions, mu, logvar = vae(inputs)
            
            # VAE Loss (Reconstruction + KL Divergence)
            recon_loss = F.mse_loss(reconstructions, inputs, reduction='mean')
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / inputs.size(0)
            vae_loss = recon_loss + kl_div

            # Adversarial Loss
            with torch.no_grad():
                discriminator_outputs = discriminator(reconstructions).squeeze()
            adversarial_loss = -torch.mean(torch.log(discriminator_outputs + 1e-8))  # Maximize discriminator's output

            # Combine Losses
            total_loss = vae_loss + adversarial_weight * adversarial_loss
            total_loss.backward()
            vae_optimizer.step()

            # Track losses
            total_vae_loss += vae_loss.item()
            total_adv_loss += adversarial_loss.item()

        avg_vae_loss = total_vae_loss / len(dataloader)
        avg_adv_loss = total_adv_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], VAE Loss: {avg_vae_loss:.4f}, Adv Loss: {avg_adv_loss:.4f}")

    # Save the trained VAE
    torch.save(vae.state_dict(), "vae_with_discriminator.pth")
    print("VAE training complete and saved as 'vae_with_discriminator.pth'.")

if __name__ == "__main__":
    train_vae_with_discriminator()