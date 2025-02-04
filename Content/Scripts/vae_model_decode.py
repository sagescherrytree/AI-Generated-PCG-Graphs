import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import json

# Simple VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 512)  # Encoder layer
        self.fc21 = nn.Linear(512, latent_dim)  # Mean for latent space
        self.fc22 = nn.Linear(512, latent_dim)  # Log variance for latent space
        self.fc3 = nn.Linear(latent_dim, 512)  # Decoder layer
        self.fc4 = nn.Linear(512, input_dim)  # Output layer

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum')

        # KL divergence
        # The term ensures the latent space distribution is close to a normal distribution
        KL_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1. - logvar)
        
        return BCE + KL_div
    
input_dim = 18  # This should match the dimension of your input data
latent_dim = 10  # This should match the latent dimension used in training
vae = VAE(input_dim, latent_dim)

# Load model.
# checkpoint = torch.load("vae_model.pth")

# Load VAE model with discriminator.
checkpoint = torch.load("vae_model_with_discriminator.pth")
# vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
vae.load_state_dict(torch.load("vae_model_with_discriminator.pth"), strict=False)
vae.eval() # Set to evaluation mode.

# Helper function to sample from latent space
def sample_latent_space(latent_dim, num_samples=1):
    """Sample latent vectors from the latent space."""
    return torch.randn(num_samples, latent_dim)  # Generate num_samples latent vectors

# Generate data for a new PCG graph
def generate_pcg_graph(vae, latent_dim, categories, num_nodes_per_category=5):
    """Generate a new PCG graph with feature vectors for each category."""
    pcg_graph = {}
    for category in categories:
        # Generate multiple latent vectors for each category
        latent_vectors = sample_latent_space(latent_dim, num_samples=num_nodes_per_category)
        generated_vectors = vae.decode(latent_vectors).detach().numpy()  # Detach gradients
        pcg_graph[category] = [vector.tolist() for vector in generated_vectors]
    return pcg_graph

# Define categories for the PCG graph
categories = ["Grass", "GroundItems", "Structures", "Trees"]

# Generate a new PCG graph
new_pcg_graph = generate_pcg_graph(vae, latent_dim, categories, num_nodes_per_category=10)

# # Save the generated PCG graph to a JSON file
# def save_pcg_graph_to_json(pcg_graph, filename="generated_pcg_graph.json"):
#     with open(filename, "w") as json_file:
#         json.dump(pcg_graph, json_file, indent=4)
#     print(f"Generated PCG graph saved to {filename}")

def save_pcg_graph_to_json(pcg_graph, filename="generated_pcg_graph.json"):
    def convert_to_number(value):
        # Convert strings to floats if possible
        try:
            return float(value)
        except ValueError:
            return value

    def format_data(data):
        # Recursively process data to ensure all values are numbers
        if isinstance(data, list):
            return [format_data(v) for v in data]
        elif isinstance(data, dict):
            return {k: format_data(v) for k, v in data.items()}
        elif isinstance(data, str):
            return convert_to_number(data)
        return data
    
    formatted_graph = format_data(pcg_graph)
    
    with open(filename, "w") as json_file:
        json.dump(formatted_graph, json_file, indent=4)
    print(f"Generated PCG graph saved to {filename}")

# Save the newly generated PCG graph
save_pcg_graph_to_json(new_pcg_graph)