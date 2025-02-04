import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import torch.nn.functional as F

# Discriminator Model
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

def prepare_discriminator_dataset(dataset):
    """Prepare labeled data for the discriminator."""
    labeled_data = []
    labels = []

    for category, envs in dataset.items():
        label = 1 if category == "General" else 0  # Desired = 1, Undesired = 0
        for env_name, subcategories in envs.items():
            for subcategory_name, vectors in subcategories.items():
                if isinstance(vectors, list):  # Ensure the subcategory contains lists of feature vectors
                    for vector in vectors:
                        labeled_data.append(vector)
                        labels.append(label)

    return labeled_data, labels

def train_discriminator():
    # Load the preprocessed dataset
    with open("pcg_preprocessed_global_normalized.json", "r") as file:
        dataset = json.load(file)

    # Prepare labeled data
    data, labels = prepare_discriminator_dataset(dataset)

    # Prepare DataLoader
    class LabeledDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return (torch.tensor(self.data[idx], dtype=torch.float32),
                    torch.tensor(self.labels[idx], dtype=torch.float32))

    labeled_dataset = LabeledDataset(data, labels)
    dataloader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)

    # Initialize Discriminator
    input_dim = len(data[0])  # Dimension of each feature vector
    discriminator = Discriminator(input_dim=input_dim)
    optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        discriminator.train()
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = discriminator(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the trained discriminator
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Discriminator training complete and saved as 'discriminator.pth'.")

if __name__ == "__main__":
    train_discriminator()