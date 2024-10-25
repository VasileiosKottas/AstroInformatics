# models/embedding.py
import torch.nn as nn
import torch
class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(EmbeddingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        nn.init.xavier_uniform_(self.latent.weight)
        nn.init.zeros_(self.latent.bias)

    def forward(self, x):
        """
        Expected input shape: (batch_size, sequence_length, input_dim)
        """
        batch_size, sequence_length, input_dim = x.size()
        x = x.reshape(batch_size * sequence_length, input_dim)  # Flatten the input
        x = torch.relu(self.fc(x))  # Apply the first FC layer
        latent = self.latent(x)  # Get latent representation
        return latent.view(batch_size, sequence_length, -1)  # Reshape back to (batch_size, sequence_length, latent_dim)
