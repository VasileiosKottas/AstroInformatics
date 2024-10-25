# models/recovery.py
import torch.nn as nn

class RecoveryNetwork(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(RecoveryNetwork, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, output_dim, batch_first=True)

    def forward(self, h):
        """
        Forward pass for the Recovery Network.

        Parameters:
        - h: Tensor of shape (batch_size, sequence_length, latent_dim)

        Returns:
        - output: Tensor of shape (batch_size, sequence_length, output_dim)
        """
        # Ensure h is shaped correctly
        batch_size, sequence_length, latent_dim = h.size()

        # Transform the latent vector into the hidden dimension
        h = h.reshape(batch_size * sequence_length, latent_dim)  # Flatten to apply FC layer
        h = self.fc(h)  # Shape: (batch_size * sequence_length, hidden_dim)
        h = h.view(batch_size, sequence_length, -1)  # Reshape back to (batch_size, sequence_length, hidden_dim)
        
        # Pass through the GRU
        output, _ = self.rnn(h)  # Shape: (batch_size, sequence_length, output_dim)
        return output
