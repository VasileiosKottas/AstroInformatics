import torch.nn as nn
import torch
from .attention import Attention

class ComplexGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(ComplexGenerator, self).__init__()
        
        self.rnn = nn.LSTM(latent_dim + 1, hidden_dim, batch_first=True, num_layers=5, bidirectional=True, dropout=0.3)
        self.attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=0.3)  # Attention mechanism
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # Increased to handle bidirectional output
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        # Additional layers for increased complexity
        self.residual_fc = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.dropout = nn.Dropout(0.4)

    def forward(self, z, wavelengths):
        if wavelengths.dim() == 2:
            wavelengths = wavelengths.unsqueeze(-1)
        z = torch.cat([z, wavelengths], dim=-1)
        h, _ = self.rnn(z)

        # Apply attention mechanism
        h = h.permute(1, 0, 2)  # (seq_len, batch, hidden_dim * 2)
        h_attn, _ = self.attn(h, h, h)  # Self-attention
        h_attn = h_attn.permute(1, 0, 2)  # (batch, seq_len, hidden_dim * 2)

        h = self.dropout(h_attn)  # Apply dropout
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.ln1(self.fc2(h)))
        output = self.fc3(h)

        # Apply residual connection
        output = output + self.residual_fc(output)
        return output