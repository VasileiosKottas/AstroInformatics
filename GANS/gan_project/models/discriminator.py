import torch.nn as nn
import torch
from .attention import Attention

class ComplexDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ComplexDiscriminator, self).__init__()
        # Deeper RNN with Bidirectional LSTM
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=5, dropout=0.3)
        self.attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=0.3)  # Attention mechanism
        self.dropout = nn.Dropout(0.4)  # Increase dropout rate
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim)

        # Additional Residual Blocks
        self.residual_fc = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

    def forward(self, x):
        h, _ = self.rnn(x)

        # Apply attention mechanism
        h = h.permute(1, 0, 2)  # (seq_len, batch, hidden_dim * 2)
        h_attn, _ = self.attn(h, h, h)  # Self-attention
        h_attn = h_attn.permute(1, 0, 2)  # (batch, seq_len, hidden_dim * 2)

        # Pool across time steps (mean) and pass through the rest of the network
        h = self.dropout(h_attn.mean(dim=1))  # Apply dropout after attention
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.ln1(self.fc2(h)))
        h = torch.relu(self.ln2(self.fc3(h)))
        
        # Residual connection
        h = h + self.residual_fc(h)
        
        return torch.sigmoid(self.fc4(h))