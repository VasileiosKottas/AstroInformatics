import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states):
        attn_weights = torch.softmax(self.attn(hidden_states), dim=1)
        context_vector = torch.sum(attn_weights * hidden_states, dim=1)
        return context_vector

class ComplexGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(ComplexGenerator, self).__init__()
        # Using a deeper RNN architecture with LSTMs
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

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(EmbeddingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        nn.init.xavier_uniform_(self.latent.weight)
        nn.init.zeros_(self.latent.bias)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.latent(x)

class RecoveryNetwork(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(RecoveryNetwork, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, output_dim, batch_first=True)

    def forward(self, h):
        h = self.fc(h).unsqueeze(1)
        output, _ = self.rnn(h.repeat(1, h.shape[1], 1))
        return output

class TimeGAN:
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        self.device = device
        self.embedding_net = EmbeddingNetwork(input_dim, hidden_dim, latent_dim).to(device)
        self.recovery_net = RecoveryNetwork(latent_dim, hidden_dim, input_dim).to(device)
        self.generator_net = ComplexGenerator(latent_dim, hidden_dim, 10).to(device)  # Predict 10 fluxes
        self.discriminator_net = ComplexDiscriminator(10, hidden_dim, 1).to(device)
