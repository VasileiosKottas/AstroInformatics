import torch.nn as nn

import torch

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(EmbeddingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        # Custom initialization for latent layer
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
