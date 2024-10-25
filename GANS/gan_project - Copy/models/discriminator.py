import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h, _ = self.rnn(x)  # h has shape (batch_size, sequence_length, hidden_dim * 2)
        h = h.mean(dim=1)   # Aggregate across the sequence length, producing (batch_size, hidden_dim * 2)
        return torch.sigmoid(self.fc(h))  # Output shape will be (batch_size, output_dim)
