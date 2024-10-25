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