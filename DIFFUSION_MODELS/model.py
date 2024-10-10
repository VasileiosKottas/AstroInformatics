import torch
import torch.nn as nn

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, n_heads, n_layers, output_dim, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        # Transformer expects [seq_len, batch_size, input_dim]
        transformer_output = self.transformer_encoder(src)
        output = self.fc_out(transformer_output[-1])  # Use the last time step's output
        return output