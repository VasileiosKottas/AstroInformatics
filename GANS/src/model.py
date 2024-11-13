# src/model.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(EmbeddingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
        
    def forward(self, x):
        return torch.tanh(self.fc(x))

class RecoveryNetwork(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(RecoveryNetwork, self).__init__()
        self.fc = nn.Linear(latent_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x)

class TransformerGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, nhead, hidden_dim, num_layers):
        super(TransformerGenerator, self).__init__()
        self.transformer_layer = TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer = TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        transformed = self.transformer(z)
        return self.fc(transformed)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.fc(x)

class PhotometryToSpectraModel(nn.Module):
    def __init__(self, photometry_dim, spectra_dim, latent_dim, transformer_hidden_dim, transformer_nhead, transformer_layers):
        super(PhotometryToSpectraModel, self).__init__()
        self.embedding = EmbeddingNetwork(input_dim=photometry_dim, latent_dim=latent_dim)
        self.generator = TransformerGenerator(latent_dim=latent_dim, output_dim=spectra_dim, nhead=transformer_nhead, hidden_dim=transformer_hidden_dim, num_layers=transformer_layers)
        self.recovery = RecoveryNetwork(latent_dim=latent_dim, output_dim=spectra_dim)
        self.discriminator = Discriminator(input_dim=spectra_dim)

    def forward_generator(self, photometry):
        latent = self.embedding(photometry)
        return self.generator(latent)

    def forward_discriminator(self, spectra):
        return self.discriminator(spectra)

    def forward_embedding_recovery(self, photometry):
        latent = self.embedding(photometry)
        return self.recovery(latent)
