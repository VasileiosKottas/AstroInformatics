# models/timegan.py
from .generator import ComplexGenerator
from .discriminator import ComplexDiscriminator
from .embedding import EmbeddingNetwork
from .recovery import RecoveryNetwork

class TimeGAN:
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        self.device = device
        self.embedding_net = EmbeddingNetwork(input_dim, hidden_dim, latent_dim).to(device)
        self.recovery_net = RecoveryNetwork(latent_dim, hidden_dim, input_dim).to(device)
        self.generator_net = ComplexGenerator(latent_dim, hidden_dim, 10).to(device)
        self.discriminator_net = ComplexDiscriminator(10, hidden_dim, 1).to(device)

    def get_networks(self):
        return {
            "embedding": self.embedding_net,
            "recovery": self.recovery_net,
            "generator": self.generator_net,
            "discriminator": self.discriminator_net
        }
