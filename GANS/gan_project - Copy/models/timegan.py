from .embedding import EmbeddingNetwork, RecoveryNetwork
from .generator import Generator
from .discriminator import Discriminator

class TimeGAN:
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        self.device = device
        self.embedding_net = EmbeddingNetwork(input_dim, hidden_dim, latent_dim).to(device)
        self.recovery_net = RecoveryNetwork(latent_dim, hidden_dim, input_dim).to(device)
        self.generator_net = Generator(latent_dim, hidden_dim, input_dim).to(device)
        self.discriminator_net = Discriminator(input_dim, hidden_dim, 1).to(device)

