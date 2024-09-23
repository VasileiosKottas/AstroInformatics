import torch
import torch.nn as nn


# Define the model building block: A recurrent neural network with GRU layers followed by a Dense (Linear) output layer
class RNNModule(nn.Module):
    def __init__(self, n_layers, input_size, hidden_units, output_units, output_activation=None):
        super(RNNModule, self).__init__()
        self.gru_layers = nn.ModuleList(
            [nn.GRU(input_size=input_size if i == 0 else hidden_units,  # Match input size for the first layer
                    hidden_size=hidden_units,
                    batch_first=True,
                    num_layers=1) for i in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_units, output_units)
        self.output_activation = output_activation

    def forward(self, x):
        for gru in self.gru_layers:
            x, _ = gru(x)
        x = self.output_layer(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x
    
# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, embedder, recovery):
        super(Autoencoder, self).__init__()
        self.embedder = embedder
        self.recovery = recovery

    def forward(self, x):
        # Pass input through the embedder (encoder)
        H = self.embedder(x)
        # Pass the embedded representation through the recovery (decoder)
        X_tilde = self.recovery(H)
        return X_tilde
    
class AdversarialNetSupervised(nn.Module):
    def __init__(self, generator, supervisor, discriminator):
        super(AdversarialNetSupervised, self).__init__()
        self.generator = generator
        self.supervisor = supervisor
        self.discriminator = discriminator

    def forward(self, Z):
        E_hat = self.generator(Z)

        H_hat = self.supervisor(E_hat)

        Y_fake = self.discriminator(H_hat)
        return Y_fake
    
class AdversarialNet(nn.Module):
    def __init__(self, generator, discriminator):
        super(AdversarialNet, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, Z):
        E_hat = self.generator(Z)
        Y_fake_e = self.discriminator(E_hat)
        return Y_fake_e

class SyntheticData(nn.Module):
    def __init__(self, generator, supervisor, recovery):
        super(SyntheticData, self).__init__()
        self.generator = generator
        self.supervisor = supervisor
        self.recovery = recovery

    def forward(self, Z):
        E_hat = self.generator(Z)    # Latent representation from generator
        H_hat = self.supervisor(E_hat)  # Supervised latent representation
        X_hat = self.recovery(H_hat)  # Recovered data from supervisor's output
        return X_hat
    
class DiscriminatorReal(nn.Module):
    def __init__(self, discriminator):
        super(DiscriminatorReal, self).__init__()
        self.discriminator = discriminator

    def forward(self, X):
        Y_real = self.discriminator(X)
        return Y_real