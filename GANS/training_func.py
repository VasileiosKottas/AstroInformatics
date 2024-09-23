import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from utils import *
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

mse = nn.MSELoss()
bce = nn.BCELoss()

def train_autoencoder_step(x, autoencoder, optimizer):
    autoencoder.to(device)
    autoencoder.train()  # Set the model to training mode
    
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    x_tilde = autoencoder(x)

    # Compute the loss
    embedding_loss_t0 = F.mse_loss(x_tilde, x)
    e_loss_0 = 10 * torch.sqrt(embedding_loss_t0)

    # Backward pass
    e_loss_0.backward()

    # Update the weights
    optimizer.step()

    # Return the loss value
    return torch.sqrt(embedding_loss_t0).item()

def train_supervisor(x, embedder, supervisor, supervisor_optimizer):
    x = torch.tensor(x, requires_grad=True).to(device)

    # Enable gradient tracking
    h = embedder(x).to(device)

    # Forward pass through the supervisor
    h_hat_supervised = supervisor(h)

    # Calculate the loss
    g_loss_s = mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

    # Backward pass and optimization
    supervisor_optimizer.zero_grad()  # Clear previous gradients
    g_loss_s.backward()  # Compute gradients
    supervisor_optimizer.step()  # Update parameters

    return g_loss_s

def train_generator(x, z, generator, supervisor, embedder, 
                    synthetic_data, adversarial_supervised,
                    adversarial_emb,
                    generator_optimizer):
    # Ensure the models are in training mode
    generator.train()
    supervisor.train()
    embedder.train()
    synthetic_data.train()
    adversarial_supervised.train()
    adversarial_emb.train()
    # Forward pass through the models
    y_fake = adversarial_supervised(z).to(device)
    generator_loss_unsupervised = F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))

    y_fake_e = adversarial_emb(z).to(device)
    generator_loss_unsupervised_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))
    h = embedder(x).to(device)
    h_hat_supervised = supervisor(h).to(device)
    generator_loss_supervised = F.mse_loss(h[:, 1:, :], h_hat_supervised[:, 1:, :])

    x_hat = synthetic_data(z).to(device)
    generator_moment_loss = get_generator_moment_loss(x, x_hat)

    # Compute the total generator loss
    generator_loss = (generator_loss_unsupervised +
                      generator_loss_unsupervised_e +
                      100 * torch.sqrt(generator_loss_supervised) +
                      100 * generator_moment_loss).to(device)

    # Zero the gradients
    generator_optimizer.zero_grad()

    # Backward pass
    generator_loss.backward()

    # Update the model parameters
    generator_optimizer.step()

    return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss


def train_embedder(x, embedder, supervisor, autoencoder, recovery, embedding_optimizer):
    # Ensure the models are in training mode
    embedder.train()
    supervisor.train()
    autoencoder.train()
    recovery.train()

    # Forward pass through the models
    h = embedder(x).to(device)
    h_hat_supervised = supervisor(h).to(device)
    generator_loss_supervised = F.mse_loss(h[:, 1:, :], h_hat_supervised[:, 1:, :])

    x_tilde = autoencoder(x).to(device)
    embedding_loss_t0 = F.mse_loss(x, x_tilde)
    e_loss = 10 * torch.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

    # Zero the gradients
    embedding_optimizer.zero_grad()

    # Backward pass
    e_loss.backward()

    # Update the model parameters
    embedding_optimizer.step()

    return torch.sqrt(embedding_loss_t0)

def train_discriminator(x, z, gamma, discriminator_model,
                        discriminator_optimizer, adversarial_supervised, adversarial_emb):
    # Ensure the discriminator is in training mode
    discriminator_model.train()

    # Forward pass and compute loss
    discriminator_loss = get_discriminator_loss(x, z, gamma, discriminator_model, adversarial_supervised, adversarial_emb)

    # Zero the gradients
    discriminator_optimizer.zero_grad()

    # Backward pass to compute gradients
    discriminator_loss.backward()

    # Update model parameters
    discriminator_optimizer.step()

    return discriminator_loss