import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


def get_generator_moment_loss(y_true, y_pred):
    # Compute the mean and variance for y_true and y_pred
    y_true_mean = torch.mean(y_true, dim=0)
    y_true_var = torch.var(y_true, dim=0, unbiased=False)

    y_pred_mean = torch.mean(y_pred, dim=0)
    y_pred_var = torch.var(y_pred, dim=0, unbiased=False)

    # Calculate the mean and variance losses
    g_loss_mean = torch.mean(torch.abs(y_true_mean - y_pred_mean))
    g_loss_var = torch.mean(torch.abs(torch.sqrt(y_true_var + 1e-6) - torch.sqrt(y_pred_var + 1e-6)))

    # Return the sum of the two losses
    return g_loss_mean + g_loss_var

def get_discriminator_loss(x, z, gamma, discriminator_model, adversarial_supervised, adversarial_emb):
    # Ensure the models are in evaluation mode if needed
    # discriminator_model.eval()
    # adversarial_supervised.eval()
    # adversarial_emb.eval()

    # Forward pass through the discriminator model
    y_real = discriminator_model(x).to(device)
    discriminator_loss_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))

    # Forward pass through the adversarial models
    y_fake = adversarial_supervised(z).to(device)
    discriminator_loss_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))

    y_fake_e = adversarial_emb(z).to(device)
    discriminator_loss_fake_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))

    # Combine the losses
    total_discriminator_loss = (discriminator_loss_real +
                                discriminator_loss_fake +
                                gamma * discriminator_loss_fake_e)

    return total_discriminator_loss
