# src/train.py
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.model import PhotometryToSpectraModel
from src.data_loader import load_data
from config.hyperparameters import DEVICE, EPOCHS, LEARNING_RATE, PHOTOMETRY_DIM, SPECTRA_DIM, LATENT_DIM, TRANSFORMER_HIDDEN_DIM, TRANSFORMER_NHEAD, TRANSFORMER_LAYERS
import os
import torch.distributions as dist

def compute_crps(y_true, y_pred):
    """
    Compute Continuous Ranked Probability Score (CRPS).
    Assumes a normal distribution for predicted values.
    """
    std = 1.0  # Standard deviation for the predicted distribution
    normal = dist.Normal(loc=y_pred, scale=std)
    cdf = normal.cdf(y_true)
    crps = torch.mean(torch.abs(y_true - cdf))
    return crps.item()

def train_model(model, photometry_data, spectra_data):
    # Setup
    criterion = nn.MSELoss()
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(model.generator.parameters(), lr=LEARNING_RATE)
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE)
    optimizer_ER = optim.Adam(list(model.embedding.parameters()) + list(model.recovery.parameters()), lr=LEARNING_RATE)

    # Track metrics
    losses_g, losses_d, losses_er, ep = [], [], [], []
    metrics_mse, metrics_mae, metrics_crps = [], [], []
    output_dir = "./training_plots"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
        valid = torch.ones((spectra_data.shape[0], 1), requires_grad=False).to(DEVICE)
        fake = torch.zeros((spectra_data.shape[0], 1), requires_grad=False).to(DEVICE)

        # Train Generator
        optimizer_G.zero_grad()
        generated_spectra = model.forward_generator(photometry_data)
        g_loss_supervised = criterion(generated_spectra, spectra_data)
        g_loss_adversarial = adversarial_loss(model.forward_discriminator(generated_spectra), valid)
        g_loss = g_loss_supervised + g_loss_adversarial
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(model.forward_discriminator(spectra_data), valid)
        fake_loss = adversarial_loss(model.forward_discriminator(generated_spectra.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Embedding and Recovery Networks
        optimizer_ER.zero_grad()
        recovered_spectra = model.forward_embedding_recovery(photometry_data)
        er_loss = criterion(recovered_spectra, spectra_data)
        er_loss.backward()
        optimizer_ER.step()

        # Compute Metrics
        mse = nn.functional.mse_loss(generated_spectra, spectra_data).item()
        mae = nn.functional.l1_loss(generated_spectra, spectra_data).item()
        crps = compute_crps(spectra_data, generated_spectra)

        # Track metrics
        losses_g.append(g_loss.item())
        losses_d.append(d_loss.item())
        losses_er.append(er_loss.item())
        metrics_mse.append(mse)
        metrics_mae.append(mae)
        metrics_crps.append(crps)
        ep.append(epoch)

        if epoch % 100 == 0:
            print(
                f"[Epoch {epoch}/{EPOCHS}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [ER loss: {er_loss.item()}] "
                f"[MSE: {mse}] [MAE: {mae}] [CRPS: {crps}]"
            )

    # Save combined plot for Generator and Discriminator Losses
    save_combined_plot(
        ep,
        {"Generator Loss": losses_g, "Discriminator Loss": losses_d},
        "Epoch",
        "Loss",
        "Generator and Discriminator Losses",
        os.path.join(output_dir, "generator_discriminator_loss.png"),
    )

    # Save individual plots for MSE, MAE, and CRPS
    save_plot(ep, metrics_mse, "Epoch", "MSE", "Mean Squared Error", os.path.join(output_dir, "mse.png"))
    save_plot(ep, metrics_mae, "Epoch", "MAE", "Mean Absolute Error", os.path.join(output_dir, "mae.png"))
    save_plot(ep, metrics_crps, "Epoch", "CRPS", "Continuous Ranked Probability Score", os.path.join(output_dir, "crps.png"))

    # Save the trained model
    torch.save(model.state_dict(), "./gan_model.pth")


def save_combined_plot(x, y_dict, xlabel, ylabel, title, filepath):
    """Helper function to save a combined plot for multiple metrics."""
    plt.figure(figsize=(12, 8))
    for label, y in y_dict.items():
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()


def save_plot(x, y, xlabel, ylabel, title, filepath):
    """Helper function to save a single plot."""
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=title, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
