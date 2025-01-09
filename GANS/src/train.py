# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from config.hyperparameters import EPOCHS, DEVICE

def train(model, photometry_data, spectra_data, val_photometry_data, val_spectra_data):
    # Setup
    criterion = nn.MSELoss()
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(model.generator.parameters(), lr=0.001)
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=0.001)

    # Track metrics
    train_losses_g, train_losses_d = [], []
    val_losses_g, val_losses_d = [], []
    output_dir = "./training_plots"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
        model.train()
        train_loss_g, train_loss_d = 0, 0

        valid = torch.ones((spectra_data.size(0), 1), requires_grad=False).to(DEVICE)
        fake = torch.zeros((spectra_data.size(0), 1), requires_grad=False).to(DEVICE)

        # Train Generator
        optimizer_G.zero_grad()
        generated_spectra = model.forward_generator(photometry_data)
        g_loss_supervised = criterion(generated_spectra, spectra_data)
        g_loss_adversarial = adversarial_loss(model.forward_discriminator(generated_spectra), valid)
        g_loss = g_loss_supervised + g_loss_adversarial
        g_loss.backward()
        optimizer_G.step()
        train_loss_g += g_loss.item()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(model.forward_discriminator(spectra_data), valid)
        fake_loss = adversarial_loss(model.forward_discriminator(generated_spectra.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        train_loss_d += d_loss.item()

        # Validation step
        model.eval()
        val_loss_g, val_loss_d = 0, 0
        with torch.no_grad():
            val_generated_spectra = model.forward_generator(val_photometry_data)
            val_g_loss = criterion(val_generated_spectra, val_spectra_data)
            val_loss_g += val_g_loss.item()

            real_loss = adversarial_loss(model.forward_discriminator(val_spectra_data), valid[:val_spectra_data.size(0)])
            fake_loss = adversarial_loss(model.forward_discriminator(val_generated_spectra), fake[:val_spectra_data.size(0)])
            val_d_loss = (real_loss + fake_loss) / 2
            val_loss_d += val_d_loss.item()

        train_losses_g.append(train_loss_g)
        train_losses_d.append(train_loss_d)
        val_losses_g.append(val_loss_g)
        val_losses_d.append(val_loss_d)

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] Train G Loss: {train_loss_g:.4f}, Train D Loss: {train_loss_d:.4f}, "
                  f"Val G Loss: {val_loss_g:.4f}, Val D Loss: {val_loss_d:.4f}")

    # Save loss plots
    plt.figure(figsize=(10, 6))
    plt.plot(range(EPOCHS), train_losses_g, label='Train G Loss')
    plt.plot(range(EPOCHS), val_losses_g, label='Val G Loss')
    plt.plot(range(EPOCHS), train_losses_d, label='Train D Loss')
    plt.plot(range(EPOCHS), val_losses_d, label='Val D Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    # Save the trained model
    torch.save(model.state_dict(), "./gan_model.pth")
