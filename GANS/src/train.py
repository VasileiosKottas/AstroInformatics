# src/train.py
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.model import PhotometryToSpectraModel
from src.data_loader import load_data
from config.hyperparameters import DEVICE, EPOCHS, LEARNING_RATE, PHOTOMETRY_DIM, SPECTRA_DIM, LATENT_DIM, TRANSFORMER_HIDDEN_DIM, TRANSFORMER_NHEAD, TRANSFORMER_LAYERS

def train_model(model, photometry_data, spectra_data):
    criterion = nn.MSELoss()
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(model.generator.parameters(), lr=LEARNING_RATE)
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE)
    optimizer_ER = optim.Adam(list(model.embedding.parameters()) + list(model.recovery.parameters()), lr=LEARNING_RATE)
    losses_g, losses_d, ep = [], [], []

    # Wrap the training loop with tqdm for a progress bar
    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
        valid = torch.ones((spectra_data.shape[0], 1), requires_grad=False).to(DEVICE)
        fake = torch.zeros((spectra_data.shape[0], 1), requires_grad=False).to(DEVICE)

        # ========================= Train Generator =========================
        optimizer_G.zero_grad()
        generated_spectra = model.forward_generator(photometry_data)
        g_loss_supervised = criterion(generated_spectra, spectra_data)
        g_loss_adversarial = adversarial_loss(model.forward_discriminator(generated_spectra), valid)
        g_loss = g_loss_supervised + g_loss_adversarial
        g_loss.backward()
        optimizer_G.step()

        # ========================= Train Discriminator =========================
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(model.forward_discriminator(spectra_data), valid)
        fake_loss = adversarial_loss(model.forward_discriminator(generated_spectra.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # ========================= Train Embedding and Recovery Networks =========================
        optimizer_ER.zero_grad()
        recovered_spectra = model.forward_embedding_recovery(photometry_data)
        er_loss = criterion(recovered_spectra, spectra_data)
        er_loss.backward()
        optimizer_ER.step()

        # Track losses
        losses_g.append(g_loss.item())
        losses_d.append(d_loss.item())
        ep.append(epoch)
        
        # Display loss values at intervals
        if epoch % 100 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [ER loss: {er_loss.item()}]")
    
    # Plot loss after training
    plt.plot(ep, losses_d, label='Discriminator Loss', linestyle='--', alpha=0.7)
    plt.plot(ep, losses_g, label='Generator Loss', alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "./gan_model.pth")
