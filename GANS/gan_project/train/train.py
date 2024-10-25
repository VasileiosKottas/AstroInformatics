# train/train.py
import torch
import torch.optim as optim
from tqdm import tqdm
from models.timegan import TimeGAN
from utils.helper import gradient_penalty
from torch.nn import MSELoss, BCEWithLogitsLoss

def train_timegan(train_loader, scaler, device, epochs=1000, lambda_gp=10):
    """
    Train the TimeGAN model with progress tracking using tqdm.

    Parameters:
    - train_loader: DataLoader for the training data.
    - scaler: Fitted scaler for normalizing data.
    - device: Device (CPU/GPU) for model training.
    - epochs: Number of epochs to train.
    - lambda_gp: Gradient penalty coefficient.
    """
    timegan = TimeGAN(input_dim=11, hidden_dim=64, latent_dim=32, device=device)
    mse_loss = MSELoss()
    discriminator_criterion = BCEWithLogitsLoss()

    generator_optimizer = optim.Adam(timegan.generator_net.parameters(), lr=0.00005, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(timegan.discriminator_net.parameters(), lr=0.0001, betas=(0.5, 0.999))
    embedding_optimizer = optim.Adam(timegan.embedding_net.parameters(), lr=0.00005, betas=(0.5, 0.999))
    recovery_optimizer = optim.Adam(timegan.recovery_net.parameters(), lr=0.00005, betas=(0.5, 0.999))

    for epoch in range(epochs):
        epoch_loss = 0
        for real_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            real_data = real_data.to(device)
            wavelengths = real_data[:, :, 0:1]

            # Embedder training
            embeddings = timegan.embedding_net(real_data)
            recovered = timegan.recovery_net(embeddings)
            embed_loss = mse_loss(recovered, real_data)

            embedding_optimizer.zero_grad()
            embed_loss.backward()
            embedding_optimizer.step()

            # Recovery training
            recovery_optimizer.zero_grad()
            embed_loss.backward()
            recovery_optimizer.step()

            # Discriminator Training
            noise = torch.randn(real_data.size(0), real_data.size(1), 32).to(device)
            generated_data = timegan.generator_net(noise, wavelengths)

            real_labels = torch.ones(real_data.size(0), 1).to(device) * 0.9
            fake_labels = torch.zeros(real_data.size(0), 1).to(device) * 0.1

            real_data_noisy = real_data + 0.05 * torch.randn_like(real_data)
            generated_data_noisy = generated_data + 0.05 * torch.randn_like(generated_data)

            real_loss = discriminator_criterion(timegan.discriminator_net(real_data_noisy[:, :, 1:]), real_labels)
            fake_loss = discriminator_criterion(timegan.discriminator_net(generated_data_noisy.detach()), fake_labels)

            discriminator_loss = real_loss + fake_loss

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Generator Training
            generator_loss = 0
            for _ in range(2):
                noise = torch.randn(real_data.size(0), real_data.size(1), 32).to(device)
                generated_data = timegan.generator_net(noise, wavelengths)
                generator_loss = mse_loss(generated_data, real_data[:, :, 1:])
                
                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()

            # Accumulate losses
            epoch_loss += generator_loss.item() + discriminator_loss.item() + embed_loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    # Save model weights after training
    torch.save({
        'embedding_net': timegan.embedding_net.state_dict(),
        'recovery_net': timegan.recovery_net.state_dict(),
        'generator_net': timegan.generator_net.state_dict(),
        'discriminator_net': timegan.discriminator_net.state_dict(),
    }, 'new_timegan_weights.pth')
