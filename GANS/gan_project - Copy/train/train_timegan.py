import torch.optim as optim
import torch
import torch.nn as nn
from models.timegan import TimeGAN
from data.prepare_data import normalize_data, inverse_normalize_data
from train.hyperparameters import input_dim, hidden_dim, latent_dim, batch_size, epochs, learning_rate


def train_timegan(train_loader, scaler, device, epochs=1000):
    # Initialize TimeGAN model and optimizers
    input_dim = 11
    hidden_dim = 64
    latent_dim = 32
    sequence_length = 17
    
    timegan = TimeGAN(input_dim, hidden_dim, latent_dim, device)
    
    # Define loss criteria
    mse_loss = nn.MSELoss()  # Use MSE for the generator loss
    discriminator_criterion = nn.BCEWithLogitsLoss()  # Still use BCE for discriminator

    # Initialize optimizers with different learning rates
    generator_optimizer = optim.Adam(timegan.generator_net.parameters(), lr=0.0001, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(timegan.discriminator_net.parameters(), lr=0.0004, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(epochs):
        for real_data in train_loader:
            real_data = real_data.to(device)

            # Train Discriminator (multiple steps)
            for _ in range(5):  # More discriminator updates to address mode collapse
                noise = torch.randn(real_data.size(0), sequence_length, latent_dim).to(device)
                generated_data = timegan.generator_net(noise)
                
                # Label smoothing for real data
                # Label smoothing for real data
                real_labels = torch.ones(real_data.size(0), 1).to(device) * 0.9  # Real labels slightly less than 1
                fake_labels = torch.zeros(real_data.size(0), 1).to(device)      # Fake labels as 0

                # Discriminator loss
                # Discriminator loss
                real_loss = discriminator_criterion(timegan.discriminator_net(real_data), real_labels)
                fake_loss = discriminator_criterion(timegan.discriminator_net(generated_data.detach()), fake_labels)

                discriminator_loss = (real_loss + fake_loss) / 2
                
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()

            # Train Generator (single step with MSE)
            noise = torch.randn(real_data.size(0), sequence_length, latent_dim).to(device)
            generated_data = timegan.generator_net(noise)
            
            # Calculate the generator loss as MSE between real and generated data
            generator_loss = mse_loss(generated_data, real_data)
            
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        # Print loss periodically
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Generator MSE Loss: {generator_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}")

    # Save trained model weights
    torch.save({
        'embedding_net': timegan.embedding_net.state_dict(),
        'recovery_net': timegan.recovery_net.state_dict(),
        'generator_net': timegan.generator_net.state_dict(),
        'discriminator_net': timegan.discriminator_net.state_dict(),
    }, 'timegan_weights.pth')
