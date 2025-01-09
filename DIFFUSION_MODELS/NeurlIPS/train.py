import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, device, criterion, optimizer, output_dir="training_metrics"):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, num_epochs, timesteps):
        train_losses = []
        val_losses = []
        epochs = []

        for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
            self.model.train()
            train_loss = 0

            for photometry_data, real_spectra in self.train_dataloader:
                photometry_data = photometry_data.to(self.device)
                real_spectra = real_spectra.to(self.device)

                # Model prediction
                generated_spectra = self.model(photometry_data, t=None)
                
                # Loss computation
                loss = self.criterion(generated_spectra.squeeze(-1), real_spectra)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Validation step
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for photometry_data, real_spectra in self.val_dataloader:
                    photometry_data = photometry_data.to(self.device)
                    real_spectra = real_spectra.to(self.device)

                    generated_spectra = self.model(photometry_data, t=None)
                    loss = self.criterion(generated_spectra.squeeze(-1), real_spectra)
                    val_loss += loss.item()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            epochs.append(epoch)

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.7)
        plt.plot(epochs, val_losses, label='Val Loss', color='orange', alpha=0.7)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.output_dir, f"loss_num_epochs_{len(epochs)}.png"))
        plt.close()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
