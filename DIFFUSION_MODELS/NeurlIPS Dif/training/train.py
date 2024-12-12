# training/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import SpectraDataset
from models.tsdiff import SelfGuidedTSDiff
from hyperparameters import batch_size, num_epochs, learning_rate, input_dim, time_steps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Add imports
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.distributions as dist

# Import necessary modules
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.distributions as dist
import os
from tqdm import tqdm  # Import tqdm for progress tracking

class Trainer:
    def __init__(self, model, dataloader, device, criterion, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.output_dir = "training_metrics"
        os.makedirs(self.output_dir, exist_ok=True)

    def crps(self, y_true, y_pred):
        """Compute Continuous Ranked Probability Score (CRPS)"""
        normal = dist.Normal(loc=y_pred, scale=1.0)  # Assuming fixed std
        return torch.mean(torch.abs(y_true - normal.cdf(y_true)))

    def train(self, num_epochs):
        losses, mses, maes, crps_scores = [], [], [], []

        for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
            self.model.train()
            epoch_loss, epoch_mse, epoch_mae, epoch_crps = 0, 0, 0, 0

            for real_spectra, photometry_data in self.dataloader:
                real_spectra = real_spectra.to(self.device)
                photometry_data = photometry_data.to(self.device)
                t = torch.randint(0, time_steps, (real_spectra.size(0),)).to(self.device)
                photometry_data = photometry_data.unsqueeze(2)

                # Forward pass
                output = self.model(photometry_data, t)

                # Compute loss
                loss = self.criterion(output, real_spectra)

                # Metrics
                mse = mean_squared_error(real_spectra.cpu().numpy(), output.detach().cpu().numpy())
                mae = mean_absolute_error(real_spectra.cpu().numpy(), output.detach().cpu().numpy())
                crps_score = self.crps(real_spectra, output)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_mse += mse
                epoch_mae += mae
                epoch_crps += crps_score.item()

            # Log metrics
            losses.append(epoch_loss / len(self.dataloader))
            mses.append(epoch_mse / len(self.dataloader))
            maes.append(epoch_mae / len(self.dataloader))
            crps_scores.append(epoch_crps / len(self.dataloader))

            tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {losses[-1]:.4f}, MSE: {mses[-1]:.4f}, MAE: {maes[-1]:.4f}, CRPS: {crps_scores[-1]:.4f}")

        # Save plots for each metric
        self.save_plot(range(num_epochs), losses, "Epoch", "Loss", "Training Loss", os.path.join(self.output_dir, "loss_plot.png"))
        self.save_plot(range(num_epochs), mses, "Epoch", "MSE", "Mean Squared Error (MSE)", os.path.join(self.output_dir, "mse_plot.png"))
        self.save_plot(range(num_epochs), maes, "Epoch", "MAE", "Mean Absolute Error (MAE)", os.path.join(self.output_dir, "mae_plot.png"))
        self.save_plot(range(num_epochs), crps_scores, "Epoch", "CRPS", "Continuous Ranked Probability Score (CRPS)", os.path.join(self.output_dir, "crps_plot.png"))

    def save_plot(self, x, y, xlabel, ylabel, title, save_path):
        """Helper function to save plots."""
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label=title, color='blue')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
