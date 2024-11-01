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

class Trainer:
    def __init__(self, model, dataloader, device, criterion, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            for i, (real_spectra, photometry_data) in enumerate(self.dataloader):
                real_spectra = real_spectra.to(self.device)
                photometry_data = photometry_data.to(self.device)

                # Randomly generate time steps for each sample in the batch
                t = torch.randint(0, time_steps, (real_spectra.size(0),)).to(self.device)

                # Reshape photometry data to be [batch_size, input_dim, seq_len]
                photometry_data = photometry_data.unsqueeze(2)  # Ensure [batch_size, 17, 1]

                # Forward pass
                output = self.model(photometry_data, t)

                # Compute loss
                loss = self.criterion(output, real_spectra)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(self.dataloader):.4f}")
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
