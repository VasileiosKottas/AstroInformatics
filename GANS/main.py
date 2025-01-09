# main.py
import sys
from config.hyperparameters import *
from src.train import train
from src.evaluate import evaluate_model
from src.model import PhotometryToSpectraModel
from src.data_loader import load_data, load_data_eval
from torch.utils.data import random_split, TensorDataset

# Device configuration
device = DEVICE
print("Using device:", device)

# Step 1: Load data
photometry_data, spectra_data = load_data(device)

# Step 2: Create TensorDataset
dataset = TensorDataset(photometry_data, spectra_data)

# Step 3: Train-validation-test split (80% train, 10% validation, 10% test)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Extract full datasets for training and validation
train_photometry_data = torch.cat([train_dataset[i][0].unsqueeze(0) for i in range(len(train_dataset))], dim=0)
train_spectra_data = torch.cat([train_dataset[i][1].unsqueeze(0) for i in range(len(train_dataset))], dim=0)

val_photometry_data = torch.cat([val_dataset[i][0].unsqueeze(0) for i in range(len(val_dataset))], dim=0)
val_spectra_data = torch.cat([val_dataset[i][1].unsqueeze(0) for i in range(len(val_dataset))], dim=0)

# Extract test data
test_photometry_data = torch.cat([test_dataset[i][0].unsqueeze(0) for i in range(len(test_dataset))], dim=0)
test_spectra_data = torch.cat([test_dataset[i][1].unsqueeze(0) for i in range(len(test_dataset))], dim=0)

# Load wavelengths for evaluation
spectra_wavelengths, photometry_wavelengths = load_data_eval(device)

# Step 4: Initialize model
model = PhotometryToSpectraModel(
    photometry_dim=PHOTOMETRY_DIM,
    spectra_dim=SPECTRA_DIM,
    latent_dim=LATENT_DIM,
    transformer_hidden_dim=TRANSFORMER_HIDDEN_DIM,
    transformer_nhead=TRANSFORMER_NHEAD,
    transformer_layers=TRANSFORMER_LAYERS
).to(device)

# Step 5: Train the model
train(model, train_photometry_data, train_spectra_data, val_photometry_data, val_spectra_data)

# Step 6: Evaluate the model
evaluate_model(
    model, 
    test_photometry_data.to(device), 
    test_spectra_data.to(device), 
    photometry_wavelengths, 
    spectra_wavelengths
)
