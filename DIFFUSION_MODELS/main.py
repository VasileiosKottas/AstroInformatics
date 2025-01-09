import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
from tsdiff import SelfGuidedTSDiff
from data_loader import SpectraDataset
from train import Trainer
from hyperparameters import batch_size, num_epochs, learning_rate, time_steps, input_dim
from plotting import plot_spectra
from chi_squared import calculate_chi2
import matplotlib.pyplot as plt
import os
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df_spectra = pd.read_csv('../data/spectra.csv', header=None)
df_photometry = pd.read_csv('../data/interpolated_spectra.csv', header=None)

# Extract wavelengths
spectra_wavelengths = df_spectra.iloc[0].values.astype(np.float64)
photometry_wavelengths = df_photometry.iloc[0].values.astype(np.float64)

# Flux values only
data_spectra = df_spectra.iloc[1:].values.astype(np.float64)
data_photometry = df_photometry.iloc[1:].values.astype(np.float64)

# Dataset initialization
dataset = SpectraDataset(data_photometry, data_spectra, normalise=True)

# Train-validation-test split
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# Model, optimizer, and criterion
model = SelfGuidedTSDiff(input_dim=input_dim, time_steps=time_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()  # Use MSE for spectra prediction

# Train the model
trainer = Trainer(model, train_dataloader, val_dataloader, device, criterion, optimizer)
trainer.train(num_epochs, time_steps)
trainer.save_model('trained_model.pth')

# Evaluate the model
print("Evaluating the model...")
model.load_state_dict(torch.load("trained_model.pth"))  # Load the trained model
model.eval()

all_real_spectra, all_generated_spectra, all_photometry_data = [], [], []
with torch.no_grad():
    for photometry_data, real_spectra in test_dataloader:
        photometry_data, real_spectra = photometry_data.to(device), real_spectra.to(device)
        generated_spectra = model(photometry_data, t=None)
        all_real_spectra.append(real_spectra)
        all_generated_spectra.append(generated_spectra.squeeze(-1))
        all_photometry_data.append(photometry_data)

all_real_spectra = torch.cat(all_real_spectra)
all_generated_spectra = torch.cat(all_generated_spectra)
all_photometry_data = torch.cat(all_photometry_data)

# Calculate metrics
mse = torch.mean((all_real_spectra - all_generated_spectra) ** 2).item()
chi2_values = calculate_chi2(all_real_spectra, all_generated_spectra)
chi2_mean = torch.mean(chi2_values).item()

print(f"Evaluation Metrics:\nMSE: {mse:.6f}\nChi^2 Mean: {chi2_mean:.6f}")

# Plot Chi-Squared distribution
plt.figure(figsize=(8, 5))
plt.hist(chi2_values.cpu().numpy(), bins=30, color='orange', alpha=0.7)
plt.xlabel('Chi-Squared Value')
plt.ylabel('Frequency')
plt.title('Chi-Squared Value Distribution')
plt.grid(True)
plt.savefig("chi2_distribution.png")
plt.show()

# Plot results
print("Plotting results...")
for i in range(5):  # Plot first 5 samples
    plot_spectra(
        real_spectra=all_real_spectra,
        photometry=all_photometry_data,
        generated_spectra=all_generated_spectra,
        photometry_wavelengths=photometry_wavelengths,
        spectra_wavelengths=spectra_wavelengths,
        index=i
    )

print("Evaluation complete.")
