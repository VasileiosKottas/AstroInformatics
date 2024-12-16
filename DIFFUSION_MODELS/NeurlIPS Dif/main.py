import torch
from models.tsdiff import SelfGuidedTSDiff
from utils.data_loader import SpectraDataset
from utils.plotting import plot_spectra
from utils.chi_squared import compute_chi_squared
from training.train import Trainer
from hyperparameters import batch_size, num_epochs, input_dim, time_steps, learning_rate
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import os

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Paths for data
spectra_file = '../../data/spectra.csv'
photometry_file = '../../data/interpolated_spectra.csv'

# Load and preprocess data
df_spectra = pd.read_csv(spectra_file).T
data_sp = df_spectra.values.astype(np.float64)
data_scaled_sp = data_sp / np.max(data_sp)

df_photo = pd.read_csv(photometry_file).T
data_ph = df_photo.values.astype(np.float64)
data_scaled_ph = data_ph / np.max(data_sp)

# Initialize dataset
dataset = SpectraDataset(data_scaled_ph.T, data_scaled_sp.T)

# Train-test split (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Initialize model, optimizer, and criterion
model = SelfGuidedTSDiff(input_dim=input_dim, time_steps=time_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Train the model
trainer = Trainer(model, train_dataloader, device, criterion, optimizer)
trainer.train(num_epochs)
trainer.save_model('trained_model.pth')

# Evaluate and generate spectra after training
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Assuming test_dataloader is already defined
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
def evaluate_on_test(model, test_dataloader, device):
    all_real_spectra = []
    all_generated_spectra = []
    all_photometry = []

    with torch.no_grad():  # No gradients needed during evaluation
        for real_spectra, photometry_data in test_dataloader:
            # Move data to the device
            real_spectra = real_spectra.to(device)
            photometry_data = photometry_data.to(device).unsqueeze(2)  # Add sequence dimension

            # Start denoising from Gaussian noise
            batch_size = real_spectra.size(0)  # Get batch size
            noise = torch.randn((batch_size, 17, 206)).to(device)  # Match model input
            for t in reversed(range(time_steps)):
                t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
                noise = model.denoise(noise, t_tensor)  # Apply denoising

            # Append results for comparison
            all_real_spectra.append(real_spectra.detach().cpu())
            all_generated_spectra.append(noise.detach().cpu())
            all_photometry.append(photometry_data.detach().cpu())

    return torch.cat(all_real_spectra), torch.cat(all_generated_spectra), torch.cat(all_photometry)

# Evaluate on the test set
real_spectra, generated_spectra, photometry = evaluate_on_test(model, test_dataloader, device)

# Plot a few test results
for i in range(1):  # Visualize 5 examples
    plot_spectra(
        real_spectra=real_spectra,
        photometry=photometry,
        generated_spectra=generated_spectra,
        photometry_wavelengths=data_scaled_ph[0],  # Replace with photometry wavelengths
        spectra_wavelengths=data_scaled_sp[0],     # Replace with spectra wavelengths
        index=i
    )

# # Denoising and generating spectra
# def generate_spectra(model, num_samples, sequence_length=206):
#     # Initialize noise with the correct shape
#     noise = torch.randn((num_samples, 17, sequence_length)).to(device)  # Match model input
    
#     # Reverse diffusion process
#     for t in reversed(range(time_steps)):
#         t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)  # Timestep tensor
#         noise = model.denoise(noise, t_tensor)
#     return noise


# # Generate new spectra
# num_samples = 2
# with torch.no_grad():
#     generated_spectra = generate_spectra(model, num_samples)

# # Visualize the generated spectra
# real_spectra, photometry_data = next(iter(test_dataloader))
# test_photometry_data = photometry_data[:num_samples].to(device)
# test_real_spectra = real_spectra[:num_samples].to(device)

# # Plot the results
# plot_spectra(
#     test_real_spectra,        # Real spectra
#     test_photometry_data,     # Input photometry
#     generated_spectra,        # Generated spectra
#     data_scaled_ph[0],        # Photometry wavelengths
#     data_scaled_sp[0],        # Spectra wavelengths
# )
