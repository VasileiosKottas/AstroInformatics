import torch
import matplotlib.pyplot as plt
from tsdiff import SelfGuidedTSDiff
from data_loader import SpectraDataset
from torch.utils.data import DataLoader, random_split
from plotting import plot_spectra
from hyperparameters import batch_size, time_steps, input_dim
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import random

def calculate_chi2(real_spectra, generated_spectra):
    uncertainty = 0.1  # Assumed uncertainty as 10% of values
    diff = real_spectra - generated_spectra
    denom = (uncertainty * real_spectra) ** 2 + (uncertainty * generated_spectra) ** 2
    chi2 = torch.sum((diff ** 2) / denom, dim=1) / real_spectra.size(1)
    return chi2

def evaluate_model(model, test_dataloader, device, spectra_wavelengths, photometry_wavelengths):
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

    # Filter wavelengths and fluxes for wavelengths < 35
    valid_spectra_indices = spectra_wavelengths < 35
    valid_photometry_indices = photometry_wavelengths < 35

    filtered_spectra_wavelengths = spectra_wavelengths[valid_spectra_indices]
    filtered_photometry_wavelengths = photometry_wavelengths[valid_photometry_indices]

    # Filter the spectra data (fluxes corresponding to the valid wavelengths)
    all_real_spectra = all_real_spectra[:, valid_spectra_indices]
    all_generated_spectra = all_generated_spectra[:, valid_spectra_indices]
    all_photometry_data = all_photometry_data[:, valid_photometry_indices]

    # Calculate chi-squared values
    chi2_values = calculate_chi2(all_real_spectra, all_generated_spectra)

    # Select 100 random galaxies (samples)
    num_samples = 100
    random_indices = random.sample(range(len(all_real_spectra)), num_samples)

    # Extract chi-squared values for the selected galaxies
    selected_chi2_values = chi2_values[random_indices]

    # Filter chi-squared values to include only those < 5
    filtered_chi2_values = selected_chi2_values[selected_chi2_values < 5]

    # Calculate evaluation metrics
    mse = mean_squared_error(all_real_spectra.cpu().numpy(), all_generated_spectra.cpu().numpy())
    chi2_mean = torch.mean(filtered_chi2_values).item()

    print(f"Evaluation Metrics:\nMSE: {mse:.6f}\nChi^2 Mean: {chi2_mean:.6f}")

    # Plot Chi-Squared distribution for filtered values
    plt.figure(figsize=(8, 5))
    plt.hist(filtered_chi2_values.cpu().numpy(), bins=30, color='orange', alpha=0.7)
    plt.xlabel('Chi-Squared Value')
    plt.ylabel('Frequency')
    plt.title('Chi-Squared Value Distribution (Filtered χ² < 5)')
    plt.grid(True)
    plt.savefig("chi2_distribution_filtered.png")
    plt.show()

    # Create a histogram for the chi-squared values from the random samples
    plt.figure(figsize=(8, 5))
    plt.hist(selected_chi2_values.cpu().numpy(), bins=30, color='blue', alpha=0.7)
    plt.xlabel('Chi-Squared Value')
    plt.ylabel('Frequency')
    plt.title('Chi-Squared Value Distribution (Random Samples)')
    plt.grid(True)
    plt.savefig("chi2_distribution_random_samples.png")
    plt.show()

    # Create a single plot with 6 subplots for the first 6 samples
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns for 6 samples
    axes = axes.flatten()  # Flatten to make indexing easier

    for i in range(6):  # Plot the first 6 samples
        real_sample = all_real_spectra[random_indices[i], :].cpu().numpy()
        photometry_data = all_photometry_data[random_indices[i], :].cpu().numpy()
        generated_sample = all_generated_spectra[random_indices[i], :].cpu().numpy()

        axes[i].plot(filtered_spectra_wavelengths, real_sample, label='Real Spectra')
        axes[i].plot(filtered_spectra_wavelengths, generated_sample, label='Generated Spectra', linestyle='--')
        axes[i].scatter(filtered_photometry_wavelengths, photometry_data, label='Photometry', color='red')
        axes[i].set_title(f'Galaxy {i+1}')
        axes[i].set_xlabel('Wavelength (µm)')
        axes[i].set_ylabel('Flux')
        axes[i].legend()
        axes[i].grid(True)

    # Adjust layout and save the combined plot with 6 subplots
    plt.tight_layout()
    plt.savefig("Diffusion_eval_with_metrics_subplots.png")  # Save the combined figure with 6 subplots
    plt.show()

if __name__ == "__main__":
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

    # Load trained model
    model = SelfGuidedTSDiff(input_dim=input_dim, time_steps=time_steps).to(device)
    model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))

    # Run evaluation
    evaluate_model(model, test_dataloader, device, spectra_wavelengths, photometry_wavelengths)
