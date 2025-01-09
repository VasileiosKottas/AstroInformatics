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

    # Calculate metrics
    mse = mean_squared_error(all_real_spectra.cpu().numpy(), all_generated_spectra.cpu().numpy())
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
    plt.savefig("chi2_distribution_diffusion.png")
    plt.show()

    # Visualise spectra
    for i in range(5):  # Plot first 5 samples
        plot_spectra(
            real_spectra=all_real_spectra,
            photometry=all_photometry_data,
            generated_spectra=all_generated_spectra,
            photometry_wavelengths=photometry_wavelengths,
            spectra_wavelengths=spectra_wavelengths,
            index=i
        )

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
    model.load_state_dict(torch.load("trained_model.pth"))

    # Run evaluation
    evaluate_model(model, test_dataloader, device, spectra_wavelengths, photometry_wavelengths)
