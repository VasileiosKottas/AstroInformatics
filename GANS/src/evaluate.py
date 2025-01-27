import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.model import PhotometryToSpectraModel
from src.data_loader import load_data
from config.hyperparameters import DEVICE, PHOTOMETRY_DIM, SPECTRA_DIM, LATENT_DIM, TRANSFORMER_HIDDEN_DIM, TRANSFORMER_NHEAD, TRANSFORMER_LAYERS
import random

def calculate_chi2(real_spectra, generated_spectra):
    uncertainty = 0.1  # Assumed uncertainty as 10% of values
    diff = real_spectra - generated_spectra
    denom = (uncertainty * real_spectra) ** 2 + (uncertainty * generated_spectra) ** 2
    chi2 = torch.sum((diff ** 2) / denom, dim=1) / real_spectra.size(1)
    return chi2

def evaluate_model(model, test_photometry_data, test_spectra_data, photometry_wavelengths, spectra_wavelengths):
    model.eval()
    with torch.no_grad():
        generated_spectra = model.forward_generator(test_photometry_data)
        
        # Calculate MSE and MAE
        mse = mean_squared_error(test_spectra_data.cpu(), generated_spectra.cpu())
        mae = mean_absolute_error(test_spectra_data.cpu(), generated_spectra.cpu())
        
        # Calculate chi-squared values
        chi2_values = calculate_chi2(test_spectra_data, generated_spectra)
        chi2_mean = torch.mean(chi2_values).item()

        print(f"Test MSE: {mse}")
        print(f"Test MAE: {mae}")
        print(f"Chi^2 Mean: {chi2_mean}")

        # Filter wavelengths and fluxes for wavelengths < 35
        valid_spectra_indices = spectra_wavelengths < 35
        valid_photometry_indices = photometry_wavelengths < 35

        filtered_spectra_wavelengths = spectra_wavelengths[valid_spectra_indices]
        filtered_photometry_wavelengths = photometry_wavelengths[valid_photometry_indices]

        # Filter the spectra data (fluxes corresponding to the valid wavelengths)
        test_spectra_data = test_spectra_data[:, valid_spectra_indices]
        generated_spectra = generated_spectra[:, valid_spectra_indices]
        test_photometry_data = test_photometry_data[:, valid_photometry_indices]

        # Calculate filtered chi-squared values
        chi2_values = calculate_chi2(test_spectra_data, generated_spectra)

        # Create a single plot with 6 subplots for the first 6 samples
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns for 6 samples
        axes = axes.flatten()  # Flatten to make indexing easier

        # Loop over the first 6 samples to plot each
        for i in range(6):
            real_sample = test_spectra_data[i, :].cpu().numpy()
            photometry_data = test_photometry_data[i, :].cpu().numpy()
            generated_sample = generated_spectra[i, :].cpu().numpy()

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
        plt.savefig("Gan_eval_with_metrics_subplots.png")  # Save the combined figure
        plt.show()

        # Chi-Squared distribution plot for filtered chi^2 < 5
        filtered_chi2_values = chi2_values[chi2_values < 5]
        
        plt.figure(figsize=(8, 5))
        plt.hist(filtered_chi2_values.cpu().numpy(), bins=30, color='orange', alpha=0.7)
        plt.xlabel('Chi-Squared Value')
        plt.ylabel('Frequency')
        plt.title('Chi-Squared Value Distribution (Filtered χ² < 5)')
        plt.grid(True)
        plt.savefig("chi2_distribution_filtered.png")  # Save the filtered Chi-squared histogram
        plt.show()

        # Create a histogram for the chi-squared values from the whole batch
        plt.figure(figsize=(8, 5))
        plt.hist(chi2_values.cpu().numpy(), bins=30, color='blue', alpha=0.7)
        plt.xlabel('Chi-Squared Value')
        plt.ylabel('Frequency')
        plt.title('Chi-Squared Value Distribution (All Samples)')
        plt.grid(True)
        plt.savefig("chi2_distribution_all_samples.png")  # Save the chi-squared histogram for all samples
        plt.show()