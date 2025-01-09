import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.model import PhotometryToSpectraModel
from src.data_loader import load_data
from config.hyperparameters import DEVICE, PHOTOMETRY_DIM, SPECTRA_DIM, LATENT_DIM, TRANSFORMER_HIDDEN_DIM, TRANSFORMER_NHEAD, TRANSFORMER_LAYERS

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
        mse = mean_squared_error(test_spectra_data.cpu(), generated_spectra.cpu())
        mae = mean_absolute_error(test_spectra_data.cpu(), generated_spectra.cpu())
        chi2_values = calculate_chi2(test_spectra_data, generated_spectra)
        chi2_mean = torch.mean(chi2_values).item()

        print(f"Test MSE: {mse}")
        print(f"Test MAE: {mae}")
        print(f"Chi^2 Mean: {chi2_mean}")

        # Plot example spectra
        real_sample = test_spectra_data[4, :].cpu().numpy()
        photometry_data = test_photometry_data[4, :].cpu().numpy()
        generated_sample = generated_spectra[4, :].cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(spectra_wavelengths, real_sample, label='Real Spectra')
        plt.plot(spectra_wavelengths, generated_sample, label='Generated Spectra', linestyle='--')
        plt.scatter(photometry_wavelengths, photometry_data, label='Photometry', color='red')
        plt.xlabel('Wavelength (µm)')
        plt.ylabel('Flux')
        plt.legend()
        plt.grid(True)
        plt.savefig("Gan_eval_with_metrics.png")  # Save the example spectra plot
        plt.show()

        # Chi-Squared distribution plot
        plt.figure(figsize=(8, 5))
        plt.hist(chi2_values.cpu().numpy(), bins=30, color='orange', alpha=0.7)
        plt.xlabel('Chi-Squared Value')
        plt.ylabel('Frequency')
        plt.title('Chi-Squared Value Distribution')
        plt.grid(True)
        plt.savefig("chi2_distribution.png")  # Save the Chi-squared histogram
        plt.show()
