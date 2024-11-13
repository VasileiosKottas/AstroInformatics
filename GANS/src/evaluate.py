# src/evaluate.py
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.model import PhotometryToSpectraModel
from src.data_loader import load_data
from config.hyperparameters import DEVICE, PHOTOMETRY_DIM, SPECTRA_DIM, LATENT_DIM, TRANSFORMER_HIDDEN_DIM, TRANSFORMER_NHEAD, TRANSFORMER_LAYERS

def evaluate_model(model, test_photometry_data, test_spectra_data, photometry_wavelengths, spectra_wavelengths):
    model.eval()
    with torch.no_grad():
        generated_spectra = model.forward_generator(test_photometry_data)
        mse = mean_squared_error(test_spectra_data.cpu(), generated_spectra.cpu())
        mae = mean_absolute_error(test_spectra_data.cpu(), generated_spectra.cpu())
        
        print(f"Test MSE: {mse}")
        print(f"Test MAE: {mae}")
        
        real_sample = test_spectra_data[4,:].cpu().numpy()
        photometry_data = test_photometry_data[4,:].cpu().numpy()
        generated_sample = generated_spectra[4,:].cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(spectra_wavelengths, real_sample, label='Real Spectra')
        plt.plot(spectra_wavelengths, generated_sample, label='Generated Spectra', linestyle='--')
        plt.scatter(photometry_wavelengths, photometry_data, label='Photometry', color='red')
        plt.xlabel('Wavelength (µm)')
        plt.ylabel('Flux')
        plt.legend()
        plt.grid(True)
        plt.show()
