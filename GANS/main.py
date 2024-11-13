# main.py
import sys
from config.hyperparameters import *
from src.train import train_model
from src.evaluate import evaluate_model
from src.model import PhotometryToSpectraModel
from src.data_loader import load_data, load_data_eval

device = DEVICE
photometry_data, spectra_data = load_data(device)
model = PhotometryToSpectraModel(
    photometry_dim=PHOTOMETRY_DIM,
    spectra_dim=SPECTRA_DIM,
    latent_dim=LATENT_DIM,
    transformer_hidden_dim=TRANSFORMER_HIDDEN_DIM,
    transformer_nhead=TRANSFORMER_NHEAD,
    transformer_layers=TRANSFORMER_LAYERS
).to(device)

# Train the model
train_model(model, photometry_data, spectra_data)

spectra_wavelengths, photometry_wavelengths = load_data_eval(device)
# Evaluate the model
evaluate_model(model, photometry_data, spectra_data, photometry_wavelengths, spectra_wavelengths)
