# main.py
import sys
from config.hyperparameters import *
from src.train import train_model
from src.evaluate import evaluate_model
from src.model import PhotometryToSpectraModel
from src.data_loader import load_data, load_data_eval
from torch.utils.data import random_split, TensorDataset
if __name__ == "__main__":
    # Device configuration
    device = DEVICE

    # Step 1: Load data
    photometry_data, spectra_data = load_data(device)

    # Step 2: Create TensorDataset
    dataset = TensorDataset(photometry_data, spectra_data)

    # Step 3: Train-test split (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Step 4: Extract full datasets for training
    train_photometry_data = torch.cat([train_dataset[i][0].unsqueeze(0) for i in range(len(train_dataset))], dim=0)
    train_spectra_data = torch.cat([train_dataset[i][1].unsqueeze(0) for i in range(len(train_dataset))], dim=0)

    # Extract test data
    test_photometry_data = torch.cat([test_dataset[i][0].unsqueeze(0) for i in range(len(test_dataset))], dim=0)
    test_spectra_data = torch.cat([test_dataset[i][1].unsqueeze(0) for i in range(len(test_dataset))], dim=0)
    spectra_wavelengths, photometry_wavelengths = load_data_eval(device)
    model = PhotometryToSpectraModel(
        photometry_dim=PHOTOMETRY_DIM,
        spectra_dim=SPECTRA_DIM,
        latent_dim=LATENT_DIM,
        transformer_hidden_dim=TRANSFORMER_HIDDEN_DIM,
        transformer_nhead=TRANSFORMER_NHEAD,
        transformer_layers=TRANSFORMER_LAYERS
    ).to(device)
    model.load_state_dict(torch.load("./gan_model.pth"))
    print("Model loaded successfully.")
    
    # Step 3: Evaluate the model
    evaluate_model(
        model, 
        test_photometry_data, 
        test_spectra_data, 
        photometry_wavelengths, 
        spectra_wavelengths
    )
