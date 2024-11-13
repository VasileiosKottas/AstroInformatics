# src/data_loader.py
import pandas as pd
import numpy as np
import torch

def load_data(device):
    df_spectra = pd.read_csv('../data/spectra.csv').T
    df_spectra.columns = range(len(df_spectra.columns))
    data_sp = df_spectra.values.astype(np.float64) / np.max(df_spectra.values)
    
    df_photo = pd.read_csv('../data/interpolated_spectra.csv').T
    df_photo.columns = range(len(df_photo.columns))
    data_ph = df_photo.values.astype(np.float64) / np.max(df_photo.values)

    photometry_data = torch.tensor(data_ph.T, dtype=torch.float32).to(device)
    spectra_data = torch.tensor(data_sp.T, dtype=torch.float32).to(device)
    
    return photometry_data, spectra_data

def load_data_eval(device):
    
    df_spectra = pd.read_csv('../data/spectra.csv').T
    df_spectra.columns = range(len(df_spectra.columns))
    data_sp = df_spectra.values.astype(np.float64)
    data_scaled_sp = data_sp / np.max(data_sp)
    fluxes_sp = data_sp[:, :]  # Shape [10000, 17]

    df_photo = pd.read_csv('../data/interpolated_spectra.csv').T
    df_photo.columns = range(len(df_photo.columns))
    data_ph = df_photo.values.astype(np.float64)
    data_scaled_ph = data_ph / np.max(data_ph)

    # Load your photometry and spectra data here
    photometry_data = data_scaled_ph
    spectra_data = data_scaled_sp
    photometry_data = torch.tensor(photometry_data.T, dtype=torch.float32).to(device)
    spectra_data = torch.tensor(spectra_data.T, dtype=torch.float32).to(device)
    # After training, evaluate on a test dataset
    df_wavelengths = pd.read_csv("../data/interpolated_spectra.csv").T
    df_wavelengths.reset_index(inplace=True)
    df_wavelengths.columns = range(len(df_wavelengths.columns))
    df_wavelengths = df_wavelengths.values.astype(np.float64)
    df_wavelengths_scaled = df_wavelengths / np.max(df_wavelengths)
    data_wavelengths_photo = df_wavelengths[:, 0]  # Shape: [17]
    fluxes_ph = data_ph[:, 0:]  # Transpose to get shape [10000, 17]
    # Spectra
    df_wavelengths_spectra = pd.read_csv("../data/spectra.csv").T
    df_wavelengths_spectra.reset_index(inplace=True)
    df_wavelengths_spectra.columns = range(len(df_wavelengths_spectra.columns))
    df_wavelengths_spectra = df_wavelengths_spectra.values.astype(np.float64)
    df_wavelengths_scaled_spectra = df_wavelengths_spectra / np.max(df_wavelengths_spectra)
    data_wavelengths_spectra = df_wavelengths_spectra[:, 0]  # Shape: [17]

    photometry_wavelengths = data_wavelengths_photo   # 17 photometry wavelength points
    spectra_wavelengths = data_wavelengths_spectra   # 17 photometry wavelength points
    test_photometry_data = torch.tensor(data_scaled_ph.T, dtype=torch.float32).to(device)  # Reshape if necessary
    test_spectra_data = torch.tensor(data_scaled_sp.T, dtype=torch.float32).to(device)
    return spectra_wavelengths, photometry_wavelengths
