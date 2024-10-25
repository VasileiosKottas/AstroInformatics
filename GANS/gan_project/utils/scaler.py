# utils/scaler.py
from sklearn.preprocessing import StandardScaler
import numpy as np

def fit_scaler(data):
    fluxes = data[:, :, 1:]
    scaler = StandardScaler()
    scaler.fit(fluxes.reshape(-1, fluxes.shape[-1]))
    return scaler

def normalize_data(data, scaler):
    wavelengths = data[:, :, 0:1]
    fluxes = data[:, :, 1:]
    normalized_fluxes = scaler.transform(fluxes.reshape(-1, fluxes.shape[-1]))
    return np.concatenate([wavelengths, normalized_fluxes.reshape(data.shape[0], data.shape[1], -1)], axis=2)

def inverse_normalize_data(data, scaler):
    fluxes = data[:, :, 1:]
    original_fluxes = scaler.inverse_transform(fluxes.reshape(-1, fluxes.shape[-1]))
    return original_fluxes.reshape(data.shape[0], data.shape[1], -1)
