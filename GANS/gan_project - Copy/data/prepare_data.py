import numpy as np
from sklearn.preprocessing import StandardScaler

def fit_scaler(data):
    num_samples, seq_len, num_features = data.shape
    data_reshaped = data.reshape(-1, num_features)
    scaler = StandardScaler()
    scaler.fit(data_reshaped)
    return scaler

def normalize_data(data, scaler):
    num_samples, seq_len, num_features = data.shape
    data_reshaped = data.reshape(-1, num_features)
    normalized = scaler.transform(data_reshaped)
    return normalized.reshape(num_samples, seq_len, num_features)

def inverse_normalize_data(data, scaler):
    num_samples, seq_len, num_features = data.shape
    data_reshaped = data.reshape(-1, num_features)
    original = scaler.inverse_transform(data_reshaped)
    return original.reshape(num_samples, seq_len, num_features)
