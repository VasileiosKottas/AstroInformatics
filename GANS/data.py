import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch 

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        # Directly convert numpy array to torch tensor
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        # Length is reduced since we create sequences of seq_length
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Return sequences of shape (seq_length, 1)
        seq = self.data[idx:idx + self.seq_length]
        # seq = seq.view(1, 1)
        return seq

def load_data():
    df = pd.read_csv('../data/interpolated_spectra.csv')
    df = df.dropna()
    df = df.T
    print(len(df))
    print(df.shape)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values).astype(np.float32)
    
    # Reshape to (17, 1, 1)
    # Assuming you want to take the first time step from each sample
    scaled_data = scaled_data[:, 0, np.newaxis]  # This will give you shape (17, 1, 1)
    
    print(type(scaled_data), scaled_data.shape)  # Check the type and shape
    n_windows = len(scaled_data)
    
    return scaled_data, n_windows
