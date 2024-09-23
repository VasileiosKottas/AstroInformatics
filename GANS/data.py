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
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        # Return sequences of shape (seq_length, 1)
        seq = self.data[idx:idx + self.seq_length]
        return seq

def load_data():
    df = pd.read_csv('../data/interpolated_spectra.csv')
    print(df.columns)
    df = df.drop('Galaxy1_Interpolated_Wavelength', axis='columns')
    df = df.dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values).astype(np.float32)
    len(scaled_data)
    
    data = scaled_data[1:]
    
    n_windows = len(data)
        
    
    return data, n_windows
