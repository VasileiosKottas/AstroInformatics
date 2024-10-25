import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from hyperparameters import *

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, index):
        return self.data[index:index + self.seq_len]

def data():
    # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    df = pd.read_csv('../data/interpolated_spectra.csv')
    df = df.dropna()
    df = df.T
    # Assuming df is your DataFrame
    df = df.reset_index()  # Move the index to a new column
    wave = df['index']
    df = df.drop(columns='index')
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values).astype(np.float32)
    data = []
    for i in range(len(df) - SEQ_LEN):
        data.append(scaled_data[1:])
    # data.append(scaled_data[1])
    data = torch.tensor(data)
    n_windows = len(data)
    
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    real_batch = dataloader
    
    random_data = data * torch.rand_like(data).float()
    
    random_dataset = TimeSeriesDataset(random_data, SEQ_LEN)
    random_dataloader = random_dataset
    random_batch = random_dataloader
    # print(data.shape)
    return real_batch, random_batch
