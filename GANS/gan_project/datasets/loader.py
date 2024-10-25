# datasets/loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

class GalaxyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)
    
def get_dataloader(data, batch_size=32, shuffle=True):
    dataset = GalaxyDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
