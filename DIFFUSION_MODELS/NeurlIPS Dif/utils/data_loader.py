import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class SpectraDataset(Dataset):
    def __init__(self, photometry_data, spectra_data):
        assert photometry_data.shape[0] == spectra_data.shape[0], "Mismatch in number of samples"
        self.photometry = torch.tensor(photometry_data, dtype=torch.float32)
        self.spectra = torch.tensor(spectra_data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.photometry)
    
    def __getitem__(self, idx):
        return self.spectra[idx], self.photometry[idx]