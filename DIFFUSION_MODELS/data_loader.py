import torch
from torch.utils.data import Dataset
import numpy as np

class SpectraDataset(Dataset):
    def __init__(self, photometry_data, spectra_data, normalise=True):
        assert photometry_data.shape[0] == spectra_data.shape[0], "Mismatch in dataset sizes"
        
        # Normalise photometry and spectra if required
        if normalise:
            photometry_data /= np.max(np.abs(photometry_data), axis=0, keepdims=True)
            spectra_data /= np.max(np.abs(spectra_data), axis=0, keepdims=True)

        self.photometry = torch.tensor(photometry_data, dtype=torch.float32)
        self.spectra = torch.tensor(spectra_data, dtype=torch.float32)

    def __len__(self):
        return len(self.photometry)

    def __getitem__(self, idx):
        photometry = self.photometry[idx].unsqueeze(-1)  # Add sequence dimension
        spectra = self.spectra[idx]
        return photometry, spectra
