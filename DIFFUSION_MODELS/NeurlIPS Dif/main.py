import torch
from models.tsdiff import SelfGuidedTSDiff
from utils.data_loader import SpectraDataset
from utils.plotting import plot_spectra
from training.train import Trainer
from hyperparameters import batch_size, num_epochs, input_dim, time_steps, learning_rate
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
spectra_file = '../../data/spectra.csv'
photometry_file = '../../data/interpolated_spectra.csv'

df_spectra = pd.read_csv(spectra_file).T
data_sp = df_spectra.values.astype(np.float64)
data_scaled_sp = data_sp / np.max(data_sp)

df_photo = pd.read_csv(photometry_file).T
data_ph = df_photo.values.astype(np.float64)
data_scaled_ph = data_ph / np.max(data_ph)

# Initialize dataset and DataLoader
dataset = SpectraDataset(data_scaled_ph.T, data_scaled_sp.T)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize model, optimizer, and criterion
model = SelfGuidedTSDiff(input_dim=input_dim, time_steps=time_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Initialize trainer and train the model
trainer = Trainer(model, dataloader, device, criterion, optimizer)
trainer.train(num_epochs)
trainer.save_model('trained_model.pth')

# Optionally: Evaluate and plot results
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

real_spectra, photometry_data = next(iter(dataloader))
test_photometry_data = torch.tensor(data_scaled_ph.T, dtype=torch.float32).unsqueeze(2).to(device)  # Reshape if necessary
test_spectra_data = torch.tensor(data_scaled_sp.T, dtype=torch.float32).to(device)

with torch.no_grad():
    t = torch.randint(0, time_steps, (test_spectra_data.size(0),)).to(device)
    generated_spectra = model(test_photometry_data.to(device), t)

df_wavelengths = pd.read_csv("../../data/interpolated_spectra.csv").T
df_wavelengths.reset_index(inplace=True)
df_wavelengths.columns = range(len(df_wavelengths.columns))
df_wavelengths = df_wavelengths.values.astype(np.float64)
df_wavelengths_scaled = df_wavelengths / np.max(df_wavelengths)
data_wavelengths_photo = df_wavelengths[:, 0] 
photometry_wavelengths = data_wavelengths_photo

df_wavelengths_spectra = pd.read_csv("../../data/spectra.csv").T
df_wavelengths_spectra.reset_index(inplace=True)
df_wavelengths_spectra.columns = range(len(df_wavelengths_spectra.columns))
df_wavelengths_spectra = df_wavelengths_spectra.values.astype(np.float64)
df_wavelengths_scaled_spectra = df_wavelengths_spectra / np.max(df_wavelengths_spectra)
data_wavelengths_spectra = df_wavelengths_spectra[:, 0]
spectra_wavelengths =  data_wavelengths_spectra 

plot_spectra(test_spectra_data, test_photometry_data, generated_spectra, photometry_wavelengths, spectra_wavelengths)