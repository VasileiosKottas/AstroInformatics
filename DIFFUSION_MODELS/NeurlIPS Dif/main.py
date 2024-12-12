import torch
from models.tsdiff import SelfGuidedTSDiff
from utils.data_loader import SpectraDataset
from utils.plotting import plot_spectra
from utils.chi_squared import compute_chi_squared
from training.train import Trainer
from hyperparameters import batch_size, num_epochs, input_dim, time_steps, learning_rate
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

import torch
from models.tsdiff import SelfGuidedTSDiff
from utils.data_loader import SpectraDataset
from utils.plotting import plot_spectra
from utils.chi_squared import compute_chi_squared
from training.train import Trainer
from hyperparameters import batch_size, num_epochs, input_dim, time_steps, learning_rate
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
spectra_file = 'C:/Users/vasil/Desktop/AstroInformatics/data/spectra.csv'
photometry_file = 'C:/Users/vasil/Desktop/AstroInformatics/data/interpolated_spectra.csv'

df_spectra = pd.read_csv(spectra_file).T
data_sp = df_spectra.values.astype(np.float64)
data_scaled_sp = data_sp / np.max(data_sp)

df_photo = pd.read_csv(photometry_file).T
data_ph = df_photo.values.astype(np.float64)
data_scaled_ph = data_ph / np.max(data_sp)

# Initialize dataset
dataset = SpectraDataset(data_scaled_ph.T, data_scaled_sp.T)

# Train-test split (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Initialize model, optimizer, and criterion
model = SelfGuidedTSDiff(input_dim=input_dim, time_steps=time_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Initialize trainer and train the model
trainer = Trainer(model, train_dataloader, device, criterion, optimizer)
trainer.train(num_epochs)
trainer.save_model('trained_model.pth')

# Evaluate the model on the test set
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

real_spectra, photometry_data = next(iter(test_dataloader))
test_photometry_data = photometry_data.unsqueeze(2).to(device)
test_spectra_data = real_spectra.to(device)

with torch.no_grad():
    t = torch.randint(0, time_steps, (test_spectra_data.size(0),)).to(device)
    generated_spectra = model(test_photometry_data, t)

# Plot results
chi_squared = compute_chi_squared(test_spectra_data, generated_spectra)
print(f"Chi-squared: {chi_squared}")

df_wavelengths = pd.read_csv(photometry_file).T
df_wavelengths.reset_index(inplace=True)
df_wavelengths.columns = range(len(df_wavelengths.columns))
data_wavelengths_photo = df_wavelengths.values[:, 0]

df_wavelengths_spectra = pd.read_csv(spectra_file).T
df_wavelengths_spectra.reset_index(inplace=True)
df_wavelengths_spectra.columns = range(len(df_wavelengths_spectra.columns))
data_wavelengths_spectra = df_wavelengths_spectra.values[:, 0]
# Ensure photometry and spectra wavelengths are aligned
# print("Photometry Wavelengths:", data_wavelengths_photo)
# print("Spectra Wavelengths:", data_wavelengths_spectra)

plot_spectra(
    test_spectra_data,
    test_photometry_data,
    generated_spectra,
    data_wavelengths_photo,
    data_wavelengths_spectra,
)
