import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from spectra_create_dataset import *
from model import TransformerEncoderModel
from utils import DiffusionProcess
from params import (input_dim, n_heads, 
                    n_layers, 
                    output_dim,
                    learning_rate,
                    num_steps,
                    beta_start,
                    beta_end,
                    )


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df, sorted_wavelengths = use_data(data = 'photometry')
df_spectra = pd.read_csv('../data/spectra.csv')
sorted_wavelengths_spectra = df_spectra.columns.values.astype(float)
df_spectra = df_spectra.T.to_numpy()
# Initialize your data and model
data = torch.from_numpy(df).float().to(device)  # Move data to GPU
df_spectra_tensor = torch.from_numpy(df_spectra).float().to(device)

# Initialize model, optimizer, and diffusion process
model = TransformerEncoderModel(input_dim, n_heads, n_layers, output_dim).to(device)  # Move model to GPU
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
diffusion = DiffusionProcess(num_steps, beta_start, beta_end)

# Store predictions for visualization
predictions = []
noise = []
# Example training loop
epochs = 1000  # Number of epochs
for epoch in range(epochs):
    for step in range(diffusion.steps):
        noisy_data = diffusion.add_noise(data, step)
        # print(model.shape)
        optimizer.zero_grad()
        denoised_data = diffusion.denoise(model, noisy_data, step)

        # Define loss for regression (on GPU)
        loss = nn.MSELoss()(denoised_data, data)
        loss.backward()
        optimizer.step()
        noise.append(noisy_data)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Store denoised data after each epoch for visualization
    with torch.no_grad():
        final_denoised = diffusion.denoise(model, noisy_data, diffusion.steps - 1)  # Get final step prediction
        predictions.append(final_denoised.cpu())  # Move predictions to CPU for visualization

torch.save(model.state_dict(), 'model_weights.pth')
# Convert data to numpy for plotting
real_data_np = data.cpu().detach().numpy()  # Move data to CPU for plotting
predicted_data_np = [pred.cpu().detach().numpy() for pred in predictions]  # Convert to CPU and numpy for plotting
noise_data_np = [noisy.cpu().detach().numpy() for noisy in noise] 
spectra = df_spectra_tensor.cpu().detach().numpy()
# Visualization
for i in range(epochs):
    if i % 1001 == 0:
        plt.figure(figsize=(10, 6))

        # Plot original data (first feature)
        plt.scatter(sorted_wavelengths, real_data_np[:, 0], label='Real Data', color='blue')

        # Plot denoised data (first feature)
        plt.plot(sorted_wavelengths, predicted_data_np[i][:, 0], label=f'Denoised Data after Epoch {i+1}', linestyle='--', color='red')
        
        # Plot the Noise
        plt.plot(sorted_wavelengths_spectra, spectra[:, 0], label=f'Noise Data after Epoch', linestyle='-', color='green')
        
        plt.xlabel('Data Index')
        plt.ylabel('Feature Value')
        plt.title(f'Real vs Denoised Data  - Epoch {i+1}')
        plt.legend()
        plt.show()
        
with torch.no_grad():
        final_denoised = diffusion.denoise(model, noisy_data, diffusion.steps - 1)  # Get final step prediction
        predictions.append(final_denoised.cpu())  # Move predictions to CPU for visualization

# Convert data to numpy for plotting
real_data_np = data.cpu().detach().numpy()  # Move data to CPU for plotting
predicted_data_np = [pred.cpu().detach().numpy() for pred in predictions]  # Convert to CPU and numpy for plotting

# Visualization
# for i in range(epochs):
plt.figure(figsize=(10, 6))

# Plot original data (first feature)
plt.scatter(sorted_wavelengths, real_data_np[:, 2], label='Real Data (Feature 1)', color='blue')

# Plot denoised data (first feature)
plt.plot(sorted_wavelengths, predicted_data_np[1][:, 2], label=f'Denoised Data after Epoch {i+1}', linestyle='--', color='red')
plt.plot(sorted_wavelengths_spectra, spectra[:, 2], label=f'Noise Data after Epoch', linestyle='-', color='green')
plt.xlabel('Data Index')
plt.ylabel('Feature Value')
plt.title(f'Real vs Denoised Data (Feature 1) - Epoch {i+1}')
plt.legend()
plt.show()
