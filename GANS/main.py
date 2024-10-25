import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from hyperparameters import *
from data import data
from models import *
from training_func import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

real_batch, random_batch, wavelength, scaled_data = data()
print(next(iter(real_batch)).shape)
# Define the models with the correct input sizes
embedder = RNNModule(n_layers=4, input_size=N_SEQ, hidden_units=HIDDEN_DIM, output_units=HIDDEN_DIM).to(device)  # Input size = n_seq for the first layer
recovery = RNNModule(n_layers=4, input_size=HIDDEN_DIM, hidden_units=HIDDEN_DIM, output_units=N_SEQ).to(device)  # Input size = hidden_dim to match embedder output
generator = RNNModule(n_layers=4, input_size=N_SEQ, hidden_units=HIDDEN_DIM, output_units=HIDDEN_DIM).to(device)
discriminator = RNNModule(n_layers=4, input_size=N_SEQ, hidden_units=HIDDEN_DIM, output_units=1, output_activation=torch.sigmoid).to(device)
supervisor = RNNModule(n_layers=3, input_size=HIDDEN_DIM, hidden_units=HIDDEN_DIM, output_units=HIDDEN_DIM).to(device)

train_steps = 10000
gamma = 1

mse = nn.MSELoss()
bce = nn.BCELoss()

# Train Autoencoder
autoencoder = Autoencoder(embedder, recovery)
autoencoder.to(device)
autoencoder_optimizer = Adam(autoencoder.parameters(), lr=1e-3)

for step in tqdm(range(train_steps)):


    for i in range(1):
        # Get a batch of data
        X_ = next(iter(real_batch)).to(device)



        # Convert X_ to a PyTorch tensor if it’s not already
        if not isinstance(X_, torch.Tensor):
            X_ = torch.tensor(X_, dtype=torch.float32).unsqeeze(1)
            print(X_.shape)

        # Train the autoencoder and get the loss
        step_e_loss_t0 = train_autoencoder_step(X_, autoencoder, autoencoder_optimizer)

# Supervised training
supervisor_optimizer = Adam(supervisor.parameters(), lr=1e-3)


for step in tqdm(range(train_steps)):

    for i in range(1):
        X_ = next(iter(real_batch)).to(device)

        step_g_loss_s = train_supervisor(X_, embedder, supervisor, supervisor_optimizer)  # Train the supervisor and get the loss


# Join Training

adversarial_supervised = AdversarialNetSupervised(generator, supervisor, discriminator).to(device)
adversarial_emb = AdversarialNet(generator, discriminator).to(device)
synthetic_data = SyntheticData(generator, supervisor, recovery).to(device)
discriminator_model = DiscriminatorReal(discriminator).to(device)

# Optimizer's
generator_optimizer = Adam(generator.parameters(), lr=1.0e-3)
discriminator_optimizer = Adam(discriminator.parameters(), lr=1.0e-3)
embedding_optimizer = Adam(embedder.parameters(), lr=1.0e-3)

# Train
# Initialize losses
step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0

real_data_list = []
generated_data_list = []
losses = {
    'step_d_loss': [],
    'step_g_loss_u': [],
    'step_g_loss_s': [],
    'step_g_loss_v': [],
    'step_e_loss_t0': []
}

for step in tqdm(range(train_steps)):

    # Train generator (twice as often as discriminator)
    for kk in range(2):
        X_ = next(iter(real_batch)).to(device)

        Z_ = next(iter(random_batch)).to(device)

        # Train generator
        step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(X_, Z_, generator, supervisor, embedder, synthetic_data, adversarial_supervised, adversarial_emb, generator_optimizer)
        # Train embedder
        step_e_loss_t0 = train_embedder(X_, embedder, supervisor, autoencoder, recovery, embedding_optimizer)

    X_ = next(iter(real_batch)).to(device)

    Z_ = next(iter(random_batch)).to(device)

    step_d_loss = get_discriminator_loss(X_, Z_, gamma, discriminator_model, adversarial_supervised, adversarial_emb)

    if step_d_loss > 0.15:
        step_d_loss = train_discriminator(X_, Z_, gamma, discriminator_model,
                        discriminator_optimizer, adversarial_supervised, adversarial_emb)

    if step % 1000 == 0:
        print(f'{step:6,.0f} | d_loss: {step_d_loss.item():6.4f} | g_loss_u: {step_g_loss_u.item():6.4f} | '
                f'g_loss_s: {step_g_loss_s.item():6.4f} | g_loss_v: {step_g_loss_v.item():6.4f} | e_loss_t0: {step_e_loss_t0.item():6.4f}')
    losses['step_d_loss'].append(step_d_loss.item())
    losses['step_g_loss_u'].append(step_g_loss_u.item())
    losses['step_g_loss_s'].append(step_g_loss_s.item())
    losses['step_g_loss_v'].append(step_g_loss_v.item())
    losses['step_e_loss_t0'].append(step_e_loss_t0.item())

    real_data_list.append(X_.cpu().detach().numpy())
    generated_data_list.append(synthetic_data(Z_).cpu().detach().numpy())

torch.save(synthetic_data, 'time_gan/TimeGAN.pth')

# Plot the losses
plt.figure(figsize=(10, 6))

# Plot each type of loss
plt.plot(losses['step_d_loss'], label='Discriminator Loss')
plt.plot(losses['step_g_loss_u'], label='Generator Loss U')
plt.plot(losses['step_g_loss_s'], label='Generator Loss S')
plt.plot(losses['step_g_loss_v'], label='Generator Loss V')
plt.plot(losses['step_e_loss_t0'], label='Embedder Loss T0')

plt.title('Training Losses Over Steps')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()

# Show the plot
plt.show()

generated_data = []

# Loop over the number of batches
for i in range(int(1)):

    Z_ = next(iter(random_batch))  # Assuming random_series is an iterator
    Z_ = torch.tensor(Z_, dtype=torch.float32).to(device)

    # Generate data using the synthetic data model
    with torch.no_grad():  # Disable gradient computation for inference
        d = synthetic_data(Z_)
        print(d.shape)
    # Store the generated data
    generated_data.append(d.cpu().detach().numpy())  # Convert tensor to numpy array


generated_data = np.concatenate(generated_data, axis=0)

plt.figure(figsize=(12, 6))

# Plot real data
# plt.plot(wavelength[:-1], scaled_data[0,:,1])


# Plot generated data
plt.scatter(wavelength[:-1], generated_data[:,:,:],c = 'r')  # Adjust based on your data shape
plt.savefig("generatedvsreal.png")
plt.show()

