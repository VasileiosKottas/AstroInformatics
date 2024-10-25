from data.dataset import get_dataloader
from data.prepare_data import fit_scaler
from train.train_timegan import train_timegan
from train.hyperparameters import *
from visualize.tsne_visualization import visualize_tsne
from visualize.plot_real_vs_generated import plot_real_vs_generated
from models.timegan import TimeGAN
import numpy as np
import pandas as pd
import torch
# import sys
# sys.path.append('AstroInformatics/data')
def load_partial_state_dict(model, state_dict):
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            if model_state[name].size() == param.size():
                # If sizes match, load the parameter
                model_state[name].copy_(param)
            else:
                # If sizes don't match, initialize it instead of copying
                print(f"Skipping loading parameter: {name} due to size mismatch")
        else:
            print(f"Skipping loading parameter: {name} not found in model")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv('../../data/interpolated_spectra.csv')
    df = df.T
    df.reset_index(inplace=True)
    # Assuming 'df' is your DataFrame
    df.columns = range(len(df.columns))
    print(df.shape[1])
    total_rows = len(df)
    sequence_length = 17
    num_features = 11
    # Ensure that the DataFrame has the correct number of columns
    assert df.shape[1] == num_features, f"Expected {num_features} features, but got {df.shape[1]}"

    # Calculate the number of samples
    if total_rows % sequence_length == 0:
        num_samples = total_rows // sequence_length
    else:
        raise ValueError("The total number of rows is not divisible by sequence length. Please adjust the data.")

    data = df.values.astype(np.float64).reshape(num_samples, sequence_length, num_features)
    # Train the TimeGAN model
    # Fit the scaler and prepare the data loader
    scaler = fit_scaler(data)
    train_loader = get_dataloader(data, batch_size=batch_size)

    # Train the TimeGAN model
    train_timegan(train_loader, scaler, device, epochs=epochs)

    # Load the trained model weights
    timegan = TimeGAN(input_dim, hidden_dim, latent_dim, device)

    # Load the saved weights
    checkpoint = torch.load('timegan_weights.pth', map_location=device)
    load_partial_state_dict(timegan.embedding_net, checkpoint['embedding_net'])
    load_partial_state_dict(timegan.recovery_net, checkpoint['recovery_net'])
    load_partial_state_dict(timegan.generator_net, checkpoint['generator_net'])
    load_partial_state_dict(timegan.discriminator_net, checkpoint['discriminator_net'])


    # Generate synthetic data from the model
    noise = torch.randn(batch_size, sequence_length, latent_dim).to(device)
    generated_data_batch = timegan.generator_net(noise)

    # Get a batch of real data, ensuring the correct batch size
    real_data_batch = next(iter(train_loader))
    real_data_batch = real_data_batch[:batch_size].to(device)

    # Visualize t-SNE of Real vs Generated Data
    # visualize_tsne(real_data_batch, generated_data_batch, scaler, device, num_samples=batch_size)

    # Simple plot of real vs generated data
# Simple plot of real vs generated data
    plot_real_vs_generated(real_data_batch, generated_data_batch, scaler, num_samples=5)
