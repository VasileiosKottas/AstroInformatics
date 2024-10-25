# main.py
import torch
import pandas as pd
import numpy as np
from datasets.loader import get_dataloader
from utils.scaler import fit_scaler
from train.train import train_timegan
from predict.predictor import generate_single_galaxy_flux
from utils.plotting import plot_real_vs_generated
from utils.evaluation import evaluate_model
from config.config import DEVICE, EPOCHS

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('../../data/interpolated_spectra.csv').T
    df.reset_index(inplace=True)
    df.columns = range(len(df.columns))
    total_rows = len(df)
    sequence_length = 17
    num_features = 11

    if total_rows % sequence_length == 0:
        num_samples = total_rows // sequence_length
    else:
        raise ValueError("The total number of rows is not divisible by sequence length. Please adjust the data.")

    data = df.values.astype(np.float64).reshape(num_samples, sequence_length, num_features)
    scaler = fit_scaler(data)
    train_loader = get_dataloader(data, batch_size=32)

    # Train TimeGAN
    train_timegan(train_loader, scaler, device, epochs=EPOCHS)

    # Load the trained model weights
    timegan = TimeGAN(1, 64, 32, device)
    checkpoint = torch.load('timegan_weights.pth', map_location=device)
    timegan.generator_net.load_state_dict(checkpoint['generator_net'])

    # Evaluate the model
    avg_mse, avg_mae = evaluate_model(timegan, train_loader, scaler, device)
    print(f"Average MSE: {avg_mse}, Average MAE: {avg_mae}")

    # Predict example fluxes for a single galaxy
    example_wavelengths = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0],
                                        [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7]])
    generated_fluxes = generate_single_galaxy_flux(timegan, example_wavelengths, scaler, device)
    print("Generated Fluxes for Single Galaxy:", generated_fluxes)

    # Get a batch of real data from the DataLoader for comparison
    real_data_batch = next(iter(train_loader))
    real_data_batch = real_data_batch[:32].to(device)

    # Generate data using the model
    noise = torch.randn(32, sequence_length, 32).to(device)
    wavelengths = real_data_batch[0, :, 0:1].unsqueeze(0).repeat(32, 1, 1)
    generated_data_batch = timegan.generator_net(noise, wavelengths)

    # Visualize the results
    plot_real_vs_generated(real_data_batch[:, :, 0:], generated_data_batch, scaler, num_samples=5)
