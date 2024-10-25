from model import TSDiff
from train import train_model, generate_fluxes, print_evaluation
from visualize import plot_real_vs_generated, plot_generated_samples
import torch
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler  # or MinMaxScaler
import joblib
import matplotlib.pyplot as plt
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your dataset, assumed shape [17, 11] (1 wavelength, 10 fluxes)
    df = pd.read_csv('../../data/interpolated_spectra.csv').T
    
    df.reset_index(inplace=True)
    df.columns = range(len(df.columns))
    data = df.values.astype(np.float64)
    # data = data / data.values.max()
    
    data_scaled = data / np.max(data)
    # Separate wavelengths and fluxes
    wavelengths = data_scaled[:, 0]  # Shape: [17,]
    fluxes = data_scaled[:, 1:]  # Shape: [10, 17] - Transpose to match [num_fluxes, num_wavelengths]
    # Step 1: Scale the flux data
    # Step 1: Scale the flux data
    # scaler = StandardScaler()  # or MinMaxScaler(feature_range=(0, 1))
    # scaled_fluxes = scaler.fit_transform(fluxes)
    # scaled_fluxes = fluxes / 
    # Save the scaler for future inverse transformation
    # joblib.dump(scaler, 'scaler.pkl')
    # Step 2: Prepare the input tensor for training
    input_tensor = torch.tensor(fluxes, dtype=torch.float32).unsqueeze(0).to(device)   # Shape: [1, 16, 10]
    plt.plot(wavelengths,input_tensor[0,:,1].to("cpu"))
    plt.title("real")
    print(input_tensor.shape)
    # Set model parameters
    model_params = {
        "input_dim": input_tensor.shape[1],
        "time_steps": 1000,
        "learning_rate": 0.001,
    }

    # Initialize and train the model
    from model import SelfGuidedTSDiff  # Assuming your model class is here
    model = SelfGuidedTSDiff(input_dim=model_params["input_dim"]).to(device)
    train_model(model, input_tensor, model_params)

    # # Generate new flux data for a single galaxy
    # # new_wavelength = torch.tensor(np.random.rand(1, 17, 1), dtype=torch.float32).to(device)  # Replace with actual new wavelengths
    # new_wavelength = torch.tensor(wavelengths, dtype=torch.float32).unsqueeze(0).to(device)   # Shape: [1, 16, 10]
    
    # print(new_wavelength.shape)
    # generated_fluxes = generate_fluxes(model, new_wavelength.unsqueeze(2))
    # print(generated_fluxes.shape)
    # # Step 3: Inverse transform the generated data to original scale
    # generated_fluxes_np = generated_fluxes.squeeze(0).cpu().numpy().T  # Shape should now be (1, 16)
    # print(generated_fluxes_np.shape)
    # # # scaler = joblib.load('scaler.pkl')
    # # # original_scale_fluxes = scaler.inverse_transform(generated_fluxes_np)  # Correct shape (1, 16)

    # # # Print evaluation metrics (compares a single real flux against generated)
    # # print_evaluation(input_tensor.to("cpu"), generated_fluxes_np)
    # # # Plot results for the single galaxy
    # plot_generated_samples(wavelengths, generated_fluxes_np)
    # plot_real_vs_generated(wavelengths, fluxes, generated_fluxes_np)

if __name__ == "__main__":
    main()