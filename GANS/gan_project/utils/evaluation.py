# utils/evaluation.py
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(timegan, real_loader, scaler, device, latent_dim=32):
    """
    Evaluate the TimeGAN model using MSE and MAE.
    
    Parameters:
    - timegan: The trained TimeGAN model.
    - real_loader: DataLoader containing real data samples.
    - scaler: Fitted scaler for inverse normalization.
    - device: The device (CPU/GPU) on which the model is running.
    - latent_dim: Dimension of the noise vector used as input to the generator.
    
    Returns:
    - avg_mse: Average Mean Squared Error across all batches.
    - avg_mae: Average Mean Absolute Error across all batches.
    """
    mse_list = []
    mae_list = []

    timegan.generator_net.eval()  # Set model to evaluation mode

    for real_data in real_loader:
        real_data = real_data.to(device)
        real_wavelengths = real_data[:, :, 0:1]  # Extract real wavelengths
        real_fluxes = real_data[:, :, 1:]        # Extract real fluxes
        
        # Generate fluxes using the same wavelengths
        noise = torch.randn(real_data.size(0), real_data.size(1), latent_dim).to(device)
        generated_fluxes = timegan.generator_net(noise, real_wavelengths)
        
        # De-scale the generated and real fluxes
        real_fluxes = scaler.inverse_transform(real_fluxes.detach().cpu().numpy().reshape(-1, real_fluxes.shape[-1]))
        generated_fluxes = scaler.inverse_transform(generated_fluxes.detach().cpu().numpy().reshape(-1, generated_fluxes.shape[-1]))
        
        # Calculate evaluation metrics
        mse = mean_squared_error(real_fluxes, generated_fluxes)
        mae = mean_absolute_error(real_fluxes, generated_fluxes)

        mse_list.append(mse)
        mae_list.append(mae)
    
    avg_mse = sum(mse_list) / len(mse_list)
    avg_mae = sum(mae_list) / len(mae_list)

    print(f"Evaluation - Average MSE: {avg_mse:.4f}, Average MAE: {avg_mae:.4f}")
    return avg_mse, avg_mae
