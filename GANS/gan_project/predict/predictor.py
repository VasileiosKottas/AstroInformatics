# predict/predictor.py
import torch
from models.timegan import TimeGAN
from utils.scaler import inverse_normalize_data

def generate_single_galaxy_flux(timegan, wavelengths, scaler, device, latent_dim=32):
    """
    Generate flux predictions for a single galaxy using the trained TimeGAN model.
    
    Parameters:
    - timegan: Trained TimeGAN model.
    - wavelengths: Tensor of wavelengths for which to generate predictions (shape: [sequence_length, 1]).
    - scaler: Fitted scaler used during training for normalization.
    - device: Device (CPU/GPU) on which the model is running.
    - latent_dim: Dimension of the noise vector used as input to the generator.
    
    Returns:
    - generated_fluxes: De-scaled predicted fluxes for a single galaxy.
    """
    # Ensure wavelengths are on the correct device
    wavelengths = wavelengths.to(device).unsqueeze(0)  # Add batch dimension (1, seq_len, 1)
    
    # Generate a noise vector (batch_size=1 for a single prediction)
    noise = torch.randn(1, wavelengths.size(1), latent_dim).to(device)
    
    # Use the generator to produce predicted fluxes
    with torch.no_grad():
        generated_fluxes = timegan.generator_net(noise, wavelengths)
    
    # De-scale the generated data
    generated_fluxes = generated_fluxes.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
    generated_fluxes = scaler.inverse_transform(generated_fluxes)  # Inverse scaling
    
    return generated_fluxes

def predict_fluxes_for_wavelengths(timegan, wavelengths, scaler, device, latent_dim=32):
    """
    Predict new fluxes using the trained TimeGAN model.
    
    Parameters:
    - timegan: Trained TimeGAN model.
    - wavelengths: Tensor of wavelengths for which to generate predictions.
    - scaler: Fitted scaler used during training for normalization.
    - device: Device (CPU/GPU) on which the model is running.
    - latent_dim: Dimension of the noise vector used as input to the generator.
    
    Returns:
    - generated_fluxes: De-scaled predicted fluxes.
    """
    wavelengths = wavelengths.to(device)
    noise = torch.randn(1, wavelengths.size(0), latent_dim).to(device)
    with torch.no_grad():
        generated_fluxes = timegan.generator_net(noise, wavelengths)
    generated_fluxes = generated_fluxes.detach().cpu().numpy().reshape(wavelengths.size(0), -1)
    generated_fluxes = scaler.inverse_transform(generated_fluxes)
    return generated_fluxes
