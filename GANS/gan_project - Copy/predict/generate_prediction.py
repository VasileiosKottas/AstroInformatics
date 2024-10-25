import torch

def predict_fluxes_for_wavelengths(timegan, wavelengths, scaler, device, latent_dim=32):
    """
    Predicts fluxes for a new galaxy using specific wavelengths as input.
    
    Parameters:
    - timegan: Trained TimeGAN model.
    - wavelengths: Tensor of specific wavelengths to use as input (shape: [sequence_length, input_dim]).
    - scaler: Fitted scaler used during training for normalization.
    - device: Device (CPU/GPU) on which the model is running.
    - latent_dim: Dimension of the noise vector used as input to the generator.
    
    Returns:
    - generated_fluxes: De-scaled predicted fluxes for the new galaxy.
    """
    
    # Ensure wavelengths are on the correct device
    wavelengths = wavelengths.to(device)
    
    # Generate a noise vector (batch_size=1 for a single prediction)
    noise = torch.randn(1, wavelengths.size(0), latent_dim).to(device)
    
    # Use the generator to produce predicted fluxes
    with torch.no_grad():  # No need to calculate gradients
        generated_fluxes = timegan.generator_net(noise, wavelengths)
    
    # De-scale the generated data
    generated_fluxes = generated_fluxes.detach().cpu().numpy().reshape(wavelengths.size(0), -1)
    generated_fluxes = scaler.inverse_transform(generated_fluxes)
    
    return generated_fluxes
