import torch 

def calculate_chi2(real_spectra, generated_spectra):
    """
    Calculates the chi-squared metric for a batch of spectra.
    Args:
        real_spectra (torch.Tensor): Ground truth spectra, shape [batch_size, num_wavelengths].
        generated_spectra (torch.Tensor): Predicted spectra, shape [batch_size, num_wavelengths].
    Returns:
        torch.Tensor: chi-squared values for the batch.
    """
    uncertainty = 0.1  # Assumed uncertainty as 10% of values
    diff = real_spectra - generated_spectra
    denom = (uncertainty * real_spectra) ** 2 + (uncertainty * generated_spectra) ** 2
    chi2 = torch.sum((diff ** 2) / denom, dim=1) / real_spectra.size(1)
    return chi2