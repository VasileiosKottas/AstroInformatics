import numpy as np

def compute_chi_squared(real_spectra, generated_spectra):
    # Move tensors to CPU and convert to NumPy arrays
    real_spectra = real_spectra.cpu().numpy()
    generated_spectra = generated_spectra.cpu().numpy()

    # Ensure generated_spectra is non-negative
    generated_spectra = np.abs(generated_spectra) + 1e-8  # Avoid division by zero or negatives

    # Compute chi-squared
    chi_squared = np.sum((real_spectra[0] - generated_spectra[0]) ** 2 / generated_spectra[0])
    return chi_squared
