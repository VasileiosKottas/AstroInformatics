import matplotlib.pyplot as plt
import numpy as np
def plot_spectra(real_spectra, photometry, generated_spectra, photometry_wavelengths, spectra_wavelengths, index=0):
    # Extract data for the given index
    real_spectra = real_spectra[index].detach().cpu().numpy()
    generated_spectra = generated_spectra[index].detach().cpu().numpy()
    photometry = photometry[index].detach().cpu().numpy()

    # Ensure wavelengths are numerical and sorted
    photometry_wavelengths = np.array(photometry_wavelengths, dtype=np.float32)
    spectra_wavelengths = np.array(spectra_wavelengths, dtype=np.float32)

    # Plot spectrometry (real and generated)
    plt.figure(figsize=(10, 6))
    plt.plot(spectra_wavelengths, real_spectra, label='Real Spectra', color='blue', alpha=0.7)
    plt.plot(spectra_wavelengths, generated_spectra, label='Generated Spectra', color='orange', linestyle='--', alpha=0.7)

    # Plot sparse photometry points
    plt.scatter(photometry_wavelengths, photometry, label='Photometry', color='red', s=50)

    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Flux')
    plt.title('Real vs. Generated Spectra with Photometry Points')

    # Rotate x-axis ticks and improve readability
    plt.xticks(rotation=0)
    plt.tick_params(axis='x', which='major', labelsize=8)

    plt.legend()
    plt.grid(True)
    plt.show()

