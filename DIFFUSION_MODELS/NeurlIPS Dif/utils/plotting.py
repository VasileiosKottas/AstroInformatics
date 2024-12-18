import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_spectra(real_spectra, photometry, generated_spectra, photometry_wavelengths, spectra_wavelengths, index=0):
    # Extract data for the given index
    real_spectra = real_spectra[index].detach().cpu().numpy()
    generated_spectra = generated_spectra[index].mean(axis=0).detach().cpu().numpy()  # Average across channels
    photometry = photometry[index].detach().cpu().numpy()

    # Ensure photometry is 1D and matches wavelengths
    photometry = photometry.squeeze(-1)  # Remove the extra dimension
    photometry_wavelengths = np.array(photometry_wavelengths[:len(photometry)], dtype=np.float32)
    spectra_wavelengths = np.array(spectra_wavelengths[:real_spectra.shape[0]], dtype=np.float32)

    # Debugging to confirm shapes
    print(f"Real spectra shape: {real_spectra.shape}")
    print(f"Generated spectra shape: {generated_spectra.shape}")
    print(f"Photometry shape: {photometry.shape}")
    print(f"Photometry wavelengths shape: {photometry_wavelengths.shape}")
    print(f"Spectra wavelengths shape: {spectra_wavelengths.shape}")

    # Plot spectrometry (real and generated)
    plt.figure(figsize=(10, 6))
    plt.plot(spectra_wavelengths, real_spectra, label='Real Spectra', color='blue', alpha=0.7)
    plt.plot(spectra_wavelengths, generated_spectra, label='Generated Spectra', color='orange', linestyle='--', alpha=0.7)

    # Plot sparse photometry points
    plt.scatter(photometry_wavelengths, photometry, label='Photometry', color='red', s=50)

    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Flux')
    plt.title('Real vs. Generated Spectra with Photometry Points')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(f"generated_spectra_index_{index}.png")
    plt.close()

