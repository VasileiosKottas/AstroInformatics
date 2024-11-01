import matplotlib.pyplot as plt

def plot_spectra(real_spectra, photometry, generated_spectra, photometry_wavelengths, spectra_wavelengths, index=0):
    real_spectra = real_spectra[index].detach().cpu().numpy()
    generated_spectra = generated_spectra[index].detach().cpu().numpy()
    photometry = photometry[index].detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(spectra_wavelengths, real_spectra, label='Real Spectra', color='blue', alpha=0.7)
    plt.plot(spectra_wavelengths, generated_spectra, label='Generated Spectra', color='orange', linestyle='--', alpha=0.7)
    plt.scatter(photometry_wavelengths, photometry, label='Photometry', color='red', s=50)
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Flux')
    plt.title('DIFFUSION: Real vs. Generated Spectra with Photometry Points')
    plt.legend()
    plt.grid(True)
    plt.show()