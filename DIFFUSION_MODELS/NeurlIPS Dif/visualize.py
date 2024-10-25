import matplotlib.pyplot as plt
import numpy as np

# import matplotlib.pyplot as plt

def plot_real_vs_generated(wavelengths, real_fluxes, generated_fluxes):
    """
    Plot real fluxes vs. generated fluxes for comparison.
    """
    plt.figure(figsize=(10, 6))
    num_samples = min(real_fluxes.shape[1], generated_fluxes.shape[1], 5)  # Plot up to 5 samples to avoid clutter
    print(generated_fluxes[:, 1:])
    for i in range(1):
        plt.plot(wavelengths, real_fluxes[:, i], label=f'Real Flux {i + 1}', linestyle='-', marker='o')
        plt.plot(wavelengths, generated_fluxes[i, :], label=f'Generated Flux {i + 1}', linestyle='--', marker='x')

    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title("Real vs Generated Flux Samples")
    plt.legend()
    plt.savefig('realvsgenerated.png')
    plt.show()
def plot_generated_samples(wavelengths, generated_fluxes):
    """
    Plot only the generated fluxes to see their behavior.
    """
    plt.figure(figsize=(10, 6))
    num_samples = min(generated_fluxes.shape[0], 5)  # Limit to 5 samples

    for i in range(1):
        plt.plot(wavelengths, generated_fluxes[i, :], linestyle='-', marker='x', label=f'Generated Sample {i + 1}')

    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title("Generated Flux Samples")
    plt.legend()
    plt.savefig('generated_samples.png')
    plt.show()