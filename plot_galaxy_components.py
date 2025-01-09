import os
import numpy as np
import matplotlib.pyplot as plt
from star_data_starb_agn_sph_dust import create_data, Galaxy

# Ensure Matplotlib does not crash on headless systems
import matplotlib
matplotlib.use('Agg')

# Create a folder for plots
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

# Helper function to save plots
def save_plot(filename):
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot components

def plot_starburst():
    galaxy = Galaxy()
    wave_synth, f1 = galaxy.create_starburst()
    plt.figure()
    plt.plot(wave_synth, np.log10(f1), label="Starburst")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Log Flux (Fv)")
    plt.title("Starburst Component")
    save_plot("starburst_component.png")
    return wave_synth

def plot_agn(wave_synth):
    galaxy = Galaxy()
    agn_types = ['cygnus']
    for agn_type in agn_types:
        flux_agn = galaxy.create_agn(agn_type)
        plt.figure()
        plt.plot(wave_synth, np.log10(flux_agn), label=agn_type)
        plt.xlabel("Wavelength (μm)")
        plt.ylabel("Log Flux (Fv)")
        plt.title(f"AGN Component: {agn_type.capitalize()}")
        save_plot(f"agn_{agn_type}_component.png")

def plot_spheroid(wave_synth):
    # Load spheroid models
    filename = './spheroid_models_5/spheroid_array1_z=0.100_met=0.00130000_X.npz'
    data_sph = np.load(filename)
    models_fnu_full = data_sph['spheroid_models_fnu']

    galaxy = Galaxy()
    flux_spheroid = galaxy.create_spheroid(models_fnu_full)
    plt.figure()
    plt.plot(wave_synth, np.log10(flux_spheroid), label="Spheroid")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Log Flux (Fv)")
    plt.title("Spheroid Component")
    save_plot("spheroid_component.png")

def plot_polar_dust(wave_synth):
    galaxy = Galaxy()
    flux_dust = galaxy.create_polar_dust()
    plt.figure()
    plt.plot(wave_synth, np.log10(flux_dust), label="Polar Dust")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Log Flux (Fv)")
    plt.title("Polar Dust Component")
    save_plot("polar_dust_component.png")

def plot_combined_components(wave_synth):
    # Load spheroid models
    filename = './spheroid_models_5/spheroid_array1_z=0.100_met=0.00130000_X.npz'
    data_sph = np.load(filename)
    models_fnu_full = data_sph['spheroid_models_fnu']

    galaxy = Galaxy()
    wave_synth, f1 = galaxy.create_starburst()
    f2 = galaxy.create_agn('cygnus')
    f3 = galaxy.create_spheroid(models_fnu_full)
    f4 = galaxy.create_polar_dust()
    fall = f1 + f2 + f3 + f4

    plt.figure()
    plt.plot(wave_synth, np.log10(fall), label="Combined Components")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Log Flux (Fv)")
    plt.title("Combined Galaxy Spectrum")
    save_plot("combined_components.png")

if __name__ == "__main__":
    # Generate starburst wavelengths and plot each component
    wave_synth = plot_starburst()
    plot_agn(wave_synth)
    plot_spheroid(wave_synth)
    plot_polar_dust(wave_synth)
    plot_combined_components(wave_synth)

    print(f"Plots have been generated and saved to the '{output_folder}' directory.")
