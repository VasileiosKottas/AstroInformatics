import numpy as np
import csv
from scipy import interpolate
from galaxy import Galaxy
import matplotlib.pyplot as plt
from tqdm import tqdm

np.seterr(all='ignore')  # Ignore over/underflow errors

def create_data(iterations, **kwargs):
    """Generates galaxy spectra and saves data for spectra and interpolated spectra.
    Only saves galaxies with positive flux values.

    Parameters:
        iterations (int): Total number of galaxies to generate.
        kwargs: Additional options including:
            - plot (bool): Whether to plot the spectra.
            - agntype (str): AGN type.
            - photometry (bool): Whether to compute photometry.
            - ULIRG (str): ULIRG type.

    Returns:
        None
    """
    ulirgs = kwargs.get('ULIRG')
    data = np.loadtxt(f'Filters/{ulirgs}_photometry.txt', skiprows=1)
    filters = data[:, 0]

    # Determine redshift based on ULIRG type
    redshift_dict = {
        'U5652': 1.618,
        'U16526': 1.749,
        'U5150': 1.898,
        'U5632': 2.016
    }
    redshift = redshift_dict.get(ulirgs, 0.1)
    zstr = str(round((redshift / 0.2)) * 0.2)
    metal = '0.0080000' if redshift < 0.29 else '0.00130000'
    filename = f'./spheroid_models_5/spheroid_array1_z={zstr[0:3]}0000_met={metal}_X.npz'
    data_sph = np.load(filename)
    models_fnu_full = data_sph['spheroid_models_fnu']

    galaxy = Galaxy()

    # Initialize lists for storing flux data
    wave_synth_list = []
    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []

    positive_output_wave_lists = []
    positive_output_flux_lists = []
    positive_wave_inter_lists = []
    positive_flux_inter_lists = []

    count = 0
    for _ in tqdm(range(iterations * 2), desc="Processing Galaxies"):
        if count >= iterations:
            break

        # Generate galaxy components
        wave_synth, f1 = galaxy.create_starburst()
        f2 = galaxy.create_agn(kwargs.get('agntype'))
        f3 = galaxy.create_spheroid(models_fnu_full)
        f4 = galaxy.create_polar_dust()
        fall = f1 + f2 + f3 + f4

        # Store the arrays
        wave_synth_list.append(wave_synth)
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

        # Check if all flux values are positive
        if np.all(fall > 0):
            fall_log = np.log10(fall) 
            output_wave_list = wave_synth[wave_synth < 1200]
            output_flux_list = fall_log[wave_synth < 1200]

            # Perform photometry if enabled
            if kwargs.get('photometry', False):
                observed_wave = filters / (1 + redshift)
                interpolation_function = interpolate.interp1d(wave_synth, fall_log, bounds_error=False, fill_value=np.nan)
                photometry = interpolation_function(observed_wave)
                # if np.all(photometry > 0):
                positive_wave_inter_lists.append(observed_wave)
                positive_flux_inter_lists.append(photometry)

            positive_output_wave_lists.append(output_wave_list)
            positive_output_flux_lists.append(output_flux_list)
            count += 1

    # Save spectra data
    with open('data/spectra.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(positive_output_wave_lists[0])  # Write wavelengths as headers
        writer.writerows(positive_output_flux_lists)  # Write flux values

    # Save interpolated spectra data
    with open('data/interpolated_spectra.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(positive_wave_inter_lists[0])  # Write interpolated wavelengths as headers
        writer.writerows(positive_flux_inter_lists)  # Write interpolated flux values

    # Add plots for f1, f2, f3, f4
    plot_components(wave_synth_list, f1_list, "Starburst", "starburst_flux.png")
    plot_components(wave_synth_list, f2_list, "AGN", "agn_flux.png")
    plot_components(wave_synth_list, f3_list, "Spheroid", "spheroid_flux.png")
    plot_components(wave_synth_list, f4_list, "Polar Dust", "polar_dust_flux.png")

def plot_components(wave_synth_list, flux_list, title, filename):
    """Plots 6 subplots for each component and saves as a PNG file.

    Parameters:
        wave_synth_list (list): List of wavelength arrays.
        flux_list (list): List of flux arrays.
        title (str): Title for the plot.
        filename (str): Filename to save the plot.

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 6 subplots (2x3 grid)
    fig.suptitle(f'{title} Flux for Wavelength (λ < 350)', fontsize=16)
    axes = axes.flatten()

    for i, (wave, flux, ax) in enumerate(zip(wave_synth_list[:6], flux_list[:6], axes)):
        # Filter wavelengths < 350
        valid_indices = (wave < 350) & (flux > 0)
        ax.plot(wave[valid_indices], flux[valid_indices])
        ax.set_title(f"Galaxy {i+1}", fontsize=10)
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')
        ax.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close(fig)


