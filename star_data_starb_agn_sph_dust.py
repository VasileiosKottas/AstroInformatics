import numpy as np
import sys
from scipy import interpolate
sys.path.append('./SMART_code')
from likelihood import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from galaxy import *
sys.path.append('./SMART_code')

np.seterr(all='ignore') #ignore over/underflow 0 errors

def create_data(iterations,**kwargs):
    '''This function outputs the flux and wavelengths from the starburts, AGN, spheroidal, polar dust. The input is the number of spectra to create. 
    As kwargs give, plot:True or False, AGNType, phtometry:True or False, redshift'''

# Lists to store all galaxy data
    all_output_wave_lists = []
    all_output_flux_lists = []
    all_starburst_flux_lists = []
    all_agn_flux_lists = []
    all_spheroid_flux_lists = []
    all_polar_dust_flux_lists = []
    all_wave_inter_lists = []
    all_flux_inter_lists = []
    
    # Shperoid
    # Choose the model with mettalicity 0.008
    ulirgs = kwargs.get('ULIRG')
    data = np.loadtxt(f'Filters/{ulirgs}_photometry.txt', skiprows=1)

    # Extract the first column (micron values)
    filters = data[:, 0]
    if ulirgs == str('U5652'):
        redshift = 1.6180000305175781
    elif ulirgs == str('U16526'):
        redshift = 1.7489999532699585
    elif ulirgs == str('U5150'):
        redshift = 1.8980000019073486
    elif ulirgs == str('U5632'):
        redshift = 2.0160000324249268
        
    zstr=str(round((redshift/0.2))*0.2)
    metal='0.00130000'

    if redshift < 0.29:
        metal='0.0080000'
        zstr='0.1000'
    filename = './spheroid_models_5/spheroid_array1_z='+zstr[0:3]+'0000_met='+metal+'_X.npz'  
    data_sph = np.load(filename)

    # Get the models from the spheroid_models file
    models_fnu_full = data_sph['spheroid_models_fnu']

    total_iterations = iterations # Choose random values within the specified ranges

    agn_kw = str(kwargs.get('agntype'))
    
    
    galaxy = Galaxy()
    for iteration in tqdm(range(int(total_iterations)), desc="Processing"):  # Add progress bar
        # Add Starburst emission
        wave_synth, f1 = galaxy.create_starburst()
        # Add AGN
        f2 = galaxy.create_agn(agn_kw)
        # Add the Spheroid
        f3 = galaxy.create_spheroid(models_fnu_full)
        # Add the Polar Dust
        f4 = galaxy.create_polar_dust()

        # Combine all components
        fall = f1 + f2 + f3 + f4

        # Temporary lists to store data for the current iteration
        output_wave_list = []
        output_flux_list = []
        starburst_flux_list = []
        agn_flux_list = []
        spheroid_flux_list = []
        polar_dust_flux_list = []
        wave_inter_list = []
        flux_inter_list = []

        # Loop through and collect data for the current iteration
        for wave, flux_ass, starburst_flux, agn_flux, spheroid_flux, polar_flux in zip(
            wave_synth, np.log10(fall), np.log10(f1), np.log10(f2), np.log10(f3), np.log10(f4)
        ):
            if wave < 1200:
                output_wave_list.append(np.log10(wave) if kwargs.get("log_wave", False) else wave)
                output_flux_list.append(flux_ass)
                starburst_flux_list.append(starburst_flux)
                agn_flux_list.append(agn_flux)
                spheroid_flux_list.append(spheroid_flux)
                polar_dust_flux_list.append(polar_flux)

        # If photometry is enabled, process interpolated values
        if kwargs.get('photometry', False):
            observed_wave = filters / (1 + redshift)
            wave_interpol = np.log10(observed_wave) if kwargs.get("log_wave", False) else observed_wave
            interpolation_function = interpolate.interp1d(
                np.log10(wave_synth), np.log10(fall)
            ) if kwargs.get("log_wave", False) else interpolate.interp1d(wave_synth, np.log10(fall))
            photometry = interpolation_function(wave_interpol)

            wave_inter_list.extend(wave_interpol)
            flux_inter_list.extend(photometry)

        # Store the collected data for this iteration
        all_output_wave_lists.append(output_wave_list)
        all_output_flux_lists.append(output_flux_list)
        all_starburst_flux_lists.append(starburst_flux_list)
        all_agn_flux_lists.append(agn_flux_list)
        all_spheroid_flux_lists.append(spheroid_flux_list)
        all_polar_dust_flux_lists.append(polar_dust_flux_list)
        all_wave_inter_lists.append(wave_inter_list)
        all_flux_inter_lists.append(flux_inter_list)

    # Determine the maximum length of the lists to pad shorter columns with NaN
    max_len = max(
        len(max(all_output_wave_lists, key=len)),
        len(max(all_output_flux_lists, key=len)),
        len(max(all_starburst_flux_lists, key=len)),
        len(max(all_agn_flux_lists, key=len)),
        len(max(all_spheroid_flux_lists, key=len)),
        len(max(all_polar_dust_flux_lists, key=len)),
        len(max(all_wave_inter_lists, key=len)),
        len(max(all_flux_inter_lists, key=len))
    )

    with open('data/spectra.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Get the wavelengths from the first iteration to use as column headers
        wavelength_headers = all_output_wave_lists[0]

        # Write the headers (wavelengths)
        writer.writerow(wavelength_headers)

        # Now write flux values for each iteration under the corresponding wavelength
        for iteration in range(total_iterations):
            # Write the flux values for the current iteration
            writer.writerow(all_output_flux_lists[iteration])

        # # Writing the data row by row, padding shorter columns with NaN
        # for i in range(max_len):
        #     row = []
        #     for j in range(total_iterations):
        #         # Pad shorter lists with NaN where needed
        #         row.append(all_output_wave_lists[j][i] if i < len(all_output_wave_lists[j]) else np.nan)
        #         row.append(all_output_flux_lists[j][i] if i < len(all_output_flux_lists[j]) else np.nan)
        #         # row.append(all_starburst_flux_lists[j][i] if i < len(all_starburst_flux_lists[j]) else np.nan)
        #         # row.append(all_agn_flux_lists[j][i] if i < len(all_agn_flux_lists[j]) else np.nan)
        #         # row.append(all_spheroid_flux_lists[j][i] if i < len(all_spheroid_flux_lists[j]) else np.nan)
        #         # row.append(all_polar_dust_flux_lists[j][i] if i < len(all_polar_dust_flux_lists[j]) else np.nan)
        #         # row.append(all_wave_inter_lists[j][i] if i < len(all_wave_inter_lists[j]) else np.nan)
        #         # row.append(all_flux_inter_lists[j][i] if i < len(all_flux_inter_lists[j]) else np.nan)
            
        #     # Write the row to the CSV
        #     writer.writerow(row)
    
        # Writing all data to a single CSV file
    with open('data/interpolated_spectra.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Get the interpolated wavelengths from the first iteration to use as column headers
        interpolated_wavelength_headers = all_wave_inter_lists[0]

        # Write the headers (interpolated wavelengths)
        writer.writerow(interpolated_wavelength_headers)

        # Write interpolated flux values for each iteration under the corresponding wavelength
        for iteration in range(total_iterations):
            writer.writerow(all_flux_inter_lists[iteration])

        # # Writing the data row by row, padding shorter columns with NaN
        # for i in range(max_len):
        #     row_inter = []

        #     # Add the wavelength only from the first iteration
        #     row_inter.append(all_wave_inter_lists[0][i] if i < len(all_wave_inter_lists[0]) else np.nan)

        #     # Add flux data for all iterations
        #     for j in range(total_iterations):
        #         row_inter.append(all_flux_inter_lists[j][i] if i < len(all_flux_inter_lists[j]) else np.nan)

        #     # Write the row to the CSV
        #     writer.writerow(row_inter)
    plt = kwargs.get('plot')
    if plt == True:
        plot_data(all_output_wave_lists[1], all_output_flux_lists[1], all_wave_inter_lists[1], all_flux_inter_lists[1])
    return output_wave_list, output_flux_list, wave_inter_list, flux_inter_list



def plot_data(output_wave_list, output_flux_list, wave_interpol, photometry):
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.figure()
    plt.plot(output_wave_list, output_flux_list, zorder=1)
    
    plt.scatter(wave_interpol, photometry,s=15,c='red',zorder=2)
    plt.legend(['Simulated Spectra','Photometry'])
    plt.ylabel('Normalized Flux (fv)')
    plt.xlabel("Wavelength (μm)")
    
    plt.savefig('synth.png', dpi=300, bbox_inches='tight')
    plt.show()

