import numpy as np
import pandas as pd
import sys
import math
from scipy import interpolate
sys.path.append('./SMART_code')
from likelihood import *
import SMART
import synthesis_routine_SMART
from synthesis_routine_SMART import (galaxy_starburst2_fnu, 
                                     tapered_disc, 
                                     galaxy_spheroid_fnu, 
                                     polar_dust_fnu2,
                                     flared_disc,
                                     st16_disc,
                                     s15_disc)
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import csv

np.seterr(all='ignore') #ignore over/underflow 0 errors

def create_flux(iterations,**kwargs):
    '''This function outputs the flux and wavelengths from the starburts, AGN, spheroidal, polar dust. The input is the number of spectra to create. 
    As kwargs give, plot:True or False, AGNType, phtometry:True or False, redshift'''

    # StarBurst
    t_e = np.log10([2.e7, 3.5e7])
    age = np.log10([5.e6, 3.e7])
    tau_v = np.log10([51., 250.])
    fsb = [-5.,5.]
    polt = np.log10([ 800., 1200. ])
    # Spheroid
    tvv = np.log10([ 0.1, 15.])
    psi = np.log10([ 1.1, 16.9])
    cirr_tau = np.log10([ 1.26e8, 7.9e9 ])
    iview = np.log10([ 0.1, 89.9 ])
    fsph = [-5.,5.]
    # AGN1
    theta_1 = np.log10([ 16., 58. ])
    tau_uv = np.log10([ 260., 1490. ])
    theta_v = np.log10([3, 90.0])
    r2tor1 = np.log10([21 , 100])
    fagn = [-5.,5.]
    # AGN2
    ct= np.log10([ 31., 69. ])
    rm= np.log10([ 11., 149. ])
    ta = np.log10([2., 6.])
    thfr06 = np.log10([1., 10.])
    # AGN3
    oa = np.log10([ 21., 75.])
    rr = np.log10([ 10., 30. ])
    tt = np.log10([ 3., 11. ])
    thst16 = np.log10([0., 89.])
    # AGN4
    vc = np.log10([ 16., 770.])
    ac = np.log10([ 1., 440. ])
    ad = np.log10([ 50., 499. ])
    th = np.log10([0.1, 10])
    # Polar Dust
    fpol = [-5., 5.]

    # Empty Lists
    output_wave_list = []
    output_flux_list = []
    flux_list_starb = []
    flux_list_sph = []
    flux_list_agn = []
    flux_list_polar = []
    wave_inter = []
    flux_inter = []
    flux_csv = []
    data_dict_gal = {}

    # Initialize the dictionary
    data_dict = {
    "t_e_r": [],
    "age_r": [],
    "tau_v_r": [],
    "tm": [],
    "fsb_r": [],
    "fagn_r": [],
    "tau_uv_r": [],
    "r2tor1_r": [],
    "theta_1_r": [],
    "theta_v_r": [],
    "ct_r": [],
    "rm_r": [],
    "ta_r": [],
    "thfr06_r": [],
    "oa_r": [],
    "rr_r": [],
    "tt_r": [],
    "thst16_r": [],
    "vc_r": [],
    "ac_r": [],
    "ad_r": [],
    "th_r": [],
    "tvv_r": [],
    "psi_r": [],
    "iview_r": [],
    "cirr_tau_r": [],
    "fsph_r": [],
    "polt_r": [],
    "fpol_r": []
    }

    # Shperoid
    # Choose the model with mettalicity 0.008
    redshift = kwargs.get('redshift')
    zstr=str(round((redshift/0.2))*0.2)
    metal='0.00130000'

    if redshift < 0.29:
        metal='0.0080000'
        zstr='0.1000'
    filename = './spheroid_models_5/spheroid_array1_z='+zstr[0:3]+'0000_met='+metal+'_X.npz'  
    data_sph = np.load(filename)

    # Get the models from the spheroid_models file
    models_fnu_full = data_sph['spheroid_models_fnu']
    tvv = np.log10([ 0.1, 15.])
    psi = np.log10([ 1.1, 16.9])
    cirr_tau = np.log10([ 1.26e8, 7.9e9 ])
    iview = np.log10([ 0.1, 89.9 ])
    
    iterations = iterations
    total_iterations = iterations # Choose random values within the specified ranges

    agn_kw = str(kwargs.get('agntype'))
    #   AGN CYGNUS keywords

    cy_r2tor1 = 50.
    cy_theta_1 = 45.
    cy_tau_uv = 750.
    if 'cygnus' in agn_kw:
      r2tor1 = [0.99*cy_r2tor1,1.01*cy_r2tor1]  
      theta_1 = [0.99*cy_theta_1,1.01*cy_theta_1]
      tau_uv = [0.99*cy_tau_uv,1.01*cy_tau_uv]

    #   AGN Fritz keywords

    fr_ct = 45.
    fr_rm=50.
    if 'fritz' in agn_kw:
      ct = [0.99*fr_ct,1.01*fr_ct]
      rm = [0.99*fr_rm,1.01*fr_rm]

    #   AGN SKIRTOR keywords

    sk_oa=50.
    sk_rr=20.
    sk_tt=7.5
    if 'skirtor' in agn_kw:
      oa = [0.99*sk_oa,1.01*sk_oa]
      rr = [0.99*sk_rr,1.01*sk_rr]
      tt = [0.99*sk_tt,1.01*sk_tt] 

    #   AGN Siebenmorgen keywords

    si_vc=20.
    si_ac=20.
    si_ad=250.
    if 'siebenmorgen' in agn_kw:
      vc = [0.99*si_vc,1.01*si_vc]
      ac = [0.99*si_ac,1.01*si_ac]
      ad = [0.99*si_ad,1.01*si_ad]

            # Load the wavelength to interpolate
    data = np.genfromtxt("filters1.txt",skip_header=1,dtype=None,encoding='ascii')
    filters, filter_names=[data[i] for i in data.dtype.names]
    filter_name = []

    

    for iteration in tqdm(range(int(total_iterations)), desc="Processing"):  # Add progress bar
        loop_key = f'Galaxy{iteration+1}'
        #    Starburst emission
        t_e_r = random.uniform(t_e[0], t_e[1])
        age_r = random.uniform(age[0], age[1])
        tau_v_r = random.uniform(tau_v[0], tau_v[1])
        tm = random.randrange(4.9e7, 6.9e7)
        fsb_r = random.uniform(fsb[0],fsb[1])
        
        starb=galaxy_starburst2_fnu([10.**(tau_v_r),10.**(age_r),10.**(t_e_r),5.9e7])

        wave_synth=starb[0]
        f1=(10.**(fsb_r))*starb[1]
        #  AGN emission

        fagn_r = random.uniform(fagn[0],fagn[1])

        # AGN1
        tau_uv_r = random.uniform(tau_uv[0], tau_uv[1])
        r2tor1_r = random.uniform(r2tor1[0], r2tor1[1])
        theta_1_r = random.uniform(theta_1[0], theta_1[1])
        theta_v_r = random.uniform(theta_v[0],theta_v[1])
        
        # AGN2
        ct_r = random.uniform(ct[0], ct[1])
        rm_r = random.uniform(rm[0], rm[1])
        ta_r = random.uniform(ta[0], ta[1])
        thfr06_r = random.uniform(thfr06[0], thfr06[1])
        # AGN3
        oa_r = random.uniform(oa[0], oa[1])
        rr_r = random.uniform(rr[0], rr[1])
        tt_r = random.uniform(tt[0], tt[1])
        thst16_r = random.uniform(thst16[0], thst16[1])
        # AGN4
        vc_r = random.uniform(vc[0], vc[1])
        ac_r = random.uniform(ac[0], ac[1])
        ad_r = random.uniform(ad[0], ad[1])
        th_r = random.uniform(th[0], th[1])

        # Choose AGNType
        if 'cygnus' in agn_kw:
            cor_theta_v=theta_v_r
            if theta_v_r > theta_1_r and theta_v_r < 65.:
        #      
                cor_theta_v=0.9*theta_1_r
        #
            agn=tapered_disc([tau_uv_r,r2tor1_r,theta_1_r,cor_theta_v])
        #
            flux= (agn[2] + 1.e-40) * agn[1]
     
        if 'fritz' in agn_kw:
            cor_thfr06=thfr06_r
            if thfr06_r > 0.9*ct_r and thfr06_r < 80.:
        # 
                cor_thfr06=0.9*ct_r
            
            agn=flared_disc([10.**ct_r,rm_r,10.**ta_r,10.**cor_thfr06])
        #
            flux= (agn[2] + 1.e-40) * agn[1]

     
        if 'skirtor' in agn_kw:
            cor_thst16=thst16_r
            if thst16_r > 0.9*oa_r and thst16_r < 80.:
        
                cor_thst16=0.9*oa_r
            
            agn=st16_disc([10.**oa_r,10.**rr_r,10.**tt_r,10.**cor_thst16])
        
            flux= (agn[2] + 1.e-40) * agn[1]

        if 'siebenmorgen' in agn_kw:
            cor_th=th_r
        
            agn=s15_disc([10.**vc_r,10.**ac_r,10.**ad_r,10.**cor_th])
        
            flux= (agn[2] + 1.e-40) * agn[1]

        mm=np.amax(flux)
        bb=flux/mm

        agn_f=(10.**fagn_r)*bb + 1.e-40   

        #  interpolate the AGN emission onto the SB grid

        wave_agn=agn[1]
        lwave_agn=wave_agn*0.
        lagn_f=agn_f*0.

        for l in range(len(wave_agn)):
                
            lwave_agn[l]=math.log10(wave_agn[l])
            lagn_f[l]=math.log10(agn_f[l])

        agn_func=interpolate.interp1d(lwave_agn,lagn_f)

        f2 = f1*0. + 1.e-40

        for l in range(len(wave_synth)):
            if wave_synth[l] < np.amax(wave_agn):         
                fff=agn_func(math.log10(wave_synth[l]))
                f2[l]=10.**fff


        # Add the Spheroid
        tvv_r = random.uniform(tvv[0], tvv[1])
        psi_r = random.uniform(psi[0], psi[1])
        iview_r = random.uniform(iview[0], iview[1],)
        cirr_tau_r = random.uniform(cirr_tau[0],cirr_tau[1])
        fsph_r = random.uniform(fsph[0],fsph[1])

        spheroid=galaxy_spheroid_fnu([10.**tvv_r,10.**psi_r,10.**cirr_tau_r,1.,models_fnu_full,wave_synth])

        f3=(10.**(fsph_r))*spheroid[1]   * spheroid[0] 

        polt_r = random.uniform(polt[0], polt[1])
        fpol_r = random.uniform(fpol[0], fpol[1])

        temp=polt_r
        polar_dust = polar_dust_fnu2([10.**temp])


        f4 = (10.**(fpol_r))*polar_dust[1]

        #   Add  all components
        
        fall = f1 + f2 + f3 + f4
        
        output_wave=wave_synth
        output_flux=fall
        
        for wave, flux_ass, f1, f2, f3, f4 in zip(output_wave, np.log10(output_flux), 
                                np.log10(f1), 
                                np.log10(f2), 
                                np.log10(f3), 
                                np.log10(f4)):
            if wave < 1300:
                if kwargs.get("log_wave",False):
                    output_wave_list.append(np.log10(wave))
                    
                else:
                    output_wave_list.append(wave)
                output_flux_list.append(flux_ass)
                
                flux_list_starb.append(f1)

                flux_list_agn.append(f2)

                flux_list_sph.append(f3)

                flux_list_polar.append(f4)
                
                data_dict_gal[loop_key] = {
                    # 'wavelength': output_wave,  # Assuming output_wave is the generated wavelength data
                    'flux': flux_ass         # Assuming output_flux is the generated flux data
                }
                # Load the filters
        flux_csv.append(np.log10(output_flux))

    # Save the random values
        data_dict["t_e_r"].append(t_e_r)
        data_dict["age_r"].append(age_r)
        data_dict["tau_v_r"].append(tau_v_r)
        data_dict["tm"].append(tm)
        data_dict["fsb_r"].append(fsb_r)
        data_dict["fagn_r"].append(fagn_r)
        data_dict["tau_uv_r"].append(tau_uv_r)
        data_dict["r2tor1_r"].append(r2tor1_r)
        data_dict["theta_1_r"].append(theta_1_r)
        data_dict["theta_v_r"].append(theta_v_r)
        data_dict["ct_r"].append(ct_r)
        data_dict["rm_r"].append(rm_r)
        data_dict["ta_r"].append(ta_r)
        data_dict["thfr06_r"].append(thfr06_r)
        data_dict["oa_r"].append(oa_r)
        data_dict["rr_r"].append(rr_r)
        data_dict["tt_r"].append(tt_r)
        data_dict["thst16_r"].append(thst16_r)
        data_dict["vc_r"].append(vc_r)
        data_dict["ac_r"].append(ac_r)
        data_dict["ad_r"].append(ad_r)
        data_dict["th_r"].append(th_r)
        data_dict["tvv_r"].append(tvv_r)
        data_dict["psi_r"].append(psi_r)
        data_dict["iview_r"].append(iview_r)
        data_dict["cirr_tau_r"].append(cirr_tau_r)
        data_dict["fsph_r"].append(fsph_r)
        data_dict["polt_r"].append(polt_r)
        data_dict["fpol_r"].append(fpol_r)

        # Save the wave and flux to lists
        output_wave_list.append(np.nan)
        output_flux_list.append(np.nan)
        flux_list_starb.append(np.nan)
        flux_list_agn.append(np.nan)
        flux_list_sph.append(np.nan)
        flux_list_polar.append(np.nan)
        if kwargs.get('photometry', False):

            observed_wave = filters/(1+redshift)
            if kwargs.get('log_wave',False):
                wave_interpol = np.log10(observed_wave)
                interpolation_function = interpolate.interp1d(np.log10(wave_synth), np.log10(fall))
                photometry = interpolation_function(wave_interpol)
                filter_name.append(filter_names)
                wave_inter.extend(wave_interpol)
                flux_inter.extend(photometry)
            else:
                wave_interpol = observed_wave
                interpolation_function = interpolate.interp1d(wave_synth, np.log10(fall))
                photometry = interpolation_function(wave_interpol)

    if kwargs.get('plot', False):
        # Plot the Data
        plot1 = plt.subplot2grid((4, 4), (0, 0), colspan=2)
        plot2 = plt.subplot2grid((4, 4), (0, 2), rowspan=3, colspan=2)
        plot3 = plt.subplot2grid((4, 4), (1, 0), rowspan=2, colspan=2)
        plot4 = plt.subplot2grid((4, 4), (3, 0), colspan=2)

        # Plotting
        plot1.scatter(output_wave_list, flux_list_starb, s=5)
        plot1.set_title('StarBurst')

        plot2.scatter(output_wave_list, flux_list_agn, s=5)
        plot2.set_title('AGN')

        plot3.scatter(output_wave_list, flux_list_sph, s=5)
        plot3.set_title('Spheroid')

        plot4.scatter(output_wave_list, flux_list_polar, s=5)
        plot4.set_title('Polar Dust')

        plt.tight_layout()
        plt.savefig('subplots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    plot_data(output_wave_list, output_flux_list, wave_inter, flux_inter)
    
    wave_clean = np.array(output_wave_list)
    flux = np.array(output_flux_list)
    # Remove NAN used to differentiate the data for every galaxy
    w = wave_clean[~np.isnan(wave_clean)] 
    f = flux[~np.isnan(flux)]
    fil = pd.DataFrame(filter_names, index=None)
    
    
    # w = data['wave']
    # f = data['flux']

    w_i = wave_inter

    f_i = flux_inter
    # flux_array = np.vstack(flux_all_iterations)
    np.savez('starb_agn_sph_wave_flux_photometry.npz', wave=output_wave_list, flux=data_dict, wave_interpol = wave_inter, flux_interpol = flux_inter,filter_names = np.array(filter_name), values = data_dict)
    # np.savetxt('galaxy_star.csv', flux_all_iterations)
    
    transposed_data = list(zip(*flux_csv))
    with open('./GANS/flux_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(transposed_data)
    return output_wave_list, output_flux_list, wave_inter, flux_inter



def plot_data(output_wave_list, output_flux_list, wave_interpol, photometry):
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.figure()
    plt.plot(output_wave_list, output_flux_list, zorder=1)
    
    plt.scatter(wave_interpol, photometry,s=10,c='red',zorder=2)
    plt.legend(['Simulated Spectra','Photometry'])
    plt.ylabel('Normalized Flux (fv)')
    plt.xlabel("Wavelength (μm)")
    
    plt.savefig('synth.png', dpi=300, bbox_inches='tight')
    plt.show()

# Execute the main function
if __name__ == "__main__":

    num_galaxies = 2
    

    wave, flux, wave_interpol, photometry = create_flux(num_galaxies, plot=True, redshift=0.21, agntype = 'cygnus', photometry = True, log_wave = True)
