import random
import sys
sys.path.append('./SMART_code')
from synthesis_routine_SMART import (galaxy_starburst2_fnu, 
                                     tapered_disc, 
                                     galaxy_spheroid_fnu, 
                                     polar_dust_fnu2,
                                     flared_disc,
                                     st16_disc,
                                     s15_disc)

import math
import numpy as np
from scipy import interpolate
import galaxy_settings
import sys


class Galaxy:
    
    def __init__(self):
        self
        
    
    def create_starburst(self):
        t_e_r = random.uniform(galaxy_settings.t_e[0], galaxy_settings.t_e[1])
        age_r = random.uniform(galaxy_settings.age[0], galaxy_settings.age[1])
        tau_v_r = random.uniform(galaxy_settings.tau_v[0], galaxy_settings.tau_v[1])
        # tm = random.randrange(4.9e7, 6.9e7)
        fsb_r = random.uniform(galaxy_settings.fsb[0],galaxy_settings.fsb[1])
        
        starb=galaxy_starburst2_fnu([10.**(tau_v_r),10.**(age_r),10.**(t_e_r),5.9e7])

        wave_synth=starb[0]
        f1=(10.**(fsb_r))*starb[1]
        return wave_synth, f1
    
    
    
    def create_agn(self, agn_kw):
        #  AGN emission
        fagn_r = random.uniform(galaxy_settings.fagn[0],galaxy_settings.fagn[1])
        cy_r2tor1 = 50.
        cy_theta_1 = 45.
        cy_tau_uv = 750.
        # Choose AGNType
        if 'cygnus' in agn_kw:
            # AGN1
            r2tor1 = [0.99*cy_r2tor1,1.01*cy_r2tor1]  
            theta_1 = [0.99*cy_theta_1,1.01*cy_theta_1]
            tau_uv = [0.99*cy_tau_uv,1.01*cy_tau_uv]
            tau_uv_r = random.uniform(tau_uv[0], tau_uv[1])
            r2tor1_r = random.uniform(r2tor1[0], r2tor1[1])
            theta_1_r = random.uniform(theta_1[0], theta_1[1])
            theta_v_r = random.uniform(galaxy_settings.theta_v[0], galaxy_settings.theta_v[1])
            cor_theta_v=theta_v_r
            if theta_v_r > theta_1_r and theta_v_r < 65.:
        #      
                cor_theta_v=0.9*theta_1_r
        #
            agn=tapered_disc([tau_uv_r,r2tor1_r,theta_1_r,cor_theta_v])
        #
            flux= (agn[2] + 1.e-40) * agn[1]

        fr_ct = 45.
        fr_rm=50.
        if 'fritz' in agn_kw:
            ct = [0.99*fr_ct,1.01*fr_ct]
            rm = [0.99*fr_rm,1.01*fr_rm]
            # AGN2
            ct_r = random.uniform(ct[0], ct[1])
            rm_r = random.uniform(rm[0], rm[1])
            ta_r = random.uniform(galaxy_settings.ta[0], galaxy_settings.ta[1])
            thfr06_r = random.uniform(galaxy_settings.thfr06[0], galaxy_settings.thfr06[1])
            cor_thfr06=thfr06_r
            if thfr06_r > 0.9*ct_r and thfr06_r < 80.:
        # 
                cor_thfr06=0.9*ct_r
            
            agn=flared_disc([10.**ct_r,rm_r,10.**ta_r,10.**cor_thfr06])
        #
            flux= (agn[2] + 1.e-40) * agn[1]

        sk_oa=50.
        sk_rr=20.
        sk_tt=7.5
        if 'skirtor' in agn_kw:
            oa = [0.99*sk_oa,1.01*sk_oa]
            rr = [0.99*sk_rr,1.01*sk_rr]
            tt = [0.99*sk_tt,1.01*sk_tt] 
            # AGN3
            oa_r = random.uniform(oa[0], oa[1])
            rr_r = random.uniform(rr[0], rr[1])
            tt_r = random.uniform(tt[0], tt[1])
            thst16_r = random.uniform(galaxy_settings.thst16[0], galaxy_settings.thst16[1])
            cor_thst16=thst16_r
            if thst16_r > 0.9*oa_r and thst16_r < 80.:
        
                cor_thst16=0.9*oa_r
            
            agn=st16_disc([10.**oa_r,10.**rr_r,10.**tt_r,10.**cor_thst16])
        
            flux= (agn[2] + 1.e-40) * agn[1]

        si_vc=20.
        si_ac=20.
        si_ad=250.
        if 'siebenmorgen' in agn_kw:
            vc = [0.99*si_vc,1.01*si_vc]
            ac = [0.99*si_ac,1.01*si_ac]
            ad = [0.99*si_ad,1.01*si_ad]
            # AGN4
            vc_r = random.uniform(vc[0], vc[1])
            ac_r = random.uniform(ac[0], ac[1])
            ad_r = random.uniform(ad[0], ad[1])
            th_r = random.uniform(galaxy_settings.th[0], galaxy_settings.th[1])
            cor_th=th_r
        
            agn=s15_disc([10.**vc_r,10.**ac_r,10.**ad_r,10.**cor_th])
        
            flux= (agn[2] + 1.e-40) * agn[1]

        mm= np.amax(flux)
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
        star = self.create_starburst()
        f2 = star[1]*0. + 1.e-40
        
        for l in range(len(star[0])):
            if star[0][l] < np.amax(wave_agn):         
                fff= agn_func(math.log10(star[0][l]))
                f2[l]=10.**fff
        return f2
    
    def create_spheroid(self, models_fnu_full):
                # Add the Spheroid
        tvv_r = random.uniform(galaxy_settings.tvv[0], galaxy_settings.tvv[1])
        psi_r = random.uniform(galaxy_settings.psi[0], galaxy_settings.psi[1])
        iview_r = random.uniform(galaxy_settings.iview[0], galaxy_settings.iview[1],)
        cirr_tau_r = random.uniform(galaxy_settings.cirr_tau[0],galaxy_settings.cirr_tau[1])
        fsph_r = random.uniform(galaxy_settings.fsph[0],galaxy_settings.fsph[1])
        star = self.create_starburst()
        spheroid=galaxy_spheroid_fnu([10.**tvv_r,10.**psi_r,10.**cirr_tau_r,1.,models_fnu_full,star[0]])

        f3=(10.**(fsph_r))*spheroid[1]   * spheroid[0] 
        return f3

    def create_polar_dust(self):
        polt_r = random.uniform(galaxy_settings.polt[0], galaxy_settings.polt[1])
        fpol_r = random.uniform(galaxy_settings.fpol[0], galaxy_settings.fpol[1])

        temp=polt_r
        polar_dust = polar_dust_fnu2([10.**temp])


        f4 = (10.**(fpol_r))*polar_dust[1]
        return f4
