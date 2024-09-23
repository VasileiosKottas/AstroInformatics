import numpy as np
import matplotlib.pyplot as plt
import emcee
import os as asd

import sys
sys.path.append('./SMART_code')

from synthesis_routine_SMART import * 

from likelihood import *

from multiprocessing import Pool


def define_p00_host(nwalkers,ndim,host,hostType,**kwargs):
    
    if hostType==1:
            sph=host
            p00 =  [-0.8, 1.6, -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.] + 0.1 * np.random.randn(nwalkers, ndim)
            
    #  Spheroidal parameters
            p01 =  sph[0][0][0]  +  (sph[0][0][1] - sph[0][0][0]) * np.random.rand(nwalkers, 1)
            p02 =  sph[0][1][0]  +  (sph[0][1][1] - sph[0][1][0] ) * np.random.rand(nwalkers, 1)
            p03 =  sph[0][2][0]  +  (sph[0][2][1] - sph[0][2][0] ) * np.random.rand(nwalkers, 1)
            p04 =  sph[0][3][0]  +  (sph[0][3][1] - sph[0][3][0] ) * np.random.rand(nwalkers, 1)
            
    #  Starburst parameters
            p05 =  sph[0][4][0]  +  ( sph[0][4][1] - sph[0][4][0]) * np.random.rand(nwalkers, 1)
            p06 =  sph[0][5][0]  +  (sph[0][5][1] - sph[0][5][0] ) * np.random.rand(nwalkers, 1)
            p07 =  sph[0][6][0]  +  (sph[0][6][1] - sph[0][6][0] ) * np.random.rand(nwalkers, 1)
            p08 =  sph[0][7][0]  +  (sph[0][7][1] - sph[0][7][0] ) * np.random.rand(nwalkers, 1)
            
    #   AGN parameters
            p09 =  sph[0][8][0]  +  (sph[0][8][1] - sph[0][8][0] ) * np.random.rand(nwalkers, 1)
            p10 =  sph[0][9][0]  +  (sph[0][9][1] - sph[0][9][0] ) * np.random.rand(nwalkers, 1)
            p11 =  sph[0][10][0] +  (sph[0][10][1] - sph[0][10][0] ) * np.random.rand(nwalkers, 1)
            p12 =  sph[0][11][0] +  (sph[0][11][1] - sph[0][11][0] ) * np.random.rand(nwalkers, 1)
            p13 =  sph[0][12][0] +  (sph[0][12][1] - sph[0][12][0] ) * np.random.rand(nwalkers, 1)
            
    #   Polar dust parameters
            p14 =  sph[0][13][0] +  (sph[0][13][1] - sph[0][13][0] ) * np.random.rand(nwalkers, 1)
            p15 =  sph[0][14][0] +  (sph[0][14][1] - sph[0][14][0] ) * np.random.rand(nwalkers, 1)
    
            for i in range(nwalkers):
              p00[i,0] =   p01[i]
              p00[i,1] =   p02[i]
              p00[i,2] =   p03[i]
              p00[i,3] =   p04[i]
              p00[i,4] =   p05[i]
              p00[i,5] =   p06[i]
              p00[i,6] =   p07[i]
              p00[i,7] =   p08[i]
              p00[i,8] =   p09[i]
              p00[i,9] =   p10[i]
              p00[i,10] =  p11[i]
              p00[i,11] =  p12[i]
              p00[i,12] =  p13[i]
              p00[i,13] =  p14[i]
              p00[i,14] =  p15[i]
    
    
    if hostType==2:
            disc=host
            p00 =  [-0.8, 1.6, -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.] + 0.1 * np.random.randn(nwalkers, ndim)
            
    #  Disc parameters
            p01 =  disc[0][0][0]      +  (disc[0][0][1] - disc[0][0][0]) * np.random.rand(nwalkers, 1)
            p02 =  disc[0][1][0]  +  (disc[0][1][1] - disc[0][1][0] ) * np.random.rand(nwalkers, 1)
            p03 =  disc[0][2][0]  +  (disc[0][2][1] - disc[0][2][0] ) * np.random.rand(nwalkers, 1)
            p04 =  disc[0][3][0]  +  (disc[0][3][1] - disc[0][3][0] ) * np.random.rand(nwalkers, 1)
            p05 =  disc[0][4][0]      +  (disc[0][4][1] - disc[0][4][0] ) * np.random.rand(nwalkers, 1)
            
    #  Starburst parameters
            p06 =  disc[0][5][0]      +  ( disc[0][5][1] - disc[0][5][0]) * np.random.rand(nwalkers, 1)
            p07 =  disc[0][6][0]    +  (disc[0][6][1] - disc[0][6][0] ) * np.random.rand(nwalkers, 1)
            p08 =  disc[0][7][0]     +  (disc[0][7][1] - disc[0][7][0] ) * np.random.rand(nwalkers, 1)
            p09 =  disc[0][8][0]    +  (disc[0][8][1] - disc[0][8][0] ) * np.random.rand(nwalkers, 1)
            
    #   AGN parameters
            p10 =  disc[0][9][0]    +  (disc[0][9][1] - disc[0][9][0] ) * np.random.rand(nwalkers, 1)
            p11 =  disc[0][10][0]  +  (disc[0][10][1] - disc[0][10][0] ) * np.random.rand(nwalkers, 1)
            p12 =  disc[0][11][0]  +  (disc[0][11][1] - disc[0][11][0] ) * np.random.rand(nwalkers, 1)
            p13 =  disc[0][12][0] +  (disc[0][12][1] - disc[0][12][0] ) * np.random.rand(nwalkers, 1)
            p14 =  disc[0][13][0] +  (disc[0][13][1] - disc[0][13][0] ) * np.random.rand(nwalkers, 1)
            
    #   Polar dust parameter
            p15 =  disc[0][14][0] +  (disc[0][14][1] - disc[0][14][0] ) * np.random.rand(nwalkers, 1)
            p16 =  disc[0][15][0] +  (disc[0][15][1] - disc[0][15][0] ) * np.random.rand(nwalkers, 1)
    
            for i in range(nwalkers):
              p00[i,0] =   p01[i]
              p00[i,1] =   p02[i]
              p00[i,2] =   p03[i]
              p00[i,3] =   p04[i]
              p00[i,4] =   p05[i]
              p00[i,5] =   p06[i]
              p00[i,6] =   p07[i]
              p00[i,7] =   p08[i]
              p00[i,8] =   p09[i]
              p00[i,9] =   p10[i]
              p00[i,10] =  p11[i]
              p00[i,11] =  p12[i]
              p00[i,12] =  p13[i]
              p00[i,13] =  p14[i]
              p00[i,14] =  p15[i]
              p00[i,15] =  p16[i]
    
    return p00     


def SMART(*argv,**kwargs):
#
#+ 
# NAME:
#   SMART
# 
# PURPOSE:
# 
#    This is a routine that uses emcee (Foreman-MAcKey et al 2013). It reads a list of objects from a file and     
#    decides depending on the flag which objects to fit (it fits the objects with flag=flag_select).
# 
#
# CATEGORY:
# 
# CALLING SEQUENCE:
# 
#       SMART
# 
# INPUTS:
# 
#    flag_select:  fit only objects with given flag
# 
# 
# OPTIONAL INPUTS:
#     
#    All the keywords
# 
# KEYWORD PARAMETERS:
# 
#    Explained in SMART User Manual
# 
# OUTPUTS:
# 
#  Plots. No other output, everything is written to files for each fitted galaxy      
# 
# OPTIONAL OUTPUTS:
# 
# COMMON BLOCKS:
# 
# SIDE EFFECTS:
#   None known
# 
# RESTRICTIONS:
#   None known
# 
# PROCEDURE:
# 
# 
# EXAMPLE:
# 
#      Python>from SMART import *
# 
# MODIFICATION HISTORY:
# 
#   Written by:  Charalambia Varnava, September 2020, & Andreas Efstathiou, May 2021
#                                  
# 
#-
    np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
    nfixed = 0
#
#  Starburst keywords
#
    t_e = np.log10([ 1.e7, 3.5e7 ])
    cy_t_e=2.e7
    if 'cy_t_e' in kwargs:
      cy_t_e=kwargs.get('cy_t_e')
      t_e = np.log10([0.99*cy_t_e,1.01*cy_t_e])
      nfixed = nfixed + 1
#
    age = np.log10([ 5.e6, 3.5e7 ])
    cy_age=2.e7
    if 'cy_age' in kwargs:
      cy_age=kwargs.get('cy_age')
      age = np.log10([0.99*cy_age,1.01*cy_age])
      nfixed = nfixed + 1
#
    tau_v = np.log10([ 51., 250. ])
    cy_tau_v=100.
    if 'cy_tau_v' in kwargs:
      cy_tau_v=kwargs.get('cy_tau_v')
      tau_v = np.log10([0.99*cy_tau_v,1.01*cy_tau_v])
      nfixed = nfixed + 1
#
#  Spheroidal keywords
#
    tvv = np.log10([ 0.1, 15.])
    cy_tvv=5.
    if 'cy_tvv' in kwargs:
      cy_tvv=kwargs.get('cy_tvv')
      tvv = np.log10([0.99*cy_tvv,1.01*cy_tvv])
      nfixed = nfixed + 1
#
    psi = np.log10([ 1.1, 16.9])
    cy_psi=5.
    if 'cy_psi' in kwargs:
      cy_psi=kwargs.get('cy_psi')
      psi = np.log10([0.99*cy_psi,1.01*cy_psi])
      nfixed = nfixed + 1
#
    cirr_tau = np.log10([ 1.26e8, 7.9e9 ])
    cy_cirr_tau=5.
    if 'cy_cirr_tau' in kwargs:
      cy_cirr_tau=kwargs.get('cy_cirr_tau')
      cirr_tau = np.log10([0.99*cy_cirr_tau,1.01*cy_cirr_tau])
      nfixed = nfixed + 1
#
#  Disc keywords
#
    iview = [ np.log10(0.1), np.log10(89.9) ]
    cy_iview=45.
    if 'cy_iview' in kwargs:
      cy_iview=kwargs.get('cy_iview')
      iview = np.log10([0.99*cy_iview,1.01*cy_iview])
      nfixed = nfixed + 1
#
#   AGN CYGNUS keywords
#
    r2tor1 = np.log10([ 21., 99. ])
    cy_r2tor1 = 50.
    if 'cy_r2tor1' in kwargs:
      cy_r2tor1=kwargs.get('cy_r2tor1')
      r2tor1 = np.log10([0.99*cy_r2tor1,1.01*cy_r2tor1])
      nfixed = nfixed + 1
#      
    theta_1 = np.log10([ 16., 58. ])
    cy_theta_1 = 45.
    if 'cy_theta_1' in kwargs:
      cy_theta_1=kwargs.get('cy_theta_1')    
      theta_1 = np.log10([0.99*cy_theta_1,1.01*cy_theta_1])
      nfixed = nfixed + 1
#      
    tau_uv = np.log10([ 260., 1490. ])
    cy_tau_uv = 750.
    if 'cy_tau_uv' in kwargs:
      cy_tau_uv=kwargs.get('cy_tau_uv')    
      tau_uv = np.log10([0.99*cy_tau_uv,1.01*cy_tau_uv])
      nfixed = nfixed + 1
#
#   AGN Fritz keywords
#
    ct = np.log10([ 31., 69. ])
    fr_ct = 45.
    if 'fr_ct' in kwargs:
      fr_ct=kwargs.get('fr_ct')
      ct = np.log10([0.99*fr_ct,1.01*fr_ct])
      nfixed = nfixed + 1
#
    rm = np.log10([ 11., 149. ])
    fr_rm=50.
    if 'fr_rm' in kwargs:
      fr_rm=kwargs.get('fr_rm')
      rm = np.log10([0.99*fr_rm,1.01*fr_rm])
      nfixed = nfixed + 1
#
#   AGN SKIRTOR keywords
#
    oa = np.log10([ 21., 75.])
    sk_oa=50.
    if 'sk_oa' in kwargs:
      sk_oa=kwargs.get('sk_oa')
      oa = np.log10([0.99*sk_oa,1.01*sk_oa])
      nfixed = nfixed + 1
#
    rr = np.log10([ 10., 30. ])
    sk_rr=20.
    if 'sk_rr' in kwargs:
      sk_rr=kwargs.get('sk_rr')
      rr = np.log10([0.99*sk_rr,1.01*sk_rr])
      nfixed = nfixed + 1
#
    tt = np.log10([ 3., 11. ])
    sk_tt=7.5
    if 'sk_tt' in kwargs:
      sk_tt=kwargs.get('sk_tt')
      tt = np.log10([0.99*sk_tt,1.01*sk_tt])
      nfixed = nfixed + 1      
#
#   AGN Siebenmorgen keywords
#
    vc = np.log10([ 16., 770.])
    si_vc=20.
    if 'si_vc' in kwargs:
      si_vc=kwargs.get('si_vc')
      vc = np.log10([0.99*si_vc,1.01*si_vc])
      nfixed = nfixed + 1
#
    ac = np.log10([ 1., 440. ])
    si_ac=20.
    if 'si_ac' in kwargs:
      si_ac=kwargs.get('si_ac')
      ac = np.log10([0.99*si_ac,1.01*si_ac])
      nfixed = nfixed + 1
#
    ad = np.log10([ 50., 499. ])
    si_ad=250.
    if 'si_ad' in kwargs:
      si_ad=kwargs.get('si_ad')
      ad = np.log10([0.99*si_ad,1.01*si_ad])
      nfixed = nfixed + 1 
#
#   Polar dust keywords
#
    polt = np.log10([ 800., 1200. ])
    po_polt=1000.
    if 'polt' in kwargs:
      po_polt=kwargs.get('po_polt')
      polt = np.log10([0.99*po_polt,1.01*po_polt])
      nfixed = nfixed + 1
# 
#  Run objects from the list in file "./"+data_file+"_list.txt" which have flag=flag_select
# 
    flag_select=argv[0]
# 
#  
    data_file='objects'
    if 'data_file' in kwargs:    
      data_file=kwargs.get('data_file')  

    metallicity=0.008
#  
    if 'metallicity' in kwargs:    
      metallicity=kwargs.get('metallicity')  
# 
    yy_axis=[1.e10,1.e14]
    if 'y_axis' in kwargs:    
      yy_axis=kwargs.get('y_axis')  
#
    xx_axis=[0.1, 1000.] 
    if 'x_axis' in kwargs:    
      xx_axis=kwargs.get('x_axis')  
# 
#  
    host_geometry='sph'
    if 'host_geometry' in kwargs:    
      host_geometry=kwargs.get('host_geometry')  
    if host_geometry=='sph':
         hostType=1
    elif host_geometry=='disc':
        hostType=2
# 
    host_gal='yes'
    if 'host_gal' in kwargs: 
        host_gal=kwargs.get('host_gal')  
    if host_gal=='yes':
         if hostType==1:
             xxfsph=[ -5., 5. ]
         if hostType==2:
             xxfdisc=[ -5., 5. ]
# 
    starburst_gal='yes'
    if 'starburst_gal' in kwargs: 
        starburst_gal=kwargs.get('starburst_gal')  
    if starburst_gal=='yes':
         xxfsb=[ -5., 5. ]
#
    AGN_model='CYGNUS'
    if 'AGN_model' in kwargs:    
      AGN_model=kwargs.get('AGN_model')  
    if AGN_model=='CYGNUS':
         AGNType=1
    elif AGN_model=='Fritz':
        AGNType=2
    elif AGN_model=='SKIRTOR':
        AGNType=3
    elif AGN_model=='Siebenmorgen':
        AGNType=4
# 
    AGN_gal='yes'
    if 'AGN_gal' in kwargs:      
        AGN_gal=kwargs.get('AGN_gal')  
    if AGN_gal=='yes':
         xxfagn=[ -5., 5. ]  

# 
    polar='no'
    if 'polar' in kwargs:    
      polar=kwargs.get('polar')  
    if polar=='yes':
         xxfpol=[ -5., 5. ]       
# 
    numOfsamples = 500    
#
#   Read the list of galaxies to be fitted
#
    data=np.genfromtxt("./"+data_file+"_list.txt",skip_header=1,dtype=None,encoding='ascii')
    ruben_name_x,redshifts_x,flag_x,flag_xx=[data[i] for i in data.dtype.names]
# 
    gg=flag_x==int(flag_select)
#    
    redshifts=redshifts_x[gg]
    ruben_name=ruben_name_x[gg]
    flag=flag_x[gg]
    flag_type=flag_xx[gg]
#    
#    
    nn=0  
#    
    run_name='S'
    if 'run_name' in kwargs:
      run_name=kwargs.get('run_name')     
#     
    if not asd.path.exists(data_file+'_results/'):
        asd.makedirs('./'+data_file+'_results')
#     
    ii=0
    num_of_runs=1
    if 'num_of_runs' in kwargs:
        num_of_runs=kwargs.get('num_of_runs')
#   
    while ii<len(redshifts):
#       
        zz=redshifts[ii]
#
        name=ruben_name[ii]
#    
#        
        filename='./'+data_file+'/'+name+'.txt'
#
#    Read the data for current galaxy
#    
        data=np.genfromtxt(filename,skip_header=1,dtype=None,encoding='ascii')
#        ww,ff,ee,flag=[data[i] for i in data.dtype.names]
        ww,ff,ee,flag,source=[data[i] for i in data.dtype.names]
#
#
        www = ww / (1. + zz)
#
        igood = [k <= 1500. for k in www]
#
        wws = sorted(www[igood])
        models_fnu_x=select_library([zz,hostType,wws,metallicity])
#
        models_fnu_red=models_fnu_x[0]
        models_fnu_full=models_fnu_x[1]
#
        ii+=1
        run_id=1
        while run_id <= num_of_runs:
    #        
            run_id_temp = run_id
            XXX= run_name + str(run_id_temp)
    #
            print ('Fitting : ', name) 
#                  
    #        
    #    Take into account redshift
    #
            if run_id==1:
               ww=ww / (1. + zz)
               freq=3.e14/ww
    
            file=open('./'+data_file+'_results/'+name+'_'+XXX+'_fit_limits_and_data.npy','w')
            print (name,ww,ff,ee,flag, file=file)
            file.close()
          
            nwalkers=128
            if 'walkers' in kwargs:
              nwalkers=kwargs.get('walkers')     
                                  
            if hostType==1:
                ndim=15
                
            if hostType==2:
                ndim=16
    # 
    #    Work out scale
    #
            kgd=flag==int(0)
            ffx=ff[kgd]
            wwx=ww[kgd]
            high=np.where(wwx>3.)
            low=np.where(wwx<3.)
            scale=np.array([1.,1.])
            scale[0]=np.log10(np.max(ffx[high])) 
            scale[1]=scale[0] - 4.
            if len(low) > 1 and flag_type[ii-1]==2:
                scale[1]=np.log10(np.max(ffx[low])) 

    #  
    #
            host=priors_host_cygnus(scale,flag_type[ii-1],hostType,AGNType,host_gal=host_gal,starburst_gal=starburst_gal,AGN_gal=AGN_gal,polar=polar,
                               r2tor1=r2tor1,theta_1=theta_1,tau_uv=tau_uv,t_e=t_e,age=age,tau_v=tau_v,tvv=tvv,psi=psi,cirr_tau=cirr_tau,
                               iview=iview,ct=ct,rm=rm,oa=oa,rr=rr,tt=tt,vc=vc,ac=ac,ad=ad,polt=polt)
            
            p00 = define_p00_host(nwalkers,ndim,host,hostType)
           
            with Pool() as pool: 
                                                         
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_host, args=(ww, ff, ee, flag, models_fnu_red, host,hostType,AGNType, wws),pool=pool)
        
                sampler.run_mcmc(p00, numOfsamples,progress=True)
        
                samples = sampler.get_chain(flat=True)
                fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
                samples = sampler.get_chain()
            
            if hostType==1 and AGNType==1:
                labels = ["fsph","tvv","psi","cirr_tau","fsb","tau_v","age","t_e","fagn",
                                    "r2tor1","tau_uv","theta_1","theta_v","fpol","polt"]
                
            if hostType==1 and AGNType==2:
                labels = ["fsph","tvv","psi","cirr_tau","fsb","tau_v","age","t_e","fagn",
                                    "ct","rm","ta","thfr06","fpol","polt"]
                
            if hostType==1 and AGNType==3:
                labels = ["fsph","tvv","psi","cirr_tau","fsb","tau_v","age","t_e","fagn",
                                    "oa","rr","tt","thst16","fpol","polt"]
                
            if hostType==1 and AGNType==4:
                labels = ["fsph","tvv","psi","cirr_tau","fsb","tau_v","age","t_e","fagn",
                                    "vc","ac","ad","th","fpol","polt"]
            
            if hostType==2 and AGNType==1:
                labels = ["fdisc","tv","psi","cirr_tau","iview","fsb","tau_v","age","t_e","fagn",
                                    "r2tor1","tau_uv","theta_1","theta_v","fpol","polt"]
                
            if hostType==2 and AGNType==2:
                labels = ["fdisc","tv","psi","cirr_tau","iview","fsb","tau_v","age","t_e","fagn",
                                    "ct","rm","ta","thfr06","fpol","polt"]
                
            if hostType==2 and AGNType==3:
                labels = ["fdisc","tv","psi","cirr_tau","iview","fsb","tau_v","age","t_e","fagn",
                                    "oa","rr","tt","thst16","fpol","polt"]
                
            if hostType==2 and AGNType==4:
                labels = ["fdisc","tv","psi","cirr_tau","iview","fsb","tau_v","age","t_e","fagn",
                                    "vc","ac","ad","th","fpol","polt"]
                
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
                
            axes[-1].set_xlabel("step number")
    #
    #   Plot samples
    #
            samples_plot='no'
            if 'samples_plot' in kwargs:   
                samples_plot=kwargs.get('samples_plot')
            if samples_plot=='yes':
                plt.savefig('./'+data_file+'_results/'+name+'_'+XXX+'_samples.png')
    # 
            flat_samples_raw = sampler.get_chain(discard=100, thin=15, flat=True)
    # 
    # save numpy array as npz file
    #  
            from numpy import asarray
            from numpy import save
            filename2='./'+data_file+'_results/'+name+'_'+XXX+'_flat_samples.npz'          
    #
    #   Corner plot
    #
            corner_plot='no'
            if 'corner_plot' in kwargs:   
                corner_plot=kwargs.get('corner_plot')
            if corner_plot=='yes':
          
              # import corner
              
              import sys, os
              sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
              from getdist import plots, MCSamples
              import getdist
              import IPython

              if hostType==1 and polar=='no':
                  param=[1,2,3,5,6,7,9,10,11,12]
                  
              if hostType==1 and polar=='yes':
                  param=[1,2,3,5,6,7,9,10,11,12,14]

              if hostType==2 and polar=='no':
                  param=[1,2,3,4,6,7,8,10,11,12,13]  
                  
              if hostType==2 and polar=='yes':
                  param=[1,2,3,4,6,7,8,10,11,12,13,15]                 
              
              flat_samples_raw_corner=np.zeros((len(flat_samples_raw),len(param)))
              
              # host_corner=np.zeros((len(param),2))
              
              k=0
              
              for i in range(len(param)):
                     flat_samples_raw_corner[:,k]=flat_samples_raw[:,(param[i])]
                     
                     # host_corner[k]= host[('xx'+ labels[(param[i])])]
                     
                     flat_samples_raw_corner[:,k]=10.**flat_samples_raw_corner[:,k]
                     
                     if hostType==1 and AGNType==1:
                         if param[i]==11 or param[i]==12:
                             flat_samples_raw_corner[:,k]=90-(flat_samples_raw_corner[:,k])
                             
                     if hostType==1 and AGNType==2 or hostType==1 and AGNType==3:
                         if param[i]==9 or param[i]==12:
                             flat_samples_raw_corner[:,k]=90-(flat_samples_raw_corner[:,k])
                             
                     if hostType==1 and AGNType==4:
                         if param[i]==12:
                             flat_samples_raw_corner[:,k]=90-(flat_samples_raw_corner[:,k])  
                             
                     if hostType==2 and AGNType==1:
                         if param[i]==4 or param[i]==12 or param[i]==13:
                             flat_samples_raw_corner[:,k]=90-(flat_samples_raw_corner[:,k])
                             
                     if hostType==2 and AGNType==2 or hostType==2 and AGNType==3:
                         if param[i]==4 or param[i]==10 or param[i]==13:
                             flat_samples_raw_corner[:,k]=90-(flat_samples_raw_corner[:,k])
                             
                     if hostType==2 and AGNType==4:
                         if param[i]==4 or param[i]==13:
                             flat_samples_raw_corner[:,k]=90-(flat_samples_raw_corner[:,k])       
                         
                     k=k+1

              if hostType==1 and AGNType==1 and polar=='no':
                  labels_corner=[r'\tau_{v}^s',r'\psi^s',r'\tau^s',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'r_2/r_1',r'\tau_{uv}',r'\theta_o',r'\theta_i']    
                     
              if hostType==1 and AGNType==1 and polar=='yes':
                  labels_corner=[r'\tau_{v}^s',r'\psi^s',r'\tau^s',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'r_2/r_1',r'\tau_{uv}',r'\theta_o',r'\theta_i',r'T_p']     
                  


              elif hostType==1 and AGNType==2 and polar=='no' or hostType==1 and AGNType==3 and polar=='no':
                  labels_corner=[r'\tau_{v}^s',r'\psi^s',r'\tau^s',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'\theta_o',r'r_2/r_1',r'\tau_{9.7\mu m}',r'\theta_i'] 
                  
              elif hostType==1 and AGNType==2 and polar=='yes' or hostType==1 and AGNType==3 and polar=='yes':
                  labels_corner=[r'\tau_{v}^s',r'\psi^s',r'\tau^s',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'\theta_o',r'r_2/r_1',r'\tau_{9.7\mu m}',r'\theta_i',r'T_p'] 
                  
                  
                  
              elif hostType==1 and AGNType==4 and polar=='no':
                  labels_corner=[r'\tau_{v}^s',r'\psi^s',r'\tau^s',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'V_c',r'A_c',r'A_d',r'theta']  
                  
              elif hostType==1 and AGNType==4 and polar=='yes':
                  labels_corner=[r'\tau_{v}^s',r'\psi^s',r'\tau^s',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'V_c',r'A_c',r'A_d',r'theta',r'T_p']                    
                  
                  
                  
              elif hostType==2 and AGNType==1 and polar=='no':                  
                  labels_corner=[r'\tau_{v}^d',r'\psi^d',r'\tau^d',r'\theta_d',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'r_2/r_1',r'\tau_{uv}',r'\theta_o',r'\theta_i'] 
                  
              elif hostType==2 and AGNType==1 and polar=='yes':                  
                  labels_corner=[r'\tau_{v}^d',r'\psi^d',r'\tau^d',r'\theta_d',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'r_2/r_1',r'\tau_{uv}',r'\theta_o',r'\theta_i',r'T_p'] 
                  
                              
                  
              elif hostType==2 and AGNType==2 and polar=='no' or hostType==2 and AGNType==3 and polar=='no':
                  labels_corner=[r'tau_{v}^d',r'\psi^d',r'\tau^d',r'\theta_d',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'theta_o',r'r_2/r_1',r'\tau_{9.7\mu m}',r'\theta_i'] 
                  
              elif hostType==2 and AGNType==2 and polar=='yes' or hostType==2 and AGNType==3 and polar=='yes':
                  labels_corner=[r'tau_{v}^d',r'\psi^d',r'\tau^d',r'\theta_d',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'theta_o',r'r_2/r_1',r'\tau_{9.7\mu m}',r'\theta_i',r'T_p'] 
                  
                  

              elif hostType==2 and AGNType==4 and polar=='no':
                  labels_corner=[r'tau_{v}^d',r'\psi^d',r'\tau^d',r'\theta_d',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'V_c',r'A_c',r'A_d',r'\theta']     
                  
              elif hostType==2 and AGNType==4 and polar=='yes':
                  labels_corner=[r'tau_{v}^d',r'\psi^d',r'\tau^d',r'\theta_d',r'\tau_v',r't_{\rm *}',r'\tau_{\rm *}',
                                 r'V_c',r'A_c',r'A_d',r'\theta',r'T_p']                       
                  
              # fig = corner.corner(
              #    10.**flat_samples_raw_corner, range=10.**host_corner, labels=labels_corner
              # )

              # plt.ioff()
              
              samples = MCSamples(samples=flat_samples_raw_corner,names = labels_corner, labels = labels_corner)
              
              g = plots.get_subplot_plotter()
              g.settings.axes_labelsize=28
              g.settings.axes_fontsize=24
              g.triangle_plot(samples, filled=True)  
              
              plt.ioff()
              
              plt.savefig('./'+data_file+'_results/'+name+'_'+XXX+'_corner_plot.png')   
    # 
    #      
            inds = np.random.randint(len(flat_samples_raw), size=100)
    #     
    #    Filter out bad flat_samples with high chi^2
    #
            chi2=[]
            for ind in inds:
              xchi2=chi_squared_host(flat_samples_raw[ind],ww,ff,ee,flag,models_fnu_full,hostType,AGNType,wavesb)
              chi2.append(xchi2)
    #        
            gg=chi2<=(np.min(chi2)+10.)
    #
            flat_samples=flat_samples_raw[inds[gg]]
    #
    # save to npz file  
    # 
            num_of_free_pars = ndim - nfixed 
            np.savez(filename2,flat_samples=flat_samples,num_of_free_pars=num_of_free_pars)

            np.savez('./'+data_file+'_results/' + name + '_' +  XXX + '_keywords.npz',data_file=data_file,metallicity=metallicity,host_geometry=host_geometry,
                  host_gal=host_gal,starburst_gal=starburst_gal,AGN_model=AGN_model,AGN_gal=AGN_gal,
                  polar=polar,num_of_runs=num_of_runs)
            
            inds = np.random.randint(len(flat_samples), size=100)   
    # 
    # 
            if hostType==1:
                b_fit_par=[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
                
            if hostType==2:
                b_fit_par=[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
            
            file2 = open('./'+data_file+'_results/' + name + '_' + XXX + '_pars' + '.txt', 'w', encoding='utf-8')
            print (name, file=file2)
            print (' ',  file=file2)  
    
            for i in range(ndim):
    #           
              mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    #          
              mcmcp = np.percentile(10.**flat_samples[:, i], [16, 50, 84])
    # 
              q  = np.diff(mcmc)
              qp = np.diff(mcmcp)
    #
              b_fit_par[i]=mcmc[1]
              print (labels[i],' : ', "{:.3f}".format(mcmcp[1]), "{:.3f}".format(qp[0]), "{:.3f}".format(qp[1]),file=file2)
              
            min_chi_squared=chi_squared_host(b_fit_par,ww,ff,ee,flag,models_fnu_full,hostType,AGNType,wavesb)
            print('Min_chi_squared : ',min_chi_squared,file=file2)
            file2.close()

            for ind in inds:
              sample = flat_samples[ind]
              print(sample)
              if hostType==1:

                  model1=synthesis_routine_SMART(10.**sample[4],10.**sample[5],10.**sample[6],10.**sample[7],
                         10.**sample[8],10.**sample[9],10.**sample[10],10.**sample[11],10.**sample[12],10.**sample[13],10.**sample[14],AGNType)

                  model2 = synthesis_routine_host_SMART(10.**sample[0],10.**sample[1],10.**sample[2],10.**sample[3],1.,
                         models_fnu_full,hostType,wavesb)
                  model = model1 + model2
                   
              if hostType==2:
                  model1=synthesis_routine_SMART(10.**sample[5],10.**sample[6],10.**sample[7],10.**sample[8],
                         10.**sample[9],10.**sample[10],10.**sample[11],10.**sample[12],10.**sample[13],10.**sample[14],10.**sample[15],AGNType)

                  model2=synthesis_routine_host_SMART(10.**sample[0],10.**sample[1],10.**sample[2],10.**sample[3],10.**sample[4],
                         models_fnu_full,hostType,wavesb) 

                  model = model1 + model2

    #
    #    Plot best fit model (total)
    # 
            frequency=3.e14/model[0]  
    #
    #    Plot data
    # 
            kgd=flag==int(0)
            plt.errorbar(ww[kgd], ff[kgd]*freq[kgd], yerr=ee[kgd]*freq[kgd], fmt=".k", capsize=0, linewidth=3,markersize=12)
    #
    #    Calculate total, sph, disc, starb, agn models
    #
            if hostType==1:
                
                mmodel_tot1=synthesis_routine_SMART(10.**b_fit_par[4],10.**b_fit_par[5],10.**b_fit_par[6],10.**b_fit_par[7],
                    10.**b_fit_par[8],10.**b_fit_par[9],10.**b_fit_par[10],10.**b_fit_par[11],10.**b_fit_par[12],10.**b_fit_par[13],10.**b_fit_par[14],AGNType)

                mmodel_tot2=synthesis_routine_host_SMART(10.**b_fit_par[0],10.**b_fit_par[1],10.**b_fit_par[2],10.**b_fit_par[3],1.,
                        models_fnu_full,hostType,wavesb)
        
                mmodel_agn=synthesis_routine_SMART(1.e-30,10.**b_fit_par[5],10.**b_fit_par[6],10.**b_fit_par[7],
                    10.**b_fit_par[8],10.**b_fit_par[9],10.**b_fit_par[10],10.**b_fit_par[11],10.**b_fit_par[12],1.e-30,10.**b_fit_par[14],AGNType)

                mmodel_sb=synthesis_routine_SMART(10.**b_fit_par[4],10.**b_fit_par[5],10.**b_fit_par[6],10.**b_fit_par[7],
                        1.e-30,10.**b_fit_par[9],10.**b_fit_par[10],10.**b_fit_par[11],10.**b_fit_par[12],1.e-30,10.**b_fit_par[14],AGNType)

                mmodel_sph=mmodel_tot2
                
                mmodel_pol=synthesis_routine_SMART(1.e-30,10.**b_fit_par[5],10.**b_fit_par[6],10.**b_fit_par[7],
                        1.e-30,10.**b_fit_par[9],10.**b_fit_par[10],10.**b_fit_par[11],10.**b_fit_par[12],10.**b_fit_par[13],10.**b_fit_par[14],AGNType)
                            
            if hostType==2:
                mmodel_tot1=synthesis_routine_SMART(10.**b_fit_par[5],10.**b_fit_par[6],10.**b_fit_par[7],10.**b_fit_par[8],
                   10.**b_fit_par[9],10.**b_fit_par[10],10.**b_fit_par[11],10.**b_fit_par[12],10.**b_fit_par[13],10.**b_fit_par[14],10.**b_fit_par[15],AGNType)

                mmodel_tot2=synthesis_routine_host_SMART(10.**b_fit_par[0],10.**b_fit_par[1],10.**b_fit_par[2],10.**b_fit_par[3],10.**b_fit_par[4],
                          models_fnu_full,hostType,wavesb)
        
                mmodel_agn=synthesis_routine_SMART(1.e-30,10.**b_fit_par[6],10.**b_fit_par[7],10.**b_fit_par[8],
                    10.**b_fit_par[9],10.**b_fit_par[10],10.**b_fit_par[11],10.**b_fit_par[12],10.**b_fit_par[13],1.e-30,10.**b_fit_par[15],AGNType)
        
                mmodel_sb=synthesis_routine_SMART(10.**b_fit_par[5],10.**b_fit_par[6],10.**b_fit_par[7],10.**b_fit_par[8],
                          1.e-30,10.**b_fit_par[10],10.**b_fit_par[11],10.**b_fit_par[12],10.**b_fit_par[13],1.e-30,10.**b_fit_par[15],AGNType)
                
                mmodel_disc=mmodel_tot2
                
                mmodel_pol=synthesis_routine_SMART(1.e-30,10.**b_fit_par[6],10.**b_fit_par[7],10.**b_fit_par[8],
                      1.e-30,10.**b_fit_par[10],10.**b_fit_par[11],10.**b_fit_par[12],10.**b_fit_par[13],10.**b_fit_par[14],10.**b_fit_par[15],AGNType)       
    # 
    #    Plot model components
    #            
            rel_residual_plot='no'
            if 'rel_residual_plot' in kwargs:   
                rel_residual_plot=kwargs.get('rel_residual_plot')
                
            if rel_residual_plot=='no':
                  fig, (ax1) = plt.subplots(nrows=1,figsize=(10,7),sharex=True)
                  
            elif rel_residual_plot=='yes':
                  fig, (ax1,ax2) = plt.subplots(nrows=2,figsize=(10,7),sharex=True, gridspec_kw={'height_ratios': [5, 1], 'hspace': 0})
                 

            if AGN_gal=='yes':
                ax1.plot(mmodel_agn[0],mmodel_agn[1]*frequency,label="AGN torus",color="blue", linewidth=3)
            
            if starburst_gal=='yes':
                ax1.plot(mmodel_sb[0],mmodel_sb[1]*frequency,label="Starburst",color="red", linewidth=3)
            
            if host_gal=='yes':
    
                if hostType==1:
                    ax1.plot(mmodel_sph[0],mmodel_sph[1]*frequency,label="Spheroidal",color="orange",linewidth=3)           
            
                if hostType==2:
                    ax1.plot(mmodel_disc[0],mmodel_disc[1]*frequency,label="Disc",color="green", linewidth=3)

            if polar=='yes':
                ax1.plot(mmodel_pol[0],mmodel_pol[1]*frequency,label="Polar dust",color="magenta", linewidth=3)            
            
            ax1.plot(mmodel_tot1[0],(mmodel_tot1[1]+mmodel_tot2[1])*frequency,label="Total",color="black", linewidth=3)
            ax1.legend(fontsize=14)
            
            ax1.set_xlim([0.1, 1000.])
            if 'x_axis' in kwargs:    
                ax1.set_xlim(xx_axis[0],xx_axis[1])
            
            ax1.set_ylim(1.e10*10.**scale[0],1.e14*10.**scale[0])
            if 'y_axis' in kwargs:    
                ax1.set_ylim(yy_axis[0],yy_axis[1])
            
            ax1.set_xlabel('Rest $\lambda$ ($\mu m$)',size=18)
            ax1.set_ylabel(r'$\nu~S_\nu$ (Jy Hz)',size=18)
            ax1.tick_params('x',labelsize=18)
            ax1.tick_params('y',labelsize=18)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            kgd=flag==int(0)
            ax1.errorbar(ww[kgd], ff[kgd]*freq[kgd], yerr=ee[kgd]*freq[kgd], fmt=".k", capsize=0, markersize=12)
            k=flag==int(1)
            www=ww[k]
            freqq=freq[k]
            fff=ff[k]
            eee=ee[k]
            ax1.errorbar(www, (fff+3.*eee)*freqq, yerr=eee*freqq, fmt=".k", uplims=True, capsize=0, markersize=12, linewidth=3)
            ax1.set_title(name,fontsize=18)           
#      
            if rel_residual_plot=='yes':
#
                model_func=interpolate.interp1d(mmodel_tot1[0],mmodel_tot1[1] + mmodel_tot2[1])
#
                model=model_func(ww[kgd])
#
                ax2.plot(mmodel_tot1[0],np.zeros(len(mmodel_tot1[0])), linewidth=3, linestyle = 'dashed', color = 'black')
                ax2.errorbar(ww[kgd], (ff[kgd]-model)/ff[kgd], yerr=((model**2/(ff[kgd]**4))*(ee[kgd]**2)+((0.15*model)**2)/(ff[kgd]**2))**.5, fmt=".k", capsize=0,markersize=6)
                ax2.set_xlim([0.1, 1000.])
                ax2.set_ylim([-1., 1.])
                ax2.set_xlabel('Rest $\lambda$ ($\mu m$)',size=18)
                ax2.set_ylabel('(Obs-Mod)/Obs')
                ax2.tick_params('x',labelsize=18)
#            
            plt.savefig('./'+data_file+'_results/'+name+'_'+XXX+'.png')
#        
            run_id = run_id + 1
        
            