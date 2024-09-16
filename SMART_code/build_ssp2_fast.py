import numpy as np
# 
def restore(file):
    """
    Read data saved with save function.
    Usage: datos = restore('misdatos.pypic')
    """
    import os
    from os.path import dirname, join as pjoin
    from scipy.io import readsav
    filepath = os.getcwd()
    filepath = filepath +'\starburst_models0'
    sav_fname = pjoin(filepath, file)
    sav_data = readsav(sav_fname)
    return(sav_data['new_models'])
      
def build_ssp2_fast(*argv,**kwargs):
# +
# NAME:
# 	build_ssp2_fast
# 
# 
# PROCEDURE:
# Combines the dust free Bruzual & Charlot SSP with the
# one computed by Efstathiou, Rowan-Robinson & Siebenmorgen (2000)
# depending on f and t_m (see description in Efstathiou  &
# Rowan-Robinson 2003) to produce structure models. 
# 
#                                                                        
# MODIFICATION HISTORY:
# Written by:	Andreas Efstathiou, February 2003, and translated to Python by Charalambia Varnava, Dec 2020
# -   

    np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
    if len(argv)<2:
        print('USAGE:build_ssp2_fast,f=f,t_m=t_m,models,modelsfile=modelsfile,tau_v=tau_v')
        return 
# 
    f=float(argv[0])
    t_m=float(argv[1])
#        
    tau_v=argv[2]
# 
    tau_vv=round(tau_v/25.)*25.
    if tau_vv == 10.:
        new_models=restore('bc98_starb10_leak=0_t=0.dat')
    elif tau_vv == 20.:
        new_models=restore('bc98_starb20_leak=0_t=0.dat')
    elif tau_vv == 25.:
        new_models=restore('bc98_starb25_leak=0_t=0.dat')
    elif tau_vv ==  30.:
        new_models=restore('bc98_starb30_leak=0_t=0.dat')
    elif tau_vv == 40.:
        new_models=restore('bc98_starb40_leak=0_t=0.dat')
    elif tau_vv == 50.:
        new_models=restore('bc98_starb50_leak=0_t=0.dat')
    elif tau_vv == 60.:
        new_models=restore('bc98_starb60_leak=0_t=0.dat')
    elif tau_vv == 70.:
        new_models=restore('bc98_starb70_leak=0_t=0.dat')
    elif tau_vv == 75.:
        new_models=restore('bc98_starb75_leak=0_t=0.dat')
    elif tau_vv == 80.:
        new_models=restore('bc98_starb80_leak=0_t=0.dat')
    elif tau_vv == 90.:
        new_models=restore('bc98_starb90_leak=0_t=0.dat')
    elif tau_vv == 100.:
        new_models=restore('bc98_starb100_leak=0_t=0.dat')
    elif tau_vv == 120.:
        new_models=restore('bc98_starb125_leak=0_t=0.dat')
    elif tau_vv == 125.:
        new_models=restore('bc98_starb125_leak=0_t=0.dat')
    elif tau_vv == 150.:
        new_models=restore('bc98_starb150_leak=0_t=0.dat')
    elif tau_vv == 175.:
        new_models=restore('bc98_starb175_leak=0_t=0.dat')
    elif tau_vv == 200.:
        new_models=restore('bc98_starb200_leak=0_t=0.dat')
    elif tau_vv == 225.:
        new_models=restore('bc98_starb225_leak=0_t=0.dat')
    elif tau_vv == 250.:
        new_models=restore('bc98_starb225_leak=0_t=0.dat')
# 
    else:
        print('tau_vv should be one of 10, 20, 30, 40, 50, 60, 70, 80, 90 or 100')
        return
# 
    models0=new_models
#     
    new_models=restore('bc98_starb1_leak=1.0_t=0.dat')

    models1=new_models
# 
    change=models0.age >= t_m

    models=models0
    for i in range(len(change)):
        if change[i]==True:
            models[i].spectrum.FLAMBDA=(1.-f)*models0[i].spectrum.FLAMBDA+f*models1[i].spectrum.FLAMBDA   
       
    models.extra='bc95 initial GMC t_v='+str(tau_v)+', f='+str(f)+', t_m='+str(t_m)
    

    return models       