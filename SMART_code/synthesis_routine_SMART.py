import numpy as np
#

import scipy.io
import os
import math

#

from scipy import interpolate

import time
start_time = time.time()
    
filename = 'starb_array12_low_res.npz'  
data_starburst = np.load(filename)

global starb_models, starb_models_fnu, wavesb, x1_tau_v, x2_age, x3_t_e, x4_t_m

starb_models     = data_starburst['starb_models']
starb_models_fnu = data_starburst['starb_models_fnu']
wavesb           = data_starburst['wavesb']
x1_tau_v         = data_starburst['x1_tau_v']
x2_age           = data_starburst['x2_age']
x3_t_e           = data_starburst['x3_t_e']
x4_t_m           = data_starburst['x4_t_m']

#   Define global spheroidal and disc variables

redshift=0.1
metal='0.00130000'

if redshift < 0.29:
   metal='0.0080000'
   zstr='0.1000'
   zdstr='0.000'

filename = './spheroid_models_5/spheroid_array1_z='+zstr[0:3]+'0000_met='+metal+'_X.npz'  
data_sph = np.load(filename)

global wavez,xsph1_tv,xsph2_psi,xsph3_tau,xsph4_iview

wavez               = data_sph['wavez']
xsph1_tv            = data_sph['xsph1_tv']
xsph2_psi           = data_sph['xsph2_psi']
xsph3_tau           = data_sph['xsph3_tau']
xsph4_iview         = data_sph['xsph4_iview']

#  Discs

filename = './disc_models_1/disc_array_z=0.01_v8X.npz'  

data_disc = np.load(filename)

global wavex,x1_tv,x2_psi,x3_tau,x4_iview  

wavex             = data_disc['wavex']
x1_tv             = data_disc['x1_tv']
x2_psi            = data_disc['x2_psi']
x3_tau            = data_disc['x3_tau']
x4_iview          = data_disc['x4_iview']

#
#  Polar dust 
#

filename = 'polar_dust_lib1.npz' 
data_polar = np.load(filename)

global polar_dust_lib,wpol,x1_polt 

polar_dust_lib  = data_polar['polar_dust_lib']
wpol            = data_polar['wpol']
x1_polt         = data_polar['x1_polt']
    

def select_library(argv,**kwargs):
#
  redshift = argv[0]
  hostType = argv[1]
  wwsorted = argv[2]
  metallicity = argv[3]
#
  index='4'
#
  if metallicity==0.02:
     index='5'
  if metallicity==0.0080:
     index='4'
  if metallicity==0.0013:
     index='3'
#
  zstr=str(round((redshift/0.2))*0.2)
  zdstr=str(round((redshift/0.4))*0.4)
#
  metal='0.00130000'
#
  if redshift < 0.29:
     metal='0.0080000'
     zstr='0.1000'
     zdstr='0.000'
#     
  if hostType==1:   
#
      filename = './spheroid_models_5/spheroid_array1_z='+zstr[0:3]+'0000_met='+metal+'_X.npz'  
      data_sph = np.load(filename)
#
      spheroid_models = data_sph['spheroid_models']
      models_fnu_full = data_sph['spheroid_models_fnu']
#      
      n1=len(xsph1_tv)
      n2=len(xsph2_psi)
      n3=len(xsph3_tau)
      n4=len(xsph4_iview)
#
#  
      ndata=len(wwsorted)
#
#      print (n1,n2,n3,n4)
#
      models_fnu_red=np.zeros([ndata,n4,n3,n2,n1])
#
      for i1 in range(n1):
#
       for i2 in range(n2):
# 
         for i3 in range(n3):
#
          for i4 in range(n4):
#
#           print (i1,i2,i3,i4)
           model=models_fnu_full[:,i4,i3,i2,i1]
           func=interpolate.interp1d(wavesb,model)           
           model_red=func(wwsorted)
           models_fnu_red[:,i4,i3,i2,i1]=model_red
#   
  if hostType==2:
      filename = './disc_models_1/disc_array_z=0.01_v8X.npz'  
#      filename = './disc_models_2/disc_array_10YY_z='+zdstr[0:3]+'_PAH=2_cygnus+'+index+'.npz'  
#
      data_disc = np.load(filename)
#
      cirrus_models = data_disc['cirrus_models']
      models_fnu_full = data_disc['cirrus_models_fnu']
#
      n1=len(x1_tv)
      n2=len(x2_psi)
      n3=len(x3_tau)
      n4=len(x4_iview)
#
      ndata=len(wwsorted)
#
#      print (n1,n2,n3,n4)
#
      models_fnu_red=np.zeros([ndata,n4,n3,n2,n1])
#
      for i1 in range(n1):
#
        for i2 in range(n2):
#
          for i3 in range(n3):
#
             for i4 in range(n4):
#
#                print (i1,i2,i3,i4)   
                model=models_fnu_full[:,i4,i3,i2,i1]
                func=interpolate.interp1d(wavesb,model)           
                model_red=func(wwsorted)
                models_fnu_red[:,i4,i3,i2,i1]=model_red

  return models_fnu_red, models_fnu_full


# CYGNUS AGN model

filename = './AGN_models/tapered_discs_5d_ir1.npz'  
data_AGN_CYGNUS = np.load(filename)
     
global tapered_discs, wavey, x1_tau_uv, x2_r2tor1, x3_theta_1, x4_theta_v, aniso_factor, tau_array, ww, qtot
  
tapered_discs = data_AGN_CYGNUS['tapered_discs']
wavey         = data_AGN_CYGNUS['wavey']
x1_tau_uv     = data_AGN_CYGNUS['x1_tau_uv']
x2_r2tor1     = data_AGN_CYGNUS['x2_r2tor1']
x3_theta_1    = data_AGN_CYGNUS['x3_theta_1']
x4_theta_v    = data_AGN_CYGNUS['x4_theta_v']
aniso_factor  = data_AGN_CYGNUS['aniso_factor']
tau_array     = data_AGN_CYGNUS['tau_array']
ww            = data_AGN_CYGNUS['ww']
qtot          = data_AGN_CYGNUS['qtot']     
 

# Fritz AGN model

filename = './AGN_models/fr06_discs_bs=0.0_qs=4_ir1_fnu.npz'
data_AGN_Fritz = np.load(filename)

global fr06_discs_r, aniso_factor_fr06, wavefr06, x1_ct, x2_rm, x3_ta, x4_thfr06    
    
fr06_discs_r      = data_AGN_Fritz['fr06_discs_r']
aniso_factor_fr06 = data_AGN_Fritz['aniso_factor_fr06'] 
wavefr06          = data_AGN_Fritz['wavefr06']
x1_ct             = data_AGN_Fritz['x1_ct']
x2_rm             = data_AGN_Fritz['x2_rm']
x3_ta             = data_AGN_Fritz['x3_ta']
x4_thfr06         = data_AGN_Fritz['x4_thfr06']    


# SKIRTOR AGN model

filename = './AGN_models/st16_discs_ps=1.0_qs=1.0_ir1_fnu.npz'
data_AGN_SKIRTOR = np.load(filename)

global st16_discs_r, aniso_factor_st16, wavest16, x1_oa, x2_rr, x3_tt, x4_thst16
       
st16_discs_r      = data_AGN_SKIRTOR['st16_discs_r']
aniso_factor_st16 = data_AGN_SKIRTOR['aniso_factor_st16'] 
wavest16          = data_AGN_SKIRTOR['wavest16']
x1_oa             = data_AGN_SKIRTOR['x1_oa']
x2_rr             = data_AGN_SKIRTOR['x2_rr']
x3_tt             = data_AGN_SKIRTOR['x3_tt']
x4_thst16         = data_AGN_SKIRTOR['x4_thst16']


#  Siebenmorgen AGN model

filename = './AGN_models/s15_discs_1000_ir1_fnu.npz'
data_AGN_Siebenmorgen = np.load(filename)

global s15_discs_r, aniso_factor_s15, waves15, x1_vc, x2_ac, x3_ad, x4_th
       
s15_discs_r      = data_AGN_Siebenmorgen['s15_discs_r']
aniso_factor_s15 = data_AGN_Siebenmorgen['aniso_factor_s15'] 
waves15          = data_AGN_Siebenmorgen['waves15']
x1_vc            = data_AGN_Siebenmorgen['x1_vc']
x2_ac            = data_AGN_Siebenmorgen['x2_ac']
x3_ad            = data_AGN_Siebenmorgen['x3_ad']
x4_th            = data_AGN_Siebenmorgen['x4_th']
    

def polar_dust_fnu2(argv,**kwargs):

#+
# NAME:
#	polar_dust_fnu2
#
# PURPOSE:
#
# CATEGORY:
#
# CALLING SEQUENCE:
#       polar_dust_fnu2, w, s, temp
#	
# INPUTS:
# 
# Temp      =  temperature
#
# OPTIONAL INPUTS:
#	
# KEYWORD PARAMETERS:
#
#   
# OUTPUTS:
#
#     w      =  wavelength in microns
#     s      =  nuSnu
# 
# OPTIONAL OUTPUTS:
#
# COMMON BLOCKS:
#	None
#
# SIDE EFFECTS:
#	None known
#
# RESTRICTIONS:
#	None known
#
# PROCEDURE:
#
#     Calculate the spectrum of polar dust, given its temperature
#
# EXAMPLE:
#                                                                      
# MODIFICATION HISTORY:
# 	Written by:	Andreas Efstathiou, May 2023
#-      
     np.seterr(all='ignore') #ignore over/underflow 0 errors
#  
#
     Temp=argv[0]
#
     w=wpol
#
     ixx1 = [k >= Temp for k in x1_polt]
#    
     ixx1_index = [i for i, x in enumerate(ixx1) if x]
#
     ix1=ixx1_index[0]-1
#     
     s0=polar_dust_lib[:,ix1]
     s1=polar_dust_lib[:,ix1+1]
#
     dtemp=x1_polt[ix1+1] - x1_polt[ix1]
     s= (s0 * (x1_polt[ix1+1] - Temp) + s1 * (Temp - x1_polt[ix1]))/dtemp

     return w, s


def galaxy_spheroid_fnu(argv,**kwargs):
# 
# +
# NAME:
#      galaxy_spheroid_fnu	
# 
# PURPOSE:
# 
# CATEGORY:
# 
# CALLING SEQUENCE:
#
#       galaxy_spheroid_fnu, w, s, tvv, psi, cirr_tau, iview
# 	
# INPUTS:
# 
# tvv      =  optical depth of the spheroid
# psi      =  intensity of starlight
# cirr_tau =  e-folding time of SFR
# iview    =  viewing angle index (measured from the
#             equator)
# 
# OPTIONAL INPUTS:
# 	
# KEYWORD PARAMETERS:
# 
# 
# OUTPUTS:
# 
#     w      =  wavelength in microns
#     s      =  nuSnu
# 
# OPTIONAL OUTPUTS:
# 
# COMMON BLOCKS:
# 	None
# 
# SIDE EFFECTS:
# 	None known
# 
# RESTRICTIONS:
# 	None known
# 
# PROCEDURE:
#
# Calculate the spectrum of a spheroid galaxy by doing
# multi-linear interpolation on a pre-computed grid
# of models
#
# EXAMPLE:
#                                                                        
# MODIFICATION HISTORY:
# Written originally in IDL by:	Andreas Efstathiou, February 2003
# -      

      np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
      w=argv[5]
#
      tvv         = float(argv[0])
      psi         = argv[1]
      cirr_tau    = argv[2]
      iview       = argv[3]
      spheroid_models_fnu=argv[4]
#
#
      dtv=xsph1_tv[1]-xsph1_tv[0]
      dcirr_tau=xsph3_tau[1]/xsph3_tau[0]
      dpsi=xsph2_psi[1]-xsph2_psi[0]
      ix1=np.int(tvv/dtv)
      ix2=np.int((psi-1.)/dpsi)
      ix3=np.int(math.log10(cirr_tau/xsph3_tau[0])/math.log10(dcirr_tau))
      ix4=np.int(iview) - 1
#
#
      dx1=xsph1_tv[ix1+1]-xsph1_tv[ix1]
      dx2=xsph2_psi[ix2+1]-xsph2_psi[ix2]
      dx3=xsph3_tau[ix3+1]-xsph3_tau[ix3]
      dx4=xsph4_iview[ix4+1]-xsph4_iview[ix4]
      dv=dx1*dx2*dx3
#
#
      Ea=(xsph1_tv[ix1+1]-tvv)*(xsph2_psi[ix2+1]-psi)*(cirr_tau-xsph3_tau[ix3])/dv
      Eb=(xsph1_tv[ix1+1]-tvv)*(psi-xsph2_psi[ix2])*(cirr_tau-xsph3_tau[ix3])/dv
      Ec=(tvv-xsph1_tv[ix1])*(xsph2_psi[ix2+1]- psi)*(cirr_tau-xsph3_tau[ix3])/dv
      Ed=(tvv-xsph1_tv[ix1])*(psi-xsph2_psi[ix2])*(cirr_tau-xsph3_tau[ix3])/dv
      Ee=(xsph1_tv[ix1+1]-tvv)*(xsph2_psi[ix2+1]-psi)*(xsph3_tau[ix3+1]-cirr_tau)/dv
      Ef=(xsph1_tv[ix1+1]-tvv)*(psi-xsph2_psi[ix2])*(xsph3_tau[ix3+1]-cirr_tau)/dv
      Eg=(tvv-xsph1_tv[ix1])*(xsph2_psi[ix2+1]-psi)*(xsph3_tau[ix3+1]-cirr_tau)/dv
      Eh=(tvv-xsph1_tv[ix1])*(psi-xsph2_psi[ix2])*(xsph3_tau[ix3+1]-cirr_tau)/dv
#
      z0=spheroid_models_fnu[:,ix4,ix3+1,ix2,ix1]  
      z1=spheroid_models_fnu[:,ix4,ix3+1,ix2+1,ix1]  
      z2=spheroid_models_fnu[:,ix4,ix3+1,ix2,ix1+1]  
      z3=spheroid_models_fnu[:,ix4,ix3+1,ix2+1,ix1+1]  
      z4=spheroid_models_fnu[:,ix4,ix3,ix2,ix1]   
      z5=spheroid_models_fnu[:,ix4,ix3,ix2,ix1]      
      z6=spheroid_models_fnu[:,ix4,ix3,ix2,ix1+1]  
      z7=spheroid_models_fnu[:,ix4,ix3,ix2+1,ix1+1]   
#
      s=z0*Ea + z1*Eb + z2*Ec + z3*Ed + z4*Ee + z5*Ef + z6*Eg + z7*Eh

      s = s   - np.log10(w) 
      s=s-np.amax(s)
      s=10.**s

      return   w, s   
  
def galaxy_disc_fnu(argv,**kwargs):
# 
# +
# NAME:
# 	galaxy_disc_fnu
# 
# PURPOSE:
# 
# CATEGORY:
# 
# CALLING SEQUENCE:
# 
# galaxy_disc_fnu, w, s, tv, psi, tau, iview
# 	
# INPUTS:
# 
# tv       =  equatorial optical depth of the disc
# psi      =  intensity of starlight
# cirr_tau =  e-folding time of SFR
# iview    =  viewing angle index (measured from the
#             equator)
# 
# OPTIONAL INPUTS:
# 	
# KEYWORD PARAMETERS:
# 
# 
# OUTPUTS:
# 
#     w      =  wavelength in microns
#     s      =  nuSnu
# 
# OPTIONAL OUTPUTS:
# 
# COMMON BLOCKS:
# 	None
# 
# SIDE EFFECTS:
# 	None known
# 
# RESTRICTIONS:
# 	None known
# 
# PROCEDURE:
# Calculate the spectrum of a disc galaxy by doing
# multi-linear interpolation on a pre-computed grid
# of models
# 
# EXAMPLE:
#                                                                      
# MODIFICATION HISTORY:
# Written by:	Andreas Efstathiou, June 2014, and translated to Python by Charalambia Varnava, September 2022
# -        

     np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
     w=argv[5]
#
     tv          = float(argv[0])
     psi         = argv[1]
     cirr_tau    = argv[2]
     iview       = argv[3]
     cirrus_models_fnu=argv[4]
#
#
     dtv=x1_tv[1]-x1_tv[0]
     dpsi=x2_psi[1]-x2_psi[0]
     dcirr_tau=x3_tau[1]/x3_tau[0]
# 
     ix1=np.int((tv-x1_tv[0])/dtv)
     ix2=np.int((psi-x2_psi[0])/dpsi)
     ix3=np.int(math.log10(cirr_tau/x3_tau[0])/math.log10(dcirr_tau))
     ixx4 = [n >= iview for n in x4_iview]

#   
     ixx4_index = [i for i, x in enumerate(ixx4) if x]
     ix4=ixx4_index[0]-1
#
#
     dx1=x1_tv[ix1+1]-x1_tv[ix1]
     dx2=x2_psi[ix2+1]-x2_psi[ix2]
     dx3=x3_tau[ix3+1]-x3_tau[ix3]
     dx4=x4_iview[1]-x4_iview[0]
# 
# 
     dv=dx1*dx2*dx3
#
#
     Ea=(x1_tv[ix1+1]-tv)*(x2_psi[ix2+1]-psi)*(cirr_tau-x3_tau[ix3])/dv
     Eb=(x1_tv[ix1+1]-tv)*(psi-x2_psi[ix2])*(cirr_tau-x3_tau[ix3])/dv
     Ec=(tv-x1_tv[ix1])*(x2_psi[ix2+1]- psi)*(cirr_tau-x3_tau[ix3])/dv
     Ed=(tv-x1_tv[ix1])*(psi-x2_psi[ix2])*(cirr_tau-x3_tau[ix3])/dv
     Ee=(x1_tv[ix1+1]-tv)*(x2_psi[ix2+1]-psi)*(x3_tau[ix3+1]-cirr_tau)/dv
     Ef=(x1_tv[ix1+1]-tv)*(psi-x2_psi[ix2])*(x3_tau[ix3+1]-cirr_tau)/dv
     Eg=(tv-x1_tv[ix1])*(x2_psi[ix2+1]-psi)*(x3_tau[ix3+1]-cirr_tau)/dv
     Eh=(tv-x1_tv[ix1])*(psi-x2_psi[ix2])*(x3_tau[ix3+1]-cirr_tau)/dv
#
     z0=cirrus_models_fnu[:,ix4,ix3+1,ix2,ix1]  
     z1=cirrus_models_fnu[:,ix4,ix3+1,ix2+1,ix1]  
     z2=cirrus_models_fnu[:,ix4,ix3+1,ix2,ix1+1]  
     z3=cirrus_models_fnu[:,ix4,ix3+1,ix2+1,ix1+1]  
     z4=cirrus_models_fnu[:,ix4,ix3,ix2,ix1]   
     z5=cirrus_models_fnu[:,ix4,ix3,ix2,ix1]      
     z6=cirrus_models_fnu[:,ix4,ix3,ix2,ix1+1]  
     z7=cirrus_models_fnu[:,ix4,ix3,ix2+1,ix1+1]   
#
     s=z0*Ea + z1*Eb + z2*Ec + z3*Ed + z4*Ee + z5*Ef + z6*Eg + z7*Eh
#
     s = s   - np.log10(w) 
     s=s-np.amax(s)
#
     s=10.**s
#
     return   w, s   

def galaxy_starburst2_fnu(argv,**kwargs):
# 
# +
# NAME:
#      galaxy_starburst2_fnu
# 
# PURPOSE:
# 
# CATEGORY:
# 
# CALLING SEQUENCE:
#
#       galaxy_starburst2_fnu, w, s, tau_v, age, t_e, t_m	
# 	
# INPUTS:
# 
# tau_v       =  initial optical depth of GMCs
# age         =  age of starburst
# t_e         =  e-folding time of SFR
# t_m         =  time after which GMCs become non-spherical
# 
# OPTIONAL INPUTS:
# 	
# KEYWORD PARAMETERS:
# 
# 
# OUTPUTS:
# 
#     w      =  wavelength in microns
#     s      =  nuSnu
# 
# OPTIONAL OUTPUTS:
# 
# COMMON BLOCKS:
# 	None
# 
# SIDE EFFECTS:
# 	None known
# 
# RESTRICTIONS:
# 	None known
# 
# PROCEDURE:
#
# Calculate the spectrum of starburst by doing
# multi-linear interpolation on a pre-computed grid
# of models
#
# EXAMPLE:
#                                                                        
# MODIFICATION HISTORY:
# Written originally in IDL by:	Andreas Efstathiou, February 2003
# -      

     np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
     w=wavesb
#
     tau_v = float(argv[0])
     age   = argv[1]
     t_e   = argv[2]
     t_m   = argv[3]
#
#
     dtv=x1_tau_v[1]-x1_tau_v[0]
     ix1=np.int(tau_v/dtv) - 1
     dage=x2_age[1]-x2_age[0]
     ix2=np.int(age/dage)
     dsb_tau=x3_t_e[1]/x3_t_e[0]
     ix3=np.int(math.log10(t_e/x3_t_e[0])/math.log10(dsb_tau))
     dt_m=x4_t_m[1]/x4_t_m[0]
     ix4=np.int(math.log10(t_m/x4_t_m[0])/math.log10(dt_m))
#
#
     dx1=x1_tau_v[ix1+1]-x1_tau_v[ix1]
     dx2=x2_age[ix2+1]-x2_age[ix2]
     dx3=x3_t_e[ix3+1]-x3_t_e[ix3]
     dx4=x4_t_m[ix4+1]-x4_t_m[ix4]
     dv=dx1*dx2*dx3
#
#
     Ea=(x1_tau_v[ix1+1]-tau_v)*(x2_age[ix2+1]-age)*(t_e-x3_t_e[ix3])/dv
     Eb=(x1_tau_v[ix1+1]-tau_v)*(age-x2_age[ix2])*(t_e-x3_t_e[ix3])/dv
     Ec=(tau_v-x1_tau_v[ix1])*(x2_age[ix2+1]- age)*(t_e-x3_t_e[ix3])/dv
     Ed=(tau_v-x1_tau_v[ix1])*(age-x2_age[ix2])*(t_e-x3_t_e[ix3])/dv
     Ee=(x1_tau_v[ix1+1]-tau_v)*(x2_age[ix2+1]-age)*(x3_t_e[ix3+1]-t_e)/dv
     Ef=(x1_tau_v[ix1+1]-tau_v)*(age-x2_age[ix2])*(x3_t_e[ix3+1]-t_e)/dv
     Eg=(tau_v-x1_tau_v[ix1])*(x2_age[ix2+1]-age)*(x3_t_e[ix3+1]-t_e)/dv
     Eh=(tau_v-x1_tau_v[ix1])*(age-x2_age[ix2])*(x3_t_e[ix3+1]-t_e)/dv
#
     z0=starb_models_fnu[:,ix4,ix3+1,ix2,ix1]  
     z1=starb_models_fnu[:,ix4,ix3+1,ix2+1,ix1]  
     z2=starb_models_fnu[:,ix4,ix3+1,ix2,ix1+1]  
     z3=starb_models_fnu[:,ix4,ix3+1,ix2+1,ix1+1]  
     z4=starb_models_fnu[:,ix4,ix3,ix2,ix1]   
     z5=starb_models_fnu[:,ix4,ix3,ix2,ix1]      
     z6=starb_models_fnu[:,ix4,ix3,ix2,ix1+1]  
     z7=starb_models_fnu[:,ix4,ix3,ix2+1,ix1+1]   
#
     s=z0*Ea + z1*Eb + z2*Ec + z3*Ed + z4*Ee + z5*Ef + z6*Eg + z7*Eh

#     s = s - np.log10(w)
     s=s-np.amax(s)
     
     s=10.**s

     return   w, s   

def tapered_disc(argv,**kwargs):
# 
# +
# NAME:
# 	tapered_disc
# 
# PURPOSE:
# 
# CATEGORY:
# 
# CALLING SEQUENCE:
#
#       tapered_disc, w, s, tau_v, age, t_e, t_m	
# 	
# INPUTS:
# 
# tau_uv      =  equatorial UV optical depth of the torus
# r2tor1      =  ratio of outer to inner radius
# theta_1     =  opening angle of the torus
# theta_v     =  viewing angle (measured from the equator)
# 
# OPTIONAL INPUTS:
# 	
# KEYWORD PARAMETERS:
# 
# 
# OUTPUTS:
# 
#     w      =  wavelength in microns
#     s      =  nuSnu
#     aniso  =  anistropy correction factor (see Efstathiou 2006)    
# 
# OPTIONAL OUTPUTS:
# 
# COMMON BLOCKS:
# 	None
# 
# SIDE EFFECTS:
# 	None known
# 
# RESTRICTIONS:
# 	None known
# 
# PROCEDURE:
#
#            Calculate the spectrum of a tapered disc by doing
#            multi-linear interpolation on a pre-computed grid
#            of models
#
# EXAMPLE:
#                                                                        
# MODIFICATION HISTORY:
# Written originally in IDL by:	Andreas Efstathiou, February 2003
# -      
     np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
     
#
     tau_uv = float(argv[0])
     r2tor1 = argv[1]
     theta_1 = argv[2]
     theta_v = argv[3]
#
     w=wavey
     
     ix1=np.int(tau_uv/250.) - 1
     ix2=np.int(r2tor1/20.) - 1
     ix3=np.int(theta_1/7.5) -2
#
     ix4=np.int(theta_v/(x4_theta_v[1]-x4_theta_v[0])) -1 
#
#
     dx1=x1_tau_uv[ix1+1]-x1_tau_uv[ix1]
     dx2=x2_r2tor1[ix2+1]-x2_r2tor1[ix2]
     dx3=x3_theta_1[ix3+1]-x3_theta_1[ix3]
     dx4=x4_theta_v[ix4+1]-x4_theta_v[ix4]
     dv=dx1*dx2*dx3
#
#
     Ea=(x1_tau_uv[ix1+1]-tau_uv)*(x2_r2tor1[ix2+1]-r2tor1)*(theta_1-x3_theta_1[ix3])/dv
     Eb=(x1_tau_uv[ix1+1]-tau_uv)*(r2tor1-x2_r2tor1[ix2])*(theta_1-x3_theta_1[ix3])/dv
     Ec=(tau_uv-x1_tau_uv[ix1])*(x2_r2tor1[ix2+1]- r2tor1)*(theta_1-x3_theta_1[ix3])/dv
     Ed=(tau_uv-x1_tau_uv[ix1])*(r2tor1-x2_r2tor1[ix2])*(theta_1-x3_theta_1[ix3])/dv
     Ee=(x1_tau_uv[ix1+1]-tau_uv)*(x2_r2tor1[ix2+1]-r2tor1)*(x3_theta_1[ix3+1]-theta_1)/dv
     Ef=(x1_tau_uv[ix1+1]-tau_uv)*(r2tor1-x2_r2tor1[ix2])*(x3_theta_1[ix3+1]-theta_1)/dv
     Eg=(tau_uv-x1_tau_uv[ix1])*(x2_r2tor1[ix2+1]-r2tor1)*(x3_theta_1[ix3+1]-theta_1)/dv
     Eh=(tau_uv-x1_tau_uv[ix1])*(r2tor1-x2_r2tor1[ix2])*(x3_theta_1[ix3+1]-theta_1)/dv
#
#
     z0=tapered_discs[:,ix4,ix3+1,ix2,ix1]  
     z1=tapered_discs[:,ix4,ix3+1,ix2+1,ix1]  
     z2=tapered_discs[:,ix4,ix3+1,ix2,ix1+1]  
     z3=tapered_discs[:,ix4,ix3+1,ix2+1,ix1+1]  
     z4=tapered_discs[:,ix4,ix3,ix2,ix1]   
     z5=tapered_discs[:,ix4,ix3,ix2,ix1]      
     z6=tapered_discs[:,ix4,ix3,ix2,ix1+1]  
     z7=tapered_discs[:,ix4,ix3,ix2+1,ix1+1]  
#
     a0=aniso_factor[ix4,ix3+1,ix2,ix1]   
     a1=aniso_factor[ix4,ix3+1,ix2+1,ix1]
     a2=aniso_factor[ix4,ix3+1,ix2,ix1+1]
     a3=aniso_factor[ix4,ix3+1,ix2+1,ix1+1]
     a4=aniso_factor[ix4,ix3,ix2,ix1]   
     a5=aniso_factor[ix4,ix3,ix2+1,ix1]
     a6=aniso_factor[ix4,ix3,ix2,ix1+1]
     a7=aniso_factor[ix4,ix3,ix2+1,ix1+1]
#
     aniso=a0*Ea+a1*Eb+a2*Ec+a3*Ed+a4*Ee+a5*Ef+a6*Eg+a7*Eh
#
     s=z0*Ea + z1*Eb + z2*Ec + z3*Ed + z4*Ee + z5*Ef + z6*Eg + z7*Eh

     s=s - np.amax(s)

     s=10.**s
     
     return aniso, w, s
     
def flared_disc(argv,**kwargs):    
# 
#+
# NAME:
#	flared_disc
#
# PURPOSE:
#
# CATEGORY:
#
# CALLING SEQUENCE:
#       flared_disc, w, s, aniso, tau_uv, r2tor1, theta_1, theta_v
#	
# INPUTS:
# 
#      tau_uv   =  equatorial UV optical depth of the torus
#      r2tor1   =  ratio of outer to inner radius
#      theta_1  =  opening angle of the torus
#      theta_v  =  viewing angle (measured from the equator)
#
# OPTIONAL INPUTS:
#	
# KEYWORD PARAMETERS:
#
#   
# OUTPUTS:
#     w      =  wavelength in microns
#     s      =  nuSnu
#     aniso  =  anistropy correction factor (see Efstathiou 2006)
#
# OPTIONAL OUTPUTS:
#
# COMMON BLOCKS:
#	None
#
# SIDE EFFECTS:
#	None known
#
# RESTRICTIONS:
#	None known
#
# PROCEDURE:
#            Calculate the spectrum of a tapered disc by doing
#            multi-linear interpolation on a pre-computed grid
#            of models
#
# EXAMPLE:
#                                                                      
# MODIFICATION HISTORY:
# 	Written by:	Andreas Efstathiou, June 2014, and translated to Python by Charalambia Varnava, July 2022
#-      
    np.seterr(all='ignore') #ignore over/underflow 2 errors
# 
    w=wavefr06
# 
    tau_uv=float(argv[0])
    r2tor1=argv[1]
    theta_1=argv[2]
    theta_v=argv[3]
# 

    ixx1 = [k <= tau_uv for k in x1_ct]
    ixx2 = [l >= r2tor1 for l in x2_rm]
    ixx3 = [m >= theta_1 for m in x3_ta]
    ixx4 = [n >= theta_v for n in x4_thfr06]
#    
    ixx1_index = [i for i, x in enumerate(ixx1) if x]
    ixx2_index = [i for i, x in enumerate(ixx2) if x]
    ixx3_index = [i for i, x in enumerate(ixx3) if x]
    ixx4_index = [i for i, x in enumerate(ixx4) if x]
    
    
    ix1=ixx1_index[0]-1
    ix2=ixx2_index[0]-1
    ix3=ixx3_index[0]-1
    ix4=ixx4_index[0]-1
# 

#
    dx1=x1_ct[ix1+1]-x1_ct[ix1]
    dx2=x2_rm[ix2+1]-x2_rm[ix2]
    dx3=x3_ta[ix3+1]-x3_ta[ix3]
    dx4=x4_thfr06[ix4+1]-x4_thfr06[ix4]
    dv=dx1*dx2*dx3
# 
# 
    Ea=(x1_ct[ix1+1]-tau_uv)*(x2_rm[ix2+1]-r2tor1)*(theta_1-x3_ta[ix3])/dv
    Eb=(x1_ct[ix1+1]-tau_uv)*(r2tor1-x2_rm[ix2])*(theta_1-x3_ta[ix3])/dv
    Ec=(tau_uv-x1_ct[ix1])*(x2_rm[ix2+1]- r2tor1)*(theta_1-x3_ta[ix3])/dv
    Ed=(tau_uv-x1_ct[ix1])*(r2tor1-x2_rm[ix2])*(theta_1-x3_ta[ix3])/dv
    Ee=(x1_ct[ix1+1]-tau_uv)*(x2_rm[ix2+1]-r2tor1)*(x3_ta[ix3+1]-theta_1)/dv
    Ef=(x1_ct[ix1+1]-tau_uv)*(r2tor1-x2_rm[ix2])*(x3_ta[ix3+1]-theta_1)/dv
    Eg=(tau_uv-x1_ct[ix1])*(x2_rm[ix2+1]-r2tor1)*(x3_ta[ix3+1]-theta_1)/dv
    Eh=(tau_uv-x1_ct[ix1])*(r2tor1-x2_rm[ix2])*(x3_ta[ix3+1]-theta_1)/dv
# 
    z0=fr06_discs_r[:,ix4,ix3+1,ix2,ix1]
    z1=fr06_discs_r[:,ix4,ix3+1,ix2+1,ix1]
    z2=fr06_discs_r[:,ix4,ix3+1,ix2,ix1+1]
    z3=fr06_discs_r[:,ix4,ix3+1,ix2+1,ix1+1]
    z4=fr06_discs_r[:,ix4,ix3,ix2,ix1]
    z5=fr06_discs_r[:,ix4,ix3,ix2+1,ix1]
    z6=fr06_discs_r[:,ix4,ix3,ix2,ix1+1]
    z7=fr06_discs_r[:,ix4,ix3,ix2+1,ix1+1]
# 
# 
# 
    a0=aniso_factor_fr06[ix4,ix3+1,ix2,ix1]
    a1=aniso_factor_fr06[ix4,ix3+1,ix2+1,ix1]
    a2=aniso_factor_fr06[ix4,ix3+1,ix2,ix1+1]
    a3=aniso_factor_fr06[ix4,ix3+1,ix2+1,ix1+1]
    a4=aniso_factor_fr06[ix4,ix3,ix2,ix1]
    a5=aniso_factor_fr06[ix4,ix3,ix2+1,ix1]
    a6=aniso_factor_fr06[ix4,ix3,ix2,ix1+1]
    a7=aniso_factor_fr06[ix4,ix3,ix2+1,ix1+1]
# 
    aniso=a0*Ea + a1*Eb + a2*Ec + a3*Ed + a4*Ee + a5*Ef + a6*Eg + a7*Eh
# 
    s=z0*Ea+z1*Eb+z2*Ec+z3*Ed+z4*Ee+z5*Ef+z6*Eg+z7*Eh
#
    s = s - np.log10(w) 
    s=s-np.amax(s)
    
    s=10.**s

    return aniso, w, s

def st16_disc(argv,**kwargs):
# 
#+
# NAME:
#	st16_disc
#
# PURPOSE:
#
# CATEGORY:
#
# CALLING SEQUENCE:
#       st16_disc, w, s, aniso, tau_uv, r2tor1, theta_1, theta_v
#	
# INPUTS:
# 
#      tau_uv   =  equatorial UV optical depth of the torus
#      r2tor1   =  ratio of outer to inner radius
#      theta_1  =  opening angle of the torus
#      theta_v  =  viewing angle (measured from the equator)
#
# OPTIONAL INPUTS:
#	
# KEYWORD PARAMETERS:
#
#   
# OUTPUTS:
#     w      =  wavelength in microns
#     s      =  nuSnu
#     aniso  =  anistropy correction factor (see Efstathiou 2006)
#
# OPTIONAL OUTPUTS:
#
# COMMON BLOCKS:
#	None
#
# SIDE EFFECTS:
#	None known
#
# RESTRICTIONS:
#	None known
#
# PROCEDURE:
#            Calculate the spectrum of a tapered disc by doing
#            multi-linear interpolation on a pre-computed grid
#            of models
#
# EXAMPLE:
#                                                                      
# MODIFICATION HISTORY:
# 	Written by:	Andreas Efstathiou, June 2014, and translated to Python by Charalambia Varnava, July 2022
#-      
    np.seterr(all='ignore') #ignore over/underflow 2 errors
# 
    w=wavest16
# 
    tau_uv = float(argv[0])
    r2tor1 = argv[1]
    theta_1 = argv[2]
    theta_v = argv[3]
# 
    ixx1 = [k >= tau_uv for k in x1_oa]
    ixx2 = [l >= r2tor1 for l in x2_rr]
    ixx3 = [m >= theta_1 for m in x3_tt]
    ixx4 = [n >= theta_v for n in x4_thst16]
#   
    ixx1_index = [i for i, x in enumerate(ixx1) if x]
    ixx2_index = [i for i, x in enumerate(ixx2) if x]
    ixx3_index = [i for i, x in enumerate(ixx3) if x]
    ixx4_index = [i for i, x in enumerate(ixx4) if x]
    
    
    ix1=ixx1_index[0]-1
    ix2=ixx2_index[0]-1
    ix3=ixx3_index[0]-1
    ix4=ixx4_index[0]-1
# 
#

    dx1=x1_oa[ix1+1]-x1_oa[ix1]
    dx2=x2_rr[ix2+1]-x2_rr[ix2]
    dx3=x3_tt[ix3+1]-x3_tt[ix3]
    dx4=x4_thst16[ix4+1]-x4_thst16[ix4]
    dv=dx1*dx2*dx3
#     
#
    Ea=(x1_oa[ix1+1]-tau_uv)*(x2_rr[ix2+1]-r2tor1)*(theta_1-x3_tt[ix3])/dv
    Eb=(x1_oa[ix1+1]-tau_uv)*(r2tor1-x2_rr[ix2])*(theta_1-x3_tt[ix3])/dv
    Ec=(tau_uv-x1_oa[ix1])*(x2_rr[ix2+1]- r2tor1)*(theta_1-x3_tt[ix3])/dv
    Ed=(tau_uv-x1_oa[ix1])*(r2tor1-x2_rr[ix2])*(theta_1-x3_tt[ix3])/dv
    Ee=(x1_oa[ix1+1]-tau_uv)*(x2_rr[ix2+1]-r2tor1)*(x3_tt[ix3+1]-theta_1)/dv
    Ef=(x1_oa[ix1+1]-tau_uv)*(r2tor1-x2_rr[ix2])*(x3_tt[ix3+1]-theta_1)/dv
    Eg=(tau_uv-x1_oa[ix1])*(x2_rr[ix2+1]-r2tor1)*(x3_tt[ix3+1]-theta_1)/dv
    Eh=(tau_uv-x1_oa[ix1])*(r2tor1-x2_rr[ix2])*(x3_tt[ix3+1]-theta_1)/dv
# 
    z0=st16_discs_r[:,ix4,ix3+1,ix2,ix1]
    z1=st16_discs_r[:,ix4,ix3+1,ix2+1,ix1]
    z2=st16_discs_r[:,ix4,ix3+1,ix2,ix1+1]
    z3=st16_discs_r[:,ix4,ix3+1,ix2+1,ix1+1]
    z4=st16_discs_r[:,ix4,ix3,ix2,ix1]
    z5=st16_discs_r[:,ix4,ix3,ix2+1,ix1]
    z6=st16_discs_r[:,ix4,ix3,ix2,ix1+1]
    z7=st16_discs_r[:,ix4,ix3,ix2+1,ix1+1]
# 
# 
# 
    a0=aniso_factor_st16[ix4,ix3+1,ix2,ix1]
    a1=aniso_factor_st16[ix4,ix3+1,ix2+1,ix1]
    a2=aniso_factor_st16[ix4,ix3+1,ix2,ix1+1]
    a3=aniso_factor_st16[ix4,ix3+1,ix2+1,ix1+1]
    a4=aniso_factor_st16[ix4,ix3,ix2,ix1]
    a5=aniso_factor_st16[ix4,ix3,ix2+1,ix1]
    a6=aniso_factor_st16[ix4,ix3,ix2,ix1+1]
    a7=aniso_factor_st16[ix4,ix3,ix2+1,ix1+1]
# 
    aniso=a0*Ea + a1*Eb + a2*Ec + a3*Ed + a4*Ee + a5*Ef + a6*Eg + a7*Eh
# 
    s=z0*Ea+z1*Eb+z2*Ec+z3*Ed+z4*Ee+z5*Ef+z6*Eg+z7*Eh
# 
    s = s - np.log10(w)
    s=s-np.amax(s)

    s=10.**s

    return aniso, w, s

def s15_disc(argv,**kwargs):
# 
#+
# NAME:
#	s15_disc
#
# PURPOSE:
#
# CATEGORY:
#
# CALLING SEQUENCE:
#       s15_disc, w, s, aniso, s15_vc, s15_ac, s15_ad, s15_th
#	
# INPUTS:
# 
#      s15_vc   =  cloud volume filling factor (%)
#      s15_ac   =  optical depth of the individual clouds
#      s15_ad   =  optical depth of the disk mid-plane
#      s15_th   =  inclination    
#
# OPTIONAL INPUTS:
#	
# KEYWORD PARAMETERS:
#
#   
# OUTPUTS:
#     w      =  wavelength in microns
#     s      =  nuSnu
#     aniso  =  anistropy correction factor (see Efstathiou 2006)
#
# OPTIONAL OUTPUTS:
#
# COMMON BLOCKS:
#	None
#
# SIDE EFFECTS:
#	None known
#
# RESTRICTIONS:
#	None known
#
# PROCEDURE:
#            Calculate the spectrum of a S15 disc by doing
#            multi-linear interpolation on a pre-computed grid
#            of models
#
# EXAMPLE:
#                                                                      
# MODIFICATION HISTORY:
# 	Written by:	Andreas Efstathiou, March 2016, and translated to Python by Charalambia Varnava, July 2022
#-      
    np.seterr(all='ignore') #ignore over/underflow 2 errors
# 
    w=waves15
# 
    s15_vc=float(argv[0])
    s15_ac=argv[1]
    s15_ad=argv[2]
    s15_th=argv[3]
# 
    ixx1 = [k >= s15_vc for k in x1_vc]
    ixx2 = [l >= s15_ac for l in x2_ac]
    ixx3 = [m >= s15_ad for m in x3_ad]
    ixx4 = [n >= s15_th for n in x4_th]
#   
    ixx1_index = [i for i, x in enumerate(ixx1) if x]
    ixx2_index = [i for i, x in enumerate(ixx2) if x]
    ixx3_index = [i for i, x in enumerate(ixx3) if x]
    ixx4_index = [i for i, x in enumerate(ixx4) if x]
    
    
    ix1=ixx1_index[0]-1
    ix2=ixx2_index[0]-1
    ix3=ixx3_index[0]-1
    ix4=ixx4_index[0]-1
# 
#
    dx1=x1_vc[ix1+1]-x1_vc[ix1]
    dx2=x2_ac[ix2+1]-x2_ac[ix2]
    dx3=x3_ad[ix3+1]-x3_ad[ix3]
    dx4=x4_th[ix4+1]-x4_th[ix4]
    dv=dx1*dx2*dx3
# 
    tau_uv=s15_vc
    r2tor1=s15_ac
    theta_1=s15_ad
# 
# 
    Ea=(x1_vc[ix1+1]-s15_vc)*(x2_ac[ix2+1]-s15_ac)*(s15_ad-x3_ad[ix3])/dv
    Eb=(x1_vc[ix1+1]-s15_vc)*(s15_ac-x2_ac[ix2])*(s15_ad-x3_ad[ix3])/dv
    Ec=(tau_uv-x1_vc[ix1])*(x2_ac[ix2+1]- r2tor1)*(theta_1-x3_ad[ix3])/dv
    Ed=(tau_uv-x1_vc[ix1])*(r2tor1-x2_ac[ix2])*(theta_1-x3_ad[ix3])/dv
    Ee=(x1_vc[ix1+1]-tau_uv)*(x2_ac[ix2+1]-r2tor1)*(x3_ad[ix3+1]-theta_1)/dv
    Ef=(x1_vc[ix1+1]-tau_uv)*(r2tor1-x2_ac[ix2])*(x3_ad[ix3+1]-theta_1)/dv
    Eg=(tau_uv-x1_vc[ix1])*(x2_ac[ix2+1]-r2tor1)*(x3_ad[ix3+1]-theta_1)/dv
    Eh=(tau_uv-x1_vc[ix1])*(r2tor1-x2_ac[ix2])*(x3_ad[ix3+1]-theta_1)/dv
# 
    z0=s15_discs_r[:,ix4,ix3+1,ix2,ix1]
    z1=s15_discs_r[:,ix4,ix3+1,ix2+1,ix1]
    z2=s15_discs_r[:,ix4,ix3+1,ix2,ix1+1]
    z3=s15_discs_r[:,ix4,ix3+1,ix2+1,ix1+1]
    z4=s15_discs_r[:,ix4,ix3,ix2,ix1]
    z5=s15_discs_r[:,ix4,ix3,ix2+1,ix1]
    z6=s15_discs_r[:,ix4,ix3,ix2,ix1+1]
    z7=s15_discs_r[:,ix4,ix3,ix2+1,ix1+1]
# 
# 
# 
    a0=aniso_factor_s15[ix4,ix3+1,ix2,ix1]
    a1=aniso_factor_s15[ix4,ix3+1,ix2+1,ix1]
    a2=aniso_factor_s15[ix4,ix3+1,ix2,ix1+1]
    a3=aniso_factor_s15[ix4,ix3+1,ix2+1,ix1+1]
    a4=aniso_factor_s15[ix4,ix3,ix2,ix1]
    a5=aniso_factor_s15[ix4,ix3,ix2+1,ix1]
    a6=aniso_factor_s15[ix4,ix3,ix2,ix1+1]
    a7=aniso_factor_s15[ix4,ix3,ix2+1,ix1+1]
# 
    aniso=a0*Ea + a1*Eb + a2*Ec + a3*Ed + a4*Ee + a5*Ef + a6*Eg + a7*Eh
# 
    s=z0*Ea+z1*Eb+z2*Ec+z3*Ed+z4*Ee+z5*Ef+z6*Eg+z7*Eh
# 
    s = s - np.log10(w)
    s = s-np.amax(s)
    
    s = 10.**s

    return aniso, w, s


def synthesis_routine_SMART(fsb,tau_v,age,t_e,fagn,
          agn1,agn2,agn3,agn4,fpol,polt, AGNType,**kwargs):
#    
#+
# NAME:
#       synthesis_routine_SMART
#         
# PURPOSE:
#         
# CATEGORY:
#
# CALLING SEQUENCE:
#
#       synthesis_routine_SMART    
#	
# INPUTS:
# 
# parameters of the models
#
# OPTIONAL INPUTS:
#	
# KEYWORD PARAMETERS:
#
#   
# OUTPUTS:
# 
#  wavelength, flux
#
# OPTIONAL OUTPUTS:
#
# COMMON BLOCKS:
#	   None
#
# SIDE EFFECTS:
#	   None known
#
# RESTRICTIONS:
#	   None known
#
# PROCEDURE:
# 
# Claculate the wavelength and the flux           
#
# EXAMPLE:
#                                                                      
# MODIFICATION HISTORY:
# 	   Written originally in IDL by: Andreas Efstathiou, March 2015
#                                    Charalambia Varnava modified the code, December 2022
#-      
    np.seterr(all='ignore') #ignore over/underflow 0 errors
#

# 
    if AGNType==1:
         r2tor1=agn1
         tau_uv=agn2
         theta_1=agn3
         theta_v=agn4
    elif AGNType==2:
         ct=agn1
         rm=agn2
         ta=agn3
         thfr06=agn4
    elif AGNType==3:
         oa=agn1
         rr=agn2
         tt=agn3
         thst16=agn4
    elif AGNType==4:
         vc=agn1
         ac=agn2
         ad=agn3
         th=agn4
#
#    Starburst emission
#
    starb=galaxy_starburst2_fnu([tau_v,age,t_e,5.9e7])
#
    wave_synth=starb[0]
    f1=fsb*starb[1]
#
#  AGN emission
#
    if AGNType==1:
        cor_theta_v=theta_v
        if theta_v > theta_1 and theta_v < 65.:
    #      
            cor_theta_v=0.9*theta_1
    #
        agn=tapered_disc([tau_uv,r2tor1,theta_1,cor_theta_v])
    #
        flux= (agn[2] + 1.e-40) * agn[1]
# 
    if AGNType==2:
        cor_thfr06=thfr06
        if thfr06 > 0.9*ct and thfr06 < 80.:
    # 
            cor_thfr06=0.9*ct
    #
        agn=flared_disc([ct,rm,ta,cor_thfr06])
    #
        flux= (agn[2] + 1.e-40) * agn[1]

# 
    if AGNType==3:
        cor_thst16=thst16
        if thst16 > 0.9*oa and thst16 < 80.:
    # 
            cor_thst16=0.9*oa
    #
        agn=st16_disc([oa,rr,tt,cor_thst16])
    #
        flux= (agn[2] + 1.e-40) * agn[1]
# 
    if AGNType==4:
        cor_th=th
    #
        agn=s15_disc([vc,ac,ad,cor_th])
    #
        flux= (agn[2] + 1.e-40) * agn[1]
#
    mm=np.amax(flux)
    bb=flux/mm
#
#
    agn_f=fagn*bb + 1.e-40   
#
#   interpolate the AGN emission onto the SB grid
#
    wave_agn=agn[1]
    lwave_agn=wave_agn*0.
    lagn_f=agn_f*0.
#
    for l in range(len(wave_agn)):
#           
        lwave_agn[l]=math.log10(wave_agn[l])
        lagn_f[l]=math.log10(agn_f[l])
#
    agn_func=interpolate.interp1d(lwave_agn,lagn_f)
#
    f2 = f1*0. + 1.e-40
#
    for l in range(len(wave_synth)):
        if wave_synth[l] < np.amax(wave_agn):         
            fff=agn_func(math.log10(wave_synth[l]))
            f2[l]=10.**fff
#
#  Add polar dust if appropriate
#
    temp=polt
    polar_dust = polar_dust_fnu2([temp])
#
#
    f4 = fpol*polar_dust[1]
#
#   Add  all components
#
    fall = f1 + f2 + f4
#
    output_wave=wave_synth
    output_flux=fall

    return output_wave, output_flux

def synthesis_routine_host_SMART(fhost,tvhost,psi,cirr_tau,iview,models_fnu,hostType,wwsorted,**kwargs):
#    
#+
# NAME:
#       synthesis_routine_host_SMART
#         
# PURPOSE:
#         
# CATEGORY:
#
# CALLING SEQUENCE:
#
#       synthesis_routine_host_SMART    
#	
# INPUTS:
# 
# parameters of the models
#
# OPTIONAL INPUTS:
#	
# KEYWORD PARAMETERS:
#
#   
# OUTPUTS:
# 
#  wavelength, flux
#
# OPTIONAL OUTPUTS:
#
# COMMON BLOCKS:
#	   None
#
# SIDE EFFECTS:
#	   None known
#
# RESTRICTIONS:
#	   None known
#
# PROCEDURE:
# 
# Calculate the wavelength and the flux of the host           
#
# EXAMPLE:
#                                                                      
# MODIFICATION HISTORY:
# 	   Written originally in IDL by: Andreas Efstathiou, March 2015
#                                    CHharalambia Varnava modified the code, December 2022
#-      
    np.seterr(all='ignore') #ignore over/underflow 0 errors
#
#
    if hostType==1:
         fsph=fhost
         tvv=tvhost
    elif hostType == 2:
         fdisc=fhost
         tv=tvhost
#
# #    Spheroid emission
# 
    f3=0.
    if hostType==1:
        spheroid=galaxy_spheroid_fnu([tvv,psi,cirr_tau,1., models_fnu,wwsorted])
#
        f3=fsph*spheroid[1]   * spheroid[0]
# 
#    Disc emission
#
# Add disc if appropriate
# 
# Calculate disc component from the appropriate grid
#
    f5=0.
    if hostType==2:
        disc=galaxy_disc_fnu([tv,psi,cirr_tau,iview, models_fnu,wwsorted])
        f5 = fdisc*disc[1]  * disc[0]

#
#   Add  all components
#
    fall =  f3 + f5
#
    output_wave=wwsorted
    output_flux=fall

    return output_wave, output_flux   

