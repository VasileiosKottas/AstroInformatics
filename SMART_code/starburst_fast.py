import numpy as np
import build_ssp2_fast as bu
import math
import new_cirrus_model_fast as ne
import build_ssp2_fast as bu
import matplotlib.pyplot as plt
# 
def starburst_fast(*argv,**kwargs):
# +
# NAME:
# 	starburst
# 
# PURPOSE:
# 
#                                                                       
# MODIFICATION HISTORY:
# Written by:	Andreas Efstathiou, March 2007, and translated to Python by Charalambia Varnava, December 2020
# -      
    np.seterr(all='ignore') #ignore over/underflow 0 errors
# 

    t_e=argv[0]
    f=argv[1]
    t_m=argv[2]
    age=argv[3]
    tau_v=argv[4]
    models=argv[5]
# 
# 
    dt=0.1e6
    chi=1.
# 
    time=dt*np.arange(15000)
#  
# 
    sfr=np.zeros(len(time))
    for i in range(len(time)):
        sfr[i]=math.exp(-time[i]/t_e)
# 
    gal_models=ne.new_cirrus_model_fast(time, sfr, age, chi, models,Av=1.e-6, leak=f)
  
    w=gal_models[0]
    s=gal_models[1]

    plot=kwargs.get('plot')
    if plot!=None:    
      plt.plot(w,s,label="Starburst",color="black")
      plt.legend(fontsize=14)
      plt.xlim(0.01,1000.)
      plt.ylim(np.amax(s)*1.e-3,np.amax(s)*10.)
      plt.xlabel("$\lambda$ ($\mu m$)")
      plt.ylabel("nu F_nu")
      plt.xscale('log')
      plt.yscale('log')
      plt.show()

    return w, s
     