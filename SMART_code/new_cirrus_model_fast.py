import numpy as np
from scipy import interpolate
import matplotlib.pyplot  as pyplot
# 
def ts_diff(ttt, interval):
    value=np.zeros(len(ttt))
    j=0
    for i in range(len(ttt)-1):
        if i==0:
            value[j]=ttt[i]
        else:
            value[j]=ttt[i]-ttt[i-interval]
        j=j+1
    return value

def new_cirrus_model_fast(*argv,**kwargs):
# +
# NAME:
# 	new_cirrus_model_fast
# 
# PURPOSE:
#  
# MODIFICATION HISTORY:
# Written by:	Andreas Efstathiou, August 2001, and translated to Python by Charalambia Varnava, December 2020
# 	
# -
    np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
    if len(argv)<4:
        print('USAGE: new_cirrus_model_fast,  time, sfr, age, chi, lambda, gal_models')
        return
#  
    time=argv[0]
    sfr=argv[1]
    age=float(argv[2])
    chi=argv[3]
    models=argv[4]
# 
    Av=kwargs.get('Av')
    if Av != None:
        Av=0.
        
    leak=kwargs.get('leak')
    if leak!=None:
        leak=0.
        
    file=kwargs.get('file')
    if file!=None:
        file='.spec'
# 
    if age < np.min(time) or age > np.max(time):
        print('ERROR: Requested age outside  allowed range')
        return
# 
# 
    t=models.age
    w=models[0].spectrum.LAMBDA
    f=np.zeros((len(w),len(t)))

    for i in range(len(t)-1):
        f[:,i]=models[i].spectrum.FLAMBDA
# 
    nend=len(time)
    dt=ts_diff(time,1)
# 
    nwave=len(w)
# 
    CnuSnu=np.zeros(nwave)
    gal_models=CnuSnu
# 
# 
    good = time <= age 

    InuSnu=np.zeros((nwave,len(good)))

    xx=np.zeros(len(good))
    for k in range(len(w)):
 
            local=f[k,:]
            xx_f=interpolate.interp1d(t,local)
            xx=xx_f(age-time[good])

            InuSnu[k,good]=xx
    
    plot=kwargs.get('plot')
    if plot!=None:  
      pyplot.plot(w/10000.,InuSnu[:,0], color='red', lw=2)
      pyplot.yscale('log')
      pyplot.xscale('log')
      pyplot.show()    

    for k in range(len(good)):
        dtt=dt[k]
        if (k == 0 or k == (len(good)-1)):
            dtt=dtt/2.
        CnuSnu=CnuSnu + dtt*sfr[k]*InuSnu[:,k]

#  
    lambdaa = w/10000.
# 
    nusnu=CnuSnu*w
    gal_models=nusnu
# 
   
    plot=kwargs.get('plot')
    if plot!=None:       
        pyplot.plot(lambdaa,gal_models, color='red', lw=2)
        pyplot.yscale('log')
        pyplot.xscale('log')
        
        pyplot.show()
    w_s=[]
    w_s.insert(0,lambdaa)
    w_s.insert(1,gal_models)

    return w_s
    