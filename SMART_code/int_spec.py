import numpy as np
import math
from scipy import interpolate
from pylab import *
#***********************************************************************

def int_spec(*args,**kwargs):
#+
# NAME:
# 	int_spec
# 
# PURPOSE:
# 	Integrates a nuSnu distribution to give bolometric flux
# 
# MODIFICATION HISTORY:
#   Written by:	Andreas Efstathiou, Feb 97, and translated to Python by Charalambia Varnava, Dec 2020
# 	
#-
    np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
    wl=args[0]
    fjj=args[1]
    
    wrange=[float(args[2]),float(args[3])]    
    fine=kwargs.get('fine')
    if fine==None:
        fwl=wl
        ffjj=fjj
    else:
        fwl=(wl[len(wl)-1]-wl[0])/float(fine)*np.arange(float(fine)+1) 
        fwl=fwl + wl[0]
        fjj= np.asarray(fjj, dtype=np.float64, order='C')
        wl= np.asarray(wl, dtype=np.float64, order='C')
        ffjj_f=interpolate.interp1d(wl,fjj)
        ffjj=ffjj_f(fwl)
# 
# 
    good=[k > wrange[0] and k < wrange[1] for k in fwl]
    fwl= np.asarray(fwl, dtype=np.float64, order='C')
    k=0
    wl11=np.zeros(len(good))
    ffjjkk=np.zeros(len(good))
    for i in range(len(good)):
        if good[i]==True:
            wl11[k]=math.log10(fwl[i])
            ffjjkk[k]=ffjj[i]
            k=k+1
# 
# 
    nla=k
    dalt1=np.zeros(nla)
    bol_flux=0.
    if nla>1:
        dalt1[0]=wl11[0]-wl11[1]  
        dalt1[nla-1]=wl11[nla-2]-wl11[nla-1]  
        i1=1
        while i1<nla-2:
            dalt1[i1]=0.5*(wl11[i1-1]-wl11[i1+1])
            i1=i1+1
    # 
        
        bol_flux=0.
        i1=0
        while i1<nla-1:
            dalt=dalt1[i1]      
            alt=math.log(10.0)*dalt
            bol_flux=bol_flux + ffjjkk[i1]*alt
            #
            i1=i1+1

    int_spec=bol_flux
    return int_spec
     