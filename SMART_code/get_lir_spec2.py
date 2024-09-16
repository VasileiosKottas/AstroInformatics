import numpy as np
from pylab import *
from int_spec import *

#***********************************************************************
def get_lir_spec2(*args,**kwargs):
#+
# NAME:
# 	get_lir_spec2
# 
# PURPOSE:
#       returns bolometric luminosity given a spectrum in
#       units of Jy versus lambda
#
#
# MODIFICATION HISTORY:
#   Written by:	Andreas Efstathiou, Feb 97
# 	
#-
    np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
    wl=args[0]
    fjj=args[1]
    d_co=args[2]
#   
    freq=3.e14 / wl

    mm = fjj * freq

    mm = mm*1.e-26    # to convert to W m^-2
    factor= 1.e6 * 3.086e16    # factor to convert from Mpc to m
    mm = 4.*3.14159 * d_co**2. * mm # in W
#
    mm = mm / 3.9e26
    mm = mm * factor*factor

    get_lir_spec2=int_spec(wl,mm,args[3],args[4])

    return get_lir_spec2
   
        