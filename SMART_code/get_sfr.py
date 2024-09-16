import numpy as np
from pylab import *
from int_spec import *

#***********************************************************************
def get_sfr(*args,**kwargs):
#+
# NAME:
# 	get_sfr
# 
# PURPOSE:
#       returns star formation rate
#
#
# MODIFICATION HISTORY:
#   Written by:	Andreas Efstathiou, Feb 97
#-
 np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
 pi=3.14159
 w=args[0]
 s=args[1]
 L=args[2]
 t_s=args[3]
 tau=args[4]
#
 t_av=kwargs.get('t_av')
 if t_av != None:
    t_av=t_s
#
 bol=int_spec(w,s,0.1,1000.)
 sfr_0=L/bol
#
 sfr=-sfr_0*tau*(exp(-t_s/tau) - 1.)/t_av
#
 print,'Maximum sfr = ',sfr_0,' Average sfr = ',sfr

 return sfr     