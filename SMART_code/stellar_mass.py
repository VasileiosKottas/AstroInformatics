import numpy as np
import math
import csv
from scipy import interpolate
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
# 
def stellar_mass(*argv):
# +
# NAME:
# stellar_mass
# 
# PURPOSE:
# 
# 
# MODIFICATION HISTORY:
# Written by:  Andreas Efstathiou, August 2013, and translated to Python by Charalambia Varnava, December 2020
# 
#-
    np.seterr(all='ignore') #ignore over/underflow 0 errors
# 
    if len(argv)==1:
        print,'USAGE: stellar_mass, t_e, age, alpha, beta, Mstars'
        return
# 
    t_e=argv[0]
    age=argv[1]
    alpha=argv[2]
    beta=argv[3]
#
# 
# Read the Bruzual & Charlot table
# mass is in Mo at time tt
# 
    file=r'./bc2003_lr_m42_salp_ssp.4color'

    f=open(file,"r")
    lines = list(csv.reader(f, delimiter = ' ', skipinitialspace = True))
    tt = np.zeros(len(lines))
    mass = np.zeros(len(lines))
    for i in range(len(lines)):
        tt[i] = float(lines[i][0]) #read column 1

        mass[i] = lines[i][6] #read column 7
#   
    tt=(10.**tt)
 
    j=0
    for i in range(len(tt)):
        if i==0:
            tt[j] = 0
            mass[j] = 0
        else:
            tt[j] = tt[i]
            mass[j] = mass[i]
        j+=1
        
    ttt=tt
    mmass=mass
# 
# 
    nend=len(ttt)
    dt=ts_diff(ttt,1)
# 
# 
    sfrate=np.zeros(nend)
    y = [x <= float(age) for x in ttt]
    i=0
    for j in range(len(y)):  
        if np.any(y):
            if  y[j] == True:
                if t_e !=0:
                    sfrate[i]=ttt[j]**alpha*math.exp(-(ttt[j]/t_e)**beta)
                    i=i+1
# 
    x = [k <= float(age) for k in ttt]
# 
    stellar_mass=0.
# 
    s_mass_f=interpolate.interp1d(ttt,mmass)
# 
# 
    k=0
    for l in range(len(x)):
        if x[l] == True:         

            stellar_mass=stellar_mass+sfrate[l]*s_mass_f(age - ttt[l])*dt[l]  
# 
# 
    Mstars=stellar_mass  

    return Mstars
