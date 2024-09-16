import numpy as np
import os
from likelihood import *
from os import path


def post_SMART_select(*argv,**kwargs):
#
#
# NAME:
#   post_SMART_select
# 
# PURPOSE:
# 
#    This is a routine that selects the run that has the minimum chi-squared.
#
# CATEGORY:
# 
# CALLING SEQUENCE:
# 
#       from post_SMART_select import *
# 
# INPUTS:
# 
#    flag_select:  post-process only objects with given flag
# 
# 
# OPTIONAL INPUTS:
# 
#    All the keywords
# 
# KEYWORD PARAMETERS:
# 
#  Explained in SMART User Manual
# 
# OUTPUTS:
# 
#  No output, everything is written to files for each fitted galaxy      
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
# EXAMPLE:
# 
#      Python>from post_SMART_select import *      
# 
# MODIFICATION HISTORY:
# 
#   Written by:  Charalambia Varnava, May 2022
# 

    if 'data_file' in kwargs: 
      data_file=kwargs.get('data_file')
    elif 'data_file' not in kwargs:
      data_file='objects'

    metallicity=0.008
#  
    if 'metallicity' in kwargs:    
      metallicity=kwargs.get('metallicity')  

#
    host_geometry='sph'
    if 'host_geometry' in kwargs:    
      host_geometry=kwargs.get('host_geometry')  
    if host_geometry=='sph':
         hostType=1
    elif host_geometry=='disc':
        hostType=2  
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
#  Run objects from the list in file "./"+data_file+"_list.txt" which have flag=flag_select
# 
    flag_select=argv[0]
    ii=0
#
#   Read the list of galaxies to be fitted
#
    data=np.genfromtxt('./'+data_file+'_list.txt',skip_header=1,dtype=None,encoding='ascii')
    ruben_name_x,redshifts_x,flag_x,flag_xx=[data[i] for i in data.dtype.names]
# 
    gg=flag_x==int(flag_select)
#    
    redshifts=redshifts_x[gg]
    ruben_name=ruben_name_x[gg]
# 
    run_name='S'
    if 'run_name' in kwargs:
      run_name=kwargs.get('run_name')
      
    num_of_runs=1
    if 'num_of_runs' in kwargs:
        num_of_runs=kwargs.get('num_of_runs')

    min_chi_squared=np.zeros(num_of_runs,dtype='float32');  
#     
    while ii<len(redshifts):  
       
        zz=redshifts[ii]
        name=ruben_name[ii] 
        models_fnu_x=select_library([zz,hostType,wavesb,metallicity])
        models_fnu=models_fnu_x[1]
        ii+=1
        run_id=1
        while run_id <= num_of_runs:
            run_id_temp = run_id
            XXX= run_name + str(run_id_temp)
#
            filename='./'+data_file+'/'+name+'.txt'     
#
#    read data for current galaxy
#    
            data=np.genfromtxt(filename,skip_header=1,dtype=None,encoding='ascii')
            ww,ff,ee,flag,source=[data[i] for i in data.dtype.names]
            file= './'+data_file+'_results/' + name+'_'+XXX+'_flat_samples.npz'

            if not path.exists(file):
                return
            
            data_emcee=np.load(file)
            flat_samples=data_emcee['flat_samples']
            data_emcee.close() 
            
            if hostType==1:
                ndim = 15
                b_fit_par=[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]

            if hostType==2:
                ndim = 16
                b_fit_par=[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
            
            for i in range(ndim):
                mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
                q = np.diff(mcmc)

                b_fit_par[i]=mcmc[1]
            
            min_chi_squared[run_id-1]=chi_squared_host(b_fit_par,ww,ff,ee,flag,models_fnu,hostType,AGNType,wavesb)   
            
            run_id = run_id + 1  
            
            
        minimum_chi_squared_min = np.argmin(min_chi_squared)
       
        i=1
        os.chdir('./'+data_file+'_results/')
        
        list_files = os.listdir( './')
        nameToRemoveTemp = minimum_chi_squared_min+1
        
        while i<=num_of_runs:
           
            if i != nameToRemoveTemp:
                
                nameToRemove = name + '_' + run_name + str(i)
               
                for item in list_files:
                    
                    if nameToRemove in item:
                        
                        if path.exists(item):
                            os.remove('./'+item)
           
            else: 
                
                nameToRename=  name + '_' + run_name + str(i)
            
                for item in list_files:
                     
                     if nameToRename in item:
                         newName = item.replace(run_name+ str(i), run_name)
                       
                         if path.exists(newName):
                             os.remove(newName)
                         os.rename(item,newName)
            i=i+1
        os.chdir('..')    