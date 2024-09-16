import numpy as np

from synthesis_routine_SMART import * 
import numpy as geek

#

def priors_host_cygnus(*argv,**kwargs):
    
        scale = argv[0]
        flag  = argv[1]
        hostType = argv[2]
        AGNType = argv[3]
        
        r2tor1=kwargs.get('r2tor1')
        theta_1=kwargs.get('theta_1')
        tau_uv=kwargs.get('tau_uv')
        t_e=kwargs.get('t_e') 
        age=kwargs.get('age')
        tau_v=kwargs.get('tau_v')
        tvv=kwargs.get('tvv') 
        psi=kwargs.get('psi')
        cirr_tau=kwargs.get('cirr_tau')
        iview=kwargs.get('iview')
        polt=kwargs.get('polt')
        ct=kwargs.get('ct') 
        rm=kwargs.get('rm') 
        oa=kwargs.get('oa') 
        rr=kwargs.get('rr') 
        tt=kwargs.get('tt') 
        vc=kwargs.get('vc') 
        ac=kwargs.get('ac')
        ad=kwargs.get('ad')
#        
        if hostType==1 and AGNType==1:
#
#    Parameters for a spheroidal run using the CYGNUS AGN model
#
            xxfsph      = [ scale[1] - 5., scale[1] + 5. ]
            xxtvv       = tvv
            xxpsi       = psi
            xxcirr_tau  = cirr_tau      
            xxfsb       = [ scale[0] - 5., scale[0] + 5. ]
            xxtau_v     = tau_v
            xxage       = age
            xxt_e       = t_e
            xxfagn      = [ scale[0] - 5., scale[0] + 5. ]
            xxr2tor1    = r2tor1
            xxtau_uv    = tau_uv
            xxtheta_1   = theta_1
            xxtheta_v   = [ np.log10(5.), np.log10(58.) ]
            xxfpol      = [ scale[0] - 5., scale[0] + 5. ]
            xxpolt      = polt
    #
            if flag==1:
               xxtheta_v=[ np.log10(70.), np.log10(80.) ]
    #
            host_gal=kwargs.get('host_gal')  
            if host_gal=='no':
                   xxfsph=[ -20., -19. ]
    #
            starburst_gal=kwargs.get('starburst_gal')  
            if starburst_gal=='no':
                   xxfsb=[ -20., -19. ] 
    #  
            AGN_gal=kwargs.get('AGN_gal')  
            if AGN_gal=='no':
                   xxfagn=[- 20., -19.]  
    # 
            polar=kwargs.get('polar')  
            if polar=='no':
                   xxfpol=[ -20., -19. ]
        #  
            sph=np.empty(1,dtype=[('xxfsph','f8',(2,)),('xxtvv','f8',(2,))
            ,('xxpsi','f8',(2,)),('xxcirr_tau','f8',(2,))    
            ,('xxfsb','f8',(2,)),('xxtau_v','f8',(2,)),('xxage','f8',(2,)),('xxt_e','f8',(2,))            
            ,('xxfagn','f8',(2,)),('xxr2tor1','f8',(2,)),('xxtau_uv','f8',(2,)),('xxtheta_1','f8',(2,))
            ,('xxtheta_v','f8',(2,)),('xxfpol','f8',(2,)),('xxpolt','f8',(2,))
            ])
            sph['xxfsph']=xxfsph
            sph['xxtvv']=xxtvv
            sph['xxpsi']=xxpsi
            sph['xxcirr_tau']=xxcirr_tau
            sph['xxfsb']=xxfsb
            sph['xxtau_v']=xxtau_v
            sph['xxage']=xxage
            sph['xxt_e']=xxt_e
            sph['xxfagn']=xxfagn
            sph['xxr2tor1']=xxr2tor1
            sph['xxtau_uv']=xxtau_uv
            sph['xxtheta_1']=xxtheta_1
            sph['xxtheta_v']=xxtheta_v
            sph['xxfpol']=xxfpol
            sph['xxpolt']=xxpolt
#
            return sph
# 
# 
        elif hostType==1 and AGNType==2:
#
#    Parameters for a spheroidal run using the Fritz AGN model
#
            xxfsph      = [ scale[1] - 5., scale[1] + 5. ]
            xxtvv       = tvv
            xxpsi       = psi
            xxcirr_tau  = cirr_tau      
            xxfsb       = [ scale[0] - 5., scale[0] + 5. ]
            xxtau_v     = tau_v
            xxage       = age
            xxt_e       = t_e
            xxfagn      = [ scale[0] - 5., scale[0] + 5. ]
            xxct        = ct
            xxrm        = rm
            xxta        = [ np.log10(1.), np.log10(9.9) ]      #   using 1. as lower range although library goes down to 0.1
            xxthfr06    = [ np.log10(5.), np.log10(56.) ]
            xxfpol      = [ scale[0] - 5., scale[0] + 5. ]
            xxpolt      = polt
    #
            if flag==1:
               xxthfr06=[ np.log10(82.), np.log10(88.) ]
    #
            host_gal=kwargs.get('host_gal')  
            if host_gal=='no':
                   xxfsph=[ -20., -19. ]
    #
            starburst_gal=kwargs.get('starburst_gal')  
            if starburst_gal=='no':
                   xxfsb=[ -20., -19. ] 
    #  
            AGN_gal=kwargs.get('AGN_gal')  
            if AGN_gal=='no':
                   xxfagn=[- 20., -19.]  
    # 
            polar=kwargs.get('polar')  
            if polar=='no':
                   xxfpol=[ -20., -19. ]
        #  
            sph=np.empty(1,dtype=[('xxfsph','f8',(2,)),('xxtvv','f8',(2,))
            ,('xxpsi','f8',(2,)),('xxcirr_tau','f8',(2,))    
            ,('xxfsb','f8',(2,)),('xxtau_v','f8',(2,)),('xxage','f8',(2,)),('xxt_e','f8',(2,))            
            ,('xxfagn','f8',(2,)),('xxct','f8',(2,)),('xxrm','f8',(2,)),('xxta','f8',(2,))
            ,('xxthfr06','f8',(2,)),('xxfpol','f8',(2,)),('xxpolt','f8',(2,))
            ])
            sph['xxfsph']=xxfsph
            sph['xxtvv']=xxtvv
            sph['xxpsi']=xxpsi
            sph['xxcirr_tau']=xxcirr_tau
            sph['xxfsb']=xxfsb
            sph['xxtau_v']=xxtau_v
            sph['xxage']=xxage
            sph['xxt_e']=xxt_e
            sph['xxfagn']=xxfagn
            sph['xxct']=xxct
            sph['xxrm']=xxrm
            sph['xxta']=xxta
            sph['xxthfr06']=xxthfr06
            sph['xxfpol']=xxfpol
            sph['xxpolt']=xxpolt
#
            return sph
# 
# 
        elif hostType==1 and AGNType==3:
            
#    Parameters for a spheroidal run using the SKIRTOR AGN model
#
            xxfsph      = [ scale[1] - 5., scale[1] + 5. ]
            xxtvv       = tvv
            xxpsi       = psi
            xxcirr_tau  = cirr_tau      
            xxfsb       = [ scale[0] - 5., scale[0] + 5. ]
            xxtau_v     = tau_v
            xxage       = age
            xxt_e       = t_e
            xxfagn      = [ scale[0] - 5., scale[0] + 5. ]
            xxoa        = oa
            xxrr        = rr
            xxtt        = tt
            xxthst16    = [ np.log10(5.), np.log10(60.) ]
            xxfpol      = [ scale[0] - 5., scale[0] + 5. ]
            xxpolt      = polt
    #
            if flag==1:
               xxthst16=[ np.log10(82.), np.log10(88.) ]
    #
            host_gal=kwargs.get('host_gal')  
            if host_gal=='no':
                   xxfsph=[ -20., -19. ]
    #
            starburst_gal=kwargs.get('starburst_gal')  
            if starburst_gal=='no':
                   xxfsb=[ -20., -19. ] 
    #  
            AGN_gal=kwargs.get('AGN_gal')  
            if AGN_gal=='no':
                   xxfagn=[- 20., -19.]  
    # 
            polar=kwargs.get('polar')  
            if polar=='no':
                   xxfpol=[ -20., -19. ]
        #  
            sph=np.empty(1,dtype=[('xxfsph','f8',(2,)),('xxtvv','f8',(2,))
            ,('xxpsi','f8',(2,)),('xxcirr_tau','f8',(2,))    
            ,('xxfsb','f8',(2,)),('xxtau_v','f8',(2,)),('xxage','f8',(2,)),('xxt_e','f8',(2,))            
            ,('xxfagn','f8',(2,)),('xxoa','f8',(2,)),('xxrr','f8',(2,)),('xxtt','f8',(2,))
            ,('xxthst16','f8',(2,)),('xxfpol','f8',(2,)),('xxpolt','f8',(2,))
            ])
            sph['xxfsph']=xxfsph
            sph['xxtvv']=xxtvv
            sph['xxpsi']=xxpsi
            sph['xxcirr_tau']=xxcirr_tau
            sph['xxfsb']=xxfsb
            sph['xxtau_v']=xxtau_v
            sph['xxage']=xxage
            sph['xxt_e']=xxt_e
            sph['xxfagn']=xxfagn
            sph['xxoa']=xxoa
            sph['xxrr']=xxrr
            sph['xxtt']=xxtt
            sph['xxthst16']=xxthst16
            sph['xxfpol']=xxfpol
            sph['xxpolt']=xxpolt
#
            return sph
# 
# 
        elif hostType==1 and AGNType==4:
#            
#    Parameters for a spheroidal run using the Siebenmorgen AGN model
#
            xxfsph      = [ scale[1] - 5., scale[1] + 5. ]
            xxtvv       = tvv
            xxpsi       = psi
            xxcirr_tau  = cirr_tau      
            xxfsb       = [ scale[0] - 5., scale[0] + 5. ]
            xxtau_v     = tau_v
            xxage       = age
            xxt_e       = t_e
            xxfagn      = [ scale[0] - 5., scale[0] + 5. ]
            xxvc        = vc
            xxac        = ac
            xxad        = ad
            xxth        = [ np.log10(0.1), np.log10(44.9) ]
            xxfpol      = [ scale[0] - 5., scale[0] + 5. ]
            xxpolt      = polt
    #
            if flag==1:
               xxth=[ np.log10(45.1), np.log10(89.9)]
    #
            host_gal=kwargs.get('host_gal')  
            if host_gal=='no':
                   xxfsph=[ -20., -19. ]
    #
            starburst_gal=kwargs.get('starburst_gal')  
            if starburst_gal=='no':
                   xxfsb=[ -20., -19. ] 
    #  
            AGN_gal=kwargs.get('AGN_gal')  
            if AGN_gal=='no':
                   xxfagn=[- 20., -19.]  
    # 
            polar=kwargs.get('polar')  
            if polar=='no':
                   xxfpol=[ -20., -19. ]
        #  
            sph=np.empty(1,dtype=[('xxfsph','f8',(2,)),('xxtvv','f8',(2,))
            ,('xxpsi','f8',(2,)),('xxcirr_tau','f8',(2,))    
            ,('xxfsb','f8',(2,)),('xxtau_v','f8',(2,)),('xxage','f8',(2,)),('xxt_e','f8',(2,))            
            ,('xxfagn','f8',(2,)),('xxvc','f8',(2,)),('xxac','f8',(2,)),('xxad','f8',(2,))
            ,('xxth','f8',(2,)),('xxfpol','f8',(2,)),('xxpolt','f8',(2,))
            ])
            sph['xxfsph']=xxfsph
            sph['xxtvv']=xxtvv
            sph['xxpsi']=xxpsi
            sph['xxcirr_tau']=xxcirr_tau
            sph['xxfsb']=xxfsb
            sph['xxtau_v']=xxtau_v
            sph['xxage']=xxage
            sph['xxt_e']=xxt_e
            sph['xxfagn']=xxfagn
            sph['xxvc']=xxvc
            sph['xxac']=xxac
            sph['xxad']=xxad
            sph['xxth']=xxth
            sph['xxfpol']=xxfpol
            sph['xxpolt']=xxpolt
#
            return sph
# 
#             
        elif hostType==2 and AGNType==1:
#
#    Parameters for a disc run using the CYGNUS AGN model
#
            xxfdisc     = [ scale[1] - 5., scale[1] + 5. ]
            xxtv        = [ np.log10(0.01), np.log10(28.) ]
            xxpsi       = [ np.log10(1.), np.log10(9.) ] 
            xxcirr_tau  = [ np.log10(0.51e9), np.log10(7.9e9) ]  
            xxiview     = iview
            xxfsb       = [ scale[0] - 5., scale[0] + 5. ]
            xxtau_v     = tau_v
            xxage       = age
            xxt_e       = t_e
            xxfagn      = [ scale[0] - 5., scale[0] + 5. ]
            xxr2tor1    = r2tor1
            xxtau_uv    = tau_uv
            xxtheta_1   = theta_1
            xxtheta_v   = [ np.log10(5.), np.log10(58.) ]
            xxfpol      = [ scale[0] - 5., scale[0] + 5. ]
            xxpolt      = polt
    # 
            if flag==1:
               xxtheta_v=[ np.log10(70.), np.log10(80.) ]
    #
            host_gal=kwargs.get('host_gal')  
            if host_gal=='no':
                   xxfdisc=[ -20., -19. ]
    #
            starburst_gal=kwargs.get('starburst_gal')  
            if starburst_gal=='no':
                   xxfsb=[ -20., -19. ] 
    #       
            AGN_gal=kwargs.get('AGN_gal')  
            if AGN_gal=='no':
                   xxfagn=[ -20., -19. ]  
    # 
            polar=kwargs.get('polar')  
            if polar=='no':
                   xxfpol=[ -20., -19. ]
        #  
            disc=np.empty(1,dtype=[('xxfdisc','f8',(2,)),('xxtv','f8',(2,))
            ,('xxpsi','f8',(2,)),('xxcirr_tau','f8',(2,)),('xxiview','f8',(2,))   
            ,('xxfsb','f8',(2,)),('xxtau_v','f8',(2,)),('xxage','f8',(2,)),('xxt_e','f8',(2,))            
            ,('xxfagn','f8',(2,)),('xxr2tor1','f8',(2,)),('xxtau_uv','f8',(2,)),('xxtheta_1','f8',(2,))
            ,('xxtheta_v','f8',(2,)),('xxfpol','f8',(2,)),('xxpolt','f8',(2,))
            ])
            disc['xxfdisc']=xxfdisc
            disc['xxtv']=xxtv
            disc['xxpsi']=xxpsi
            disc['xxcirr_tau']=xxcirr_tau
            disc['xxiview']=xxiview
            disc['xxfsb']=xxfsb
            disc['xxtau_v']=xxtau_v
            disc['xxage']=xxage
            disc['xxt_e']=xxt_e
            disc['xxfagn']=xxfagn
            disc['xxr2tor1']=xxr2tor1
            disc['xxtau_uv']=xxtau_uv
            disc['xxtheta_1']=xxtheta_1
            disc['xxtheta_v']=xxtheta_v
            disc['xxfpol']=xxfpol
            disc['xxpolt']=xxpolt
    #
            return disc
# 
# 
        elif hostType==2 and AGNType==2:

#    Parameters for a disc run using the Fritz AGN model
#
            xxfdisc     = [ scale[1] - 5., scale[1] + 5. ]
            xxtv        = [ np.log10(0.01), np.log10(28.) ]
            xxpsi       = [ np.log10(1.), np.log10(9.) ] 
            xxcirr_tau  = [ np.log10(0.51e9), np.log10(7.9e9) ]  
            xxiview     = [ np.log10(1.), np.log10(23.) ]
            xxfsb       = [ scale[0] - 5., scale[0] + 5. ]
            xxtau_v     = tau_v
            xxage       = age
            xxt_e       = t_e
            xxfagn      = [ scale[0] - 5., scale[0] + 5. ]
            xxct        = ct
            xxrm        = rm
            xxta        = [ np.log10(1.), np.log10(9.9) ]      #   using 1. as lower range although library goes down to 0.1
            xxthfr06    = [ np.log10(5.), np.log10(56.) ]
            xxfpol      = [ scale[0] - 5., scale[0] + 5. ]
            xxpolt      = polt
    # 
            if flag==1:
               xxthst16=[ np.log10(82.), np.log10(88.) ]
    #
            host_gal=kwargs.get('host_gal')  
            if host_gal=='no':
                   xxfdisc=[ -20., -19. ]
    #
            starburst_gal=kwargs.get('starburst_gal')  
            if starburst_gal=='no':
                   xxfsb=[ -20., -19. ] 
    #       
            AGN_gal=kwargs.get('AGN_gal')  
            if AGN_gal=='no':
                   xxfagn=[ -20., -19. ]  
    # 
            polar=kwargs.get('polar')  
            if polar=='no':
                   xxfpol=[ -20., -19. ]
        #  
            disc=np.empty(1,dtype=[('xxfdisc','f8',(2,)),('xxtv','f8',(2,))
            ,('xxpsi','f8',(2,)),('xxcirr_tau','f8',(2,)),('xxiview','f8',(2,))   
            ,('xxfsb','f8',(2,)),('xxtau_v','f8',(2,)),('xxage','f8',(2,)),('xxt_e','f8',(2,))            
            ,('xxfagn','f8',(2,)),('xxct','f8',(2,)),('xxrm','f8',(2,)),('xxta','f8',(2,))
            ,('xxthfr06','f8',(2,)),('xxfpol','f8',(2,)),('xxpolt','f8',(2,))
            ])
            disc['xxfdisc']=xxfdisc
            disc['xxtv']=xxtv
            disc['xxpsi']=xxpsi
            disc['xxcirr_tau']=xxcirr_tau
            disc['xxiview']=xxiview
            disc['xxfsb']=xxfsb
            disc['xxtau_v']=xxtau_v
            disc['xxage']=xxage
            disc['xxt_e']=xxt_e
            disc['xxfagn']=xxfagn
            disc['xxct']=xxct
            disc['xxrm']=xxrm
            disc['xxta']=xxta
            disc['xxthfr06']=xxthfr06
            disc['xxfpol']=xxfpol
            disc['xxpolt']=xxpolt
    #
            return disc
# 
# 
        elif hostType==2 and AGNType==3:

#    Parameters for a disc run using the SKIRTOR AGN model
#
            xxfdisc     = [ scale[1] - 5., scale[1] + 5. ]
            xxtv        = [ np.log10(0.01), np.log10(28.) ]
            xxpsi       = [ np.log10(1.), np.log10(9.) ] 
            xxcirr_tau  = [ np.log10(0.51e9), np.log10(7.9e9) ]  
            xxiview     = [ np.log10(1.), np.log10(23.) ]
            xxfsb       = [ scale[0] - 5., scale[0] + 5. ]
            xxtau_v     = tau_v
            xxage       = age
            xxt_e       = t_e
            xxfagn      = [ scale[0] - 5., scale[0] + 5. ]
            xxoa        = oa
            xxrr        = rr
            xxtt        = tt
            xxthst16    = [ np.log10(5.), np.log10(60.) ]
            xxfpol      = [ scale[0] - 5., scale[0] + 5. ]#
            xxpolt      = polt
    # 
            if flag==1:
               xxthst16=[ np.log10(82.), np.log10(88.) ]
    #
            host_gal=kwargs.get('host_gal')  
            if host_gal=='no':
                   xxfdisc=[ -20., -19. ]
    #
            starburst_gal=kwargs.get('starburst_gal')  
            if starburst_gal=='no':
                   xxfsb=[ -20., -19. ] 
    #       
            AGN_gal=kwargs.get('AGN_gal')  
            if AGN_gal=='no':
                   xxfagn=[ -20., -19. ]  
    # 
            polar=kwargs.get('polar')  
            if polar=='no':
                   xxfpol=[ -20., -19. ]
        #  
            disc=np.empty(1,dtype=[('xxfdisc','f8',(2,)),('xxtv','f8',(2,))
            ,('xxpsi','f8',(2,)),('xxcirr_tau','f8',(2,)),('xxiview','f8',(2,))   
            ,('xxfsb','f8',(2,)),('xxtau_v','f8',(2,)),('xxage','f8',(2,)),('xxt_e','f8',(2,))            
            ,('xxfagn','f8',(2,)),('xxoa','f8',(2,)),('xxrr','f8',(2,)),('xxtt','f8',(2,))
            ,('xxthst16','f8',(2,)),('xxfpol','f8',(2,)),('xxpolt','f8',(2,))
            ])
            disc['xxfdisc']=xxfdisc
            disc['xxtv']=xxtv
            disc['xxpsi']=xxpsi
            disc['xxcirr_tau']=xxcirr_tau
            disc['xxiview']=xxiview
            disc['xxfsb']=xxfsb
            disc['xxtau_v']=xxtau_v
            disc['xxage']=xxage
            disc['xxt_e']=xxt_e
            disc['xxfagn']=xxfagn
            disc['xxoa']=xxoa
            disc['xxrr']=xxrr
            disc['xxtt']=xxtt
            disc['xxthst16']=xxthst16
            disc['xxfpol']=xxfpol
            disc['xxpolt']=xxpolt
    #
            return disc
# 
# 
        elif hostType==2 and AGNType==4:

#    Parameters for a disc run using the Siebenmorgen AGN model
#
            xxfdisc     = [ scale[1] - 5., scale[1] + 5. ]
            xxtv        = [ np.log10(0.01), np.log10(28.) ]
            xxpsi       = [ np.log10(1.), np.log10(9.) ] 
            xxcirr_tau  = [ np.log10(0.51e9), np.log10(7.9e9) ]  
            xxiview     = [ np.log10(1.), np.log10(23.) ]
            xxfsb       = [ scale[0] - 5., scale[0] + 5. ]
            xxtau_v     = tau_v
            xxage       = age
            xxt_e       = t_e
            xxfagn      = [ scale[0] - 5., scale[0] + 5. ]
            xxvc        = vc
            xxac        = ac
            xxad        = ad
            xxth        = [ np.log10(0.1), np.log10(44.9) ]
            xxfpol      = [ scale[0] - 5., scale[0] + 5. ]
            xxpolt      = polt
    # 
            if flag==1:
               xxth=[ np.log10(45.1), np.log10(89.9)]
    #
            host_gal=kwargs.get('host_gal')  
            if host_gal=='no':
                   xxfdisc=[ -20., -19. ]
    #
            starburst_gal=kwargs.get('starburst_gal')  
            if starburst_gal=='no':
                   xxfsb=[ -20., -19. ] 
    #       
            AGN_gal=kwargs.get('AGN_gal')  
            if AGN_gal=='no':
                   xxfagn=[ -20., -19. ]  
    # 
            polar=kwargs.get('polar')  
            if polar=='no':
                   xxfpol=[ -20., -19. ]
        #  
            disc=np.empty(1,dtype=[('xxfdisc','f8',(2,)),('xxtv','f8',(2,))
            ,('xxpsi','f8',(2,)),('xxcirr_tau','f8',(2,)),('xxiview','f8',(2,))   
            ,('xxfsb','f8',(2,)),('xxtau_v','f8',(2,)),('xxage','f8',(2,)),('xxt_e','f8',(2,))            
            ,('xxfagn','f8',(2,)),('xxvc','f8',(2,)),('xxac','f8',(2,)),('xxad','f8',(2,))
            ,('xxth','f8',(2,)),('xxfpol','f8',(2,)),('xxpolt','f8',(2,))
            ])
            disc['xxfdisc']=xxfdisc
            disc['xxtv']=xxtv
            disc['xxpsi']=xxpsi
            disc['xxcirr_tau']=xxcirr_tau
            disc['xxiview']=xxiview
            disc['xxfsb']=xxfsb
            disc['xxtau_v']=xxtau_v
            disc['xxage']=xxage
            disc['xxt_e']=xxt_e
            disc['xxfagn']=xxfagn
            disc['xxvc']=xxvc
            disc['xxac']=xxac
            disc['xxad']=xxad
            disc['xxth']=xxth
            disc['xxfpol']=xxfpol
            disc['xxpolt']=xxpolt
    #
            return disc

def log_prior_host(theta,models_fnu,host):

    for i in range(len(theta)):
 
      if host[0][i][0] > theta[i]  or  host[0][i][1] < theta[i]:  
           return -np.inf

    return 0.0


def log_likelihood_host(theta, x, y, yerr, flag, models_fnu,hostType,AGNType,wws):

    if hostType==1 and AGNType==1:
        fsph,tvv,psi,cirr_tau,fsb,tau_v,age,t_e,fagn,r2tor1,tau_uv,theta_1,theta_v,fpol,polt = theta 
        
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**r2tor1,10.**tau_uv,10.**theta_1,10.**theta_v,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fsph,10.**tvv,10.**psi,10.**cirr_tau,1.,models_fnu,hostType,wws)
        
    if hostType==1 and AGNType==2:
        fsph,tvv,psi,cirr_tau,fsb,tau_v,age,t_e,fagn,ct,rm,ta,thfr06,fpol,polt = theta 
        
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**ct,10.**rm,10.**ta,10.**thfr06,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fsph,10.**tvv,10.**psi,10.**cirr_tau,1.,models_fnu,hostType,wws)

        
    if hostType==1 and AGNType==3:
        fsph,tvv,psi,cirr_tau,fsb,tau_v,age,t_e,fagn,oa,rr,tt,thst16,fpol,polt = theta 
        
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**oa,10.**rr,10.**tt,10.**thst16,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fsph,10.**tvv,10.**psi,10.**cirr_tau,1.,models_fnu,hostType,wws)
        
    if hostType==1 and AGNType==4:
        fsph,tvv,psi,cirr_tau,fsb,tau_v,age,t_e,fagn,vc,ac,ad,th,fpol,polt = theta 
        
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**vc,10.**ac,10.**ad,10.**th,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fsph,10.**tvv,10.**psi,10.**cirr_tau,1.,models_fnu,hostType,wws)
        
    if hostType==2 and AGNType==1:
        fdisc,tv,psi,cirr_tau,iview,fsb,tau_v,age,t_e,fagn,r2tor1,tau_uv,theta_1,theta_v,fpol,polt = theta 
    
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**r2tor1,10.**tau_uv,10.**theta_1,10.**theta_v,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fdisc,10.**tv,10.**psi,10.**cirr_tau,10.**iview,models_fnu,hostType,wws)
        
    if hostType==2 and AGNType==2:
        fdisc,tv,psi,cirr_tau,iview,fsb,tau_v,age,t_e,fagn,ct,rm,ta,thfr06,fpol,polt = theta 
    
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**ct,10.**rm,10.**ta,10.**thfr06,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fdisc,10.**tv,10.**psi,10.**cirr_tau,10.**iview,models_fnu,hostType,wws)
        
    if hostType==2 and AGNType==3:
        fdisc,tv,psi,cirr_tau,iview,fsb,tau_v,age,t_e,fagn,oa,rr,tt,thst16,fpol,polt = theta 
    
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**oa,10.**rr,10.**tt,10.**thst16,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fdisc,10.**tv,10.**psi,10.**cirr_tau,10.**iview,models_fnu,hostType,wws)
        
    if hostType==2 and AGNType==4:
        fdisc,tv,psi,cirr_tau,iview,fsb,tau_v,age,t_e,fagn,vc,ac,ad,th,fpol,polt = theta 
    
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**vc,10.**ac,10.**ad,10.**th,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fdisc,10.**tv,10.**psi,10.**cirr_tau,10.**iview,models_fnu,hostType,wws)
            
    model_func=interpolate.interp1d(mmodel[0],mmodel[1])

    model_all=model_func(x)

    model_func_host=interpolate.interp1d(wws,mmodel_host[1])

    model_host=model_func_host(x)

    model = model_all + model_host
    
    sigma2 = yerr ** 2
    
    d = geek.subtract(model, y + 3.*yerr)
    
    if (any(excess>0.0 for excess in d*(flag+0.0)) == True):
        return -np.inf
       
    return -0.5 * np.sum((1. - flag)*(y - model) ** 2 / sigma2)


def log_probability_host(theta, x, y, yerr, flag, models_fnu, host,hostType,AGNType,wws):
    lp = log_prior_host(theta, models_fnu, host)

    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_host(theta, x, y, yerr, flag, models_fnu, hostType, AGNType,wws)


def chi_squared_host(theta, x, y, yerr, flag, models_fnu, hostType,AGNType,wws):

    
    if hostType==1 and AGNType==1:
        fsph,tvv,psi,cirr_tau,fsb,tau_v,age,t_e,fagn,r2tor1,tau_uv,theta_1,theta_v,fpol,polt = theta 
 
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**r2tor1,10.**tau_uv,10.**theta_1,10.**theta_v,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fsph,10.**tvv,10.**psi,10.**cirr_tau,1.,models_fnu,hostType,wws)
    
    if hostType==1 and AGNType==2:
        fsph,tvv,psi,cirr_tau,fsb,tau_v,age,t_e,fagn,ct,rm,ta,thfr06,fpol,polt = theta 
        
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**ct,10.**rm,10.**ta,10.**thfr06,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fsph,10.**tvv,10.**psi,10.**cirr_tau,1.,models_fnu,hostType,wws)
        
        
    if hostType==1 and AGNType==3:
        fsph,tvv,psi,cirr_tau,fsb,tau_v,age,t_e,fagn,oa,rr,tt,thst16,fpol,polt = theta 
        
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**oa,10.**rr,10.**tt,10.**thst16,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fsph,10.**tvv,10.**psi,10.**cirr_tau,1.,models_fnu,hostType,wws)
        
    if hostType==1 and AGNType==4:
        fsph,tvv,psi,cirr_tau,fsb,tau_v,age,t_e,fagn,vc,ac,ad,th,fpol,polt = theta 
        
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**vc,10.**ac,10.**ad,10.**th,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fsph,10.**tvv,10.**psi,10.**cirr_tau,1.,models_fnu,hostType,wws)
        
    if hostType==2 and AGNType==1:
        fdisc,tv,psi,cirr_tau,iview,fsb,tau_v,age,t_e,fagn,r2tor1,tau_uv,theta_1,theta_v,fpol,polt = theta 
        
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**r2tor1,10.**tau_uv,10.**theta_1,10.**theta_v,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fdisc,10.**tv,10.**psi,10.**cirr_tau,10.**iview,models_fnu,hostType,wws)
        
    if hostType==2 and AGNType==2:
        fdisc,tv,psi,cirr_tau,iview,fsb,tau_v,age,t_e,fagn,ct,rm,ta,thfr06,fpol,polt = theta 
    
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**ct,10.**rm,10.**ta,10.**thfr06,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fdisc,10.**tv,10.**psi,10.**cirr_tau,10.**iview,models_fnu,hostType,wws)
        
    if hostType==2 and AGNType==3:
        fdisc,tv,psi,cirr_tau,iview,fsb,tau_v,age,t_e,fagn,oa,rr,tt,thst16,fpol,polt = theta 
    
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**oa,10.**rr,10.**tt,10.**thst16,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fdisc,10.**tv,10.**psi,10.**cirr_tau,10.**iview,models_fnu,hostType,wws)
        
    if hostType==2 and AGNType==4:
        fdisc,tv,psi,cirr_tau,iview,fsb,tau_v,age,t_e,fagn,vc,ac,ad,th,fpol,polt = theta 
    
        mmodel = synthesis_routine_SMART(10.**fsb,10.**tau_v,10.**age,10.**t_e,10.**fagn,
              10.**vc,10.**ac,10.**ad,10.**th,10.**fpol,10.**polt,AGNType)

        mmodel_host = synthesis_routine_host_SMART(10.**fdisc,10.**tv,10.**psi,10.**cirr_tau,10.**iview,models_fnu,hostType,wws)        

    model_func=interpolate.interp1d(mmodel[0],mmodel[1])

    model_all=model_func(x)

    model_func_host=interpolate.interp1d(wws,mmodel_host[1])

    model_host=model_func_host(x)

    model = model_all + model_host    

    sigma2 = (0.15*model) ** 2 + (0.15*y) ** 2
    return np.sum((1. - flag)*(y - model) ** 2 / sigma2)
