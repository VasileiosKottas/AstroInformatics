from star_data_starb_agn_sph_dust import *

if __name__ == "__main__":

    ULIRG = (
        'U5652',
        'U16526',
        'U5150',
        'U5632'
    )
    
    num_galaxies = 10000
    
    create_data(num_galaxies, plot=True, ULIRG = ULIRG[3], agntype = 'cygnus', photometry = True, log_wave = True)