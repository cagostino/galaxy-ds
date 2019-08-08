import astropy.cosmology as apc
cosmo = apc.Planck15
import numpy as np
def getlumfromflux(flux, z):
    distances=np.array(cosmo.luminosity_distance(z))* (3.086e+24) #Mpc to cm
    lum = 4*np.pi*(distances**2)*flux
    return lum