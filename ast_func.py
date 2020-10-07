import astropy.cosmology as apc
cosmo = apc.Planck15
import numpy as np
from astropy.coordinates import SkyCoord 
from astropy import units as  u

def getlumfromflux(flux, z):
    '''
    getting the luminosity for a given flux, based on L= 4pi*d^2 *flux
    -uses luminosity distance here so additional factors not necessary
    -in cgs so need to convert lum dist from Mpc to cm 
    '''
    distances=np.array(cosmo.luminosity_distance(z))* (3.086e+24) #Mpc to cm
    lum = 4*np.pi*(distances**2)*flux
    return lum

def comp_skydist(ra1,dec1,ra2,dec2):
    '''
    computing distance between two objects on the sky
    '''
    c1 = SkyCoord(ra1,dec1, frame='icrs')
    c2 = SkyCoord(ra2,dec2, frame='icrs')
    return c1.separation(c2)

def conv_ra(ra):
    '''
    convert ra for plotting spherical aitoff projection.
    '''
    copra = np.copy(ra)
    for i in range(len(ra)):
        if copra[i] > 270:
            copra[i] =- 360 + copra[i]

    return (copra)*(-1)