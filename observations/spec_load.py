import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf
from specutils import Spectrum1D

data_fold = './reduced_data/'
xr82 = 'CombXR82.ms.fits'

#spec= Spectrum1D.read(data_fold+xr82)
class MSObject:
    def __init__(self, filename):
        fil = pf.open(filename)
        specs = []
        wavelengths = []
        for spec in fil:
            wavelength_units = u.Unit(spec.header.get('TUNIT1', 'um'))
            
xr82_img = pf.open(data_fold+xr82)[0]
xr82_data = xr82_img.data
xr82_header = xr82_img.header