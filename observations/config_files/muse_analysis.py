import numpy as np
import matplotlib.pyplot as plt

from mpdaf.obj import Cube
from mpdaf.drs import PixTable
from specutils import Spectrum1D
import astropy.units as u 

from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

from specutils.manipulation import FluxConservingResampler
#xu_104_cube = Cube('/N/slate/cjagosti/muse/MUSE_data/xu-104/ADP.2020-12-23T12:45:53.705.fits')
#xu_104_cube.z = 0.118
xr_31_cube = Cube('/home/caug/config_files/xr31_starlighted.fits',ext=1)#.)#'/N/slate/cjagosti/muse/MUSE_data/xr-31/ADP.2021-01-21T03:24:21.579.fits')
xr_31_cube = Cube('/home/caug/config_files/xr31_skysub_stamp.fits',ext=1)#.)#'/N/slate/cjagosti/muse/MUSE_data/xr-31/ADP.2021-01-21T03:24:21.579.fits')


wcs = WCS(xr_31_cube.data_header)
plt.subplot(projection=imwcs)

xr_31_cube.red_bptlines_wl = bptlines_wl*xr_31_cube.z+bptlines_wl
def y1_kauffmann(xvals):
    yline1_kauffmann = 10**(0.61 / (xvals - 0.05) + 1.3) 
    return yline1_kauffmann


from ppxf.ppxf import ppxf



def getwave(spec):
    start = spec.wave.get_crval()
    delta = spec.wave.get_step()
    wave = np.arange(len(spec.data.data))*delta+start
    return wave
spec_104 = xu_104_cube[:,153,161]
spec_mask_104 = np.where(np.isfinite(spec_104.data.data))[0]
spec_31 = xr_31_cube[:,149,161]
spec_mask_31 = np.where(np.isfinite(spec_31.data.data))[0]


spec_104.wavelengths = getwave(spec_104)
spec_31.wavelengths = getwave(spec_31)

shift_wl_104 = spec_104.wavelengths/(1+xu_104_cube.z)
shift_wl_31 = spec_31.wavelengths/(1+xr_31_cube.z)

fluxcon = FluxConservingResampler()

spe_104 = Spectrum1D(spectral_axis=shift_wl_104[spec_mask_104]*u.AA, flux=spec_104.data.data[spec_mask_104]*spec_104.unit)

spe_31 = Spectrum1D(spectral_axis=shift_wl_31[spec_mask_31]*u.AA, flux=spec_31.data.data[spec_mask_31]*spec_31.unit)

new_disp_grid_104 = np.arange(min(shift_wl_104), max(shift_wl_104), 1.5)*u.AA

new_disp_grid_31 = np.arange(min(shift_wl_31), max(shift_wl_31), 1.5)*u.AA


new_spec_fluxcon_104 = fluxcon(spe, new_disp_grid_104)
new_spec_fluxcon_31 = fluxcon(spe, new_disp_grid_31)


wave_inp = np.array(new_spec_fluxcon_31.spectral_axis)# np.array(spec_31.wavelengths) # 
flux_inp = np.array(new_spec_fluxcon_31.flux)# np.array(spec_31.data.data)#


