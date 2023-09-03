from os import path, makedirs
from astropy import log

# In/out & maths
import numpy as np
import pandas as pd
from astropy.table import Table

# Plots
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

# For spec reading & preprocessing
from astropy.io import fits
import astropy.units as u
from pycasso2.starlight.io import read_output_tables_v4
from pycasso2.resampling import resample_cube
from pycasso2.resampling.core import *
#from spectral_cube import SpectralCube

from pycasso2.resampling import vac2air
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord

# For dobby
from pycasso2.dobby.fitting import fit_strong_lines
from pycasso2.dobby.utils import plot_el, read_summary_from_file, get_el_info
from scipy.interpolate import interp1d
import astropy.io.fits as pf
from pycasso2.resampling.core import *
from pycasso2 import cosmology as pyc2c
from pycasso2.starlight.io import *
from photutils.centroids import centroid_com

import sys
sys.path.append("../..")
from Fits_set import *
from ast_func import *
from ELObj import *

log.setLevel('INFO')

from mpdaf.obj import Cube
#import bces.bces as BCES

arcsec =  1./3600 #in degrees
H0 = 70.0  # km / s / Mpc
omega0 = 0.300
flux_unit=1e-20*u.erg/u.s/u.cm/u.cm/u.AA
c = 299792.458  # km / sinterp1d_spectra

muse_wls = np.array([4650, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9350])
muse_res = np.array([1609, 1750, 1978, 2227, 2484, 2737, 2975, 3183, 3350, 3465, 3506])
resampled_wl_starlight = np.arange(4750, 8951,1.25)

        
muse_disp = muse_wls/muse_res/2.355 #/2.5
wl_interp = interp1d(muse_wls, muse_disp)
muse_wdisp = wl_interp(resampled_wl_starlight)

muse_disp_kms = c/muse_res/2.355
kms_interp = interp1d(muse_wls, muse_disp_kms)
muse_disp_kms_int = kms_interp(resampled_wl_starlight)





def y1_kauffmann(xvals):
    yline1_kauffmann = 10**(0.61 / (xvals - 0.05) + 1.3) 
    return yline1_kauffmann
bptlines_wl = np.array([4861,5007,6563,6583, 6717, 6731, 6300])
def get_dobby_spec(spec, spec_err, spec_flag,spec_cont,wdisp = muse_wdisp, 
                             lamb=[],
                             enable_kin_ties=True,get_el=False,
                            enable_balmer_lim = True, load=False, 
                            plot=False, save=True, outfile='', save_plot=True):
    '''
    Parameters
    ----------
    spec : TYPE
        DESCRIPTION.
    spec_err : TYPE
        DESCRIPTION.
    spec_cont : TYPE
        DESCRIPTION.
    spec_flag : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    if len(lamb)==0:
        wave_rang = spec.wave.get_range()
        lamb = np.arange(wave_rang[0], wave_rang[1]+spec.wave.get_step(), spec.wave.get_step())*1e10

    overwrite = True
    
    model = 'gaussian'
    enable_kin_ties = enable_kin_ties
    # Do not allow Ha/Hb < 2.6
    enable_balmer_lim = enable_balmer_lim
    # Degree for Legendre polynomial fits in the local pseudocontinuum
    degree = 16
    
    # Instrumental dispersion spectrum in km/s? (True = yes, False = in angstroms)
    vd_kms = False
    model='gaussian'
    #fits_suffix='test'
    #_k=1*enable_kin_ties
    #_b = 1*enable_balmer_lim
    #suffix = 'El%sk%ib%i' % (_m, _k, _b)    
    if not load:
        el = fit_strong_lines(lamb, spec.data, np.ma.array(spec_cont.data), spec_err.data, vd_inst = wdisp,
                            kinematic_ties_on = enable_kin_ties, 
                            balmer_limit_on = enable_balmer_lim, 
                            model = model,
                            saveAll = save, 
                            outname = outfile, 
                            outdir = '', 
                            overwrite = True, 
                            degree=degree,
                            vd_kms = vd_kms)
    if plot and not load:
        plot_el(lamb, spec.data, el )
        if save_plot:
            plt.savefig(outfile+'.pdf', dpi=250, format='pdf', bbox_inches='tight')
            plt.savefig(outfile+'.png', dpi=250, format='png', bbox_inches='tight')
            plt.close()
    if get_el:
        return lamb, el
    
    elines, pseudo_cont_spec = read_summary_from_file(outfile+'.hdf5')
    return lamb, elines, pseudo_cont_spec
def get_bpt_from_dobby(dob_out, het=False):
    lamb, elines, pseudo_cont_spec = dob_out
    out = []
    lines = [4861,5007,6563,6584,6300,6716,6731]
    if het:
        lines.append(3727)
    for i, line in enumerate(lines):
        flux = get_el_info(elines, line, 'El_F')[0]
        vd = get_el_info(elines, line, 'El_vd')[0]
        vdins = get_el_info(elines, line, 'El_vdins')[0]
        l0 = get_el_info(elines, line, 'El_l0')[0]
        lcrms = get_el_info(elines, line, 'El_lcrms')[0]
        vdtot_ang = l0*np.sqrt(vd**2+vdins**2)/c
        A = flux/(np.sqrt(2*np.pi) *vdtot_ang)
        dl = 1
        eF = lcrms*np.sqrt(6.*vdtot_ang*dl)
        sn = flux/eF
        out.append([ [elines, pseudo_cont_spec], flux,eF, sn])
    s2_flux = out[-2][1] +out[-1][1]
    s2_flux_error = np.sqrt(out[-2][1]+out[-1][1])
    s2_sn = s2_flux/s2_flux_error
    out.append([ s2_flux, s2_flux_error, s2_sn])
    n2ha = np.log10(out[3][1]/out[2][1])
    s2ha = np.log10(out[-1][1]/out[2][1])
    o1ha = np.log10(out[4][1]/out[2][1])
    o3hb = np.log10(out[1][1]/out[0][1])
    out.append([o3hb, n2ha,s2ha, o1ha])
    ykau = np.log10(y1_kauffmann(n2ha))
    if np.isinf(o3hb) or np.isinf(n2ha):
        bpt_map = np.nan
    elif n2ha>=0: 
        bpt_map= 2
    elif o3hb > ykau:
        bpt_map=2
    else:
        bpt_map=1
    out.append(bpt_map)
    if out[3][3]>2 and out[2][3]>2 and out[1][3]>2 and out[0][3]>2:
        low_sn=0
    else:
        low_sn=1
    out.append(low_sn)
    return out
    
  

def get_bptlinefluxes_single(spec):       

    try:
        hb = spec.gauss_fit(bptlines_wl[0]-5,bptlines_wl[0]+5)
        hb_flux = hb.flux
        hb_flux_error = hb.err_flux
    except:
        hb = np.nan
        hb_flux = np.nan
        hb_flux_error = np.nan
        
    try:
        o3 = spec.gauss_fit(bptlines_wl[1]-5,bptlines_wl[1]+5)
        o3_flux = o3.flux
        o3_flux_error = o3.err_flux
    except:
        o3 = np.nan
        o3_flux = np.nan
        o3_flux_error = np.nan

    try:
        ha = spec.gauss_fit(bptlines_wl[2]-5,bptlines_wl[2]+5)
        ha_flux = ha.flux
        ha_flux_error = ha.err_flux

    except:
        ha = np.nan
        ha_flux = np.nan
        ha_flux_error = np.nan            
    try:
        n2 = spec.gauss_fit(bptlines_wl[3]-5,bptlines_wl[3]+5)
        n2_flux = n2.flux
        n2_flux_error = n2.err_flux

    except:
        n2 = np.nan
        n2_flux = np.nan
        n2_flux_error = np.nan
    try:
        s2_6717 = spec.gauss_fit(bptlines_wl[4]-5,bptlines_wl[4]+5)
        s2_6717_flux = s2_6717.flux
        s2_6717_flux_error = s2_6717.err_flux

    except:
        s2_6717 = np.nan
        s2_6717_flux= np.nan
        s2_6717_flux_error = np.nan

    try:
        s2_6731 = spec.gauss_fit(bptlines_wl[5]-5,bptlines_wl[5]+5)
        s2_6731_flux = s2_6731.flux
        s2_6731_flux_error = s2_6731.err_flux

    except:
        s2_6731= np.nan
        s2_6731_flux = np.nan
        s2_6731_flux_error = np.nan

    try:
        o1 = spec.gauss_fit(bptlines_wl[6]-5,bptlines_wl[6]+5)
        o1_flux = o1.flux
        o1_flux_error = o1.err_flux

    except:
        o1 = np.nan
        o1_flux = np.nan
        o1_flux_error = np.nan
    out = [[hb,hb_flux, hb_flux_error],
           [ o3, o3_flux, o3_flux_error], 
           [ha,ha_flux, ha_flux_error], 
           [n2, n2_flux, n2_flux_error],
           [s2_6717, s2_6717_flux, s2_6717_flux_error],
           [s2_6731, s2_6731_flux, s2_6731_flux_error],
           [o1, o1_flux, o1_flux_error]]
    return out

def get_bptlinefluxes_cube(cube, cube_err=[], cube_flag=[], cube_cont=[], 
                           method='simple',sncut=1, gal_id='', 
                           i_start=0, j_start=0, i_end = 0, j_end=0,load=False, 
                           het = False, wdisp=muse_wdisp):
    ha_fits = np.empty((cube.data.shape[1],cube.data.shape[2]), dtype=object)
    ha_flux = np.zeros((cube.data.shape[1],cube.data.shape[2]))
    ha_flux_error = np.zeros_like(ha_flux)
    ha_sn = np.zeros_like(ha_flux)
    
    hb_fits =np.empty((cube.data.shape[1],cube.data.shape[2]), dtype=object)
    hb_flux =  np.zeros_like(ha_flux)
    hb_flux_error = np.zeros_like(ha_flux)
    hb_sn = np.zeros_like(ha_flux)
    
    o3_fits =np.empty((cube.data.shape[1],cube.data.shape[2]), dtype=object)
    o3_flux=  np.zeros_like(ha_flux)    
    o3_flux_error = np.zeros_like(ha_flux)
    o3_sn = np.zeros_like(ha_flux)
    
    n2_fits = np.empty((cube.data.shape[1],cube.data.shape[2]), dtype=object)
    n2_flux = np.zeros_like(ha_flux)
    n2_flux_error = np.zeros_like(ha_flux)
    n2_sn = np.zeros_like(ha_flux)

    o1_fits = np.empty((cube.data.shape[1],cube.data.shape[2]), dtype=object)
    o1_flux = np.zeros_like(ha_flux)
    o1_flux_error = np.zeros_like(ha_flux)
    o1_sn = np.zeros_like(ha_flux)

    o2_fits = np.empty((cube.data.shape[1],cube.data.shape[2]), dtype=object)
    o2_flux = np.zeros_like(ha_flux)
    o2_flux_error = np.zeros_like(ha_flux)
    o2_sn = np.zeros_like(ha_flux)

    s2_6717_fits = np.empty((cube.data.shape[1],cube.data.shape[2]), dtype=object)
    s2_6717_flux = np.zeros_like(ha_flux)
    s2_6717_flux_error = np.zeros_like(ha_flux)
    s2_6717_sn = np.zeros_like(ha_flux)

    s2_6731_fits = np.empty((cube.data.shape[1],cube.data.shape[2]), dtype=object)
    s2_6731_flux = np.zeros_like(ha_flux)
    s2_6731_flux_error = np.zeros_like(ha_flux)
    s2_6731_sn = np.zeros_like(ha_flux)

    s2_flux = np.zeros_like(ha_flux)
    s2_flux_error = np.zeros_like(ha_flux)
    s2_sn = np.zeros_like(ha_flux)
    

    n2ha = np.zeros_like(ha_flux)
    s2ha = np.zeros_like(ha_flux)
    o3hb = np.zeros_like(ha_flux)
    o1ha = np.zeros_like(ha_flux)
    bpt_map = np.zeros_like(ha_flux)
    low_sn_mask = np.zeros_like(ha_flux)
    
    
    if j_end == 0:
        j_end = cube.data.shape[2]
        if i_end ==0:
            i_end = cube.data.shape[1]
    
    for i in range(i_start,i_end):
        for j in range(j_start,j_end ):
            spec = cube[:,i,j]
            if method=='simple':
                line_fits = get_bptlinefluxes_single(spec)
            elif method =='dobby':
                out_dir = 'dobby_out/'+gal_id+'/'  
                outname = '%s_%s_%s' % (gal_id, i,j)
                outfile = path.join(out_dir, '%s.hdf5' % outname)  
                
                spec_err = cube_err[:,i,j]
                spec_cont = cube_cont[:,i,j]
                spec_flag = cube_flag[:,i,j]

                dob = get_dobby_spec(spec, 
                                     spec_err, 
                                     spec_flag,
                                     spec_cont,
                                     wdisp = wdisp,
                                     enable_kin_ties=True,
                                     enable_balmer_lim = True, outfile=outfile, load=load   )
                 
                line_fits = get_bpt_from_dobby(dob)
            if not het:
                hb, o3, ha, n2,  o1, s2_6717, s2_6731, s2, bpt_quants, bpt_class, low_sn  = line_fits
            else:
                hb, o3, ha, n2,  o1, s2_6717, s2_6731, o2, s2, bpt_quants, bpt_class, low_sn  = line_fits
                
            hb_,hb_flux_, hb_flux_error_, hb_sn_ = hb
            o3_, o3_flux_, o3_flux_error_, o3_sn_ = o3
            ha_,ha_flux_, ha_flux_error_, ha_sn_ = ha
            n2_, n2_flux_, n2_flux_error_, n2_sn_ = n2
            s2_6717_, s2_6717_flux_, s2_6717_flux_error_, s2_6717_sn_ = s2_6717 
            s2_6731_, s2_6731_flux_, s2_6731_flux_error_, s2_6731_sn_ = s2_6731
            s2_flux_, s2_flux_error_, s2_sn_= s2
            
            if het:
                o2_, o2_flux, o2_flux_error, o2_sn = o2

            o1_, o1_flux_, o1_flux_error_, o1_sn_ = o1
            
            
            hb_fits[i,j] = hb_
            hb_flux[i,j] = hb_flux_
            hb_flux_error[i,j] = hb_flux_error_
            hb_sn[i,j] = hb_sn_
                
            o3_fits[i,j] = o3_
            o3_flux[i,j] = o3_flux_
            o3_flux_error[i,j] = o3_flux_error_
            o3_sn[i,j] = o3_sn_

            ha_fits[i,j] = ha_         
            ha_flux[i,j] = ha_flux_
            ha_flux_error[i,j] = ha_flux_error_
            ha_sn[i,j] = ha_sn_

            n2_fits[i,j] = n2_
            n2_flux[i,j] = n2_flux_
            n2_flux_error[i,j] = n2_flux_error_
            n2_sn[i,j] = n2_sn_

            s2_6717_fits[i,j] = s2_6717_
            s2_6717_flux[i,j] = s2_6717_flux_
            s2_6717_flux_error[i,j] = s2_6717_flux_error_
            s2_6717_sn[i,j] = s2_6717_sn_

            s2_6731_fits[i,j] = s2_6731_
            s2_6731_flux[i,j] = s2_6731_flux_
            s2_6731_flux_error[i,j] = s2_6731_flux_error_
            s2_6731_sn[i,j] = s2_6731_sn_

            s2_flux[i,j] = s2_flux_
            s2_flux_error[i,j] = s2_flux_error_
            s2_sn[i,j] = s2_sn_

            o1_fits[i,j] = o1_
            o1_flux[i,j] = o1_flux_
            o1_flux_error[i,j] = o1_flux_error_
            o1_sn[i,j] = o1_sn_
            
            
            o3hb[i,j] = bpt_quants[0]
            n2ha[i,j] = bpt_quants[1]
            s2ha[i,j] = bpt_quants[2]
            o1ha[i,j] = bpt_quants[3]
            
            bpt_map[i,j] = bpt_class
            low_sn_mask[i,j] = low_sn
        if j_start>0:
            j_start=0
        if i_start-i_end==1:
            break
  
    '''
    hb_fits[low_sn_mask_x, low_sn_mask_y] = np.nan
    ha_fits[low_sn_mask_x, low_sn_mask_y] = np.nan
    n2_fits[low_sn_mask_x, low_sn_mask_y] = np.nan
    o3_fits[low_sn_mask_x, low_sn_mask_y] = np.nan   
    n2ha[low_sn_mask_x, low_sn_mask_y] = np.nan
    o3hb[low_sn_mask_x, low_sn_mask_y] = np.nan
    
    bpt_map[low_sn_mask_x, low_sn_mask_y] = np.nan
    '''
    out = [[hb_fits, hb_flux, hb_flux_error, hb_sn],
            [o3_fits, o3_flux, o3_flux_error, o3_sn],
            [ha_fits, ha_flux,ha_flux_error, ha_sn], 
            [n2_fits, n2_flux, n2_flux_error, n2_sn],
            [o1_fits, o1_flux, o1_flux_error, o1_sn],
            [s2_6717_fits, s2_6717_flux, s2_6717_flux_error, s2_6717_sn], 
            [s2_6731_fits, s2_6731_flux, s2_6731_flux_error, s2_6731_sn],            
            [s2_flux, s2_flux_error, s2_sn], 
            [o3hb, n2ha, s2ha, o1ha], bpt_map,
            low_sn_mask]
    return out
            
class ELines(object):
    def __init__(self):
        pass

from mpl_toolkits.axes_grid1 import make_axes_locatable

class Reduced_Cube(object):
    def __init__(self, 
                 fname, 
                 gal_id, 
                 z, 
                 flux_unit=1e-20,
                 het=False, 
                 load_cutouts=True, 
                 plot=False, 
                 load_apertures=False, 
                 write_sl=False, 
                 load_sl=True, 
                 bco3n=False):
        '''
        Parameters
        ----------
        fname : TYPE
            DESCRIPTION.
        gal_id : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.
        flux_unit : FLOAT, optional
            DESCRIPTION. The default is 1e-20.
        het : BOOLEAN, optional
            DESCRIPTION. The default is False.
        load_cutouts : BOOLEAN, optional
            DESCRIPTION. The default is True.
        plot : BOOLEAN, optional
            DESCRIPTION. The default is False.
        load_aps : BOOLEAN, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        if het:
            wdisp = het_wdisp
        else:
            wdisp = muse_wdisp
        self.wdisp = wdisp
        self.het = het
        self.load_cutouts = load_cutouts
        self.write_sl = write_sl
        self.load_sl = load_sl
        self.load_apertures = load_apertures
        self.bco3n = bco3n
        self.plot = plot
        self.wdisp = wdisp
        
        self.fname=fname
        self.gal_id=gal_id
        self.obs = Cube(fname,ext=1, unit=flux_unit)
        self.z = z
        self.obs_err = Cube(fname,ext=2, unit=flux_unit)
        self.obs_flag = Cube(fname,ext=3)
        self.obs_cont = Cube(fname,ext=4, unit=flux_unit)
        self.obs_im_full = self.obs.get_image((3000,10000))
        #self.obs_im_u = self.obs.get_band_image('SDSS_u')
        self.obs_im_g = self.obs.get_band_image('SDSS_g')
        self.obs_im_r = self.obs.get_band_image('SDSS_r')
        self.obs_im_i = self.obs.get_band_image('SDSS_i')
        self.obs_rgb = make_lupton_rgb(self.obs_im_i.data,self.obs_im_r.data,self.obs_im_g.data)
        self.center = np.unravel_index(np.argmax(self.obs_im_r.data), self.obs_im_r.shape) #centroid_com(self.obs_im_r.data)[::-1]
        xcoord = np.arange(self.obs_im_g.shape[0])
        ycoord = np.arange(self.obs_im_g.shape[1])
        xcentered = self.center[1]-xcoord
        ycentered = self.center[0]-ycoord
        meshx, meshy = np.meshgrid(xcentered, ycentered)
        self.dists_from_center = np.sqrt(meshy**2+meshy**2)        
        self.obs_im_halp = self.obs.get_image((6558, 6568))
        self.obs_im_n2 = self.obs.get_image((6578, 6588))
        self.obs_im_s2 = self.obs.get_image((6710, 6740))
        self.obs_im_hbeta = self.obs.get_image((4856,4866))
        self.obs_im_o3 = self.obs.get_image((5002,5012))
        self.obs_im_o1  = self.obs.get_image((6295, 6305))
        self.obs_rgb_o3_n2_halp = make_lupton_rgb(self.obs_im_n2.data,self.obs_im_halp.data,self.obs_im_o3.data)
        self.obs_rgb_s2_n2_halp = make_lupton_rgb(self.obs_im_halp.data,self.obs_im_n2.data,self.obs_im_s2.data)
        self.obs_rgb_o3_s2_halp = make_lupton_rgb(self.obs_im_s2.data,self.obs_im_halp.data,self.obs_im_o3.data)
        self.obs_rgb_o3_s2_n2 = make_lupton_rgb(self.obs_im_s2.data,self.obs_im_n2.data,self.obs_im_s2.data)

        
        
        #self.obs_im_z = self.obs.get_band_image('SDSS_z')
        self.cont_sub = self.obs-self.obs_cont
        
        self.cont_sub_im_halp = self.cont_sub.get_image((6558, 6568))
        self.cont_sub_im_n2 = self.cont_sub.get_image((6578, 6588))
        self.cont_sub_im_s2 = self.cont_sub.get_image((6710, 6740))
        self.cont_sub_im_hbeta = self.cont_sub.get_image((4856,4866))
        self.cont_sub_im_o3 = self.cont_sub.get_image((5002,5012))
        self.cont_sub_im_o1  = self.cont_sub.get_image((6295, 6305))
        
        self.cont_sub_rgb_o3_n2_halp = make_lupton_rgb(self.cont_sub_im_n2.data,self.cont_sub_im_halp.data,self.cont_sub_im_o3.data)
        self.cont_sub_rgb_s2_n2_halp = make_lupton_rgb(self.cont_sub_im_s2.data,self.cont_sub_im_n2.data,self.cont_sub_im_halp.data)
        self.cont_sub_rgb_o3_s2_halp = make_lupton_rgb(self.cont_sub_im_s2.data,self.cont_sub_im_halp.data,self.cont_sub_im_o3.data)
        self.cont_sub_rgb_o3_s2_n2 = make_lupton_rgb(self.cont_sub_im_s2.data,self.cont_sub_im_n2.data,self.cont_sub_im_o3.data)
        spec_ = self.obs[:,0,0]
        wave_rang_ = spec_.wave.get_range()
        self.lamb = np.arange(wave_rang_[0], wave_rang_[1]+spec_.wave.get_step(), spec_.wave.get_step())*1e10
        if load_apertures:
            self.load_aps()
    def load_aps(self):

        self.spec_35arc, self.cont_35arc, self.spec_sub_35arc, self.spec_sub_35arc_dob, self.spec_err_35arc,self.spec_flag_35arc = self.get_within_radius(1.75, 
                                                                                                                wdisp=self.wdisp, 
                                                                                                                het=self.het, 
                                                                                                                load= load_cutouts, 
                                                                                                                bco3n = self.bco3n,
                                                                                                                plot=self.plot, 
                                                                                                                write_sl=self.write_sl, load_sl=self.load_sl)
       


        self.spec_3arc, self.cont_3arc, self.spec_sub_3arc, self.spec_sub_3arc_dob, self.spec_err_3arc,self.spec_flag_3arc = self.get_within_radius(1.5, 
                                                                                                            wdisp=self.wdisp, 
                                                                                                            het=self.het, 
                                                                                                            load= load_cutouts, 
                                                                                                            bco3n = self.bco3n,
                                                                                                            plot=self.plot, write_sl=self.write_sl, load_sl=self.load_sl)

        self.spec_25arc, self.cont_25arc, self.spec_sub_25arc, self.spec_sub_25arc_dob, self.spec_err_25arc,self.spec_flag_25arc = self.get_within_radius(1.25,
                                                                                                                wdisp=self.wdisp, 
                                                                                                                het=self.het,
                                                                                                                load= load_cutouts, bco3n = self.bco3n,
                                                                                                                plot=self.plot, write_sl=self.write_sl, load_sl=self.load_sl)

        self.spec_2arc, self.cont_2arc, self.spec_sub_2arc, self.spec_sub_2arc_dob , self.spec_err_2arc,self.spec_flag_2arc= self.get_within_radius(1, 
                                                                                                            wdisp=self.wdisp, 
                                                                                                            het=self.het, 
                                                                                                            load= load_cutouts, bco3n = self.bco3n,
                                                                                                            plot=self.plot, write_sl=self.write_sl, load_sl=self.load_sl)
        self.spec_15arc, self.cont_15arc, self.spec_sub_15arc, self.spec_sub_15arc_dob, self.spec_err_15arc,self.spec_flag_15arc = self.get_within_radius(0.75,
                                                                                                                wdisp=self.wdisp, 
                                                                                                                het=self.het, 
                                                                                                                load= load_cutouts,
                                                                                                                bco3n = self.bco3n,
                                                                                                                plot=self.plot, write_sl=self.write_sl, load_sl=self.load_sl)

        self.spec_1arc, self.cont_1arc, self.spec_sub_1arc, self.spec_sub_1arc_dob, self.spec_err_1arc,self.spec_flag_1arc = self.get_within_radius(0.5, 
                                                                                                            wdisp=self.wdisp, 
                                                                                                            het=self.het, 
                                                                                                            load= load_cutouts, bco3n = self.bco3n,
                                                                                                            plot=self.plot, write_sl=self.write_sl, load_sl=self.load_sl)
        self.spec_05arc, self.cont_05arc, self.spec_sub_05arc, self.spec_sub_05arc_dob, self.spec_err_05arc,self.spec_flag_05arc = self.get_within_radius(0.25, 
                                                                                                                wdisp=self.wdisp, 
                                                                                                                het=self.het, 
                                                                                                                load= load_cutouts, bco3n = self.bco3n,
                                                                                                                plot=self.plot, write_sl=self.write_sl, load_sl=self.load_sl)
        
        self.aperture_dobs  = [self.spec_sub_05arc_dob, self.spec_sub_1arc_dob,self.spec_sub_15arc_dob,
                               self.spec_sub_2arc_dob, self.spec_sub_25arc_dob,self.spec_sub_3arc_dob]
        self.aperture_err  = [self.spec_err_05arc, self.spec_err_1arc,self.spec_err_15arc,
                               self.spec_err_2arc, self.spec_err_25arc,self.spec_err_3arc]
        self.aperture_flag  = [self.spec_flag_05arc, self.spec_flag_1arc,self.spec_flag_15arc,
                               self.spec_flag_2arc, self.spec_flag_25arc,self.spec_flag_3arc]

        self.aperture_specs  = [self.spec_05arc, self.spec_1arc,self.spec_15arc,
                               self.spec_2arc, self.spec_25arc,self.spec_3arc]
        self.aperture_conts  = [self.cont_05arc, self.cont_1arc,self.cont_15arc,
                               self.cont_2arc, self.cont_25arc,self.cont_3arc]
        self.aperture_specs_sub = [self.spec_sub_05arc, self.spec_sub_1arc,self.spec_sub_15arc,
                               self.spec_sub_2arc, self.spec_sub_25arc,self.spec_sub_3arc]

    def plot_starlight_cont_comp(self, lamb, spec, cont, filename='', save=True):
        unit= r'Flux [$10^{-20}$ erg/cm$^2$/s/\AA]'
        plt.plot(lamb, spec ,label='Data', color='k')
        plt.plot(lamb, cont ,label='Starlight Fit',color='r')
        plt.xlabel('Wavelength (angstroms)', fontsize=20)
        plt.ylabel(unit, fontsize=20)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/'+filename +'.pdf', bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/'+filename +'.png', bbox_inches='tight', dpi=250, format='png')

        plt.xlim([4840, 4880])


        rang_ = np.where((lamb>4840)&(lamb<4880))[0]

        ymin = np.min(spec[rang_])
        ymax = np.max(spec[rang_])
        plt.ylim([ymin-ymin/3,ymax+ymax/10])        
        plt.tight_layout()
        plt.savefig('plots/'+filename +'_hbeta_zoom.pdf', bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/'+filename +'_hbeta_zoom.png', bbox_inches='tight', dpi=250, format='png')
        plt.xlim([6530, 6600])
        rang_ = np.where((lamb>6530)&(lamb<6600))[0]
        ymin = np.min(spec[rang_])
        ymax = np.max(spec[rang_])
        plt.ylim([ymin-ymin/3,ymax+ymax/10])        

        plt.tight_layout()
        plt.savefig('plots/'+filename +'_halp_zoom.pdf', bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/'+filename +'_halp_zoom.png', bbox_inches='tight', dpi=250, format='png')
        plt.xlim([4940, 5020])
        rang_ = np.where((lamb>4940)&(lamb<5020))[0]
        ymin = np.min(spec[rang_])
        ymax = np.max(spec[rang_])
        plt.ylim([ymin-ymin/3,ymax+ymax/10])        

        plt.tight_layout()
        plt.savefig('plots/'+filename +'_oiii_zoom.pdf', bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/'+filename +'_oiii_zoom.png', bbox_inches='tight', dpi=250, format='png')

        rang_ = np.where((lamb>4840)&(lamb<5020))[0]
        plt.xlim([4840, 5020])
        ymin = np.min(spec[rang_])
        ymax = np.max(spec[rang_])
        plt.ylim([ymin-ymin/3,ymax+ymax/10])        
        plt.text(4861, ymin-(ymax-ymin)/10 , r'H$\beta$')
        plt.text(5007, ymax+(ymax-ymin)/10 , r'[OIII]')
    
        plt.tight_layout()
        plt.savefig('plots/'+filename +'_hbeta_oiii_zoom.pdf', bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/'+filename +'_hbeta_oiii_zoom.png', bbox_inches='tight', dpi=250, format='png')


        plt.close()
        
        
    def get_dobby_fits(self, i_start=0, j_start=0,  i_end=0, j_end=0, load_fits=False, load_full=False, het=False, wdisp=muse_wdisp):           
        self.dobby = ELines()
        if load_full:
            self.assign_fluxes(self.dobby,fluxes=[], save=False, het = het)
        else:
            self.cont_sub_fluxes_dobby = get_bptlinefluxes_cube(self.cont_sub,self.obs_err,self.obs_flag,self.obs_cont, 
                                                       method='dobby', load=load_fits, gal_id=self.gal_id, i_start=i_start, 
                                                       j_start=j_start, i_end=i_end, j_end=j_end, wdisp=wdisp)
            self.assign_fluxes(self.dobby, fluxes= self.cont_sub_fluxes_dobby , save=True, het= het)
    
    def get_simple_fits(self, load=False, het= False):
        if load:
            self.assign_fluxes(self.simple_fits, fluxes=[], save=False, het=het)
            self.assign_fluxes(self.simple_fits_el, fluxes=[], save=False, het=het)
            
        else:     
            self.cont_sub_fluxes_simple = get_bptlinefluxes_cube(self.cont_sub)
            self.obs_fluxes_simple = get_bptlinefluxes_cube(self.obs)
            
            self.simple_fits = ELines()
            self.simple_fits_el = ELines()
        
            self.assign_fluxes(self.simple_fits, self.obs_fluxes_simple)
            self.assign_fluxes(self.simple_fits_el, self.cont_sub_fluxes_simple)



    def assign_fluxes(self, obj,fluxes=[], dobby=True, load=False, save=True , het=False):     
        if dobby:
            ext = '_dob'
        else:
            ext = '_simp'
        if len(fluxes)==0:
            fluxes.append([np.load('./dobby_combined/'+self.gal_id+ext+'_hb_fit.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_hb_flux.npy',allow_pickle=True),            
                           np.load('./dobby_combined/'+self.gal_id+ext+'_hb_flux_error.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_hb_sn.npy',allow_pickle=True)])
            
            fluxes.append([np.load('./dobby_combined/'+self.gal_id+ext+'_o3_fit.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_o3_flux.npy',allow_pickle=True)            ,
                           np.load('./dobby_combined/'+self.gal_id+ext+'_o3_flux_error.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_o3_sn.npy',allow_pickle=True)]),

            fluxes.append([np.load('./dobby_combined/'+self.gal_id+ext+'_ha_fit.npy',allow_pickle=True),                       
                           np.load('./dobby_combined/'+self.gal_id+ext+'_ha_flux.npy',allow_pickle=True),                        
                           np.load('./dobby_combined/'+self.gal_id+ext+'_ha_flux_error.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_ha_sn.npy',allow_pickle=True)])
                
            fluxes.append([np.load('./dobby_combined/'+self.gal_id+ext+'_n2_fit.npy',allow_pickle=True),            
                           np.load('./dobby_combined/'+self.gal_id+ext+'_n2_flux.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_n2_flux_error.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_n2_sn.npy',allow_pickle=True)])
            
            fluxes.append([np.load('./dobby_combined/'+self.gal_id+ext+'_o1_fit.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_o1_flux.npy',allow_pickle=True), 
                           np.load('./dobby_combined/'+self.gal_id+ext+'_o1_flux_error.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_o1_sn.npy',allow_pickle=True)            ])
            
            fluxes.append([np.load('./dobby_combined/'+self.gal_id+ext+'_s2_6717_fit.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_s2_6717_flux.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_s2_6717_flux_error.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_s2_6717_sn.npy',allow_pickle=True)   ])
            
            fluxes.append([np.load('./dobby_combined/'+self.gal_id+ext+'_s2_6731_fit.npy',allow_pickle=True),           
                           np.load('./dobby_combined/'+self.gal_id+ext+'_s2_6731_flux.npy',allow_pickle=True),                           
                           np.load('./dobby_combined/'+self.gal_id+ext+'_s2_6731_flux_error.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_s2_6731_sn.npy',allow_pickle=True)])
            
            fluxes.append([np.load('./dobby_combined/'+self.gal_id+ext+'_s2_flux.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_s2_flux_error.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_s2_sn.npy',allow_pickle=True)])
            
            fluxes.append([np.load('./dobby_combined/'+self.gal_id+ext+'_o3hb.npy',allow_pickle=True),           
                           np.load('./dobby_combined/'+self.gal_id+ext+'_n2ha.npy',allow_pickle=True),            
                           np.load('./dobby_combined/'+self.gal_id+ext+'_s2ha.npy',allow_pickle=True),
                           np.load('./dobby_combined/'+self.gal_id+ext+'_o1ha.npy',allow_pickle=True)])
            
            fluxes.append(np.load('./dobby_combined/'+self.gal_id+ext+'_bpt_map.npy',allow_pickle=True))
            fluxes.append(np.load('./dobby_combined/'+self.gal_id+ext+'_lowsn_mask.npy',allow_pickle=True))
            
        hb_ = fluxes[0]
        obj.hb_fit, obj.hb_flux, obj.hb_flux_error, obj.hb_sn = hb_
        
        o3_ = fluxes[1]

        obj.o3_fit, obj.o3_flux, obj.o3_flux_error, obj.o3_sn = o3_
        
        ha_ = fluxes[2]
        obj.ha_fit, obj.ha_flux, obj.ha_flux_error, obj.ha_sn = ha_
        
        
        
        if type(obj.hb_flux) ==np.float64:
            obj.av_balmer = extinction(obj.ha_flux, obj.hb_flux, agn=True)

            if obj.av_balmer<=0 or obj.hb_flux==0 or obj.ha_flux==0:
                obj.av_balmer=0    
        else:
            obj.av_balmer = extinction(obj.ha_flux, obj.hb_flux, agn=True, zeroed=True)

        obj.o3_flux_corr = dustcorrect(obj.o3_flux, obj.av_balmer, 5007.0)
        obj.o3_flux_error_corr = dustcorrect(obj.o3_flux_error, obj.av_balmer, 5007.0)
        
        obj.o3_lum = np.log10(getlumfromflux(obj.o3_flux/1e20, self.z))

        obj.o3_lum_corr = np.log10(getlumfromflux(obj.o3_flux_corr/1e20, self.z))
        obj.o3_lum_up_corr = np.log10(getlumfromflux( (obj.o3_flux_corr+obj.o3_flux_error_corr)/1e20, self.z))
        obj.o3_lum_down_corr = np.log10(getlumfromflux((obj.o3_flux_corr-obj.o3_flux_error_corr)/1e20, self.z))
        obj.e_o3_lum_up_corr = obj.o3_lum_up_corr-obj.o3_lum_corr
        obj.e_o3_lum_down_corr = obj.o3_lum_corr-obj.o3_lum_down_corr
        
        obj.ha_flux_corr = dustcorrect(obj.ha_flux, obj.av_balmer, 6563.0)
        obj.ha_flux_error_corr = dustcorrect(obj.ha_flux_error, obj.av_balmer, 6563.0)
                
        obj.ha_lum_corr = np.log10(getlumfromflux(obj.ha_flux_corr/1e20, self.z))
        obj.ha_lum_up_corr = np.log10(getlumfromflux( (obj.ha_flux_corr+obj.ha_flux_error_corr)/1e20, self.z))
        obj.ha_lum_down_corr = np.log10(getlumfromflux((obj.ha_flux_corr-obj.ha_flux_error_corr)/1e20, self.z))
        obj.e_ha_lum_up_corr = obj.ha_lum_up_corr-obj.ha_lum_corr
        obj.e_ha_lum_down_corr = obj.ha_lum_corr-obj.ha_lum_down_corr
        
        
        n2_ = fluxes[3]
        obj.n2_fit, obj.n2_flux, obj.n2_flux_error, obj.n2_sn = n2_
        
        s2_6717_ = fluxes[5]

        obj.s2_6717_fit, obj.s2_6717_flux, obj.s2_6717_flux_error, obj.s2_6717_sn = s2_6717_
        s2_6731_ = fluxes[6]

        obj.s2_6731_fit, obj.s2_6731_flux, obj.s2_6731_flux_error, obj.s2_6731_sn = s2_6731_
        
        s2_ = fluxes[7]
        obj.s2_flux, obj.s2_flux_error, obj.s2_sn = s2_
        
        o1_ = fluxes[4]
        obj.o1_fit, obj.o1_flux, obj.o1_flux_error, obj.o1_sn = o1_
        
        bpt_quants_ = fluxes[8]      
        obj.o3hb, obj.n2ha, obj.s2ha, obj.o1ha = bpt_quants_
        
        obj.bpt_map_ = fluxes[9]
        
        obj.low_sn_mask_ = fluxes[10]
        
        if save:
            np.save('./dobby_combined/'+self.gal_id+ext+'_hb_fit',obj.hb_fit)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_hb_flux',obj.hb_flux)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_hb_flux_error',obj.hb_flux_error)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_hb_sn',obj.hb_sn)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_ha_fit',obj.ha_fit)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_ha_flux',obj.ha_flux)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_ha_flux_error',obj.ha_flux_error)
            np.save('./dobby_combined/'+self.gal_id+ext+'_ha_sn',obj.ha_sn)
            np.save('./dobby_combined/'+self.gal_id+ext+'_o3_fit',obj.o3_fit)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_o3_flux',obj.o3_flux)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_o3_flux_error',obj.o3_flux_error)
            np.save('./dobby_combined/'+self.gal_id+ext+'_o3_sn',obj.o3_sn)
            np.save('./dobby_combined/'+self.gal_id+ext+'_n2_fit',obj.n2_fit)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_n2_flux',obj.n2_flux)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_n2_flux_error',obj.n2_flux_error)
            np.save('./dobby_combined/'+self.gal_id+ext+'_n2_sn',obj.n2_sn)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_6717_fit',obj.s2_6717_fit)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_6717_flux',obj.s2_6717_flux)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_6717_flux_error',obj.s2_6717_flux_error)
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_6717_sn',obj.s2_6717_sn)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_6731_fit',obj.s2_6731_fit)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_6731_flux',obj.s2_6731_flux)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_6731_flux_error',obj.s2_6731_flux_error)
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_6731_sn',obj.s2_6731_sn)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_flux',obj.s2_flux)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_flux_error',obj.s2_flux_error)
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2_sn',obj.s2_sn)
            
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_o1_fit',obj.o1_fit)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_o1_flux',obj.o1_flux)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_o1_flux_error',obj.o1_flux_error)
            np.save('./dobby_combined/'+self.gal_id+ext+'_o1_sn',obj.o1_sn)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_o3hb',obj.o3hb)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_n2ha',obj.n2ha)
            
            np.save('./dobby_combined/'+self.gal_id+ext+'_s2ha',obj.s2ha)
            np.save('./dobby_combined/'+self.gal_id+ext+'_o1ha',obj.o1ha)
            np.save('./dobby_combined/'+self.gal_id+ext+'_bpt_map', obj.bpt_map_)
            np.save('./dobby_combined/'+self.gal_id+ext+'_lowsn_mask', obj.low_sn_mask_)
    def get_single_dobby_fit(self, spec, err, flag, cont, outfile='', het=False, load=False, wdisp=[],lamb=[], plot=False, save_plot=True):
        spec_dobby = get_dobby_spec(spec, 
                                     err, 
                                     flag ,
                                     cont,
                                     wdisp,
                                     lamb=lamb,
                                     enable_kin_ties=True,
                                     enable_balmer_lim = True, 
                                     outfile=outfile,
                                     load=load, plot=plot, save_plot=save_plot)
        spec_fluxes = get_bpt_from_dobby(spec_dobby, het=het )
        dob_ = ELines()
        self.assign_fluxes(dob_, fluxes= spec_fluxes, save=False, het = het)
        return spec_dobby, dob_            
    def get_within_radius(self, size,xoff=0, yoff=0, het=False, load=False, wdisp=[], plot=False, write_sl=False, load_sl=False, bco3n = False):
        
        if het:
            dy=0.3
            dx=0.36
        else:
            dx=dy=.2
        xoffset = round(xoff/dx)
        yoffset = round(yoff/dy)
        
        outfile='./dobby_out_combined/'+self.gal_id+'_'+str(size*2)+'arc'
        dummyspec = self.obs.subcube_circle_aperture((int(self.center[0]+yoffset), int(self.center[1]+xoffset)),float(size), unit_center=None)
        dummyerr = self.obs_err.subcube_circle_aperture((int(self.center[0]+yoffset), int(self.center[1]+xoffset)),float(size), unit_center=None)
        
        dummyim = dummyspec.get_band_image('SDSS_r')
        '''
        wcs = WCS(self.obs.data_header)
        #center = self.wcs.sky2pix(center, unit=unit_center)[0]
        wcs= wcs.dropaxis(2)
        plt.figure()
        plt.subplot(projection=wcs)
        plt.imshow(dummyim.data, origin='lower', cmap='plasma')
        plt.xlabel('RA')
        plt.ylabel('Dec.')
        '''
        
      
        
        spec = self.obs.aperture((self.center[0]+yoffset, self.center[1]+xoffset),size, unit_center=None)
        wave_rang = spec.wave.get_range()
        lamb = np.arange(wave_rang[0], wave_rang[1]+spec.wave.get_step(), spec.wave.get_step())*1e10
        val = np.where(~dummyim.mask.flatten())
        n_val = val[0].size
        
        dummyerr_ij = np.zeros(len(dummyerr[:,0,0].data))
        print(dummyerr.shape)
        for i in range(0,dummyerr.shape[1]):
            for j in range(0, dummyerr.shape[2]):
                if ~dummyim.mask[i,j]:
                    dummyerr_ij = dummyerr_ij+ dummyerr[:, i,j].data**2
        err= np.sqrt(dummyerr_ij)/np.sqrt(n_val)
        print(np.mean(err))
        print(err)
        #err = self.obs_err.aperture((self.center[0]+yoffset, self.center[1]+xoffset) , size, unit_center=None)/np.sqrt(n_val)
        
        flag = self.obs_flag.aperture((self.center[0]+yoffset, self.center[1]+xoffset) , size, unit_center=None)
        if write_sl:
            write_starlight(lamb, spec.data, err.data, flag.data, outfile+'_starlight_in.txt')
        if bco3n:
            outfile = outfile+'_bco3n'
                            
        if load_sl:
             out, spec, cont, fwei, spec_sub = load_starlight(outfile+'_starlight_in.out')
        else:
            cont = self.obs_cont.aperture((self.center[0]+yoffset, self.center[1]+xoffset), size, unit_center=None)
            spec_sub = self.cont_sub.aperture((self.center[0]+yoffset, self.center[1]+xoffset), size, unit_center=None)                
            
    
        spec_sub_dobby, dob_ =  self.get_single_dobby_fit(spec_sub, 
                                 err, 
                                 flag ,
                                 cont,
                                 lamb=lamb,
                                 wdisp=wdisp,
                                 outfile=outfile, 
                                 load=load, plot=plot)

        return spec, cont, spec_sub, dob_, err, flag
    def get_within_square(self, size,xoff=0, yoff=0, het=False, load=False, wdisp=[], plot=False):
        
        if het:
            dy=0.3
            dx=0.36
        else:
            dx=dy=.2
        xoffset = round(xoff/dx)
        yoffset = round(yoff/dy)        
        
        spec = self.obs.subcube((self.center[0]+xoff, self.center[1]+yoff),size, unit_center=None).copy().mean(axis=(1,2))
        cont = self.obs_cont.subcube((self.center[0]+xoff, self.center[1]+yoff), size, unit_center=None).copy().mean(axis=(1,2))
        spec_sub = self.cont_sub.subcube((self.center[0]+xoff, self.center[1]+yoff), size, unit_center=None).copy().mean(axis=(1,2))                
        err = self.obs_err.subcube((self.center[0]+xoff, self.center[1]+yoff) , size, unit_center=None).copy().mean(axis=(1,2))
        flag = self.obs_flag.subcube((self.center[0]+xoff, self.center[1]+yoff) , size, unit_center=None).copy().mean(axis=(1,2))
        spec_sub_dobby, dob_ =  self.get_single_dobby_fit(spec_sub, 
                                     err, 
                                     flag ,
                                     cont,
                                     wdisp=wdisp,
                                     outfile='./dobby_out_combined/'+self.gal_id+'_'+str(size)+'arc_x_'+str(xoff)+'_y_'+str(yoff), 
                                     load=load, plot=plot)

        return spec, cont, spec_sub, dob_

    def plot_im(self, im, vmin=None, vmax=None, ax=None, cbarlabel='', cbarticklabels=[],save=False, filename='', center = False):

        if not ax:
            wcs = WCS(self.obs.data_header)
            wcs= wcs.dropaxis(2)
            fig = plt.figure()
            ax= fig.add_subplot(projection=wcs)
        col_img = ax.imshow(im, origin='lower', vmin=vmin, vmax=vmax, cmap='plasma')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec.')
        
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        if len(cbarlabel) != 0:
            #cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0,0.02,ax.fet_position().height ])
            cbar = plt.colorbar(col_img,  fraction=0.046, pad=0.04, ax=ax)
            cbar.set_label(cbarlabel, fontsize=20)
        if len(cbarticklabels)>0:
            cbar.set_ticks(np.arange(len(cbarticklabels))+vmin)
            cbar.set_ticklabels(cbarticklabels)
        if center:
            
            ax.scatter(self.center[1],self.center[0],marker='x', color='k', alpha=0.5, s=35)
        plt.tight_layout()
        if save:
                ax.savefig('plots/'+filename+'.pdf', format='pdf', dpi=250,bbox_inches='tight' )
                ax.savefig('plots/'+filename+'.png', format='png', dpi=250,bbox_inches='tight' )
                ax.close()
        else:
            if len(cbarlabel)!=0:
                return ax, cbar
    
def write_starlight(wl, flux, err, flag, filename):
    out_ = np.vstack([wl, flux, err, flag]).transpose()
    np.savetxt(filename, out_, fmt='%4.f %6.4f %4.8f %4.4f')
def load_starlight(filename, flux_unit=1e20):
    out = read_output_tables(filename)
    fobs_norm = out['keywords']['fobs_norm']
    fobs = out['spectra']['f_obs']*fobs_norm*flux_unit
    fsyn = out['spectra']['f_syn']*fobs_norm*flux_unit
    fwei = out['spectra']['f_wei']*flux_unit
    res = fobs-fsyn
    return out, fobs, fsyn, fwei, res
c_kms = c/1000.
arcsec = 1./3600
plt.rc('font',family='serif')
plt.rc('text',usetex=True)
emission_lines = {'halp':6562.800, 'hbeta':4861.325, 'oiii':5006.843, 'nii':6583.460}
class SDSS_spec(Fits_set):
    def __init__(self, fname, z, order=7):
        super().__init__(fname)
        self.order = order
        self.logwl = self.getcol('loglam')
        self.vacwl = 10**(self.logwl)        
        self.wl = self.vacwl / (1.0 + 2.735182E-4 + 131.4182 / self.vacwl**2 + 2.76249E8 / self.vacwl**4)
        # vacuum to air from Morton (1991, ApJS, 77, 119)
        self.flux = self.getcol('flux')
        self.z = z
        self.flux_err = self.getcol('ivar')
        self.flux_flags = self.getcol('and_mask')
        self.badpix = self.flux_flags>0
        self.rest_wl, self.rest_flux = pyc2c.spectra2restframe(self.wl, self.flux, self.z)
        _, self.rest_flux_err = pyc2c.spectra2restframe(self.wl, self.flux_err, self.z)

        self.rest_vdisp = np.log(10)*self.getcol('wdisp') *0.0001*self.rest_wl

        
        self.resampled_wl_starlight = np.arange(4750, 8951)
        self.resampled_spec = resample_spectra(self.rest_wl, self.resampled_wl_starlight, self.rest_flux, self.rest_flux_err, self.badpix)
        self.resampled_flux, self.resampled_flux_err, self.resampled_flag = self.resampled_spec
        self.dob_flag = np.where(self.resampled_flag>100, 1, 0)
        self.resampled_wdisp,  _, _ = resample_spectra(self.rest_wl, self.resampled_wl_starlight, self.rest_vdisp, self.rest_flux_err, self.badpix*0)
        
    def plot(self, wlmin = 4800, wlmax=7500, save=False, filename='', cont=False):
        plt.plot(self.wl, self.flux, color='k', label='SDSS Spectrum')
        plt.xlabel('Wavelength (angstroms)', fontsize=20)
        plt.ylabel(r'Flux [$10^{-17}$ erg/cm$^2$/s/\AA]', fontsize=20)
        plt.xlim([wlmin, wlmax])
        plt.ylim([0, np.max(self.flux)+10])
        plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
        plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
        plt.minorticks_on()
        plt.tick_params(direction='in',axis='both',which='both')
        plt.tight_layout()
        if cont:
            plt.plot(self.wl, self.contflux, color='r', linestyle='-.', label='Median Smoothed')
            plt.plot(self.wl, self.contfit, color='g', linestyle='--', label='Fit')
        plt.legend(frameon=False,fontsize=12)
            
        if save:
            plt.savefig(filename, dpi=250)
    def write_wlflux(self, filename):
        phdu = pf.PrimaryHDU(self.flux)
        phdu.writeto(filename)
    def write_starlight(self, filename):
        out_ = np.vstack([self.resampled_wl_starlight, self.resampled_spec[0], self.resampled_spec[1], self.resampled_spec[2]]).transpose()
        np.savetxt(filename, out_, fmt='%4.2f %4.2f %4.2f %3.2f')
    def load_starlight(self, filename):
        self.starlight_out = read_output_tables(filename)
        fobs_norm = self.starlight_out['keywords']['fobs_norm']
        self.starlight_fobs = self.starlight_out['spectra']['f_obs']*fobs_norm
        self.starlight_fsyn = self.starlight_out['spectra']['f_syn']*fobs_norm
        self.starlight_fwei = self.starlight_out['spectra']['f_wei']
        self.starlight_res = self.starlight_fobs-self.starlight_fsyn
def plot_spec(x,y, label, unit= r'Flux [$10^{-17}$ erg/cm$^2$/s/\AA]'):
    plt.plot(x,y,label=label)
    plt.xlabel('Wavelength (angstroms)', fontsize=20)
    plt.ylabel(unit, fontsize=20)
    plt.legend()

load_cutouts=True
plot=False
load_aps=True
write_sl = False
load_sl=True
bco3n = True
xu104 = Reduced_Cube('data/xu104_bco3n_starlighted.fits', 'xu104',0.118, 
                     load_cutouts=load_cutouts, plot=plot, load_apertures=load_aps, write_sl=write_sl, load_sl=load_sl, bco3n=bco3n)
xr31 = Reduced_Cube('data/xr31_bco3n_starlighted.fits', 'xr31',0.0712, 
                    load_cutouts=load_cutouts, plot=plot, load_apertures=load_aps, write_sl=write_sl, load_sl=load_sl, bco3n=bco3n)

xu210 = Reduced_Cube('data/xu210_bco3n_starlighted.fits', 'xu210',0.101388, 
                     load_cutouts=load_cutouts, plot=plot, load_apertures=load_aps, write_sl=write_sl, load_sl=load_sl, bco3n=bco3n)

xu22 = Reduced_Cube('data/xu22_bco3n_starlighted.fits', 'xu22', 0.0949, 
                    load_cutouts=load_cutouts, plot=plot, load_apertures=load_aps, write_sl=write_sl, load_sl=load_sl, bco3n=bco3n)

xu23 = Reduced_Cube('data/xu23_bco3n_starlighted.fits', 'xu23', 0.096, 
                    load_cutouts=load_cutouts, plot=plot, load_apertures=load_aps, write_sl=write_sl, load_sl=load_sl, bco3n=bco3n)



xu104.get_dobby_fits(load_full=True)
xr31.get_dobby_fits( load_full=True)    
xu210.get_dobby_fits( load_full=True)
xu22.get_dobby_fits(load_full=True)
xu23.get_dobby_fits(load_full=True)

#xr10 = Cube('xr10_het_starlighted.fits', ext=1, unit=1e-20)



    
sdss_xu22 = SDSS_spec('../muse/xu22/sdss_xu22.fits',0.0949)
sdss_xu23 = SDSS_spec('../muse/xu23/sdss_xu23.fits',0.096)
sdss_xu104 = SDSS_spec('../muse/xu104/sdss_xu104.fits',0.118) #/home/caug/gdrive/FromBox/research/bpt_bias/observations/muse/xu104/sdss_xu104.fits',0.118)
sdss_xu210 = SDSS_spec('../muse/xu210/sdss_xu210.fits',0.101388)

sdss_xr31 = SDSS_spec('../muse/xr31/sdss_xr31.fits',0.0712)

sdss_xu22.load_starlight('../muse/xu22/xu22_sdss.out')
sdss_xu23.load_starlight('../muse/xu23/xu23_sdss.out')
sdss_xu104.load_starlight('../muse/xu104/xu104_sdss.out')#/home/caug/gdrive/FromBox/research/bpt_bias/observations/muse/xu104/xu104_sdss.out')
sdss_xr31.load_starlight('../muse/xr31/xr31_sdss.out')
sdss_xu210.load_starlight('../muse/xu210/xu210_sdss.out')


xr31.sdss = sdss_xr31
xu22.sdss = sdss_xu22 
xu23.sdss = sdss_xu23
xu104.sdss = sdss_xu104
xu210.sdss = sdss_xu210




muse_cubes = {'SF-1':xr31, 'WL-3':xu22, 'WL-2':xu23, 'WL-4':xu104,'WL-1':xu210}

xu_cubes = {22:xu22, 23:xu23,  104:xu104,210:xu210}


xraylums = [41.74,42.06, 42.43, 42.035, 42.17]

redshifts = [0.0712,0.0949, 0.096, 0.118, 0.101388]


