"""
Made for loading in Fits files by name and to make it easy to grab column data.
"""
import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='serif')
plt.rc('text',usetex=True)

from scipy.constants import c
from astropy.table import Table
import pandas as pd



import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coords
import astropy.cosmology as apc
cosmo = apc.Planck15

class Fits_set:


    '''
    Made for loading in astronomy data tables.
    '''
    def __init__(self,fname):
        self.fname = fname
        if 'fit' in self.fname:
            
            self.fitsimg = pf.open(fname)
            self.header = self.fitsimg[1].header
            self.data = Table(self.fitsimg[1].data)
            # Initialize an empty list to store exploded columns
            exploded_columns = []
            columns_to_remove = []

            # Iterate through the columns
            for colname in self.data.colnames:
                column = self.data[colname]
                # Check if the column is multidimensional
                if column.ndim > 1:
                    # Create a new column for each sub-element of the multidimensional column
                    for i in range(column.shape[1]):
                        new_colname = f"{colname}_{i}"
                        new_column = column[:, i]                    
                        exploded_columns.append((new_colname, new_column))
                    columns_to_remove.append(colname)

            # Add the exploded columns to the original Astropy table
            for colname, coldata in exploded_columns:
                self.data[colname] = coldata
            for colname in columns_to_remove:
                self.data.remove_column(colname)
            
            self.data = self.data.to_pandas()
        elif 'tsv' in self.fname
            self.data = pd.read_csv(self.fname, delimiter='\t')
        elif 'csv' in self.fname:
            self.data = pd.read_csv(self.fname)
        


class XMM(Fits_set):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['sourceids'] = self.data['Source']
        
        self.data['hardflux2'] = (self.data['Flux4'] + self.data['Flux5']) * 0.87
        self.data['ehardflux2'] = np.sqrt(self.data['e_Flux4'] ** 2 + self.data['e_Flux5'] ** 2) * 0.87
        self.data['hardflux'] = (self.data['Flux8'] - (self.data['Flux1'] + self.data['Flux2'] + self.data['Flux3'])) * 0.87
        self.data['ehardflux'] = np.sqrt(self.data['e_Flux8'] ** 2 + self.data['e_Flux1'] ** 2 + self.data['e_Flux2'] ** 2 + self.data['e_Flux3'] ** 2) * 0.87
        self.data['softflux'] = self.data['Flux2'] + self.data['Flux3']
        self.data['esoftflux'] = np.sqrt(self.data['e_Flux2'] ** 2 + self.data['e_Flux3'] ** 2)
        self.data['fullflux'] = (self.data['Flux8'] - self.data['Flux1']) * 0.91
        self.data['efullflux'] = np.sqrt(self.data['e_Flux8'] ** 2 + self.data['e_Flux1'] ** 2) * 0.91
        self.data['hardflux_sn'] = self.data['hardflux'] / self.data['ehardflux']
        self.data['fullflux_sn'] = self.data['fullflux'] / self.data['efullflux']
        self.data['softflux_sn'] = self.data['softflux'] / self.data['esoftflux']
        self.data['qualflag'] = self.data['S']

    def get_texp(self, matchinds):
        texps = []
        for k, source in enumerate(self.data['Source'].iloc[matchinds]):
            if k % 1000 == 0:
                print(k)
            source_obs = str(np.int64(str(source)[2:11]))
            obsids = self.data['obsids'].astype(str)
            obs_ = obsids[obsids == source_obs].index
            exp = np.array([self.data.loc[obs_, 'tpn'], self.data.loc[obs_, 'tmos1'], self.data.loc[obs_, 'tmos2']])
            try:
                texps.append(np.max(exp))
            except:
                texps.append(np.nan)
        return np.array(texps)                
        
class XMM3obs(Fits_set):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['obsids'] = self.obs_data[1]  # Assuming '1' is the column name for obsids in fits_set_obs
        self.data['tpn'] = self.obs_data['t_PN']
        self.data['tmos1'] = self.obs_data['t_M1']
        self.data['tmos2'] = self.obs_data['t_M2']

class XMM4obs(Fits_set):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['obsids'] = self.obs_data['ObsID'].astype(np.int64)
        self.data['tpn'] = self.obs_data['t.PN']
        self.data['tmos1'] = self.obs_data['t.M1']
        self.data['tmos2'] = self.obs_data['t.M2']
        exp = np.vstack([self.data['tpn'], self.data['tmos1'], self.data['tmos2']])
        self.data['texps'] = np.max(exp, axis=0)
        
class CSC(Fits_set):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['hardflux'] = self.data['Fluxh'] * 1.15
        self.data['softflux'] = self.data['Fluxs'] + self.data['Fluxm']
        self.data['fullflux'] = self.data['Fluxb'] * 1.04
        # Continue for other calculated fields
class FIRST(Fits_set):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['allplateids'] = self.data['SPEC_PLATE']
        self.data['allmjds']: self.data['SPEC_MJD'],
        self.data['allfiberids']: self.data['SPEC_FIBERID'],
        self.data['nvss_flux']: self.data['NVSS_FLUX'],
        self.data['first_flux']: self.data['FIRST_FINT'],
        self.data['wenss_flux']: self.data['WENSS_FLUX'],
        self.data['vlss_flux']: self.data['VLSS_FLUX']




class Galinfo(Fits_set):
    def __init__(self, filename):
        super().__init__(filename)        
        self.data['allplateids']= self.data['PLATEID']
        self.data['allmjds']= self.data['MJD']
        self.data['allfiberids']: self.data['FIBERID']
        self.data['all_spectype']: self.data['SPECTROTYPE']

class Galline(Fits_set):
    def __init__(self, filename):
        super().__init__(filename)        

            'allhdelta': self.data['H_DELTA_FLUX'],
            'allhdelta_err': self.data['H_DELTA_FLUX_ERR'],
            'allhgamma': self.data['H_GAMMA_FLUX'],
            'allhgamma_err': self.data['H_GAMMA_FLUX_ERR'],
            'alloIII4363': self.data['OIII_4363_FLUX'],
            'alloIII4363_err': self.data['OIII_4363_FLUX_ERR'],
            'alloIII4959': self.data['OIII_4959_FLUX'],
            'alloIII4959_err': self.data['OIII_4959_FLUX_ERR'],
            'allheI': self.data['HEI_5876_FLUX'],
            'allheI_err': self.data['HEI_5876_FLUX_ERR'],
            'alloII3726': self.data['OII_3726_FLUX'],
            'alloII3726err': self.data['OII_3726_FLUX_ERR'],
            'alloII3729': self.data['OII_3729_FLUX'],
            'alloII3729err': self.data['OII_3729_FLUX_ERR'],
            'allneIII': self.data['NEIII_3869_FLUX_ERR'],
            'allneIIIerr': self.data['NEIII_3869_FLUX'],
            'alloI': self.data['OI_6300_FLUX'],
            'alloI_err': self.data['OI_6300_FLUX_ERR'],
            'allSII_6717': self.data['SII_6717_FLUX'],
            'allSII_6717_err': self.data['SII_6717_FLUX_ERR'],
            'allSII_6731': self.data['SII_6731_FLUX'],
            'allSII_6731_err': self.data['SII_6731_FLUX_ERR'],
            'alloIII': self.data['OIII_5007_FLUX'],
            'alloIII_err': self.data['OIII_5007_FLUX_ERR'],
            'allhbeta': self.data['H_BETA_FLUX'],
            'allhbeta_err': self.data['H_BETA_FLUX_ERR'],
            'alloIII_eqw': self.data['OIII_5007_EQW'],
            'alloIII_eqw_err': self.data['OIII_5007_EQW_ERR'],
            'allnII': self.data['NII_6584_FLUX'],
            'allnII_err': self.data['NII_6584_FLUX_ERR'],
            'allhalpha': self.data['H_ALPHA_FLUX'],
            'allhalpha_err': self.data['H_ALPHA_FLUX_ERR'],
            'allha_eqw': self.data['H_ALPHA_EQW'],
            'allha_eqw_err': self.data['H_ALPHA_EQW_ERR'],
            'allnII_6548': self.data['NII_6548_FLUX'],
            'allnII_6548_err': self.data['NII_6548_FLUX_ERR'],
            'allbalmerdisperr': self.data['SIGMA_BALMER_ERR'],
            'allbalmerdisp': self.data['SIGMA_BALMER'],
            'allforbiddendisp': self.data['SIGMA_FORBIDDEN'],
            'allforbiddendisperr': self.data['SIGMA_FORBIDDEN_ERR']
class Galindx(Fits_set):
    def __init__(self, filename):
        super().__init__(filename)        
            'alld4000': self.data['D4000_N'],
            'hdelta_lick': self.data['LICK_HD_A'],
            'tauv_cont': self.data['TAUV_CONT'],
            'allvdisp': self.data['V_DISP'],        
class Gal_Fib(Fits_set):
    def __init__(self, filename, output):
        super().__init__(filename)        
        self.data[output] =  self.data['AVG'],


galmass = Gal_Fib(galmass, 'all_sdss_avgmasses')
fibmass = Gal_Fib(fibmass, 'all_fibmass')
fibsfr = Gal_Fib(fibsfr, 'all_fibmass')
fibssfr = Gal_Fib(fibssfr, 'all_fibssfr_mpa')


class SDSSObj(self):
    def __init__(self, galfiboh_file, galinfo_file, galline_file, galindx_file, fibmass_file, fibsfr_file, fibssfr_file,)

