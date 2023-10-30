import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coords
import astropy.cosmology as apc
cosmo = apc.Planck15
from Fits_set import Fits_set


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
