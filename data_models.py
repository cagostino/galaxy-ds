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
from ast_func import dustcorrect, extinction, get_bptplus_niigroups, get_bpt1_groups,get_bptplus_groups,get_classifiability, get_thom_dist,


def read_data(catfold, filename, columns, header=None):
    filepath = catfold + filename
    return pd.read_csv(filepath, delim_whitespace=True, header=None, names=columns)

def load_gsw_catalog(catfold, filename):
    # Define column names based on the provided information
    columns = [
        'ObjID', 'GLXID', 'plate', 'MJD', 'fiber_ID',
        'RA', 'Decl', 'z', '2r', 'mass', 'mass_Error',
        'sfr', 'sfr_error', 'afuv', 'afuv_error',
        'ab', 'ab_error', 'av_gsw', 'av_gsw_error', 'flag_sed',
        'uv_survey', 'flag_uv', 'flag_midir', 'flag_mgs'
    ]    
    # Load data into Pandas DataFrames
    gsw_df = read_data(catfold, filename, columns)
    gsw_df.reset_index(drop=True, inplace=True)

    # Additional data to be concatenated
    if 'M' in filename:
        additional_data_files = [
            ('sigma1_mis.dat', [2], 'sigma1_m'),
            ('envir_nyu_mis.dat', [0], 'env_nyu_m'),
            ('baldry_mis.dat', [4], 'env_bald_m'),
            ('irexcess_mis.dat', [0], 'irx_m'),
            ('simard_ellip_mis.dat', [1], 'axisrat')
        ]

        for file, cols, new_col_name in additional_data_files:
            additional_df = read_data(catfold, file, cols)
            additional_df.reset_index(drop=True, inplace=True)
            gsw_df[new_col_name] = additional_df.iloc[:, 0]

    return gsw_df
        

class AstroTablePD:
    '''
    Made for loading in astronomy data tables.
    '''
    def __init__(self,
                 fname=None, 
                 dataframe= None):
        self.fname = fname
        if fname is not None:
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
            elif 'GSW' in self.fname:
                self.data = load_gsw_catalog(self.fname)
        elif dataframe is not None:
            self.data = dataframe            
        



catfold='catalogs/'
import numpy as np
print('loading GSW')


import pandas as pd 


class GSWCat:    
    def __init__(self, goodinds, gsw_table, sedflag=0):
        self.inds = goodinds
        self.sedfilt = np.where(gsw_table['flag_sed'][self.inds]==sedflag)[0]
        self.gsw_df = gsw_table.iloc[self.inds].iloc[self.sedfilt]

class GSWCatmatch3xmm(GSWCat):
    def __init__(self, 
                 goodinds, 
                 gsw_table, 
                 xray_table,
                 sedflag=0):
        super().__init__(goodinds, gsw_table, sedflag=sedflag)
        
        self.gsw_df['softflux'] = xray_table.loc[self.sedfilt, 'softflux']
        self.gsw_df['hardflux'] = xray_table.loc[self.sedfilt,'hardflux']
        self.gsw_df['fullflux'] = xray_table.loc[self.sedfilt,'fullflux']        

        self.gsw_df['esoftflux'] = xray_table.loc[self.sedfilt,'esoftflux']
        self.gsw_df['ehardflux'] = xray_table.loc[self.sedfilt,'ehardflux']
        self.gsw_df['efullflux'] = xray_table.loc[self.sedfilt, 'efullflux']
        
        self.gsw_df['fullflux_sn'] = self.gsw_df.fullflux/self.gsw_df.efullflux
        self.gsw_df['hardflux_sn'] = self.gsw_df.hardflux/self.gsw_df.ehardflux
        self.gsw_df['softflux_sn'] = self.gsw_df.softflux/self.gsw_df.esoftflux
        
        self.gsw_df['softlums'] =getlumfromflux(self.gsw_df['softflux'], self.gsw_df['z'])
        self.gsw_df['hardlums'] =getlumfromflux(self.gsw_df['hardflux'], self.gsw_df['z'])
        self.gsw_df['fulllums'] = getlumfromflux(self.gsw_df['fullflux'], self.gsw_df['z'])
        
        self.gsw_df['efulllums_up'] =  getlumfromflux(self.gsw_df['fullflux']+self.gsw_df['efullflux'],self.gsw_df['z'])
        self.gsw_df['efulllums_down'] = getlumfromflux(self.gsw_df['fullflux']-self.gsw_df['efullflux'],self.gsw_df['z'])
        
        self.gsw_df['softlumsrf'] = np.log10(self.gsw_df.softlums*(1+self.gsw_df['z'])**(1.7-2))
        self.gsw_df['hardlumsrf'] = np.log10(self.gsw_df.hardlums*(1+self.gsw_df['z'])**(1.7-2))
        self.gsw_df['fulllumsrf'] = np.log10(self.gsw_df.fulllums*(1+self.z)**(1.7-2))
        self.gsw_df['efulllumsrf_down'] = np.log10(self.gsw_df.efulllums_down*(1+self.z)**(1.7-2))
        self.gsw_df['efulllumsrf_up'] = np.log10(self.gsw_df.efulllums_up*(1+self.z)**(1.7-2))\        
        
        self.gsw_df['ehardlums_up'] =   getlumfromflux(self.gsw_df['hardflux']+self.gsw_df['ehardflux'],self.gsw_df['z'])
        self.gsw_df['ehardlums_down'] = getlumfromflux(self.gsw_df['hardflux']-self.gsw_df['ehardflux'],self.gsw_df['z'])
        self.gsw_df['ehardlumsrf_down'] = np.log10(self.gsw_df.ehardlums_down*(1+self.z)**(1.7-2))
        self.gsw_df['ehardlumsrf_up'] = np.log10(self.gsw_df.ehardlums_up*(1+self.z)**(1.7-2))        


        self.gsw_df['hr1'] = xray_table.loc[self.sedfilt, 'hr1']
        self.gsw_df['hr2'] = xray_table.loc[self.sedfilt, 'hr2']
        self.gsw_df['hr3'] = xray_table.loc[self.sedfilt, 'hr3']
        self.gsw_df['hr4'] = xray_table.loc[self.sedfilt, 'hr4']
        
        #self.ext = ext[self.sedfilt]
        #self.xrayflag = xrayflag[self.sedfilt]
        self.gsw_df['exptimes'] = xray_table.loc[self.sedfilt, 'logtimes']
        self.gsw_df['matchxrayra'] =  xray_table.loc[self.sedfilt, 'ra']
        self.gsw_df['matchxraydec'] =  xray_table.loc[self.sedfilt, 'logtimes']
        
        
class GSWCatmatch_CSC(GSWCat):
    def __init__(self, goodinds, sfrplus,  fullflux_filt, 
                  hardflux_filt, softflux_filt, sedflag=0): #, ext, hr1, hr2, hr3, hr4,efullflux_filtxrayflag
        super().__init__(goodinds, sfrplus, sedflag=sedflag)
        
        self.gsw_df['softflux'] = softflux_filt[self.sedfilt]
        self.gsw_df['hardflux'] = hardflux_filt[self.sedfilt]
        self.gsw_df['fullflux'] = fullflux_filt[self.sedfilt]
        
        #self.efullflux = efullflux_filt[self.sedfilt]
        
        self.gsw_df['softlums'] =getlumfromflux(softflux_filt[self.sedfilt], self.z)
        self.gsw_df['hardlums'] =getlumfromflux(hardflux_filt[self.sedfilt], self.z)
        #self.hardlums2 =getlumfromflux(hardflux2_filt[self.sedfilt], self.z)
        self.gsw_df['fulllums'] = getlumfromflux(fullflux_filt[self.sedfilt], self.z)
        #self.efulllums_up =  getlumfromflux(fullflux_filt[self.sedfilt]+efullflux_filt[self.sedfilt],self.z)
        #self.efulllums_down = getlumfromflux(fullflux_filt[self.sedfilt]-efullflux_filt[self.sedfilt],self.z)
        self.gsw_df['softlumsrf'] = np.log10(self.gsw_df.softlums*(1+self.z)**(1.7-2))
        self.gsw_df['hardlumsrf'] = np.log10(self.gsw_df.hardlums*(1+self.z)**(1.7-2))
        

        
        #self.hardlums2rf = np.log10(self.hardlums2*(1+self.z)**(1.7-2))
        '''self.hr1 = hr1[self.sedfilt]
        self.hr2 = hr2[self.sedfilt]
        self.hr3 = hr3[self.sedfilt]
        self.hr4 = hr4[self.sedfilt]
        
        self.xrayflag = xrayflag[self.sedfilt]
        '''
        self.gsw_df['fulllumsrf'] = np.log10(self.gsw_df.fulllums*(1+self.z)**(1.7-2))
        '''
        self.efulllumsrf_down = np.log10(self.efulllums_down*(1+self.z)**(1.7-2))
        self.efulllumsrf_up = np.log10(self.efulllums_up*(1+self.z)**(1.7-2))
        self.ext = ext[self.sedfilt]
        '''
        
class GSWCatmatch_radio(GSWCat):
    def __init__(self, goodinds,  sfrplus,  nvss_flux,
            first_flux , wenss_flux ,vlss_flux, sedflag=0): 
        super().__init__(goodinds, sfrplus, sedflag=sedflag)
        
        self.gsw_df['nvss_flux'] = nvss_flux[self.sedfilt]
        self.gsw_df['first_flux'] = first_flux[self.sedfilt]
        self.gsw_df['wenss_flux'] = wenss_flux[self.sedfilt]
        self.gsw_df['vlss_flux'] = vlss_flux[self.sedfilt]        

        self.gsw_df['nvss_lums'] =getlumfromflux(nvss_flux[self.sedfilt], self.z)
        self.gsw_df['firstlums'] =getlumfromflux(first_flux[self.sedfilt], self.z)
        self.gsw_df['wensslums'] = getlumfromflux(wenss_flux[self.sedfilt], self.z)
        self.gsw_df['vlsslums'] = getlumfromflux(vlss_flux[self.sedfilt], self.z)
        


class XMM(AstroTablePD):
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
        
class XMM3obs(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['obsids'] = self.obs_data[1]  # Assuming '1' is the column name for obsids in AstroTablePD_obs
        self.data['tpn'] = self.obs_data['t_PN']
        self.data['tmos1'] = self.obs_data['t_M1']
        self.data['tmos2'] = self.obs_data['t_M2']

class XMM4obs(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['obsids'] = self.obs_data['ObsID'].astype(np.int64)
        self.data['tpn'] = self.obs_data['t.PN']
        self.data['tmos1'] = self.obs_data['t.M1']
        self.data['tmos2'] = self.obs_data['t.M2']
        exp = np.vstack([self.data['tpn'], self.data['tmos1'], self.data['tmos2']])
        self.data['texps'] = np.max(exp, axis=0)
        
class CSC(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['hardflux'] = self.data['Fluxh'] * 1.15
        self.data['softflux'] = self.data['Fluxs'] + self.data['Fluxm']
        self.data['fullflux'] = self.data['Fluxb'] * 1.04
        # Continue for other calculated fields
class FIRST(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['allplateids'] = self.data['SPEC_PLATE']
        self.data['allmjds']: self.data['SPEC_MJD'],
        self.data['allfiberids']: self.data['SPEC_FIBERID'],
        self.data['nvss_flux']: self.data['NVSS_FLUX'],
        self.data['first_flux']: self.data['FIRST_FINT'],
        self.data['wenss_flux']: self.data['WENSS_FLUX'],
        self.data['vlss_flux']: self.data['VLSS_FLUX']




class Galinfo(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)        
        self.data['allplateids']= self.data['PLATEID']
        self.data['allmjds']= self.data['MJD']
        self.data['allfiberids']: self.data['FIBERID']
        self.data['all_spectype']: self.data['SPECTROTYPE']
    
class Galline(AstroTablePD):
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
                self.halp_sn = np.reshape(sdss.allhalpha[sdssinds]/sdss.allhalpha_err[sdssinds],-1)
        self.nii_sn = np.reshape(sdss.allnII[sdssinds]/sdss.allnII_err[sdssinds],-1)
        self.halpnii_sn = 1./np.sqrt( (1./self.halp_sn)**2 +(1/self.nii_sn)**2)
        self.oi_sn = np.reshape(sdss.alloI[sdssinds]/sdss.alloI_err[sdssinds],-1)
        self.oiii_sn = np.reshape(sdss.alloIII[sdssinds]/sdss.alloIII_err[sdssinds],-1)
        self.hbeta_sn = np.reshape(sdss.allhbeta[sdssinds]/sdss.allhbeta_err[sdssinds],-1)
        self.hbetaoiii_sn = 1./np.sqrt( (1./self.hbeta_sn)**2 + (1/self.oiii_sn)**2 )

        self.sii6731_sn = np.reshape(sdss.allSII_6731[sdssinds]/sdss.allSII_6731_err[sdssinds],-1)
        self.sii6717_sn = np.reshape(sdss.allSII_6717[sdssinds]/sdss.allSII_6717_err[sdssinds],-1)
        self.sii_sn = np.reshape(sdss.allSII[sdssinds]/sdss.allSII_err[sdssinds],-1)
        
        self.neIII_sn = np.reshape(sdss.allneIII[sdssinds]/sdss.allneIIIerr[sdssinds],-1)
        self.oII_sn = np.reshape(sdss.alloII[sdssinds]/sdss.alloII_err[sdssinds],-1)
        #BPT filt used for everything
        self.bpt_sn_filt_bool = (self.halp_sn>sncut) & (self.hbeta_sn > sncut) & (self.oiii_sn > sncut) & (self.nii_sn > sncut)
        self.bpt_sn_filt = np.where(self.bpt_sn_filt_bool)[0]
        self.high_sn_o3 = np.where(self.oiii_sn>sncut)[0]
        
        self.not_bpt_sn_filt_bool  = np.logical_not(self.bpt_sn_filt_bool)

        self.halp_nii_filt_bool = ( (self.halp_sn > sncut) & (self.nii_sn > sncut) &( (self.oiii_sn<=sncut) | (self.hbeta_sn<=sncut) ) )
        self.halp_nii_filt = np.where(self.halp_nii_filt_bool)[0]
        
        self.not_bpt_sn_filt_bool  = np.logical_not(self.bpt_sn_filt_bool)
        self.not_bpt_sn_filt  = np.where(self.not_bpt_sn_filt_bool)[0]
        
        self.neither_filt_bool = np.logical_not( ( (self.bpt_sn_filt_bool) | (self.halp_nii_filt_bool) ))#neither classifiable by BPT, or just by NII
        self.neither_filt = np.where(self.neither_filt_bool)[0]
        #testing S/N each line >3
        self.bpt_sn3_filt_bool = ((self.halp_sn > 3) & (self.hbeta_sn > 3) & (self.oiii_sn > 3) & (self.nii_sn > 3))
        self.bpt_sn3_filt = np.where(self.bpt_sn3_filt_bool)[0]
        
        self.halp_nii3_filt_bool = ( (self.halp_sn > 3) & (self.nii_sn > 3) &( (self.oiii_sn<=3) | (self.hbeta_sn<=3) ))
        self.halp_nii3_filt = np.where(self.halp_nii3_filt_bool)[0]
        
        self.neither3_filt_bool = np.logical_not( ((self.bpt_sn3_filt_bool ) | (self.halp_nii3_filt_bool)))#neither classifiable by BPT, or just by NII
        self.neither3_filt = np.where(self.neither3_filt_bool)[0]

        #testing S/N each line >1
        self.bpt_sn1_filt_bool = ((self.halp_sn > 1) & (self.hbeta_sn > 1) & (self.oiii_sn > 1) & (self.nii_sn > 1))
        self.bpt_sn1_filt = np.where(self.bpt_sn1_filt_bool)[0]
        
        self.halp_nii1_filt_bool = ( (self.halp_sn > 1) & (self.nii_sn > 1) &( (self.oiii_sn<=1) | (self.hbeta_sn<=1) ))
        self.halp_nii1_filt = np.where(self.halp_nii1_filt_bool)[0]
        
        self.neither1_filt_bool = np.logical_not( ((self.bpt_sn1_filt_bool ) | (self.halp_nii1_filt_bool)))#neither classifiable by BPT, or just by NII
        self.neither1_filt = np.where(self.neither1_filt_bool)[0]


        #testing S/N line ratio >3
        self.bpt_sn_lr3_filt_bool = ((self.halpnii_sn>3) &(self.hbetaoiii_sn >3) & (self.halp_sn > 0) &
                                        (self.hbeta_sn > 0) & (self.oiii_sn > 0) & (self.nii_sn > 0) &
                                        (np.isfinite(self.halpnii_sn)) & (np.isfinite(self.hbetaoiii_sn)))
        self.bpt_sn_lr3_filt = np.where( self.bpt_sn_lr3_filt_bool )[0]

        self.halp_nii_lr3_filt_bool =((self.halpnii_sn>3) &(self.hbetaoiii_sn <=3) & (self.halp_sn > 0) &
                                        (self.nii_sn > 0) & (np.isfinite(self.halpnii_sn)))
        self.halp_nii_lr3_filt = np.where(self.halp_nii_lr3_filt_bool)[0]
        
        self.neither_lr3_filt_bool = np.logical_not(((self.bpt_sn_lr3_filt_bool ) | (self.halp_nii_lr3_filt_bool)) )
        self.neither_lr3_filt = np.where(self.neither_lr3_filt_bool)[0]
    
        #testing S/N line ratio >3/sqrt(2)
        self.bpt_sn_lr2_filt_bool = ((self.halpnii_sn>2.12) &(self.hbetaoiii_sn >2.12) & (self.halp_sn > 0) &
                                        (self.hbeta_sn > 0) & (self.oiii_sn > 0) & (self.nii_sn > 0) &
                                        (np.isfinite(self.halpnii_sn)) & (np.isfinite(self.hbetaoiii_sn)))
        self.bpt_sn_lr2_filt = np.where( self.bpt_sn_lr2_filt_bool)[0]
        
        self.halp_nii_lr2_filt_bool = ((self.halpnii_sn > 2.12) & (self.hbetaoiii_sn <=2.12) & (self.halp_sn > 0) &
                                        (self.nii_sn > 0) & (np.isfinite(self.halpnii_sn)))
        self.halp_nii_lr2_filt = np.where( self.halp_nii_lr2_filt_bool)[0]
   
        self.neither_lr2_filt_bool=  np.logical_not(((self.halp_nii_lr2_filt_bool) | self.bpt_sn_lr2_filt_bool))
        self.neither_lr2_filt = np.where(self.neither_lr2_filt_bool)[0]
        #testing S/N line ratio > 2/sqrt(2)
        self.bpt_sn_minus_bpt_sn_lr2_filt_bool = ( (self.bpt_sn_filt_bool) & np.logical_not(self.bpt_sn_lr2_filt_bool))
        self.bpt_sn_minus_bpt_sn_lr2_filt = np.where(self.bpt_sn_minus_bpt_sn_lr2_filt_bool)[0]        
        self.halp_nii_minus_halp_nii_lr2_filt_bool = ( (self.halp_nii_filt_bool) & np.logical_not(self.halp_nii_lr2_filt_bool))
        self.halp_nii_minus_halp_nii_lr2_filt = np.where(self.halp_nii_minus_halp_nii_lr2_filt_bool)[0]
        
        
        self.bpt_sn_lr1_filt_bool = ((self.halpnii_sn > 1.41) &(self.hbetaoiii_sn > 1.41)& (self.halp_sn > 0) &
                                        (self.hbeta_sn > 0) & (self.oiii_sn > 0) & (self.nii_sn > 0) &
                                        (np.isfinite(self.halpnii_sn)) & (np.isfinite(self.hbetaoiii_sn)))        
        self.bpt_sn_lr1_filt = np.where(self.bpt_sn_lr1_filt_bool)[0]        
        self.halp_nii_lr1_filt_bool = ((self.halpnii_sn > 1.41) &(self.hbetaoiii_sn <= 1.41) & (self.halp_sn > 0) &
                                        (self.nii_sn > 0) & (np.isfinite(self.halpnii_sn)))
        self.halp_nii_lr1_filt = np.where(self.halp_nii_lr1_filt_bool )[0]
        self.neither_lr1_filt_bool = np.logical_not(( (self.bpt_sn_lr1_filt_bool) | (self.halp_nii_lr1_filt_bool)  ))
        self.neither_lr1_filt = np.where(self.neither_lr1_filt_bool)[0]
        
        self.vo87_1_filt = np.where(self.sii_sn[self.bpt_sn_filt]>2)[0]
        self.vo87_2_filt = np.where(self.oi_sn[self.bpt_sn_filt]>2)[0]
        
        self.sdss_filt = sdssinds[self.bpt_sn_filt]
        self.sdss_filt_plus = sdssinds[self.halp_nii_filt]
        self.sdss_filt_neither = sdssinds[self.neither_filt]
        self.sdss_filt_weak = sdssinds[self.not_bpt_sn_filt]

        self.tbtx = sdss.allneIII[sdssinds]/sdss.alloII[sdssinds]
             
  

        self.EL_dict = {'oiiiflux': sdss.alloIII/1e17,
                        'hbetaflux': sdss.allhbeta/1e17,
                        'halpflux': sdss.allhalpha/1e17,
                        'niiflux': sdss.allnII/1e17,
                        'siiflux': sdss.allSII/1e17,
                        'sii6731flux': sdss.allSII_6731/1e17,
                        'sii6717flux': sdss.allSII_6717/1e17,
                        'oiflux': sdss.alloI/1e17,
                        'hdeltaflux': sdss.allhdelta/1e17,
                        'hgammaflux': sdss.allhgamma/1e17,
                        'oiii4363flux': sdss.alloIII4363/1e17,
                        'oiii4959flux': sdss.alloIII4959/1e17,
                        'heIflux': sdss.allheI/1e17,
                        'nII6548flux': sdss.allnII_6548/1e17,
                        'oiiflux': sdss.alloII/1e17,
                        
                        'tauv_cont': sdss.tauv_cont,
                        'd4000': sdss.alld4000,
                        'hdelta_lick' : sdss.hdelta_lick,
                        'halp_eqw': sdss.allha_eqw,
                        'oiii_eqw': sdss.alloIII_eqw,
                        'vdisp': sdss.allvdisp,
                        'mbh': np.log10((3*(sdss.allvdisp/200)**4)*1e8),
                        'edd_lum': np.log10(3*1e8*1.38e38*((sdss.allvdisp/200)**4)),

                        'forbiddenfwhm':  sdss.allforbiddendisp*2 * np.sqrt(2*np.log(2)),
                        'balmerfwhm': sdss.allbalmerdisp*2 * np.sqrt(2*np.log(2)),
                        'fibmass': sdss.all_fibmass,
                        'fibsfr_mpa' : sdss.all_fibsfr_mpa,
                        'fibssfr_mpa' : sdss.all_fibssfr_mpa,                 
                        'ohabund':  sdss.fiboh,

                        'oiii_err': sdss.alloIII_err/1e17,
                        'hbeta_err': sdss.allhbeta_err/1e17,
                        'halp_err': sdss.allhalpha_err/1e17,
                        'nii_err': sdss.allnII_err/1e17,
                        'sii_err': sdss.allSII_err/1e17,
                        'sii6731_err': sdss.allSII_6731_err/1e17,
                        'sii6717_err': sdss.allSII_6717_err/1e17,
                        'oi_err': sdss.alloI_err/1e17,
                        'oiii4363_err': sdss.alloIII4363_err/1e17,
                        'oiii4959_err': sdss.alloIII4959_err/1e17,
                        'oii_err': sdss.alloII_err/1e17,
                        
                        'halpflux_sn':  sdss.allhalpha/sdss.allhalpha_err,
                        'niiflux_sn': sdss.allnII/sdss.allnII_err,
                        'oiflux_sn': sdss.alloI/sdss.alloI_err,
                        'oiiiflux_sn': sdss.alloIII/sdss.alloIII_err,
                        'hbetaflux_sn':  sdss.allhbeta/sdss.allhbeta_err,
                        'sii6731flux_sn':  sdss.allSII_6731/sdss.allSII_6731_err,
                        'sii6717flux_sn': sdss.allSII_6717/sdss.allSII_6717_err,
                        'siiflux_sn': sdss.allSII/sdss.allSII_err,
                        'oiiflux_sn': sdss.alloII/sdss.alloII_err,
                        'forbiddenfwhmerr': sdss.allforbiddendisperr*2*np.sqrt(2*np.log(2)),
                        'balmerfwhmerr': sdss.allbalmerdisperr*2*np.sqrt(2*np.log(2))
                        }
 

        self.EL_dict['massfrac'] = 10**(sdss.all_fibmass)/10**(sdss.all_sdss_avgmasses)
        self.EL_dict['yvals_bpt'] =self.EL_dict['oiiiflux']/self.EL_dict['hbetaflux']
        self.EL_dict['xvals1_bpt'] =self.EL_dict['niiflux']/self.EL_dict['halpflux']
        self.EL_dict['xvals2_bpt'] =self.EL_dict['siiflux']/self.EL_dict['halpflux']
        self.EL_dict['xvals3_bpt'] =self.EL_dict['oiflux']/self.EL_dict['halpflux']
            
        self.EL_dict['av'] = extinction(self.EL_dict['halpflux'], self.EL_dict['hbetaflux'])
        self.EL_dict['av_agn'] = extinction(self.EL_dict['halpflux'], self.EL_dict['hbetaflux'],agn=True)
        
        self.EL_dict['niiha'] = np.log10(self.EL_dict['xvals1_bpt'])
        self.EL_dict['siiha'] = np.log10(self.EL_dict['xvals2_bpt'])
        self.EL_dict['oiha'] = np.log10(self.EL_dict['xvals3_bpt'])
        self.EL_dict['oiiihb'] = np.log10(self.EL_dict['yvals_bpt'])
        self.EL_dict['thom_dist'] = get_thom_dist(self.EL_dict['niiha'], 
                                                  self.EL_dict['oiiihb']) 

        self.EL_dict['ji_p1'] =   (self.EL_dict['niiha']*0.63+self.EL_dict['siiha']*0.51+self.EL_dict['oiiihb']*0.59)
        self.EL_dict['ji_p2'] =   (-self.EL_dict['niiha']*0.63+self.EL_dict['siiha']*0.78)
        self.EL_dict['ji_p3'] =   (-self.EL_dict['niiha']*0.46-self.EL_dict['siiha']*0.37+0.81*self.EL_dict['oiiihb'])
        
        
        
        self.EL_dict_gsw = {}
        for key in self.EL_dict.keys():
            self.EL_dict_gsw[key] = self.EL_dict[key][sdssinds]
        self.EL_dict_gsw['mass'] = self.gswcat.mass[self.make_spec]
        self.EL_dict_gsw['sedflags'] = (self.gswcat.sedflags[self.make_spec])

        self.EL_dict_gsw['massfracgsw'] = 10**( self.EL_dict_gsw['fibmass'])/10**(self.EL_dict_gsw['mass'])

        self.EL_dict_gsw['ids'] = self.gswcat.ids[self.make_spec]
        self.EL_dict_gsw['sfr'] = self.gswcat.sfr[self.make_spec]
        self.EL_dict_gsw['sfr_error'] = self.gswcat.sfr_error[self.make_spec]
        self.EL_dict_gsw['mass_error'] = self.gswcat.mass_error[self.make_spec]
        
        self.EL_dict_gsw['sigma1'] = self.gswcat.sigma1[self.make_spec]
        self.EL_dict_gsw['nyuenv'] = np.log10(self.gswcat.nyuenv[self.make_spec])+2*0.2
        
        self.EL_dict_gsw['baldenv'] = self.gswcat.baldenv[self.make_spec]
        self.EL_dict_gsw['irx'] = self.gswcat.irx[self.make_spec]
        self.EL_dict_gsw['nuv'] = self.gswcat.nuv[self.make_spec]
        self.EL_dict_gsw['fuv'] = self.gswcat.fuv[self.make_spec]
        self.EL_dict_gsw['uv_col'] = self.EL_dict_gsw['fuv']-self.EL_dict_gsw['nuv']
        bad_uv = np.where((self.EL_dict_gsw['nuv']==-99) |(self.EL_dict_gsw['nuv']==-999) |(self.EL_dict_gsw['fuv']==-99) |(self.EL_dict_gsw['nuv']==-999) )[0]
        self.EL_dict_gsw['uv_col'][bad_uv] = np.nan        
        self.EL_dict_gsw['axisrat'] = self.gswcat.axisrat[self.make_spec]
        
        
        self.EL_dict_gsw['ssfr'] = self.EL_dict_gsw['sfr'] - self.EL_dict_gsw['mass']
        self.EL_dict_gsw['delta_ssfr'] = get_deltassfr(self.EL_dict_gsw['mass'], self.EL_dict_gsw['ssfr'])

        self.EL_dict_gsw['irx'][np.where(self.EL_dict_gsw['ssfr']<-11)] = np.nan
        self.EL_dict_gsw['irx'][np.where(self.EL_dict_gsw['irx']==-99)] = np.nan
        self.EL_dict_gsw['axisrat'][np.where(self.EL_dict_gsw['axisrat']==100)] = np.nan

        self.EL_dict_gsw['irx'][np.where((self.EL_dict_gsw['irx']==-99)&(self.EL_dict_gsw['irx']>10))] = np.nan

        self.EL_dict_gsw['z'] =  self.gswcat.z[self.make_spec]
        
        self.EL_dict_gsw['ra'] = self.gswcat.ra[self.make_spec]
        self.EL_dict_gsw['dec'] = self.gswcat.dec[self.make_spec]
        self.EL_dict_gsw['av_gsw'] = self.gswcat.av[self.make_spec]
        self.EL_dict_gsw['a_uv_gsw'] = self.gswcat.a_uv[self.make_spec]

        self.EL_dict_gsw['fibsfr']= self.EL_dict_gsw['sfr']+np.log10(self.EL_dict_gsw['massfrac'])
        self.EL_dict_gsw['fibsfrgsw']= self.EL_dict_gsw['sfr']+np.log10(self.EL_dict_gsw['massfracgsw'])
        
        self.EL_dict_gsw['dmpc_samir']=np.log10(samircosmo.luminosity_distance(self.EL_dict_gsw['z']).value)
        
        self.EL_dict_gsw['logoiii_sf'] = (-0.54004897*self.EL_dict_gsw['mass']+0.89790634*self.EL_dict_gsw['fibmass']+
                                        0.11895356*self.EL_dict_gsw['sfr']-0.077016435*self.EL_dict_gsw['a_uv_gsw']+
                                        1.2047141-0.00056521663*self.EL_dict_gsw['hdelta_lick']-1.1270720*(np.log10(self.EL_dict_gsw['d4000']-1.09))-
                                        0.17276752*self.EL_dict_gsw['av_gsw']-1.4106159*self.EL_dict_gsw['dmpc_samir'])
        self.EL_dict_gsw['oiii_sf_sub_samir'] = self.EL_dict_gsw['oiiiflux']-10**(self.EL_dict_gsw['logoiii_sf'])/1e17
        self.oiiinegsfsub = np.where(self.EL_dict_gsw['oiii_sf_sub_samir']<=0)[0]
        self.EL_dict_gsw['oiii_sf_sub_samir'][self.oiiinegsfsub] = np.nan #self.EL_dict_gsw['oiiiflux'][self.oiiinegsfsub]
        

        high_sn10_hb = np.where((self.EL_dict_gsw['hbetaflux_sn']>10)&(self.EL_dict_gsw['halpflux_sn']>0))
        print(high_sn10_hb)
        if weak_lines or not empirdust:

            self.reg_av = None
            self.reg_av_sf = None

        elif empirdust:
            
            self.y_reg = np.array(self.EL_dict_gsw['av_agn'][high_sn10_hb])
            self.y_reg_sf = np.array(self.EL_dict_gsw['av'][high_sn10_hb])
            self.X_reg_av = np.vstack([self.EL_dict_gsw['av_gsw'][high_sn10_hb]]).transpose()
            self.reg_av = LinearRegression().fit(self.X_reg_av,self.y_reg)
            self.reg_av_sf = LinearRegression().fit(self.X_reg_av,self.y_reg_sf)
        
        self.x_pred_av = np.vstack([self.EL_dict_gsw['av_gsw']]).transpose()
        self.EL_dict_gsw['corrected_presub_av'] = correct_av(self.reg_av, self.x_pred_av, 
                                                         np.array(self.EL_dict_gsw['av_agn']),
                                                         np.array(self.EL_dict_gsw['hbetaflux_sn']), empirdust=empirdust)
        self.EL_dict_gsw['corrected_presub_av_sf'] = correct_av(self.reg_av_sf, self.x_pred_av, 
                                                         np.array(self.EL_dict_gsw['av']),
                                                         np.array(self.EL_dict_gsw['hbetaflux_sn']), empirdust=empirdust)
        
        
        '''
        self.X_reg_av_sfr = np.vstack([self.EL_dict_gsw['av_gsw'][high_sn10_hb], self.EL_dict_gsw['sfr'][high_sn10_hb]]).transpose()
        self.reg_av_sfr = LinearRegression().fit(self.X_reg_av_sfr,self.y_reg)
        self.x_pred_av_sfr = np.vstack([self.EL_dict_gsw['av_gsw'], self.EL_dict_gsw['sfr']]).transpose()
        self.EL_dict_gsw['corrected_presub_av_sfr'] = correct_av(self.reg_av_sfr, self.x_pred_av_sfr, 
                                                         np.array(self.EL_dict_gsw['av_agn']),
                                                         np.array(self.EL_dict_gsw['hbetaflux_sn']))
        self.X_reg_av_mass = np.vstack([self.EL_dict_gsw['av_gsw'][high_sn10_hb], self.EL_dict_gsw['mass'][high_sn10_hb]]).transpose()
        self.reg_av_mass = LinearRegression().fit(self.X_reg_av_mass,self.y_reg)
        self.x_pred_av_mass = np.vstack([self.EL_dict_gsw['av_gsw'], self.EL_dict_gsw['mass']]).transpose()
        self.EL_dict_gsw['corrected_presub_av_mass'] = correct_av(self.reg_av_mass, self.x_pred_av_mass, 
                                                         np.array(self.EL_dict_gsw['av_agn']),
                                                         np.array(self.EL_dict_gsw['hbetaflux_sn']))

        self.X_reg_av_mass_sfr = np.vstack([self.EL_dict_gsw['av_gsw'][high_sn10_hb], self.EL_dict_gsw['mass'][high_sn10_hb],self.EL_dict_gsw['sfr'][high_sn10_hb]]).transpose()
        self.reg_av_mass_sfr = LinearRegression().fit(self.X_reg_av_mass_sfr,self.y_reg)
        self.x_pred_av_mass_sfr = np.vstack([self.EL_dict_gsw['av_gsw'], self.EL_dict_gsw['mass'],self.EL_dict_gsw['sfr']]).transpose()
        self.EL_dict_gsw['corrected_presub_av_mass_sfr'] = correct_av(self.reg_av_mass_sfr, self.x_pred_av_mass_sfr, 
                                                         np.array(self.EL_dict_gsw['av_agn']),
                                                         np.array(self.EL_dict_gsw['hbetaflux_sn']))
        '''
        self.EL_dict_gsw['hbetaflux_corr'] = dustcorrect(self.EL_dict_gsw['hbetaflux'], self.EL_dict_gsw['corrected_presub_av'], 4861.0)
        self.EL_dict_gsw['halpflux_corr'] = dustcorrect(self.EL_dict_gsw['halpflux'], self.EL_dict_gsw['corrected_presub_av'], 6563.0)
        
        self.EL_dict_gsw['oiiiflux_corr'] = dustcorrect(self.EL_dict_gsw['oiiiflux'], self.EL_dict_gsw['corrected_presub_av'], 5007.0)
        self.EL_dict_gsw['oiiiflux_corr_sf_sub_samir'] = dustcorrect(self.EL_dict_gsw['oiii_sf_sub_samir'], self.EL_dict_gsw['corrected_presub_av'], 5007.0)

        self.EL_dict_gsw['oiii_err_corr'] = dustcorrect(self.EL_dict_gsw['oiii_err'], self.EL_dict_gsw['corrected_presub_av'], 5007.0)
        
        self.EL_dict_gsw['oiiflux_corr'] = dustcorrect(self.EL_dict_gsw['oiiflux'], self.EL_dict_gsw['corrected_presub_av'], 3727.0)
        self.EL_dict_gsw['niiflux_corr'] = dustcorrect(self.EL_dict_gsw['niiflux'], self.EL_dict_gsw['corrected_presub_av'], 6583.0)
        self.EL_dict_gsw['oiflux_corr'] = dustcorrect(self.EL_dict_gsw['oiflux'], self.EL_dict_gsw['corrected_presub_av'], 6300.0)
        self.EL_dict_gsw['siiflux_corr'] = dustcorrect(self.EL_dict_gsw['siiflux'], self.EL_dict_gsw['corrected_presub_av'], 6724.0)
        self.EL_dict_gsw['sii6717flux_corr'] = dustcorrect(self.EL_dict_gsw['sii6717flux'], self.EL_dict_gsw['corrected_presub_av'], 6717.0)
        self.EL_dict_gsw['sii6731flux_corr'] = dustcorrect(self.EL_dict_gsw['sii6731flux'], self.EL_dict_gsw['corrected_presub_av'], 6731.0)


        self.EL_dict_gsw['hbetaflux_corr_sf'] = dustcorrect(self.EL_dict_gsw['hbetaflux'], self.EL_dict_gsw['corrected_presub_av_sf'], 4861.0)
        self.EL_dict_gsw['halpflux_corr_sf'] = dustcorrect(self.EL_dict_gsw['halpflux'], self.EL_dict_gsw['corrected_presub_av_sf'], 6563.0)
        
        self.EL_dict_gsw['oiiiflux_corr_sf'] = dustcorrect(self.EL_dict_gsw['oiiiflux'], self.EL_dict_gsw['corrected_presub_av_sf'], 5007.0)
        self.EL_dict_gsw['oiii_err_corr_sf'] = dustcorrect(self.EL_dict_gsw['oiii_err'], self.EL_dict_gsw['corrected_presub_av_sf'], 5007.0)
        
        self.EL_dict_gsw['oiiflux_corr_sf'] = dustcorrect(self.EL_dict_gsw['oiiflux'], self.EL_dict_gsw['corrected_presub_av_sf'], 3727.0)
        self.EL_dict_gsw['niiflux_corr_sf'] = dustcorrect(self.EL_dict_gsw['niiflux'], self.EL_dict_gsw['corrected_presub_av_sf'], 6583.0)
        self.EL_dict_gsw['oiflux_corr_sf'] = dustcorrect(self.EL_dict_gsw['oiflux'], self.EL_dict_gsw['corrected_presub_av_sf'], 6300.0)
        self.EL_dict_gsw['siiflux_corr_sf'] = dustcorrect(self.EL_dict_gsw['siiflux'], self.EL_dict_gsw['corrected_presub_av_sf'], 6724.0)
        self.EL_dict_gsw['sii6717flux_corr_sf'] = dustcorrect(self.EL_dict_gsw['sii6717flux'], self.EL_dict_gsw['corrected_presub_av_sf'], 6717.0)
        self.EL_dict_gsw['sii6731flux_corr_sf'] = dustcorrect(self.EL_dict_gsw['sii6731flux'], self.EL_dict_gsw['corrected_presub_av_sf'], 6731.0)

        self.EL_dict_gsw['hbetaflux_corr_p1'] = dustcorrect(self.EL_dict_gsw['hbetaflux'], self.EL_dict_gsw['av'], 4861.0)
        self.EL_dict_gsw['halpflux_corr_p1'] = dustcorrect(self.EL_dict_gsw['halpflux'], self.EL_dict_gsw['av'], 6563.0)
        
        self.EL_dict_gsw['oiiiflux_corr_p1'] = dustcorrect(self.EL_dict_gsw['oiiiflux'], self.EL_dict_gsw['av'], 5007.0)
        
        self.EL_dict_gsw['oiiflux_corr_p1'] = dustcorrect(self.EL_dict_gsw['oiiflux'], self.EL_dict_gsw['av'], 3727.0)
        self.EL_dict_gsw['niiflux_corr_p1'] = dustcorrect(self.EL_dict_gsw['niiflux'], self.EL_dict_gsw['av'], 6583.0)
        self.EL_dict_gsw['oiflux_corr_p1'] = dustcorrect(self.EL_dict_gsw['oiflux'], self.EL_dict_gsw['av'], 6300.0)
        self.EL_dict_gsw['siiflux_corr_p1'] = dustcorrect(self.EL_dict_gsw['siiflux'], self.EL_dict_gsw['av'], 6724.0)
        self.EL_dict_gsw['sii6717flux_corr_p1'] = dustcorrect(self.EL_dict_gsw['sii6717flux'], self.EL_dict_gsw['av'], 6717.0)
        self.EL_dict_gsw['sii6731flux_corr_p1'] = dustcorrect(self.EL_dict_gsw['sii6731flux'], self.EL_dict_gsw['av'], 6731.0)

        self.EL_dict_gsw['oiii_oii'] = np.log10(self.EL_dict_gsw['oiiiflux_corr']/self.EL_dict_gsw['oiiflux_corr'])
        self.EL_dict_gsw['oiii_oi'] = np.log10(self.EL_dict_gsw['oiiiflux_corr']/self.EL_dict_gsw['oiflux_corr'])
        self.EL_dict_gsw['oiii_nii'] = np.log10(self.EL_dict_gsw['oiiiflux_corr']/self.EL_dict_gsw['niiflux_corr'])

        self.EL_dict_gsw['U'] =oiii_oii_to_U(self.EL_dict_gsw['oiii_oii'] )
        self.EL_dict_gsw['nii_oii'] =np.log10(self.EL_dict_gsw['niiflux']/self.EL_dict_gsw['oiiflux_corr'])
        self.EL_dict_gsw['log_oh'] = nii_oii_to_oh(self.EL_dict_gsw['niiflux_corr'], self.EL_dict_gsw['oiiflux_corr'])  
        self.EL_dict_gsw['log_oh_ke02'] = nii_oii_to_oh_ke02(self.EL_dict_gsw['niiflux_corr'], self.EL_dict_gsw['oiiflux_corr'])  
           
        self.EL_dict_gsw['sii_ratio'] =(self.EL_dict_gsw['sii6717flux_corr']/self.EL_dict_gsw['sii6731flux_corr'])
        self.EL_dict_gsw['sii_oii'] =np.log10(self.EL_dict_gsw['siiflux']/self.EL_dict_gsw['oiiflux'])
        self.EL_dict_gsw['oi_sii'] =np.log10(self.EL_dict_gsw['oiflux']/self.EL_dict_gsw['siiflux'])

        
        self.EL_dict_gsw['oiiilum'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux_corr'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['edd_ratio'] = self.EL_dict_gsw['oiiilum']+np.log10(600)-self.EL_dict_gsw['edd_lum']

        self.EL_dict_gsw['oilum'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiflux_corr'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['oiilum'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiflux_corr'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['niilum'] = np.log10(getlumfromflux(self.EL_dict_gsw['niiflux_corr'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['siilum'] = np.log10(getlumfromflux(self.EL_dict_gsw['siiflux_corr'],self.EL_dict_gsw['z']))

        self.EL_dict_gsw['oiiilum_sfsub_samir'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux_corr_sf_sub_samir'],self.EL_dict_gsw['z']))

        self.EL_dict_gsw['oiiilum_up'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux_corr']+self.EL_dict_gsw['oiii_err_corr'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['oiiilum_up2'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux_corr']+2*self.EL_dict_gsw['oiii_err_corr'],self.EL_dict_gsw['z']))

        self.EL_dict_gsw['oiiilum_down'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux_corr']-self.EL_dict_gsw['oiii_err_corr'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['e_oiiilum_down'] = self.EL_dict_gsw['oiiilum']-self.EL_dict_gsw['oiiilum_down']
        self.EL_dict_gsw['e_oiiilum_up'] = self.EL_dict_gsw['oiiilum_up']- self.EL_dict_gsw['oiiilum']

        self.EL_dict_gsw['nlr_rad_from_lo3'] = self.EL_dict_gsw['oiiilum']*(0.42)-13.97
        
        self.EL_dict_gsw['fibsize'] = cosmo.angular_diameter_distance(self.EL_dict_gsw['z']/206265.)*1000.*1.5 #in kpc
        self.EL_dict_gsw['nlr_fib_ratio'] = self.EL_dict_gsw['nlr_rad_from_lo3']/self.EL_dict_gsw['fibsize']
        self.EL_dict_gsw['halplum'] = np.log10(getlumfromflux(self.EL_dict_gsw['halpflux_corr'], self.EL_dict_gsw['z']))
        self.EL_dict_gsw['edd_par'] = self.EL_dict_gsw['oiiilum']-self.EL_dict_gsw['mbh']

        self.EL_dict_gsw['oiiilum_sf'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux_corr_sf'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['halplum_sf'] = np.log10(getlumfromflux(self.EL_dict_gsw['halpflux_corr_sf'], self.EL_dict_gsw['z']))

        self.EL_dict_gsw['oiiilum_p1'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux_corr_p1'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['halplum_p1'] = np.log10(getlumfromflux(self.EL_dict_gsw['halpflux_corr_p1'], self.EL_dict_gsw['z']))

        self.EL_dict_gsw['oiiilum_uncorr'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux'],self.EL_dict_gsw['z']))
        
        self.EL_dict_gsw['oiiilum_up_uncorr'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux']+self.EL_dict_gsw['oiii_err'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['oiiilum_down_uncorr'] = np.log10(getlumfromflux(self.EL_dict_gsw['oiiiflux']-self.EL_dict_gsw['oiii_err'],self.EL_dict_gsw['z']))
        self.EL_dict_gsw['e_oiiilum_up_uncorr'] = self.EL_dict_gsw['oiiilum_up_uncorr']-self.EL_dict_gsw['oiiilum_uncorr']
        self.EL_dict_gsw['e_oiiilum_down_uncorr'] = self.EL_dict_gsw['oiiilum_uncorr']- self.EL_dict_gsw['oiiilum_down_uncorr']
        
        self.EL_dict_gsw['halplum_uncorr'] = np.log10(getlumfromflux(self.EL_dict_gsw['halpflux'], self.EL_dict_gsw['z']))
        self.EL_dict_gsw['halpfibsfr'] = halptofibsfr_corr(10**self.EL_dict_gsw['halplum'])
        self.EL_dict_gsw['halpfibsfr_uncorr'] = halptofibsfr_corr(10**self.EL_dict_gsw['halplum_uncorr'])

        self.EL_dict_gsw['halpfibsfr_sf'] = halptofibsfr_corr(10**self.EL_dict_gsw['halplum_sf'])

        if self.xr:
            self.EL_dict_gsw['full_xraylum'] = self.gswcat.gsw_df.fulllumsrf.iloc[self.make_spec]
            self.EL_dict_gsw['soft_xraylum'] = self.gswcat.gsw_df.softlumsrf.iloc[self.make_spec]
            self.EL_dict_gsw['hard_xraylum'] = self.gswcat.gsw_df.hardlumsrf.iloc[self.make_spec]
            self.EL_dict_gsw['edd_par_xr'] = self.EL_dict_gsw['hard_xraylum']-self.EL_dict_gsw['edd_lum']

            self.EL_dict_gsw['lo3_pred_fromlx'] = (self.EL_dict_gsw['hard_xraylum']+7.55)/(1.22)
            self.EL_dict_gsw['nlr_rad_from_lo3_pred_fromlx'] = self.EL_dict_gsw['lo3_pred_fromlx']*(0.42)-13.97
            self.EL_dict_gsw['nlr_fib_ratio_pred_fromlx'] = 10**self.EL_dict_gsw['nlr_rad_from_lo3_pred_fromlx']/1000/self.EL_dict_gsw['fibsize']
                    
            self.EL_dict_gsw['lo3_offset'] = -(self.EL_dict_gsw['lo3_pred_fromlx']-self.EL_dict_gsw['oiiilum']    )  
            self.EL_dict_gsw['lo3_offset_up'] = -(self.EL_dict_gsw['lo3_pred_fromlx']-self.EL_dict_gsw['oiiilum_up']    )  
            self.EL_dict_gsw['lo3_offset_up2'] = -(self.EL_dict_gsw['lo3_pred_fromlx']-self.EL_dict_gsw['oiiilum_up2']    )  
            
            self.EL_dict_gsw['fo3_pred_fromlx'] = redden(getfluxfromlum(10**self.EL_dict_gsw['lo3_pred_fromlx'], self.EL_dict_gsw['z']),
                                                         self.EL_dict_gsw['corrected_presub_av'], 5007.0)
            self.EL_dict_gsw['lo3_minus_pred_fromlx'] = (self.EL_dict_gsw['hard_xraylum']+7.55)/(1.22)-0.587 #subtracting dispersion?
            self.EL_dict_gsw['fo3_minus_pred_fromlx'] = redden(getfluxfromlum(10**self.EL_dict_gsw['lo3_minus_pred_fromlx'], self.EL_dict_gsw['z']),
                                                               self.EL_dict_gsw['corrected_presub_av'], 5007.0 )
            
            self.EL_dict_gsw['fo3_dev'] = (self.EL_dict_gsw['fo3_pred_fromlx']-self.EL_dict_gsw['oiiiflux'])/(self.EL_dict_gsw['fo3_pred_fromlx']-self.EL_dict_gsw['fo3_minus_pred_fromlx'])
            
            self.EL_dict_gsw['fullflux'] = self.gswcat.gsw_df.fullflux.iloc[self.make_spec]
            
            self.EL_dict_gsw['softflux'] = self.gswcat.gsw_df.softflux.iloc[self.make_spec]
            self.EL_dict_gsw['hardflux'] = self.gswcat.gsw_df.hardflux.iloc[self.make_spec]
            self.EL_dict_gsw['full_lxsfr'] = np.log10(xrayranallidict['full']*10**(self.EL_dict_gsw['sfr']))
            self.EL_dict_gsw['hard_lxsfr'] = np.log10(xrayranallidict['hard']*10**(self.EL_dict_gsw['sfr']))
            self.EL_dict_gsw['xray_agn_status'] = self.EL_dict_gsw['full_xraylum']-0.6 > self.EL_dict_gsw['full_lxsfr']
            self.EL_dict_gsw['xray_excess'] = self.EL_dict_gsw['full_xraylum']- self.EL_dict_gsw['full_lxsfr']
            
            self.EL_dict_gsw['hardlx_gas'] = np.log10(7.3e39*10**(self.EL_dict_gsw['sfr'])) #mineo+2012
            self.EL_dict_gsw['hardlx_xrb'] = np.log10( (10**(29.37)*(10**self.EL_dict_gsw['mass'])*(1+self.EL_dict_gsw['z'])**2.03 )+
                                                              (10**self.EL_dict_gsw['sfr'])*(10**39.28)*(1+self.EL_dict_gsw['z'])**1.31)
            self.EL_dict_gsw['softlx_xrb'] = np.log10( (10**(29.04)*(10**self.EL_dict_gsw['mass'])*(1+self.EL_dict_gsw['z'])**3.78 )+
                                                              (10**self.EL_dict_gsw['sfr'])*(10**39.28)*(1+self.EL_dict_gsw['z'])**0.99)
            self.EL_dict_gsw['fulllx_xrb'] = np.log10(10**self.EL_dict_gsw['softlx_xrb']+10**self.EL_dict_gsw['hardlx_xrb'])
            self.EL_dict_gsw['hardlx_xrb_local'] = np.log10( 10**(28.96)*(10**self.EL_dict_gsw['mass'])+
                                                              10**self.EL_dict_gsw['sfr']*(10**39.21))

            
            
            self.EL_dict_gsw['hardlx_sfr'] = np.log10(xrayranallidict['hard']*10**(self.EL_dict_gsw['sfr']))
            self.EL_dict_gsw['softlx_sfr'] = np.log10(xrayranallidict['soft']*10**(self.EL_dict_gsw['sfr']))
            self.EL_dict_gsw['full_lxagn'] = np.log10(10**self.EL_dict_gsw['full_xraylum']-10**self.EL_dict_gsw['full_lxsfr'])
            self.EL_dict_gsw['hard_lxagn'] = np.log10(10**self.EL_dict_gsw['hard_xraylum']-10**self.EL_dict_gsw['hardlx_sfr'])
            self.EL_dict_gsw['soft_lxagn'] = np.log10(10**self.EL_dict_gsw['soft_xraylum']-10**self.EL_dict_gsw['softlx_sfr'])

            if xmm:
                self.EL_dict_gsw['exptimes'] = self.gswcat.gsw_df.exptimes.iloc[self.make_spec]
                self.EL_dict_gsw['xrayflag'] = self.gswcat.xrayflag[self.make_spec]
                self.EL_dict_gsw['ext'] = self.gswcat.ext[self.make_spec]
                self.EL_dict_gsw['efullflux'] = self.gswcat.gsw_df.efullflux.iloc[self.make_spec]
                self.EL_dict_gsw['fullflux_sn'] = self.EL_dict_gsw['fullflux']/self.EL_dict_gsw['efullflux']
                
                self.EL_dict_gsw['ehardflux'] = self.gswcat.gsw_df.efullflux.iloc[self.make_spec]
                self.EL_dict_gsw['hardflux_sn'] = self.EL_dict_gsw['hardflux']/self.EL_dict_gsw['ehardflux']
                
                self.EL_dict_gsw['esoftflux'] = self.gswcat.gsw_df.efullflux.iloc[self.make_spec]
                self.EL_dict_gsw['softflux_sn'] = self.EL_dict_gsw['softflux']/self.EL_dict_gsw['esoftflux']

            
            
                self.EL_dict_gsw['full_xraylum_up'] = self.gswcat.gsw_df.efulllumsrf_up.iloc[self.make_spec]
                self.EL_dict_gsw['full_xraylum_down'] = self.gswcat.gsw_df.efulllumsrf_down.iloc[self.make_spec]

                self.EL_dict_gsw['e_full_xraylum_up'] = self.EL_dict_gsw['full_xraylum_up'] -self.EL_dict_gsw['full_xraylum']
                self.EL_dict_gsw['e_full_xraylum_down'] = self.EL_dict_gsw['full_xraylum'] -self.EL_dict_gsw['full_xraylum_down']

                self.EL_dict_gsw['hard_xraylum_up'] = self.gswcat.gsw_df.ehardlumsrf_up.iloc[self.make_spec]
                self.EL_dict_gsw['hard_xraylum_down'] = self.gswcat.gsw_df.ehardlumsrf_down.iloc[self.make_spec]

                self.EL_dict_gsw['e_hard_xraylum_up'] = self.EL_dict_gsw['hard_xraylum_up'] - self.EL_dict_gsw['hard_xraylum']
                self.EL_dict_gsw['e_hard_xraylum_down'] = self.EL_dict_gsw['hard_xraylum'] - self.EL_dict_gsw['hard_xraylum_down']

            
                self.EL_dict_gsw['hr1'] = self.gswcat.gsw_df.hr1.iloc[self.make_spec]
                self.EL_dict_gsw['hr2'] = self.gswcat.gsw_df.hr2.iloc[self.make_spec]
                self.EL_dict_gsw['hr3'] = self.gswcat.gsw_df.hr3.iloc[self.make_spec]
                self.EL_dict_gsw['hr4'] = self.gswcat.gsw_df.hr4.iloc[self.make_spec]
            
        if radio:   
            self.EL_dict_gsw['nvss_flux'] = self.gswcat.gsw_df.nvss_flux.iloc[self.make_spec]
            self.EL_dict_gsw['first_flux'] = self.gswcat.gsw_df.first_flux.iloc[self.make_spec]
            self.EL_dict_gsw['wenss_flux'] = self.gswcat.gsw_df.wenss_flux.iloc[self.make_spec]
            self.EL_dict_gsw['vlss_flux'] = self.gswcat.gsw_df.vlss_flux.iloc[self.make_spec]
            self.EL_dict_gsw['nvss_lums'] = self.gswcat.gsw_df.nvss_lums.iloc[self.make_spec]
            self.EL_dict_gsw['first_lums'] = self.gswcat.gsw_df.firstlums.iloc[self.make_spec]
            self.EL_dict_gsw['wenss_lums'] = self.gswcat.gsw_df.wensslums.iloc[self.make_spec]
            self.EL_dict_gsw['vlss_lums'] = self.gswcat.gsw_df.vlsslums.iloc[self.make_spec]

        self.EL_gsw_df = pd.DataFrame.from_dict(self.EL_dict_gsw)
        allbptgroups, allbptsf, allbptagn = get_bpt1_groups( np.log10(self.EL_gsw_df['xvals1_bpt']), np.log10(self.EL_gsw_df['yvals_bpt'] ) )
        allbptplusgroups, allbptplssf, allbptplsagn = get_bptplus_groups(np.log10(self.EL_gsw_df['xvals1_bpt']), np.log10(self.EL_gsw_df['yvals_bpt']))
        allbptplusniigroups, allbptplsnii_sf, allbptplsnii_agn= get_bptplus_niigroups(np.log10(self.EL_gsw_df['xvals1_bpt']))

        self.EL_gsw_df['bptplusgroups'] = allbptplusgroups
        self.EL_gsw_df['bptplusniigroups'] = allbptplusniigroups

        self.EL_gsw_df['bptgroups'] = allbptgroups
        classifiability = get_classifiability(self.EL_gsw_df['niiflux_sn'], 
            self.EL_gsw_df['halpflux_sn'],self.EL_gsw_df['oiiiflux_sn'],self.EL_gsw_df['hbetaflux_sn'] )
        self.EL_gsw_df['classifiability'] = classifiability
        
        self.not_bpt_EL_gsw_df = self.EL_gsw_df.iloc[self.not_bpt_sn_filt].copy() #unclassifiables from Paper 1
        self.high_sn_o3_EL_gsw_df = self.EL_gsw_df.iloc[self.high_sn_o3].copy() #unclassifiables from Paper 1
        
        self.bpt_EL_gsw_df = self.EL_gsw_df.iloc[self.bpt_sn_filt].copy()
        
        high_sn_all7 = np.where(( self.bpt_EL_gsw_df.oiiflux_sn>2) &(self.bpt_EL_gsw_df.oiflux_sn>2) &(self.bpt_EL_gsw_df.siiflux_sn>2))
        self.alllines_bpt_EL_gsw_df = self.bpt_EL_gsw_df.iloc[high_sn_all7].copy()
        
        self.vo87_1_EL_gsw_df = self.bpt_EL_gsw_df.iloc[self.vo87_1_filt].copy()
        self.vo87_2_EL_gsw_df = self.bpt_EL_gsw_df.iloc[self.vo87_2_filt].copy()
        

        bptgroups, bptsf, bptagn = get_bpt1_groups( np.log10(self.bpt_EL_gsw_df['xvals1_bpt']), np.log10(self.bpt_EL_gsw_df['yvals_bpt'] ) )

 
        bptplsugroups, bptplssf, bptplsagn = get_bptplus_groups(np.log10(self.bpt_EL_gsw_df['xvals1_bpt']), np.log10(self.bpt_EL_gsw_df['yvals_bpt']))
        
        
        
        self.bpt_sf_df = self.bpt_EL_gsw_df.iloc[bptsf].copy()

        high_sn_all7_sf = np.where(( self.bpt_sf_df.oiiflux_sn>2) &(self.bpt_sf_df.oiflux_sn>2) &(self.bpt_sf_df.siiflux_sn>2))

        self.alllines_bpt_sf_df = self.bpt_sf_df.iloc[high_sn_all7_sf]
        self.bpt_agn_df = self.bpt_EL_gsw_df.iloc[bptagn].copy()
        
        self.bptplus_sf_df = self.bpt_EL_gsw_df.iloc[bptplssf].copy()
        self.bptplus_agn_df = self.bpt_EL_gsw_df.iloc[bptplsagn].copy()
        high_sn_all7_bptplussf = np.where(( self.bptplus_sf_df.oiiflux_sn>2) &(self.bptplus_sf_df.oiflux_sn>2) &(self.bptplus_sf_df.siiflux_sn>2))
        self.alllines_bptplus_sf_df = self.bptplus_sf_df.iloc[high_sn_all7_bptplussf]
                                                                                                                                                                                           
        self.plus_EL_gsw_df = self.EL_gsw_df.iloc[self.halp_nii_filt].copy()

        groups_bptplusnii, bptplsnii_sf, bptplsnii_agn= get_bptplus_niigroups(np.log10(self.plus_EL_gsw_df['xvals1_bpt']))

        self.bptsf = bptsf
        self.bptagn = bptagn
        self.bptplssf = bptplssf
        self.bptplsagn = bptplsagn
        self.bptplsnii_sf = bptplsnii_sf
        self.bptplsnii_agn = bptplsnii_agn
        #self.plus_EL_gsw_df['bptplusniigroups'] = groups_bptplusnii
        
        self.bptplusnii_sf_df = self.plus_EL_gsw_df.iloc[bptplsnii_sf].copy()
        self.bptplusnii_agn_df = self.plus_EL_gsw_df.iloc[bptplsnii_agn].copy()        
        self.neither_EL_gsw_df = self.EL_gsw_df.iloc[self.neither_filt].copy()
        self.allnonagn_df=pd.concat([self.bptplus_sf_df,self.bptplusnii_sf_df, self.neither_EL_gsw_df],join='outer')
        self.lines = ['oiiiflux','hbetaflux','halpflux','niiflux', 'siiflux',
                  'sii6731flux', 'sii6717flux','oiflux','oiii4363flux',
                  'oiii4959flux','oiiflux']
        
        self.line_errs = ['oiii_err', 'hbeta_err', 'halp_err', 'nii_err', 
                    'sii_err','sii6731_err', 'sii6717_err', 'oi_err',
                    'oiii4363_err', 'oiii4959_err', 'oii_err']
    def get_pred_av(self):
        xmids_gsw = []
        yavs_gsw = []
        av_avgs = []
        xmids_agn = []
        yavs_agn = []
        av_avgs_agn = []
        xmids_agn2 = []
        av_avgs_agn2 = []
        yavs_agn2 = []
        
        
        for i in range(len(self.allnonagn_avgsw_filts_df)):
            plt.figure()
            xmid, yav, av_av = plot2dhist(self.allnonagn_avgsw_filts_df[i]['sfr'], 
                                          self.allnonagn_avgsw_filts_df[i]['av'],
                                          bin_quantity = self.allnonagn_avgsw_filts_df[i]['av_gsw'],
                                          nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5.,
                                          lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$',
                                          setplotlims=True, bin_y=True)
            av_avgs.append(av_av)  
            xmids_gsw.append(xmid)
            yavs_gsw.append(yav)
            plt.close()
            xmid, yav, av_agn2 = plot2dhist(self.agn_avgsw_filts_df[i]['sfr'], self.agn_avgsw_filts_df[i]['av_agn'],
                                   bin_quantity = self.agn_avgsw_filts_df[i]['av_gsw'],
                                   nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., 
                                   lim=True, xlabel='log(SFR)', ylabel=r'A$_{\mathrm{V}}$', 
                                   setplotlims=True, bin_y=True)
        
            xmids_agn2.append(xmid)
            yavs_agn2.append(yav)
            av_avgs_agn2.append(av_agn2)
            plt.close()
            
            xmid, yav, av_agn = plot2dhist(self.agn_avgsw_filts_df[i]['sfr'], self.agn_avgsw_filts_df[i]['av'], 
                                   bin_quantity = self.agn_avgsw_filts_df[i]['av_gsw'],
                                   nan=True, maxx=3.5, minx=-3.5, miny=-5, maxy=5., lim=True, xlabel='log(SFR)',
                                   ylabel=r'A$_{\mathrm{V}}$', setplotlims=True, bin_y=True)
        
            xmids_agn.append(xmid)
            yavs_agn.append(yav)
            av_avgs_agn.append(av_agn)
            plt.close()            
        self.xmids_gsw = np.array(xmids_gsw)
        self.yavs_gsw = np.array(yavs_gsw)
        self.av_avgs = np.array(av_avgs)
        self.xmids_agn = np.array(xmids_agn)
        self.xmids_agn2 = np.array(xmids_agn2)
        self.yavs_agn = np.array(yavs_agn)
        self.yavs_agn2 = np.array(yavs_agn2)
        self.av_avgs_agn = np.array(av_avgs_agn)
        self.av_avgs_agn2 = np.array(av_avgs_agn)
        
        X = np.vstack([self.allnonagn_hb_sn_filts_df[2].sfr, self.allnonagn_hb_sn_filts_df[2].av_gsw]).transpose()
        X_agn = np.vstack([self.agn_avgsw_filts_df[2].sfr, self.agn_avgsw_filts_df[2].av_gsw]).transpose()
        
        self.X_reg = X
        self.X_reg_agn = X_agn
        y = np.array(np.transpose(self.allnonagn_hb_sn_filts_df[2].av))
        y_agn = np.array(np.transpose(self.agn_avgsw_filts_df[2].av))
        y_agn2 = np.array(np.transpose(self.agn_avgsw_filts_df[2].av_agn))

        reg = LinearRegression().fit(X,y)
        reg_agn = LinearRegression().fit(X_agn,y_agn)
        reg_agn2 = LinearRegression().fit(X_agn,y_agn2)

        ypreds = []
        for i in range(len(xmids_gsw)):
            X_test = np.vstack([xmids_gsw[i],av_avgs[i]]).transpose()
            ypred = reg.predict(X_test)
            ypreds.append(ypred)
        self.sfr_av_reg = reg
        self.sfr_av_ypreds = ypreds
        
        ypreds_agn = []
        for i in range(len(xmids_agn)):
            X_test = np.vstack([xmids_agn[i],av_avgs_agn[i]]).transpose()
            ypred = reg_agn.predict(X_test)
            ypreds_agn.append(ypred)
        self.sfr_av_reg_agn = reg_agn
        self.sfr_av_ypreds_agn = ypreds_agn
        
        ypreds_agn2 = []
        for i in range(len(xmids_agn2)):
            X_test = np.vstack([xmids_agn2[i],av_avgs_agn2[i]]).transpose()
            ypred = reg_agn2.predict(X_test)
            ypreds_agn2.append(ypred)
        self.sfr_av_reg_agn2 = reg_agn2
        self.sfr_av_ypreds_agn2 = ypreds_agn2




    def correct_av(self,reg, sfr, av_gsw, av_balm, hb_sn):
        x_test = np.vstack([sfr, av_gsw])
        av_balm_fixed = []
        for i in range(len(hb_sn)):

            if hb_sn[i] <5:
                x_samp = np.array([x_test[:,i]])
                av_fix = np.float64(reg.predict(x_samp))
                if av_fix<0:
                    av_fix=0
            elif av_balm[i] <0:
                av_fix = 0
            else:
                av_fix = av_balm[i]
            av_balm_fixed.append(av_fix)
        return np.array(av_balm_fixed)    
            
            
class Galindx(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)        
            'alld4000': self.data['D4000_N'],
            'hdelta_lick': self.data['LICK_HD_A'],
            'tauv_cont': self.data['TAUV_CONT'],
            'allvdisp': self.data['V_DISP'],        
class Gal_Fib(AstroTablePD):
    def __init__(self, filename, output):
        super().__init__(filename)        
        self.data[output] =  self.data['AVG'],


galmass = Gal_Fib(galmass, 'all_sdss_avgmasses')
fibmass = Gal_Fib(fibmass, 'all_fibmass')
fibsfr = Gal_Fib(fibsfr, 'all_fibmass')
fibssfr = Gal_Fib(fibssfr, 'all_fibssfr_mpa')


class SDSSObj(self):
    def __init__(self, galfiboh_file, galinfo_file, galline_file, galindx_file, fibmass_file, fibsfr_file, fibssfr_file,)







import numpy as np

lum_arr = np.logspace(34,46,1000)
loglum_arr = np.log10(lum_arr)

sfrsoft = 2.2e-40*lum_arr
logsfrsoft = np.log10(sfrsoft)#from salpeter IMF
logsfrsoft = logsfrsoft - 0.2 #converted to Chabrier IMF

sfrhard = 2.0e-40*lum_arr
logsfrhard = np.log10(sfrhard)#from salpeter IMF
logsfrhard =logsfrhard - 0.2 #converted to Chabrier IMF
#combine the supposed sfr form soft and hard to get the full
sfrfull= 1.05e-40*lum_arr
logsfrfull = np.log10(sfrfull) -0.2
xrayranallidict = {'soft':1/(1.39e-40),'hard':1/(1.26e-40),'full':1/(0.66e-40)}


class Xraysfr:
    def __init__(self, xraylums, gswcat, filt, agn, nonagn, typ):
        self.typ = typ
        self.filt = filt
        #self.unclassifiable = np.array([i for i in range(len(xraylums)) if i not in self.filt   ])
        self.mass = gswcat.mass
        self.z = gswcat.z
        self.z_filt = self.z[self.filt]
        self.ra = gswcat.ra
        self.ra_filt = gswcat.ra[self.filt]
        self.dec = gswcat.dec
        self.dec_filt = gswcat.dec[self.filt]
        #self.xrayra = gswcat.matchxrayra
        #self.xrayra_filt = gswcat.matchxrayra[self.filt]
        #self.xraydec = gswcat.matchxraydec
        #self.xraydec_filt = gswcat.matchxraydec[self.filt]
        self.lum_mass =  xraylums-np.array(gswcat.gsw_df.mass) #mass_m2_match
        self.lum = xraylums
        self.sfr_mass = np.array(gswcat.gsw_df.sfr)-np.array(gswcat.gsw_df.mass)# sfr_m2_match-mass_m2_match
        self.sfr = np.array(gswcat.gsw_df.sfr)
        self.lxsfr = np.log10(xrayranallidict[typ]*10**(self.sfr))
        self.lxsfr_filt = np.log10(xrayranallidict[typ]*10**(self.sfr[self.filt]))
 
        self.notagn = np.where(self.lum[self.filt]==0)
        self.mass_filt = self.mass[self.filt]
        self.lum_mass_filt = self.lum_mass[self.filt]
        self.sfr_filt = self.sfr[self.filt]
        self.sfr_mass_filt = self.sfr_mass[self.filt]
        self.lum_filt = self.lum[self.filt]
        #self.mess = np.where((self.lum_mass_val >29) & (self.lum_mass_val<30) & (self.softsfr_mass_val >-11))[0]
        self.valid = np.where(self.lum[self.filt] >0)[0]
        self.validagn = np.where(self.lum[self.filt][agn]>0)[0]
        self.validnoagn = np.where(self.lum[self.filt][nonagn]>0)[0]
        #valid overall
        self.lum_mass_val_filt =  self.lum_mass[self.filt][self.valid] #mass_m2_match
        self.lum_val_filt = self.lum[self.filt][self.valid]
        self.sfr_mass_val_filt = self.sfr_mass[self.filt][self.valid]
        self.sfr_val_filt = self.sfr[self.filt][self.valid]
        self.mass_val_filt = self.mass[self.filt][self.valid]
        #valid bpt agn
        self.lum_mass_val_filtagn =  self.lum_mass[self.filt][agn][self.validagn]
        self.lum_val_filtagn = self.lum[self.filt][agn][self.validagn]
        self.sfr_mass_val_filtagn = self.sfr_mass[self.filt][agn][self.validagn]
        self.sfr_val_filtagn = self.sfr[self.filt][agn][self.validagn]
        self.mass_filtagn = self.mass[self.filt][agn][self.validagn]

        #valid bpt hii
        self.lum_mass_val_filtnoagn =  self.lum_mass[self.filt][nonagn][self.validnoagn]
        self.lum_val_filtnoagn = self.lum[self.filt][nonagn][self.validnoagn]
        self.sfr_mass_val_filtnoagn = self.sfr_mass[self.filt][nonagn][self.validnoagn]
        self.sfr_val_filtnoagn = self.sfr[self.filt][nonagn][self.validnoagn]
        self.mass_filtnoagn = self.mass[self.filt][nonagn][self.validnoagn]

        self.likelyagn_xr = np.where((self.lxsfr[self.filt][self.valid] < self.lum[self.filt][self.valid] - 0.6) &
                                     (self.lum[self.filt][self.valid] > 0))[0]
        self.likelyagnbpthii = np.where((self.lxsfr[self.filt][nonagn][self.validnoagn] < self.lum[self.filt][nonagn][self.validnoagn]-0.6) &
                                        (self.lum[self.filt][nonagn][self.validnoagn] >0))[0]
        self.likelyagnbptagn = np.where((self.lxsfr[self.filt][agn][self.validagn] < self.lum[self.filt][agn][self.validagn]-0.6) &
                                        (self.lum[self.filt][agn][self.validagn] >0))[0]
        self.likelysf = np.where((abs(self.lxsfr[self.filt][self.valid] - self.lum[self.filt][self.valid]) < 0.6) &
                                     (self.lum[self.filt][self.valid] > 0))[0]
        self.likelysfbpthii = np.where((abs(self.lxsfr[self.filt][nonagn][self.validnoagn] - self.lum[self.filt][nonagn][self.validnoagn])<0.6) &
                                        (self.lum[self.filt][nonagn][self.validnoagn] >0))[0]
        self.likelysfbptagn = np.where((abs(self.lxsfr[self.filt][agn][self.validagn] - self.lum[self.filt][agn][self.validagn])<0.6) &
                                        (self.lum[self.filt][agn][self.validagn] >0))[0]
        #valid xray agn
        self.lum_mass_val_filt_xrayagn =  self.lum_mass[self.filt][self.valid][self.likelyagn_xr]
        self.lum_val_filt_xrayagn = self.lum[self.filt][self.valid][self.likelyagn_xr ]
        self.sfr_mass_val_filt_xrayagn = self.sfr_mass[self.filt][self.valid][self.likelyagn_xr]
        self.sfr_val_filt_xrayagn = self.sfr[self.filt][self.valid][self.likelyagn_xr ]
        self.mass_filt_xrayagn = self.mass[self.filt][self.valid][self.likelyagn_xr ]
        self.z_filt_xrayagn = self.z[self.filt][self.valid][self.likelyagn_xr]
        
        #valid bpt agn

        self.lum_mass_val_filtagn_xrayagn =  self.lum_mass[self.filt][agn][self.validagn][self.likelyagnbptagn]
        self.lum_val_filtagn_xrayagn = self.lum[self.filt][agn][self.validagn][self.likelyagnbptagn]
        self.sfr_mass_val_filtagn_xrayagn = self.sfr_mass[self.filt][agn][self.validagn][self.likelyagnbptagn]
        self.sfr_val_filtagn_xrayagn = self.sfr[self.filt][agn][self.validagn][self.likelyagnbptagn]
        self.mass_filtagn_xrayagn = self.mass[self.filt][agn][self.validagn][self.likelyagnbptagn]
        self.lxsfr_val_filtagn_xrayagn = self.lxsfr[self.filt][agn][self.validagn][self.likelyagnbptagn]
        #valid bpt hii
        self.lum_mass_val_filtnoagn_xrayagn =  self.lum_mass[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.lum_val_filtnoagn_xrayagn = self.lum[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.sfr_mass_val_filtnoagn_xrayagn = self.sfr_mass[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.sfr_val_filtnoagn_xrayagn = self.sfr[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.mass_filtnoagn_xrayagn = self.mass[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]
        self.lxsfr_val_filtnoagn_xrayagn = self.lxsfr[self.filt][nonagn][self.validnoagn][self.likelyagnbpthii]