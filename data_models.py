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

from sklearn.linear_model import LinearRegression


import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coords
import astropy.cosmology as apc
from ast_utils import oiii_oii_to_U, nii_oii_to_oh,nii_oii_to_oh_ke02,getlumfromflux,halptofibsfr_corr,get_extinction, correct_av,get_av_bd_a21,get_deltassfr,samircosmo,redden, getfluxfromlum
cosmo = apc.Planck15


from chroptiks.plotting_utils import hist1d, hist2d, scat, plt
from ast_utils import dustcorrect, extinction, get_bptplus_niigroups, get_bpt1_groups,get_bptplus_groups,get_classifiability, get_thom_dist


def read_data(filename, columns, header=None):
    return pd.read_csv(filename, delim_whitespace=True, header=header, names=columns)


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


def load_gsw_catalog(filename, catfold='./catalogs/'):
    # Define column names based on the provided information
    columns = [
        'ObjID', 'GLXID', 'plate', 'MJD', 'fiber_ID',
        'RA', 'Decl', 'z', '2r', 'mass', 'mass_Error',
        'sfr', 'sfr_error', 'afuv', 'afuv_error',
        'ab', 'ab_error', 'av_gsw', 'av_gsw_error', 'flag_sed',
        'uv_survey', 'flag_uv', 'flag_midir', 'flag_mgs'
    ]    
    # Load data into Pandas DataFrames
    gsw_df = read_data( filename, columns)
    gsw_df.reset_index(drop=True, inplace=True)

    # Additional data to be concatenated
    print(gsw_df)
    if 'M' in filename:
        additional_data_files = [
            (catfold+'sigma1_mis.dat', [2], 'sigma1_m'),
            (catfold+'envir_nyu_mis.dat', [0], 'env_nyu_m'),
            (catfold+'baldry_mis.dat', [4], 'env_bald_m'),
            (catfold+'irexcess_mis.dat', [0], 'irx_m'),
            (catfold+'simard_ellip_mis.dat', [1], 'axisrat')
        ]

        for file, cols, new_col_name in additional_data_files:
            additional_df = read_data( file, cols)
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
                exploded_columns = []
                columns_to_remove = []
                for colname in self.data.colnames:
                    column = self.data[colname]
                    if column.ndim > 1:
                        for i in range(column.shape[1]):
                            new_colname = f"{colname}_{i}"
                            new_column = column[:, i]                    
                            exploded_columns.append((new_colname, new_column))
                        columns_to_remove.append(colname)
                for colname, coldata in exploded_columns:
                    self.data[colname] = coldata
                for colname in columns_to_remove:
                    self.data.remove_column(colname)
                
                self.data = self.data.to_pandas()
            elif 'tsv' in self.fname:
                self.data = pd.read_csv(self.fname, delimiter='\t')
            elif 'csv' in self.fname:
                self.data = pd.read_csv(self.fname)   
            elif 'GSW' in self.fname:
                self.data = load_gsw_catalog(self.fname)
        elif dataframe is not None:
            self.data = dataframe          
        print(self.fname, self.data.columns)
        
        
catfold='catalogs/'
import numpy as np
print('loading GSW')


import pandas as pd 

class XMM3obs(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['obsids'] = self.data['ObsID']  # Assuming '1' is the column name for obsids in AstroTablePD_obs
        self.data['tpn'] = self.data['t_PN']
        self.data['tmos1'] = self.data['t_M1']
        self.data['tmos2'] = self.data['t_M2']

class XMM4obs(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)
        self.data['obsids'] = self.data['ObsID'].astype(np.int64)
        self.data['tpn'] = self.data['t.PN']
        self.data['tmos1'] = self.data['t.M1']
        self.data['tmos2'] = self.data['t.M2']
        exp = np.vstack([self.data['tpn'], self.data['tmos1'], self.data['tmos2']])
        self.data['texps'] = np.max(exp, axis=0)
def get_texp(XR, XRObs):
    # Convert the obsids to strings and set them as index
    XRObs.data['obsids_str'] = XRObs.data['obsids'].astype(str)
    XRObs.data.set_index('obsids_str', inplace=True)

    # Create a new column with the transformed 'Source' values
    XR.data['source_obs'] = XR.data['Source'].apply(lambda x: str(np.int64(str(x)[2:11])))

    # Perform a join operation to get the relevant 'tpn', 'tmos1', 'tmos2' for each 'source_obs'
    merged_df = XR.data.join(XRObs.data[['tpn', 'tmos1', 'tmos2']], how='left', on='source_obs')

    # Calculate the maximum along the specified columns, and handle NaNs
    texps = merged_df[['tpn', 'tmos1', 'tmos2']].apply(np.nanmax, axis=1).values

    return texps

class XMM(AstroTablePD):
    def __init__(self, filename, XRObs):
        super().__init__(filename)
        self.XRObs = XRObs
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
        if get_texp:
            try:
                self.data['texp'] = pd.read_csv(catfold+filename.split('/')[-1].split('.')[0]+'texps.csv')['texp']
            except:        
                self.data['texp'] = get_texp(self, self.XRObs)
                self.data[['texp']].to_csv(catfold+filename.split('/')[-1].split('.')[0]+'texps.csv', index=False)
        else:
            self.data['texp']=None
                
        
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
        self.data['allmjds']: self.data['SPEC_MJD']
        self.data['allfiberids']: self.data['SPEC_FIBERID']
        self.data['nvss_flux']: self.data['NVSS_FLUX']
        self.data['first_flux']: self.data['FIRST_FINT']
        self.data['wenss_flux']: self.data['WENSS_FLUX']
        self.data['vlss_flux']: self.data['VLSS_FLUX']



class Gal_Info(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)        
        self.data['allplateids']= self.data['PLATEID']
        self.data['allmjds']= self.data['MJD']
        self.data['allfiberids']: self.data['FIBERID']
        self.data['all_spectype']: self.data['SPECTROTYPE']
    
class Gal_Line(AstroTablePD):
    def __init__(self, 
                 filename, filt=None,
                 weaklines=None, 
                 empirdust=None,
                 lines =  {'H_ALPHA': 4861.0, 
                 'OI_6300':6100.0, 
                 'OII_3726':3727.0, 
                 'NII_6584':6583.0,
                 'NII_6548':6548.0, 
                 'OIII_5007':5007.0, 
                 'H_BETA':4861.0, 
                 'H_GAMMA':4000.0, 
                 'H_DELTA':4000.0, 
                 'SII_6717':6717.0, 
                 'SII_6731':6731.0, 
                 'NEIII_3869':3700.0,
                 'OIII_4363':4363.0,
                 'OIII_4959':4959.0,
                 'SII':6724.0
                 
                 },
                 line_sncut=2
                 ):
        
        super().__init__(filename) 
        if filt is not None:
            self.data = self.data[filt]
        all_suffixes= ['CONT', 'CONT_ERR', 
                       'REQW', 'REQW_ERR',
                       'EQW', 'EQW_ERR',
                       'SEQW', 'SEQW_ERR', 
                       'FLUX','FLUX_ERR', 
                       'INST_RES', 'CHISQ']
        #maybe use the suffixes and lines to create some kind of multi-indexed data frame for looking at things on a line by line bases?
        self.lines =lines
        self.data['av_bd_sf'] = get_extinction(self.data['H_ALPHA_FLUX'], self.data['H_BETA_FLUX'])
        self.data['av_bd_agn'] = get_extinction(self.data['H_ALPHA_FLUX'], self.data['H_BETA_FLUX'], dec_rat=3.1)
        self.data['SII_FLUX']    = self.data['SII_6717_FLUX'] +self.data['SII_6731_FLUX']
        self.data['SII_FLUX_ERR'] = np.sqrt(self.data['SII_6717_FLUX']**2+self.data['SII_6731_FLUX']**2)
        self.data['SII_FLUX_SN']=self.data['SII_FLUX']/self.data['SII_FLUX_ERR']
        self.data['TBT_X'] = self.data['NEIII_3869_FLUX']/self.data['OII_3726_FLUX']
        self.data['yvals_bpt'] =self.data['OIII_5007_FLUX']/self.data['H_BETA_FLUX']
        self.data['xvals1_bpt'] =self.data['NII_6584_FLUX']/self.data['H_ALPHA_FLUX']
        self.data['xvals2_bpt'] =self.data['SII_FLUX']/self.data['H_ALPHA_FLUX']
        self.data['xvals3_bpt'] =self.data['OI_6300_FLUX']/self.data['H_ALPHA_FLUX']

        self.data['n2ha'] = np.log10(self.data['xvals1_bpt'])
        self.data['s2ha'] = np.log10(self.data['xvals2_bpt'])
        self.data['o1ha'] = np.log10(self.data['xvals3_bpt'])
        self.data['o3hb'] = np.log10(self.data['yvals_bpt'])
        self.data['thom_dist'] = get_thom_dist(self.data['n2ha'], 
                                                  self.data['o3hb']) 
        self.data['ji_p1'] =   (self.data['n2ha']*0.63+self.data['s2ha']*0.51+self.data['o3hb']*0.59)
        self.data['ji_p2'] =   (-self.data['n2ha']*0.63+self.data['s2ha']*0.78)
        self.data['ji_p3'] =   (-self.data['n2ha']*0.46-self.data['s2ha']*0.37+0.81*self.data['o3hb'])        
        for line in self.lines:
            self.data[line+'_FLUX_SN'] = self.data[line+'_FLUX']/self.data[line+'_FLUX_ERR']    
            self.data[line + '_SN_PASS'] = self.data[line+'_FLUX_SN']>line_sncut
            self.data[line+'_FLUX_corr_a19'] = dustcorrect(self.data[line+'_FLUX'], self.data['av_bd_agn'],self.lines[line])
            self.data[line+'_FLUX_ERR_corr_a19'] = dustcorrect(self.data[line+'_FLUX_ERR'], self.data['av_bd_agn'],self.lines[line])


    #        self.data[line+'_FLUX_corr_a21'] = dustcorrect(self.data[line+'_FLUX'], elf.data['av_bd_agn'],self.lines[line])
        
        
            
def add_dust_corrected_fluxes_by_model(data, lines, modelfn=get_av_bd_a21, av_col = 'av_bd_agn', model='a21', **kwargs):
    if model =='a21':
        
        data['av_'+model] = modelfn(data, av_col, data['H_BETA_FLUX_SN'], 'av_gsw', **kwargs)
    elif model =='a19':
        data['av_'+model] = modelfn(data['H_ALPHA_FLUX'], data['H_BETA_FLUX'], **kwargs)
    
    for line in list(lines.keys()):
        data[line+'_FLUX_corr_'+model] = dustcorrect(data[line+'_FLUX'], data["av_" +model ],lines[line])
        data[line+'_FLUX_ERR_corr_'+model] = dustcorrect(data[line+'_FLUX'], data["av_" +model ],lines[line])

    return data
def get_dust_correction_quantities(data, model='a19', z='Z'):
    data['oiii_oii'] = np.log10(data['OIII_5007_FLUX_corr_'+model]/data['OII_3726_FLUX_corr_'+model])
    data['oiii_oi'] = np.log10(data['OIII_5007_FLUX_corr_'+model]/data['OI_6300_FLUX_corr_'+model])
    data['oiii_nii'] = np.log10(data['OIII_5007_FLUX_corr_'+model]/data['NII_6584_FLUX_corr_'+model])

    data['U'] =oiii_oii_to_U(data['oiii_oii'] )
    data['nii_oii'] =np.log10(data['NII_6584_FLUX_corr_'+model]/data['OII_3726_FLUX_corr_'+model])
    data['log_oh'] = nii_oii_to_oh(data['NII_6584_FLUX_corr_'+model], data['OII_3726_FLUX_corr_'+model])  
    
    data['log_oh_ke02'] = nii_oii_to_oh_ke02(data['NII_6584_FLUX_corr_'+model], data['OII_3726_FLUX_corr_'+model])             
    data['sii_ratio'] =(data['SII_6717_FLUX_corr_'+model]/data['SII_6717_FLUX_corr_'+model])
    data['sii_oii'] =np.log10(data['SII_FLUX_corr_'+model]/data['OII_3726_FLUX_corr_'+model])
    data['oi_sii'] =np.log10(data['OI_6300_FLUX_corr_'+model]/data['SII_FLUX_corr_'+model])
    
    data['oiiilum'] = np.log10(getlumfromflux(data['OIII_5007_FLUX_corr_'+model],data[z]))
    #data['edd_ratio'] = data['oiiilum']+np.log10(600)-data['edd_lum']

    data['oilum'] = np.log10(getlumfromflux(data['OI_6300_FLUX_corr_'+model],data[z]))
    data['oiilum'] = np.log10(getlumfromflux(data['OII_3726_FLUX_corr_'+model],data[z]))
    data['niilum'] = np.log10(getlumfromflux(data['NII_6584_FLUX_corr_'+model],data[z]))
    data['siilum'] = np.log10(getlumfromflux(data['SII_FLUX_corr_'+model],data[z]))

    #data['oiiilum_sfsub_samir'] = np.log10(getlumfromflux(data['oiiiflux_corr_sf_sub_samir'],data[z]))

    data['oiiilum_up'] = np.log10(getlumfromflux(data['OIII_5007_FLUX_corr_'+model]+data['OIII_5007_FLUX_ERR_corr_'+model],data[z]))
    data['oiiilum_up2'] = np.log10(getlumfromflux(data['OIII_5007_FLUX_corr_'+model]+2*data['OIII_5007_FLUX_ERR_corr_'+model],data[z]))

    data['oiiilum_down'] = np.log10(getlumfromflux(data['OIII_5007_FLUX_corr_'+model]-data['OIII_5007_FLUX_ERR_corr_'+model],data[z]))
    data['e_oiiilum_down'] = data['oiiilum']-data['oiiilum_down']
    data['e_oiiilum_up'] = data['oiiilum_up']- data['oiiilum']

    data['nlr_rad_from_lo3'] = data['oiiilum']*(0.42)-13.97
    
    data['fibsize'] = np.array(cosmo.angular_diameter_distance(data[z]/206265.).value)*1000/2. #in kpc
    data['nlr_fib_ratio'] = data['nlr_rad_from_lo3']/data['fibsize']
    #data['edd_par'] = data['oiiilum']-data['mbh']

    data['halplum'] = np.log10(getlumfromflux(data['H_ALPHA_FLUX_corr_'+model], data[z]))

    data['halpfibsfr'] = halptofibsfr_corr(10**data['halplum'])
    #data['halpfibsfr_uncorr'] = halptofibsfr_corr(10**data['halplum_uncorr'])  
    return data
def get_line_filters(data, sncut=2):
        
    data['bpt_sn_filt_bool'] = (data['H_ALPHA_FLUX_SN']>2) & (data['H_BETA_FLUX_SN'] > sncut) & (data['OIII_5007_FLUX_SN'] > sncut) & (data['NII_6584_FLUX_SN'] > sncut)
    data['high_sn_o3'] =data['OIII_5007_FLUX_SN'] > sncut        
    data['not_bpt_sn_filt_bool']  = np.logical_not(data['bpt_sn_filt_bool'])
    data['halp_nii_filt_bool'] = ( (data['H_ALPHA_FLUX_SN'] > sncut) & (data['NII_6584_FLUX_SN']> sncut) &( (data['OIII_5007_FLUX_SN']<=sncut) | (data['H_BETA_FLUX_SN']<=sncut) ) )        
    data['neither_filt_bool'] = np.logical_not( ( (data['bpt_sn_filt_bool']) | (data['halp_nii_filt_bool']) ))#neither classifiable by BPT, or just by NII        
    data['vo87_1_filt_bool'] = (data['SII_FLUX_SN']>sncut) &(data['bpt_sn_filt_bool'])
    data['vo87_2_filt_bool'] =(data['OI_6300_FLUX_SN']>sncut) &(data['bpt_sn_filt_bool'])
    data['high_sn_all7 ']= (data['bpt_sn_filt_bool']) &( data['OII_3726_FLUX_SN']>sncut) &(data['OI_6300_FLUX_SN']>2) &(data['SII_FLUX_SN']>sncut)

    return data
def apply_line_filter(data, subset = None, subset_name = ''):
    if subset is not None:
        data = data[subset]
    
    bpt_EL_df = data.iloc[data['bpt_sn_filt_bool']]
    
    alllines_bpt_EL_df = data.iloc['high_sn_all7']
    
    vo87_1_EL_df = data['vo87_1_filt_bool']
    vo87_2_EL_df = data['vo87_2_filt_bool']
    bptgroups, bptsf, bptagn = get_bpt1_groups( np.log10(bpt_EL_df['xvals1_bpt']), np.log10(bpt_EL_df['yvals_bpt'] ) )
    bptplsugroups, bptplssf, bptplsagn = get_bptplus_groups(np.log10(bpt_EL_df['xvals1_bpt']), np.log10(bpt_EL_df['yvals_bpt']))
            
    bpt_sf_df = bpt_EL_df.iloc[bptsf]

    high_sn_all7_sf = np.where(( bpt_sf_df.oiiflux_sn>2) &(bpt_sf_df.oiflux_sn>2) &(bpt_sf_df.siiflux_sn>2))

    alllines_bpt_sf_df = bpt_sf_df.iloc[high_sn_all7_sf]
    bpt_agn_df = bpt_EL_df.iloc[bptagn]
    bptplus_sf_df = bpt_EL_df.iloc[bptplssf]
    bptplus_agn_df = bpt_EL_df.iloc[bptplsagn]
    high_sn_all7_bptplussf = np.where(( bptplus_sf_df.oiiflux_sn>2) &(bptplus_sf_df.oiflux_sn>2) &(bptplus_sf_df.siiflux_sn>2))
    alllines_bptplus_sf_df = bptplus_sf_df.iloc[high_sn_all7_bptplussf]
                                                                    
    plus_EL_df = data.iloc[halp_nii_filt]
    
    return

    #self.plus_EL_gsw_df['bptplusniigroups'] = groups_bptplusnii
    '''
    self.bptplusnii_sf_df = self.plus_EL_gsw_df.iloc[bptplsnii_sf].copy()
    self.bptplusnii_agn_df = self.plus_EL_gsw_df.iloc[bptplsnii_agn].copy()        
    self.neither_EL_gsw_df = self.EL_gsw_df.iloc[self.neither_filt].copy()
    self.allnonagn_df=pd.concat([self.bptplus_sf_df,self.bptplusnii_sf_df, self.neither_EL_gsw_df],join='outer')
    '''


class Gal_Indx(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)                    
        self.data['d4000'] = self.data['D4000_N']
        self.data['hdelta_lick'] = self.data['LICK_HD_A']
        self.data['tauv_cont'] = self.data['TAUV_CONT']
        #self.data['vdisp'] = self.data['V_DISP']            
class Gal_Fib(AstroTablePD):
    def __init__(self, filename, output):
        '''
        '''
        super().__init__(filename)        
        self.data[output] =  self.data['AVG']




class SDSSObj:
    def __init__(self, galfiboh_file, galinfo_file, galline_file, galindx_file, fibmass_file, fibsfr_file, fibssfr_file):
        pass

class Xraysfr:
    def __init__(self, xraylums, gswcat, filt, agn, nonagn, typ):
        '''
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
        '''        
        self.typ = typ
        self.filt = filt

        # Assuming gswcat.gsw_df is a DataFrame that contains the columns 'mass', 'z', 'ra', 'dec', 'sfr'
        self.df = gswcat.gsw_df.copy()
        self.df['lum'] = xraylums
        self.df['lum_mass'] = xraylums - self.df['mass']
        self.df['sfr_mass'] = self.df['sfr'] - self.df['mass']
        self.df['lxsfr'] = np.log10(xrayranallidict[typ] * 10 ** self.df['sfr'])
        
        # Filter rows based on filt, agn, nonagn
        self.df['filt'] = self.filt
        self.df['agn'] = agn
        self.df['nonagn'] = nonagn
        
        # Create filters
        self.df['valid'] = self.df['lum'] > 0
        self.df['likelyagn_xr'] = (self.df['lxsfr'] < self.df['lum'] - 0.6) & self.df['valid']
        self.df['likelysf'] = (abs(self.df['lxsfr'] - self.df['lum']) < 0.6) & self.df['valid']

        # Using these filters, you can select rows that satisfy these conditions whenever you need them.
        # For example: self.df[self.df['likelyagn_xr']]

    # If you need to access filtered data frequently, you can write methods that return the filtered data.
    def get_valid_data(self):
        return self.df[self.df['valid']]

    def get_likelyagn_data(self):
        return self.df[self.df['likelyagn_xr']]
        
    # You can add more methods to manipulate or analyze the data as needed.
        
class GSWCat:    
    def __init__(self, 
                 goodinds, 
                 gsw_table, 
                 sedflag=0):
        self.inds = goodinds
        self.sedfilt = np.where(gsw_table['flag_sed'][self.inds]==sedflag)[0]
        self.gsw_df = gsw_table.iloc[self.inds].iloc[self.sedfilt]
        


        self.gsw_table['uv_col'] = self.gsw_df['fuv']-self.gsw_df['nuv']
        bad_uv = np.where((self.gsw_df['nuv']==-99) |(self.gsw_df['nuv']==-999) |(self.gsw_df['fuv']==-99) |(self.gsw_df['nuv']==-999) )[0]
        self.gsw_df['uv_col'][bad_uv] = np.nan        
 
        self.gsw_df['ssfr'] = self.gsw_df['sfr'] - self.gsw_df['mass']
        self.gsw_df['delta_ssfr'] = get_deltassfr(self.gsw_df['mass'], self.gsw_df['ssfr'])
        
        self.gsw_df['irx'][np.where(self.gsw_df['ssfr']<-11)] = np.nan
        self.gsw_df['irx'][np.where(self.gsw_df['irx']==-99)] = np.nan
        self.gsw_df['axisrat'][np.where(self.gsw_df['axisrat']==100)] = np.nan        
        self.data['dmpc_samir']=np.array(np.log10(samircosmo.luminosity_distance(self.data['z']).value))


class GSWCatmatch_XMM:
    def __init__(self, gsw_df):
        self.gsw_df = gsw_df
        self.gsw_df['softlums'] =getlumfromflux(self.gsw_df['softflux'], self.gsw_df['z'])
        self.gsw_df['hardlums'] =getlumfromflux(self.gsw_df['hardflux'], self.gsw_df['z'])
        self.gsw_df['fulllums'] = getlumfromflux(self.gsw_df['fullflux'], self.gsw_df['z'])
        
        self.gsw_df['efulllums_up'] =  getlumfromflux(self.gsw_df['fullflux']+self.gsw_df['efullflux'],self.gsw_df['z'])
        self.gsw_df['efulllums_down'] = getlumfromflux(self.gsw_df['fullflux']-self.gsw_df['efullflux'],self.gsw_df['z'])
        
        self.gsw_df['softlumsrf'] = np.log10(self.gsw_df.softlums*(1+self.gsw_df['z'])**(1.7-2))
        self.gsw_df['hardlumsrf'] = np.log10(self.gsw_df.hardlums*(1+self.gsw_df['z'])**(1.7-2))
        self.gsw_df['fulllumsrf'] = np.log10(self.gsw_df.fulllums*(1+self.gsw_df['z'])**(1.7-2))
        self.gsw_df['efulllumsrf_down'] = np.log10(self.gsw_df.efulllums_down*(1+self.gsw_df['z'])**(1.7-2))
        self.gsw_df['efulllumsrf_up'] = np.log10(self.gsw_df.efulllums_up*(1+self.gsw_df['z'])**(1.7-2))        
        
        self.gsw_df['ehardlums_up'] =   getlumfromflux(self.gsw_df['hardflux']+self.gsw_df['ehardflux'],self.gsw_df['z'])
        self.gsw_df['ehardlums_down'] = getlumfromflux(self.gsw_df['hardflux']-self.gsw_df['ehardflux'],self.gsw_df['z'])
        self.gsw_df['ehardlumsrf_down'] = np.log10(self.gsw_df.ehardlums_down*(1+self.gsw_df['z'])**(1.7-2))
        self.gsw_df['ehardlumsrf_up'] = np.log10(self.gsw_df.ehardlums_up*(1+self.gsw_df['z'])**(1.7-2))                
        #self.data['edd_par_xr'] = self.data['hard_xraylum']-self.data['edd_lum']

        self.gsw_df['lo3_pred_fromlx'] = (self.gsw_df['hardlumsrf']+7.55)/(1.22)
        self.gsw_df['nlr_rad_from_lo3_pred_fromlx'] = self.gsw_df['lo3_pred_fromlx']*(0.42)-13.97
        self.gsw_df['nlr_fib_ratio_pred_fromlx'] = 10**self.gsw_df['nlr_rad_from_lo3_pred_fromlx']/1000/self.gsw_df['fibsize']
                    
        self.gsw_df['lo3_offset'] = -(self.gsw_df['lo3_pred_fromlx']-self.gsw_df['oiiilum']    )  
        self.gsw_df['lo3_offset_up'] = -(self.gsw_df['lo3_pred_fromlx']-self.gsw_df['oiiilum_up']    )  
        self.gsw_df['lo3_offset_up2'] = -(self.gsw_df['lo3_pred_fromlx']-self.gsw_df['oiiilum_up2']    )  
            
        self.gsw_df['fo3_pred_fromlx'] = redden(getfluxfromlum(10**self.gsw_df['lo3_pred_fromlx'], self.gsw_df['z']),
                                                         self.gsw_df['corrected_presub_av'], 5007.0)
        self.gsw_df['lo3_minus_pred_fromlx'] = (self.gsw_df['hardlumsrf']+7.55)/(1.22)-0.587 #subtracting dispersion?
        self.gsw_df['fo3_minus_pred_fromlx'] = redden(getfluxfromlum(10**self.gsw_df['lo3_minus_pred_fromlx'], self.gsw_df['z']),
                                                               self.gsw_df['corrected_presub_av'], 5007.0 )            
        self.gsw_df['fo3_dev'] = (self.gsw_df['fo3_pred_fromlx']-self.gsw_df['oiiiflux'])/(self.gsw_df['fo3_pred_fromlx']-self.gsw_df['fo3_minus_pred_fromlx'])
            
        self.gsw_df['full_lxsfr'] = np.log10(xrayranallidict['full']*10**(self.gsw_df['sfr']))
        self.gsw_df['hard_lxsfr'] = np.log10(xrayranallidict['hard']*10**(self.gsw_df['sfr']))
        self.gsw_df['xray_agn_status'] = self.gsw_df['fulllumsrf']-0.6 > self.gsw_df['full_lxsfr']
        self.gsw_df['xray_excess'] = self.gsw_df['fulllumsrf']- self.gsw_df['full_lxsfr']
        self.gsw_df['hardlx_gas'] = np.log10(7.3e39*10**(self.gsw_df['sfr'])) #mineo+2012
        self.gsw_df['hardlx_xrb'] = np.log10( (10**(29.37)*(10**self.gsw_df['mass'])*(1+self.gsw_df['z'])**2.03 )+
                                                              (10**self.gsw_df['sfr'])*(10**39.28)*(1+self.gsw_df['z'])**1.31)
        self.gsw_df['softlx_xrb'] = np.log10( (10**(29.04)*(10**self.gsw_df['mass'])*(1+self.gsw_df['z'])**3.78 )+
                                                              (10**self.gsw_df['sfr'])*(10**39.28)*(1+self.gsw_df['z'])**0.99)
        self.gsw_df['fulllx_xrb'] = np.log10(10**self.gsw_df['softlx_xrb']+10**self.gsw_df['hardlx_xrb'])
        self.gsw_df['hardlx_xrb_local'] = np.log10( 10**(28.96)*(10**self.gsw_df['mass'])+
                                                              10**self.gsw_df['sfr']*(10**39.21))            
        self.gsw_df['hardlx_sfr'] = np.log10(xrayranallidict['hard']*10**(self.gsw_df['sfr']))
        self.gsw_df['softlx_sfr'] = np.log10(xrayranallidict['soft']*10**(self.gsw_df['sfr']))
        self.gsw_df['full_lxagn'] = np.log10(10**self.gsw_df['fulllumsrf']-10**self.gsw_df['full_lxsfr'])
        self.gsw_df['hard_lxagn'] = np.log10(10**self.gsw_df['hardlumsrf']-10**self.gsw_df['hardlx_sfr'])
        self.gsw_df['soft_lxagn'] = np.log10(10**self.gsw_df['softlumsrf']-10**self.gsw_df['softlx_sfr'])
        
        
        
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
        self.hr1 = hr1[self.sedfilt]
        self.hr2 = hr2[self.sedfilt]
        self.hr3 = hr3[self.sedfilt]
        self.hr4 = hr4[self.sedfilt]
        
        self.xrayflag = xrayflag[self.sedfilt]
        self.gsw_df['fulllumsrf'] = np.log10(self.gsw_df.fulllums*(1+self.z)**(1.7-2))
        self.efulllumsrf_down = np.log10(self.efulllums_down*(1+self.z)**(1.7-2))
        self.efulllumsrf_up = np.log10(self.efulllums_up*(1+self.z)**(1.7-2))
        self.ext = ext[self.sedfilt]
              
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
        
        
        self.EL_dict['massfrac'] = 10**(sdss.all_fibmass)/10**(sdss.all_sdss_avgmasses)
