import matplotlib
import matplotlib.pyplot as plt
import astropy.cosmology as apc
cosmo = apc.Planck15
from loaddata_sdss_xr import *

import astropy.coordinates as coords
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata
from astropy import units as u
#from sklearn import linear_model
import numpy as np

from scipy.ndimage.filters import gaussian_filter


arcsec = 1/3600. #degrees



from demarcations import *
from Gdiffs import *
from Fits_set import *
from setops import *
from matchspec import *
from loaddata_m2 import *
from ast_func import *
from catmatching_xr import *
from ELObj import *
from XMM3_obj import *
from xraysfr_obj import *
from gsw_3xmm_match import *
catfold='catalogs/'
plt.rc('font', family='serif')





#commid, ind1_m2, ind2_m2phot = commonpts1d(m2[0], m2_photcatids)
m2_photrfluxmatched = m1_modelrflux[ind2_m1phot]
posflux = np.where(m2_photrfluxmatched >0 )[0]
#np.savetxt(catfold+'photmatchinginds.txt', ind2_m2phot,fmt='%s')
def get_massfrac(fibmass, totmass):
    '''
    get the mass fraction of the fibers
    '''
    val_masses = np.where((fibmass > 6) & ( fibmass < 13)  &(totmass > 6) &(totmass < 13))[0]
    inval_masses =[(fibmass < 6) | (fibmass > 13) | (totmass < 6) | (totmass > 13)]
    mass_frac = np.ones_like(fibmass)
    mass_frac[val_masses] = 10**(fibmass[val_masses])/(10**(totmass[val_masses]))
    mass_frac[inval_masses] *= -99
    return mass_frac, val_masses
all_sdss_massfrac, val_massinds  = get_massfrac(sdssobj.all_fibmass, sdssobj.all_sdss_avgmasses)

def get_stat_quant(quant, quantmin=0, quantmax=999):
    quant_val = quant[(quant>quantmin)&(quant<quantmax)]
    mn = np.nanmean(quant_val)
    med = np.nanmedian(quant_val)
    q16 = np.quantile(quant_val,.16)
    q84 = np.quantile(quant_val, .84)
    return mn, med,q16, q84
    

'''
matching the xray catalog to the gswlc
'''

# %% x-ray
print('matching catalogs')
x3 = XMM3obs(xmm3,xmm3obs)
x4 = XMM4obs(xmm4,xmm4obs)

#coordchand = SkyCoord(ra95*u.degree, dec95*u.degree)
'''
actmatch_3xmm_gsw2 = catmatch_act(x3.ra,x3.dec,m2Cat_GSW_3xmm.allra,m2Cat_GSW.alldec, m2_photrfluxmatched, x3.fullflux)
#np.array(match_7), np.array(match_7_dists), np.array(matchxflux), np.array(matchedind), np.array(matchrflux), np.array(matchra_diff), np.array(matchdec_diff)
idm2 = actmatch_3xmm_gsw2[0]
goodm2 = actmatch_3xmm_gsw2[3]
d2d2m = actmatch_3xmm_gsw2[1]
radiff = actmatch_3xmm_gsw2[5]
decdiff = actmatch_3xmm_gsw2[6]

actmatch_4xmm_gsw2 = catmatch_act(x4.ra,x4.dec,m2Cat_GSW.allra,m2Cat_GSW.alldec, m2_photrfluxmatched, x4.fullflux, np.arange(len(x4.ra)))
#np.array(match_7), np.array(match_7_dists), np.array(matchxflux), np.array(matchedind), np.array(matchrflux), np.array(matchra_diff), np.array(matchdec_diff)
idm2x4 = actmatch_4xmm_gsw2[0]
goodm2x4 = actmatch_4xmm_gsw2[3]
d2d2mx4 = actmatch_4xmm_gsw2[1]
radiffx4 = actmatch_4xmm_gsw2[5]
decdiffx4 = actmatch_4xmm_gsw2[6]



m1_thin = combo_query_dict[succ[0]][:0].copy().to_pandas()
m1_med = combo_query_dict[succ[0]][:0].copy().to_pandas()
m1_thick = combo_query_dict[succ[0]][:0].copy().to_pandas()
m2_thin = combo_query_dict[succ[0]][:0].copy().to_pandas()
m2_med = combo_query_dict[succ[0]][:0].copy().to_pandas()
m2_thick = combo_query_dict[succ[0]][:0].copy().to_pandas()
pn_thin = combo_query_dict[succ[0]][:0].copy().to_pandas()
pn_med = combo_query_dict[succ[0]][:0].copy().to_pandas()
pn_thick = combo_query_dict[succ[0]][:0].copy().to_pandas()

m1_thin_inds =[]
m1_med_inds = []
m1_thick_inds = []
m2_thin_inds =[]
m2_med_inds = []
m2_thick_inds = []
pn_thin_inds =[]
pn_med_inds = []
pn_thick_inds = []

table_dicts = {'M1_Thin1': [m1_thin,m1_thin_inds],
               'M1_Medium': [m1_med,m1_med_inds],
               'M1_Thick': [m1_thick, m1_thick_inds],
               'M2_Thin1': [m2_thin, m2_thin_inds],
               'M2_Medium': [m2_med, m2_med_inds],
               'M2_Thick': [m2_thick, m2_thick_inds],
               'PN_Thin1': [pn_thin, pn_thin_inds],
               'PN_Medium': [pn_med, pn_med_inds],
               'PN_Thick': [pn_thick, pn_thick_inds]    
               }
ecfs = {
        'PN':{'Thin1_soft':7.3866,
              'Thin1_hard': 1.086,
              'Thin1_full': 3.3243,
              'Medium_soft': 7.0298,
              'Medium_hard': 1.0992,
              'Medium_full':3.1924,
              'Thick_soft': 5.4091,
              'Thick_hard': 1.056,
              'Thick_full': 2.5928
              },
        'M1':{'Thin1_soft':1.929,
              'Thin1_hard': 0.3832,
              'Thin1_full': 0.929,
              'Medium_soft': 1.8531,
              'Medium_hard': 0.3797,
              'Medium_full': 0.900,
              'Thick_soft': 1.5301,
              'Thick_hard': 0.3672,
              'Thick_full': 0.779
              },
        'M2':{'Thin1_soft':1.9322,
              'Thin1_hard': 0.3919,
              'Thin1_full': 0.9358,
              'Medium_soft': 1.8548,
              'Medium_hard': 0.3881,
              'Medium_full': 0.9358,
              'Thick_soft': 1.5301,
              'Thick_hard': 0.3749,
              'Thick_full': 0.7829
              }
        }
for i in range(len(succ)):
    tab_ = combo_query_dict[succ[i]]    
    if i %100 ==0:
        print(i)
    for filt in filts:
        for inst in insts:
            subtab = tab_[(tab_['instrum'] ==inst)&(tab_['filt'] ==filt)].filled(-9999)
            if len(subtab) != 0:
                subtab_ = subtab.to_pandas()
                table_dicts[inst+'_'+filt][1].append(succ[i])
                table_dicts[inst+'_'+filt][0] = table_dicts[inst+'_'+filt][0].append( subtab_)


xraytable_queries = np.load('combo_queries_xraycovered.npy', allow_pickle=True)
xraytable_queries = dict(xraytable_queries.item())
succ = [] 
failed = []

for key in xraytable_queries.keys():
    if len(xraytable_queries[key]) >0:
        succ.append(key)
    else:
        failed.append(key)
succ = np.array(succ)

combo_table_xr= xraytable_queries[succ[0]][:0].copy().to_pandas()

nan_flx_ul = []
succ_flx_ul = []
for i in range(len(succ)):
    tab_ = xraytable_queries[succ[i]]    
    if i %100 ==0:
        print(i)
    band7_sigs = []
    band8_sigs = []
    band6_sigs = []
    band7_insts = []
    band7_exps = []
    

    for j in range(len(tab_)):
        row_ = tab_[j]
        inst_ = row_['instrum']
        filt_ = row_['filt']
        ecf_7 = ecfs[inst_][filt_+'_hard']
        ecf_8 = ecfs[inst_][filt_+'_full']
        ecf_6 = ecfs[inst_][filt_+'_soft']

        band7_sigs.append(np.float64(row_['band7_ul_sigma2'])/ecf_7)
        band8_sigs.append(np.float64(row_['band8_ul_sigma2'])/ecf_8)
        band6_sigs.append(np.float64(row_['band6_ul_sigma2'])/ecf_6)
        band7_insts.append(row_['instrum'])
        band7_exps.append(np.float64(row_['band7_exposure']))
        
    band7_sigs = np.array(band7_sigs)
    band8_sigs = np.array(band8_sigs)
    band6_sigs = np.array(band6_sigs)
    
    minsig7 = np.where((band7_sigs==np.nanmin(band7_sigs ))&(np.isfinite(band7_sigs)))[0]
    minsig8 = np.where((band8_sigs==np.nanmin(band8_sigs ))&(np.isfinite(band8_sigs)))[0]
    minsig6 = np.where((band6_sigs==np.nanmin(band6_sigs ))&(np.isfinite(band6_sigs)))[0]
    
    
    if len(minsig7)!=0:        
        combo_table_xr=combo_table_xr.append(tab_[minsig7].to_pandas())
        succ_flx_ul.append(i)
    elif len(minsig8)!=0:        
        combo_table_xr=combo_table_xr.append(tab_[minsig8].to_pandas())
        succ_flx_ul.append(i)
    elif len(minsig6)!=0:        
        combo_table_xr=combo_table_xr.append(tab_[minsig6].to_pandas())
        succ_flx_ul.append(i)
    else:
        band7_sigs_est = []
        for k in range(len(band7_exps)):
            if band7_insts[k] == 'PN':
                band7_sigs_est.append(10**(-np.log10(band7_exps[k])*5/7-11 ))
            else:
                band7_sigs_est.append(10**(-np.log10(band7_exps[k])*5/6-41/4 ))
        minsig7_est = np.argmin(band7_sigs_est)
        combo_table_xr=combo_table_xr.append(tab_[[np.argmax(minsig7_est)]].to_pandas())
        nan_flx_ul.append(i)
for key in table_dicts.keys():
    ecf_key = ecfs[key.split('_')[0]]
    ecf_soft = ecf_key[key.split('_')[1]+'_soft']
    ecf_hard = ecf_key[key.split('_')[1]+'_hard']
    ecf_full = ecf_key[key.split('_')[1]+'_full']  
    
band6_flx_1sig =[]
band7_flx_1sig =[]
band8_flx_1sig =[]
band6_flx_2sig =[]
band7_flx_2sig =[]
band7_flx_2sig_est =[]

band8_flx_2sig =[]
band6_flx_ =[]
band7_flx_ =[]
band8_flx_ =[]

for i in range(len(combo_table_xr)):
    row_ = combo_table_xr.iloc[i]
    inst_ = row_['instrum']
    filt_ = row_['filt']
    ecf_7 = ecfs[inst_][filt_+'_hard']
    ecf_8 = ecfs[inst_][filt_+'_full']
    ecf_6 = ecfs[inst_][filt_+'_soft']
    band7_flx_.append(np.float64(row_['band7_src_counts']-row_['band7_bck_counts'])/(np.float64(row_['band7_exposure'])*ecf_7*1e11*np.float64(row_['eef'])))
    band8_flx_.append(np.float64(row_['band8_src_counts']-row_['band8_bck_counts'])/(np.float64(row_['band8_exposure'])*ecf_8*1e11*np.float64(row_['eef'])))
    band6_flx_.append(np.float64(row_['band6_src_counts']-row_['band6_bck_counts'])/(np.float64(row_['band6_exposure'])*ecf_6*1e11*np.float64(row_['eef'])))

    if inst_ == 'PN':
        band7_flx_2sig_est.append(10**(-np.log10(row_['band7_exposure'])*5/7-11 ))
    else:
        band7_flx_2sig_est.append(10**(-np.log10(row_['band7_exposure'])*5/6-41/4 ))
    band7_flx_1sig.append(np.float64(row_['band7_ul_sigma1'])/ecf_7/1e11)
    band8_flx_1sig.append(np.float64(row_['band8_ul_sigma1'])/ecf_8/1e11)
    band6_flx_1sig.append(np.float64(row_['band6_ul_sigma1'])/ecf_6/1e11)
    band7_flx_2sig.append(np.float64(row_['band7_ul_sigma2'])/ecf_7/1e11)
    band8_flx_2sig.append(np.float64(row_['band8_ul_sigma2'])/ecf_8/1e11)
    band6_flx_2sig.append(np.float64(row_['band6_ul_sigma2'])/ecf_6/1e11)

mpa_mags = galinfo.getcol('PLUG_MAG')
mpa_rmag = mpa_mags[:,2]
mpa_rflux = - mpa_rmag    

actmatch_4xmm_mpa = catmatch_act(x4.ra,x4.dec,mpa_ra, mpa_dec, mpa_rflux, x4.fullflux, np.arange(len(x4.ra)))
#np.array(match_7), np.array(match_7_dists), np.array(matchxflux), np.array(matchedind), np.array(matchrflux), np.array(matchra_diff), np.array(matchdec_diff)
idm2x4_mpa = actmatch_4xmm_mpa[0]
goodm2x4_mpa = actmatch_4xmm_mpa[3]
d2d2mx4_mpa = actmatch_4xmm_mpa[1]
radiffx4_mpa = actmatch_4xmm_mpa[5]
decdiffx4_mpa = actmatch_4xmm_mpa[6]

e, a, b = np.intersect1d(srcids_x4_full, x4.sourceids, return_indices=True)
actmatch_4xmm_mpa_res = catmatch_act(x4.ra[b],x4.dec[b],mpa_ra[succ], mpa_dec[succ], mpa_rflux[succ], x4.fullflux[b], np.arange(len(x4.ra[b])))
#np.array(match_7), np.array(match_7_dists), np.array(matchxflux), np.array(matchedind), np.array(matchrflux), np.array(matchra_diff), np.array(matchdec_diff)
idm2x4_mpa_res = actmatch_4xmm_mpa_res[0]
goodm2x4_mpa_res = actmatch_4xmm_mpa_res[3]
d2d2mx4_mpa_res = actmatch_4xmm_mpa_res[1]
radiffx4_mpa_res = actmatch_4xmm_mpa_res[5]
decdiffx4_mpa_res = actmatch_4xmm_mpa_res[6]
            
            
rates = {'PN':xmm4_full.getcol('PN_4_RATE')+ xmm4_full.getcol('PN_5_RATE'),
         'M1':xmm4_full.getcol('M1_4_RATE')+xmm4_full.getcol('M1_5_RATE'),
         'M2': xmm4_full.getcol('M2_4_RATE')+xmm4_full.getcol('M2_5_RATE')}            
cts = {'PN':xmm4_full.getcol('PN_8_CTS'),
         'M1':xmm4_full.getcol('M1_8_CTS') ,
         'M2':xmm4_full.getcol('M2_8_CTS')}          
fluxes = {'PN':xmm4_full.getcol('PN_8_FLUX'),
         'M1':xmm4_full.getcol('M1_8_FLUX') ,
         'M2':xmm4_full.getcol('M2_8_FLUX')}          
  

rates_xmm = []
rates_rap = []

cts_xmm = []
cts_rap = []

flx_rap = []
flx_xmm = []

for i in range(len(idm2x4_mpa_res)):
    cts_rapidxmm = np.float64(combo_queries_xr['band8_src_counts'][idm2x4_mpa_res[i]])    
    rate7_rapidxmm = ((combo_queries_xr['band7_src_counts']-combo_queries_xr['band7_bck_counts'])/combo_queries_xr['band7_exposure']/combo_queries_xr['eef'])[idm2x4_mpa_res[i]]
    rate8_rapidxmm = ((combo_queries_xr['band8_src_counts']-combo_queries_xr['band8_bck_counts'])/combo_queries_xr['band8_exposure']/combo_queries_xr['eef'])[idm2x4_mpa_res[i]]

    inst = combo_queries_xr['instrum'][idm2x4_mpa_res[i]]
    filt = combo_queries_xr['filt'][idm2x4_mpa_res[i]]
    
    flux8_rapidxmm = rate8_rapidxmm/ecfs[inst][filt+'_full']/1e11
    x4_rate =  rates[inst][a[goodm2x4_mpa_res[i]]]
    x4_cts = cts[inst][a[goodm2x4_mpa_res[i]]]
    x4_flx = fluxes[inst][a[goodm2x4_mpa_res[i]]]
    
    rates_rap.append(rate7_rapidxmm)
    cts_rap.append(cts_rapidxmm)
    rates_xmm.append(x4_rate)
    cts_xmm.append(x4_cts)
    flx_xmm.append(x4_flx)
    flx_rap.append(flux8_rapidxmm)
    
rates_xmm = np.array(rates_xmm)
rates_rap = np.array(rates_rap)
cts_xmm = np.array(cts_xmm)
cts_rap = np.array(cts_rap)
flx_xmm = np.array(flx_xmm)
flx_rap = np.array(flx_rap)
    
actmatch_4xmm_mpa_xrcov = catmatch_act(x4.ra,x4.dec,mpa_ra[succ], mpa_dec[succ], mpa_rflux[succ], x4.fullflux, np.arange(len(x4.ra)))
#np.array(match_7), np.array(match_7_dists), np.array(matchxflux), np.array(matchedind), np.array(matchrflux), np.array(matchra_diff), np.array(matchdec_diff)
idm2x4_mpa_xrcov = actmatch_4xmm_mpa_xrcov[0]
goodm2x4_mpa_xrcov = actmatch_4xmm_mpa_xrcov[3]
d2d2mx4_mpa_xrcov = actmatch_4xmm_mpa_xrcov[1]
radiffx4_mpa_xrcov = actmatch_4xmm_mpa_xrcov[5]
decdiffx4_mpa_xrcov = actmatch_4xmm_mpa_xrcov[6]

actmatch_4xmmbig_mpa_xrcov = catmatch_act(rabigxmm,decbigxmm,mpa_ra[succ], mpa_dec[succ], mpa_rflux[succ], fluxbigxmm, np.arange(len(fluxbigxmm)))
idm2x4_big_mpa_xrcov = actmatch_4xmmbig_mpa_xrcov[0]
goodm2x4_big_mpa_xrcov = actmatch_4xmmbig_mpa_xrcov[3]
d2d2mx4_big_mpa_xrcov = actmatch_4xmmbig_mpa_xrcov[1]
radiffx4_big_mpa_xrcov = actmatch_4xmmbig_mpa_xrcov[5]
decdiffx4_big_mpa_xrcov = actmatch_4xmmbig_mpa_xrcov[6]

combo_queries = {}
for i in range(len(combo_matched_gals_mpaxmm4)):
        query = rpi.query_radec([mpa_ra[combo_matched_gals_mpaxmm4[i]]],[mpa_dec[combo_matched_gals_mpaxmm4[i]]], obstype='pointed')
        combo_queries[combo_matched_gals_mpaxmm4[i]] = query
        if i%100 == 0:
            print(query)
            print(i)


dataset = {} 
dataset['PLATEID'] = galinfo.getcol('PLATEID')
dataset['FIBERID'] = galinfo.getcol('FIBERID')
dataset['MJD'] = galinfo.getcol('MJD')

dataset['RA'] = galinfo.getcol('RA')
dataset['DEC'] = galinfo.getcol('DEC')

dataset_df = pd.DataFrame(dataset)

dataset_df['XRAYRA'] = np.nan
dataset_df['XRAYDEC'] = np.nan
dataset_df['hard_exp_time'] = np.nan
dataset_df['hard_band_2sigma_ul_flux'] = np.nan
dataset_df['hard_band_2sigma_ul_flux_est'] = np.nan
dataset_df['hard_band_flux'] = np.nan
dataset_df['mass_gsw'] = np.nan
dataset_df['sfr_gsw'] = np.nan


dataset_df.loc[succ,'hard_exp_time'] = np.array( np.log10(combo_table_xr.band7_exposure))
dataset_df.loc[succ,'hard_band_2sigma_ul_flux'] = band7_flx_2sig
dataset_df.loc[succ,'hard_band_2sigma_ul_flux_est'] = band7_flx_2sig_est
dataset_df.loc[succ,'hard_band_flux'] = band7_flx_
dataset_df.loc[succ,'hard_band_ul_flags'] = np.array(combo_table_xr.band7_flags)

dataset_df.loc[succ,'XRAYRA'] = np.array(combo_table_xr.ra)
dataset_df.loc[succ,'XRAYDEC'] = np.array(combo_table_xr.dec)

dataset_df.loc[mpa_spec_allm2.spec_inds_prac,'sfr_gsw'] = allm2[sfrind][mpa_spec_allm2.make_prac]
dataset_df.loc[mpa_spec_allm2.spec_inds_prac,'mass_gsw'] =   allm2[massind][mpa_spec_allm2.make_prac]

dataset_df['band7_background_cts'] = np.nan
dataset_df['band7_src_cts'] = np.nan
#dataset_df['band7_1sigma_ul_flux'] = np.nan

dataset_df['band7_exposure_time'] = np.nan



dataset_df['band6_1sigma_ul_flux'] = np.nan
dataset_df['band6_2sigma_ul_flux'] = np.nan
dataset_df['band6_exposure_time'] = np.nan
dataset_df['band6_background_cts'] = np.nan
dataset_df['band6_src_cts'] = np.nan

dataset_df['band8_1sigma_ul_flux'] = np.nan
dataset_df['band8_2sigma_ul_flux'] = np.nan
dataset_df['band8_exposure_time'] = np.nan
dataset_df['band8_background_cts'] = np.nan
dataset_df['band8_src_cts'] = np.nan


dataset_df.loc[succ,'band7_1sigma_ul_flux'] = band7_flx_1sig
dataset_df.loc[succ,'band7_2sigma_ul_flux'] = band7_flx_2sig
dataset_df.loc[succ,'band7_background_cts'] = np.array(combo_table_xr['band7_bck_counts'])
dataset_df.loc[succ,'band7_src_cts'] =np.array( combo_table_xr['band7_src_counts'])
dataset_df.loc[succ,'band7_exposure_time'] = np.array(combo_table_xr['band7_exposure'])
dataset_df.loc[succ, 'XRAYRA'] = np.array(combo_table_xr['ra'])
dataset_df.loc[succ, 'XRAYDEC'] = np.array(combo_table_xr['dec'])


dataset_df.loc[succ,'band8_1sigma_ul_flux'] = band8_flx_1sig
dataset_df.loc[succ,'band8_2sigma_ul_flux'] = band8_flx_2sig
dataset_df.loc[succ,'band8_background_cts'] = np.array(combo_table_xr['band8_bck_counts'])
dataset_df.loc[succ,'band8_src_cts'] =np.array( combo_table_xr['band8_src_counts'])
dataset_df.loc[succ,'band8_exposure_time'] = np.array(combo_table_xr['band8_exposure'])


dataset_df.loc[succ,'band6_1sigma_ul_flux'] = band6_flx_1sig
dataset_df.loc[succ,'band6_2sigma_ul_flux'] = band6_flx_2sig
dataset_df.loc[succ,'band6_background_cts'] = np.array(combo_table_xr['band6_bck_counts'])
dataset_df.loc[succ,'band6_src_cts'] = np.array(combo_table_xr['band6_src_counts'])
dataset_df.loc[succ,'band6_exposure_time'] =np.array( combo_table_xr['band6_exposure'])



dataset_df['4xmm_ra'] = np.nan
dataset_df['4xmm_dec'] = np.nan
dataset_df['4xmm_band6_flux'] = np.nan
dataset_df['4xmm_band7_flux'] = np.nan
dataset_df['4xmm_band8_flux'] = np.nan
dataset_df['4xmm_band6_flux_error'] = np.nan
dataset_df['4xmm_band7_flux_error'] = np.nan
dataset_df['4xmm_band8_flux_error'] = np.nan

dataset_df['4xmm_qual_flag'] = np.nan
dataset_df['4xmm_extent'] = np.nan


dataset_df.loc[ idm2x4_mpa,'4xmm_ra'] = x4.ra[goodm2x4_mpa]
dataset_df.loc[ idm2x4_mpa,'4xmm_dec'] = x4.dec[goodm2x4_mpa]
dataset_df.loc[ idm2x4_mpa,'4xmm_extent'] = x4.ext[goodm2x4_mpa]
dataset_df.loc[ idm2x4_mpa,'4xmm_band6_flux'] = (x4.flux1+x4.flux2+x4.flux3)[goodm2x4_mpa]
dataset_df.loc[ idm2x4_mpa,'4xmm_band7_flux'] = (x4.flux4+x4.flux5)[goodm2x4_mpa]
dataset_df.loc[ idm2x4_mpa,'4xmm_band8_flux'] = (x4.flux1+x4.flux2+x4.flux3+x4.flux4+x4.flux5)[goodm2x4_mpa]

dataset_df.loc[ idm2x4_mpa,'4xmm_band6_flux_error'] = np.sqrt(x4.eflux1**2+x4.eflux2**2+x4.eflux3**2)[goodm2x4_mpa]
dataset_df.loc[ idm2x4_mpa,'4xmm_band7_flux_error'] = np.sqrt(x4.eflux4**2+x4.eflux5**2)[goodm2x4_mpa]
dataset_df.loc[ idm2x4_mpa,'4xmm_band8_flux_error'] = np.sqrt(x4.eflux1**2+x4.eflux2**2+x4.eflux3**2+x4.eflux4**2+x4.eflux5**2)[goodm2x4_mpa]
dataset_df.loc[ idm2x4_mpa,'4xmm_qual_flag'] = x4.qualflag[goodm2x4_mpa]


np.savetxt(catfold+'xmm3gswamatch1ind.txt',ida1)
np.savetxt(catfold+'xmm3gswamatch1good.tx4xt',gooda1)
np.savetxt(catfold+'xmm3gswamatchd2d1.txt',d2d1a)

np.savetxt(catfold+'xmm3gswmmatch1ind.txt',idm2)
np.savetxt(catfold+'xmm3gswmmatch1good.txt',goodm2)
np.savetxt(catfold+'xmm3gswmmatchd2d1.txt',d2d1m)

np.savetxt(catfold+'xmm3gswdmatch1ind.txt',idd1)
np.savetxt(catfold+'xmm3gswdmatch1good.txt',goodd1)
np.savetxt(catfold+'xmm3gswdmatchd2d1.txt',d2d1d)





np.savetxt(catfold+'xmm4gswmmatch1ind.txt',idm2x4)
np.savetxt(catfold+'xmm4gswmmatch1good.txt',goodm2x4)
np.savetxt(catfold+'xmm4gswmmatchd2d1.txt',d2d2mx4)

np.savetxt(catfold+'xmm4gswmmatch1deltaranew.txt',radiffx4)
np.savetxt(catfold+'xmm4gswmmatch1deltadecnew.txt',decdiffx4)



'''
idm2 = np.loadtxt(catfold+'xmm3gswmatch1indnew.txt')
goodm2 = np.loadtxt(catfold+'xmm3gswmatch1goodnew.txt')
d2d1m = np.loadtxt(catfold+'xmm3gswmmatchd2d1new.txt')
decdiff = np.loadtxt(catfold+'xmm3gswmatchdeltadecnew.txt')
radiff = np.loadtxt(catfold+'xmm3gswmatchdeltaranew.txt')

idm2x4 = np.loadtxt(catfold+'xmm4gswmmatch1ind.txt')
goodm2x4 = np.loadtxt(catfold+'xmm4gswmmatch1good.txt')
d2d2mx4 = np.loadtxt(catfold+'xmm4gswmmatchd2d1.txt')
decdiffx4 = np.loadtxt(catfold+'xmm4gswmmatch1deltaranew.txt')
radiffx4 = np.loadtxt(catfold+'xmm4gswmmatch1deltadecnew.txt')


x3.idm2 = np.int64(idm2)
x3.good_med = np.int64(goodm2)
x3.medexptimes = x3.get_texp(x3.good_med)

x4.idm2 = np.int64(idm2x4)
x4.good_med = np.int64(goodm2x4)
x4.medexptimes = x4.get_texp(x4.good_med)
x4.sternexptimes = x4.get_texp(match_to_xmm)

#x3.medmatchindsall, x3.medmatchsetidsall = x3.obsmatching(np.int64(np.ones_like(x3.ra)))
#x3.goodobsids = np.array(x3.obsids[np.where(x3.medmatchinds !=0)[0]])
#x4.goodobsids = np.array(x4.obsids[np.where(x4.medmatchinds !=0)[0]])

#np.savetxt(catfold+'sampleobstimes.txt', np.array(x3.goodobsids), fmt='%s')
#x3.medtimes_all,x3.medtimes_allt, x3.obsids_allt = x3.singtimearr(np.int64(np.ones_like(x3.tpn)))
#x4.medtimes_all,x4.medtimes_allt, x4.obsids_allt = x4.singtimearr(np.int64(np.ones_like(x4.tpn)))
#x4.sterntimes_all,x4.sterntimes_allt, x4.sternobsids_allt = x4.singtimearr(x4.sternmatchinds)

x3.logmedtimes_all = np.log10(x3.medexptimes)
x3.medtimes_allfilt = np.where((x3.logmedtimes_all>4.1)&(x3.logmedtimes_all <4.5))
#x3.medcovgobs = x3.obsids[x3.medtimes_allfilt]
#np.savetxt(catfold+'goodobsalltimes.txt',np.array(x3.medcovgobs), fmt='%s')
x4.logmedtimes_all = np.log10(x4.medexptimes)
x4.medtimes_allfilt = np.where((x4.logmedtimes_all>4.1)&(x4.logmedtimes_all <4.5))
x4.medcovgobs = x4.obsids[x4.medtimes_allfilt]


x3.goodra = np.array(x3.ra[x3.good_med])
x3.gooddec = np.array(x3.dec[x3.good_med])
x4.goodra = np.array(x4.ra[x4.good_med])
x4.gooddec = np.array(x4.dec[x4.good_med])

#x3.allxmm3times, x3.allxmm3timesobs = x3.obsmatching(np.int64(np.ones(x3.ra.size)))
#x3.medtimes, x3.medtimesobs, x3.obsids_matches = x3.singtimearr(x3.medmatchinds)
#x3.medtimes, x3.medtimesobs, x3.obsids_matches = x3.singtimearr(x3.medmatchindsobsinds)
#x4.medtimes, x4.medtimesobs, x4.obsids_matches = x4.singtimearr(x4.medmatchinds)

x3.logmedtimes = np.log10(x3.medexptimes)
#x3.logmedtimesobs = np.log10(x3.medtimesobs)
x3.medtimefilt = np.where((x3.logmedtimes > 4.1) & (x3.logmedtimes < 4.5))[0]
#x3.medtimeobsfilt = np.where((x3.logmedtimesobs >4.1) & (x3.logmedtimesobs < 4.5))[0]

x4.logmedtimes = np.log10(x4.medexptimes)
#x4.logmedtimesobs = np.log10(x4.medtimesobs)
x4.medtimefilt = np.where((x4.logmedtimes > 4.1) & (x4.logmedtimes < 4.5))[0]
#x4.medtimeobsfilt = np.where((x4.logmedtimesobs >4.1) & (x4.logmedtimesobs < 4.5))[0]

x3.goodratimefilt = x3.goodra[x3.medtimefilt]
x3.gooddectimefilt = x3.gooddec[x3.medtimefilt]
#np.savetxt(catfold+'goodobstimes.txt',np.array(x3.goodobsids[x3.medtimeobsfilt]), fmt='%s')
#x3.medtimefilt = np.where((x3.logmedtimes > 3.5) & (x3.logmedtimes <3.9))[0] #shorter, s82x-esque exposures
x3.medtimefilt_all = np.where(x3.logmedtimes >0 )[0] #all times


x4.goodratimefilt = x4.goodra[x4.medtimefilt]
x4.gooddectimefilt = x4.gooddec[x4.medtimefilt]
#np.savetxt(catfold+'goodobstimes.txt',np.array(x3.goodobsids[x3.medtimeobsfilt]), fmt='%s')
#x3.medtimefilt = np.where((x3.logmedtimes > 3.5) & (x3.logmedtimes <3.9))[0] #shorter, s82x-esque exposures
x4.medtimefilt_all = np.where(x4.logmedtimes >0 )[0] #all times


x3.softflux_filt = x3.softflux[x3.good_med][x3.medtimefilt]
x3.hardflux_filt = x3.hardflux[x3.good_med][x3.medtimefilt]
x3.hardflux2_filt = x3.hardflux2[x3.good_med][x3.medtimefilt]
x3.fullflux_filt = x3.fullflux[x3.good_med][x3.medtimefilt]
x3.efullflux_filt = x3.efullflux[x3.good_med][x3.medtimefilt]
x3.ehardflux_filt = x3.ehardflux[x3.good_med][x3.medtimefilt]
x3.esoftflux_filt = x3.esoftflux[x3.good_med][x3.medtimefilt]

x3.qualflag_filt = x3.qualflag[x3.good_med][x3.medtimefilt]
x3.ext_filt = x3.ext[x3.good_med][x3.medtimefilt_all]

x3.HR1_filt = x3.HR1[x3.good_med][x3.medtimefilt]
x3.HR2_filt = x3.HR2[x3.good_med][x3.medtimefilt]
x3.HR3_filt = x3.HR3[x3.good_med][x3.medtimefilt]
x3.HR4_filt = x3.HR4[x3.good_med][x3.medtimefilt]



x3.softflux_all = x3.softflux[x3.good_med][x3.medtimefilt_all]
x3.hardflux_all = x3.hardflux[x3.good_med][x3.medtimefilt_all]
x3.hardflux2_all = x3.hardflux2[x3.good_med][x3.medtimefilt_all]
x3.fullflux_all = x3.fullflux[x3.good_med][x3.medtimefilt_all]
x3.efullflux_all = x3.efullflux[x3.good_med][x3.medtimefilt_all]
x3.ehardflux_all = x3.ehardflux[x3.good_med][x3.medtimefilt_all]
x3.esoftflux_all = x3.esoftflux[x3.good_med][x3.medtimefilt_all]

x3.qualflag_all = x3.qualflag[x3.good_med][x3.medtimefilt_all]
x3.ext_all = x3.ext[x3.good_med][x3.medtimefilt_all]

x3.HR1_all = x3.HR1[x3.good_med][x3.medtimefilt_all]
x3.HR2_all = x3.HR2[x3.good_med][x3.medtimefilt_all]
x3.HR3_all = x3.HR3[x3.good_med][x3.medtimefilt_all]
x3.HR4_all = x3.HR4[x3.good_med][x3.medtimefilt_all]


x4.softflux_filt = x4.softflux[x4.good_med][x4.medtimefilt]
x4.hardflux_filt = x4.hardflux[x4.good_med][x4.medtimefilt]
x4.hardflux2_filt = x4.hardflux2[x4.good_med][x4.medtimefilt]
x4.fullflux_filt = x4.fullflux[x4.good_med][x4.medtimefilt]
x4.efullflux_filt = x4.efullflux[x4.good_med][x4.medtimefilt]
x4.ehardflux_filt = x4.ehardflux[x4.good_med][x4.medtimefilt]
x4.esoftflux_filt = x4.esoftflux[x4.good_med][x4.medtimefilt]

x4.qualflag_filt = x4.qualflag[x4.good_med][x4.medtimefilt]
x4.ext_filt = x4.ext[x4.good_med][x4.medtimefilt_all]

x4.HR1_filt = x4.HR1[x4.good_med][x4.medtimefilt]
x4.HR2_filt = x4.HR2[x4.good_med][x4.medtimefilt]
x4.HR3_filt = x4.HR3[x4.good_med][x4.medtimefilt]
x4.HR4_filt = x4.HR4[x4.good_med][x4.medtimefilt]



x4.softflux_all = x4.softflux[x4.good_med][x4.medtimefilt_all]
x4.hardflux_all = x4.hardflux[x4.good_med][x4.medtimefilt_all]
x4.hardflux2_all = x4.hardflux2[x4.good_med][x4.medtimefilt_all]
x4.fullflux_all = x4.fullflux[x4.good_med][x4.medtimefilt_all]
x4.efullflux_all = x4.efullflux[x4.good_med][x4.medtimefilt_all]
x4.ehardflux_all = x4.ehardflux[x4.good_med][x4.medtimefilt_all]
x4.esoftflux_all = x4.esoftflux[x4.good_med][x4.medtimefilt_all]

x4.qualflag_all = x4.qualflag[x4.good_med][x4.medtimefilt_all]
x4.ext_all = x4.ext[x4.good_med][x4.medtimefilt_all]

x4.HR1_all = x4.HR1[x4.good_med][x4.medtimefilt_all]
x4.HR2_all = x4.HR2[x4.good_med][x4.medtimefilt_all]
x4.HR3_all = x4.HR3[x4.good_med][x4.medtimefilt_all]
x4.HR4_all = x4.HR4[x4.good_med][x4.medtimefilt_all]


sternobj_df_spec_xr['fullflux'] = x4.fullflux[match_to_xmm]
sternobj_df_spec_xr['hardflux'] = x4.hardflux[match_to_xmm]

sternobj_df_spec_xr['efullflux'] = x4.efullflux[match_to_xmm]
sternobj_df_spec_xr['fulllum'] = np.log10(getlumfromflux(x4.fullflux[match_to_xmm], sternobj_df_spec_xr['z']))
sternobj_df_spec_xr['hardlum'] = np.log10(getlumfromflux(x4.hardflux[match_to_xmm], sternobj_df_spec_xr['z']))

sternobj_df_spec_xr['fulllumsrf'] = np.log10((getlumfromflux(x4.fullflux[match_to_xmm], sternobj_df_spec_xr['z'])*(1+sternobj_df_spec_xr['z'])**(1.7-2)))
sternobj_df_spec_xr['hardlumsrf'] = np.log10((getlumfromflux(x4.hardflux[match_to_xmm], sternobj_df_spec_xr['z'])*(1+sternobj_df_spec_xr['z'])**(1.7-2)))

sternobj_df_spec_xr['texp'] = x4.sternexptimes
sternobj_df_spec_xr.to_csv(catfold+'sternxr_match.csv')

sternobj_df_spec_xr_rob = sternobj_df_spec_xr.iloc[np.where((sternobj_df_spec_xr.robust =='+')&(sternobj_df_spec_xr['abs'] =='+'))].copy()
liu_obj = {}

liu_obj['z'] = liu_basic.getcol('Z')
liuobj_df = pd.DataFrame(liu_obj)


liuobj_df['lbha'], liuobj_df['logMBH'], liuobj_df['lbhb'] = liu_spec.getcol(['LBHA','MBH','LBHB']) 

liuobj_df['lha'], liuobj_df['e_lha'], liuobj_df['lhb'], liuobj_df['e_lhb']  = liu_spec.getcol(['LNHA','LNHA_ERR', 'LNHB', 'LNHB_ERR']) 

liuobj_df['e_lo3'], liuobj_df['lo3'], liuobj_df['ln2'], liuobj_df['e_ln2']  = liu_spec.getcol(['LOIII5007_ERR','LOIII5007', 'LNII6583', 'LNII6583_ERR']) 
liuobj_df['e_ls2_6716'], liuobj_df['ls2_6716'], liuobj_df['e_ls2_6731'],liuobj_df['ls2_6731'], liuobj_df['lo1'], liuobj_df['e_lo1']  = liu_spec.getcol(['LSII6716_ERR','LSII6716','LSII6731_ERR','LSII6731',  'LOI6300', 'LOI6300_ERR']) 

liuobj_df['fs2_6716'] = getfluxfromlum(10**np.array(liuobj_df['ls2_6716']), np.array(liuobj_df['z']))
liuobj_df['fs2_6731'] = getfluxfromlum(10**np.array(liuobj_df['ls2_6731']), np.array(liuobj_df['z']))
liuobj_df['fs2'] = liuobj_df['fs2_6716']+liuobj_df['fs2_6731']
liuobj_df['ls2'] = np.log10(getlumfromflux(liuobj_df['fs2_6716']+liuobj_df['fs2_6731'], np.array(liuobj_df['z'])))


liuobj_df['fo1'] = getfluxfromlum(10**np.array(liuobj_df['lo1']), np.array(liuobj_df['z']))
liuobj_df['fn2'] = getfluxfromlum(10**np.array(liuobj_df['ln2']), np.array(liuobj_df['z']))


liuobj_df['fo3'] = getfluxfromlum(10**np.array(liuobj_df['lo3']), np.array(liuobj_df['z']))
liuobj_df['fo3_up'] = getfluxfromlum(10**(np.array(liuobj_df['lo3'])+np.array(liuobj_df['e_lo3'])), np.array(liuobj_df['z']))
liuobj_df['fo3_down'] = getfluxfromlum(10**(np.array(liuobj_df['lo3'])-np.array(liuobj_df['e_lo3'])), np.array(liuobj_df['z']))

liuobj_df['e_fo3_up'] = liuobj_df['fo3_up']-liuobj_df['fo3'] 
liuobj_df['e_fo3_down'] = liuobj_df['fo3']-liuobj_df['fo3_down']         

liuobj_df['e_fo3'] = np.mean(np.vstack([liuobj_df['e_fo3_up'], liuobj_df['e_fo3_down']]), axis=0)

liuobj_df['sn_fo3'] = np.array(liuobj_df['fo3'])/np.array(liuobj_df['e_fo3'])

liuobj_df['fha'] = getfluxfromlum(10**np.array(liuobj_df['lha']), np.array(liuobj_df['z']))
liuobj_df['fha_up'] = getfluxfromlum(10**(np.array(liuobj_df['lha'])+np.array(liuobj_df['e_lha'])), np.array(liuobj_df['z']))
liuobj_df['fha_down'] = getfluxfromlum(10**(np.array(liuobj_df['lha'])-np.array(liuobj_df['e_lha'])), np.array(liuobj_df['z']))

liuobj_df['e_fha_up'] = liuobj_df['fha_up']-liuobj_df['fha'] 
liuobj_df['e_fha_down'] = liuobj_df['fha']-liuobj_df['fha_down']         

liuobj_df['e_fha'] = np.mean(np.vstack([liuobj_df['e_fha_up'], liuobj_df['e_fha_down']]), axis=0)

liuobj_df['sn_fha'] = np.array(liuobj_df['fha'])/np.array(liuobj_df['e_fha'])

liuobj_df['fhb'] = getfluxfromlum(10**np.array(liuobj_df['lhb']), np.array(liuobj_df['z']))
liuobj_df['fhb_up'] = getfluxfromlum(10**(np.array(liuobj_df['lhb'])+np.array(liuobj_df['e_lhb'])), np.array(liuobj_df['z']))
liuobj_df['fhb_down'] = getfluxfromlum(10**(np.array(liuobj_df['lhb'])-np.array(liuobj_df['e_lhb'])), np.array(liuobj_df['z']))

liuobj_df['e_fhb_up'] = liuobj_df['fhb_up']-liuobj_df['fhb'] 
liuobj_df['e_fhb_down'] = liuobj_df['fhb']-liuobj_df['fhb_down']         

liuobj_df['e_fhb'] = np.mean(np.vstack([liuobj_df['e_fhb_up'], liuobj_df['e_fhb_down']]), axis=0)

liuobj_df['sn_fhb'] = np.array(liuobj_df['fhb'])/np.array(liuobj_df['e_fhb'])



liuobj_df['av_balm'] = extinction(liuobj_df['fha'], liuobj_df['fhb'], agn=True, zeroed=True)
liuobj_df['fo3_corr'] = dustcorrect(liuobj_df['fo3'], liuobj_df['av_balm'], 5007.0)
liuobj_df['lo3_corr'] = np.log10(getlumfromflux(liuobj_df['fo3_corr'], liuobj_df['z']))
liuobj_df['fha_corr'] = dustcorrect(liuobj_df['fha'], liuobj_df['av_balm'], 6563.0)
liuobj_df['lha_corr'] = np.log10(getlumfromflux(liuobj_df['fha_corr'], liuobj_df['z']))

liuobj_xmm_df = liuobj_df.iloc[match_x_liu].copy()
liuobj_xmm_df['hardflux']=x4.hardflux[match_liu_to_xmm]
liuobj_xmm_df['softflux']=x4.softflux[match_liu_to_xmm]
liuobj_xmm_df['fullflux']=x4.fullflux[match_liu_to_xmm]
liuobj_xmm_df['hardflux_sn']=x4.hardflux_sn[match_liu_to_xmm]
liuobj_xmm_df['softflux_sn']=x4.softflux_sn[match_liu_to_xmm]
liuobj_xmm_df['fullflux_sn']=x4.fullflux_sn[match_liu_to_xmm]

liuobj_xmm_df['fulllums_rf'] =  np.log10(getlumfromflux(liuobj_xmm_df['fullflux'], 
                                                        liuobj_xmm_df['z'])*(1+liuobj_xmm_df['z'])**(1.7-2))
liuobj_xmm_df['softlums_rf'] =  np.log10(getlumfromflux(liuobj_xmm_df['softflux'], 
                                                        liuobj_xmm_df['z'])*(1+liuobj_xmm_df['z'])**(1.7-2))
liuobj_xmm_df['hardlums_rf'] =  np.log10(getlumfromflux(liuobj_xmm_df['hardflux'], 
                                                        liuobj_xmm_df['z'])*(1+liuobj_xmm_df['z'])**(1.7-2))

liuobj_xmm_z07_df = liuobj_xmm_df.iloc[np.where(liuobj_xmm_df.z<=0.07)[0]].copy()

liuobj_xmm_fx_z07_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.07)&(liuobj_xmm_df.fullflux_sn>2))[0]].copy()
liuobj_xmm_hx_z07_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.07)&(liuobj_xmm_df.hardflux_sn>2))[0]].copy()
liuobj_xmm_fx_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.3)&(liuobj_xmm_df.fullflux_sn>2))[0]].copy()
liuobj_xmm_hx_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.3)&(liuobj_xmm_df.hardflux_sn>2))[0]].copy()

liuobj_xmm_fx_o3_dust_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.3)&
                                                       (liuobj_xmm_df.fullflux_sn>2)&
                                                       (liuobj_xmm_df.sn_fo3>3)&
                                                       (liuobj_xmm_df.sn_fhb>2)&
                                                       (liuobj_xmm_df.sn_fha>2)
                                                       )[0]].copy()
liuobj_xmm_hx_o3_dust_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.3)&
                                                       (liuobj_xmm_df.hardflux_sn>2)&
                                                       (liuobj_xmm_df.sn_fo3>3)&
                                                       (liuobj_xmm_df.sn_fhb>2)&
                                                       (liuobj_xmm_df.sn_fha>2))[0]].copy()
liuobj_xmm_fx_o3_dust_z07_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.07)&
                                                       (liuobj_xmm_df.fullflux_sn>2)&
                                                       (liuobj_xmm_df.sn_fo3>3)&
                                                       (liuobj_xmm_df.sn_fhb>2)&
                                                       (liuobj_xmm_df.sn_fha>2)
                                                       )[0]].copy()
liuobj_xmm_hx_o3_dust_z07_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.07)&
                                                       (liuobj_xmm_df.hardflux_sn>2)&
                                                       (liuobj_xmm_df.sn_fo3>3)&
                                                       (liuobj_xmm_df.sn_fhb>2)&
                                                       (liuobj_xmm_df.sn_fha>2))[0]].copy()

liuobj_xmm_fx_o3_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.3)&
                                                       (liuobj_xmm_df.fullflux_sn>2)&
                                                       (liuobj_xmm_df.sn_fo3>3)
                                                       )[0]].copy()
liuobj_xmm_hx_o3_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.3)&
                                                       (liuobj_xmm_df.hardflux_sn>2)&
                                                       (liuobj_xmm_df.sn_fo3>3))[0]].copy()
liuobj_xmm_fx_o3_z07_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.07)&
                                                       (liuobj_xmm_df.fullflux_sn>2)&
                                                       (liuobj_xmm_df.sn_fo3>3)
                                                       )[0]].copy()
liuobj_xmm_hx_o3_z07_df = liuobj_xmm_df.iloc[np.where((liuobj_xmm_df.z<=0.07)&
                                                       (liuobj_xmm_df.hardflux_sn>2)&
                                                       (liuobj_xmm_df.sn_fo3>3))[0]].copy()



csc_cat = CSC(merged_csc)

comm_csc_gsw,  gsw_csc, csc_gsw= np.intersect1d(  m2[0], merged_csc.SDSSDR15, return_indices=True)

csc_cat.gswmatch_inds = gsw_csc
#%% GSW loading

#a1Cat_GSW_3xmm = GSWCatmatch3xmm(x3,x3.ida[x3.good_all][x3.alltimefilt], a1, redshift_a1, alla1)
m2Cat_GSW_qsos = GSWCat( np.arange(len(m2[0])), m2, redshift_m2, allm2, sedflag=1)

m2Cat_GSW = GSWCat( np.arange(len(m2[0])), m2, redshift_m2, allm2)
x2Cat_GSW = GSWCat( np.arange(len(x2[0])), x2, redshift_x2, allx2)


m2Cat_GSW_3xmm = GSWCatmatch3xmm(x3.idm2[x3.medtimefilt], m2, redshift_m2, allm2, x3.qualflag_filt,
                                 x3.fullflux_filt, x3.efullflux_filt, 
                                 x3.hardflux_filt, x3.ehardflux_filt, x3.hardflux2_filt, x3.softflux_filt, x3.esoftflux_filt, x3.ext_filt,
                                 x3.HR1_filt, x3.HR2_filt, x3.HR3_filt, x3.HR4_filt)
m2Cat_GSW_4xmm = GSWCatmatch3xmm(x4.idm2[x4.medtimefilt], m2, redshift_m2, allm2, x4.qualflag, #softflux here should be qualflag
                                 x4.fullflux_filt, x4.efullflux_filt, 
                                 x4.hardflux_filt, x4.ehardflux_filt,x4.hardflux2_filt, x4.softflux_filt,x4.esoftflux_filt, x4.ext_filt, #second softflux should be ext
                                 x4.HR1_filt, x4.HR2_filt, x4.HR3_filt, x4.HR4_filt)

m2Cat_GSW_csc = GSWCatmatch_CSC(csc_cat.gswmatch_inds, m2, redshift_m2, allm2, 
                                 x3.fullflux, #x3.efullflux_filt, 
                                 x3.hardflux, x3.softflux)#, x3.ext_filt,
                                 #x3.HR1_filt, x3.HR2_filt, x3.HR3_filt, x3.HR4_filt)


m2Cat_GSW_3xmm_all = GSWCatmatch3xmm( x3.idm2[x3.medtimefilt_all], m2, 
                                     redshift_m2, allm2, x3.qualflag_all, x3.fullflux_all, 
                                     x3.efullflux_all,x3.hardflux_all, x3.ehardflux_all,
                                     x3.hardflux2_all, x3.softflux_all,x3.esoftflux_all, x3.ext_all,
                                     x3.HR1_all, x3.HR2_all, x3.HR3_all, x3.HR4_all)
m2Cat_GSW_4xmm_all = GSWCatmatch3xmm( x4.idm2[x4.medtimefilt_all], m2, 
                                     redshift_m2, allm2, x4.qualflag_all, x4.fullflux_all, 
                                     x4.efullflux_all,x4.hardflux_all, x4.ehardflux_all,
                                     x4.hardflux2_all, x4.softflux_all,x4.esoftflux_all, x4.ext_all,
                                     x4.HR1_all, x4.HR2_all, x4.HR3_all, x4.HR4_all)

m2Cat_GSW_3xmm_all_qsos = GSWCatmatch3xmm( x3.idm2[x3.medtimefilt_all], m2, 
                                     redshift_m2, allm2, x3.qualflag_all, x3.fullflux_all, 
                                     x3.efullflux_all,x3.hardflux_all, x3.ehardflux_all, 
                                     x3.hardflux2_all, x3.softflux_all,x3.esoftflux_all, x3.ext_all,
                                     x3.HR1_all, x3.HR2_all, x3.HR3_all, x3.HR4_all, sedflag=1)
m2Cat_GSW_4xmm_all_qsos = GSWCatmatch3xmm( x4.idm2[x4.medtimefilt_all], m2, 
                                     redshift_m2, allm2, x4.qualflag_all, x4.fullflux_all, 
                                     x4.efullflux_all,x4.hardflux_all, x4.ehardflux_all,
                                     x4.hardflux2_all, x4.softflux_all, x4.esoftflux_all,x4.ext_all,
                                     x4.HR1_all, x4.HR2_all, x4.HR3_all, x4.HR4_all, sedflag=1)


m2Cat_GSW_3xmm.gsw_df['exptimes'] = x3.logmedtimes[x3.medtimefilt][m2Cat_GSW_3xmm.sedfilt]
m2Cat_GSW_3xmm_all.gsw_df['exptimes'] = x3.logmedtimes[x3.medtimefilt_all][m2Cat_GSW_3xmm_all.sedfilt]
m2Cat_GSW_3xmm.gsw_df['matchxrayra'] = x3.goodra[x3.medtimefilt][m2Cat_GSW_3xmm.sedfilt]
m2Cat_GSW_3xmm.gsw_df['matchxraydec'] = x3.gooddec[x3.medtimefilt][m2Cat_GSW_3xmm.sedfilt]
#m2Cat_GSW_3xmm.xraysourceids = x3.sourceids[x3.good_med][x3.medtimefilt][m2Cat_GSW_3xmm.sedfilt]
#m2Cat_GSW_3xmm.gsw_df['xrayobsids'] =  x3.obsids_matches[x3.medtimefilt][m2Cat_GSW_3xmm.sedfilt]
m2Cat_GSW_3xmm_all.gsw_df['matchxrayra'] = x3.goodra[x3.medtimefilt_all][m2Cat_GSW_3xmm_all.sedfilt]
m2Cat_GSW_3xmm_all.gsw_df['matchxraydec'] = x3.gooddec[x3.medtimefilt_all][m2Cat_GSW_3xmm_all.sedfilt]



m2Cat_GSW_4xmm.gsw_df['exptimes'] = x4.logmedtimes[x4.medtimefilt][m2Cat_GSW_4xmm.sedfilt]
m2Cat_GSW_4xmm_all.gsw_df['exptimes'] = x4.logmedtimes[x4.medtimefilt_all][m2Cat_GSW_4xmm_all.sedfilt]
m2Cat_GSW_4xmm.gsw_df['matchxrayra'] = x4.goodra[x4.medtimefilt][m2Cat_GSW_4xmm.sedfilt]
m2Cat_GSW_4xmm.gsw_df['matchxraydec'] = x4.gooddec[x4.medtimefilt][m2Cat_GSW_4xmm.sedfilt]
#m2Cat_GSW_3xmm.xraysourceids = x3.sourceids[x3.good_med][x3.medtimefilt][m2Cat_GSW_3xmm.sedfilt]
#m2Cat_GSW_4xmm.gsw_df['xrayobsids'] =  x4.obsids_matches[x4.medtimefilt][m2Cat_GSW_4xmm.sedfilt]
#m2Cat_GSW_4xmm_all.gsw_df['xrayobsids'] =  x4.obsids_matches[x4.medtimefilt_all][m2Cat_GSW_4xmm_all.sedfilt]

m2Cat_GSW_4xmm_all.gsw_df['matchxrayra'] = x4.goodra[x4.medtimefilt_all][m2Cat_GSW_4xmm_all.sedfilt]
m2Cat_GSW_4xmm_all.gsw_df['matchxraydec'] = x4.gooddec[x4.medtimefilt_all][m2Cat_GSW_4xmm_all.sedfilt]

m2Cat_GSW_3xmm_all_qsos.gsw_df['exptimes'] = x3.logmedtimes[x3.medtimefilt_all][m2Cat_GSW_3xmm_all_qsos.sedfilt]
m2Cat_GSW_4xmm_all_qsos.gsw_df['exptimes'] = x4.logmedtimes[x4.medtimefilt_all][m2Cat_GSW_4xmm_all_qsos.sedfilt]

#m2Cat_GSW_3xmm_all_qsos.gsw_df['xrayobsids'] =  x4.obsids_matches[x4.medtimefilt_all][m2Cat_GSW_3xmm_all_qsos.sedfilt]

m2Cat_GSW_3xmm_all_qsos.gsw_df['matchxrayra'] = x4.goodra[x4.medtimefilt_all][m2Cat_GSW_3xmm_all_qsos.sedfilt]
m2Cat_GSW_3xmm_all_qsos.gsw_df['matchxraydec'] = x4.gooddec[x4.medtimefilt_all][m2Cat_GSW_3xmm_all_qsos.sedfilt]

#m2Cat_GSW_4xmm_all_qsos.gsw_df['xrayobsids'] =  x4.obsids_matches[x4.medtimefilt_all][m2Cat_GSW_4xmm_all_qsos.sedfilt]

m2Cat_GSW_4xmm_all_qsos.gsw_df['matchxrayra'] = x4.goodra[x4.medtimefilt_all][m2Cat_GSW_4xmm_all_qsos.sedfilt]
m2Cat_GSW_4xmm_all_qsos.gsw_df['matchxraydec'] = x4.gooddec[x4.medtimefilt_all][m2Cat_GSW_4xmm_all_qsos.sedfilt]




#%% mpa matching

mpa_spec_m2_3xmm = MPAJHU_Spec(m2Cat_GSW_3xmm, sdssobj)
mpa_spec_m2_4xmm = MPAJHU_Spec(m2Cat_GSW_4xmm, sdssobj)

mpa_spec_m2_csc= MPAJHU_Spec(m2Cat_GSW_csc, sdssobj)

mpa_spec_qsos = MPAJHU_Spec(m2Cat_GSW_qsos, sdssobj, sedtyp=1)

mpa_spec_m2_3xmm_all = MPAJHU_Spec(m2Cat_GSW_3xmm_all, sdssobj)
mpa_spec_m2_4xmm_all = MPAJHU_Spec(m2Cat_GSW_4xmm_all, sdssobj)

mpa_spec_m2_3xmm_all_qsos = MPAJHU_Spec(m2Cat_GSW_3xmm_all_qsos, sdssobj, sedtyp=1)
mpa_spec_m2_4xmm_all_qsos = MPAJHU_Spec(m2Cat_GSW_4xmm_all_qsos, sdssobj, sedtyp=1)


mpa_spec_allm2 = MPAJHU_Spec(m2Cat_GSW, sdssobj, find=False, gsw=True)
mpa_spec_allx2 = MPAJHU_Spec(x2Cat_GSW, sdssobj, find=False, gsw=True)

first_spec_allm2 = FIRST_Spec(m2Cat_GSW, firstobj, find=False, gsw=True)
first_spec_allm2.spec_inds_prac, first_spec_allm2.spec_plates_prac, first_spec_allm2.spec_fibers_prac, first_spec_allm2.make_prac = np.int64(np.loadtxt(catfold+'first_gsw2_matching_info.txt', dtype=np.float64))

m2Cat_GSW_first = GSWCatmatch_radio( np.arange(len(m2[0]))[first_spec_allm2.make_prac], 
                                    m2, redshift_m2, allm2,
                                    firstobj.nvss_flux[first_spec_allm2.spec_inds_prac],
                                    firstobj.first_flux[first_spec_allm2.spec_inds_prac],
                                    firstobj.wenss_flux[first_spec_allm2.spec_inds_prac],
                                    firstobj.vlss_flux [first_spec_allm2.spec_inds_prac])


mpa_spec_allm2_first = MPAJHU_Spec(m2Cat_GSW_first, sdssobj, find=False, gsw=True)

#gsw_2matching_info = np.vstack([mpa_spec_allm2.spec_inds_prac, mpa_spec_allm2.spec_plates_prac, mpa_spec_allm2.spec_fibers_prac, mpa_spec_allm2.spec_mass_prac, mpa_spec_allm2.make_prac ])
#gsw_x2matching_info = np.vstack([mpa_spec_allx2.spec_inds_prac, mpa_spec_allx2.spec_plates_prac, mpa_spec_allx2.spec_fibers_prac, mpa_spec_allx2.spec_mass_prac, mpa_spec_allx2.make_prac ])

#first_gswmatching_info = np.vstack([first_spec_allm2.spec_inds_prac, first_spec_allm2.spec_plates_prac, first_spec_allm2.spec_fibers_prac, first_spec_allm2.make_prac ])
#first_dr7matching_info = np.vstack([mpa_spec_allm2_first.spec_inds_prac, mpa_spec_allm2_first.spec_plates_prac, mpa_spec_allm2_first.spec_fibers_prac, mpa_spec_allm2_first.make_prac ])

mpa_spec_allm2.spec_inds_prac, mpa_spec_allm2.spec_plates_prac, mpa_spec_allm2.spec_fibers_prac, mpa_spec_allm2.spec_mass_prac, mpa_spec_allm2.make_prac = np.loadtxt(catfold+'gsw2_dr7_matching_info.txt')
mpa_spec_allx2.spec_inds_prac, mpa_spec_allx2.spec_plates_prac, mpa_spec_allx2.spec_fibers_prac, mpa_spec_allx2.spec_mass_prac, mpa_spec_allx2.make_prac = np.loadtxt(catfold+'gsw2_x2_dr7_matching_info.txt')

mpa_spec_allm2_first.spec_inds_prac, mpa_spec_allm2_first.spec_plates_prac, mpa_spec_allm2_first.spec_fibers_prac, mpa_spec_allm2_first.make_prac = np.loadtxt(catfold+'first_dr7_matching_info.txt', unpack=True)

inds_comm, gsw_sedfilt_mpamake,mpamake_gsw_sedfilt=np.intersect1d(m2Cat_GSW.sedfilt, mpa_spec_allm2.make_prac, return_indices=True)

#new way of setting up gsw_3xmm_match filters sed flag beforehand so some of 
#the previous objects are now eing leftout (all QSOs/bad SEDfits so not a problem)
#need to combine inds from m2Cat_GSW.sedfilt ana mpa_spec_allm2.make_prac




#spec_inds_m2_3xmm, spec_plates_m2_3xmm, spec_fibers_m2_3xmm, specmass_m2_3xmm, make_m2_3xmm, miss_m2_3xmm, ids_sp_m2_3xmm  = matchspec_full(m2Cat_GSW_3xmm, sdssobj)
#spec_inds_m2_3xmm, spec_plates_m2_3xmm, spec_fibers_m2_3xmm, specmass_m2_3xmm, make_m2_3xmm, miss_m2_3xmm, ids_sp_m2_3xmm  = matchspec_prac(m2Cat_GSW_3xmm, sdssobj)
#spec_inds_m2_3xmm_all, spec_plates_m2_3xmm_all, spec_fibers_m2_3xmm_all, specmass_m2_3xmm_all, make_m2_3xmm_all, miss_m2_3xmm_all, ids_sp_m2_3xmm_all  = matchspec_prac(m2Cat_GSW_3xmm_all, sdssobj)

#spec_inds_allm2, spec_plates_allm2, spec_fibers_allm2, mass_allm2,make_allm2 = np.loadtxt(catfold+'gsw_dr7_matching_info.txt')
mpa_spec_allm2.spec_inds_prac = np.int64(mpa_spec_allm2.spec_inds_prac).reshape(-1)
mpa_spec_allm2.make_prac = np.int64(mpa_spec_allm2.make_prac).reshape(-1)
mpa_spec_allx2.spec_inds_prac = np.int64(mpa_spec_allx2.spec_inds_prac).reshape(-1)
mpa_spec_allx2.make_prac = np.int64(mpa_spec_allx2.make_prac).reshape(-1)

mpa_spec_qsos.spec_inds_prac = np.int64(mpa_spec_qsos.spec_inds_prac).reshape(-1)
mpa_spec_qsos.make_prac = np.int64(mpa_spec_qsos.make_prac).reshape(-1)
mpa_spec_m2_3xmm.spec_inds_prac = np.int64(mpa_spec_m2_3xmm.spec_inds_prac ).reshape(-1)
mpa_spec_m2_3xmm_all.spec_inds_prac  = np.int64(mpa_spec_m2_3xmm_all.spec_inds_prac ).reshape(-1)
mpa_spec_m2_4xmm.spec_inds_prac = np.int64(mpa_spec_m2_4xmm.spec_inds_prac ).reshape(-1)
mpa_spec_m2_4xmm_all.spec_inds_prac  = np.int64(mpa_spec_m2_4xmm_all.spec_inds_prac ).reshape(-1)
mpa_spec_m2_3xmm_all_qsos.spec_inds_prac = np.int64(mpa_spec_m2_3xmm_all_qsos.spec_inds_prac ).reshape(-1)
mpa_spec_m2_4xmm_all_qsos.spec_inds_prac  = np.int64(mpa_spec_m2_4xmm_all_qsos.spec_inds_prac ).reshape(-1)

mpa_spec_m2_3xmm.make_prac = np.int64(mpa_spec_m2_3xmm.make_prac ).reshape(-1)
mpa_spec_m2_3xmm_all.make_prac  = np.int64(mpa_spec_m2_3xmm_all.make_prac ).reshape(-1)
mpa_spec_m2_4xmm.make_prac = np.int64(mpa_spec_m2_4xmm.make_prac ).reshape(-1)
mpa_spec_m2_4xmm_all.make_prac  = np.int64(mpa_spec_m2_4xmm_all.make_prac ).reshape(-1)
mpa_spec_m2_3xmm_all_qsos.make_prac = np.int64(mpa_spec_m2_3xmm_all_qsos.make_prac ).reshape(-1)
mpa_spec_m2_4xmm_all_qsos.make_prac  = np.int64(mpa_spec_m2_4xmm_all_qsos.make_prac ).reshape(-1)


mpa_spec_m2_csc.spec_inds_prac = np.int64(mpa_spec_m2_csc.spec_inds_prac ).reshape(-1)
mpa_spec_allm2_first.spec_inds_prac = np.int64(mpa_spec_allm2_first.spec_inds_prac ).reshape(-1)
mpa_spec_allm2_first.make_prac = np.int64(mpa_spec_allm2_first.make_prac ).reshape(-1)
mpa_spec_allm2_first.spec_plates_prac = np.int64(mpa_spec_allm2_first.spec_plates_prac ).reshape(-1)
mpa_spec_allm2_first.spec_fibers_prac = np.int64(mpa_spec_allm2_first.spec_fibers_prac ).reshape(-1)

def latextable(table):
    for row in table:
        out= '%s & '*(len(row)-1) +'%s \\\\'
        print(out % tuple(row))


'''
actual analysis begins below
'''

#%% Emission line objects
EL_qsos = ELObj(mpa_spec_qsos.spec_inds_prac , sdssobj, mpa_spec_qsos.make_prac,m2Cat_GSW_qsos,gsw=True)
EL_m2 = ELObj(mpa_spec_allm2.spec_inds_prac , sdssobj, gsw_sedfilt_mpamake, m2Cat_GSW,gsw=True, dustbinning=True, empirdust=True)
EL_first = ELObj(mpa_spec_allm2_first.spec_inds_prac , sdssobj, mpa_spec_allm2_first.make_prac,m2Cat_GSW_first,gsw=True, dustbinning=False, empirdust=False, radio = True )

EL_3xmm  = ELObj(mpa_spec_m2_3xmm.spec_inds_prac , sdssobj, mpa_spec_m2_3xmm.make_prac,m2Cat_GSW_3xmm, xr=True, xmm=True, empirdust=False)
EL_4xmm  = ELObj(mpa_spec_m2_4xmm.spec_inds_prac , sdssobj, mpa_spec_m2_4xmm.make_prac,m2Cat_GSW_4xmm, xr=True, xmm=True, empirdust=False)

EL_csc  = ELObj(mpa_spec_m2_csc.spec_inds_prac , sdssobj, mpa_spec_m2_csc.make_prac,m2Cat_GSW_csc, xr=True)

EL_3xmm_all = ELObj(mpa_spec_m2_3xmm_all.spec_inds_prac , sdssobj, mpa_spec_m2_3xmm_all.make_prac, m2Cat_GSW_3xmm_all, xr=True, xmm=True, empirdust=False)
EL_4xmm_all = ELObj(mpa_spec_m2_4xmm_all.spec_inds_prac , sdssobj, mpa_spec_m2_4xmm_all.make_prac, m2Cat_GSW_4xmm_all, xr=True, xmm=True, empirdust=False)

EL_3xmm_all_qsos = ELObj(mpa_spec_m2_3xmm_all_qsos.spec_inds_prac , sdssobj, mpa_spec_m2_3xmm_all_qsos.make_prac, m2Cat_GSW_3xmm_all_qsos, xr=True, xmm=True, empirdust=False)
EL_4xmm_all_qsos = ELObj(mpa_spec_m2_4xmm_all_qsos.spec_inds_prac , sdssobj, mpa_spec_m2_4xmm_all_qsos.make_prac, m2Cat_GSW_4xmm_all_qsos, xr=True, xmm=True, empirdust=False)

#%% X-ray lum sfr filtering
'''
XRAY LUM -SFR
'''


softxray_xmm = Xraysfr(np.array(m2Cat_GSW_3xmm.gsw_df.softlumsrf), m2Cat_GSW_3xmm, 
                       mpa_spec_m2_3xmm.make_prac[EL_3xmm.bpt_sn_filt], 
                       EL_3xmm.bptagn,EL_3xmm.bptsf, 'soft')
hardxray_xmm = Xraysfr(np.array(m2Cat_GSW_3xmm.gsw_df.hardlumsrf), m2Cat_GSW_3xmm, 
                       mpa_spec_m2_3xmm.make_prac[EL_3xmm.bpt_sn_filt], 
                       EL_3xmm.bptagn,EL_3xmm.bptsf,'hard')
fullxray_xmm = Xraysfr(np.array(m2Cat_GSW_3xmm.gsw_df.fulllumsrf), m2Cat_GSW_3xmm,
                       mpa_spec_m2_3xmm.make_prac[EL_3xmm.bpt_sn_filt], 
                       EL_3xmm.bptagn, EL_3xmm.bptsf, 'full')
fullxray_xmm_bptplus = Xraysfr(np.array(m2Cat_GSW_3xmm.gsw_df.fulllumsrf), m2Cat_GSW_3xmm,
                       mpa_spec_m2_3xmm.make_prac[EL_3xmm.bpt_sn_filt], 
                       EL_3xmm.bptplsagn, EL_3xmm.bptplssf, 'full')

fullxray_xmm4 = Xraysfr(np.array(m2Cat_GSW_4xmm.gsw_df.fulllumsrf), m2Cat_GSW_4xmm,
                       mpa_spec_m2_4xmm.make_prac[EL_4xmm.bpt_sn_filt], 
                       EL_4xmm.bptagn, EL_4xmm.bptsf, 'full')
fullxray_xmm4_bptplus = Xraysfr(np.array(m2Cat_GSW_4xmm.gsw_df.fulllumsrf), m2Cat_GSW_4xmm,
                       mpa_spec_m2_4xmm.make_prac[EL_4xmm.bpt_sn_filt], 
                       EL_4xmm.bptplsagn, EL_4xmm.bptplssf, 'full')
fullxray_csc = Xraysfr(np.array(m2Cat_GSW_csc.gsw_df.fulllumsrf), m2Cat_GSW_csc,
                       mpa_spec_m2_csc.make_prac[EL_csc.bpt_sn_filt], 
                       EL_csc.bptagn, EL_csc.bptsf, 'full')

fullxray_xmm_dr7 = Xraysfr(np.array(m2Cat_GSW_3xmm.gsw_df.fulllumsrf), m2Cat_GSW_3xmm, mpa_spec_m2_3xmm.make_prac,  EL_3xmm.bptagn, EL_3xmm.bptsf, 'full')
fullxray_xmm4_dr7 = Xraysfr(np.array(m2Cat_GSW_4xmm.gsw_df.fulllumsrf), m2Cat_GSW_4xmm, mpa_spec_m2_4xmm.make_prac,  EL_4xmm.bptagn, EL_4xmm.bptsf, 'full')

softxray_xmm_all = Xraysfr(np.array(m2Cat_GSW_3xmm_all.gsw_df.softlumsrf), m2Cat_GSW_3xmm_all, 
                           mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt], 
                           EL_3xmm_all.bptagn,EL_3xmm_all.bptsf, 'soft')
hardxray_xmm_all = Xraysfr(np.array(m2Cat_GSW_3xmm_all.gsw_df.hardlumsrf), m2Cat_GSW_3xmm_all,
                           mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt],
                           EL_3xmm_all.bptagn,EL_3xmm_all.bptsf, 'hard')
fullxray_xmm_all = Xraysfr(np.array(m2Cat_GSW_3xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_3xmm_all, 
                           mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt],
                           EL_3xmm_all.bptagn, EL_3xmm_all.bptsf, 'full')
fullxray_xmm4_all = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all, 
                           mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.bpt_sn_filt],
                           EL_4xmm_all.bptagn, EL_4xmm_all.bptsf, 'full')


fullxray_xmm_all_high_sn_o3 = Xraysfr(np.array(m2Cat_GSW_3xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_3xmm_all, 
                           mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.high_sn_o3],
                           np.arange(len(EL_3xmm_all.high_sn_o3)), [], 'full')
fullxray_xmm_all_xragn = Xraysfr(np.array(m2Cat_GSW_3xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_3xmm_all, 
                           mpa_spec_m2_3xmm_all.make_prac,
                           np.arange(len(EL_3xmm_all.EL_gsw_df)), [], 'full')
fullxray_xmm_all_bptplus = Xraysfr(np.array(m2Cat_GSW_3xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_3xmm_all, 
                                   mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt], 
                                   EL_3xmm_all.bptplsagn, EL_3xmm_all.bptplssf, 'full')

fullxray_xmm4_all_high_sn_o3 = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all, 
                           mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.high_sn_o3],
                           np.arange(len(EL_4xmm_all.high_sn_o3)), [], 'full')
fullxray_xmm4_all_xragn = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all, 
                           mpa_spec_m2_4xmm_all.make_prac,
                           np.arange(len(EL_4xmm_all.EL_gsw_df)), [], 'full')
fullxray_xmm4_all_bptplus = Xraysfr(np.array(m2Cat_GSW_4xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_4xmm_all, 
                                   mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.bpt_sn_filt], 
                                   EL_4xmm_all.bptplsagn, EL_4xmm_all.bptplssf, 'full')



softxray_xmm_no = Xraysfr(np.array(m2Cat_GSW_3xmm.gsw_df.softlumsrf), m2Cat_GSW_3xmm, mpa_spec_m2_3xmm.make_prac[EL_3xmm.not_bpt_sn_filt_bool], [], [], 'soft')
hardxray_xmm_no = Xraysfr(np.array(m2Cat_GSW_3xmm.gsw_df.hardlumsrf), m2Cat_GSW_3xmm, mpa_spec_m2_3xmm.make_prac[EL_3xmm.not_bpt_sn_filt_bool], [], [], 'hard')
fullxray_xmm_no = Xraysfr(np.array(m2Cat_GSW_3xmm.gsw_df.fulllumsrf), m2Cat_GSW_3xmm, mpa_spec_m2_3xmm.make_prac[EL_3xmm.not_bpt_sn_filt_bool], [], [], 'full')
fullxray_xmm_all_no = Xraysfr(np.array(m2Cat_GSW_3xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_3xmm_all,
                              mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.not_bpt_sn_filt_bool], [], [], 'full')
fullxray_xmm_all_unclass_p2 = Xraysfr(np.array(m2Cat_GSW_3xmm_all.gsw_df.fulllumsrf), m2Cat_GSW_3xmm_all,
                              mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.neither_filt], [], [], 'full')


#%% refiltering emission line objects by x-ray properties
xmm3eldiagmed_xrfilt = ELObj(mpa_spec_m2_3xmm.spec_inds_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr], 
                             sdssobj,
                             mpa_spec_m2_3xmm.make_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr], 
                             m2Cat_GSW_3xmm,xr=True, xmm=True)
xmm4eldiagmed_xrfilt = ELObj(mpa_spec_m2_4xmm.spec_inds_prac[EL_4xmm.bpt_sn_filt][fullxray_xmm4.valid][fullxray_xmm4.likelyagn_xr], 
                             sdssobj,
                             mpa_spec_m2_4xmm.make_prac[EL_4xmm.bpt_sn_filt][fullxray_xmm4.valid][fullxray_xmm4.likelyagn_xr], 
                             m2Cat_GSW_4xmm,xr=True, xmm=True)

csceldiagmed_xrfilt = ELObj(mpa_spec_m2_csc.spec_inds_prac[EL_csc.bpt_sn_filt][fullxray_csc.valid][fullxray_csc.likelyagn_xr], 
                             sdssobj,
                             mpa_spec_m2_csc.make_prac[EL_csc.bpt_sn_filt][fullxray_csc.valid][fullxray_csc.likelyagn_xr], 
                             m2Cat_GSW_csc,xr=True, xmm=False)

xmm3eldiagmed_xrfiltbptplus = ELObj(mpa_spec_m2_3xmm.spec_inds_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm_bptplus.valid][fullxray_xmm_bptplus.likelyagn_xr], sdssobj,
                             mpa_spec_m2_3xmm.make_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm_bptplus.valid][fullxray_xmm_bptplus.likelyagn_xr], 
                             m2Cat_GSW_3xmm,xr=True, xmm=True)

xmm3eldiagmed_xrsffilt = ELObj(mpa_spec_m2_3xmm.spec_inds_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelysf], sdssobj,
                             mpa_spec_m2_3xmm.make_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelysf], m2Cat_GSW_3xmm,xr=True, xmm=True)

xmm3eldiagmed_xrfilt_all =  ELObj(mpa_spec_m2_3xmm_all.spec_inds_prac[EL_3xmm_all.bpt_sn_filt][fullxray_xmm_all.valid][fullxray_xmm_all.likelyagn_xr], sdssobj,
                                  mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt][fullxray_xmm_all.valid][fullxray_xmm_all.likelyagn_xr], 
                                  m2Cat_GSW_3xmm_all,xr=True, xmm=True)
xmm4eldiagmed_xrfilt_all =  ELObj(mpa_spec_m2_4xmm_all.spec_inds_prac[EL_4xmm_all.bpt_sn_filt][fullxray_xmm4_all.valid][fullxray_xmm4_all.likelyagn_xr], sdssobj,
                                  mpa_spec_m2_4xmm_all.make_prac[EL_4xmm_all.bpt_sn_filt][fullxray_xmm4_all.valid][fullxray_xmm4_all.likelyagn_xr], 
                                  m2Cat_GSW_4xmm_all,xr=True, xmm=True)

xmm3eldiagmed_xrfilt_all_high_sn_o3 =  ELObj(mpa_spec_m2_3xmm_all.spec_inds_prac[EL_3xmm_all.high_sn_o3][fullxray_xmm_all_high_sn_o3.valid][fullxray_xmm_all_high_sn_o3.likelyagn_xr], sdssobj,
                                  mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.high_sn_o3][fullxray_xmm_all_high_sn_o3.valid][fullxray_xmm_all_high_sn_o3.likelyagn_xr], m2Cat_GSW_3xmm_all,xr=True, xmm=True)

xmm3eldiagmed_xrfiltbptplus_all =  ELObj(mpa_spec_m2_3xmm_all.spec_inds_prac[EL_3xmm_all.bpt_sn_filt][fullxray_xmm_all_bptplus.valid][fullxray_xmm_all_bptplus.likelyagn_xr], 
                                         sdssobj,
                                  mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt][fullxray_xmm_all_bptplus.valid][fullxray_xmm_all_bptplus.likelyagn_xr], 
                                  m2Cat_GSW_3xmm_all, xr=True, xmm=True)

xmm3eldiagmed_xrfilt_xragn =  ELObj(mpa_spec_m2_3xmm_all.spec_inds_prac[fullxray_xmm_all_xragn.valid][fullxray_xmm_all_xragn.likelyagn_xr], 
                                    sdssobj,
                                  mpa_spec_m2_3xmm_all.make_prac[fullxray_xmm_all_xragn.valid][fullxray_xmm_all_xragn.likelyagn_xr], 
                                  m2Cat_GSW_3xmm_all,xr=True, xmm=True)
xmm4eldiagmed_xrfilt_xragn =  ELObj(mpa_spec_m2_4xmm_all.spec_inds_prac[fullxray_xmm4_all_xragn.valid][fullxray_xmm4_all_xragn.likelyagn_xr], 
                                    sdssobj,
                                  mpa_spec_m2_4xmm_all.make_prac[fullxray_xmm4_all_xragn.valid][fullxray_xmm4_all_xragn.likelyagn_xr], 
                                  m2Cat_GSW_4xmm_all,xr=True, xmm=True)
xmm3eldiagmed_xrfilt_unclass_p2 =  ELObj(mpa_spec_m2_3xmm_all.spec_inds_prac[EL_3xmm_all.neither_filt][fullxray_xmm_all_unclass_p2.valid][fullxray_xmm_all_unclass_p2.likelyagn_xr], 
                                    sdssobj,
                                  mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.neither_filt][fullxray_xmm_all_unclass_p2.valid][fullxray_xmm_all_unclass_p2.likelyagn_xr], 
                                  m2Cat_GSW_3xmm_all,xr=True, xmm=True, weak_lines=True)
xmm3eldiagmed_xrfilt_unclass_p1 =  ELObj(mpa_spec_m2_3xmm_all.spec_inds_prac[EL_3xmm_all.not_bpt_sn_filt_bool][fullxray_xmm_all_no.valid][fullxray_xmm_all_no.likelyagn_xr], 
                                    sdssobj,
                                  mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.not_bpt_sn_filt_bool][fullxray_xmm_all_no.valid][fullxray_xmm_all_no.likelyagn_xr], 
                                  m2Cat_GSW_3xmm_all,xr=True, xmm=True)





#obs_gals_inds = [1,2,4,5,6,8,9,10,12,13,14,16,21,52,60,76,78,81,82,83,84,85,86,87]


#unclass = mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.not_bpt_sn_filt][fullxray_xmm_all_no.likelyagn_xr]
#unclass_EL_3xmm_all_xragn = EL_3xmm_all.not_bpt_sn_filt[fullxray_xmm_all_no.likelyagn_xr]
#muse_samp_inds = [3,6, 28, 57] #7,27, 46
#unclass_muse_samp_inds = [22,23,57,104,210]

#het_samp_inds = [11, 13, 16,  21, 22,  79, 82, 85, 87] #9,10, 14, 20, 77
#unclass_het_samp_inds = [68, 87, 91, 102, 110, 483, 487, 497, 510, 523, 525, 526, 534]


#%% Gdiffs

'''
Gdiffs
'''
xmm3inds = m2Cat_GSW_3xmm.inds[m2Cat_GSW_3xmm.sedfilt][mpa_spec_m2_3xmm.make_prac][EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr]
xmm3ids = m2Cat_GSW_3xmm.ids[mpa_spec_m2_3xmm.make_prac][EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr]
gswids = m2[0][mpa_spec_allm2.make_prac][EL_m2.bpt_sn_filt]
covered_gsw_x3 = np.int64(np.loadtxt('catalogs/xraycov/matched_gals_gsw2xmm3_xrcovg_fields_set.txt'))

'''
xmm3gdiff = Gdiffs(xmm3ids, gswids, xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, EL_m2.bpt_EL_gsw_df, x_xr=xmm3eldiagmed_xrfilt.bpt_EL_gsw_df['niiha'],
y_xr=xmm3eldiagmed_xrfilt.bpt_EL_gsw_df['oiiihb'], x_gsw=EL_m2.bpt_EL_gsw_df['niiha'], y_gsw=EL_m2.bpt_EL_gsw_df['oiiihb'])
    
xmm3gdiff.get_filt(covered_gsw_x3)
commdiffidsxmm3, xmm3diffcomm, xmm3gswdiffcomm = commonpts1d(xmm3ids, gswids[covered_gsw_x3])
xmm3gdiff.nearbyx(xmm3gswdiffcomm)
xmm3gdiff.getdist_by_thresh(3.0)
'''

def get_gdiffs_set(xr_ids, gswids, covered_inds, xr_df, gsw_df, quantities):
    
    gdiff_set = Gdiffs(xr_ids, gswids, xr_df, gsw_df, quantities)
    commdiffids, x3diffcomm, x3gswdiffcomm = commonpts1d(xr_ids, gswids[covered_inds])
    gdiff_set.get_filt(covered_inds)    

    gdiff_set.nearbyx(x3gswdiffcomm)
    gdiff_set.getdist_by_thresh(3.0)
    return gdiff_set



xmm3gdiff =get_gdiffs_set(xmm3ids, gswids, covered_gsw_x3,                                  
                                  xmm3eldiagmed_xrfilt.bpt_EL_gsw_df,EL_m2.bpt_EL_gsw_df, ['mass', 'z', 'niiha', 'oiiihb'])

xmm3gdiff_set_mass_ssfr =get_gdiffs_set(xmm3ids, gswids, covered_gsw_x3,                                  
                                  xmm3eldiagmed_xrfilt.bpt_EL_gsw_df,EL_m2.bpt_EL_gsw_df, ['mass', 'z', 'ssfr'])


binnum = 60
contaminations_xmm3 = xmm3gdiff.xrgswfracs[:,binnum]
contaminations_xmm3_2 = xmm3gdiff.xrgswfracs[:, 40]
contaminations_xmm3_25 = xmm3gdiff.xrgswfracs[:, 50]
contaminations_xmm3_3 = xmm3gdiff.xrgswfracs[:, 60]
contaminations_xmm3_35 = xmm3gdiff.xrgswfracs[:, 70]
contaminations_xmm3_4 = xmm3gdiff.xrgswfracs[:, 80]
#xmm3gdiff.interpdistgrid(11,11,50,method='linear')
xmm3eldiagmed_xrfilt.bpt_EL_gsw_df['xrfracs']=contaminations_xmm3_25


#%% sfrm

'''

match sfr
'''


#kmnclust = np.loadtxt('catalogs/kmeans2.dat', unpack=True)



#kmnclust2 =  kmnclust[1]
'''
val2 = np.where(kmnclust2 != 0 )[0]
sy2_1 = np.where(kmnclust2==1)[0]
hliner_1 = np.where(kmnclust2==2)[0]
sliner_1 = np.where(kmnclust2==3)[0]
'''
from sfrmatch import SFRMatch

sfrm_gsw2 = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df)

sfrm_gsw2.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, second=True,
                                load=True,  sncut_forb=2,sncut_balm=2, with_av=True, fname='_second_1117')

sfrm_gsw2.subtract_elflux(sncut=2,  halphbeta_sncut=2, second=False)

'''



sfrm_gsw2_old = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df )
sfrm_gsw2_old.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True,   sncut_forb=-9999,sncut_balm=-9999, with_av=True, fname='', old=True, subfold='old_matching/')

sfrm_gsw2_old.subtract_elflux(sncut=2,  halphbeta_sncut=2, second=False)


'''

'''
sfrm_gsw2_best = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df )
sfrm_gsw2_best.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True,  sncut_forb=-9999,sncut_balm=-9999, with_av=True, fname='_no_pos_sn_req', old=False)
sfrm_gsw2_best.subtract_elflux(sncut=2,  halphbeta_sncut=2, second=False)

sfrm_gsw2_half = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df )
sfrm_gsw2_half.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True, sncut_forb=2, sncut_balm= -99, with_av=True, fname='_no_pos_balm_req', old=False)
sfrm_gsw2_half.subtract_elflux(sncut=2,  halphbeta_sncut=2, second=False)



sfrm_gsw2_best0 = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df )
sfrm_gsw2_best0.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True,  sncut=0, with_av=True, fname='_pos_sn_req', old=False)

67071

sfrm_gsw2_best0.agn_dist_inds = np.delete(sfrm_gsw2_best0.agn_dist_inds,22309)
sfrm_gsw2_best0.agn_plus_dist_inds = np.insert(sfrm_gsw2_best0.agn_plus_dist_inds, 10702, 67071)

sfrm_gsw2_best0.subtract_elflux(sncut=2,  halphbeta_sncut=2, second=False)


'''

full_agn_filt= np.where(sfrm_gsw2.allagn_df.full_agn)[0]
#sfrm_gsw2.fullagn_df = sfrm_gsw2.allagn_df.iloc[full_agn_filt].copy()


kmnclust = np.loadtxt('catalogs/kmeans_sfrm.dat', unpack=True)

kmnclust1 =  kmnclust[full_agn_filt]
val1 = np.where(kmnclust1 != 0 )[0]
sy2_1 = np.where(kmnclust1==1)[0]
hliner_1 = np.where(kmnclust1==2)[0]
sliner_1 = np.where(kmnclust1==3)[0]
allliners = np.append(sliner_1, hliner_1)



kmnclust_woo1 = np.loadtxt('catalogs/kmeans_sfrm_woo1.dat', unpack=True)

kmnclust2=  kmnclust_woo1[full_agn_filt]
val2 = np.where(kmnclust2 != 0 )[0]
sy2_2 = np.where(kmnclust2==1)[0]
hliner_2 = np.where(kmnclust2==2)[0]
sliner_2 = np.where(kmnclust2==3)[0]
allliners_2 = np.append(sliner_2, hliner_2)

combo_classes =[]
for i in range(len(kmnclust)):
    if kmnclust[i] !=0:
        combo_classes.append(kmnclust[i])
    else:
        combo_classes.append(kmnclust_woo1[i])
combo_classes=np.array(combo_classes) 

combo_classes_full = combo_classes[full_agn_filt]

combo_val = np.where(combo_classes_full !=0)[0]
combo_sy2 = np.where(combo_classes_full ==1)[0]
combo_hliner = np.where(combo_classes_full ==2)[0]
combo_sliner = np.where(combo_classes_full ==3)[0]


kmnclust1[kmnclust1==0] = 9
combo_classes_full[combo_classes_full==0] = 9

sfrm_gsw2.fullagn_df['sevenline_classes'] = kmnclust1
sfrm_gsw2.fullagn_df['sixline_classes'] = combo_classes_full

fulltab = pd.merge(EL_m2.EL_gsw_df, sfrm_gsw2.fullagn_df, on ='ids', how='left')

fulltab['sevenline_classes'] = fulltab['sevenline_classes'].fillna(0).copy()

fulltab['sixline_classes'] = fulltab['sixline_classes'].fillna(0).copy()
fulltab['genclass'] = np.ones_like(fulltab['sixline_classes'])*9

fulltab.loc[EL_m2.neither_filt,'genclass']=0

fulltab.loc[EL_m2.bpt_sn_filt[EL_m2.bptplssf],'genclass']=1
fulltab.loc[EL_m2.halp_nii_filt[EL_m2.bptplsnii_sf],'genclass']=2

fulltab.loc[EL_m2.halp_nii_filt[EL_m2.bptplsnii_agn],'genclass']=3

fulltab.loc[np.where(fulltab['sixline_classes']!=0)[0],'genclass']=5
fulltab.loc[np.where(fulltab['genclass']==9)[0], 'genclass']=4

low_z = np.where(sfrm_gsw2.fullagn_df.z<=0.07)[0]
low_z_val1 =np.where(kmnclust1[low_z] !=0 )[0]
low_z_sy2_1=np.where(kmnclust1[low_z] ==1 )[0]
low_z_hliner_1 =np.where(kmnclust1[low_z] ==2 )[0]
low_z_sliner_1 =np.where(kmnclust1[low_z] ==3 )[0]
low_z_allliners = np.append(low_z_sliner_1, low_z_hliner_1)

allnon_sy2 = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.allnonagn_df.copy()])
allnon_sy2_highsn = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.alllines_bpt_sf_df.copy()])

allnon_sy2_lowz = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.allnonagn_df.iloc[np.where(EL_m2.allnonagn_df.z<=0.07)].copy()])
allnon_sy2_highsn_lowz = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.alllines_bpt_sf_df.iloc[np.where(EL_m2.alllines_bpt_sf_df.z<=0.07)].copy()])

allnonsy2_lowz = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.allnonagn_df.iloc[np.where(EL_m2.allnonagn_df.z<=0.07)].copy()])
allnonagn_lowz = EL_m2.allnonagn_df.iloc[np.where(EL_m2.allnonagn_df.z<=0.07)].copy()

sf_lowz = EL_m2.bptplus_sf_df.iloc[np.where(EL_m2.bptplus_sf_df.z<=0.07)].copy()

sf_highsn_lowz = EL_m2.alllines_bptplus_sf_df.iloc[np.where(EL_m2.alllines_bptplus_sf_df.z<=0.07)].copy()
sf_highsn_lowz = pd.concat([sfrm_gsw2.low_z_df.iloc[low_z_allliners].copy(), EL_m2.alllines_bptplus_sf_df.iloc[np.where(EL_m2.alllines_bptplus_sf_df.z<=0.07)].copy()])

sfplus_lowz = EL_m2.bptplusnii_sf_df.iloc[np.where(EL_m2.bptplusnii_sf_df.z<=0.07)].copy()

'''

val1_n2o2_match = np.where((sfrm_gsw2.fullmatch_df.niiflux_sn.iloc[val1]>2)&(sfrm_gsw2.fullmatch_df.oiiflux_sn.iloc[val1]>2))[0]
sy2_n2o2_match = np.where((sfrm_gsw2.fullmatch_df.niiflux_sn.iloc[sy2_1]>2)&(sfrm_gsw2.fullmatch_df.oiiflux_sn.iloc[sy2_1]>2))[0]
sliner_n2o2_match = np.where((sfrm_gsw2.fullmatch_df.niiflux_sn.iloc[sliner_1]>2)&(sfrm_gsw2.fullmatch_df.oiiflux_sn.iloc[sliner_1]>2))[0]
hliner_n2o2_match = np.where((sfrm_gsw2.fullmatch_df.niiflux_sn.iloc[hliner_1]>2)&(sfrm_gsw2.fullmatch_df.oiiflux_sn.iloc[hliner_1]>2))[0]


val1_oh = np.where(sfrm_gsw2.fullmatch_df.ohabund.iloc[val1]>0)[0]
sy2_oh = np.where(sfrm_gsw2.fullmatch_df.ohabund.iloc[sy2_1]>0)[0]
sliner_oh = np.where(sfrm_gsw2.fullmatch_df.ohabund.iloc[sliner_1]>0)[0]
hliner_oh = np.where(sfrm_gsw2.fullmatch_df.ohabund.iloc[hliner_1]>0)[0]



kmnclust_hd = np.loadtxt('catalogs/kmeans_d4000_hdelta.dat', unpack=True)

full_agn_filt_hd= np.where(d4000mhd_gsw2.allagn_df.full_agn)[0]

kmnclust1_hd =  kmnclust_hd[full_agn_filt_hd]
val1_hd = np.where(kmnclust1_hd != 0 )[0]
sy2_1_hd = np.where(kmnclust1_hd==1)[0]
hliner_1_hd = np.where(kmnclust1_hd==2)[0]
sliner_1_hd = np.where(kmnclust1_hd==3)[0]


full_agn = np.where(sfrm_gsw2.allagn_df.full_agn==1)[0]
val_full_agn = np.where(sfrm_gsw2.fullagn_df.iloc[val1]==1)
sy2_full_agn = np.where(sfrm_gsw2.allagn_df.iloc[sy2_1]==1)
sf_full_agn = np.where(sfrm_gsw2.allagn_df.iloc[sliner_1]==1)
liner2_full_agn = np.where(sfrm_gsw2.allagn_df.iloc[hliner_1]==1)


sfrm_gsw2_sec = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df)

sfrm_gsw2_sec.get_highsn_match_only(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True,  sncut=2, with_av=True, fname='_second_')
sfrm_gsw2_sec.subtract_elflux(sncut=2,  halphbeta_sncut=2)
'''

val2_sii_doub= np.where((kmnclust1 != 0) &(sfrm_gsw2.fullagn_df.sii6717flux_sub_sn>2) &(sfrm_gsw2.fullagn_df.sii6731flux_sub_sn>2))[0]
sy2_1_sii_doub = np.where((kmnclust1==1) &(sfrm_gsw2.fullagn_df.sii6717flux_sub_sn>2) &(sfrm_gsw2.fullagn_df.sii6731flux_sub_sn>2))[0]
sliner_1_sii_doub = np.where((kmnclust1==3) &(sfrm_gsw2.fullagn_df.sii6717flux_sub_sn>2) &(sfrm_gsw2.fullagn_df.sii6731flux_sub_sn>2))[0]
hliner_1_sii_doub = np.where((kmnclust1==2) &(sfrm_gsw2.fullagn_df.sii6717flux_sub_sn>2) &(sfrm_gsw2.fullagn_df.sii6731flux_sub_sn>2))[0]


val2_oi_3 =  np.where((kmnclust1 != 0) &(sfrm_gsw2.fullagn_df.oiflux_sub_sn>3) )[0]
sy2_1_oi_3 = np.where((kmnclust1==1) &(sfrm_gsw2.fullagn_df.oiflux_sub_sn>3) )[0]
sliner_1_oi_3 = np.where((kmnclust1==3) &(sfrm_gsw2.fullagn_df.oiflux_sub_sn>3) )[0]
hliner_1_oi_3 = np.where((kmnclust1==2) &(sfrm_gsw2.fullagn_df.oiflux_sub_sn>3))[0]




sncut=2
combo_sncut=2, 
delta_ssfr_cut=-0.7
minmass=10.2
maxmass=10.4, 
d4000_cut=1.6


upperoiiilum=40.14
loweroiiilum=40.06

upperU_cut=-0.2
lowerU_cut=-0.3,


filts = {'delta_ssfr': {'cut':[delta_ssfr_cut]},
             #'mass':{'cut': [minmass,maxmass]},
             #'d4000':{'cut':[d4000_cut]},
             #'oiiilum': {'cut':[loweroiiilum,upperoiiilum]},
             #'oiiilum_sub_dered': {'cut':[loweroiiilum,upperoiiilum]},
             #'U_sub': {'cut':[lowerU_cut,upperU_cut]},
             #'U': {'cut':[lowerU_cut,upperU_cut]},
             #'qc_sub':{'cut':[lowerqc_cut, upperqc_cut]},
             #'q_sub':{'cut':[lowerq_cut, upperq_cut]}
             'sy2_liner_bool':{'cut':[0.5]}
              }
gen_filts = {}
match_filts = {'delta_ssfr': {'cut':[delta_ssfr_cut]},
             #'mass':{'cut': [minmass,maxmass]},
             #'d4000':{'cut':[d4000_cut]},
             #'oiiilum': {'cut':[loweroiiilum,upperoiiilum]},
             #'oiiilum_sub_dered': {'cut':[loweroiiilum,upperoiiilum]},
             #'U_sub': {'cut':[lowerU_cut,upperU_cut]},
             #'U': {'cut':[lowerU_cut,upperU_cut]},
             'sy2_liner_bool':{'cut':[0.5]}
             
             #'qc_sub':{'cut':[lowerq_cut, upperq_cut]},
             #'q_sub':{'cut':[lowerq_cut, upperq_cut]}

              }
line_filts = [['oiflux',sncut],
              ['oiiflux',sncut],
              ['siiflux',sncut],
              ['oiflux_sub',sncut],
              ['oiiflux_sub',sncut],
              ['siiflux_sub',sncut]                      
              ]
line_filts_comb = {'sii_oii_sub':[['siiflux_sub', 'oiiflux_sub'],combo_sncut],
                   'sii_oi_sub':[['siiflux_sub','oiflux_sub'],3],
                   'oi_oii_sub':[['oiflux_sub', 'oiiflux_sub'],combo_sncut],
                   'sii_oii_oi_sub':[['siiflux_sub', 'oiiflux_sub', 'oiflux_sub'], combo_sncut],
                   'sii_oii_ha_hb_sub':[['siiflux_sub','oiiflux_sub','halpflux_sub','hbetaflux_sub'],combo_sncut],
                   #'o32_sub':[['oiii4959flux_sub','oiiflux_sub'], combo_sncut],
                   'siidoub_sub':[['sii6717flux_sub', 'sii6731flux_sub'],combo_sncut ],
                   
                   #'sii_oii':[['siiflux', 'oiiflux'],combo_sncut],
                   #'sii_oi':[['siiflux','oiflux'],combo_sncut],
                   #'oi_oii':[['oiflux', 'oiiflux'],combo_sncut],
                   #'sii_oii_ha_hb':[['siiflux','oiiflux','halpflux','hbetaflux'],combo_sncut],
                   #'siidoub':[['sii6717flux', 'sii6731flux'],combo_sncut ]
                   
                   }
#sfrm_gsw2.get_filt_dfs(filts, gen_filts, match_filts, line_filts, line_filts_comb, loweroiiilum=40.0, upperoiiilum=40.2)
#get_filt_dfs(self, sncut=2, combo_sncut=2, 
#                     delta_ssfr_cut=-0.7, 
#                     minmass=10.2, maxmass=10.4, 
#                     d4000_cut=1.6, 
#                     loweroiiilum=40.2, upperoiiilum=40.3, 
#                     upperU_cut=-0.2, lowerU_cut=-0.3,):



#sfrm_gsw2.bin_by_bpt( val1, sy2_1, sliner_1, hliner_1,binsize=0.1)


cols_to_use = sfrm_gsw2.fullagn_df.columns.difference(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df.columns)
cols_to_use  = cols_to_use.tolist()
cols_to_use.append('ids')


cols_to_use_csc = sfrm_gsw2.fullagn_df.columns.difference(csceldiagmed_xrfilt.bpt_EL_gsw_df.columns)
cols_to_use_csc  = cols_to_use_csc.tolist()
cols_to_use_csc.append('ids')


merged_xr = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[ cols_to_use], on='ids')
merged_xr_all = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[ cols_to_use], on='ids')

merged_xr_val = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[val1], on='ids')
merged_xr_val_all = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[val1], on='ids')

merged_xr_val_combo = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_val], on='ids')
merged_xr_val_all_combo = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_val], on='ids')

merged_xr_sy2 = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sy2_1], on='ids')
merged_xr_sf = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sliner_1], on='ids')
merged_xr_liner2 = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[hliner_1], on='ids')

merged_xr_sy2_all = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sy2_1], on='ids')
merged_xr_sf_all = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sliner_1], on='ids')
merged_xr_liner2_all = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[hliner_1], on='ids')



merged_xr_sy2_combo = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_sy2], on='ids')
merged_xr_sf_combo = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_sliner], on='ids')
merged_xr_liner2_combo = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_hliner], on='ids')

merged_xr_sy2_all_combo = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_sy2], on='ids')
merged_xr_sf_all_combo = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_sliner], on='ids')
merged_xr_liner2_all_combo = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[combo_hliner], on='ids')


merged_xr_sy2_all.to_csv(catfold+'merged_xr_sy2_all.csv')
merged_xr_liner2_all.to_csv(catfold+'merged_xr_hliner_all.csv')
merged_xr_sf_all.to_csv(catfold+'merged_xr_sliner_all.csv')
merged_xr_val_all.to_csv(catfold+'merged_xr_val_all.csv')
merged_xr_all.to_csv(catfold+'merged_xr_all.csv')

merged_xr_sy2_all_combo.to_csv(catfold+'merged_xr_sy2_all_combo.csv')
merged_xr_liner2_all_combo.to_csv(catfold+'merged_xr_hliner_all_combo.csv')
merged_xr_sf_all_combo.to_csv(catfold+'merged_xr_sliner_all_combo.csv')
merged_xr_val_all_combo.to_csv(catfold+'merged_xr_val_all_combo.csv')




merged_xr_val2 = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[val2], on='ids')
merged_xr_val2_all = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[val2], on='ids')

merged_xr_sy2_woo1 = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sy2_2], on='ids')
merged_xr_sf_woo1 = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sliner_2], on='ids')
merged_xr_liner2_woo1 = pd.merge(xmm3eldiagmed_xrfilt.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[hliner_2], on='ids')

merged_xr_sy2_all_woo1 = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sy2_2], on='ids')
merged_xr_sf_all_woo1 = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[sliner_2], on='ids')
merged_xr_liner2_all_woo1 = pd.merge(xmm3eldiagmed_xrfilt_all.bpt_EL_gsw_df, sfrm_gsw2.fullagn_df[cols_to_use].iloc[hliner_2], on='ids')


merged_xr_sy2_all_woo1.to_csv(catfold+'merged_xr_sy2_all_woo1.csv')
merged_xr_liner2_all_woo1.to_csv(catfold+'merged_xr_hliner_all_woo1.csv')
merged_xr_sf_all_woo1.to_csv(catfold+'merged_xr_sliner_all_woo1.csv')
merged_xr_val2_all.to_csv(catfold+'merged_xr_val_all_woo1.csv')


EL_3xmm_all.high_sn_o3_EL_gsw_df.to_csv(catfold+'high_sn_o3_xray_all_sample.csv')
EL_3xmm.high_sn_o3_EL_gsw_df.to_csv(catfold+'high_sn_o3_xray_sample.csv')
EL_4xmm_all.EL_gsw_df.to_csv(catfold+'x4_xray_all_sample.csv')
xmm4eldiagmed_xrfilt_xragn.EL_gsw_df.to_csv(catfold+'x4_xragn_all_sample.csv')

xmm3eldiagmed_xrfilt_all.bpt_sf_df.to_csv(catfold+'xragn_bptsf.csv')
xmm3eldiagmed_xrfilt_all_high_sn_o3.high_sn_o3_EL_gsw_df.to_csv(catfold+'xragn_high_sn_o3_sample.csv')
xmm3eldiagmed_xrfilt_xragn.EL_gsw_df.to_csv(catfold+'xragn_sample_no_sn_cuts.csv')

xmm3eldiagmed_xrfilt_unclass_p1.EL_gsw_df.to_csv(catfold+'xragn_sample_unclass_p1_cuts.csv')
xmm3eldiagmed_xrfilt_unclass_p2.EL_gsw_df.to_csv(catfold+'xragn_sample_unclass_p2_cuts.csv')


comm_covered_bptplsagn, comm_bptpls, comm_covered = np.intersect1d(EL_m2.bptplsagn, covered_gsw_x3, return_indices=True)
commdiffidsxmm3_sub, xmm3diffcomm_sub, xmm3gswdiffcomm_sub = commonpts1d(xmm3ids, gswids[covered_gsw_x3[comm_covered]])
#intersect_covered_, comm_first, comm_second = np.intersect1d(relevant_inds, covered_gsw, return_indices=True)
comm_covered_bptplsagn_val1, comm_bptpls_val1, comm_covered_val1 = np.intersect1d(np.array(sfrm_gsw2.fullagn_df.ids.iloc[val1]), 
                                                                                  gswids[covered_gsw_x3[comm_covered]], return_indices=True)
comm_covered_bptplsagn_sy2, comm_bptpls_sy2, comm_covered_sy2 = np.intersect1d(np.array(sfrm_gsw2.fullagn_df.ids.iloc[sy2_1]),
                                                                               gswids[covered_gsw_x3[comm_covered]],return_indices=True)
comm_covered_bptplsagn_liner2, comm_bptpls_liner2, comm_covered_liner2 = np.intersect1d(np.array(sfrm_gsw2.fullagn_df.ids.iloc[hliner_1]), 
                                                                                        gswids[covered_gsw_x3[comm_covered]], return_indices=True)
comm_covered_bptplsagn_sf, comm_bptpls_sf, comm_covered_sf = np.intersect1d(np.array(sfrm_gsw2.fullagn_df.ids.iloc[sliner_1]),
                                                                            gswids[covered_gsw_x3[comm_covered]], return_indices=True)


xmm3gdiff_sub_val1 =get_gdiffs_set(merged_xr_val.ids, np.array(sfrm_gsw2.fullagn_df.ids.iloc[val1]), comm_bptpls_val1,                                  
                                  merged_xr_val,sfrm_gsw2.fullagn_df.iloc[val1].copy(), ['mass', 'z', 'niiha_sub', 'oiiihb_sub'])

xmm3gdiff_sub_sy2 =get_gdiffs_set(merged_xr_sy2.ids, np.array(sfrm_gsw2.fullagn_df.ids.iloc[sy2_1]), comm_bptpls_sy2,                                  
                                  merged_xr_sy2,sfrm_gsw2.fullagn_df.iloc[sy2_1].copy(), ['mass', 'z', 'niiha_sub', 'oiiihb_sub'])

xmm3gdiff_sub_liner2 =get_gdiffs_set(merged_xr_liner2.ids, np.array(sfrm_gsw2.fullagn_df.ids.iloc[hliner_1]), comm_bptpls_liner2,                                  
                                  merged_xr_liner2,sfrm_gsw2.fullagn_df.iloc[hliner_1].copy(), ['mass', 'z', 'niiha_sub', 'oiiihb_sub'])
xmm3gdiff_sub_sf =get_gdiffs_set(merged_xr_sf.ids, np.array(sfrm_gsw2.fullagn_df.ids.iloc[sliner_1]), comm_bptpls_sf,                                  
                                  merged_xr_sf,sfrm_gsw2.fullagn_df.iloc[sliner_1].copy(), ['mass', 'z', 'niiha_sub', 'oiiihb_sub'])

#xmm3gdiff_set_mass_ssfr =get_gdiffs_set(xmm3ids, gswids, covered_gsw_x3,                                  
 #                                 xmm3eldiagmed_xrfilt.bpt_EL_gsw_df,EL_m2.bpt_EL_gsw_df, ['mass', 'z', 'ssfr'])


#e,a,b = np.intersect1d(EL_m2.bptplsagn[sfrm_gsw2.agn_ind_mapping], covered_gsw_x3, return_indices=True)
#xmm3gdiff_sub_set =get_gdiffs_set(xmm3ids[xmm3eldiagmed_xrfilt.bptplsagn], np.array(sfrm_gsw2.fullagn_df.ids),covered_gsw_x3[b],
#                                  merged_xr,sfrm_gsw2.fullagn_df, ['mass', 'z', 'niiha_sub','oiiihb_sub'])

''' 


xmm3gdiff_sub = Gdiffs(xmm3ids[xmm3eldiagmed_xrfilt.bptplsagn], gswids[EL_m2.bptplsagn],  merged_xr, EL_m2.bpt_EL_gsw_df,
                   ['mass', 'z', 'niiha', 'oiiihb'])

xmm3gdiff_sub_val1 = Gdiffs(xmm3ids[xmm3eldiagmed_xrfilt.bptplsagn], gswids[EL_m2.bptplsagn],  merged_xr_val, EL_m2.bpt_EL_gsw_df,
                   ['mass', 'z', 'niiha', 'oiiihb'])
xmm3gdiff_sub_sy2 = Gdiffs(xmm3ids[xmm3eldiagmed_xrfilt.bptplsagn], gswids[EL_m2.bptplsagn],  merged_xr_sy2, EL_m2.bpt_EL_gsw_df,
                   ['mass', 'z', 'niiha', 'oiiihb'])
xmm3gdiff_sub_liner2 = Gdiffs(xmm3ids[xmm3eldiagmed_xrfilt.bptplsagn], gswids[EL_m2.bptplsagn],  merged_xr_liner2, EL_m2.bpt_EL_gsw_df,
                   ['mass', 'z', 'niiha', 'oiiihb'])
xmm3gdiff_sub_sf = Gdiffs(xmm3ids[xmm3eldiagmed_xrfilt.bptplsagn], gswids[EL_m2.bptplsagn],  merged_xr_sf, EL_m2.bpt_EL_gsw_df,
                   ['mass', 'z', 'niiha', 'oiiihb'])






xmm3gdiff_sub_val1.get_filt(comm_bptpls_val1)
xmm3gdiff_sub_sy2.get_filt(comm_bptpls_sy2)
xmm3gdiff_sub_liner2.get_filt(comm_bptpls_liner2)
xmm3gdiff_sub_sf.get_filt(comm_bptpls_sf)


commdiffidsxmm3_sub_val1, xmm3diffcomm_sub, xmm3gswdiffcomm_sub = commonpts1d(xmm3ids, gswids[covered_gsw_x3[comm_covered]])

commdiffidsxmm3_sub_sy2, xmm3diffcomm_sub_sy2, xmm3gswdiffcomm_sub_sy2 = commonpts1d(merged_xr_sy2.ids, gswids[covered_gsw_x3[comm_covered]])
commdiffidsxmm3_sub_liner2, xmm3diffcomm_sub_liner2, xmm3gswdiffcomm_sub_liner2 = commonpts1d(merged_xr_liner2.ids, gswids[covered_gsw_x3[comm_covered]])
commdiffidsxmm3_sub_sf, xmm3diffcomm_sub_sf, xmm3gswdiffcomm_sub_sf = commonpts1d(merged_xr_sf.ids, gswids[covered_gsw_x3[comm_covered]])





xmm3gdiff_sub_sy2.nearbyx(xmm3gswdiffcomm_sub_sy2)
xmm3gdiff_sub_liner2.nearbyx(xmm3gswdiffcomm_sub_liner2)
xmm3gdiff_sub_sf.nearbyx(xmm3gswdiffcomm_sub_sf)

xmm3gdiff_sub.getdist_by_thresh(3.0)
binnum = 60
contaminations_xmm3_sub = xmm3gdiff_sub.xrgswfracs[:,binnum]

contaminations_xmm3_15_sub = xmm3gdiff_sub.xrgswfracs[:, 30]

contaminations_xmm3_2_sub = xmm3gdiff_sub.xrgswfracs[:, 40]
contaminations_xmm3_25_sub = xmm3gdiff_sub.xrgswfracs[:, 50]
contaminations_xmm3_3_sub = xmm3gdiff_sub.xrgswfracs[:, 60]
contaminations_xmm3_35_sub = xmm3gdiff_sub.xrgswfracs[:, 70]
contaminations_xmm3_4_sub = xmm3gdiff_sub.xrgswfracs[:, 80]
xmm3gdiff.interpdistgrid(11,11,50,method='linear')
'''



'''
d4000m_gsw2 = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df)

d4000m_gsw2.get_highsn_match_only_d4000(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True,  sncut_forb=2,sncut_balm=2, with_av=True)
d4000m_gsw2.subtract_elflux(sncut=2,  halphbeta_sncut=10, second=False)
d4000m_gsw2.bin_by_bpt(binsize=0.1)
d4000mhd_gsw2 = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df)
d4000mhd_gsw2.get_highsn_match_only_d4000_hdelta(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True, sncut_forb=2,sncut_balm=2, with_av=True, fname='second_d4_hd_1117')
d4000mhd_gsw2.subtract_elflux(sncut=2,  halphbeta_sncut=10)
d4000mhd_gsw2.bin_by_bpt(binsize=0.1)
d4000mhd_gsw2.get_filt_dfs(filts, gen_filts, match_filts, line_filts, line_filts_comb, loweroiiilum=40.0, upperoiiilum=40.2)
#d4000mhd_gsw2.bin_by_bpt(binsize=0.1)



d4000mhd_gsw2 = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df)
d4000mhd_gsw2.get_highsn_match_only_d4000_hdelta(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=False,  sncut_forb=2,sncut_balm=2, with_av=True, fname='second_d4_hd_0511_mixed_global_fiber')
d4000mhd_gsw2.subtract_elflux(sncut=2,  halphbeta_sncut=10)
#d4000mhd_gsw2.bin_by_bpt(binsize=0.1)
#d4000mhd_gsw2.get_filt_dfs(filts, gen_filts, match_filts, line_filts, line_filts_comb, loweroiiilum=40.0, upperoiiilum=40.2)




d4000mhd_gsw2 = SFRMatch(EL_m2,EL_m2.bpt_EL_gsw_df,
                     EL_m2.plus_EL_gsw_df, EL_m2.neither_EL_gsw_df)
d4000mhd_gsw2.get_highsn_match_only_d4000_hdelta(EL_m2.bptplsagn, EL_m2.bptplssf, 
                                EL_m2.bptplsnii_sf, EL_m2.bptplsnii_agn, 
                                load=True,  sncut=2, with_av=True, fname='second_d4_hd_1117')
d4000mhd_gsw2.subtract_elflux(sncut=2,  halphbeta_sncut=10)

lower_half_ms_sy2 = np.where((sfrm_gsw2.fullagn_df.delta_ssfr.iloc[sy2_1]<0 )&
                             (sfrm_gsw2.fullagn_df.delta_ssfr.iloc[sy2_1]>-0.7)&
                             (sfrm_gsw2.fullagn_df.mass.iloc[sy2_1]>10)
                             )[0]
lower_half_ms_hliner = np.where((sfrm_gsw2.fullagn_df.delta_ssfr.iloc[hliner_1]<0)&
                                (sfrm_gsw2.fullagn_df.delta_ssfr.iloc[hliner_1]>-0.7)&
                                (sfrm_gsw2.fullagn_df.mass.iloc[hliner_1]>10))[0]
lower_half_ms_sliner = np.where((sfrm_gsw2.fullagn_df.delta_ssfr.iloc[sliner_1]<0)&
                                (sfrm_gsw2.fullagn_df.delta_ssfr.iloc[sliner_1]>-0.7)&
                                (sfrm_gsw2.fullagn_df.mass.iloc[sliner_1]>10))[0]

lower_half_ms_nonagn = np.where((EL_m2.allnonagn_df.delta_ssfr<0)&
                                (EL_m2.allnonagn_df.delta_ssfr>-0.7)&
                                (EL_m2.allnonagn_df.mass>10))[0]

plt.hist(sfrm_gsw2.fullagn_df.d4000.iloc[sy2_1[lower_half_ms_sy2]],  histtype='step',label='Sy2', range=(1,2.4), bins=20, log=False)
plt.hist(sfrm_gsw2.fullagn_df.d4000.iloc[hliner_1[lower_half_ms_hliner]], histtype='step', label='H-LINER', range=(1,2.4), bins=20, log=False)
plt.hist(sfrm_gsw2.fullagn_df.d4000.iloc[sliner_1[lower_half_ms_sliner]], histtype='step', label='S-LINER', range=(1,2.4), bins=20, log=False)
plt.hist(EL_m2.allnonagn_df.d4000.iloc[lower_half_ms_nonagn], histtype='step', label='Non-AGN', range=(1,2.4), bins=20, log=False)
plt.legend()
plt.xlabel(r'D$_{4000}$')
plt.ylabel('Counts')
plt.tight_layout()


lower_half_ms_sy2 = np.where((d4000mhd_gsw2.fullagn_df.delta_ssfr.iloc[sy2_1_hd]<0 )&
                             (d4000mhd_gsw2.fullagn_df.delta_ssfr.iloc[sy2_1_hd]>-0.7)&
                             (d4000mhd_gsw2.fullagn_df.mass.iloc[sy2_1_hd]>10)
                             )[0]
lower_half_ms_hliner = np.where((d4000mhd_gsw2.fullagn_df.delta_ssfr.iloc[hliner_1_hd]<0)&
                                (d4000mhd_gsw2.fullagn_df.delta_ssfr.iloc[hliner_1_hd]>-0.7)&
                                (d4000mhd_gsw2.fullagn_df.mass.iloc[hliner_1_hd]>10))[0]
lower_half_ms_sliner = np.where((d4000mhd_gsw2.fullagn_df.delta_ssfr.iloc[sliner_1_hd]<0)&
                                (d4000mhd_gsw2.fullagn_df.delta_ssfr.iloc[sliner_1_hd]>-0.7)&
                                (d4000mhd_gsw2.fullagn_df.mass.iloc[sliner_1_hd]>10))[0]

lower_half_ms_nonagn = np.where((EL_m2.allnonagn_df.delta_ssfr<0)&
                                (EL_m2.allnonagn_df.delta_ssfr>-0.7)&
                                (EL_m2.allnonagn_df.mass>10))[0]

plt.hist(d4000mhd_gsw2.allagn_df.d4000.iloc[sy2_1_hd[lower_half_ms_sy2]],  histtype='step',label='Sy2', range=(1,2.4), bins=20, log=True)
plt.hist(d4000mhd_gsw2.allagn_df.d4000.iloc[hliner_1_hd[lower_half_ms_hliner]], histtype='step', label='H-LINER', range=(1,2.4), bins=20, log=True)
plt.hist(d4000mhd_gsw2.allagn_df.d4000.iloc[sliner_1_hd[lower_half_ms_sliner]], histtype='step', label='S-LINER', range=(1,2.4), bins=20, log=True)
plt.hist(EL_m2.allnonagn_df.d4000.iloc[lower_half_ms_nonagn], histtype='step', label='Non-AGN', range=(1,2.4), bins=20, log=True)
plt.legend()
plt.xlabel(r'D$_{4000}$')
plt.ylabel('Counts')
plt.tight_layout()

all_liners = np.append(sliner_1_hd[lower_half_ms_sliner], hliner_1_hd[lower_half_ms_hliner])
sy2_best_agn = []
sy2_best_nonagn = []
sy2_best_agn_dists = []
sy2_best_nonagn_dists = []

for i in range(len(sy2_1_hd[lower_half_ms_sy2])):
    sfr_diffs_agn = sfrm_gsw2.get_dists(d4000mhd_gsw2.fullagn_df.sfr.iloc[sy2_1_hd[lower_half_ms_sy2][i]], d4000mhd_gsw2.fullagn_df.sfr.iloc[all_liners])
    mass_diffs_agn = sfrm_gsw2.get_dists(d4000mhd_gsw2.fullagn_df.mass.iloc[sy2_1_hd[lower_half_ms_sy2][i]], d4000mhd_gsw2.fullagn_df.mass.iloc[all_liners])
    fibmass_diffs_agn = sfrm_gsw2.get_dists(d4000mhd_gsw2.fullagn_df.fibmass.iloc[sy2_1_hd[lower_half_ms_sy2][i]], d4000mhd_gsw2.fullagn_df.fibmass.iloc[all_liners])

    z_diffs_agn = sfrm_gsw2.get_dists(d4000mhd_gsw2.fullagn_df.z.iloc[sy2_1_hd[lower_half_ms_sy2][i]], d4000mhd_gsw2.fullagn_df.z.iloc[all_liners])/np.var(EL_m2.allnonagn_df.z)
    av_diffs_agn = sfrm_gsw2.get_dists(d4000mhd_gsw2.fullagn_df.av_gsw.iloc[sy2_1_hd[lower_half_ms_sy2][i]], d4000mhd_gsw2.fullagn_df.av_gsw.iloc[all_liners])

    combined_agn = np.array(np.sqrt(sfr_diffs_agn+mass_diffs_agn+fibmass_diffs_agn+z_diffs_agn+av_diffs_agn))
    
    agn_best = np.argmin(combined_agn)
    print(agn_best)
    sy2_best_agn_dists.append(combined_agn[agn_best])
    sy2_best_agn.append(agn_best)
    
    sfr_diffs_nonagn = sfrm_gsw2.get_dists(d4000mhd_gsw2.fullagn_df.sfr.iloc[sy2_1_hd[lower_half_ms_sy2][i]], EL_m2.allnonagn_df.sfr)
    mass_diffs_nonagn = sfrm_gsw2.get_dists(d4000mhd_gsw2.fullagn_df.mass.iloc[sy2_1_hd[lower_half_ms_sy2][i]], EL_m2.allnonagn_df.mass)
    fibmass_diffs_nonagn = sfrm_gsw2.get_dists(d4000mhd_gsw2.fullagn_df.fibmass.iloc[sy2_1_hd[lower_half_ms_sy2][i]], EL_m2.allnonagn_df.fibmass)
    z_diffs_nonagn = sfrm_gsw2.get_dists(d4000mhd_gsw2.fullagn_df.z.iloc[sy2_1_hd[lower_half_ms_sy2][i]],EL_m2.allnonagn_df.z)/np.var(EL_m2.allnonagn_df.z)

    av_diffs_nonagn = sfrm_gsw2.get_dists(d4000mhd_gsw2.fullagn_df.av_gsw.iloc[sy2_1_hd[lower_half_ms_sy2][i]], EL_m2.allnonagn_df.av_gsw)
    combined_nonagn = np.array(np.sqrt(sfr_diffs_nonagn+mass_diffs_nonagn+fibmass_diffs_nonagn+z_diffs_nonagn+av_diffs_nonagn))
    nonagn_best = np.argmin(combined_nonagn)
    sy2_best_nonagn_dists.append(combined_nonagn[nonagn_best])
    sy2_best_nonagn.append(nonagn_best)
    
    
    print(combined_nonagn[nonagn_best], combined_agn[agn_best])

plt.hist(sy2_best_nonagn_dists, bins=15, range=(0,0.3), histtype='step', label='Sy2 with Non-AGN Match')
plt.hist(sy2_best_agn_dists, bins=15, range=(0,0.3), histtype='step', label='Sy2 with LINER Match')
plt.xlabel(r'$d$')
plt.ylabel('Counts')
plt.legend()

plot2dhist(sfrm_gsw2.fullmatch_df.sfr.iloc[sy2_1],
           sfrm_gsw2.fullagn_df.sfr.iloc[sy2_1], 
           minx=-4, maxx=4, miny=-4, maxy=4, setplotlims=True, lim=True ,
           xlabel=r'log(SFR$_{\mathrm{Match}}$)',ylabel=r'log(SFR$_{\mathrm{AGN}}$)')
plt.plot([-4,4], [-4,4], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/sfrm_sfr.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/sfrm_sfr.png', format='png', bbox_inches='tight', dpi=250)


plot2dhist(d4000mhd_gsw2.allmatch_df.sfr.iloc[sy2_1_hd],
           d4000mhd_gsw2.allagn_df.sfr.iloc[sy2_1_hd], 
           minx=-4, maxx=4, miny=-4, maxy=4, setplotlims=True, lim=True ,
           xlabel=r'log(SFR$_{\mathrm{Match}}$)',ylabel=r'log(SFR$_{\mathrm{AGN}}$)')
plt.plot([-4,4], [-4,4], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/d4hd_sfr.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/d4hd_sfr.png', format='png', bbox_inches='tight', dpi=250)


plot2dhist(d4000m_gsw2.allmatch_df.sfr.iloc[d4000m_gsw2.sn2_filt_sy2],
           d4000m_gsw2.allagn_df.sfr.iloc[d4000m_gsw2.sn2_filt_sy2], 
           minx=-4, maxx=4, miny=-4, maxy=4, setplotlims=True, lim=True ,
           xlabel=r'log(SFR$_{\mathrm{Match}}$)',ylabel=r'log(SFR$_{\mathrm{AGN}}$)')
plt.plot([-4,4], [-4,4], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/d4_sfr.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/d4_sfr.png', format='png', bbox_inches='tight', dpi=250)



plot2dhist(sfrm_gsw2.fullmatch_df.mass.iloc[sy2_1],
           sfrm_gsw2.fullagn_df.mass.iloc[sy2_1], 
           minx=9, maxx=12, miny=9, maxy=12, setplotlims=True, lim=True ,
           xlabel=r'log(M$_{*,\mathrm{Match}})$',ylabel=r'log(M$_{*,\mathrm{AGN}})$')
plt.plot([9,12], [9,12], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/sfrm_mass.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/sfrm_mass.png', format='png', bbox_inches='tight', dpi=250)


plot2dhist(d4000mhd_gsw2.allmatch_df.mass.iloc[sy2_1_hd],
           d4000mhd_gsw2.allagn_df.mass.iloc[sy2_1_hd], 
           minx=9, maxx=12, miny=9, maxy=12, setplotlims=True, lim=True ,
           xlabel=r'log(M$_{*,\mathrm{Match}})$',ylabel=r'log(M$_{*,\mathrm{AGN}})$')
plt.plot([9,12], [9,12], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/d4hd_mass.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/d4hd_mass.png', format='png', bbox_inches='tight', dpi=250)


plot2dhist(d4000m_gsw2.allmatch_df.mass.iloc[d4000m_gsw2.sn2_filt_sy2],
           d4000m_gsw2.allagn_df.mass.iloc[d4000m_gsw2.sn2_filt_sy2], 
           minx=9, maxx=12, miny=9, maxy=12, setplotlims=True, lim=True ,
           xlabel=r'log(M$_{*,\mathrm{Match}})$',ylabel=r'log(M$_{*,\mathrm{AGN}})$')
plt.plot([9,12], [9,12], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/d4_mass.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/d4_mass.png', format='png', bbox_inches='tight', dpi=250)




plot2dhist(sfrm_gsw2.fullmatch_df.av_gsw.iloc[sy2_1],
           sfrm_gsw2.fullagn_df.av_gsw.iloc[sy2_1], 
           minx=0, maxx=1.5, miny=0, maxy=1.5, setplotlims=True, lim=True ,
           xlabel=r'A(V$_{*,\mathrm{Match}})$',ylabel=r'A(V$_{*,\mathrm{AGN}})$')
plt.plot([0,1.5], [0,1.5], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/sfrm_av_gsw.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/sfrm_av_gsw.png', format='png', bbox_inches='tight', dpi=250)


plot2dhist(d4000mhd_gsw2.allmatch_df.av_gsw.iloc[sy2_1_hd],
           d4000mhd_gsw2.allagn_df.av_gsw.iloc[sy2_1_hd], 
           minx=0, maxx=1.5, miny=0, maxy=1.5, setplotlims=True, lim=True ,
           xlabel=r'A(V$_{*,\mathrm{Match}})$',ylabel=r'A(V$_{*,\mathrm{AGN}})$')
plt.plot([0,1.5], [0,1.5], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/d4hd_av_gsw.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/d4hd_av_gsw.png', format='png', bbox_inches='tight', dpi=250)


plot2dhist(d4000m_gsw2.allmatch_df.av_gsw.iloc[d4000m_gsw2.sn2_filt_sy2],
           d4000m_gsw2.allagn_df.av_gsw.iloc[d4000m_gsw2.sn2_filt_sy2], 
           minx=0, maxx=1.5, miny=0, maxy=1.5, setplotlims=True, lim=True ,
           xlabel=r'A(V$_{*,\mathrm{Match}})$',ylabel=r'A(V$_{*,\mathrm{AGN}})$')
plt.plot([0,1.5], [0,1.5], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/d4_av_gsw.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/d4_av_gsw.png', format='png', bbox_inches='tight', dpi=250)







plot2dhist(sfrm_gsw2.fullmatch_df.sfr.iloc[sy2_1],
           sfrm_gsw2.fullagn_df.sfr.iloc[sy2_1], 
           minx=-4, maxx=4, miny=-4, maxy=4, setplotlims=True, lim=True ,
           xlabel=r'SFR$_{\mathrm{Match}}$',ylabel=r'SFR$_{\mathrm{AGN}}$')
plt.plot([-4,4], [-4,4], 'k-.')plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/sfrm_d4000.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/sfrm_d4000.png', format='png', bbox_inches='tight', dpi=250)


plot2dhist(d4000mhd_gsw2.allmatch_df.d4000.iloc[sy2_1_hd],
           d4000mhd_gsw2.allagn_df.d4000.iloc[sy2_1_hd], 
           minx=0.5, maxx=2.4, miny=0.5, maxy=2.4, setplotlims=True, lim=True ,
           xlabel=r'D$_{4000, \mathrm{Match}}$',ylabel=r'D$_{4000, \mathrm{AGN}}$')
plt.plot([0.5,2.4], [0.5,2.4], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/d4hd_d4000.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/d4hd_d4000.png', format='png', bbox_inches='tight', dpi=250)


plot2dhist(d4000m_gsw2.allmatch_df.d4000.iloc[d4000m_gsw2.sn2_filt_sy2],
           d4000m_gsw2.allagn_df.d4000.iloc[d4000m_gsw2.sn2_filt_sy2], 
           minx=0.5, maxx=2.4, miny=0.5, maxy=2.4, setplotlims=True, lim=True ,
           xlabel=r'D$_{4000, \mathrm{Match}}$',ylabel=r'D$_{4000, \mathrm{AGN}}$')
plt.plot([0.5,2.4], [0.5,2.4], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/d4_d4000.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/d4_d4000.png', format='png', bbox_inches='tight', dpi=250)




plot2dhist(sfrm_gsw2.fullmatch_df.hdelta_lick.iloc[sy2_1],
           sfrm_gsw2.fullagn_df.hdelta_lick.iloc[sy2_1], 
           minx=-10, maxx=10, miny=-10, maxy=10, setplotlims=True, lim=True ,
           xlabel=r'H$_{\delta \mathrm{Lick, Match}}$',ylabel=r'H$_{\delta \mathrm{Lick, AGN}}$')
plt.plot([-10,10], [-10,10], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/sfrm_hdelta.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/sfrm_hdelta.png', format='png', bbox_inches='tight', dpi=250)

plot2dhist(d4000mhd_gsw2.allmatch_df.hdelta_lick.iloc[sy2_1_hd],
           d4000mhd_gsw2.allagn_df.hdelta_lick.iloc[sy2_1_hd], 
           minx=-10, maxx=10, miny=-10, maxy=10, setplotlims=True, lim=True ,
           xlabel=r'H$_{\delta \mathrm{Lick, Match}}$',ylabel=r'H$_{\delta \mathrm{Lick, AGN}}$')
plt.plot([-10,10], [-10,10], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/d4hd_hdelta.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/d4hd_hdelta.png', format='png', bbox_inches='tight', dpi=250)

plot2dhist(d4000m_gsw2.allmatch_df.hdelta_lick.iloc[d4000m_gsw2.sn2_filt_sy2],
           d4000m_gsw2.allagn_df.hdelta_lick.iloc[d4000m_gsw2.sn2_filt_sy2], 
           minx=-10, maxx=10, miny=-10, maxy=10, setplotlims=True, lim=True ,
           xlabel=r'H$_{\delta \mathrm{Lick, Match}}$',ylabel=r'H$_{\delta \mathrm{Lick, AGN}}$')
plt.plot([-10,10], [-10,10], 'k-.')
plt.tight_layout()
plt.savefig('./plots/sfrmatch/pdf/diagnostic/d4_hdelta.pdf', format='pdf', bbox_inches='tight', dpi=250)
plt.savefig('./plots/sfrmatch/png/diagnostic/d4_hdelta.png', format='png', bbox_inches='tight', dpi=250)



#sandy mass cuts
mass_min=10.2
mass_max=10.4
nonagn_massfilt = np.where( (EL_m2.allnonagn_df.mass>mass_min)&(EL_m2.allnonagn_df.mass<mass_max)&(EL_m2.allnonagn_df.sigma1>0))[0]
match_sy2_massfilt = np.where( (sfrm_gsw2.fullmatch_df.mass.iloc[sy2_1]>mass_min)&
                              (sfrm_gsw2.fullmatch_df.mass.iloc[sy2_1]<mass_max) &
                              (sfrm_gsw2.fullagn_df.sigma1.iloc[sy2_1]>0)&
                              (sfrm_gsw2.fullmatch_df.sigma1.iloc[sy2_1]>0))[0]

sy2_massfilt = np.where( (sfrm_gsw2.fullagn_df.mass.iloc[sy2_1]>mass_min)&
                        (sfrm_gsw2.fullagn_df.mass.iloc[sy2_1]<mass_max)&
                        (sfrm_gsw2.fullagn_df.sigma1.iloc[sy2_1]>0)
                        #(sfrm_gsw2.fullmatch_df.sigma1.iloc[sy2_1]>0)
                        )[0]

hliner_massfilt = np.where( (sfrm_gsw2.fullagn_df.mass.iloc[hliner_1]>mass_min)&
                           (sfrm_gsw2.fullagn_df.mass.iloc[hliner_1]<mass_max)&
                           (sfrm_gsw2.fullagn_df.sigma1.iloc[hliner_1]>0)#&
                           #(sfrm_gsw2.fullmatch_df.sigma1.iloc[hliner_1]>0)
                           )[0]
sliner_massfilt = np.where( (sfrm_gsw2.fullagn_df.mass.iloc[sliner_1]>mass_min)&
                           (sfrm_gsw2.fullagn_df.mass.iloc[sliner_1]<mass_max)&
                           (sfrm_gsw2.fullagn_df.sigma1.iloc[sliner_1]>0)
                           #(sfrm_gsw2.fullmatch_df.sigma1.iloc[sliner_1]>0)
                           )[0]
both_liners=np.append(sliner_1, hliner_1)

liner_massfilt = np.where((sfrm_gsw2.fullagn_df.mass.iloc[both_liners] >mass_min)&
                          (sfrm_gsw2.fullagn_df.mass.iloc[both_liners]<mass_max)&#&
                          (sfrm_gsw2.fullagn_df.sigma1.iloc[both_liners] >0)
                          #(sfrm_gsw2.fullmatch_df.sigma1.iloc[both_liners]<mass_max)
                          )[0]

massfilt = np.where( (sfrm_gsw2.fullagn_df.mass.iloc[val1]>mass_min)&
                        (sfrm_gsw2.fullagn_df.mass.iloc[val1]<mass_max)&
                        (sfrm_gsw2.fullagn_df.sigma1.iloc[val1]>0)
                        #(sfrm_gsw2.fullmatch_df.sigma1>0)
                        )[0]

plot2dhist(EL_m2.allnonagn_df.sigma1.iloc[nonagn_massfilt], EL_m2.allnonagn_df.ssfr.iloc[nonagn_massfilt], 
           minx=7, maxx=11, maxy=-8, miny=-14,setplotlims=True, lim=True, nx=50,ny=50, xlabel=r'$\Sigma_{1}$', ylabel='log(sSFR)')
plt.gca().invert_yaxis()
plt.scatter(sfrm_gsw2.fullagn_df.sigma1.iloc[sy2_1[sy2_massfilt]], 
            sfrm_gsw2.fullagn_df.ssfr.iloc[sy2_1[sy2_massfilt]], s=7, c='r', label='Sy2')
plt.tight_layout()



plt.scatter(sfrm_gsw2.fullmatch_df.sigma1.iloc[sy2_1[match_sy2_massfilt]], 
            sfrm_gsw2.fullmatch_df.ssfr.iloc[sy2_1[match_sy2_massfilt]], s=7, facecolor='none', edgecolor='r')

plot2dhist(EL_m2.allnonagn_df.sigma1.iloc[nonagn_massfilt], EL_m2.allnonagn_df.ssfr.iloc[nonagn_massfilt], 
           minx=7, maxx=11, maxy=-8, miny=-14,setplotlims=True, lim=True, nx=50,ny=50)
plot2dhist(sfrm_gsw2.fullagn_df.sigma1.iloc[hliner_1[hliner_massfilt]], 
            sfrm_gsw2.fullagn_df.ssfr.iloc[hliner_1[hliner_massfilt]], minx=7, maxx=11, maxy=-8, miny=-14,setplotlims=True, lim=True, nx=50,ny=50)
plot2dhist(sfrm_gsw2.fullagn_df.sigma1.iloc[np.append(sliner_1, hliner_1)[liner_massfilt]], 
            sfrm_gsw2.fullagn_df.ssfr.iloc[np.append(sliner_1, hliner_1)[liner_massfilt]], minx=7, maxx=11, maxy=-8, miny=-14,setplotlims=True, lim=True, nx=50,ny=50)

plot2dhist(EL_m2.allnonagn_df.sigma1.iloc[nonagn_massfilt], EL_m2.allnonagn_df.ssfr.iloc[nonagn_massfilt], 
           minx=7, maxx=11, maxy=-8, miny=-14,setplotlims=True, lim=True, nx=50,ny=50)
plt.scatter(sfrm_gsw2.fullagn_df.sigma1.iloc[sy2_1[sy2_massfilt]], 
            sfrm_gsw2.fullagn_df.ssfr.iloc[sy2_1[sy2_massfilt]], s=7, facecolor='none', edgecolor='r')


'''

#%% 
def match_xrgal_to_sfrm_obj(xrids, sfrmids):
    matching_ind_bpt = []
    xrind_bpt = []
    matching_ind_plus = []
    xrind_plus = []
    matching_ind_neither = []
    xrind_neither = []
    for i, xrid in enumerate(xrids):
        match = np.int64(np.where(sfrmids == xrid)[0][0])
        if match <sfrm_gsw2.agns.size:
            matching_ind_bpt.append(match)
            xrind_bpt.append(i)
        elif match <sfrm_gsw2.agns.size+sfrm_gsw2.agns_plus.size:
            matching_ind_plus.append(match-sfrm_gsw2.agns.size)
            xrind_plus.append(i)
        else:
            matching_ind_neither.append(match-sfrm_gsw2.agns.size+sfrm_gsw2.agns_plus.size)
            xrind_neither.append(i)
    return np.array(matching_ind_bpt), np.array(xrind_bpt), np.array(matching_ind_plus), np.array(xrind_plus), np.array(matching_ind_neither), np.array(xrind_neither)

#xmm3all_xr_to_sfrm_bpt, xmm3all_xrind_bpt, xmm3all_xr_to_sfrm_plus,xmm3all_xrind_plus, xmm3all_xr_to_sfrm_neither, xmm3all_xrind_neither = match_xrgal_to_sfrm_obj(xmm3eldiagmed_xrfilt_all.ids[agn_3xmmmbptplus_all_xrfilt_bptplus], sfrm_gsw2.sdssids_sing)
#xmm3_xr_to_sfrm_bpt,xmm3_xrind_bpt, xmm3_xr_to_sfrm_plus,xmm3_xrind_plus, xmm3_xr_to_sfrm_neither, xmm3_xrind_neither = match_xrgal_to_sfrm_obj(xmm3eldiagmed_xrfiltbptplus.ids[agn_3xmmm_xrfilt_bptplus], sfrm_gsw2.sdssids_sing)

#sfrm_gsw2.bin_by_bpt(binsize=0.1)


'''
for creating target files
'''
def write_to_fil_obs(filnam):
    f = open(filnam,'w+')
    #f.write("#Name, OBJID, RA, DEC, BPT CAT\n")
    f.write("#Name, OBJID, RA, DEC, Exp time, SN Min, SN Min Line, SN Max, SN Max Line, z\n")

    sn =  np.vstack([xmm3eldiagmed_xrfilt.halp_sn,xmm3eldiagmed_xrfilt.nii_sn,                   xmm3eldiagmed_xrfilt.oiii_sn,xmm3eldiagmed_xrfilt.hbeta_sn])
    snmin = np.min(sn, axis= 0)
    snmax = np.max(sn, axis= 0)
    sncodemin = matchcode(sn, snmin)
    sncodemax = matchcode(sn, snmax)

    filtxmm = mpa_spec_m2_3xmm.make_prac[EL_3xmm.bpt_sn_filt][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr][nonagn_3xmmm_xrfilt]

    for i in range(len(filtxmm)):
        f.write(str(m2Cat_GSW_3xmm.matchra[filtxmm][i])+','+str(m2Cat_GSW_3xmm.matchdec[filtxmm][i])+','+str(m2Cat_GSW_3xmm.matchmjd[filtxmm][i])+ ','+str(m2Cat_GSW_3xmm.matchfiber[filtxmm][i])+','+str(m2Cat_GSW_3xmm.matchplate[filtxmm][i])+'\n')
        #f.write('3XMMHII'+'-'+str(i)+','+str(m2Cat_GSW_3xmm.ids[filtxmm][i])+','+                    str(m2Cat_GSW_3xmm.matchra[filtxmm][i])+','+str(m2Cat_GSW_3xmm.matchdec[filtxmm][i])+
        #                      ','+str(m2Cat_GSW_3xmm.exptimes[filtxmm][i])+',' +               str(snmin[nonagn_3xmmm_xrfilt][i])+','+sncodemin[nonagn_3xmmm_xrfilt][i]+','+str(snmax[nonagn_3xmmm_xrfilt][i])+','+ sncodemax[nonagn_3xmmm_xrfilt][i]+','+str(xmm3eldiagmed_xrfilt.z[nonagn_3xmmm_xrfilt][i])+'\n')

    f.close()
#write_to_fil_obs('sdssquerytable.txt')

codes = {0:'halp', 1:'nii', 2:'oiii', 3:'hbeta'}
def matchcode(sn,sndes):
    code = []
    for i in range(len(sndes)):
        whermin = np.where(sn[:,i] == sndes[i])[0]
        code.append(codes[whermin[0]])
    return np.array(code)

def write_to_fil_obs_supp(filnam):
    f = open(filnam,'w+')
    f.write("#Name, OBJID, RA, DEC, Exp time, SN Min, SN Min Line, SN Max, SN Max Line, z, ssfr \n")
    filtxmm = mpa_spec_m2_3xmm_all.make_prac[EL_3xmm_all.bpt_sn_filt][fullxray_xmm_all.valid][fullxray_xmm_all.likelyagn_xr][nonagn_3xmmm_all_xrfilt]
    sn =  np.vstack([xmm3eldiagmed_xrfilt_all.halp_sn,xmm3eldiagmed_xrfilt_all.nii_sn,xmm3eldiagmed_xrfilt_all.oiii_sn,xmm3eldiagmed_xrfilt_all.hbeta_sn])
    snmin = np.min(sn, axis= 0)
    snmax = np.max(sn, axis= 0)
    sncodemin = matchcode(sn, snmin)
    sncodemax = matchcode(sn, snmax)

    for i in range(len(filtxmm)):
        print(m2Cat_GSW_3xmm_all.matchra[filtxmm][i])
        f.write('3XMMHII'+'-'+str(i)+','+str(m2Cat_GSW_3xmm_all.ids[filtxmm][i])+','+str(m2Cat_GSW_3xmm_all.matchra[filtxmm][i])+','+str(m2Cat_GSW_3xmm_all.matchdec[filtxmm][i])+
                ','+str(m2Cat_GSW_3xmm_all.exptimes[filtxmm][i])+','+str(snmin[nonagn_3xmmm_all_xrfilt][i])+','+
                sncodemin[nonagn_3xmmm_all_xrfilt][i]+','+str(snmax[nonagn_3xmmm_all_xrfilt][i])+','+
                sncodemax[nonagn_3xmmm_all_xrfilt][i]+','+str(xmm3eldiagmed_xrfilt_all.z[nonagn_3xmmm_all_xrfilt][i])+
                ','+str(xmm3eldiagmed_xrfilt_all.ssfr[nonagn_3xmmm_all_xrfilt][i])+'\n')

    f.close()
#write_to_fil_obs_supp('observing_sample_supp_SP19.txt')
