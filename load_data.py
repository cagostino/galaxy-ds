#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Fits_set import *
catfold='catalogs/'
import numpy as np
import pandas as pd
from ast_utils import getfluxfromlum, getlumfromflux, dustcorrect, extinction
from XMM3_obj import *

m2_df, a2_df, x2_df = load_catalogs(catfold)

print('GSW loaded')



m2 = np.loadtxt(catfold+"GSWLC-M2.dat", unpack = True, usecols=(0,1), dtype=np.int64)
galinfo  = Fits_set(catfold+'gal_info_dr7_v5_2.fit')
galline = Fits_set(catfold+'gal_line_dr7_v5_2.fit')
galfiboh = Fits_set(catfold+'gal_fiboh_dr7_v5_2.fits')
galindx = Fits_set(catfold+'gal_indx_dr7_v5_2.fit')
galmass = Fits_set(catfold+'totlgm_dr7_v5_2b.fit')
fibmass = Fits_set(catfold+'fiblgm_dr7_v5_2.fit')
fibsfr = Fits_set(catfold+'gal_fibsfr_dr7_v5_2.fits')
fibssfr = Fits_set(catfold+'gal_fibspecsfr_dr7_v5_2.fits')
'''
Load info from X-ray Surveys
'''
chand =Fits_set(catfold+'Chandra_multiwavelength.fits')
xmm = Fits_set(catfold+'XMM_multiwavelength_cat.fits')
#xmmfu = fits_set(catfold+'XMM_followup.fits')
xmmfu= Fits_set(catfold+'XMM_followup_all.fits')


# Create a DataFrame
sdssobj = pd.DataFrame(data_dict)


from Fits_set import XMM, XMM3obs, XMM4obs, CSC, FIRST
x3 = XMM(catfold+'3xmm.fits')
xmm3obs = XMM3obs(catfold+'3xmmobs.fits')


x4 = XMM(catfold+'4xmm.fits')
xmm4obs = XMM4obs(catfold+'4xmmobs.tsv')

#csc = Fits_set(catfold+'csc.fits')
#csc_sdss = Fits_set(catfold+'csc_sdss.fits')

merged_csc= CSC(catfold+'merged_csc.csv')

first = FIRST(catfold+'FIRST.fits')





sterntab1 = Fits_set(catfold+'sterntab.fits')
match_to_gsw = np.load(catfold+'match_gsw_to_stern.txt.npy', allow_pickle=True)
match_stern = np.load(catfold+'match_stern_to_gsw.txt.npy',  allow_pickle=True)
match_to_xmm = np.load(catfold+'match_xmm_to_stern.npy', allow_pickle=True)
match_x_stern = np.load(catfold+'match_stern_to_xmm.npy',  allow_pickle=True)
match_liu_to_xmm = np.load(catfold+'match_xmm_to_liu.npy', allow_pickle=True)
match_x_liu = np.load(catfold+'match_x_liu.npy', allow_pickle=True)

sternspecz = np.loadtxt(catfold+'stern_z_caug37.csv', skiprows=1, delimiter=',', unpack=True)
sternspecz_ids = np.loadtxt(catfold+'stern_z_caug37.csv', skiprows=1,usecols=(6), dtype=np.int64, delimiter=',', unpack=True)

sternspec_pass = np.load(catfold+'sternspec_pass.npy', allow_pickle=True)
'''
sternspec_pass = []
for i in range(len(sterntab1.getcol('_RA'))):
    radiff = np.abs(sterntab1.getcol('_RA')[i] - sternspecz[3])
    decdiff = np.abs(sterntab1.getcol('_DE')[i] - sternspecz[4])
    if np.min(radiff) <arcsec and np.min(decdiff) <arcsec:
        sternspec_pass.append(i)
    
'''

sternobj = {}

sternobj = {
    'lbha' : sterntab1.data['logLbHa'],
    'logM' : sterntab1.data['logM_'],
    'luv' : sterntab1.data['logLUV'],
    'alpha' : sterntab1.data['alpha'],
    'lha' : sterntab1.data['logLHa'],
    'e_lha': sterntab1.data['l_logLHa'],
    'lhb' : sterntab1.data['logLHb'],
    'e_lhb' : sterntab1.data['l_logLHB'],
    'e_lo3' : sterntab1.data['l_logL_OIII_'],
    'lo3' : sterntab1.data['logL_OIII_'],
    'ln2' : sterntab1.data['logL_NII_'],    
    'e_ln2' : sterntab1.data['l_logL_NII_'],    
    'robust' : sterntab1.data['fs'],
    'abs' : sterntab1.data['fa']
    
    
    
}  
#sternobj['sdssids'] = m2[0][match_to_gsw]
sternobj_df = pd.DataFrame(sternobj)
sternobj_df_spec = sternobj_df.iloc[sternspec_pass].copy()
sternobj_df_spec['ids'] =sternspecz_ids
sternobj_df_spec['z'] = sternspecz[0]

sternobj_df_spec_xr = sternobj_df_spec.iloc[match_x_stern].copy()

sternobj_m2_df = sternobj_df.iloc[match_stern]
sternobj_m2_df.loc[:, 'ids'] = m2[0][match_to_gsw]



liu_basic = Fits_set(catfold+'liu_basic.fits')

liu_spec = Fits_set(catfold+'liu_qsos_spec.fits')
liu_mw = Fits_set(catfold+'liu_qsos_multiwavelength.fits')















print('matching catalogs')
x3 = XMM3obs(xmm3,xmm3obs)
x4 = XMM4obs(xmm4,xmm4obs)

#coordchand = SkyCoord(ra95*u.degree, dec95*u.degree)

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

csc_cat = CSC(merged_csc)

comm_csc_gsw,  gsw_csc, csc_gsw= np.intersect1d(  m2[0], merged_csc.SDSSDR15, return_indices=True)

csc_cat.gswmatch_inds = gsw_csc

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
liuobj_df['fo1_corr'] = dustcorrect(liuobj_df['fo1'], liuobj_df['av_balm'], 6001.0)
liuobj_df['lo1_corr'] = np.log10(getlumfromflux(liuobj_df['fo1_corr'], liuobj_df['z']))

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








mpa_spec_m2_4xmm = MPAJHU_Spec(m2Cat_GSW_4xmm, sdssobj)


mpa_spec_qsos = MPAJHU_Spec(m2Cat_GSW_qsos, sdssobj, sedtyp=1)

mpa_spec_m2_4xmm_all = MPAJHU_Spec(m2Cat_GSW_4xmm_all, sdssobj)

mpa_spec_m2_4xmm_all_qsos = MPAJHU_Spec(m2Cat_GSW_4xmm_all_qsos, sdssobj, sedtyp=1)


mpa_spec_allm2 = MPAJHU_Spec(m2Cat_GSW, sdssobj, find=False, gsw=True)

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

mpa_spec_qsos.spec_inds_prac = np.int64(mpa_spec_qsos.spec_inds_prac).reshape(-1)
mpa_spec_qsos.make_prac = np.int64(mpa_spec_qsos.make_prac).reshape(-1)

mpa_spec_m2_4xmm.spec_inds_prac = np.int64(mpa_spec_m2_4xmm.spec_inds_prac ).reshape(-1)
mpa_spec_m2_4xmm_all.spec_inds_prac  = np.int64(mpa_spec_m2_4xmm_all.spec_inds_prac ).reshape(-1)
mpa_spec_m2_4xmm_all_qsos.spec_inds_prac  = np.int64(mpa_spec_m2_4xmm_all_qsos.spec_inds_prac ).reshape(-1)

mpa_spec_m2_4xmm.make_prac = np.int64(mpa_spec_m2_4xmm.make_prac ).reshape(-1)
mpa_spec_m2_4xmm_all.make_prac  = np.int64(mpa_spec_m2_4xmm_all.make_prac ).reshape(-1)
mpa_spec_m2_4xmm_all_qsos.make_prac  = np.int64(mpa_spec_m2_4xmm_all_qsos.make_prac ).reshape(-1)


mpa_spec_allm2_first.spec_inds_prac = np.int64(mpa_spec_allm2_first.spec_inds_prac ).reshape(-1)
mpa_spec_allm2_first.make_prac = np.int64(mpa_spec_allm2_first.make_prac ).reshape(-1)
mpa_spec_allm2_first.spec_plates_prac = np.int64(mpa_spec_allm2_first.spec_plates_prac ).reshape(-1)
mpa_spec_allm2_first.spec_fibers_prac = np.int64(mpa_spec_allm2_first.spec_fibers_prac ).reshape(-1)
