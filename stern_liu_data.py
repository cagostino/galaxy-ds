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
sternspec_pass = []
for i in range(len(sterntab1.getcol('_RA'))):
    radiff = np.abs(sterntab1.getcol('_RA')[i] - sternspecz[3])
    decdiff = np.abs(sterntab1.getcol('_DE')[i] - sternspecz[4])
    if np.min(radiff) <arcsec and np.min(decdiff) <arcsec:
        sternspec_pass.append(i)
    
'''

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
