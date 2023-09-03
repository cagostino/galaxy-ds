#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:22:09 2022

@author: cjagosti
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle
sys.path.append("../..")
#from Fits_set import *
#from ast_func import *
#from ELObj import *

EL_m2 = pd.read_csv('../../EL_m2_df.csv')            
bpt_EL_gsw_df = pd.read_csv('../../EL_m2_bpt_EL_gsw_df.csv')            

EL_4xmm = pd.read_csv('../../EL_4xmm_df.csv')
EL_4xmm_all = pd.read_csv('../../EL_4xmm_all_df.csv')

not_bpt_EL_gsw_df = pd.read_csv('../../EL_4xmm_not_bpt_EL_gsw_df.csv')
bpt_sf_df = pd.read_csv('../../EL_4xmm_bpt_sf_df.csv')


muse_c = ['g','b','gray', 'r', 'k']
muse_sym = ['o','s','>','d', 'v']
redshifts=[]

names = np.array(['SF-1','WL-3','WL-2','WL-EXT-1', 'WL-1'])


xu = not_bpt_EL_gsw_df.iloc[[25,26, 143, 281]]
xu1  = not_bpt_EL_gsw_df.iloc[25]
xu2 = not_bpt_EL_gsw_df.iloc[26]
xu3 = not_bpt_EL_gsw_df.iloc[143]
xu4 = not_bpt_EL_gsw_df.iloc[281]

xr  = bpt_sf_df.iloc[112]
muse_samp = {'SF-1':xr,'WL-3': xu1,'WL-2': xu2,'WL-EXT-1': xu3, 'WL-1':xu4}

#EL_m2 = np.load('../../EL_m2.npy', allow_pickle=True)
muse_cubes = {'SF-1':xr31, 'WL-3':xu22, 'WL-2':xu23, 'WL-EXT-1':xu104,'WL-1':xu210}


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
lsfrrelat = {'soft': [r'SFR/M$_{*} = 1.39\cdot 10^{-40}$ L$_{\rm x}/$M$_{*}$', r'SFR = $1.39\cdot 10^{-40}$ L$_{\rm x}$',logsfrsoft],
             'hard': [r'SFR/M$_{*} = 1.26\cdot 10^{-40} $L$_{\rm x}$/M$_{*}$', r'SFR = $1.26\cdot 10^{-40}$ L$_{\rm x}$',logsfrhard],
             'full': [r'SFR/M$_{*} = 0.66\cdot 10^{-40}$ L$_{\rm x}$/M$_{*}$',r'SFR = $0.66\cdot 10^{-40}$ L$_{\rm x}$', logsfrfull]  }



'''


dob_xu22= get_dobby_spec(xu22.sdss.starlight_res, xu22.sdss.resampled_flux_err, xu22.sdss.resampled_flag, xu22.sdss.starlight_fsyn, lamb=xu22.sdss.resampled_wl_starlight, wdisp = xu22.sdss.resampled_wdisp)
dob_out_xu22 = get_bpt_from_dobby(dob_xu22)

dob_xr31= get_dobby_spec(xr31.sdss.starlight_res, xr31.sdss.resampled_flux_err, xr31.sdss.resampled_flag, xr31.sdss.starlight_fsyn, lamb=xr31.sdss.resampled_wl_starlight, wdisp = xr31.sdss.resampled_wdisp)
dob_out_xr31 = get_bpt_from_dobby(dob_xr31)

dob_xu23= get_dobby_spec(xu23.sdss.starlight_res, xu23.sdss.resampled_flux_err, xu23.sdss.resampled_flag, xu23.sdss.starlight_fsyn, lamb=xu23.sdss.resampled_wl_starlight, wdisp = xu23.sdss.resampled_wdisp)
dob_out_xu23 = get_bpt_from_dobby(dob_xu23)

dob_xu104= get_dobby_spec(xu104.sdss.starlight_res, xu104.sdss.resampled_flux_err, xu104.sdss.resampled_flag, xu104.sdss.starlight_fsyn, lamb=xu104.sdss.resampled_wl_starlight, wdisp = xu104.sdss.resampled_wdisp)
dob_out_xu104 = get_bpt_from_dobby(dob_xu104)

dob_xu210= get_dobby_spec(xu210.sdss.starlight_res, xu210.sdss.resampled_flux_err, xu210.sdss.resampled_flag, xu210.sdss.starlight_fsyn, lamb=xu210.sdss.resampled_wl_starlight, wdisp = xu210.sdss.resampled_wdisp)
dob_out_xu210 = get_bpt_from_dobby(dob_xu210)


lst_dob_sdss = [dob_out_xr31, dob_out_xu22, dob_out_xu23, dob_out_xu104, dob_out_xu210]


xu = not_bpt_EL_gsw_df.iloc[[25,26, 143, 281]]
xu1  = not_bpt_EL_gsw_df.iloc[25]
xu2 = not_bpt_EL_gsw_df.iloc[26]
xu3 = not_bpt_EL_gsw_df.iloc[143]
xu4 = not_bpt_EL_gsw_df.iloc[281]

xr  = bpt_sf_df.iloc[112]
muse_samp = {'SF-1':xr,'WL-3': xu1,'WL-2': xu2,'WL-EXT-1': xu3, 'WL-1':xu4}
sampkeys = list(muse_samp.keys())
for i, dob in enumerate(lst_dob_sdss):
    print(sampkeys[i])
    print(dob[0][2])
    print(dob[1][2])
    print(dob[2][2])
    print(dob[3][2])


[24,25,215]
muse_cubes = {'SF-1':xr31, 'WL-3':xu22, 'WL-2':xu23, 'WL-EXT-1':xu104,'WL-1':xu210}

fig = plt.figure(figsize=(4, 7))
spec = fig.add_gridspec(ncols=1, nrows=3)
muse_keys =['SF-1','WL-1', 'WL-2']
new_names = ['SF-1', 'WL-1', 'WL-2']
for i, obj in enumerate(muse_keys):
    print(i)
    o3hbs = []
    n2has = []
    n2sns = []
    hasns = []
    o3sns = []
    hbsns = []

    for dob in muse_cubes[obj].aperture_dobs:
        o3hbs.append(dob.o3hb)
        n2has.append(dob.n2ha)
        n2sns.append(dob.n2_sn)
        hasns.append(dob.ha_sn)
        hbsns.append(dob.hb_sn)
        o3sns.append(dob.o3_sn)
    ximg = muse_cubes[obj].dobby
    o3hb_copy = np.copy(ximg.o3hb)
    n2ha_copy = np.copy(ximg.n2ha)
    
    
    low_sn_o3_mask = ximg.o3_sn<2
    
    low_sn_n2_mask = ximg.n2_sn<2
    low_sn_ha_mask = ximg.ha_sn<2
    low_sn_hb_mask = ximg.hb_sn<2
    
    low_sn_n2ha = (low_sn_ha_mask | low_sn_n2_mask)
    low_sn_o3hb  = (low_sn_hb_mask | low_sn_o3_mask)

    
    low_sn_bpt = (low_sn_n2ha | low_sn_o3hb)
    n2ha_copy[low_sn_bpt]=np.nan
    o3hb_copy[low_sn_bpt]=np.nan
    

    #ax1 = fig.add_subplot(spec[0,i])
    counts,ybins,xbins = np.histogram2d(bpt_EL_gsw_df.niiha, bpt_EL_gsw_df.oiiihb, bins=(50,50))
    
    
    #plotbptnormal(n2ha_copy.flatten(), o3hb_copy.flatten(),nx=50, ny=50,lim=True, nobj=False, mod_kauff=True, fig = fig, ax=ax1)
    
    
    ax2 = fig.add_subplot(spec[i])
    #counts = scipy.ndimage.zoom(counts, 3)

    plotbptnormal(bpt_EL_gsw_df.niiha, bpt_EL_gsw_df.oiiihb, mod_kauff=True,maxx=1.1, minx=-2,maxy=1.3, miny=-1.2, setplotlims=True,nobj=False, fig=fig, ax=ax2)
    #ax2.contour(counts.transpose(),extent=[ybins.min(),ybins.max(),xbins.min(),xbins.max()],linewidths=1,levels=[50, 200, 1000], cmap='plasma')
    #ax1.set_yticks([-0.5, 0,0.5, 1])
    ax2.set_yticks([-1, 0, 1])
    print(i)
    if i!=2:# i <3:
        #ax1.set_xticklabels('')
        #ax1.set_xlabel('')
        #ax1.set_yticklabels('')
        #ax1.set_ylabel('')
        
        ax2.set_xticklabels('')
        ax2.set_xlabel('')
    else:
        #ax1.set_ylabel(r'log([OIII]/H$\beta$)')
        ax2.set_ylabel(r'log([OIII]/H$\beta$)')
        ax2.set_xlabel(r'log([NII]/H$\alpha$)')
        
    #ax1.set_xticklabels('')
    #ax1.set_xlabel('')
    if i==2:
        ax2.scatter(muse_samp[obj].niiha, -1.15, marker='x', color='r', s=50)
    else:
        print(obj)
        ax2.scatter(muse_samp[obj].niiha, muse_samp[obj].oiiihb, marker='x', color='r', s=100)

    ax2.text(-1.8, -1., new_names[i], fontsize=15)

    for k in range(len(o3hbs)):
        if n2sns[k] >2 and hasns[k]>2 and hbsns[k]>2 and o3sns[k]>2:
            ax2.scatter(n2has[k], o3hbs[k], s=(k+1)*10, marker='o', edgecolor='orange',facecolor='none', zorder=10-k)
        elif n2sns[k]>2 and hasns[k]>2:
            ok_o3_hb = np.where((np.array(o3sns)>2 )&(np.array(hbsns)>2))[0] 
            if len(ok_o3_hb)!=0:                            
                ax2.scatter(n2has[k], o3hbs[ok_o3_hb[0]], s=(k+1)*10, marker='o',edgecolor='r',facecolor='none',zorder=10-k)
            else:
                ax2.scatter(n2has[k], -0.9, s=(k+1)*10, marker='o',edgecolor='r',facecolor='none', zorder=10-k)

    
    
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plots/bpt_diag_apertures.pdf', dpi=250, format='pdf', bbox_inches='tight')
plt.savefig('plots/bpt_diag_apertures.png', dpi=250, format='png', bbox_inches='tight')
plt.close()  
    
obj = 'XR31'
fluxes_o3 = []
fluxes_n2 = []
fluxes_ha = []
fluxes_hb = []

sn_o3 = []
sn_n2 = []
sn_ha = []
sn_hb = []

for i,obj in enumerate(muse_cubes.keys()):
    xobj = muse_cubes[obj]

    sdss_spec, sdss_dob = xobj.get_single_dobby_fit(xobj.sdss.starlight_res*1e17, 
                                          xobj.sdss.resampled_flux_err, 
                                          xobj.sdss.resampled_flag, xobj.sdss.starlight_fsyn*1e17,
                                          wdisp = xobj.sdss.resampled_wdisp, 
                                          lamb=xobj.sdss.resampled_wl_starlight, 
                                          plot=False, save_plot=False, outfile='sdss_'+obj)
    fluxes_o3.append([ sdss_dob.o3_flux, muse_samp[i].oiiiflux*1e17,
                           xobj.spec_sub_3arc_dob.o3_flux,xobj.spec_sub_05arc_dob.o3_flux])
                           
    fluxes_n2.append([sdss_dob.n2_flux, muse_samp[i].niiflux*1e17,xobj.spec_sub_3arc_dob.n2_flux,
                     xobj.spec_sub_05arc_dob.n2_flux])
    fluxes_ha.append([sdss_dob.ha_flux, muse_samp[i].halpflux*1e17,xobj.spec_sub_3arc_dob.ha_flux,
                      xobj.spec_sub_05arc_dob.ha_flux])

    fluxes_hb.append([sdss_dob.hb_flux, muse_samp[i].hbetaflux*1e17,
                               xobj.spec_sub_3arc_dob.hb_flux,
                               xobj.spec_sub_05arc_dob.hb_flux])
    
    sn_o3.append([ sdss_dob.o3_sn,  muse_samp[i].oiiiflux_sn,
                  xobj.spec_sub_3arc_dob.o3_sn,xobj.spec_sub_05arc_dob.o3_sn])
    sn_n2.append([sdss_dob.n2_sn, muse_samp[i].niiflux_sn,
                  xobj.spec_sub_3arc_dob.n2_sn,xobj.spec_sub_05arc_dob.n2_sn ])
    sn_ha.append([sdss_dob.ha_sn, muse_samp[i].halpflux_sn, 
                  xobj.spec_sub_3arc_dob.ha_sn,xobj.spec_sub_05arc_dob.ha_sn ])
    sn_hb.append([sdss_dob.hb_sn,muse_samp[i].hbetaflux_sn,
                  xobj.spec_sub_3arc_dob.hb_sn,
                  xobj.spec_sub_05arc_dob.hb_sn   ])
    
plotbptnormal([], [],nx=20, ny=20, kewley=True, mod_kauff=False, nobj=False)
plt.scatter(xragn_bpt_sf_df.niiha.iloc[28], xragn_bpt_sf_df.oiiihb.iloc[28], marker='x', label='MPA SDSS')

plt.scatter(sdss_dob.n2ha, sdss_dob.o3hb, marker='^', label='Dobby SDSS')

plt.scatter(xr31.spec_sub_3arc_dob.n2ha, xr31.spec_sub_3arc_dob.o3hb, marker='^', label="Dobby 3'' MUSE")
plt.legend()

plt.savefig('plots/'+obj+'_bpt_lr_comparison_3arc.pdf', dpi=250, format='pdf', bbox_inches='tight')
plt.savefig('plots/'+obj+'_bpt_lr_comparison_3arc.png', dpi=250, format='png', bbox_inches='tight')
plt.close()


for obj in muse_cubes.keys():
    
    xobj = muse_cubes[obj]
      
    
   


    
    sdss_spec, sdss_dob = xobj.get_single_dobby_fit(xobj.sdss.starlight_res*1e17, 
                                          xobj.sdss.resampled_flux_err, 
                                          xobj.sdss.resampled_flag, xobj.sdss.starlight_fsyn*1e17,
                                          wdisp = xobj.sdss.resampled_wdisp, 
                                          lamb=xobj.sdss.resampled_wl_starlight, 
                                          plot=False, save_plot=False, outfile='sdss_'+str(obj))

    plotbptnormal([], [],nx=20, ny=20, kewley=True, mod_kauff=False, nobj=False)
    
    
    
    if xragn_unclass_p1.niiflux_sn.iloc[obj] >2 and xragn_unclass_p1.halpflux_sn.iloc[obj] >2:
        if xragn_unclass_p1.oiiiflux_sn.iloc[obj] >2 and xragn_unclass_p1.hbetaflux_sn.iloc[obj] >2:
            plt.scatter(xragn_unclass_p1.niiha.iloc[obj], xragn_unclass_p1.oiiihb.iloc[obj], marker='x', label='MPA SDSS')
        else:
            plt.scatter(xragn_unclass_p1.niiha.iloc[obj],1.7, marker='x', label='MPA SDSS')
    if sdss_dob.n2_sn >2 and sdss_dob.ha_sn >2:
        if sdss_dob.o3_sn >2 and sdss_dob.hb_sn.iloc[obj] >2:
            plt.scatter(sdss_dob.n2ha, sdss_dob.o3hb, marker='^', label='Dobby SDSS')
        else:
            plt.scatter(sdss_dob.n2ha,1.7, marker='^', label='Dobby SDSS')
    xr3arc = xobj.spec_sub_3arc_dob
    if xr3arc.n2_sn >2 and xr3arc.ha_sn >2:
        if xr3arc.o3_sn >2 and xr3arc.hb_sn.iloc[obj] >2:
            plt.scatter(xr3arc.n2ha, xr3arc.o3hb, marker='o', label="Dobby 3'' MUSE")
        else:
            plt.scatter(xr3arc.n2ha,1.7, marker='o', label="Dobby 3'' MUSE")


    plt.legend()        
    plt.savefig('plots/xu'+str(obj)+'_bpt_lr_comparison_3arc.pdf', dpi=250, format='pdf', bbox_inches='tight')
    plt.savefig('plots/xu'+str(obj)+'_bpt_lr_comparison_3arc.png', dpi=250, format='png', bbox_inches='tight')
    plt.close()    
        
    
from mpl_toolkits.axes_grid1 import make_axes_locatable
offsets = [-0.00448,-0.00448, -0.00448,-0.00448,-0.00448]

for i, obj in enumerate(muse_cubes.keys()):
    ximg = muse_cubes[obj].dobby


    wcs = WCS(muse_cubes[obj].obs.data_header)
    wcs= wcs.dropaxis(2)
    fig = plt.figure(figsize=(16,5.4))
    spec = fig.add_gridspec(ncols=4, nrows=2)

    low_sn_o3_mask = ximg.o3_sn<2
    
    low_sn_n2_mask = ximg.n2_sn<2
    low_sn_ha_mask = ximg.ha_sn<2
    low_sn_hb_mask = ximg.hb_sn<2
    
    n2_copy = np.copy(ximg.n2_flux)
    ha_copy = np.copy(ximg.ha_flux)
    hb_copy = np.copy(ximg.hb_flux)
    o3_copy = np.copy(ximg.o3_flux)
    
    o3hb_copy = np.copy(ximg.o3hb)
    n2ha_copy = np.copy(ximg.n2ha)

    low_sn_n2ha = (low_sn_ha_mask | low_sn_n2_mask)
    low_sn_o3hb  = (low_sn_hb_mask | low_sn_o3_mask)

    
    low_sn_bpt = (low_sn_n2ha | low_sn_o3hb)
    bptmap_copy = np.copy(ximg.bpt_map_)
    bptmap_copy[low_sn_bpt] = np.nan    
    ykau = np.log10(y1_kauffmann(n2ha_copy))
    bptmap_copy = np.where(o3hb_copy>ykau, 2, 1)
    bptmap_copy = np.where(n2ha_copy>0, 2, bptmap_copy)
    bptmap_copy = np.where(low_sn_bpt, np.nan, bptmap_copy)
    ax1 = fig.add_subplot(spec[0, 0], projection=wcs)        


    muse_cubes[obj].plot_im(muse_cubes[obj].obs_rgb, filename=obj+'rgb',ax=ax1, save=False)
    ax2 = fig.add_subplot(spec[0, 3], projection=wcs, sharex=ax1, sharey=ax1)    
    a, cbar1=  muse_cubes[obj].plot_im(bptmap_copy, cbarlabel=r'BPT Class', cbarticklabels=['SF','AGN'], filename=obj+'_bpt_class', save=False,ax=ax2, vmin=1, vmax=2,center=True)  

    o3_copy[low_sn_o3_mask] = np.nan
    ax3 = fig.add_subplot(spec[1,0], projection=wcs)    

    muse_cubes[obj].plot_im(np.log10(o3_copy), cbarlabel=r'log(F$_{\mathrm{[OIII}}$)',ax=ax3, filename=obj+'_o3flux',center=True, save=False)
    
    

    n2_copy[low_sn_n2_mask] = np.nan
    ax4 = fig.add_subplot(spec[1,2], projection=wcs)    

    muse_cubes[obj].plot_im(np.log10(n2_copy), cbarlabel=r'log(F$_{\mathrm{[NII}}$)', ax=ax4,filename=obj+'_n2flux', save=False,center=True)


    ha_copy[low_sn_ha_mask] = np.nan
    ax6 = fig.add_subplot(spec[1, 3], projection=wcs)    
    
    muse_cubes[obj].plot_im(np.log10(ha_copy), cbarlabel=r'log(F$_{\mathrm{H}\alpha}$)',ax=ax6, filename=obj+'_haflux', save=False,center=True)


    hb_copy[low_sn_hb_mask] = np.nan
    ax5 = fig.add_subplot(spec[1, 1], projection=wcs)    

    muse_cubes[obj].plot_im(np.log10(hb_copy), cbarlabel=r'log(F$_{\mathrm{H}\beta}$)', ax=ax5,filename=obj+'_hbflux', save=False,center=True)


    n2ha_copy[low_sn_n2ha] = np.nan
    ax8 = fig.add_subplot(spec[0, 2], projection=wcs)    

    muse_cubes[obj].plot_im(n2ha_copy, cbarlabel=r'log([NII]/H$\alpha$)', filename=obj+'_n2ha',ax=ax8, save=False, vmin=-1, vmax=0.5,center=True)
   
    o3hb_copy[low_sn_o3hb] = np.nan
    ax7 = fig.add_subplot(spec[0, 1], projection=wcs)    

    muse_cubes[obj].plot_im(o3hb_copy, cbarlabel=r'log([OIII]/H$\beta$)', filename=obj+'_o3hb',ax=ax7, save=False, vmin=-1, vmax=1,center=True)
    
    ax1.set_xlabel('')
    ra, dec = ax1.coords[0], ax1.coords[1]
    ra.set_ticklabel_visible(False)
    dec.set_axislabel('Dec.')
    ra.set_axislabel('')
    ra.set_ticklabel(exclude_overlapping=True)
    

    ra, dec = ax2.coords[0], ax2.coords[1]
    ra.set_ticklabel_visible(False)
    dec.set_ticklabel_visible(False)
    ra.set_ticklabel(exclude_overlapping=True)

    ax2.set_xlabel('')
    ax2.set_ylabel('')


    ra, dec = ax3.coords[0], ax3.coords[1]

    #ra.set_ticklabel_visible(False)
    #dec.set_ticklabel_visible(False)
    ax3.set_xlabel('RA')
    ax3.set_ylabel('Dec.')
    ra.set_ticklabel(exclude_overlapping=True)

    #ax3.set_xlabel('')
    #ax3.set_ylabel('')
    
    ra, dec = ax4.coords[0], ax4.coords[1]
    #ra.set_ticklabel_visible(False)
    ra.set_ticklabel(exclude_overlapping=True)
    dec.set_ticklabel_visible(False)
    ra.set_axislabel('RA')
    dec.set_axislabel('')

    
    #ax5.set_xlabel('')
    ra,dec =ax5.coords[0], ax5.coords[1]
    ra.set_ticklabel(exclude_overlapping=True)

    #ra.set_ticklabel_visible(False)
    dec.set_ticklabel_visible(False)
    
    dec.set_axislabel('')
    ra.set_axislabel('RA')
    
    ra.set_ticklabel(exclude_overlapping=True)
    

    ra, dec = ax6.coords[0], ax6.coords[1]
    dec.set_axislabel('')
    ra.set_axislabel('RA')
    #ra.set_ticklabel_visible(False)
    
    dec.set_ticklabel_visible(False)
    ra.set_ticklabel(exclude_overlapping=True)


    ra, dec = ax7.coords[0], ax7.coords[1]
    ra.set_ticklabel(exclude_overlapping=True)
    ra.set_axislabel('')
    dec.set_axislabel('')

    dec.set_ticklabel_visible(False)
    ra.set_ticklabel_visible(False)

    ra, dec = ax8.coords[0], ax8.coords[1]
    ra.set_ticklabel(exclude_overlapping=True)
    ra.set_axislabel('')
    dec.set_axislabel('')

    dec.set_ticklabel_visible(False)
    ra.set_ticklabel_visible(False)

    #fig.tight_layout()
    #plt.subplots_adjust( hspace=0)
    #box = ax1.get_position()
    #box.x0 = box.x0 +offsets[i]
    #box.x1 = box.x1 + offsets[i]
    #ax1.set_position(box)
    
    fig.savefig('plots/'+obj+'_maps.pdf', format='pdf', dpi=250, bbox_inches='tight')
    plt.close()
    
    par_dict = {'halpflux': [ximg.ha_flux, r'log(F$_{\mathrm{H}\alpha}$)'],'hbetaflux':[ximg.hb_flux, r'log(F$_{\mathrm{H}\beta}$)'],
                     'n2flux': [ximg.n2_flux,r'log(F$_{\mathrm{[NII]}}$)'], 'o3flux':[ximg.o3_flux, r'log(F$_{\mathrm{[OIII}}$)'], 
                     's2flux': [ximg.s2_flux,r'log(F$_{\mathrm{[SII]}}$)'], 'o1flux':[ximg.o1_flux, r'log(F$_{\mathrm{[OI}}$)'], 
                     'n2ha': [ximg.n2ha,r'log([NII]/H$\alpha$)'], 'o3hb':[ximg.o3hb, r'log([OIII]/H$\beta$)'],
                     's2ha': [ximg.s2ha,r'log([SII]/H$\alpha$)'], 'o1ha':[ximg.o1ha, r'log([OI]/H$\alpha$)'],
                     'bptmap': [ximg.bpt_map_,r'BPT Class']}
    

for i in range(len(muse_cubes['xr31'].aperture_dobs)): 
    o3fluxes_aps = []
    o3flux_errors_aps = []

    for obj in muse_cubes.keys():
    
        o3fluxes_aps.append( muse_cubes[obj].aperture_dobs[i].o3_flux)
        o3flux_errors_aps.append( muse_cubes[obj].aperture_dobs[i].o3_flux_error)
        
    o3lums = getlumfromflux(np.array(o3fluxes_aps)/1e20, np.array(redshifts))
    o3lums_down = getlumfromflux((np.array(o3fluxes_aps)-np.array(o3flux_errors_aps))/1e20, np.array(redshifts))
    o3lums_up = getlumfromflux((np.array(o3fluxes_aps)+np.array(o3flux_errors_aps))/1e20, np.array(redshifts))
    
    plt.gca().set_aspect('equal')
    plt.errorbar( xraylums[1:],np.log10(o3lums)[1:],yerr=[np.log10(o3lums[1:])-np.log10(o3lums_down[1:]), 
                                                          np.log10(o3lums_up[1:])-np.log10(o3lums[1:])],color='r', fmt='none')
    plt.tight_layout()
    plt.savefig('plots/lx_lo3_allxragn_'+str((i+1)*0.5)+'arc.pdf', bbox_inches='tight', dpi=250, format='pdf')
    plt.savefig('plots/lx_lo3_allxragn_'+str((i+1)*0.5)+'arc.png', bbox_inches='tight', dpi=250, format='png')
    plt.close()
    plt.scatter(merged_xr_all.full_xraylum,merged_xr_all.oiiilum, s=3)

    plt.ylabel('log(L$_{\mathrm{[OIII]}}$)')
    plt.xlabel('log(L$_{\mathrm{X}}$)')
    plt.ylim([37,44])
    
    plt.gca().set_aspect('equal')
    plt.errorbar( xraylums[1:],np.log10(o3lums)[1:],yerr=[np.log10(o3lums)-np.log10(o3lums_down), 
                                                          np.log10(o3lums_up)-np.log10(o3lums)],color='r', fmt='none')
    plt.tight_layout()
    plt.savefig('plots/lx_lo3_bpt_'+str((i+1)*0.5)+'arc.pdf', bbox_inches='tight', dpi=250, format='pdf')
    plt.savefig('plots/lx_lo3_bpt_'+str((i+1)*0.5)+'arc.png', bbox_inches='tight', dpi=250, format='png')
    plt.close()
    
    
    


o3fluxes_aps_3 = []
o3flux_errors_aps_3 = []

o3fluxes_corr_aps_3 = []
o3flux_errors_corr_aps_3 = []
o3fluxes_corr_aps_host_3 = []

o3fluxes_aps_05 = []
o3flux_errors_aps_05 = []

o3fluxes_corr_aps_05 = []
o3flux_errors_corr_aps_05 = []
o3fluxes_corr_aps_host_05 = []



bd_sns = []
o3_sns = []

muse_c = ['g','b','gray', 'r', 'k']
muse_sym = ['o','s','>','d', 'v']
redshifts=[]

names = np.array(['SF-1','WL-3','WL-2','WL-EXT-1', 'WL-1'])
for i,obj in enumerate(muse_cubes.keys()):
    redshifts.append(muse_samp[i]['z'])
    o3fluxes_aps_3.append( muse_cubes[obj].aperture_dobs[-1].o3_flux)
    o3flux_errors_aps_3.append( muse_cubes[obj].aperture_dobs[-1].o3_flux_error)
    
    o3fluxes_corr_aps_3.append( muse_cubes[obj].aperture_dobs[-1].o3_flux_corr[0])
    o3fluxes_corr_aps_host_3.append( dustcorrect( muse_cubes[obj].aperture_dobs[-1].o3_flux, muse_samp[i]['corrected_presub_av'], 5007.0)[0] )
    
    
    o3fluxes_aps_05.append( muse_cubes[obj].aperture_dobs[0].o3_flux)
    o3flux_errors_aps_05.append( muse_cubes[obj].aperture_dobs[0].o3_flux_error)
    
    o3fluxes_corr_aps_05.append( muse_cubes[obj].aperture_dobs[0].o3_flux_corr[0])
    o3fluxes_corr_aps_host_05.append( dustcorrect( muse_cubes[obj].aperture_dobs[0].o3_flux, muse_samp[i]['corrected_presub_av'], 5007.0)[0] )
    
    
    o3_sns.append( muse_cubes[obj].aperture_dobs[-1].o3_sn)
    
    o3flux_errors_corr_aps_3.append( muse_cubes[obj].aperture_dobs[-1].o3_flux_error_corr[0])
    
    bd_sns.append( np.min((muse_cubes[obj].aperture_dobs[-1].hb_sn,muse_cubes[obj].aperture_dobs[-1].ha_sn )))
                  

o3lums_05 = getlumfromflux(np.array(o3fluxes_aps_05)/1e20, np.array(redshifts))
o3lums_down_05 = getlumfromflux((np.array(o3fluxes_aps_05)-np.array(o3flux_errors_aps_05))/1e20, np.array(redshifts))
o3lums_up_05 = getlumfromflux((np.array(o3fluxes_aps_05)+np.array(o3flux_errors_aps_05))/1e20, np.array(redshifts))

o3lums_corr_05 = getlumfromflux(np.array(o3fluxes_corr_aps_05)/1e20, np.array(redshifts))
o3lums_corr_host_05 = getlumfromflux(np.array(o3fluxes_corr_aps_host_05)/1e20, np.array(redshifts))


o3lums_3 = getlumfromflux(np.array(o3fluxes_aps_3)/1e20, np.array(redshifts))
o3lums_down_3 = getlumfromflux((np.array(o3fluxes_aps_3)-np.array(o3flux_errors_aps_3))/1e20, np.array(redshifts))
o3lums_up_3 = getlumfromflux((np.array(o3fluxes_aps_3)+np.array(o3flux_errors_aps_3))/1e20, np.array(redshifts))

o3lums_corr_3 = getlumfromflux(np.array(o3fluxes_corr_aps_3)/1e20, np.array(redshifts))
o3lums_corr_host_3 = getlumfromflux(np.array(o3fluxes_corr_aps_host_3)/1e20, np.array(redshifts))


#o3lums_down_corr = getlumfromflux((np.array(o3fluxes_corr_aps)-np.array(o3flux_errors_corr_aps))/1e20, np.array(redshifts))
#o3lums_up_corr= getlumfromflux((np.array(o3fluxes_corr_aps)+np.array(o3flux_errors_corr_aps))/1e20, np.array(redshifts))


fig  = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('log(L$_{\mathrm{X,\ 2-10\ keV}}$)')
ax1.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].hard_xraylum, xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].oiiilum, 
        minx=38, maxx=46, miny=38, maxy=43.8, label='Point Source X-ray AGN', marker='s',edgecolor='magenta', facecolor='magenta', fig = fig, ax = ax1)
#scatter(xr_agn_props['x4_sn1_o3_hx_allz_ext_nobptsf'].hard_xraylum, xr_agn_props['x4_sn1_o3_hx_allz_ext_nobptsf'].oiiilum, 
#        minx=38, maxx=46, miny=38, maxy=43.8, label='Extended X-ray Sources',  marker='^', edgecolor='cyan', facecolor='cyan', fig = fig, ax = ax1)

ax1.plot(x_panessa, geny_, 'r--',color='r', linewidth=3, label='Panessa+06')
ax1.set_aspect('equal')


plt.gca().set_aspect('equal')
o3lums_3 = np.array(o3lums_3)
o3lums_05 = np.array(o3lums_05)

xraylums = np.array(xraylums)
o3lums_down_3 = np.array(o3lums_down_3)
o3lums_up_3 = np.array(o3lums_up_3)
o3lums_down_05 = np.array(o3lums_down_05)
o3lums_up_05 = np.array(o3lums_up_05)

o3lums_error_3 = abs(np.log10(o3lums_3) - (np.log10(o3lums_down_3)+np.log10(o3lums_up_3))/2)
o3lums_error_05= abs(np.log10(o3lums_05) - (np.log10(o3lums_down_05)+np.log10(o3lums_up_05))/2)

muse_samp = [xr, xu1, xu2, xu3, xu4]

for i in [0,2,4]:
    print(names[i])
    plt.errorbar( xraylums[i],
             np.log10(o3lums_corr_host_3)[i],
             xerr =  (muse_samp[i].e_hard_xraylum_up  + muse_samp[i].e_hard_xraylum_down)/2 ,
             yerr =  o3lums_error_3[i],
             zorder=10+i,
             marker=muse_sym[i],
             c=muse_c[i],
             markersize=5,
             capsize=10, elinewidth=0.5)
    plt.errorbar( xraylums[i],
             np.log10(o3lums_corr_host_05)[i],
             xerr =  (muse_samp[i].e_hard_xraylum_up  + muse_samp[i].e_hard_xraylum_down)/2 ,
             yerr =  o3lums_error_05[i] ,
             zorder=10+i,
             marker=muse_sym[i],
             c=muse_c[i],
             markersize=5,
             capsize=10, elinewidth=0.5)
    plt.errorbar( xraylums[i],
             (muse_samp[i].oiiilum),             
             xerr =  (muse_samp[i].e_hard_xraylum_up  + muse_samp[i].e_hard_xraylum_down)/2 ,
             yerr =  (muse_samp[i].e_oiiilum_up  + muse_samp[i].e_oiiilum_down)/2 ,
             zorder=10+i,
             marker=muse_sym[i],
             c=muse_c[i],
             label=names[i],
             markersize=5,
             capsize=10, elinewidth=0.5)

i=1
plt.errorbar( xraylums[i],
            np.log10(o3lums_corr_host_3)[i],
            xerr =  (muse_samp[i].e_hard_xraylum_up  + muse_samp[i].e_hard_xraylum_down)/2 ,
            yerr =  o3lums_error_3[i],
            zorder=10+i,
            marker=muse_sym[i],
            c=muse_c[i],
            label=names[i],

            markersize=5,
            capsize=10, elinewidth=0.5)
#plt.scatter( xraylums[i],
#         np.log10(o3lums_corr_host_05)[i],zorder=10+i,
#         marker=muse_sym[i],edgecolor=muse_c[i],facecolor='none', s=100)
#plt.scatter( xraylums[i],
#         muse_samp[i].oiiilum,zorder=10+i,
#         marker=muse_sym[i],facecolor='none', edgecolor=muse_c[i], s=200, linewidth=2)


#i=3
#plt.scatter( xraylums[i],
#         38.1,zorder=10+i,
#         marker=muse_sym[i],edgecolor=muse_c[i],facecolor='none',label=names[i], s=250, linewidth=3)
plt.tight_layout()

handles, labels = ax1.get_legend_handles_labels()
order = [0,1,2,4,3,5]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=15)

ax1.set_xticks([38,40,42,44])

plt.savefig('plots/lxo3_muse_corr.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxo3_muse_corr.png', bbox_inches='tight', dpi=250)
plt.close()


label='full'
scat=0.6

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(loglum_arr,lsfrrelat[label][2],'k--',zorder=3)

plt.fill_between(loglum_arr+scat,lsfrrelat[label][2],y2=lsfrrelat[label][2]-20,color='gray', zorder=0,alpha=0.3,linewidth=0)


plt.xlabel(r'log(L$_{\rm X, 0.5-10\ keV}$)',fontsize=20)
plt.ylabel(r'log(SFR)',fontsize=20)

for i, obj in enumerate(muse_samp.keys()):
    print(i, obj, muse_sym[i], names[i])
    plt.errorbar( muse_samp[obj].full_xraylum,
             muse_samp[obj].sfr,
             xerr =  (muse_samp[obj].e_full_xraylum_up  + muse_samp[obj].e_full_xraylum_down)/2 ,
             yerr =  muse_samp[obj].sfr_error,
             zorder=10+i,
             marker=muse_sym[i],
             color=muse_c[i],
             label=names[i],
             markersize=5,
             capsize=10, elinewidth=0.5
             )
scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].full_xraylum, xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].sfr, 
        label='X-ray AGNs', marker='s',edgecolor='magenta', facecolor='magenta')
#scatter(xr_agn_props['x4_sn1_o3_hx_allz_ext_nobptsf'].full_xraylum, xr_agn_props['x4_sn1_o3_hx_allz_ext_nobptsf'].sfr, 
#       label='Extended X-ray Sources',  marker='^', edgecolor='cyan', facecolor='cyan')
plt.text(43.1,-3.05,'X-ray AGN\n Candidates', fontsize=14, rotation=0)
plt.xlim([37.,45.5])
plt.ylim([-3.5, 5])
ax.set(adjustable='box', aspect='equal')
order = [0, 1,5, 3, 2,4]
handles, labels = ax.get_legend_handles_labels()

ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=12)
plt.tight_layout()
plt.savefig('plots/lxsfr_muse.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxsfr_muse.png', bbox_inches='tight', dpi=250)
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
for i, obj in enumerate(muse_samp.keys()):
    plt.errorbar( muse_samp[obj].mass,
             muse_samp[obj].ssfr,         
             xerr =  (muse_samp[obj].mass_error) ,
             yerr =  np.sqrt(muse_samp[obj].sfr_error**2+muse_samp[obj].mass_error**2),
             zorder=10+i,
             marker=muse_sym[i],
             color=muse_c[i],
             label=names[i],
             markersize=5,
             capsize=10, elinewidth=0.5)
    
scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].mass, xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].ssfr, 
      label='X-ray AGNs', marker='s',edgecolor='magenta', facecolor='magenta')

#scatter(xr_agn_props['x4_sn1_o3_hx_allz_ext_nobptsf'].mass, xr_agn_props['x4_sn1_o3_hx_allz_ext_nobptsf'].ssfr, 
#       label='Extended X-ray Sources',  marker='^', edgecolor='cyan', facecolor='cyan')

order = [1, 5,3,2,4,0]
handles, labels = ax.get_legend_handles_labels()

ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=12)
plot2dhist(bpt_EL_gsw_df.mass, bpt_EL_gsw_df.ssfr, minx=7.5, maxx=12.5, miny=-15, maxy=-8)
plt.ylabel(r'log(sSFR)',fontsize=20)
plt.xlabel(r'log(M$_{\mathrm{*}})$',fontsize=20)
plt.savefig('plots/ssfrm_muse.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/ssfrm_muse.png', bbox_inches='tight', dpi=250)

    

w1 = np.array([13.579,13.478,12.732,14.221])
w2 = np.array([13.427,13.378,12.544,14.052])
w3 = np.array([9.738,11.445,10.998,12.035])
w4 = np.array([8.247	,8.581,8.373,8.163])



#whan

halp_eqws=[]
niihas = []

xr31halp_eqw3 = np.log10(xr31.spec_sub_3arc_dob.ha_fit[0][12][7])
xr31halp_eqw05 = np.log10(xr31.spec_sub_05arc_dob.ha_fit[0][12][7])

xr31n2ha_3 = (xr31.spec_sub_3arc_dob.n2ha)
xr31n2ha_05 = (xr31.spec_sub_05arc_dob.n2ha)

print(xr31halp_eqw05, xr31halp_eqw3, np.log10(-muse_samp['SF-1'].halp_eqw))

xu22halp_eqw3 = np.log10(xu22.spec_sub_3arc_dob.ha_fit[0][12][7])
xu22halp_eqw05 = np.log10(xu22.spec_sub_05arc_dob.ha_fit[0][12][7])
xu22n2ha_3 = (xu22.spec_sub_3arc_dob.n2ha)
xu22n2ha_05 = (xu22.spec_sub_05arc_dob.n2ha)

print(xu22halp_eqw05, xu22halp_eqw3, np.log10(-muse_samp['WL-3'].halp_eqw))


xu23halp_eqw3 = np.log10(xu23.spec_sub_3arc_dob.ha_fit[0][12][7])
xu23halp_eqw05 = np.log10(xu23.spec_sub_05arc_dob.ha_fit[0][12][7])
xu23n2ha_3 = (xu23.spec_sub_3arc_dob.n2ha)
xu23n2ha_05 = (xu23.spec_sub_05arc_dob.n2ha)

print(xu23halp_eqw05, xu23halp_eqw3, np.log10(-muse_samp['WL-2'].halp_eqw))

xu104halp_eqw3 = np.log10(xu104.spec_sub_3arc_dob.ha_fit[0][12][7])
xu104halp_eqw05 = np.log10(xu104.spec_sub_05arc_dob.ha_fit[0][12][7])
xu104n2ha_3 = (xu104.spec_sub_3arc_dob.n2ha)
xu104n2ha_05 = (xu104.spec_sub_05arc_dob.n2ha)

print(xu104halp_eqw05, xu104halp_eqw3, np.log10(-muse_samp['WL-EXT-1'].halp_eqw))


xu210halp_eqw3 = np.log10(xu210.spec_sub_3arc_dob.ha_fit[0][12][7])
xu210halp_eqw05 = np.log10(xu210.spec_sub_05arc_dob.ha_fit[0][12][7])
xu210n2ha_3 = (xu210.spec_sub_3arc_dob.n2ha)
xu210n2ha_05 = (xu210.spec_sub_05arc_dob.n2ha)

print(xu210halp_eqw05, xu210halp_eqw3, np.log10(-muse_samp['WL-1'].halp_eqw))



fig = plt.figure()
ax = fig.add_subplot(111)
plotwhan(bpt_EL_gsw_df.niiha, 
         np.log10(-bpt_EL_gsw_df.halp_eqw), lim=False, fig = fig, ax = ax)


ax.scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].niiha, 
        np.log10(-xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].halp_eqw), 
      label='X-ray AGNs', marker='s',edgecolor='k', facecolor='magenta', linewidth=0.1, zorder=0, s= 5)
i=0
obj='SF-1'
ax.scatter( muse_samp[obj].niiha,
         np.log10(-muse_samp[obj].halp_eqw) , zorder=10+i+1,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='white',
         label=names[i],
         linewidth=1, 
         s=200)
i=0
obj='SF-1'
ax.scatter( xr31n2ha_05,
         xr31halp_eqw05 , zorder=10+i+2,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='white',
         linewidth=0.5,
         s=150)

i=0
obj='SF-1'
ax.scatter( xr31n2ha_3,
         xr31halp_eqw3 , zorder=10+i,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='white',
         linewidth=2,
         s=250)

i = 2
obj='WL-2'
ax.scatter( muse_samp[obj].niiha,
         np.log10(-muse_samp[obj].halp_eqw) , zorder=10+i,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='white',
         label=names[i],
         
         s=200, linewidth=1)
i = 2
obj='WL-2'
ax.scatter( xu23n2ha_05,
         xu23halp_eqw05, zorder=10+i,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='white',
         linewidth=0.5,
         s=150)
i = 2
obj='WL-2'
ax.scatter( xu23n2ha_3,
         xu23halp_eqw3, zorder=10+i,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='white',
         linewidth=2,
         s=250)


i = 3
obj='WL-EXT-1'
ax.scatter( muse_samp[obj].niiha,
         np.log10(-muse_samp[obj].halp_eqw) , zorder=10+i,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='none',
         label=names[i],
         linewidth = 1,
         s=200)
i = 3
obj='WL-EXT-1'
ax.scatter( xu104n2ha_05,
         xu104halp_eqw05 , zorder=10+i,
         marker=muse_sym[i],
         edgecolor=muse_c[i],
         facecolor='none', 
         
         linewidth=0.5,
         s=150)
i = 3
obj='WL-EXT-1'
ax.scatter( xu104n2ha_3,
         xu104halp_eqw3 , zorder=10+i,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='none',
         s=250, linewidth=2)

i=4
obj='WL-1'
ax.scatter( xu210n2ha_05,
         xu210halp_eqw05 , zorder=10+i,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='none',
         linewidth=0.5,
         label=names[i],
         s=150)
i=4
obj='WL-1'
ax.scatter( xu210n2ha_3,
         xu210halp_eqw3 , zorder=10+i,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='none',
         linewidth=2,
         s=250)

i=4
obj='WL-1'
ax.scatter( muse_samp[obj].niiha,
         np.log10(-muse_samp[obj].halp_eqw) , zorder=10+i,
         marker=muse_sym[i],
         edgecolor=muse_c[i], facecolor='none', hatch = '//',
         linewidth=1,
         s=200)









handles, labels = ax.get_legend_handles_labels()
order = [0, 1, 4, 2,3]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=3, bbox_to_anchor=(0.01, 0.03))
plt.savefig('plots/whan_muse.pdf', dpi=250, format='pdf', bbox_inches='tight')
plt.savefig('plots/whan_muse.png', dpi=250, format='png', bbox_inches='tight')





        
        
'''
