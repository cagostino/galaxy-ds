#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as plt 
from xray_load import *
from scipy.stats import pearsonr
def reduced_clustering(tab):
    
    subtab = xr_agn_props['x4_hx_allz_nobptsf'][['ra', 'dec','z']]
    dists = []
    zdiffs = []
    
    dists_ph = []
    zdiffs_ph = []
    
    for i in range(len(subtab)):
        ra_i, dec_i, z_i = subtab[['ra','dec','z']].iloc[i]
        tab_matched =tab[(tab['ra']==ra_i)&(tab['dec']==dec_i)]
        RAJ = tab_matched['RAJ2000']
        DEJ = tab_matched['DEJ2000']
        #RAJ = np.array(pd.to_numeric(RAJ, errors='coerce'))
        #DEJ = np.array(pd.to_numeric(DEJ, errors='coerce'))
        zsp_diffs_i = abs(np.array( pd.to_numeric(tab_matched['zsp'], errors='coerce'))-z_i)
        zph_diffs_i = abs(np.array( pd.to_numeric(tab_matched['zph'], errors='coerce'))-z_i)

        dists_i = np.array(np.sqrt(((ra_i- RAJ)*np.cos(np.radians(dec_i)))**2+
                          (dec_i-DEJ)**2)+zsp_diffs_i**2)
        dists_i_ph = np.array(np.sqrt(((ra_i- RAJ)*np.cos(np.radians(dec_i)))**2+
                          (dec_i-DEJ)**2)+zph_diffs_i**2)
        mindistind_sp=  np.nanargmin(zsp_diffs_i)
        mindistind_ph=  np.nanargmin(zsp_diffs_i)
        
        zdiffs.append(zsp_diffs_i[mindistind_sp])
        dists.append(dists_i[mindistind_sp])
        zdiffs_ph.append(zph_diffs_i[mindistind_ph])
        dists_ph.append(dists_i_ph[mindistind_ph])
        
    return zdiffs, dists, zdiffs_ph, dists_ph
        
from sklearn.svm import SVC

classes = np.where(xr_agn_props['x4_sn1_o3_hx_allz'].ext == 0, 0,1)

redshifts = [0,0.05, 0.1, 0.15, 0.2, 0.25, 0.35]
redshifts_ = np.array(redshifts)+0.025
nbins=  []
bpt_3sig = []

bpt_2sig = []

n2ha= []
bpt_2sig_n2 = []

bpt_2sig_n2_o3_1 = []

bpt_2sig_n2_o3_3 = []
o3_1 = []
o3_2 = []

for i in range(len(redshifts)):
    if i == len(redshifts)-1:
        zmin = 0
        zmax=0.3
    else:
        zmin = redshifts[i]
        zmax = redshifts[i+1]
        
    nosf_ = xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].append(xr_agn_props['x4_bad_o3_hx_allz_noext'])
    x4_noext = nosf_.iloc[np.where( (nosf_.z<=zmax) &    (nosf_.z>zmin)  )].copy()                                           
 
                                              
    x4_noextsize = len(x4_noext)
    nbins.append(x4_noextsize)
    bpt_3sig.append( np.where((x4_noext.oiiiflux_sn > 3) & (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2)& (x4_noext.niiflux_sn > 3)&
                       (x4_noext.hbetaflux_sn > 3)& (x4_noext.halpflux_sn > 3)&
                       (x4_noext.bptplusgroups=='AGN'))[0].size/x4_noextsize)
    bpt_2sig.append( np.where((x4_noext.oiiiflux_sn > 2) & (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2)& (x4_noext.niiflux_sn > 2)&
                       (x4_noext.hbetaflux_sn > 2)& (x4_noext.halpflux_sn > 2)&
                       (x4_noext.bptplusgroups=='AGN'))[0].size/x4_noextsize)
    o3_1.append( np.where((x4_noext.oiiiflux_sn > 1) & (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2))[0].size/x4_noextsize)
    o3_2.append( np.where((x4_noext.oiiiflux_sn > 2) & (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2))[0].size/x4_noextsize)
    
    bpt_2sig_n2.append( np.where( (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2)& (x4_noext.niiflux_sn > 2)&
                       (((x4_noext.oiiiflux_sn >2)  &(x4_noext.hbetaflux_sn > 2))|
                       ((x4_noext.oiiiflux_sn <2)  |(x4_noext.hbetaflux_sn < 2)))&
                       (x4_noext.halpflux_sn > 2))[0].size/x4_noextsize)
    bpt_2sig_n2_o3_1.append( np.where( (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2)& (x4_noext.niiflux_sn > 2)&
                       
                       (((x4_noext.oiiiflux_sn >2)  &(x4_noext.hbetaflux_sn > 2))|
                       ((x4_noext.oiiiflux_sn >1)  &(x4_noext.hbetaflux_sn < 2)))&
                       (x4_noext.halpflux_sn > 2))[0].size/x4_noextsize)
    bpt_2sig_n2_o3_3.append( np.where( (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2)& (x4_noext.niiflux_sn > 2)&
                       
                       (((x4_noext.oiiiflux_sn >2)  &(x4_noext.hbetaflux_sn > 2))|
                       ((x4_noext.oiiiflux_sn >3)  &(x4_noext.hbetaflux_sn < 2)))&
                       (x4_noext.halpflux_sn > 2))[0].size/x4_noextsize)
        
        
       
        
    

def plot_fiducial_pure_uncorr(xr_df, bin_y=True, percentiles=True, size_y_bin=0.5, color='k', fname='', save=False):
    xmid, avg_y, avg_quant, perc16, perc84 = scatter(xr_df.full_lxagn, xr_df.oiiilum_sub,
                                                     xerr=np.vstack([xr_df.e_full_xraylum_down,
                                                                     xr_df.e_full_xraylum_up]),
                                                     yerr=np.vstack([xr_df.e_oiiilum_down_sub,
                                                                     xr_df.e_oiiilum_up_sub]),
                                                     minx=37, maxx=46, miny=37, maxy=44, aspect='equal', alpha=1,
                                                     color=color, ecolor=color, bin_y=bin_y, size_y_bin=size_y_bin, percentiles=percentiles)
    plt.xticks([38, 40, 42, 44])
    plt.yticks([38, 40, 42, 44])

    plt.xlabel(r'log(L$_{\mathrm{X}}$)')
    plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('plots/lx_o3_uncorrected_sub'+fname+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/lx_o3_uncorrected_sub'+fname+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()
    return xmid, avg_y, avg_quant, perc16, perc84


def plot_fiducial_pure(xr_df, bin_y=True, percentiles=True, size_y_bin=0.5, color='k', fname='', save=False):
    xmid, avg_y, avg_quant, perc16, perc84 = scatter(xr_df.full_lxagn, xr_df.oiiilum_sub_dered,
                                                     xerr=np.vstack([xr_df.e_full_xraylum_down,
                                                                     xr_df.e_full_xraylum_up]),
                                                     yerr=np.vstack([xr_df.e_oiiilum_down_sub_dered,
                                                                     xr_df.e_oiiilum_up_sub_dered]),
                                                     minx=37, maxx=46, miny=37, maxy=44, aspect='equal', alpha=1,
                                                     color=color, ecolor=color, bin_y=bin_y, size_y_bin=size_y_bin, percentiles=percentiles)
    plt.xticks([38, 40, 42, 44])
    plt.yticks([38, 40, 42, 44])

    plt.xlabel(r'log(L$_{\mathrm{X}}$)')
    plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('plots/lx_o3_corrected_sub'+fname+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/lx_o3_corrected_sub'+fname+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()
    return xmid, avg_y, avg_quant, perc16, perc84


def plot_fiducial_uncorr(xr_df, bin_y=True, percentiles=True, size_y_bin=0.5, color='k', fname='', save=False):
    xmid, avg_y, avg_quant, perc16, perc84 = scatter(xr_df.full_lxagn, xr_df.oiiilum_uncorr,
                                                     xerr=np.vstack([xr_df.e_full_xraylum_down,
                                                                     xr_df.e_full_xraylum_up]),
                                                     yerr=np.vstack([xr_df.e_oiiilum_down_uncorr,
                                                                     xr_df.e_oiiilum_up_uncorr]),
                                                     minx=37, maxx=46, miny=37, maxy=44, aspect='equal', alpha=1,
                                                     color=color, ecolor=color, bin_y=bin_y, size_y_bin=size_y_bin, percentiles=percentiles)
    plt.xticks([38, 40, 42, 44])
    plt.yticks([38, 40, 42, 44])

    plt.xlabel(r'log(L$_{\mathrm{X}}$)')
    plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('plots/lx_o3_uncorrected_'+fname+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/lx_o3_uncorrected_'+fname+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()
    return xmid, avg_y, avg_quant, perc16, perc84


def plot_fiducial(xr_df, bin_y=True, percentiles=True, size_y_bin=0.5, color='k', fname='', save=False):
    xmid, avg_y, avg_quant, perc16, perc84 = scatter(xr_df.full_lxagn, xr_df.oiiilum,
                                                     xerr=np.vstack([xr_df.e_full_xraylum_down,
                                                                     xr_df.e_full_xraylum_up]),
                                                     yerr=np.vstack([xr_df.e_oiiilum_down,
                                                                     xr_df.e_oiiilum_up]),
                                                     minx=37, maxx=46, miny=37, maxy=44, aspect='equal', alpha=1,
                                                     color=color, ecolor=color, bin_y=bin_y, size_y_bin=size_y_bin, percentiles=percentiles)
    plt.xticks([38, 40, 42, 44])
    plt.yticks([38, 40, 42, 44])

    plt.xlabel(r'log(L$_{\mathrm{X}}$)')
    plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('plots/lx_o3_corrected_'+fname+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/lx_o3_corrected_'+fname+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()
    return xmid, avg_y, avg_quant, perc16, perc84


def plot_fwd_results(fwd_model, xr_df, filename):
    minks = np.argmin(fwd_model[-2])
    plt.plot(fwd_model[0], fwd_model[-2])
    plt.ylim([0, 1])
    plt.xlabel('Scatter Factor')
    plt.ylabel('KS-statistic')
    plt.tight_layout()
    plt.savefig('plots/scat_ks_' + filename+'.pdf',
                dpi=250, format='pdf', bbox_inches='tight')
    plt.savefig('plots/scat_ks_' + filename+'.png',
                dpi=250, format='png', bbox_inches='tight')
    plt.close()

    plothist(xr_df.oiiiflux, range=(-1e-13, 1e-13),
             bins=1000, cumulative=True, label='Real')
    plothist(fwd_model[1][minks], range=(-1e-13, 1e-13),
             bins=1000, cumulative=True, label='Simulated')

    plt.xlabel(r'F$_{\mathrm{[OIII]}}$')
    plt.ylabel(r'Cumulative Counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/cumulative_flux_' + filename+'.pdf',
                dpi=250, format='pdf', bbox_inches='tight')
    plt.savefig('plots/cumulative_flux_' + filename+'.png',
                dpi=250, format='png', bbox_inches='tight')
    plt.close()

    area_1 = plothist(xr_df.oiiilum-np.array(fwd_model[4][minks]), range=(-6, 6),
                      bins=30, normed=True, integrate=True)
    area_2 = plothist(np.array(fwd_model[2][minks])-np.array(fwd_model[4][minks]), range=(-6, 6),
                      bins=30, density=True, normed=True, integrate=True)
    plt.close()

    area_1 = plothist(xr_df.oiiilum-np.array(fwd_model[4][minks]), range=(-6, 6),
                      bins=30, normed=True, integrate=True, label='Real, area='+str(area_1)[0:4])
    area_2 = plothist(np.array(fwd_model[2][minks])-np.array(fwd_model[4][minks]), range=(-6, 6),
                      bins=30, density=True, normed=True, integrate=True, label='Simulated, area='+str(area_2)[0:4])

    plt.xlabel(r'$\Delta$log(L$_{\mathrm{[OIII]}}$)')
    plt.ylabel('Normalized Counts')

    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/delta_lo3_' + filename+'.pdf',
                dpi=250, format='pdf', bbox_inches='tight')
    plt.savefig('plots/delta_lo3_' + filename+'.png',
                dpi=250, format='png', bbox_inches='tight')
    plt.close()


def get_f_underlum(obs, nondetect, gauss):
    f_ul = 1-gauss.size/(obs.size+nondetect.size)
    return f_ul


'''
fig  = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)
scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].hard_xraylum, 
        xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].nlr_fib_ratio_pred_fromlx,
        ccode=xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].z,
        
             aspect='auto', bin_y=True, percentiles=False, bin_stat_y='mean',plotdata=True,linecolor='k',
             size_y_bin=0.3, counting_thresh=4,minx=39, maxx=44,miny=0, maxy=2,vmin=0.01, vmax=0.3,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.xlabel(r'log(L$_{\rm X}$)')
plt.ylabel(r'NLR Size/Fiber Size')

norm = mpl.colors.Normalize(vmin=0.01, vmax=0.3)

cbar_ax = fig.add_axes([1, 0.2, 0.03, 0.7])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='plasma'),
             cax=cbar_ax)

cbar.set_label(r'z', fontsize=20)
cbar.ax.tick_params(labelsize=20)


plt.legend()
plt.tight_layout()
plt.savefig('plots/fibnlr_sizes.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/fibnlr_sizes.png', dpi=250, bbox_inches='tight')




fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
x_, cnts_,_ = plothist(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].nlr_rad_from_lo3_pred_fromlx, range=(2,5), bins=25)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.ylabel(r'Counts')
plt.xlabel(r'log(NLR Size)[pc]')
plt.text(3.7, max(cnts_)-30 ,'Mode = ' +str(10**x_[np.argmax(cnts_)]/1000)[:3] +' kpc' )
plt.text(3.7, max(cnts_)-35 ,'Median = ' +str(10**np.median(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].nlr_rad_from_lo3_pred_fromlx)/1000)[0:3]+' kpc' )
plt.text(3.7, max(cnts_)-40 ,'Mean = ' +str(10**np.mean(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].nlr_rad_from_lo3_pred_fromlx/1000))[0:3]+' kpc' )

plt.legend()
plt.tight_layout()
plt.savefig('plots/fibnlr_sizes_hist.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/fibnlr_sizes_hist.png', dpi=250, bbox_inches='tight')





fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].d4000, 
        xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].oiiilum,
             aspect=1/2, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,linecolor='k',
             size_y_bin=0.3, counting_thresh=4,minx=1, maxx=2.5,miny=37.5, maxy=42.5,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.ylabel(r'log(L$_{\rm [OIII]}$)')
plt.xlabel(r'D$_{4000}$')

plt.legend()
plt.tight_layout()
plt.savefig('plots/o3_d4000.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/o3_d4000.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].sfr, 
        xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].oiiilum,
             aspect=1, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,linecolor='k',
             size_y_bin=0.3, counting_thresh=4,minx=-3, maxx=3,miny=37.5, maxy=42.5,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.ylabel(r'log(L$_{\rm [OIII]}$)')
plt.xlabel(r'log(SFR)')

plt.legend()
plt.tight_layout()
plt.savefig('plots/o3_sfr.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/o3_sfr.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].d4000, 
        xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].oiiilum_sfsub_samir,
             aspect=1/2, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,linecolor='k',
             size_y_bin=0.3, counting_thresh=4,minx=1, maxx=2.5,miny=37.5, maxy=42.5,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.ylabel(r'log(L$_{\rm [OIII], SF Sub.}$)')
plt.xlabel(r'D$_{4000}$')

plt.legend()
plt.tight_layout()
plt.savefig('plots/o3sf_sub_d4000.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/o3sf_sub_d4000.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].sfr, 
        xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].oiiilum_sfsub_samir,
             aspect=1, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,linecolor='k',
             size_y_bin=0.3, counting_thresh=4,minx=-3, maxx=3,miny=37.5, maxy=42.5,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.ylabel(r'log(L$_{\rm [OIII], SF Sub.}$)')
plt.xlabel(r'log(SFR)')

plt.legend()
plt.tight_layout()
plt.savefig('plots/o3sf_sub_sfr.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/o3sf_sub_sfr.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_noext_nobptsf'].mass, 
        xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_noext_nobptsf'].oiiilum,
             aspect=1, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,linecolor='k',
             size_y_bin=0.6, counting_thresh=4,minx=10, maxx=12,miny=37.5, maxy=42.5,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.ylabel(r'log(L$_{\rm [OIII]}$)')
plt.xlabel(r'log(M$_{*}$)')

#plt.xlim([-2,2])
#plt.xticks([-2,-1,0,1,2])
#plt.ylim([-3,1])
plt.legend()
plt.tight_layout()
plt.savefig('plots/lo3_mass.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/lo3_mass.png', dpi=250, bbox_inches='tight')




fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_noext_nobptsf'].mass, 
        xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_noext_nobptsf'].hard_xraylum,
             aspect=1, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,linecolor='k',
             size_y_bin=0.4, counting_thresh=4,minx=10, maxx=12,miny=39.5, maxy=44.5,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.ylabel(r'log(L$_{\rm X}$)')
plt.xlabel(r'log(M$_{*}$)')

#plt.xlim([-2,2])
#plt.xticks([-2,-1,0,1,2])
#plt.ylim([-3,1])
plt.legend()
plt.tight_layout()
plt.savefig('plots/lx_mass.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/lx_mass.png', dpi=250, bbox_inches='tight')



fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
x = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up])
y= np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].av_agn,xr_agn_props['x4_bad_o3_hx_allz_noext'].av_agn])
filt = np.where((np.isfinite(x))&(np.isfinite(y)))
r = pearsonr(x[filt],y[filt])
r = str(round(r[0],2))
print(r)
scatter(x,y,
             aspect=5/4, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,linecolor='k',label=r'',
             size_y_bin=0.6, counting_thresh=4,minx=-2.1, maxx=2.1,miny=0, maxy=2,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up, xr_agn_props['x4_bad_o3_hx_allz_noext'].ssfr, marker='s', color='blue', s=15)

#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(sSFR)')
plt.xlim([-2,2])
plt.xticks([-2,-1,0,1,2])
plt.ylim([-0,2])
plt.gca().invert_xaxis()
plt.text(1.85, 1.7, 'Pearson R ='+r, fontsize=12)

plt.tight_layout()

plt.savefig('plots/dlo3_ssfr.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_ssfr.png', dpi=250, bbox_inches='tight')



fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
x = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up])
y= np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].ssfr,xr_agn_props['x4_bad_o3_hx_allz_noext'].ssfr])
r = pearsonr(x,y)
r = str(round(r[0],2))
print(r)
scatter(x,y,
             aspect=5/4, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,linecolor='k',label=r'',
             size_y_bin=0.6, counting_thresh=4,minx=-2.1, maxx=2.1,miny=-14, maxy=-9,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up, xr_agn_props['x4_bad_o3_hx_allz_noext'].ssfr, marker='s', color='blue', s=15)

#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(sSFR)')
plt.xlim([-2,2])
plt.xticks([-2,-1,0,1,2])
plt.ylim([-14,-9])
plt.gca().invert_xaxis()
plt.text(1.85, -13.7, 'Pearson R ='+r, fontsize=12)

plt.tight_layout()

plt.savefig('plots/dlo3_ssfr.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_ssfr.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
x = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset,
               xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up])
y = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].sfr, 
               xr_agn_props['x4_bad_o3_hx_allz_noext'].sfr])
r = pearsonr(x,y)
r = str(round(r[0],2))
print(r)
scatter(x,y,
             aspect=25/16, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,linecolor='k',label=r'',
             size_y_bin=0.6, counting_thresh=4,minx=-2.1, maxx=2.1,miny=-2, maxy=1,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up, 
            xr_agn_props['x4_bad_o3_hx_allz_noext'].sfr, marker='s', color='blue', s=15)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up, 
            xr_agn_props['x4_bad_o3_hx_allz_noext'].sfr, marker='s', color='blue', s=15)

#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(SFR)')
plt.xlim([-2,2])
plt.xticks([-2,-1,0,1,2])
plt.ylim([-3,3])
plt.gca().invert_xaxis()
plt.text(1.5, 2.2, 'Pearson R ='+r, fontsize=12)

plt.tight_layout()

plt.savefig('plots/dlo3_sfr.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_sfr.png', dpi=250, bbox_inches='tight')







fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
x  = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset,
                xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up])
y= np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].mass,
              xr_agn_props['x4_bad_o3_hx_allz_noext'].mass])
r = pearsonr(x,y)
r = str(round(r[0],2))
scatter(x,y,
        minx=-2.1, maxx=2.1,miny=8.5, maxy=12.5, aspect=25/16, bin_y=True, size_y_bin=0.6, counting_thresh=4,linecolor='k',
        percentiles=False, bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up, 
            xr_agn_props['x4_bad_o3_hx_allz_noext'].mass, marker='s', color='b', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(M$_{*}$)')
plt.xlim([-2,2])
plt.ylim([8.5,12.5])
plt.xticks([-2,-1,0,1,2])

plt.yticks([9, 10, 11,12])
plt.legend()
plt.gca().invert_xaxis()
plt.text(1.9, 8.7, 'Pearson R ='+r, fontsize=12)

plt.tight_layout()
plt.savefig('plots/dlo3_mass.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_mass.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)

scatter(np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset,
                  xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up),
        np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].mbh,xr_agn_props['x4_bad_o3_hx_allz_noext'].mbh),
        
        minx=-2.1, maxx=2.1,miny=6, maxy=10, aspect=25/16, bin_y=True, size_y_bin=0.6, counting_thresh=4,linecolor='k',
        percentiles=False, bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].mbh, marker='s', color='r', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(M$_{\mathrm{BH}}$)')
plt.xlim([-2,2])
plt.ylim([6.,10.])
plt.xticks([-2,-1,0,1,2])

plt.yticks([6.5, 7.5,8.5,9.5])
plt.legend()
plt.tight_layout()
plt.savefig('plots/dlo3_mbh.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_mbh.png', dpi=250, bbox_inches='tight')



fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up),
        np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].mbh,xr_agn_props['x4_bad_o3_hx_allz_noext'].mbh),linecolor='k',
        
        minx=-2.1, maxx=2.1,miny=6, maxy=10, aspect=25/16, bin_y=True, size_y_bin=0.6, counting_thresh=4,
        percentiles=False, bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].mbh, marker='s', color='r', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(M$_{\mathrm{BH}}$)')
plt.xlim([-2,2])
plt.ylim([6.,10.])
plt.xticks([-2,-1,0,1,2])

plt.yticks([6.5, 7.5,8.5,9.5])
plt.legend()
plt.tight_layout()
plt.savefig('plots/dlo3_mbh.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_mbh.png', dpi=250, bbox_inches='tight')



fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up),
        np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].edd_ratio,xr_agn_props['x4_bad_o3_hx_allz_noext'].edd_ratio),
        
        minx=-2.1, maxx=2.1,miny=-5, maxy=-1, aspect=25/16, bin_y=True, size_y_bin=0.6, counting_thresh=4,linecolor='k',
        percentiles=False, bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].edd_ratio, marker='s', color='r', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(Edd. Ratio)')
plt.xlim([-2,2])
plt.ylim([-5.,0.])
plt.xticks([-2,-1,0,1,2])

plt.yticks([-4.5, -3.5,  -2.5, -1.5, -0.5])
plt.legend()
plt.tight_layout()
plt.savefig('plots/dlo3_edd_rat.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_edd_rat.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up),
        np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].edd_par_xr,xr_agn_props['x4_bad_o3_hx_allz_noext'].edd_par_xr),
        
        minx=-2.1, maxx=2.1,miny=-6, maxy=-1, aspect=25/16, bin_y=True, size_y_bin=0.6, counting_thresh=4,linecolor='k',
        percentiles=False, bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].edd_par_xr, marker='s', color='r', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(Edd. Par. (XR) )')
plt.xlim([-2,2])
#plt.ylim([-5.,0.])
plt.xticks([-2,-1,0,1,2])

#plt.yticks([-4.5, -3.5,  -2.5, -1.5, -0.5])
plt.legend()
plt.tight_layout()
plt.savefig('plots/dlo3_edd_parxr.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_edd_parxr.png', dpi=250, bbox_inches='tight')




fig  = plt.figure(figsize=(4,5))
x  = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up])
y = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].oiha,xr_agn_props['x4_bad_o3_hx_allz_noext'].oiha])
r = pearsonr(x,y)
r = str(round(r[0],2))

scatter(x,y, aspect=12.5/4, minx=-2.1, maxx=2.1,miny=-2, maxy=0, bin_y=True, size_y_bin=0.64, counting_thresh=4,linecolor='k',
             percentiles=False, plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s')
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_hard'].lo3_offset, 
            xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_hard'].oiha, marker='s', color='r', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log([OI]/H$\alpha$)')
plt.xlim([-2,2])
plt.ylim([-2,0])
plt.yticks([-2, -1.5, -1,-0.5, 0])
plt.xticks([-2,-1,0,1,2])

plt.legend()
plt.tight_layout()
plt.savefig('plots/dlo3_log_oiha.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_log_oiha.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
x = xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset.iloc[np.where(np.isfinite(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].U)) ], 

y= xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].U.iloc[np.where(np.isfinite(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].U)) ]
r = pearsonr(x,y)
r = str(round(r[0],2))

scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf_ionpar'].lo3_offset,
        xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf_ionpar'].U,
             aspect=3.125,minx=-2.1, maxx=2.1,miny=-4, maxy=-2, bin_y=True, size_y_bin=0.6, counting_thresh=4,linecolor='k',
             plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s')
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up, xr_agn_props['x4_bad_o3_hx_allz_noext'].U, marker='s', color='r', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(U)')
plt.xlim([-2,2])
plt.ylim([-4,-2])
plt.xticks([-2,-1,0,1,2])

plt.yticks([-4, -3.5, -3,-2.5, -2])

plt.gca().invert_xaxis()
plt.text(.2, -2.2, 'Pearson R ='+r, fontsize=12)

plt.legend()
plt.tight_layout()
plt.savefig('plots/dlo3_u.pdf',  dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_u.png',  dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))

x = xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf_oh'].lo3_offset
y=xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf_oh'].log_oh
r = pearsonr(x,y)
r = str(round(r[0],2))
scatter(np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf_oh'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_oh'].lo3_offset),
        np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf_oh'].log_oh, xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_oh'].log_oh),
        aspect=6+1/4, minx=-2.1, maxx=2.1,miny=8.2, maxy=9.2, bin_y=True, size_y_bin=0.6, counting_thresh=4,linecolor='k',
        plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s')
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_oh'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_oh'].log_oh, marker='s', color='r', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(O/H)+12')
plt.xlim([-2,2])
plt.ylim([8.2, 9.2])
plt.yticks([8.4, 8.6, 8.8, 9, 9.2])
plt.xticks([-2,-1,0,1,2])

plt.legend()
plt.tight_layout()
plt.savefig('plots/dlo3_log_oh.pdf',  dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_log_oh.png',  dpi=250, bbox_inches='tight')

fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)

x = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset, 
               xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up])
y = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].av_gsw, 
               xr_agn_props['x4_bad_o3_hx_allz_noext'].av_gsw])
r = pearsonr(x,y)
r = str(round(r[0],2))
scatter(x,y,aspect=4+1/6, minx=-2.1, maxx=2.1,miny=0, maxy=1.5, linecolor='k',bin_y=True, 
        size_y_bin=0.6, counting_thresh=4,bin_stat_y='median', plotdata=True, 
        facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up,
            xr_agn_props['x4_bad_o3_hx_allz_noext'].av_gsw, marker='s', color='b', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'$A_{V\mathrm{,\ Stellar}}$')
plt.xlim([-2,2])
plt.ylim([0,1.5])
plt.xticks([-2,-1,0,1,2])

plt.legend()

plt.gca().invert_xaxis()
plt.text(.75, 1.3, 'Pearson R ='+r, fontsize=12)

plt.tight_layout()
plt.savefig('plots/dlo3_av.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_av.png',  dpi=250, bbox_inches='tight')



fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)


x = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset.iloc[np.where(np.isfinite(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].irx))],
               xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].irx))] ]) 
y=np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].irx.iloc[np.where(np.isfinite(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].irx))],
             xr_agn_props['x4_bad_o3_hx_allz_noext'].irx.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].irx))]  ])
r = pearsonr(x,y)
r = str(round(r[0],2))
scatter(x,y,   aspect=4+1/6, minx=-2.1, maxx=2.1,miny=0, maxy=1.5, bin_y=True, size_y_bin=0.6, counting_thresh=4,linecolor='k',
        bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].irx))] ,
            xr_agn_props['x4_bad_o3_hx_allz_noext'].irx.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].irx))], marker='s', color='blue', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'IR Excess [dex]')
plt.xlim([-2,2])
plt.ylim([0,1.5])
plt.xticks([-2,-1,0,1,2])

plt.legend()

plt.gca().invert_xaxis()
plt.text(.75, 1.3, 'Pearson R ='+r, fontsize=12)

plt.tight_layout()

plt.savefig('plots/dlo3_irx.pdf',  dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_irx.png',  dpi=250, bbox_inches='tight')




fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)


x = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset.iloc[np.where(np.isfinite(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].axisrat))],xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].axisrat))]]) 
y = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].axisrat.iloc[np.where(np.isfinite(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].axisrat))], xr_agn_props['x4_bad_o3_hx_allz_noext'].axisrat.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].axisrat))]]) 
r = pearsonr(x,y)
r = str(round(r[0],2))
scatter(x,
        y, 
             aspect=25/4,linecolor='k', minx=-2.1, maxx=2.1,miny=0, maxy=1.5, bin_y=True, size_y_bin=0.6, counting_thresh=4,  bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s',ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].axisrat))] ,
            xr_agn_props['x4_bad_o3_hx_allz_noext'].axisrat.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].axisrat))], marker='s', color='blue', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'Axis Ratio')
plt.xlim([-2,2])
plt.ylim([0,1.])
plt.xticks([-2,-1,0,1,2])

plt.legend()
plt.yticks([0, 0.25, 0.5, 0.75, 1])

plt.gca().invert_xaxis()
plt.text(.75, 0.1, 'Pearson R ='+r, fontsize=12)

plt.tight_layout()
plt.savefig('plots/dlo3_axis.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_axis.png', dpi=250, bbox_inches='tight')




fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
x = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset.iloc[np.where(np.isfinite(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].d4000))],xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].d4000))]  ])
y = np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].d4000.iloc[np.where(np.isfinite(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].d4000))],xr_agn_props['x4_bad_o3_hx_allz_noext'].d4000.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].d4000))]  ])
r = pearsonr(x,y)
r = str(round(r[0],2))
scatter(x,y,             aspect=4+1/6, minx=-2.1, maxx=2.1,miny=1,linecolor='k', maxy=2.5, bin_y=True, 
        size_y_bin=0.6, counting_thresh=4,  bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s',ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].d4000))] ,
            xr_agn_props['x4_bad_o3_hx_allz_noext'].d4000.iloc[np.where(np.isfinite(xr_agn_props['x4_bad_o3_hx_allz_noext'].d4000))], marker='s', color='blue', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'D$_{4000}$')
plt.xlim([-2,2])
plt.ylim([1,2.5])
plt.xticks([-2,-1,0,1,2])

plt.legend()
plt.yticks([1, 1.25, 1.5, 1.75, 2, 2.25])

plt.gca().invert_xaxis()
plt.text(.75, 2.2, 'Pearson R ='+r, fontsize=12)

plt.tight_layout()


plt.savefig('plots/dlo3_d4000.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_d4000.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset),
        np.append(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].uv_col, xr_agn_props['x4_bad_o3_hx_allz_noext'].uv_col), 
             aspect=2, minx=-2.1, maxx=2.1,miny=-2, maxy=2, bin_y=True, size_y_bin=0.6, counting_thresh=4,  bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s',ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].uv_col, marker='s', color='r', s=15)

plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'FUV-NUV')
plt.xlim([-2,2])
plt.ylim([0,2.5])
plt.xticks([-2,-1,0,1,2])

plt.legend()
plt.yticks([1, 1.25, 1.5, 1.75, 2, 2.25])
plt.tight_layout()
plt.savefig('plots/dlo3_uvcol.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_uvcol.png', dpi=250, bbox_inches='tight')



fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset,xr_agn_props['bptplus_sf_allxragn'].lo3_offset]),
        np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].d4000, xr_agn_props['x4_bad_o3_hx_allz_noext'].d4000,xr_agn_props['bptplus_sf_allxragn'].d4000 ]), 
             aspect=4+1/6, minx=-2.1, maxx=2.1,miny=-2, maxy=2, bin_y=True, size_y_bin=0.6, counting_thresh=4,  bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s',ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].d4000, marker='s', color='r', s=15)
plt.scatter(xr_agn_props['bptplus_sf_allxragn'].lo3_offset, xr_agn_props['bptplus_sf_allxragn'].d4000, marker='^', color='g', s=15)
plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'D$_{4000}$')
plt.xlim([-2,2])
plt.ylim([1,2.5])
plt.xticks([-2,-1,0,1,2])

plt.legend()
plt.yticks([1, 1.25, 1.5, 1.75, 2, 2.25])
plt.tight_layout()
plt.savefig('plots/dlo3_d4000_w_bptsf.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_d4000_w_bptsf.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].lo3_offset, 
                   xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset,
                   xr_agn_props['bptplus_sf_allxragn'].lo3_offset2]),
        np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].d4000, xr_agn_props['x4_bad_o3_hx_allz_noext'].d4000,xr_agn_props['bptplus_sf_allxragn'].d4000 ]), 
             aspect=4+1/6, minx=-2.1, maxx=2.1,miny=-2, maxy=2, bin_y=True, size_y_bin=0.6, counting_thresh=4,  bin_stat_y='median',plotdata=True,  facecolor='magenta', edgecolor='magenta',marker='s',ax=ax1, fig = fig)
#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset, 
            xr_agn_props['x4_bad_o3_hx_allz_noext'].d4000, marker='s', color='r', s=15)
plt.scatter(xr_agn_props['bptplus_sf_allxragn'].lo3_offset2,
            xr_agn_props['bptplus_sf_allxragn'].d4000, marker='^', color='g', s=15)
plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'D$_{4000}$')
plt.xlim([-2,2])
plt.ylim([1,2.5])
plt.xticks([-2,-1,0,1,2])

plt.legend()
plt.yticks([1, 1.25, 1.5, 1.75, 2, 2.25])
plt.tight_layout()
plt.savefig('plots/dlo3_d4000_w_bptsf_sfsub.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_d4000_w_bptsf_sfsub.png', dpi=250, bbox_inches='tight')

fig  = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(redshifts_[:-1]+0.012, o3_1[:-1],  c='darkturquoise', linewidth=0.5)
ax1.scatter(redshifts_+0.012, o3_1,  c='darkturquoise', marker='*', s=50, label=r'[OIII] 1$\sigma$', zorder=10)

ax1.plot(redshifts_[:-1]+0.01, o3_2[:-1],  c='m', linewidth=0.5)
ax1.scatter(redshifts_+0.01, o3_2,  c='m', marker='o', s=50, label=r'[OIII] 2$\sigma$', zorder=10)
ax1.minorticks_off()


ax1.plot(redshifts_[:-1]+0.005, bpt_2sig_n2[:-1],  c='r', linewidth=0.5)
ax1.scatter(redshifts_+0.005, bpt_2sig_n2,  c='r', marker='^', s=50, label=r'[NII]\&H$\alpha$ 2$\sigma$', zorder=10)


ax1.plot(redshifts_[:-1], bpt_2sig_n2_o3_1[:-1],  c='g', linewidth=0.5)
ax1.scatter(redshifts_, bpt_2sig_n2_o3_1,  c='g', marker='<', s=50, label=r'[NII]\&H$\alpha$ 2$\sigma$,'+ '\n'+ r'[OIII] 1$\sigma$', zorder=10)


ax1.plot(redshifts_[:-1]-0.005, bpt_2sig[:-1],  c='k', linewidth=0.5)
ax1.scatter(redshifts_-0.005, bpt_2sig,  c='k', marker='+', s=50, label=r'BPT 2$\sigma$', zorder=10)

ax1.plot(redshifts_[:-1]-0.01, bpt_3sig[:-1],  c='b', linewidth=0.5)
ax1.scatter(redshifts_-0.01, bpt_3sig,  c='b', marker='s', s=50, label=r'BPT 3$\sigma$', zorder=10)




for i in range(6):
    ax1.axvline(0.05+0.05*i, linewidth=0.5)
    ax1.text(0.01+0.05*i, 0.1, 'N='+str(nbins[i]))
ax1.text(0.35, 0.5,'N$_{\mathrm{ Tot.}}$='+str(nbins[-1]))

ax1.set_xticks([0.05, 0.15, 0.25, 0.375])
ax1.set_xticklabels([0.05, 0.15, 0.25, 'all'])    
ax1.set_xlim([0, 0.45])
ax1.set_ylim([0,1])
plt.legend(loc=4, fontsize=10)
ax1.set_xlabel('Redshift')
ax1.set_ylabel('Fraction')


plt.savefig('plots/f_classifiable.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_classifiable.png', bbox_inches='tight', dpi=250)
xinp = np.arange(0,3,0.2)+0.1

gauss_ideal = gaussian(xinp,0.587, 0,341*2)
fig  = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(211)
ax1.set_aspect(2)
#ax1.axvline(x=0.587, label=r'1 $\sigma$')
plothist(-xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_nobptsf'].lo3_offset, range=(0, 3), 
         bins=10, norm0=True, normed=True, label='X-ray AGN Candidates', linestyle='-',linewidth=2.5)
ax1.plot(xinp, gauss_ideal/np.max(gauss_ideal), '--',linewidth=2.5, color='gray',label='Expected Gaussian')

ax1.legend(fontsize=20)
ax2 = fig.add_subplot(212)
ax2.set_aspect(2)
ax2.plot(xinp, gauss_ideal/np.max(gauss_ideal), '--',linewidth=2.5, color='gray', label='Expected Gaussian')

#bncenters, cnt1, int_ = plothist(-xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_offset.iloc[np.where(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].ext==0)[0]], range=(0, 3), bins=10, norm0=False, normed=False, label='Point Source')
plothist(-xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_noext_nobptsf'].lo3_offset, linestyle='-',
         range=(0,3), bins=10, norm0=True, normed=True, label='Point Source', c='magenta', linewidth=2.5)

bncenters,cnt2, int_ = plothist(-xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_ext_nobptsf'].lo3_offset,
                                range=(0, 3), bins=10, normval=np.max(cnt1), normed=True, linestyle='-',label='Extended',linewidth=2.5, c='red')

plt.tight_layout()
ax2.legend(fontsize=20)
#ax2.axvline(x=0.587)

ax1.set_xlabel('')
ax1.set_xticks([0.5, 1, 1.5, 2, 2.5])
ax2.set_ylim([0,1.05])
ax1.set_ylim([0,1.05])

ax1.set_xticklabels('')
ax2.set_xticks([0.5, 1, 1.5, 2, 2.5])
ax2.set_xticklabels([-0.5, -1, -1.5, -2, -2.5])
ax1.set_xlim([0,3])
ax2.set_xlim([0,3])


ax2.set_xlabel('$\Delta$log(L$_{\mathrm{[OIII]}}$)')

ax1.set_ylabel('Norm. Counts')
ax2.set_ylabel('Norm. Counts')

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig('plots/hist_delta_lo3.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hist_delta_lo3.png', bbox_inches='tight', dpi=250)



fig  = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111)
ax1.set_aspect(2)
plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_nobptsf'].lo3_offset, range=(0, 3), 
         bins=10, norm0=True, normed=True, label='X-ray AGN Candidates', linestyle='-',linewidth=2.5)
ax1.plot(xinp, gauss_ideal/np.max(gauss_ideal), '--',linewidth=2.5, color='r',label='Ideal Gaussian')
ax1.legend(fontsize=20)

plt.tight_layout()
ax1.legend(fontsize=20)
#ax2.axvline(x=0.587)
ax1.set_xticks([0.5, 1, 1.5, 2, 2.5])
ax1.set_ylim([0,1.05])
ax1.set_xlim([0,3])
ax1.set_xlabel('$\Delta$log(L$_{\mathrm{[OIII]}}$)')
ax1.set_ylabel('Norm. Counts')
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)



plt.savefig('plots/hist_delta_lo3_og.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hist_delta_lo3_og.png', bbox_inches='tight', dpi=250)




fig  = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111)
ax1.set_aspect(2)


ax1.plot(xinp, gauss_ideal/np.max(gauss_ideal), '--',linewidth=2.5, color='r')

#bncenters, cnt1, int_ = plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_offset.iloc[np.where(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].ext==0)[0]], range=(0, 3), bins=10, norm0=False, normed=False, label='Point Source')
plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_noext_nobptsf'].lo3_offset, linestyle='-',
         range=(0, 3), bins=10, norm0=True, normed=True, label='Point Source', c='magenta', linewidth=2.5)
bncenters,cnt2, int_ = plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_ext_nobptsf'].lo3_offset,
                                range=(0, 3), bins=10, normval=np.max(cnt1), normed=True, linestyle='-',label='Extended',linewidth=2.5, c='darkturquoise')
plt.tight_layout()
ax1.legend(fontsize=20)
#ax2.axvline(x=0.587)
ax1.set_xticks([0.5, 1, 1.5, 2, 2.5])
ax1.set_ylim([0,1.05])
ax1.set_xlim([0,3])
ax1.set_xlabel('$\Delta$log(L$_{\mathrm{[OIII]}}$)')
ax1.set_ylabel('Norm. Counts')
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)



plt.savefig('plots/hist_delta_lo3_split.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hist_delta_lo3_split.png', bbox_inches='tight', dpi=250)








fig  = plt.figure()

ax1 = fig.add_subplot(111)

ax1.plot(x_panessa, geny_, color='r', linewidth=3, zorder=0)

scatter(xr_agn_props['x4_sn3_o3_hx_allz_noext_nobptsf'].hard_xraylum, xr_agn_props['x4_sn3_o3_hx_allz_noext_nobptsf'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=43.8, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True,
             facecolor='none', edgecolor='k', fig=fig, ax=ax1, s=7)

#ax1.plot(xliu_hard_dc[yavliu_hard_dc>0], yavliu_hard_dc[yavliu_hard_dc>0], color='c', linewidth=3, label='Type 1')
ax1.plot(xt2_hard, yavt2_hard, color='k', linewidth=3, label='Type 2', zorder=10)


ax1.plot(xliner_hard, yavliner_hard, color='orange', linewidth=3, label='LINER')
ax1.plot(xsy2_hard, yavsy2_hard, color='b', linewidth=3, label='Sy2')

scatter(xr_agn_props['combo_all_liners_hx_noext'].hard_xraylum, xr_agn_props['combo_all_liners_hx_noext'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=43.8, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True,s=15,
             color='orange', edgecolor='orange', fig=fig, ax=ax1)
scatter(xr_agn_props['combo_sy2_hx_noext'].hard_xraylum, xr_agn_props['combo_sy2_hx_noext'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=43.8, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, s=15,
             color='blue', edgecolor='blue', fig=fig, ax=ax1)


ax1.set_xticks([38,40,42,44])
ax1.legend(fontsize=12)
#plt.yticks([38,40,42,44])
ax1.set_xlabel(r'log(L$_{\mathrm{X,\ 2-10\ keV}}$)')
ax1.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')

plt.tight_layout()
ax1.set_aspect('equal')


plt.savefig('plots/lxo3_by_type_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxo3_by_type_.png', bbox_inches='tight', dpi=250)



fig  = plt.figure()

ax1 = fig.add_subplot(111)


ax1.set_xlabel('log(L$_{\mathrm{X,\ 2-10\ keV}}$)')

ax1.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')


scatter(xr_agn_props['x4_sn1_o3_hx_allz_nobptsf'].hard_xraylum, xr_agn_props['x4_sn1_o3_hx_allz_nobptsf'].oiiilum, 
        minx=38, maxx=46, miny=38, maxy=43.8, label='X-ray AGN Candidates', s=5,edgecolor='k', facecolor='k', fig = fig, ax = ax1)

ax1.plot(x_panessa, geny_, color='red', linewidth=3, label='Panessa+06', zorder=0)
#ax1.plot(x_panessa+0.587, geny_, 'r--', linewidth=3, label=r'Panessa+06 +1$\sigma$')
ax1.set_aspect('equal')
handles, labels = ax1.get_legend_handles_labels()
order = [1,0]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=15)

ax1.set_xticks([38,40,42,44])

plt.savefig('plots/lxo3_full_selection_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxo3_full_selection_.png', bbox_inches='tight', dpi=250)



fig  = plt.figure()

ax1 = fig.add_subplot(111)


ax1.set_xlabel('log(L$_{\mathrm{X,\ 2-10\ keV}}$)')

ax1.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')


scatter(xr_agn_props['x4_sn1_o3_hx_allz_nobptsf'].hard_xraylum, 
        xr_agn_props['x4_sn1_o3_hx_allz_nobptsf'].oiiilum, 
        ccode=np.log10(xr_agn_props['x4_sn1_o3_hx_allz_nobptsf'].oiiiflux_sn), vmin=np.log10(1), vmax=np.log10(100),
        minx=38, maxx=46, miny=38, maxy=43.8, label='X-ray AGN Candidates', s=5,edgecolor='k',
        cmap='rainbow_r', facecolor='k', fig = fig, ax = ax1)

ax1.plot(x_panessa, geny_, color='red', linewidth=3, label='Panessa+06', zorder=0)
#ax1.plot(x_panessa+0.587, geny_, 'r--', linewidth=3, label=r'Panessa+06 +1$\sigma$')
ax1.set_aspect('equal')
handles, labels = ax1.get_legend_handles_labels()
order = [1,0]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=15)
norm = mpl.colors.Normalize(vmin=-1, vmax=2)

cbar_ax = fig.add_axes([0.925, 0.15, 0.03, 0.7])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='rainbow_r'),
             cax=cbar_ax)

cbar.ax.tick_params(labelsize=20)
cbar.solids.set_edgecolor('face')
cbar.set_label('log([OIII] S/N)')
ax1.set_xticks([38,40,42,44])

plt.savefig('plots/lxo3_full_selection_o3_sn.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxo3_full_selection_o3_sn.png', bbox_inches='tight', dpi=250)



fig  = plt.figure()

ax1 = fig.add_subplot(111)


ax1.set_xlabel('log(L$_{\mathrm{X,\ 2-10\ keV}}$)')

ax1.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')


scatter(xr_agn_props['bptplus_sf_allxragn'].hard_xraylum, xr_agn_props['bptplus_sf_allxragn'].oiiilum, 
        minx=38, maxx=46, miny=38, maxy=43.8, label='BPT-SF AGN Candidates', s=15,edgecolor='g', facecolor='g',marker='^', fig = fig, ax = ax1)

ax1.plot(x_panessa, geny_, color='red', linewidth=3, label='Panessa+06', zorder=0)
#ax1.plot(x_panessa+0.587, geny_, 'r--', linewidth=3, label=r'Panessa+06 +1$\sigma$')
ax1.set_aspect('equal')
handles, labels = ax1.get_legend_handles_labels()
order = [1,0]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=15)

ax1.set_xticks([38,40,42,44])

plt.savefig('plots/lxo3_bpt_sf_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxo3_bpt_sf_.png', bbox_inches='tight', dpi=250)


fig  = plt.figure()

ax1 = fig.add_subplot(111)


ax1.set_xlabel('log(L$_{\mathrm{X,\ 2-10\ keV}}$)')

ax1.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')



scatter(xr_agn_props['bptplus_sf_allxr'].hard_xraylum.iloc[np.where(( xr_agn_props['bptplus_sf_allxr'].ext==0)&
                                                                    (xr_agn_props['bptplus_sf_allxr'].xray_agn_status==False))],
        xr_agn_props['bptplus_sf_allxr'].oiiilum.iloc[np.where(( xr_agn_props['bptplus_sf_allxr'].ext==0)&
                                                                    (xr_agn_props['bptplus_sf_allxr'].xray_agn_status==False))], 
        minx=38, maxx=46, miny=38, maxy=43.8, label='X-ray non-AGNs', s=15,edgecolor='g', facecolor='g',marker='^', fig = fig, ax = ax1)


ax1.plot(x_panessa, geny_, color='red', linewidth=3, label='Panessa+06', zorder=0)
#ax1.plot(x_panessa+0.587, geny_, 'r--', linewidth=3, label=r'Panessa+06 +1$\sigma$')
ax1.set_aspect('equal')
handles, labels = ax1.get_legend_handles_labels()
order = [1,0]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=15)

ax1.set_xticks([38,40,42,44])


fig  = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
ax1.set_xlabel('log(L$_{\mathrm{X,\ 2-10\ keV}}$)')

scatter(xmm4eldiagmed_xrsffilt.bptplus_sf_df.hard_xraylum.iloc[np.where((xmm4eldiagmed_xrsffilt.bptplus_sf_df.hardflux_sn>1)&(xmm4eldiagmed_xrsffilt.bptplus_sf_df.ext==0))],
        xmm4eldiagmed_xrsffilt.bptplus_sf_df.oiiilum.iloc[np.where((xmm4eldiagmed_xrsffilt.bptplus_sf_df.hardflux_sn>1)&(xmm4eldiagmed_xrsffilt.bptplus_sf_df.ext==0))], 
        minx=38, maxx=46, miny=38, maxy=43.8, label='X-ray non-AGNs', s=15,edgecolor='g', facecolor='g',marker='^', fig = fig, ax = ax1)

ax1.plot(x_panessa, geny_, color='red', linewidth=3, label='Panessa+06', zorder=0)
#ax1.plot(x_panessa+0.587, geny_, 'r--', linewidth=3, label=r'Panessa+06 +1$\sigma$')
ax1.set_aspect('equal')
handles, labels = ax1.get_legend_handles_labels()
order = [1,0]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=15)

ax1.set_xticks([38,40,42,44])

plt.savefig('plots/lxo3_nonagns_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxo3_nonagns_.png', bbox_inches='tight', dpi=250)



fig  = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')

ax1.set_xlabel('log(L$_{\mathrm{X,\ 2-10\ keV}}$)')

ax1.plot(x_panessa, geny_, color='red', linewidth=3, zorder=0)
#ax1.plot(x_panessa+0.587, geny_, 'r--', linewidth=3)
scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].hard_xraylum, xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].oiiilum, 
        minx=38, maxx=46, miny=38, maxy=43.8, label='Point Source', facecolor='magenta', edgecolor='magenta', marker='s',s=10,fig = fig, ax = ax1)
scatter(xr_agn_props['x4_sn1_o3_hx_allz_ext_nobptsf'].hard_xraylum, xr_agn_props['x4_sn1_o3_hx_allz_ext_nobptsf'].oiiilum, 
        minx=38, maxx=46, miny=38, maxy=43.8, facecolor='red', 
        edgecolor='k',marker='^',color='red', label='Extended',s=20, fig = fig, ax = ax1)
ax1.legend(fontsize=15)

plt.tight_layout()

ax1.set_xticks([38,40,42,44])


plt.savefig('plots/lxo3_ext_noext_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxo3_ext_noext_.png', bbox_inches='tight', dpi=250)




fig  = plt.figure()

ax1 = fig.add_subplot(111)


plot2dhist(EL_m2.EL_gsw_df.mass, EL_m2.EL_gsw_df.ssfr, minx=7.5, maxx=12.5, miny=-14,maxy=-8)

scatter(xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].mass, xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'].ssfr, 
         label='Point Source', facecolor='magenta', edgecolor='magenta', marker='s',s=10,fig = fig, ax = ax1)
scatter(xr_agn_props['x4_sn1_o3_hx_allz_ext_nobptsf'].mass, xr_agn_props['x4_sn1_o3_hx_allz_ext_nobptsf'].ssfr, 
         facecolor='red', 
        edgecolor='black',marker='^',color='red', label='Extended',s=20, fig = fig, ax = ax1, aspect=2/3)
ax1.legend(fontsize=15, loc=3)

plt.tight_layout()

ax1.set_xticks([8,9,10,11,12])

ax1.set_yticks([-13,-12,-11,-10,-9])
ax1.set_xlim([7.5, 12.5])
ax1.set_ylim([-14,-8.])


ax1.set_xlabel(r'log(M$_{*}$)')

ax1.set_ylabel('log(sSFR)')
plt.savefig('plots/ssfr_m_ext_noext_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/ssfr_m_ext_noext_.png', bbox_inches='tight', dpi=250)



fig  = plt.figure()

ax1 = fig.add_subplot(111)


plotbptnormal(EL_m2.bpt_EL_gsw_df.niiha, EL_m2.bpt_EL_gsw_df.oiiihb, nobj=False, ax = ax1, fig= fig, maxy=1.4,maxx=1., minx=-2, miny=-1.4, setplotlims=True)
scatter(xr_agn_props['all_hx'].niiha[xr_agn_props['all_hx'].ext==0], xr_agn_props['all_hx'].oiiihb[xr_agn_props['all_hx'].ext==0], 
         label='Point Source', facecolor='magenta', edgecolor='magenta', marker='s',s=10,fig = fig, ax = ax1)
scatter(xr_agn_props['all_hx'].niiha[xr_agn_props['all_hx'].ext!=0], xr_agn_props['all_hx'].oiiihb[xr_agn_props['all_hx'].ext!=0], 
         facecolor='red', 
        edgecolor='k',marker='^',color='red', label='Extended',s=20,markerborder=0.5, fig = fig, ax = ax1)
ax1.legend(fontsize=15, loc=3)

plt.tight_layout()


ax1.set_xticks([-2,-1,0,1])

ax1.set_yticks([-1,-0.5,-0,0.5,1])

plt.savefig('plots/bpt_ext_noext.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/bpt_ext_noext.png', bbox_inches='tight', dpi=250)

import matplotlib as mpl


fig  = plt.figure()

ax1 = fig.add_subplot(111)


plotbptnormal(EL_m2.bpt_EL_gsw_df.niiha, EL_m2.bpt_EL_gsw_df.oiiihb, nobj=False, ax = ax1, fig= fig, maxy=1.4,maxx=1., minx=-2, miny=-1.4, setplotlims=True)
scatter(xr_agn_props['all_hx'].niiha[xr_agn_props['all_hx'].ext==0], xr_agn_props['all_hx'].oiiihb[xr_agn_props['all_hx'].ext==0], 
        ccode=xr_agn_props['all_hx'].lo3_offset[xr_agn_props['all_hx'].ext==0], cmap='Blues',
        facecolor='magenta', edgecolor='magenta', vmin=-1, vmax=1, marker='s',s=10,fig = fig, ax = ax1)
ax1.legend(fontsize=15, loc=3)

ax1.set_ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
ax1.set_xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.5)

cbar_ax = fig.add_axes([0.9, 0.2, 0.03, 0.7])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='plasma'),
             cax=cbar_ax)

cbar.set_label(r'$\Delta$ log(L$_{\mathrm{[OIII]}}$)', fontsize=20)
cbar.ax.tick_params(labelsize=20)
ax1.set_xticks([-2,-1,0,1])

ax1.set_yticks([-1,-0.5,-0,0.5,1])

plt.tight_layout()

plt.savefig('plots/bpt_lo3_ccode6.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/bpt_lo3_ccode6.png', bbox_inches='tight', dpi=250)

fig  = plt.figure()

ax1 = fig.add_subplot(111)


plotbptnormal(EL_m2.bpt_EL_gsw_df.niiha, EL_m2.bpt_EL_gsw_df.oiiihb, nobj=False, ax = ax1, fig= fig, maxy=1.4,maxx=1., minx=-2, miny=-1.4, setplotlims=True)
plot2dhist(xr_agn_props['all_hx'].niiha[xr_agn_props['all_hx'].ext==0], xr_agn_props['all_hx'].oiiihb[xr_agn_props['all_hx'].ext==0], 
        ccode=xr_agn_props['all_hx'].lo3_offset[xr_agn_props['all_hx'].ext==0], nx=7, ny=7,ccode_bin_min=3, cmap='Blues',ccodename=r'$\Delta$ log(L$_{\mathrm{[OIII]}}$)',
        ccodelim=[-0.3, 0.5],fig = fig, ax = ax1)
ax1.legend(fontsize=15, loc=3)

ax1.set_ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
ax1.set_xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
norm = mpl.colors.Normalize(vmin=-0.5, vmax=1)


ax1.set_xticks([-2,-1,0,1])

ax1.set_yticks([-1,-0.5,-0,0.5,1])

plt.tight_layout()

plt.savefig('plots/bpt_lo3_ccode3.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/bpt_lo3_ccode3.png', bbox_inches='tight', dpi=250)



xmido3,yavgo3err, av__, perc16o3err, perc84o3err= scatter(xragn_no_sn_cuts.oiiiflux, 
                                                    np.log10(xragn_no_sn_cuts.oiii_err),
                                                    aspect='auto', bin_y=True, 
                                                    percentiles=True, size_y_bin=1e-17, 
                                                    counting_thresh=0, minx=-1e-16, maxx=1e-13, 
                                                    miny=-18, maxy=-14)
xmido3,yavgo3err, av__, perc16o3err, perc84o3err= scatter(np.log10(xragn_no_sn_cuts.oiiiflux-np.min(xragn_no_sn_cuts.oiiiflux)*1.2), 
                                                    np.log10(xragn_no_sn_cuts.oiii_err),
                                                    aspect='auto', bin_y=True, 
                                                    percentiles=True, size_y_bin=0.2, 
                                                    counting_thresh=1, minx=-18, maxx=-12, 
                                                    miny=-18, maxy=-14)
xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['high_sn_o3_bd_no_sf'], 
                                                       fname='high_sn_o3_bd_no_sf', 
                                                       save=False)

plt.close()

a = fwd_model(xr_agn_props['high_sn_o3_bd_no_sf'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['high_sn_o3_bd_no_sf'], 'high_sn_o3_bd_no_sf')

xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['high_sn_lx_o3_bd_no_sf'], 
                                                       fname='high_sn_lx_o3_bd_no_sf', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['high_sn_lx_o3_bd_no_sf'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['high_sn_lx_o3_bd_no_sf'], 'high_sn_lx_o3_bd_no_sf')

xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['o3all'], 
                                                       fname='o3all', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['o3all'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['o3all'], 'o3all')

xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['xrall'], 
                                                       fname='xrall', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['xrall'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['xrall'], 'xrall')

xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['all'], 
                                                       fname='all', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['all'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['all'], 'all')


xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['combo_sy2'], 
                                                       fname='combo_sy2', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['combo_sy2'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)

plot_fwd_results(a, xr_agn_props['combo_sy2'], 'combo_sy2')



xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['combo_all_liners'], 
                                                       fname='combo_all_liners', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['combo_all_liners'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['combo_all_liners'], 'combo_all_liners')


xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['combo_sliner'], 
                                                       fname='combo_sliner', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['combo_sliner'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['combo_sliner'], 'combo_sliner')

xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['combo_hliner'], 
                                                       fname='combo_hliner', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['combo_hliner'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['combo_hliner'], 'combo_hliner')
'''


def flux_distros(lum_pred, dists, av):
    perturbs = np.abs(np.random.normal(size=len(dists), scale=0.58))
    lum_pert = lum_pred-perturbs
    flux_pert = getfluxfromlum(10**lum_pert, dists)
    redflux = redden(flux_pert, av, 5007.)
    return redflux


def bootstrapped_underlum_fracs(nboot=1000000):
    np.random.seed(13)
    f_uls_z07 = []
    f_uls_allz = []
    f_1s_z07 = []
    f_1s_allz = []

    n_z07_ul = len(xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3']) + \
        len(xr_agn_props['x4_sn_lt_1_o3_hx_z07_belowlxo3'])
    n_allz_ul = len(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']) + \
        len(xr_agn_props['x4_sn_lt_1_o3_hx_allz_belowlxo3'])

    n_z07_1 = len(xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3_1dex']) + \
        len(xr_agn_props['x4_sn_lt_1_o3_hx_z07_belowlxo3'])
    n_allz_1 = len(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_1dex']) + \
        len(xr_agn_props['x4_sn_lt_1_o3_hx_allz_belowlxo3'])

    for i in range(nboot):
        gauss_z07 = np.random.normal(
            loc=0, scale=0.59, size=xr_agn_props['x4_hx_z07'].shape[0])
        gauss_allz = np.random.normal(
            loc=0, scale=0.59, size=xr_agn_props['x4_hx_allz'].shape[0])
        npos_z07 = np.where(gauss_z07 >= 0)[0].size
        npos_allz = np.where(gauss_allz >= 0)[0].size
        f_ul_z07 = 1-npos_z07/(n_z07_ul)
        f_ul_allz = 1-npos_allz/(n_allz_ul)
        f_uls_z07.append(f_ul_z07)
        f_uls_allz.append(f_ul_allz)

        n1_z07 = np.where(gauss_z07 >= 1)[0].size
        n1_allz = np.where(gauss_allz >= 1)[0].size
        f_1_z07 = 1-n1_z07/(n_z07_1)
        f_1_allz = 1-n1_allz/(n_allz_1)
        f_1s_z07.append(f_1_z07)
        f_1s_allz.append(f_1_allz)


def gaussian(x, sigma, mean, factor=1):
    return factor*np.exp(-(x-mean)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)


def integrate(x, y):
    int_ = scipy.integrate.simps(y, x=x)
    return int_


def mocksim(xrlum, z, av, f_ul,  sig2, sl2, int2, sl1=1.22, int1=7.55, sig1=0.587, n_samps=100):

    sampd = np.random.uniform(size=(n_samps, len(xrlum)))
    mocklums = np.zeros_like(sampd)
    xrlums = np.zeros_like(sampd)

    mockfluxes = np.zeros_like(sampd)
    for i in range(n_samps):
        regsampd = np.where(sampd[i] > f_ul)[0]
        mock_sim = np.where(sampd[i] <= f_ul)[0]
        mocklums[i, regsampd] = (xrlum[regsampd]+int1)/sl1 - \
            np.abs(np.random.normal(scale=0.58, size=len(regsampd)))
        mocklums[i, mock_sim] = (xrlum[mock_sim]+int2)/sl2 + \
            np.random.normal(scale=sig2, size=len(mock_sim))
        mockfluxes[i] = redden(getfluxfromlum(10**mocklums[i], z), av, 5007.)
    return mocklums, mockfluxes


def lnprior(p):
    # The parameters are stored as a vector of values, so unpack them
    ful, sig2, sl2, int2 = p
    # We're using only uniform priors, and only eps has a lower bound
    if sig2 <= 0.3 or sig2 > 0.8 or sl2 <= 3.8 or sl2 > 4.2 or ful < 0.05 or ful > 0.4 or int2 > 112 or int2 < 108:
        return -np.inf
    return 0


def lnlike(p, xrlum, z, av, oiiiflux):
    ful, sig2, sl2, int2 = p
    modellum, modelflux = mocksim(xrlum, z, av, ful, sig2, sl2, int2)
    # the likelihood is sum of the lot of normal distributions
    ks_ = ks_2samp(np.array(oiiiflux), modelflux.flatten())[0]
    return -ks_


def lnprob(p, xrlum, z, av, oiiiflux):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p,  xrlum, z, av, oiiiflux)


def getks(x1, x2):
    ks_stat = ks_2samp(x1, x2)[0]
    return ks_stat


'''
np.random.seed(13)

import scipy.optimize as opt
nll = lambda *args: -lnprob(*args)
result = opt.minimize(nll, [0.23,0.5, 4, 110],
                      args=(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['z']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['corrected_presub_av']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['oiiiflux']),
                                      ))

ndim, nwalkers = 4, 500
p0 = [result['x']+1e-2*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,
                                args=(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['z']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['corrected_presub_av']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['oiiiflux']),
                                      ))

pos,prob,state = sampler.run_mcmc(p0, 1000, progress=True)

offsets = np.arange(0.,5,0.1)
fuls = np.arange(0.1, 0.5, 0.01)
sigmas = np.arange(0.2, 0.6, 0.02)
sl2 = np.arange(3.,5, 0.1)
int2 = np.arange(60,150,1)
sampgrid = np.meshgrid( fuls, sigmas, sl2, int2)

fuls = sampgrid[0]
sigmas=sampgrid[1]
sl2 = sampgrid[2]
int2 = sampgrid[3]

ks_samps = np.copy(sampgrid[0])*-999
ks_sampslum = np.copy(sampgrid[0])*-999

pvals = np.copy(ks_samps)
sh_samp = sampgrid[0].shape
print(len(np.ravel(sampgrid[0])))

for i in range(len(np.ravel(sampgrid[0]))):
    if i %100==0:
        print(i)
    unraved = np.unravel_index(i, sampgrid[0].shape)
    modellums, modelflux = mocksim(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['hard_xraylum']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['z']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['corrected_presub_av']),
                                      fuls[unraved],   sigmas[unraved], sl2[unraved], int2[unraved], n_samps=5)
    ks_ = ks_2samp(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['oiiiflux']), np.ravel(modelflux))
    ks_lums = ks_2samp(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['oiiilum']), np.ravel(modellums))
    
    ks_samps[unraved] =ks_[0]
    ks_sampslum[unraved] =ks_lums[0]
    
    pvals[unraved] = ks_[1]
    
    
ks_ = ks_2samp(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiiflux), mockfluxes)
ks_samps_tot = ks_samps+ks_sampslum

ds = np.argsort(ks_samps.flatten())
ds = np.argsort(ks_sampslum.flatten())
ds = np.argsort(ks_samps_tot.flatten())
i=0
mocklums, mockfluxes = mocksim(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['hard_xraylum']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['z']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['corrected_presub_av']),
                                      fuls.flatten()[ds[i]],  sigmas.flatten()[ds[i]], sl2.flatten()[ds[i]], int2.flatten()[ds[i]], 
                                      n_samps=100)
xray_lm = np.zeros_like(mocklums)

for k in range(mockfluxes.shape[0]):
    xray_lm[k] = np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum'])
    

print(ks_samps.flatten()[ds[i]],ks_sampslum.flatten()[ds[i]],fuls.flatten()[ds[i]],  sigmas.flatten()[ds[i]], sl2.flatten()[ds[i]], int2.flatten()[ds[i]])
scatter(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiilum, minx=38, maxx=46, miny=38, maxy=46)
plt.figure()
scatter(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], mocklums, minx=38, maxx=46, miny=38, maxy=46)

plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiilum, range=(38,43), bins=15,  normed=True, label='Observed', linestyle='-')
plothist(mocklums, range=(38,43), bins=15,label='Simulated', normed=True)
plt.xlim([38,43])
plt.ylim([0,1])

plt.xlabel('log(L[OIII])')
plt.ylabel('Counts')
plt.legend()

plt.tight_layout()

plt.savefig('plots/lum_hist_bestfit_comboks.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lum_hist_bestfit_comboks.png', bbox_inches='tight', dpi=250)

plothist(np.log10(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiiflux), cumulative=True, normed=True, linestyle='-', reverse=True, label='Observed')
plothist(np.log10(mockfluxes), cumulative=True, normed=True, reverse=True, label='Simulated')

plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Counts')
plt.xlim([-13,-17])
plt.ylim([0,1])

plt.legend()
plt.tight_layout()

plt.savefig('plots/flux_hist_bestfit_fluxks.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/flux_hist_bestfit_fluxks.png', bbox_inches='tight', dpi=250)

plot2dhist(np.ravel(xray_lm), np.ravel(mocklums), minx=38, maxx=46, miny=38, maxy=46, setplotlims=True, lim=True, nx=200, ny=200)

scatter(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiilum, minx=38, maxx=46, miny=38, maxy=46)

plt.xlabel('log(Lx)')
plt.ylabel('log(L[OIII])')
plt.xlim([40,46])
plt.ylim([38,44])

plt.tight_layout()

plt.savefig('plots/lxlo3_bestfit_comboks.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxlo3_bestfit_comboks.png', bbox_inches='tight', dpi=250)




plot2dhist(np.ravel(xray_lm), np.ravel(mocklums), minx=38, maxx=46, miny=38, maxy=46, setplotlims=True, lim=True, nx=200, ny=200)

scatter(xr_agn_props['sy2_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], 
        xr_agn_props['sy2_sn1_o3_hx_allz_belowlxo3'].oiiilum, minx=38, maxx=46, miny=38, maxy=46, label='Sy2 below', edgecolor='b', facecolor='b')
scatter(xr_agn_props['hliner_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], 
        xr_agn_props['hliner_sn1_o3_hx_allz_belowlxo3'].oiiilum, minx=38, maxx=46, miny=38, maxy=46, label='H-LINERs below', edgecolor='darkturquoise', facecolor='darkturquoise')
scatter(xr_agn_props['sliner_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], 
        xr_agn_props['sliner_sn1_o3_hx_allz_belowlxo3'].oiiilum, minx=38, maxx=46, miny=38, maxy=46, label='S-LINERs below', edgecolor='orange', facecolor='orange')

plt.plot(x_panessa, geny_, 'r', linewidth=3, label='Panessa+06 without QSOs')
plt.xlabel('log(Lx)')
plt.ylabel('log(L[OIII])')
plt.xlim([40,46])
plt.ylim([38,44])
plt.legend()
plt.tight_layout()

plt.savefig('plots/lxlo3_all_groups_below_relation.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxlo3_all_groups_below_relation.png', bbox_inches='tight', dpi=250)



plot2dhist(np.ravel(xray_lm), np.ravel(mocklums), minx=38, maxx=46, miny=38, maxy=46, setplotlims=True, lim=True, nx=200, ny=200)

scatter(xr_agn_props['sy2_sn1_o3_hx_allz_abovelxo3']['full_xraylum'], xr_agn_props['sy2_sn1_o3_hx_allz_abovelxo3'].oiiilum, 
        minx=38, maxx=46, miny=38, maxy=46, label='Sy2 above', edgecolor='b', facecolor='b')
plt.xlabel('log(Lx)')
plt.ylabel('log(L[OIII])')
plt.xlim([40,46])
plt.ylim([38,44])

plt.tight_layout()

plt.savefig('plots/lxlo3_sy2_above_relation.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxlo3_sy2_above_relation.png', bbox_inches='tight', dpi=250)


x = np.linspace(-5,5, 100000)
gauss_ideal = gaussian(x,0.587, 0,341*2)
hist_z07 = np.histogram(xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].lo3_offset, range=(0, 3),bins=15)[0]
hist_allz = np.histogram(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_nobptsf'].lo3_offset, range=(0, 3.0),bins=15)[0]

+7.55)/(1.22)
fact= 2.0064796705359496
xinp = np.arange(0,3,0.2)+0.1

pnts_by_bin_g = []
gauss_bnc_allz = gaussian(xinp, 0.587,0, factor=341/2.5)
gauss_bnc_z07 = gaussian(xinp, 0.587,0, factor=124/2.5)

pois_g_z07 = np.random.poisson(lam=gauss_bnc_z07, size=(1000000, len(gauss_bnc_z07)))
pois_g_allz = np.random.poisson(lam=gauss_bnc_allz, size=(1000000, len(gauss_bnc_allz)))

pois_o_z07 = np.random.poisson(lam=hist_z07, size=(1000000, len(hist_z07)))
pois_o_allz = np.random.poisson(lam=hist_allz, size=(1000000, len(hist_allz)))

pois_nd_z07 = np.random.poisson(lam=5, size=(1000000))
pois_nd_allz = np.random.poisson(lam=37, size=(1000000))
pois_tot_allz = np.random.poisson(lam=638, size=(1000000))


x_ul2 = np.where(xinp>1)[0]



o_all_z =412
nd_all_z=37
g_all_z = 355

o_all_z =318
nd_all_z=12
g_all_z = 321


o_z07 = 147
nd_z07= 5
g_z07 = 124

o_all_z2 = 87
nd_all_z2= 37
g_all_z2 = 30

o_z072 = 28
nd_z072= 5
g_z072 = 11


f_ul07_g_pois = 1-(np.sum(pois_g_z07, axis=1))/(o_z07+nd_z07)
f_ulall_g_pois = 1-(np.sum(pois_g_allz, axis=1))/(o_all_z+nd_all_z)

f_ul207_g_pois = (o_z072+nd_z072 -np.sum(pois_g_z07[:, x_ul2], axis=1))/(o_z07+nd_z07)
f_ul2all_g_pois = (o_all_z2+nd_all_z2-np.sum(pois_g_allz[:, x_ul2], axis=1))/(o_all_z+nd_all_z)

        

f_ul07_o_nd_pois = 1-g_z07/(np.sum(pois_o_z07, axis=1)+pois_nd_z07)
f_ulall_o_nd_pois = 1-g_all_z/(np.sum(pois_o_allz, axis=1)+pois_nd_allz)


f_ul207_o_nd_pois = (np.sum(pois_o_z07[:, x_ul2], axis=1)+pois_nd_z07-g_z072)/(np.sum(pois_o_z07, axis=1)+pois_nd_z07)
f_ul2all_o_nd_pois = (np.sum(pois_o_allz[:, x_ul2], axis=1)+pois_nd_allz-g_all_z2)/(np.sum(pois_o_allz, axis=1)+pois_nd_allz)


plothist(f_ul07_g_pois, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul_z07_pois_g.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_z07_pois_g.png', bbox_inches='tight', dpi=250)


plothist(f_ulall_g_pois, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul_allz_pois_g.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_allz_pois_g.png', bbox_inches='tight', dpi=250)



        
plothist(f_ul207_g_pois, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_z07_pois_g.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_z07_pois_g.png', bbox_inches='tight', dpi=250)


plothist(f_ul2all_g_pois, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_allz_pois_g.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_allz_pois_g.png', bbox_inches='tight', dpi=250)


plothist(f_ul07_o_nd_pois, bins=20, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul_z07_pois_o_nd.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_z07_pois_o_nd.png', bbox_inches='tight', dpi=250)


plothist(f_ulall_o_nd_pois, bins=20, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul_allz_pois_o_nd.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_allz_pois_o_nd.png', bbox_inches='tight', dpi=250)



        
plothist(f_ul207_o_nd_pois, bins=20, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_z07_pois_o_nd.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_z07_pois_o_nd.png', bbox_inches='tight', dpi=250)


plothist(f_ul2all_o_nd_pois, bins=20, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_allz_pois_o_nd.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_allz_pois_o_nd.png', bbox_inches='tight', dpi=250)




fuls_z07 = []
fuls_allz = []




for i in range(10000000):
    rand07 = np.random.normal(loc=g_z07, scale=np.sqrt(g_z07))
    randall = np.random.normal(loc=g_all_z, scale=np.sqrt(g_all_z))
    ful07 = 1-(rand07)/(o_z07+nd_z07)
    fulall = 1-(randall)/(o_all_z+nd_all_z)
    fuls_z07.append(ful07)
    fuls_allz.append(fulall)
    
        



o_all_z2 = 87
nd_all_z2= 37
g_all_z2 = 30

o_z072 = 28
nd_z072= 5
g_z072 = 11


fuls_z072 = []
fuls_allz2 = []

for i in range(10000000):
    rand07 = np.random.normal(loc=g_z072, scale=np.sqrt(g_z072))
    randall = np.random.normal(loc=g_all_z2, scale=np.sqrt(g_all_z2))
    ful07 = (o_z072+nd_z072-rand07)/(o_z07+nd_z07)
    fulall = (o_all_z2+nd_all_z2 -randall)/(o_all_z+nd_all_z)
    fuls_z072.append(ful07)
    fuls_allz2.append(fulall)
    
        
plothist(fuls_z072, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_z07.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_z07.png', bbox_inches='tight', dpi=250)



  
        
plothist(fuls_allz2, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_allz.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_allz.png', bbox_inches='tight', dpi=250)



for i in range(5,50,5):
    norm_int = plothist(xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].lo3_offset, bins=i, range=(0,3), normed=True, norm0=True, integrate=True, label='Observed')
    gauss_int = plothist(gauss, bins=i, range=(0,3), normed=True, norm0=True, integrate=True, label='Gaussian')

    plt.legend()
    plt.xlabel('$\Delta$L$_{\mathrm{[OIII]}}$')
    plt.ylabel('Counts')
    plt.tight_layout()
    binwidth = str(round(3/i, 3))
    print(binwidth, norm_int, gauss_int, (norm_int-gauss_int)/gauss_int)    
    plt.savefig('plots/lo3_offset_hist_z07_binwid_'+binwidth+'.pdf', bbox_inches='tight', dpi=250, format='pdf')
    plt.savefig('plots/lo3_offset_hist_z07_binwid_'+binwidth+'.png', bbox_inches='tight', dpi=250)
    plt.close()

for i in range(5,50,5):
    norm_int = plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_offset, bins=i, range=(0,3), normed=True, norm0=True, integrate=True, label='Observed')
    gauss_int = plothist(gauss, bins=i, range=(0,3), normed=True, norm0=True, integrate=True, label='Gaussian')
    
    plt.legend()
    plt.xlabel('$\Delta$L$_{\mathrm{[OIII]}}$')
    plt.ylabel('Counts')
    plt.tight_layout()
    binwidth = str(round(3/i, 3))
    print(binwidth, norm_int, gauss_int, (norm_int-gauss_int)/gauss_int)    
    plt.savefig('plots/lo3_offset_hist_binwid_'+binwidth+'.pdf', bbox_inches='tight', dpi=250, format='pdf')
    plt.savefig('plots/lo3_offset_hist_binwid_'+binwidth+'.png', bbox_inches='tight', dpi=250)
    plt.close()
    
flux_sim_allz = flux_distros(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_pred_fromlx, xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].z, xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].corrected_presub_av)
flux_sim_z07 = flux_distros(xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].lo3_pred_fromlx, xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].z, xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].corrected_presub_av)
flux_real_allz = xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiiflux
flux_real_z07 = xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].oiiiflux

plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=40, range=(-17,-13), label='Observed', normed=True)
plothist(np.log10(flux_sim_allz), cumulative=True, reverse=True, bins=40, range=(-17,-13), label='Simulated', normed=True)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/fo3_cum_dist_allz.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/fo3_cum_dist_allz.png', bbox_inches='tight', dpi=250)
plt.close()

plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=40, range=(-17,-13), label='Observed', normed=True)
plothist(np.log10(flux_sim_z07), cumulative=True, reverse=True, bins=40, range=(-17,-13), label='Simulated', normed=True)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/fo3_cum_dist_z07.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/fo3_cum_dist_z07.png', bbox_inches='tight', dpi=250)
plt.close()

  , cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.225, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_z07_scaled_15_-0.225.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_z07_scaled_15_-0.225.png', bbox_inches='tight', dpi=250)



bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.225, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_hist_z07_scaled_15_-0.225.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_hist_z07_scaled_15_-0.225.png', bbox_inches='tight', dpi=250)

bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.29, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_allz_scaled_15_-0.29.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_allz_scaled_15_-0.29.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.29, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_hist_allz_scaled_15_-0.29.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_hist_allz_scaled_15_-0.29.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.33, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_z07_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_z07_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)

bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.33, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_allz_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_allz_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.33, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_hist_allz_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_hist_allz_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_z07.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_z07.png', bbox_inches='tight', dpi=250)


scatter(np.log10(xr_agn_props['x4_sn1_o3_hx_allz_nobptsf'].oiiiflux_sn), 
        np.log10(xr_agn_props['x4_sn1_o3_hx_allz_nobptsf'].oiiiflux), 
         label='X-ray AGNs (Point Sources)', s=5,edgecolor='k', facecolor='k', aspect='auto')

plt.xlabel('log([OIII] SNR)')

plt.ylabel('log(F$_{\mathrm{[OIII]}}$)')

plt.legend(loc=4,fontsize=15)
plt.tight_layout()
plt.savefig('plots/o3_snr_flux.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/o3_snr_flux.png', dpi=250, bbox_inches='tight')



fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext'].lo3_offset,
                   xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up,
                   ] ),
        np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext'].ssfr,xr_agn_props['x4_bad_o3_hx_allz_noext'].ssfr ]),
             aspect=5/4, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,linecolor='k',
             size_y_bin=0.6, counting_thresh=4,minx=-2.1, maxx=2.1,miny=-14, maxy=-9,
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].ssfr, marker='s', color='r', s=15)
plt.scatter(xr_agn_props['bptplus_sf_allxragn'].lo3_offset, xr_agn_props['bptplus_sf_allxragn'].ssfr, marker='^', color='g', s=25)

#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(sSFR)')
plt.xlim([-2,2])
plt.xticks([-2,-1,0,1,2])
plt.ylim([-14,-9])
plt.legend()
plt.tight_layout()
plt.savefig('plots/dlo3_ssfr_w_bptsf.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_ssfr_w_bptsf.png', dpi=250, bbox_inches='tight')


fig  = plt.figure(figsize=(4,5))
ax1 = fig.add_subplot(111)
scatter(np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext'].lo3_offset, 
                   xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset_up,
                   xr_agn_props['bptplus_sf_allxragn'].lo3_offset2] ),
        np.hstack([xr_agn_props['x4_sn1_o3_hx_allz_noext'].ssfr,
                   xr_agn_props['x4_bad_o3_hx_allz_noext'].ssfr, 
                   xr_agn_props['bptplus_sf_allxragn'].ssfr ]),
             aspect=5/4, bin_y=True, percentiles=False, bin_stat_y='median',plotdata=True,
             size_y_bin=0.6, counting_thresh=4,minx=-2.1, maxx=2.1,miny=-14, maxy=-9,linecolor='k',
             facecolor='magenta', edgecolor='magenta',marker='s', ax=ax1, fig = fig)
plt.scatter(xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset, xr_agn_props['x4_bad_o3_hx_allz_noext'].ssfr, marker='s', color='r', s=15)
plt.scatter(xr_agn_props['bptplus_sf_allxragn'].lo3_offset2, 
            xr_agn_props['bptplus_sf_allxragn'].ssfr, marker='^', color='g', s=25)

#plt.plot(x_panessa, geny_, 'gray',linestyle='--', linewidth=3, zorder=1)
plt.xlabel(r'$\Delta$ log(L$_{\rm [OIII]}$)')
plt.ylabel(r'log(sSFR)')
plt.xlim([-2,2])
plt.xticks([-2,-1,0,1,2])
plt.ylim([-14,-9])
plt.legend()
plt.tight_layout()
plt.savefig('plots/dlo3_ssfr_w_bptsf_sfsub.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/dlo3_ssfr_w_bptsf_sfsub.png', dpi=250, bbox_inches='tight')


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.33, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_z07_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_z07_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.33, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_hist_z07_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_hist_z07_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_allz.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_allz.png', bbox_inches='tight', dpi=250)



bnc, cnts_real_allz, int = plothist(np.log10(flux_real_allz), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim_allz, int = plothist(np.log10(flux_sim_allz)-0.3, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Counts')
plt.legend()
plt.tight_layout()

plt.close()

plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])

bnc, cnts_real_allz, int = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim_allz, int = plothist(np.log10(flux_sim_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)

bnc, cnts_real_z07, int = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim_z07, int = plothist(np.log10(flux_sim_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)


bnc148 = np.where(bnc*10== -148)[0]
bnc15 = np.where(bnc== -15)[0]
bnc17 = np.where(bnc == -17)[0]

scaled_z07_148 = cnts_real_z07[bnc148]/cnts_sim_z07[bnc148]
scaled_allz_148 = cnts_real_allz[bnc148]/cnts_sim_allz[bnc148]
scaled_z07_15 = cnts_real_z07[bnc15]/cnts_sim_z07[bnc15]
scaled_allz_15 = cnts_real_allz[bnc15]/cnts_sim_allz[bnc15]

plt.plot(bnc, cnts_real_z07, 'k--', label='Observed')
plt.plot(bnc, cnts_sim_z07*scaled_z07_148, 'b--', label='Simulated')
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'Cumulative counts')
plt.legend()
plt.tight_layout()
plt.xlim([-13,-18])
plt.savefig('plots/f_cum_z07_scaled_14.8.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cum_z07_scaled_14.8.png', bbox_inches='tight', dpi=250)

plt.plot(bnc, cnts_real_z07, 'k--', label='Observed')
plt.plot(bnc, cnts_sim_z07*scaled_z07_15, 'b--', label='Simulated')
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'Cumulative counts')
plt.legend()
plt.tight_layout()
plt.xlim([-13,-18])
plt.savefig('plots/f_cum_z07_scaled_15.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cum_z07_scaled_15.png', bbox_inches='tight', dpi=250)

plt.plot(bnc, cnts_real_allz, 'k--', label='Observed')
plt.plot(bnc, cnts_sim_allz*scaled_allz_15, 'b--', label='Simulated')
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'Cumulative counts')
plt.legend()
plt.tight_layout()
plt.xlim([-13,-18])
plt.savefig('plots/f_cum_allz_scaled_15.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cum_allz_scaled_15.png', bbox_inches='tight', dpi=250)


plt.plot(bnc, cnts_real_allz, 'k--', label='Observed')
plt.plot(bnc, cnts_sim_allz*scaled_allz_148, 'b--', label='Simulated')
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'Cumulative counts')
plt.legend()
plt.tight_layout()
plt.xlim([-13,-18])
plt.savefig('plots/f_cum_allz_scaled_14.8.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cum_allz_scaled_14.8.png', bbox_inches='tight', dpi=250)




bnc, cnts_real, int_ = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim, int_ = plothist(np.log10(flux_sim_allz)-0.33, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()
plt.plot(bnc[::-1], (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ulcum_allz_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ulcum_allz_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.33, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()

plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ul_allz_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_allz_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)




bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.33, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()
plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ulcum_z07_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ulcum_z07_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.33, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()

plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ul_z07_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_z07_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)





bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.225, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()
plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ulcum_z07_scaled_15_-0.225.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ulcum_z07_scaled_15_-0.225.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.225, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()

plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])
plt.savefig('plots/f_ul_z07_scaled_15_-0.225.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_z07_scaled_15_-0.225.png', bbox_inches='tight', dpi=250)

bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.29, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()
plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ulcum_allz_scaled_15_-0.29.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ulcum_allz_scaled_15_-0.29.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.29, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()

plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])
plt.savefig('plots/f_ul_allz_scaled_15_-0.29.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_allz_scaled_15_-0.29.png', bbox_inches='tight', dpi=250)


'''


def fwd_model(xr_df, xmid, avg_y, perc16, perc84,
              xmido3err, avg_y_o3err, perc16_o3err, perc84_o3err):
    '''

    '''

    np.random.seed(13)
    z = xr_df.z
    av = xr_df.corrected_presub_av
    lx = xr_df.full_lxagn
    # get oiii from a given lx based on median relation
    avg_interp = interp1d(xmid, avg_y, fill_value='extrapolate')
    perc16_interp = interp1d(xmid, perc16, fill_value='extrapolate')
    perc84_interp = interp1d(xmid, perc84, fill_value='extrapolate')
    avgerr_interp = interp1d(xmido3err, avg_y_o3err, fill_value='extrapolate')
    perc16err_interp = interp1d(
        xmido3err, perc16_o3err, fill_value='extrapolate')
    perc84err_interp = interp1d(
        xmido3err, perc84_o3err, fill_value='extrapolate')

    fo3_allsim = []
    fo3_err_allsim = []
    lo3_allsim = []
    lo3_allfiducial = []
    ks_vals = []

    p_vals = []
    realcnts, realbins = np.histogram(
        xr_df.oiiiflux, bins=100, range=(-1e-16, 1e-13))
    real_dist = np.cumsum(realcnts)

    range_scatters = np.linspace(1, 10, 100)
    for scat in range_scatters:
        fo3_sim = []
        lo3_fiducial = []
        fo3_err_sim = []
        lo3_sim = []
        for i in range(len(lx)):
            lxi = lx.iloc[i]
            zi = z.iloc[i]
            avi = av.iloc[i]
            perc16_i = perc16_interp(lxi)
            perc84_i = perc84_interp(lxi)
            avg_i = avg_interp(lxi)
            lo3_fiducial.append(avg_i)
            factor = (perc84_i-avg_i)/((avg_i-perc16_i)*scat)
            unif_factor = 0.5*factor
            random_unif = np.random.uniform()
            gaussian_sample = np.random.normal()

            if random_unif < unif_factor:
                lo3_pert = avg_i+(perc84_i-avg_i)*np.abs(gaussian_sample)

            else:
                lo3_pert = avg_i-(avg_i-perc16_i)*np.abs(gaussian_sample)*scat
            fo3_pert = getfluxfromlum(10**lo3_pert, zi)
            fo3_pert_red = redden(fo3_pert, avi, 5007.0)

            logfo3_err = avgerr_interp(
                np.log10(fo3_pert_red-np.min(xragn_no_sn_cuts.oiiiflux)*1.2))

            fo3_err = 10**(logfo3_err)
            fo3_pert_red = fo3_pert_red+np.random.normal(scale=fo3_err)

            fo3_sim.append(fo3_pert_red[0])
            lo3_sim.append(lo3_pert)
            fo3_err_sim.append(fo3_err[0])
        simcnts, simbins = np.histogram(
            fo3_sim, bins=100, range=(-1e-16, 1e-13))
        sim_dist = np.cumsum(simcnts)

        ks = ks_2samp(sim_dist, real_dist)
        ks_vals.append(ks.statistic)
        p_vals.append(ks.pvalue)
        lo3_allfiducial.append(lo3_fiducial)
        fo3_allsim.append(fo3_sim)
        fo3_err_allsim.append(fo3_err_sim)
        lo3_allsim.append(lo3_sim)

    return range_scatters, fo3_allsim, lo3_allsim, fo3_err_allsim, lo3_allfiducial, ks_vals, p_vals
    # error_interp

    # get the empirical error,
    # use that error as a sigma for a gaussian to draw the error
    # add the drawn gaussian to the the flux

    # make cumulative distribution of simulated fluxes, not log
    # same for actual fluxes
    # ks teest for the two distributions
    # change the lower scatter, repeat experiement, check the ks test. repeat until minimum ks difference
    # plot offsets delta log lo3 for fiducial high snr model and for best model
    # make them have the same peak, compare areas, these are the underluminous agn
    # repeat everything with SNR>2 relation?
    # perturb lo3,


def plot_relations():
    for prop in xr_agn_props.keys():
        agn_prop = xr_agn_props[prop]
        a, b, aerr, berr, covab = BCES.bcesp(np.array(agn_prop.full_xraylum-42),
                                             np.array(np.nanmean(np.vstack([agn_prop.e_full_xraylum_down,
                                                                            agn_prop.e_full_xraylum_up]), axis=0)),
                                             np.array(agn_prop.oiiilum),
                                             np.array(np.nanmean(np.vstack([agn_prop.e_oiiilum_down,
                                                                           agn_prop.e_oiiilum_up]), axis=0)),
                                             np.zeros_like(agn_prop.full_xraylum))
        print(a, b)
        print(prop)
        scatter(agn_prop.hard_xraylum, agn_prop.oiiilum,
                # xerr=np.vstack([agn_prop.e_full_xraylum_down,
                #                agn_prop.e_full_xraylum_up]),
                # yerr=np.vstack([agn_prop.e_oiiilum_down,
                #                agn_prop.e_oiiilum_up]),
                percentiles=True,
                minx=37, maxx=46, miny=37, maxy=44, aspect='equal', alpha=1, ecolor='k', size_y_bin=0.5, bin_y=True)

        # plt.plot(np.arange(37,46,0.1), (np.arange(37,46,0.1)-42)*(a[3])+b[3],'k:',
        #         label='ODR: '+str(a[3])[:4]+r'$\cdot (x-42)$+'+str(b[3])[:4])
        plt.xticks([38, 40, 42, 44, 46])
        plt.yticks([38, 40, 42, 44])

        plt.xlabel(r'log(L$_{\mathrm{X}}$)')
        plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
        plt.legend()
        plt.savefig('plots/hx_o3_'+prop+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/hx_o3_'+prop+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()

    for prop in pure_xr_agn_props.keys():
        agn_prop = pure_xr_agn_props[prop]
        a, b, aerr, berr, covab = BCES.bcesp(np.array(agn_prop.full_lxagn-42),
                                             np.array(np.nanmean(np.vstack([agn_prop.e_full_xraylum_down,
                                                                           agn_prop.e_full_xraylum_up]), axis=0)),
                                             np.array(
                                                 agn_prop.oiiilum_sub_dered),
                                             np.array(np.nanmean(np.vstack([agn_prop.e_oiiilum_down_sub_dered,
                                                                            agn_prop.e_oiiilum_up_sub_dered]), axis=0)),
                                             np.zeros_like(agn_prop.full_lxagn))
        print(a, b)
        scatter(agn_prop.full_lxagn, agn_prop.oiiilum_sub_dered,
                xerr=np.vstack([agn_prop.e_full_xraylum_down,
                                agn_prop.e_full_xraylum_up]),
                yerr=np.vstack([agn_prop.e_oiiilum_down_sub_dered,
                                agn_prop.e_oiiilum_up_sub_dered]),
                minx=37, maxx=46, miny=37, maxy=46, aspect='equal', alpha=1, ecolor='k', bin_y=True)

        plt.plot(np.arange(37, 46, 0.1), (np.arange(37, 46, 0.1)-42)*(a[3])+b[3], 'k:',
                 label='ODR: '+str(a[3])[:4]+r'$\cdot (x-42)$+'+str(b[3])[:4])
        plt.xticks([38, 40, 42, 44, 46])
        plt.yticks([38, 40, 42, 44, 46])

        plt.xlabel(r'log(L$_{\mathrm{X}}$)')
        plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
        plt.legend()
        plt.savefig('plots/pure_lx_o3_'+prop+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/pure_lx_o3_'+prop+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()


'''
prop='unclass_p1'
agn_prop= xr_agn_props[prop]

valid_samp = np.where((np.isfinite(agn_prop.full_xraylum))& 
                      (np.isfinite(agn_prop.oiiilum)) &
                      (np.isfinite(agn_prop.e_full_xraylum_down)) &
                      (np.isfinite(agn_prop.e_full_xraylum_up)) &
                      (np.isfinite(agn_prop.e_oiiilum_down)) & 
                      (np.isfinite(agn_prop.e_oiiilum_up)))[0]
                                                                           
a,b,aerr,berr,covab=BCES.bcesp(np.array(agn_prop.full_xraylum-42)[valid_samp],
                           np.array(np.nanmean(np.vstack([agn_prop.e_full_xraylum_down,
                                                          agn_prop.e_full_xraylum_up]),axis=0))[valid_samp],
                           np.array(agn_prop.oiiilum)[valid_samp],
                           np.array(np.nanmean(np.vstack([agn_prop.e_oiiilum_down, 
                                                          agn_prop.e_oiiilum_up]), axis=0))[valid_samp], 
                           np.zeros_like(agn_prop.full_xraylum)[valid_samp])
print(a,b)
print(prop)
scatter(agn_prop.full_xraylum[valid_samp],agn_prop.oiiilum[valid_samp], 
        xerr=np.vstack([agn_prop.e_full_xraylum_down[valid_samp], 
                        agn_prop.e_full_xraylum_up[valid_samp]]),
        yerr=np.vstack([agn_prop.e_oiiilum_down[valid_samp],
         
                        agn_prop.e_oiiilum_up[valid_samp]]),
                        
        minx=37, maxx=46, miny=37, maxy=46, aspect='equal',alpha=1, ecolor='k', bin_y=True, size_y_bin=0.5)    
plt.xticks([38,40,42,44,46])
plt.yticks([38,40,42,44,46])
plt.plot(np.arange(37,46,0.1), (np.arange(37,46,0.1)-42)*(a[3])+b[3],'k:', 
         label='ODR: '+str(a[3])[:4]+r'$\cdot (x-42)$+'+str(b[3])[:4])
plt.xlabel(r'log(L$_{\mathrm{X}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.savefig('plots/lx_o3_unclass_p1.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lx_o3_unclass_p1.png', bbox_inches='tight', dpi=250, format='png')
prop='unclass_p2'
agn_prop= xr_agn_props[prop]
a,b,aerr,berr,covab=BCES.bcesp(np.array(agn_prop.full_xraylum-42)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(np.nanmean(np.vstack([agn_prop.e_full_xraylum_down,
                                                          agn_prop.e_full_xraylum_up]),axis=0))[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(agn_prop.oiiilum)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(np.nanmean(np.vstack([agn_prop.e_oiiilum_down, 
                                                          agn_prop.e_oiiilum_up]), axis=0))[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]], 
                           np.zeros_like(agn_prop.full_xraylum)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]])
print(a,b)
print(prop)
scatter(agn_prop.full_xraylum[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],agn_prop.oiiilum[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]], 
        xerr=np.vstack([agn_prop.e_full_xraylum_down[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]], 
                        agn_prop.e_full_xraylum_up[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]]]),
        yerr=np.vstack([agn_prop.e_oiiilum_down[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]],
         
                        agn_prop.e_oiiilum_up[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]]]),
                        
        minx=37, maxx=46, miny=37, maxy=46, aspect='equal',alpha=1, ecolor='k', bin_y=True, size_y_bin=0.5)    
plt.xticks([38,40,42,44,46])
plt.yticks([38,40,42,44,46])
plt.plot(np.arange(37,46,0.1), (np.arange(37,46,0.1)-42)*(a[3])+b[3],'k:', 
         label='ODR: '+str(a[3])[:4]+r'$\cdot (x-42)$+'+str(b[3])[:4])
plt.xlabel(r'log(L$_{\mathrm{X}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.savefig('plots/lx_o3_unclass_p2.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lx_o3_unclass_p2.png', bbox_inches='tight', dpi=250, format='png')
prop='xrall'
agn_prop= xr_agn_props[prop]
a,b,aerr,berr,covab=BCES.bcesp(np.array(agn_prop.full_xraylum-42)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(np.nanmean(np.vstack([agn_prop.e_full_xraylum_down,
                                                          agn_prop.e_full_xraylum_up]),axis=0))[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(agn_prop.oiiilum)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(np.nanmean(np.vstack([agn_prop.e_oiiilum_down, 
                                                          agn_prop.e_oiiilum_up]), axis=0))[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]], 
                           np.zeros_like(agn_prop.full_xraylum)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]])
print(a,b)
print(prop)
scatter(agn_prop.full_xraylum[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],agn_prop.oiiilum[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]], 
        xerr=np.vstack([agn_prop.e_full_xraylum_down[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]], 
                        agn_prop.e_full_xraylum_up[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]]]),
        yerr=np.vstack([agn_prop.e_oiiilum_down[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]],
         
                        agn_prop.e_oiiilum_up[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]]]),
                        
        minx=37, maxx=46, miny=37, maxy=46, aspect='equal',alpha=1, ecolor='k', bin_y=True, size_y_bin=0.5)    
plt.xticks([38,40,42,44,46])
plt.yticks([38,40,42,44,46])
plt.plot(np.arange(37,46,0.1), (np.arange(37,46,0.1)-42)*(a[3])+b[3],'k:', 
         label='ODR: '+str(a[3])[:4]+r'$\cdot (x-42)$+'+str(b[3])[:4])
plt.xlabel(r'log(L$_{\mathrm{X}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.savefig('plots/lx_o3_xrall.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lx_o3_xrall.png', bbox_inches='tight', dpi=250, format='png')
'''


def lin_func(p, x):
    m, b = p
    return m*x+b-42


genx_ = np.arange(37, 46)
geny_ = np.arange(37, 46)

x_panessa = geny_*1.22-7.55

panessa_xraylum = np.array([40.79, 37.55, 42.84, 42.07, 42.83, 42.31, 42.58, 40.82, 41.85, 38.21,
                            39.95, 40.25, 42.62, 41.89, 40.79, 41.74, 38.86, 39.88, 42.29, 39.10,
                            37.88, 38.88, 41.18, 41.31, 41.29, 42.47, 39.87, 42.22, 40.87, 41.72,
                            39.81, 39.32, 39.65, 39.59, 39.43, 41.03, 40.22, 39.16, 38.89, 41.08,
                            40.91, 41.36, 43.25,  41.12, 41.47])
panessa_t1_xraylum = np.array([42.83, 40.25, 41.74, 42.29,
                               41.31, 42.47, 42.22, 39.81, 41.03,
                               40.22, 41.08, 41.36, 43.25])
panessa_t1_oiiilum = np.array([41.91, 38.56, 40.51, 40.50,
                               39.81, 41.47, 40.41, 38.28, 39.42,
                               38.71, 39.72, 39.03, 41.16])
panessa_oiiilum = np.array([39.04, 37.90, 41.91, 40.76, 41.91, 40.42, 40.94, 40.40, 39.92, 38.58,
                            38.96, 38.56, 40.07, 40.07, 39.90, 40.51, 37.99, 38.86, 40.50, 38.22,
                            38.79, 38.63, 40.50, 39.81, 38.74, 41.47, 38.46, 40.41, 39.07, 40.54,
                            38.28, 37.81, 39.04, 39.23, 38.69, 39.42, 38.71, 38.86, 38.58, 39.72,
                            39.90, 39.03, 41.16, 40.25, 40.21])

'''
bin_stat_y='median'
    
x_heckman = geny_+1.59
x_heckmant2 = geny_+0.57

y_berney_all = genx_*1.23-12
y_berney_t1 = genx_*1.1-6.5
y_berney_t2 = genx_*1.3-15
y_berney_sy = genx_*1.13-7.5
y_berney_liner = genx_*1.8-37
y_berney_comp = genx_-4



xliner_full, yavliner_full, _,liner_16_full, liner_84_full = scatter(xr_agn_props['combo_all_liners_fx'].full_xraylum, 
                                                                     xr_agn_props['combo_all_liners_fx'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)


xliner_hard, yavliner_hard, _,liner_16_hard,liner_84_hard = scatter(xr_agn_props['combo_all_liners_hx_noext'].hard_xraylum,
                                                                    xr_agn_props['combo_all_liners_hx_noext'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)

xsliner_hard, yavsliner_hard, _,liner_16_hard,liner_84_hard = scatter(xr_agn_props['combo_sliner_hx_noext'].hard_xraylum,
                                                                    xr_agn_props['combo_sliner_hx_noext'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=3)

xhliner_hard, yavhliner_hard, _,liner_16_hard,liner_84_hard = scatter(xr_agn_props['combo_hliner_hx_noext'].hard_xraylum,
                                                                    xr_agn_props['combo_hliner_hx_noext'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=3)

xliner_full07, yavliner_full07, _,liner_16_full07, liner_84_full07 = scatter(xr_agn_props['combo_all_liners_fx_z07'].full_xraylum, 
                                                                             xr_agn_props['combo_all_liners_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)


xliner_hard07, yavliner_hard07, _,liner_16_hard07,liner_84_hard07 = scatter(xr_agn_props['combo_all_liners_hx_z07'].hard_xraylum,
                                                                            xr_agn_props['combo_all_liners_hx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xliner_hard07edd, yavliner_hard07edd, _,liner_16_hard07edd,liner_84_hard07edd = scatter(xr_agn_props['combo_all_liners_hx_z07'].hard_xraylum-xr_agn_props['combo_all_liners_hx_z07'].mbh,
                                                                            xr_agn_props['combo_all_liners_hx_z07'].oiiilum-xr_agn_props['combo_all_liners_hx_z07'].mbh,
                                                                             aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,
             maxx=38, minx=31, miny=30, maxy=37)

xliner_hardedd, yavliner_hardedd, _,liner_16_hardedd,liner_84_hardedd = scatter(xr_agn_props['combo_all_liners_hx'].hard_xraylum-xr_agn_props['combo_all_liners_hx'].mbh,
                                                                    xr_agn_props['combo_all_liners_hx'].oiiilum-xr_agn_props['combo_all_liners_hx'].mbh,
                    maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)



xliu_hard_dc, yavliu_hard_dc, _, liu16_hard_dc, liu84_hard_dc = scatter(liuobj_xmm_hx_o3_dust_df['hardlums_rf'], liuobj_xmm_hx_o3_dust_df['lo3_corr'] ,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)
xliu_full_dc, yavliu_full_dc, _, liu16_full_dc, liu84_full_dc = scatter(liuobj_xmm_fx_o3_dust_df['fulllums_rf'], liuobj_xmm_fx_o3_dust_df['lo3_corr'] ,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xliu_hard_dc07, yavliu_hard_dc07, _, liu16_hard_dc07, liu84_hard_dc07 = scatter(liuobj_xmm_hx_o3_dust_z07_df['hardlums_rf'], liuobj_xmm_hx_o3_dust_z07_df['lo3_corr'] ,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)
xliu_full_dc07, yavliu_full_dc07, _, liu16_full_dc07, liu84_full_dc07 = scatter(liuobj_xmm_fx_o3_dust_z07_df['fulllums_rf'], liuobj_xmm_fx_o3_dust_z07_df['lo3_corr'] ,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xliu_hard_dc07edd, yavliu_hard_dc07edd, _, liu16_hard_dc07edd, liu84_hard_dc07edd = scatter(liuobj_xmm_hx_o3_dust_z07_df['hardlums_rf']-liuobj_xmm_hx_o3_dust_z07_df['logMBH'], 
                                                                                            liuobj_xmm_hx_o3_dust_z07_df['lo3_corr']-liuobj_xmm_hx_o3_dust_z07_df['logMBH'] ,
            maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xliu_hard_dcedd, yavliu_hard_dcedd, _, liu16_hard_dcedd, liu84_hard_dcedd = scatter(liuobj_xmm_hx_o3_dust_df['hardlums_rf']-liuobj_xmm_hx_o3_dust_df['logMBH'],
                                                                                    liuobj_xmm_hx_o3_dust_df['lo3_corr']-liuobj_xmm_hx_o3_dust_df['logMBH'] ,
             maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)


xsy2_full07, yavsy2_full07, _, sy2_16_full07, sy2_84_full07 = scatter(xr_agn_props['combo_sy2_fx_z07'].full_xraylum, xr_agn_props['combo_sy2_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xsy2_hard07, yavsy2_hard07, _, sy2_16_hard07, sy2_84_hard07 = scatter(xr_agn_props['combo_sy2_hx_z07'].hard_xraylum, xr_agn_props['combo_sy2_hx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xsy2_full, yavsy2_full, _, sy2_16_full, sy2_84_full = scatter(xr_agn_props['combo_sy2_fx'].full_xraylum, xr_agn_props['combo_sy2_fx'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xsy2_hard, yavsy2_hard, _, sy2_16_hard, sy2_84_hard = scatter(xr_agn_props['combo_sy2_hx_noext'].hard_xraylum, xr_agn_props['combo_sy2_hx_noext'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)
xsy2_hard07, yavsy2_hard07, _, sy2_16_hard07, sy2_84_hard07 = scatter(xr_agn_props['combo_sy2_hx_z07'].hard_xraylum, xr_agn_props['combo_sy2_hx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)

xsy2_hardedd, yavsy2_hardedd, _, sy2_16_hardedd, sy2_84_hardedd = scatter(xr_agn_props['combo_sy2_hx'].hard_xraylum-xr_agn_props['combo_sy2_hx'].mbh, 
                                                                          xr_agn_props['combo_sy2_hx'].oiiilum-xr_agn_props['combo_sy2_hx'].mbh,
            maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xsy2_hard07edd, yavsy2_hard07edd, _, sy2_16_hard07edd, sy2_84_hard07edd = scatter(xr_agn_props['combo_sy2_hx_z07'].hard_xraylum-xr_agn_props['combo_sy2_hx_z07'].mbh, 
                                                                                  xr_agn_props['combo_sy2_hx_z07'].oiiilum-xr_agn_props['combo_sy2_hx_z07'].mbh,
             maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)

xsy2_fulledd, yavsy2_fulledd, _, sy2_16_fulledd, sy2_84_fulledd = scatter(xr_agn_props['combo_sy2_fx'].full_xraylum-xr_agn_props['combo_sy2_fx'].mbh, 
                                                                          xr_agn_props['combo_sy2_fx'].oiiilum-xr_agn_props['combo_sy2_fx'].mbh,
            maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xsy2_full07edd, yavsy2_full07edd, _, sy2_16_full07edd, sy2_84_full07edd = scatter(xr_agn_props['combo_sy2_fx_z07'].full_xraylum-xr_agn_props['combo_sy2_fx_z07'].mbh, 
                                                                                  xr_agn_props['combo_sy2_fx_z07'].oiiilum-xr_agn_props['combo_sy2_fx_z07'].mbh,
             maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)



xt2_full07, yavt2_full07, _, t2_16_full07, t2_84_full07 = scatter(xr_agn_props['x4_sn3_o3_fx_z07'].full_xraylum, xr_agn_props['x4_sn3_o3_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)
xt2_hard07, yavt2_hard07, _, t2_16_hard07, t2_84_hard07 = scatter(xr_agn_props['x4_sn3_o3_hx_z07'].hard_xraylum, xr_agn_props['x4_sn3_o3_hx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)

xt2_full, yavt2_full, _, t2_16_full, t2_84_full = scatter(xr_agn_props['x4_sn3_o3_fx_allz'].full_xraylum, xr_agn_props['x4_sn3_o3_fx_allz'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)

xt2_hard, yavt2_hard, _, t2_16_hard, t2_84_hard = scatter(xr_agn_props['x4_sn3_o3_hx_allz_noext_nobptsf'].hard_xraylum, xr_agn_props['x4_sn3_o3_hx_allz_noext_nobptsf'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)

xt2_hard07edd, yavt2_hard07edd, _, t2_16_hard07edd, t2_84_hard07edd = scatter(xr_agn_props['x4_sn3_o3_hx_z07'].hard_xraylum-xr_agn_props['x4_sn3_o3_hx_z07'].mbh, 
                                                                              xr_agn_props['x4_sn3_o3_hx_z07'].oiiilum-xr_agn_props['x4_sn3_o3_hx_z07'].mbh,
               maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)

xt2_hardedd, yavt2_hardedd, _, t2_16_hardedd, t2_84_hardedd = scatter(xr_agn_props['x4_sn3_o3_hx_allz'].hard_xraylum-xr_agn_props['x4_sn3_o3_hx_allz'].mbh, 
                                                                      xr_agn_props['x4_sn3_o3_hx_allz'].oiiilum-xr_agn_props['x4_sn3_o3_hx_allz'].mbh,
               maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)




plt.plot(xliu_full07[yavliu_full07>0], yavliu_full07[yavliu_full07>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_full07, yavt2_full07, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_full07, yavsy2_full07, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_full07, yavliner_full07, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_fx_z07'].full_xraylum, xr_agn_props['x4_sn3_o3_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_fx_z07'].full_xraylum, xr_agn_props['combo_all_liners_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_fx_z07'].full_xraylum, xr_agn_props['combo_sy2_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([38,40,42,44])
#plt.yticks([38,40,42,44])

plt.xlabel(r'log(L$_{\mathrm{X,\ 0.5-10\ keV}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.tight_layout()




plt.plot(xliu_full_dc07[yavliu_full_dc07>0], yavliu_full_dc07[yavliu_full_dc07>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_full07, yavt2_full07, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_full07, yavsy2_full07, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_full07, yavliner_full07, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_fx_z07'].full_xraylum, xr_agn_props['x4_sn3_o3_fx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_fx_z07'].full_xraylum, xr_agn_props['combo_all_liners_fx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_fx_z07'].full_xraylum, xr_agn_props['combo_sy2_fx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([38,40,42,44])
plt.yticks([38,40,42,44])


plt.xlabel(r'log(L$_{\mathrm{X,\ 0.5-10\ keV}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/fx_o3_sn3_combined_objects_z07_t1dc_med.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/fx_o3_sn3_combined_objects_z07_t1dc_med.png', bbox_inches='tight', dpi=250, format='png')
plt.close()


plt.plot(xliu_hard_dc07[yavliu_hard_dc07>0], yavliu_hard_dc07[yavliu_hard_dc07>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_hard07, yavt2_hard07, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_hard07, yavsy2_hard07, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_hard07, yavliner_hard07, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_hx_z07'].hard_xraylum, xr_agn_props['x4_sn3_o3_hx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_hx_z07'].hard_xraylum, xr_agn_props['combo_all_liners_hx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_hx_z07'].hard_xraylum, xr_agn_props['combo_sy2_hx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([38,40,42,44])
plt.yticks([38,40,42,44])
plt.plot(x_panessa, geny_, 'k--', linewidth=3, label='Panessa+07 without QSOs')

plt.xlabel(r'log(L$_{\mathrm{X,\ 2-10\ keV}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/hx_o3_sn3_combined_objects_z07_t1dc_med_p07_noqso.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hx_o3_sn3_combined_objects_z07_t1dc_med_p07_noqso.png', bbox_inches='tight', dpi=250, format='png')
plt.close()



plt.plot(xliu_hard_dc07edd[yavliu_hard_dc07edd>0], yavliu_hard_dc07edd[yavliu_hard_dc07edd>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_hard07edd, yavt2_hard07edd, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_hard07edd, yavsy2_hard07edd, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_hard07edd, yavliner_hard07edd, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_hx_z07'].hard_xraylum-xr_agn_props['x4_sn3_o3_hx_z07'].mbh, 
        xr_agn_props['x4_sn3_o3_hx_z07'].oiiilum-xr_agn_props['x4_sn3_o3_hx_z07'].mbh,
              maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_hx_z07'].hard_xraylum- xr_agn_props['combo_all_liners_hx_z07'].mbh,
        xr_agn_props['combo_all_liners_hx_z07'].oiiilum- xr_agn_props['combo_all_liners_hx_z07'].mbh,
              maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_hx_z07'].hard_xraylum-xr_agn_props['combo_sy2_hx_z07'].mbh, 
        xr_agn_props['combo_sy2_hx_z07'].oiiilum-xr_agn_props['combo_sy2_hx_z07'].mbh,
             maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([32,34,36,38])
plt.yticks([30,32,34,36])
#plt.plot(x_panessa, geny_, 'k--', linewidth=3, label='Panessa+07 without QSOs')

plt.xlabel(r'log(L$_{\mathrm{X,\ 2-10\ keV}}$/M$_{\mathrm{BH}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$/M$_{\mathrm{BH}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/hx_o3_mbh_sn3_combined_objects_z07_t1dc.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hx_o3_mbh_sn3_combined_objects_z07_t1dc.png', bbox_inches='tight', dpi=250, format='png')
plt.close()


plt.plot(xliu_hard_dcedd[yavliu_hard_dcedd>0], yavliu_hard_dcedd[yavliu_hard_dcedd>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_hardedd, yavt2_hardedd, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_hardedd, yavsy2_hardedd, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_hardedd, yavliner_hardedd, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_hx_allz'].hard_xraylum-xr_agn_props['x4_sn3_o3_hx_allz'].mbh, 
        xr_agn_props['x4_sn3_o3_hx_allz'].oiiilum-xr_agn_props['x4_sn3_o3_hx_allz'].mbh,
              maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_hx'].hard_xraylum- xr_agn_props['combo_all_liners_hx'].mbh,
        xr_agn_props['combo_all_liners_hx'].oiiilum- xr_agn_props['combo_all_liners_hx'].mbh,
              maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_hx'].hard_xraylum-xr_agn_props['combo_sy2_hx'].mbh, 
        xr_agn_props['combo_sy2_hx'].oiiilum-xr_agn_props['combo_sy2_hx'].mbh,
             maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([32,34,36,38])
plt.yticks([30,32,34,36])
#plt.plot(x_panessa, geny_, 'k--', linewidth=3, label='Panessa+07 without QSOs')

plt.xlabel(r'log(L$_{\mathrm{X,\ 2-10\ keV}}$/M$_{\mathrm{BH}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$/M$_{\mathrm{BH}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/hx_o3_mbh_sn3_combined_objects_t1dc.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hx_o3_mbh_sn3_combined_objects_t1dc.png', bbox_inches='tight', dpi=250, format='png')
plt.close()


plt.plot(xliu_full_dc[yavliu_full_dc>0], yavliu_full_dc[yavliu_full_dc>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_full, yavt2_full, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_full, yavsy2_full, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_full, yavliner_full, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_fx_allz'].full_xraylum, xr_agn_props['x4_sn3_o3_fx_allz'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_fx'].full_xraylum, xr_agn_props['combo_all_liners_fx'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_fx'].full_xraylum, xr_agn_props['combo_sy2_fx'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([38,40,42,44])
plt.yticks([38,40,42,44])

plt.xlabel(r'log(L$_{\mathrm{X,\ 0.5-10\ keV}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/fx_o3_sn3_combined_objects_allz_t1dc_med.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/fx_o3_sn3_combined_objects_allz_t1dc_med.png', bbox_inches='tight', dpi=250, format='png')
plt.close()



plt.plot(xliu_hard_dc[yavliu_hard_dc>0], yavliu_hard_dc[yavliu_hard_dc>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_hard, yavt2_hard, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_hard, yavsy2_hard, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_hard, yavliner_hard, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_hx_allz_noext_nobptsf'].hard_xraylum, xr_agn_props['x4_sn3_o3_hx_allz_noext_nobptsf'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_hx_noext'].hard_xraylum, xr_agn_props['combo_all_liners_hx_noext'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_hx_noext'].hard_xraylum, xr_agn_props['combo_sy2_hx_noext'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([38,40,42,44])
plt.yticks([38,40,42,44])
plt.plot(x_panessa, geny_, 'k--', linewidth=3, label='Panessa+06')
plt.xlabel(r'log(L$_{\mathrm{X,\ 2-10\ keV}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/hx_o3_sn3_combined_objects_allz_t1dc_med_p07_noqso_sfsub.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hx_o3_sn3_combined_objects_allz_t1dc_med_p07_noqso_sfsub.png', bbox_inches='tight', dpi=250, format='png')
plt.close()




label='full'
scat=0.6

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(loglum_arr,lsfrrelat[label][2],'k--',zorder=3)
plt.fill_between(loglum_arr+scat,lsfrrelat[label][2],y2=lsfrrelat[label][2]-20,color='gray', zorder=0,alpha=0.5,linewidth=0)


plt.xlabel(r'log(L$_{\rm X, 0.5-10\ keV}$)',fontsize=20)
plt.ylabel(r'log(SFR)',fontsize=20)
plt.text(43.1,-3.05,'X-ray AGN\n Candidates', fontsize=14, rotation=0)
plt.xlim([37.,45.5])
plt.ylim([-3.5, 5])
ax.set(adjustable='box', aspect='equal')
plt.tight_layout()
#plt.savefig('plots/lx_full_selection_.pdf', bbox_inches='tight', dpi=250, format='pdf')
#plt.savefig('plots/lx_full_selection_.png', bbox_inches='tight', dpi=250)

scatter( xr_agn_props['x4_sn1_o3_hx_allz_noext'].full_xraylum, xr_agn_props['x4_sn1_o3_hx_allz_noext'].sfr,facecolor='magenta',
        edgecolor='magenta', marker='s', s=10, label='Point Source')

scatter( xr_agn_props['x4_sn1_o3_hx_allz_ext'].full_xraylum,xr_agn_props['x4_sn1_o3_hx_allz_ext'].sfr, facecolor='red',edgecolor='black', marker='^', s=15,linewidth=0.5, label='Extended')
plt.legend(frameon=False,fontsize=15,loc=2,bbox_to_anchor = (-0.02, 0.99))

plt.savefig('plots/lx_sfr_ext_noext_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lx_sfr_ext_noext_.png', bbox_inches='tight', dpi=250)



label='full'
scat=0.6

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(loglum_arr,lsfrrelat[label][2],'k--',label='Relation for normal galaxies (non-AGN)',zorder=3)
plt.fill_between(loglum_arr+scat,lsfrrelat[label][2],y2=lsfrrelat[label][2]-20,color='gray', zorder=0,alpha=0.5,linewidth=0)

plt.scatter(xr_agn_props['all_hx'].full_xraylum.iloc[np.where(xr_agn_props['all_hx'].ext==0)[0]],
            xr_agn_props['all_hx'].sfr.iloc[np.where(xr_agn_props['all_hx'].ext==0)[0]], facecolor='none', edgecolor='k',s=15, label='Unclassified Type 2 AGNs')
plt.scatter(xr_agn_props['x4_n2agn'].full_xraylum.iloc[np.where(xr_agn_props['x4_n2agn'].ext==0)[0]],
            xr_agn_props['x4_n2agn'].sfr.iloc[np.where(xr_agn_props['x4_n2agn'].ext==0)[0]],marker='o',facecolor='none', edgecolor='k', s=15)


plt.scatter(xr_agn_props['combo_sy2_hx'].full_xraylum.iloc[np.where(xr_agn_props['combo_sy2_hx'].ext==0)[0]],
            xr_agn_props['combo_sy2_hx'].sfr.iloc[np.where(xr_agn_props['combo_sy2_hx'].ext==0)[0]], facecolor='b', edgecolor='b',s=15, label='Seyfert 2s')
plt.scatter(xr_agn_props['combo_all_liners_hx'].full_xraylum.iloc[np.where(xr_agn_props['combo_all_liners_hx'].ext==0)[0]],
            xr_agn_props['combo_all_liners_hx'].sfr.iloc[np.where(xr_agn_props['combo_all_liners_hx'].ext==0)[0]], facecolor='orange', edgecolor='orange', s=15, label='LINERs')
plt.scatter(xr_agn_props['bptplus_sf_allxr'].full_xraylum.iloc[np.where(xr_agn_props['bptplus_sf_allxr'].ext==0)[0]],
            xr_agn_props['bptplus_sf_allxr'].sfr.iloc[np.where(xr_agn_props['bptplus_sf_allxr'].ext==0)[0]],marker='^',facecolor='g', edgecolor='g', s=15,  label='BPT SF')

plt.scatter(xr_agn_props['unclass_p2_hx_o3'].full_xraylum.iloc[np.where(xr_agn_props['unclass_p2_hx_o3'].ext==0)[0]],
            xr_agn_props['unclass_p2_hx_o3'].sfr.iloc[np.where(xr_agn_props['unclass_p2_hx_o3'].ext==0)[0]],marker='+', facecolor='r', edgecolor='r', s=15, label='Weak or no emission lines')


plt.xlabel(r'log(L$_{\rm x}$)',fontsize=20)
plt.ylabel(r'log(SFR)',fontsize=20)
plt.legend(frameon=False,fontsize=10,loc=2,bbox_to_anchor = (-0.00, 0.95))
plt.text(42.8,-2.4,'X-ray AGN\n Candidates', fontsize=15)
plt.xlim([37.5,45])
plt.ylim([-3,4.5])
ax.set(adjustable='box', aspect='equal')
plt.tight_layout()




union = pd.concat([xr_agn_props['x4all_hx'],xr_agn_props['unclass_p2_hx_o3'],
                   xr_agn_props['combo_val_hx_allxr'], xr_agn_props['x4_n2agn'],
                   EL_4xmm_all.bptplus_sf_df.iloc[np.where(EL_4xmm_all.bptplus_sf_df.hardflux_sn>2)],
                   EL_4xmm_all.bptplusnii_sf_df.iloc[np.where(EL_4xmm_all.bptplusnii_sf_df.hardflux_sn>2)]])

union_t2 = pd.concat([xr_agn_props['x4_sn1_o3_hx_allz_noext_nobptsf'],xr_agn_props['unclass_p2_hx_o3'],
                   xr_agn_props['combo_val_hx_noext'], xr_agn_props['x4_n2agn'],xr_agn_props['combo_sy2_hx_noext'],xr_agn_props['combo_all_liners_hx_noext']]
                     )

                   
sym_diff = union[~union.duplicated(subset='ids', keep=False)]
sym_diff_t2 = union_t2[~union_t2.duplicated(subset='ids', keep=False)]

filt = ((sym_diff_t2.oiiiflux_sn>1)&(sym_diff_t2.niiflux_sn>2)&(sym_diff_t2.hbetaflux_sn>1)&(sym_diff_t2.halpflux_sn>2))
sym_diff_t2_filt = sym_diff_t2[filt].copy()

label='full'
scat=0.6

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(loglum_arr,lsfrrelat[label][2],'k--',label='Relation for normal galaxies (non-AGN)',zorder=3)
plt.fill_between(loglum_arr+scat,lsfrrelat[label][2],y2=lsfrrelat[label][2]-20,color='gray', zorder=0,alpha=0.5,linewidth=0)

plt.scatter(sym_diff.full_xraylum,
            sym_diff.sfr, facecolor='none', 
            edgecolor='k',s=13, label='Unclassified type 2 AGNs', zorder=1)
plt.scatter(xr_agn_props['x4_n2agn'].full_xraylum,
            xr_agn_props['x4_n2agn'].sfr,marker='o',facecolor='none', 
            edgecolor='k', s=13, zorder=2)

plt.scatter(xr_agn_props['unclass_p2_hx_o3'].full_xraylum,
            xr_agn_props['unclass_p2_hx_o3'].sfr,marker='+', facecolor='r', 
            edgecolor='r', s=15, label='Weak or no emission lines', zorder=3)



plt.scatter(xr_agn_props['bptplus_sf_allxr'].full_xraylum,
            xr_agn_props['bptplus_sf_allxr'].sfr,marker='^',facecolor='g',
            edgecolor='g', s=15,  label='BPT star-forming', zorder=15)

plt.scatter(xr_agn_props['combo_sy2_hx_allxr'].full_xraylum,
            xr_agn_props['combo_sy2_hx_allxr'].sfr, facecolor='b', 
            edgecolor='b',s=15, label='Seyfert 2s', zorder=8)
plt.scatter(xr_agn_props['combo_all_liners_hx_allxr'].full_xraylum,
            xr_agn_props['combo_all_liners_hx_allxr'].sfr, facecolor='orange',
            edgecolor='orange', s=17, label='LINERs', zorder=10)

plt.xlabel(r'log(L$_{\rm X, 0.5-10\ keV}$)',fontsize=20)
plt.ylabel(r'log(SFR)',fontsize=20)
plt.legend(frameon=False,fontsize=11,loc=2,bbox_to_anchor = (-0.00, 0.95))
plt.text(42.8,-3.1,'X-ray AGN\n Candidates', fontsize=15)
plt.xlim([37.,45.5])
plt.ylim([-3.5, 5])
ax.set(adjustable='box', aspect='equal')
plt.tight_layout() 
plt.savefig('plots/lx_full_selection_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lx_full_selection_.png', bbox_inches='tight', dpi=250)



fig = plt.figure()
ax = fig.add_subplot(111)


plotbptnormal(sym_diff_t2_filt.niiha,
            sym_diff_t2_filt.oiiihb, facecolor='none', ax=ax, fig=fig,scat=True,
            edgecolor='k',s=13, label='Unclassified type 2 AGNs', zorder=11, nobj=False)


plotbptnormal(xr_agn_props['unclass_p2_hx_o3'].niiha,
            xr_agn_props['unclass_p2_hx_o3'].oiiihb,marker='+', facecolor='r', ax=ax, fig=fig,scat=True,
            edgecolor='r', s=15, label='Weak or no emission lines', zorder=3, nobj=False)



plotbptnormal(xr_agn_props['bptplus_sf_allxr'].niiha,
            xr_agn_props['bptplus_sf_allxr'].oiiihb,marker='^',facecolor='g', ax=ax, fig=fig,scat=True,nobj=False,
            edgecolor='g', s=15,  label='BPT star-forming', zorder=15)

plotbptnormal(xr_agn_props['combo_sy2_hx_allxr'].niiha,
            xr_agn_props['combo_sy2_hx_allxr'].oiiihb, facecolor='b', ax=ax, fig=fig,scat=True,nobj=False,
            edgecolor='b',s=15, label='Seyfert 2s', zorder=10)
plotbptnormal(xr_agn_props['combo_all_liners_hx_allxr'].niiha,
            xr_agn_props['combo_all_liners_hx_allxr'].oiiihb, facecolor='orange', ax=ax,fig=fig,scat=True,nobj=False,
            edgecolor='orange', s=17, label='LINERs', zorder=10)

ax.set_ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
ax.set_xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)

plt.legend(frameon=False,fontsize=11,loc=2)#,bbox_to_anchor = (-0.00, 0.95))

ax.set(adjustable='box', aspect='equal')
plt.tight_layout() 
plt.savefig('plots/bpt_full_selection_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/bpt_full_selection_.png', bbox_inches='tight', dpi=250)



label='full'
scat=0.6

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(loglum_arr,lsfrrelat[label][2],'k--',label='Relation for normal galaxies (non-AGN)',zorder=3)
plt.fill_between(loglum_arr+scat,lsfrrelat[label][2],y2=lsfrrelat[label][2]-20,color='gray', zorder=0,alpha=0.5,linewidth=0)




plt.scatter(xr_agn_props['combo_sliner_hx_allxr'].full_xraylum,
            xr_agn_props['combo_sliner_hx_allxr'].sfr, facecolor='b', 
            edgecolor='b',s=15, label='S-LINERs', zorder=8)
plt.scatter(xr_agn_props['combo_hliner_hx_allxr'].full_xraylum,
            xr_agn_props['combo_hliner_hx_allxr'].sfr, facecolor='orange',
            edgecolor='orange', s=17, label='H-LINERs', zorder=10)

plt.xlabel(r'log(L$_{\rm X, 0.5-10\ keV}$)',fontsize=20)
plt.ylabel(r'log(SFR)',fontsize=20)
plt.legend(frameon=False,fontsize=11,loc=2,bbox_to_anchor = (-0.00, 0.95))
plt.text(42.8,-3.1,'X-ray AGN\n Candidates', fontsize=15)
plt.xlim([37.,45.5])
plt.ylim([-3.5, 5])
ax.set(adjustable='box', aspect='equal')
plt.tight_layout() 
plt.savefig('plots/lx_sfr_liners_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lx_sfr_liners_.png', bbox_inches='tight', dpi=250)






label='full'
scat=0.6

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(loglum_arr,lsfrrelat[label][2],'k--',label='Relation for normal galaxies (non-AGN)',zorder=3)
plt.fill_between(loglum_arr+scat,lsfrrelat[label][2],y2=lsfrrelat[label][2]-20,color='gray', zorder=0,alpha=0.3,linewidth=0)

plt.scatter(xr_agn_props['all_hx'].full_xraylum,
            xr_agn_props['all_hx'].sfr, facecolor='none', 
            edgecolor='k',s=10, label='Unclassified type 2 AGNs (from BPT)', zorder=1)
plt.scatter(xr_agn_props['x4_n2agn'].full_xraylum,
            xr_agn_props['x4_n2agn'].sfr,marker='*',facecolor='gray', 
            edgecolor='gray', s=12, label=r'Unclassified type 2 AGNs (from [NII]/H$\alpha$)', zorder=2)

plt.scatter(xr_agn_props['unclass_p2_hx'].full_xraylum,
            xr_agn_props['unclass_p2_hx'].sfr,marker='+', facecolor='r', 
            edgecolor='r', s=12, label='Unclassifiable (weak emission lines)', zorder=3)



plt.scatter(xr_agn_props['bptplus_sf_allxr'].full_xraylum,
            xr_agn_props['bptplus_sf_allxr'].sfr,marker='^',facecolor='g',
            edgecolor='g', s=12,  label='BPT star-forming', zorder=7)

plt.scatter(xr_agn_props['combo_sy2_hx_allxr'].full_xraylum,
            xr_agn_props['combo_sy2_hx_allxr'].sfr, facecolor='b', 
            edgecolor='b',s=12, label='Seyfert 2s', zorder=8)
plt.scatter(xr_agn_props['combo_all_liners_hx_allxr'].full_xraylum,
            xr_agn_props['combo_all_liners_hx_allxr'].sfr, facecolor='orange',
            edgecolor='orange', s=12, label='LINERs', zorder=10)

plt.xlabel(r'log(L$_{\rm x}$)',fontsize=20)
plt.ylabel(r'log(SFR)',fontsize=20)
plt.legend(frameon=False,fontsize=10,loc=2,bbox_to_anchor = (-0.02, 0.99))
plt.text(43.1,-3.05,'X-ray AGN\n Candidates', fontsize=14, rotation=0)
plt.xlim([37.,45.5])
plt.ylim([-3.5, 5])
ax.set(adjustable='box', aspect='equal')
plt.tight_layout()

plt.savefig('plots/lx_full_selection_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lx_full_selection_.png', bbox_inches='tight', dpi=250)
'''
