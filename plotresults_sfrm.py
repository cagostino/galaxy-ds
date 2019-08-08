#from matchgal_gsw2 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib
from matplotlib.colors import LogNorm
from sklearn import mixture
from ast_func import *
#import os
#os.environ['PATH']+=':~/texlive'
#from mpl_toolkits.basemap import Basemap
mydpi = 96
plt.rc('font',family='serif')
plt.rc('text',usetex=True)
#plt.rc('text',usetex=False)
def plot2dhist(x,y,nx,ny, nan=False, ax=None):
    if nan:
        fin = np.where((np.isfinite(x)) &(np.isfinite(y) ))[0]
        x= np.copy(x[fin])
        y=np.copy(y[fin])
    hist, xedges, yedges = np.histogram2d(x,y,bins = (nx,ny))
    extent= [np.min(xedges),np.max(xedges),np.min(yedges),np.max(yedges)]
    print(extent)
    if ax:
        ax.imshow((hist.transpose())**0.3, cmap='gray_r',extent=extent,origin='lower',
               aspect='auto',alpha=0.9) 
    else:
        plt.imshow((hist.transpose())**0.3, cmap='gray_r',extent=extent,origin='lower',
               aspect='auto',alpha=0.9) 


def histsf_diffs(diffs, matchtyp, save=False, filename='', bins=20, ran=(0,1)):
    if save:
        fig = plt.figure()
    plt.hist(diffs, color='k', histtype='step', bins=bins, range=ran)
    plt.xlabel('Distance to ' + match_typs[matchtyp], fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/distmet/histsf_diffs_'+ filename+'.png', dpi=250)
        fig.savefig('plots/sfrmatch/pdf/distmet/histsf_diffs_'+ filename+'.pdf', dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/histsf_diffs_'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
    
match_typs = {0:'BPT SF', 1:'BPT+ SF', 2:'Unclassifiable'}

'''

i=0
histsf_diffs(sfrm_gsw2.dists_best[i], sfrm_gsw2.best_types[i], filename =str(i)+'_avgn'+str(sfrm_gsw2.n_avg), save=True)

i=10
histsf_diffs(sfrm_gsw2.dists_best[i], Falsesfrm_gsw2.best_types[i], filename =str(i)+'_avgn'+str(sfrm_gsw2.n_avg), save=True)

i=20
histsf_diffs(sfrm_gsw2.dists_best[i], sfrm_gsw2.best_types[i], filename =str(i)+'_avgn'+str(sfrm_gsw2.n_avg), save=True)

i=30
histsf_diffs(sfrm_gsw2.dists_best[i], sfrm_gsw2.best_types[i], filename =str(i)+'_avgn'+str(sfrm_gsw2.n_avg), save=True)


'''



def get_avg_dist_n():
    avg_dists = []
    n_class = []
    n_class.append(sfrm_gsw2.bpt_sn_filt.size + sfrm_gsw2.bpt_plus_sn_filt.size + sfrm_gsw2.bpt_neither_sn_filt.size)
    avg_dists.append(sfrm_gsw2.mindists_best)

    for n_val in n_vals_used[1:]:
        sfrm_gsw2.getsfrmatch_avg(agn_gsw_bptplus, nonagn_gsw_bptplus, nonagn_gsw_bptplusnii, agn_gsw_bptplusnii, load=True, n_avg=n_val)
        sfrm_gsw2.subtract_elflux_avg(nonagn_gsw_bptplus, nonagn_gsw_bptplusnii,sncut=2)
        n_class.append(sfrm_gsw2.bpt_sn_filt_avg.size)
        avg_dists.append(sfrm_gsw2.mindists_avg)
    return np.array(avg_dists), np.array(n_class)
def get_sample_dist(dists, samp_num, save=False):
    sampl = []
    for i in range(len(dists)):
        sampl.append(dists[i][samp_num])
    fig1 = plt.figure()
    plt.scatter(n_vals_used, sampl,marker='x', color='k')
    
    plt.xlabel(r'n$_{\mathrm{Avg.}}$', fontsize=20)
    plt.ylabel('Avg. Sample Distance', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    if save:
        fig1.savefig('plots/sfrmatch/png/distmet/avg_sample_dists_'+ str(samp_num) +'.png', dpi=250)
        fig1.savefig('plots/sfrmatch/pdf/distmet/avg_sample_dists_'+ str(samp_num) +'.pdf', dpi=250)
        fig1.set_rasterized(True)
        fig1.savefig('plots/sfrmatch/eps/avg_sample_dists_'+ str(samp_num) +'.eps', dpi=150,format='eps')
        plt.close(fig1)
    fig2 = plt.figure()
    plt.scatter(np.log10(n_vals_used), sampl,marker='x', color='k')
    plt.xlabel(r'log(n$_{\mathrm{Avg.}}$)', fontsize=20)
    plt.ylabel('Avg. Sample Distance', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    if save:
        fig2.savefig('plots/sfrmatch/png/distmet/logavg_sample_dists_'+ str(samp_num) +'.png', dpi=250)
        fig2.savefig('plots/sfrmatch/pdf/distmet/logavg_sample_dists_'+ str(samp_num) +'.pdf', dpi=250)
        fig2.set_rasterized(True)
        fig2.savefig('plots/sfrmatch/eps/logavg_sample_dists_'+ str(samp_num) +'.eps', dpi=150,format='eps')
        plt.close(fig2)
def get_avg_dist(dists, save=False, filename=''):
    fig1 = plt.figure()
    plt.scatter(n_vals_used, np.mean(dists, axis=1),marker='x', color='k')    
    plt.xlabel(r'n$_{\mathrm{Avg.}}$', fontsize=20)
    plt.ylabel('Avg. Distance', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    if save:
        fig1.savefig('plots/sfrmatch/png/distmet/avg_dists.png', dpi=250)
        fig1.savefig('plots/sfrmatch/pdf/distmet/avg_dists.pdf', dpi=250)
        fig1.set_rasterized(True)
        fig1.savefig('plots/sfrmatch/eps/avg_dists.eps', dpi=150,format='eps')
        plt.close(fig1)
    fig2 = plt.figure()
    plt.scatter(np.log10(n_vals_used), np.mean(dists, axis=1),marker='x', color='k')
    plt.xlabel(r'log(n$_{\mathrm{Avg.}}$)', fontsize=20)
    plt.ylabel('Avg. Distance', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()    
    if save:
        fig2.savefig('plots/sfrmatch/png/distmet/logavg_dists.png', dpi=250)
        fig2.savefig('plots/sfrmatch/pdf/distmet/logavg_dists.pdf', dpi=250)
        fig2.set_rasterized(True)
        fig2.savefig('plots/sfrmatch/eps/logavg_dists.eps', dpi=150,format='eps')
        plt.close(fig2)
def get_med_dist(dists, save=False, filename=''):
    fig1 = plt.figure()
    plt.scatter(n_vals_used, np.median(dists, axis=1),marker='x', color='k')    
    plt.xlabel(r'n$_{\mathrm{Avg.}}$', fontsize=20)
    plt.ylabel('Median Distance', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    if save:
        fig1.savefig('plots/sfrmatch/png/distmet/med_dists.png', dpi=250)
        fig1.savefig('plots/sfrmatch/pdf/distmet/med_dists.pdf', dpi=250)
        fig1.set_rasterized(True)
        fig1.savefig('plots/sfrmatch/eps/distmet/med_dists.eps', dpi=150,format='eps')
        plt.close(fig1)
    fig2 = plt.figure()
    plt.scatter(np.log10(n_vals_used), np.median(dists, axis=1),marker='x', color='k')
    plt.xlabel(r'log(n$_{\mathrm{Avg.}}$)', fontsize=20)
    plt.ylabel('Median Distance', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()    
    if save:
        fig2.savefig('plots/sfrmatch/png/distmet/logmed_dists.png', dpi=250)
        fig2.savefig('plots/sfrmatch/pdf/distmet/logmed_dists.pdf', dpi=250)
        fig2.set_rasterized(True)
        fig2.savefig('plots/sfrmatch/eps/distmet/logmed_dists.eps', dpi=150,format='eps')
        plt.close(fig2)        


def plotnclass(n_class,matchtyp, save=False, filename=''):
    fig1 = plt.figure()
    plt.scatter(n_vals_used, n_class,marker='x', color='k')    
    plt.xlabel(r'n$_{\mathrm{Avg.}}$', fontsize=20)
    plt.ylabel(r'n$_{\mathrm{Classifiable}}$, '+matchtyp, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    if save:
        fig1.savefig('plots/sfrmatch/png/distmet/n_classifiable_'+filename+'.png', dpi=250)
        fig1.savefig('plots/sfrmatch/pdf/distmet/n_classifiable_'+filename+'.pdf', dpi=250)
        fig1.set_rasterized(True)
        fig1.savefig('plots/sfrmatch/eps/distmet/n_classifiable_'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig1)
    fig2 = plt.figure()
    plt.scatter(np.log10(n_vals_used), n_class,marker='x', color='k')
    plt.xlabel(r'log(n$_{\mathrm{Avg.}}$)', fontsize=20)
    plt.ylabel(r'n$_{\mathrm{Classifiable}}$, '+matchtyp, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()    
    if save:
        fig2.savefig('plots/sfrmatch/png/distmet/logn_classifiable_'+filename+'.png', dpi=250)
        fig2.savefig('plots/sfrmatch/pdf/distmet/logn_classifiable_'+filename+'.pdf', dpi=250)
        fig2.set_rasterized(True)
        fig2.savefig('plots/sfrmatch/eps/distmet/logn_classifiable_'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig2)                
'''
n_vals_used = [1, 3, 5, 10, 20, 50, 100]#[1,3,5,10,20,50,100]
avg_dists_n,n_class = get_avg_dist_n()
get_sample_dist(avg_dists_n, 1, save=True)
get_sample_dist(avg_dists_n, 10000, save=True)
get_sample_dist(avg_dists_n, 50000, save=True)
get_sample_dist(avg_dists_n, 80000, save=True)

get_avg_dist(avg_dists_n, save=True)
get_med_dist(avg_dists_n, save=True)

plotnclass(n_class, 'Match', filename='bpt', save=True)


'''

def plotfluxcomp(agn_flux, matches_flux, line, matchtyp, filename='', save=False):
    nagn_bigger = np.where(agn_flux > matches_flux)[0].size
    if save:
        fig = plt.figure()
    print('agn bigger: ', nagn_bigger)
    print('match bigger: ', matches_flux.size-nagn_bigger)
    mnx = np.percentile(agn_flux, 1)
    mxx = np.percentile(agn_flux, 99)
    mny = np.percentile(matches_flux, 1)
    mxy = np.percentile(matches_flux, 99)
    
    mxval = np.max([mxx,mxy])
    valbg = np.where((np.isfinite(agn_flux) & (np.isfinite(matches_flux))) &
            (matches_flux > mny) &( matches_flux < mxy) & (agn_flux< mxx)&(agn_flux > -1.5e-15) )[0]    
    plot2dhist(agn_flux[valbg], matches_flux[valbg], 250, 250)
    plt.plot([0,mxval],[0,mxval],'b--')
    plt.text(mxval/8, mxval-mxval/5, r'N$_{\mathrm{above}}$ = '+str(matches_flux.size-nagn_bigger) + '(' +str(round((matches_flux.size-nagn_bigger)/matches_flux.size,3)*100)[0:4]+'\%)', fontsize=15)
    
    plt.axvline(x=0, color='r', ls=':')
    plt.axhline(y=0, color='r', ls=':')
    plt.xlabel(line+' Flux, AGN', fontsize=20)
    plt.ylabel(line+' Flux, ' + ' Match', fontsize=20)
    
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/fluxcomp_'+ filename+'.png', dpi=250)
        fig.savefig('plots/sfrmatch/pdf/fluxcomp_'+filename+'.pdf', dpi=250)
        fig.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/fluxcomp_'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)

'''
plotfluxcomp(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.neither_agn], sfrm_gsw2.eldiag.oiiiflux_neither[sfrm_gsw2.neither_matches],'[OIII]', 'Unclassified', filename='unclass_oiii', save=False)
plotfluxcomp(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.neither_agn], sfrm_gsw2.eldiag.hbetaflux_neither[sfrm_gsw2.neither_matches],r'H$\beta$', 'Unclassified', filename='unclass_hbeta', save=False)
plotfluxcomp(sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.neither_agn], sfrm_gsw2.eldiag.halpflux_neither[sfrm_gsw2.neither_matches],r'H$\alpha$', 'Unclassified', filename='unclass_halp', save=False)
plotfluxcomp(sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.neither_agn], sfrm_gsw2.eldiag.niiflux_neither[sfrm_gsw2.neither_matches],'[NII]', 'Unclassified', filename='unclass_nii', save=False)

plotfluxcomp(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.agns_plus], sfrm_gsw2.eldiag.oiiiflux_plus[sfrm_gsw2.sfs_plus],'[OIII]', 'NII/Ha SF', filename='plus_oiii', save=True)
plotfluxcomp(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.agns_plus], sfrm_gsw2.eldiag.hbetaflux_plus[sfrm_gsw2.sfs_plus],r'H$\beta$', 'NII/Ha SF', filename='plus_hbeta', save=True)
plotfluxcomp(sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.agns_plus], sfrm_gsw2.eldiag.halpflux_plus[sfrm_gsw2.sfs_plus],r'H$\alpha$', 'NII/Ha SF', filename='plus_halp', save=True)
plotfluxcomp(sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.agns_plus], sfrm_gsw2.eldiag.niiflux_plus[sfrm_gsw2.sfs_plus],'[NII]', 'NII/Ha SF', filename='plus_nii', save=True)

plotfluxcomp(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.sfs],'[OIII]', 'BPT SF', filename='bpt_oiii', save=True)
plotfluxcomp(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.sfs],r'H$\beta$', 'BPT SF', filename='bpt_hbeta', save=True)
plotfluxcomp(sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.sfs],r'H$\alpha$', 'BPT SF', filename='plus_halp', save=True)
plotfluxcomp(sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.sfs],'[NII]', 'BPT SF', filename='plus_nii', save=True)



n_pos_unclass = np.where((sfrm_gsw2.niiflux_sub_sn_neither>0)&
         (sfrm_gsw2.oiiiflux_sub_sn_neither>0) &
         (sfrm_gsw2.halpflux_sub_sn_neither>0) &
         (sfrm_gsw2.hbetaflux_sub_sn_neither>0))[0].size

perc_pos_unclass = n_pos_unclass /sfrm_gsw2.neither_agn.size

n_pos_niiha = np.where((sfrm_gsw2.niiflux_sub_sn_plus>0)&
         (sfrm_gsw2.oiiiflux_sub_sn_plus>0) &
         (sfrm_gsw2.halpflux_sub_sn_plus>0) &
         (sfrm_gsw2.hbetaflux_sub_sn_plus>0))[0].size

perc_pos_niiha = n_pos_niiha /sfrm_gsw2.agns_plus.size

n_pos_bpt = np.where((sfrm_gsw2.niiflux_sub_sn>0)&
         (sfrm_gsw2.oiiiflux_sub_sn>0) &
         (sfrm_gsw2.halpflux_sub_sn>0) &
         (sfrm_gsw2.hbetaflux_sub_sn>0))[0].size
perc_pos_bpt = n_pos_bpt /sfrm_gsw2.agns.size
                       
'''
def plthist(bincenters, counts):

    plt.plot(bincenters, counts,color='k', drawstyle='steps-mid')
def hist_sfrdiffs(diffs, lab, ymax=16000, col='k', ran=[], bins=30, save = False, filename=''):
    fig = plt.figure()
    if len(ran) !=0:
        plt.hist(diffs,color=col, bins=bins, range=ran, histtype='step', label=lab+', Median dist: '+str(np.median(diffs))[0:5])
    else:
        plt.hist(diffs,color=col, bins=bins, histtype='step', label=lab+', Median dist: '+str(np.median(diffs))[0:5])
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.xlabel('Distance to closest match',fontsize=20)
    plt.ylabel(r'N$_{matches}$',fontsize=20)    
    plt.legend(frameon=False,fontsize=15)
    plt.ylim([0,ymax])
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/match_dists_'+filename+'.png', dpi=250)
        fig.savefig('plots/sfrmatch/pdf/match_dists_'+filename+'.pdf', dpi=250)
        fig.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/match_dists_'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)

    else:
        plt.show()

'''
hist_sfrdiffs(sfrm_gsw2.mindists_best, 'Combined', ymax=30000, col='k', ran=(0,1), bins=40, save=True, filename='combined')


hist_sfrdiffs(sfrm_gsw2.mindists[0], 'BPT+', col='r',  ymax=30000,ran=(0,1), bins=40, save=False, filename='bptplus')

hist_sfrdiffs(sfrm_gsw2.mindists[1],'NII/Ha', col='g', ymax=30000, ran=(0,1), bins=40, save=False, filename='niihalp')

hist_sfrdiffs(sfrm_gsw2.mindists[2],'Unclassified', col='b',  ymax=30000, ran=(0,1), bins=40, save=False, filename='unclass')

hist_sfrdiffs(sfrm_gsw2.mindistsagn_best, 'AGN Combined', ymax=40000, col='k', ran=(0,1), bins=40, save=True, filename='combined_agn')
hist_sfrdiffs(sfrm_gsw2.mindists_agn[0], 'BPT AGN', ymax=40000, col='r', ran=(0,1), bins=40, save=True, filename='agn')
hist_sfrdiffs(sfrm_gsw2.mindists_agn[1], 'NII/Ha AGN', ymax=40000, col='g', ran=(0,1), bins=40, save=True, filename='agnplus')



hist_sfrdiffs(sfrm_gsw2.mindists_best, 'Combined', ymax=30000, col='k', ran=(0,1), bins=40, save=True, filename='old_combined')


hist_sfrdiffs(sfrm_gsw2.mindists[0], 'BPT+', col='r',  ymax=30000,ran=(0,1), bins=40, save=True, filename='old_bptplus')

hist_sfrdiffs(sfrm_gsw2.mindists[1],'NII/Ha', col='g', ymax=30000, ran=(0,1), bins=40, save=True, filename='old_niihalp')

hist_sfrdiffs(sfrm_gsw2.mindists[2],'Unclassified', col='b',  ymax=30000, ran=(0,1), bins=40, save=True, filename='old_unclass')

hist_sfrdiffs(sfrm_gsw2.mindistsagn_best, 'AGN Combined', ymax=40000, col='k', ran=(0,1), bins=40, save=True, filename='old_combined_agn')
hist_sfrdiffs(sfrm_gsw2.mindists_agn[0], 'BPT AGN', ymax=40000, col='r', ran=(0,1), bins=40, save=True, filename='old_agn')
hist_sfrdiffs(sfrm_gsw2.mindists_agn[1], 'NII/Ha AGN', ymax=40000, col='g', ran=(0,1), bins=40, save=True, filename='old_agnplus')
'''    


#plot_xmmtime(np.log10(x3.medtimes),r'log(t$_{\mathrm{exp}}$)[s]', save=True)
#plot_xmmtime(np.log10(x3.alltimes),r'log(t$_{\mathrm{exp}}$)')
#plot_xmmtime(np.log10(x3.deeptimes),r'log(t$_{\mathrm{exp}}$)',nbins=55)

def plotdistrat(dists, label, rang=(0.5, 1.5), bins=40, save = False, filename=''):
    if save:
        fig = plt.figure()
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    
    plt.hist(dists, range=rang, histtype='step', bins=bins)    
    plt.xlabel(r'$(d_{Match}/d_{AGN})^2$', fontsize=20)
    plt.title(label, fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/distrat_'+filename+'.png',dpi=250)
        fig.savefig('plots/sfrmatch/pdf/distrat_'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/distrat_'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotdistrat(sfrm_gsw2.bptdistrat, 'BPT',save=True, filename='bpt')
plotdistrat(sfrm_gsw2.bptdistrat_plus, 'BPT Plus', save=True, filename='bptplus')


plotdistrat(sfrm_gsw2.bptdistrat_neither, 'Unclassifiable', save=True,filename='unclassifiable')


'''

def reject_outliers(data, m=2):
    return np.array([abs(data - np.mean(data)) < m * np.std(data)])[0]


#getallgdiffhistsbptagn(xmm3gdiff, xmm3ids, agn_3xmmm_xrfilt)
def massfrachist(massfrac,lab=''):
    val = np.where((massfrac >0)&(massfrac<2))[0]
    plt.hist(massfrac[val],label=lab)
    plt.xlabel('Mass Fraction')
    plt.ylabel('Counts')
    plt.xlim([0,1])
    plt.tight_layout()
#massfrachist(all_sdss_massfrac)
def xrayhists(prop,propname,nbins=20):
    plt.hist(prop,histtype='step',bins=nbins)
    plt.xlabel(propname,fontsize=20)
    stdprop = np.std(prop)
    meanprop  =np.mean(prop)
    print('var: ', stdprop)
    print('mean: ',meanprop)
    plt.axvline(x=meanprop,label='Mean: '+ str(meanprop)[0:5])
    plt.axvline(x=meanprop+stdprop, ls='--',label=r'$\sigma$: ' +str(stdprop)[0:5])
    plt.axvline(x=meanprop-stdprop,ls='--')
    plt.ylabel('Counts',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.legend()
    plt.tight_layout()

def plotbpt(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=False,filename='',labels=True, leg=True,title=None, minx=-2, maxx=1, miny=-1.2, maxy=1.2):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()

    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > miny) &( bgy < maxy) & (bgx<maxx)&(bgx > minx) )[0]
    nx = 3/0.01
    ny = 2.4/0.01
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(list(nonagnfilt))!=0:
        plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
    if len(list(agnfilt)) !=0:
        plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    if labels:
        plt.text(0.6,0.75,'AGN', fontsize=15)
        plt.text(-1.15,-0.3,'HII',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([miny-0.1,maxy])
    plt.xlim([minx-0.1, maxx+0.1])
    if leg:
        plt.legend(fontsize=15,frameon=False,loc=3,bbox_to_anchor=(-0.02, -0.02))
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/NII_OIII_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/NII_OIII_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        #fig.savefig('plots/xmm3/eps/diagnostic/NII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()

def plotbptnormal(bgx,bgy,save=False,filename='',labels=True, title=None, minx=-2, maxx=1, miny=-1.2, maxy=1.2):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > miny) &( bgy < maxy) & (bgx<maxx)&(bgx > minx) )[0]
    nx = 3/0.01
    ny = 2.4/0.01
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    plt.text(-1.7, -1, r"N$_{\mathrm{obj}}$: "+str(len(valbg)), fontsize=20)
    if labels:
        plt.text(0.6,0.75,'AGN', fontsize=15)
        plt.text(-1.15,-0.3,'HII',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([miny-0.1,maxy])
    plt.xlim([minx-0.1, maxx+0.1])
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/NII_OIII_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/sfrmatch/pdf/diagnostic/NII_OIII_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/NII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt]),filename='sn2', save=True, minx=-3.0, maxx=2.0, miny=-2, maxy=2)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,filename='fluxsub_highsn_bpt', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,filename='fluxsub_highsn_plus', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]),filename='fluxsub_x_only_highsn_sfplus', save=True)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,filename='fluxsub_highsn_neither', save=True)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub[sfrm_gsw2.bpt_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.bpt_sn_filt],filename='fluxsub_old_bpt', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.bpt_plus_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.bpt_plus_sn_filt],filename='fluxsub_old_plus', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.bpt_plus_sn_filt] ,  np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus])[sfrm_gsw2.bpt_plus_sn_filt],filename='fluxsub_x_only_old_sfplus', save=True)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.bpt_neither_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.bpt_neither_sn_filt],filename='fluxsub_old_neither', save=True)


plotbptnormal(sfrm_gsw2.log_nii_halpha_sub[sfrm_gsw2.bpt_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.bpt_sn_filt],filename='fluxsub_bpt_snfilt2', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.bpt_plus_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.bpt_plus_sn_filt],filename='fluxsub_plus_snfilt0', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.bpt_neither_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.bpt_neither_sn_filt],filename='fluxsub_neither_snfilt0', save=False, minx=-3.0, maxx=2.0, miny=-2, maxy=2)


sfrm_gsw2.getsfrmatch(agn_gsw_bptplus, nonagn_gsw_bptplus, nonagn_gsw_bptplusnii, agn_gsw_bptplusnii, load=True)
sfrm_gsw2.subtract_elflux(sncut=1)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.bpt_sing_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.bpt_sing_sn_filt],filename='fluxsub_highsn_bpt_sing', 
              save=True)

for nn in n_vals_used[1:]:
    sfrm_gsw2.getsfrmatch_avg(agn_gsw_bptplus, nonagn_gsw_bptplus, nonagn_gsw_bptplusnii, agn_gsw_bptplusnii, load=True, n_avg=nn)
    sfrm_gsw2.subtract_elflux_avg(nonagn_gsw_bptplus, nonagn_gsw_bptplusnii,sncut=1)
    
    plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_avg[sfrm_gsw2.bpt_sn_filt_avg] ,  sfrm_gsw2.log_oiii_hbeta_sub_avg[sfrm_gsw2.bpt_sn_filt_avg],filename='fluxsub_bpt_snfilt1_avg'+str(sfrm_gsw2.n_avg),
                  save=True, minx=-3.0, maxx=2.0, miny=-2, maxy=2)


plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus_avg[sfrm_gsw2.bpt_plus_sn_filt_avg] ,  sfrm_gsw2.log_oiii_hbeta_sub_plus_avg[sfrm_gsw2.bpt_plus_sn_filt_avg],filename='fluxsub_plus_snfilt2_avg'+str(sfrm_gsw2.n_avg),
              save=True, minx=-3.0, maxx=2.0, miny=-2, maxy=2)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither_avg[sfrm_gsw2.bpt_neither_sn_filt_avg] ,  sfrm_gsw2.log_oiii_hbeta_sub_neither_avg[sfrm_gsw2.bpt_neither_sn_filt_avg],filename='fluxsub_neither_snfilt2_avg'+str(sfrm_gsw2.n_avg),
              save=False, minx=-3.0, maxx=2.0, miny=-2, maxy=2)



plotbptnormal(sfrm_gsw2.log_nii_halpha_sub[sfrm_gsw2.bpt_sn_filt_intermed] ,  sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.bpt_sn_filt_intermed],filename='fluxsub_bpt_snfilt_0-1', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.bpt_plus_sn_filt_intermed] ,  sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.bpt_plus_sn_filt_intermed],filename='fluxsub_plus_snfilt_0-1', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.bpt_neither_sn_filt_intermed] ,  sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.bpt_neither_sn_filt_intermed],filename='fluxsub_neither_snfilt_0-1', save=False)



plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.bpt_plus_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.bpt_plus_sn_filt],filename='fluxsub_plus_snfilt0', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.bpt_neither_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.bpt_neither_sn_filt],filename='fluxsub_neither_snfilt0', save=False, minx=-3.0, maxx=2.0, miny=-2, maxy=2)

#high_sn things

plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns]),filename='_highsn_agns', save=True)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]),filename='_highsn_agns_plus', save=True)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.neither_agn]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.neither_agn]),filename='_highsn_neither_agns', save=True)

#no sn req
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns]),filename='_old_agns', save=True)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]),filename='_old_agns_plus', save=True)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.neither_agn]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.neither_agn]),filename='_old_neither_agns', save=True)

plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_selfmatch]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_selfmatch]),filename='_agns_selfmatch', save=False)

plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agnsplus_selfmatch]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agnsplus_selfmatch]),filename='_agnsplus_selfmatch', save=False)

plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.sfs]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.sfs]),filename='_old_sfs', save=True)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.sfs]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.sfs]),filename='_highsn_sfs', save=True)

plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_snlr_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_snlr_filt]),filename='sn_lr3', save=True)
    
'''
def plotbptplus(bgx, bgy, bgxhist, unclass, nonagn=[], agn=[], save=False,filename='',labels=True, title=None,nii_bound=nii_bound):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    fig = plt.figure(figsize=(6,5))
    if title:
        plt.title(title,fontsize=30)    
    ax1 = plt.subplot2grid((3,2),(0,0), colspan=2, rowspan=2)

    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > -1.2) &( bgy < 1.2) & (bgx<1)&(bgx > -2) )[0]
    nx = 3/0.01
    ny = 2.4/0.01
    bgxhist_finite = bgxhist#[np.where(np.isfinite(bgxhist))[0]]

    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ax = ax1)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    ax1.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    ax1.plot(np.log10(xline1_kauffman),np.log10(yline_stasinska),c='k',ls='-.')#,label='Kauffman Line')

    ax1.set_xlim([-2.1,1])
    ax1.set_ylim([-1.3,1.2])
    ax1.axvline(x=nii_bound,color='k', alpha=0.8, linewidth=1, ls='-')
    #ax1.axvline(x=nii_bound+0.05,color='k', alpha=0.8, linewidth=1, ls='-')
    
    plt.text(-1.8, -0.4, r"N$_{\mathrm{obj}}$: "+str(len(bgx))+'('+str(round(100*len(bgx)/(len(bgxhist_finite)+len(bgx) +len(unclass))))+'\%)', fontsize=15)
    if len(nonagn) != 0 and len(agn) != 0:
        plt.text(-1.8, -1, r"N$_{\mathrm{AGN}}$: "+str(len(agn)) +'('+str(round(100*len(agn)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)
        plt.text(-1.8, -0.7, r"N$_{\mathrm{SF}}$: "+str(len(nonagn))+'('+str(round(100*len(nonagn)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)
        
    if labels:
        ax1.text(0.6,0.75,'AGN', fontsize=15)
        ax1.text(-1.15,-0.3,'HII',fontsize=15)
    ax1.set_ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.setp(ax1.get_xticklabels(), visible=False)
   
    #plt.xlim([-2.1,1])
    ax2 =  plt.subplot2grid((3,2),(2,0), colspan=2, rowspan=1)
    #fig.add_subplot(212,sharex=ax1)
    print('len(bgxhist): ',len(bgxhist))
    
    print('len(bgxhist_finite): ',len(bgxhist_finite))
    
    cnts, bins = np.histogram(bgxhist, bins=100, range=(-2,1))#ax2.hist(bgxhist_finite, bins=250,range = (-2,1), histtype='step', density=True, stacked=True)
    bncenters = (bins[1:] + bins[0:-1])/2
    plthist(bncenters, cnts/len(bgxhist_finite))
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    ax2.set_xlim([-2.1,1])
    ax2.set_xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    ax2.set_ylabel(r'Fraction',fontsize=20)
    #ax2.set_xticks(np.arange(0.1, 1, 0.1), minor = True)
    ax2.axvline(x=nii_bound,ls='-',linewidth=1, color='k', alpha=0.8 )
    #ax2.axvline(x=nii_bound+0.05,ls='-',linewidth=1, color='k', alpha=0.8 )
    
    nii_agn = np.where(bgxhist_finite >nii_bound)[0]
    nii_sf = np.where(bgxhist_finite <nii_bound)[0]
    #axs[1].set_xlim([-1.3,1.2])
    plt.text(-1.8, 6.25/8 * np.max(cnts)/len(bgxhist_finite), r"N$_{\mathrm{obj}}$: " + str(len(bgxhist_finite)) +'('+str(round(100*len(bgxhist_finite)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)
    plt.text(-1.8, 4.25/8 * np.max(cnts)/len(bgxhist_finite), r"N$_{\mathrm{AGN}}$: "+str(len(nii_agn))+'('+str(round(100*len(nii_agn)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)
    plt.text(-1.8, 2.25/8 * np.max(cnts)/len(bgxhist_finite), r"N$_{\mathrm{SF}}$: "+str(len(nii_sf))+'('+str(round(100*len(nii_sf)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)
    if len(unclass) != 0:
        plt.text(-1.8, 0.25/8 * np.max(cnts)/len(bgxhist_finite), r"N$_{\mathrm{unclass}}$: "+str(len(unclass)) +'('+str(round(100*len(unclass)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)

    #ax2.set_aspect(10)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if save:
        fig.savefig('./plots/sfrmatch/png/diagnostic/NII_OIII_scat'+filename+'.png',dpi=250)
        fig.savefig('./plots/sfrmatch/pdf/diagnostic/NII_OIII_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        #fig.savefig('./plots/sfrmatch/eps/NII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    
'''
plotbptplus(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agnsplus_selfmatch]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agnsplus_selfmatch]),
            np.log10(EL_m2.xvals1_bpt[EL_m2.halp_nii_filt][sfrm_gsw2.agnsplus_selfmatch_other]), [], nonagn=[], agn= [],
            filename='_bptplus_agns_selfmatch', save=False, labels=False)


plotbptplus(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]),
            np.log10(EL_m2.xvals1_bpt[EL_m2.halp_nii_filt][sfrm_gsw2.sfs_plus]), [], nonagn=[], agn= [],
            filename='_old_bptplus_agns_sfs', save=True, labels=False)

plotbptplus(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]),
            np.log10(EL_m2.xvals1_bpt[EL_m2.halp_nii_filt][sfrm_gsw2.sfs_plus]), [], nonagn=[], agn= [],
            filename='_highsn_bptplus_agns_sfs', save=True, labels=False)

plotbptplus(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_minus_bpt_sn_lr2_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_minus_bpt_sn_lr2_filt]),
            np.log10(EL_m2.xvals1_bpt[EL_m2.halp_nii_minus_halp_nii_lr2_filt]), [], nonagn=nonagn_gsw_sn2_minus_lr2, agn= agn_gsw_sn2_minus_lr2,
            filename='bptplus_sn2_minus_lr3', save=False, labels=False)

plotbptplus(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt]),
            np.log10(EL_m2.xvals1_bpt[EL_m2.halp_nii_filt]), EL_m2.neither_filt, nonagn=nonagn_gsw_sn2, agn= agn_gsw_sn2,
            filename='bptplus_sn2', save=False, labels=False)

plotbptplus(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn3_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn3_filt]),
            np.log10(EL_m2.xvals1_bpt[EL_m2.halp_nii3_filt]),EL_m2.neither3_filt, nonagn =nonagn_gsw_sn3, agn=agn_gsw_sn3, 
            filename='bptplus_sn3', save=True, labels=False)


plotbptplus(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_lr3_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_lr3_filt]),
            np.log10(EL_m2.xvals1_bpt[EL_m2.halp_nii_lr3_filt]), EL_m2.neither_lr3_filt, nonagn =nonagn_gsw_snlr3, agn=agn_gsw_snlr3, 
            filename='bptplus_sn_lr3', save=True, labels=False)
    
plotbptplus(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_lr2_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_lr2_filt]),
            np.log10(EL_m2.xvals1_bpt[EL_m2.halp_nii_lr2_filt]), EL_m2.neither_lr2_filt,  nonagn =nonagn_gsw_snlr2, agn=agn_gsw_snlr2, 
            filename='bptplus_sn_lr2', save=True, labels=False)
    
plotbptplus(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_lr1_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_lr1_filt]),
            np.log10(EL_m2.xvals1_bpt[EL_m2.halp_nii_lr1_filt]), EL_m2.neither_lr1_filt,  nonagn =nonagn_gsw_snlr1, agn=agn_gsw_snlr1, 
            filename='bptplus_sn_lr1', save=False, labels=False)
    


plotbptnormal( EL_m2.niiha,EL_m2.oiiihb,save=False)

plotbptplus( EL_m2.niiha,EL_m2.oiiihb, EL_m2.niihaplus, save=False)

plotbpt(EL_3xmm.niiha,EL_3xmm.oiiihb,
         nonagn_3xmmm,
         agn_3xmmm,
         EL_m1.niiha,EL_m1.oiiihb,save=False)


'''

def mass_z(bgx,bgy,save=True,filename='',title=None, leg=False):
    '''
    for doing mass against z with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > 8) &( bgy < 12.5) & (bgx<0.3)&(bgx > 0) )[0]
    nx = 0.3/0.001
    ny = 4/0.01
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([8, 12.5])
    plt.xlim([0,0.3])
    plt.ylabel(r'log(M$_{\rm *}$)',fontsize=20)
    plt.xlabel(r'z',fontsize=20)

    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/mass_z'+filename+'.png',dpi=250)
        fig.savefig('plots/sfrmatch/pdf/mass_z'+filename+'.pdf',dpi=250, format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/mass_z'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''


mass_z(EL_m2.z[agn_gsw_bptplus],EL_m2.mass[agn_gsw_bptplus],save=True, filename= '_agn_bptplus')
mass_z(EL_m2.z[nonagn_gsw_bptplus],EL_m2.mass[nonagn_gsw_bptplus],save=True, filename= '_sf_bptplus')

mass_z(EL_m2.z_plus[agn_gsw_bptplusnii],EL_m2.mass_plus[agn_gsw_bptplusnii],save=True, filename= '_agn_bptplusnii')
mass_z(EL_m2.z_plus[nonagn_gsw_bptplusnii],EL_m2.mass_plus[nonagn_gsw_bptplusnii],save=True, filename= '_sf_bptplusnii')

mass_z(EL_m2.z_neither,EL_m2.mass_neither,save=True, filename= '_unclass')


'''

def dust_comp_sf_sub(bgx,bgy,save=True,filename='',title=None, leg=False):
    '''
    for doing A_V for SF versus AGN
    '''
    if save:
        fig = plt.figure()
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -6) &( bgy < 6) & (bgx<6)&(bgx > -6) )[0]
    nx = 12/0.02
    ny = 12/0.02
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([-6, 6])
    plt.xlim([-6, 6])
    plt.ylabel(r'A$_{\rm V, AGN-match}$',fontsize=20)
    plt.xlabel(r'A$_{\rm V, match}$',fontsize=20)

    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/dustcomp_sf_sub'+filename+'.png',dpi=250)
        fig.savefig('plots/sfrmatch/pdf/dustcomp_sf_sub'+filename+'.pdf',dpi=250, format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/dustcomp_sf_sub'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''
dust_comp_sf_sub(sfrm_gsw2.av_sf, sfrm_gsw2.av_sub,save=True, filename= '_fluxsub_bpt')
dust_comp_sf_sub(sfrm_gsw2.av_sf[sfrm_gsw2.bpt_sn_filt] ,  sfrm_gsw2.av_sub[sfrm_gsw2.bpt_sn_filt],filename='_fluxsub_bpt_snfilt', save=True)

dust_comp_sf_sub(sfrm_gsw2.av_plus_sf, sfrm_gsw2.av_sub_plus,save=True, filename= '_fluxsub_plus')
dust_comp_sf_sub(sfrm_gsw2.av_plus_sf[sfrm_gsw2.bpt_plus_sn_filt] ,  sfrm_gsw2.av_sub_plus[sfrm_gsw2.bpt_plus_sn_filt],filename='_fluxsub_plus_snfilt', save=True)


dust_comp_sf_sub(sfrm_gsw2.av_neither_sf ,  sfrm_gsw2.av_sub_neither,filename='_fluxsub_neither', save=True)
dust_comp_sf_sub(sfrm_gsw2.av_neither_sf[sfrm_gsw2.bpt_neither_sn_filt] ,  sfrm_gsw2.av_sub_neither[sfrm_gsw2.bpt_neither_sn_filt],filename='_fluxsub_neither_snfilt', save=True)
'''


def dust_comp_agn_sub(bgx,bgy,save=True,filename='',title=None, leg=False):
    '''
    for doing A_V for SF versus AGN
    '''
    if save:
        fig = plt.figure()
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -6) &( bgy < 6) & (bgx<6)&(bgx > -6) )[0]
    nx = 12/0.02
    ny = 12/0.02
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([-6, 6])
    plt.xlim([-6, 6])
    plt.ylabel(r'A$_{\rm V, AGN-match}$',fontsize=20)
    plt.xlabel(r'A$_{\rm V, AGN}$',fontsize=20)

    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/dustcomp_agn_sub'+filename+'.png',dpi=250)
        fig.savefig('plots/sfrmatch/pdf/dustcomp_agn_sub'+filename+'.pdf',dpi=250, format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/dustcomp_agn_sub'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''
dust_comp_agn_sub(sfrm_gsw2.av_agn, sfrm_gsw2.av_sub,save=True, filename= '_fluxsub_bpt')
dust_comp_agn_sub(sfrm_gsw2.av_agn[sfrm_gsw2.bpt_sn_filt] ,  sfrm_gsw2.av_sub[sfrm_gsw2.bpt_sn_filt],filename='_fluxsub_bpt_snfilt', save=True)

dust_comp_agn_sub(sfrm_gsw2.av_plus_agn, sfrm_gsw2.av_sub_plus,save=True, filename= '_fluxsub_plus')
dust_comp_agn_sub(sfrm_gsw2.av_plus_agn[sfrm_gsw2.bpt_plus_sn_filt] ,  sfrm_gsw2.av_sub_plus[sfrm_gsw2.bpt_plus_sn_filt],filename='_fluxsub_plus_snfilt', save=True)


dust_comp_agn_sub(sfrm_gsw2.av_neither_agn ,  sfrm_gsw2.av_sub_neither,filename='_fluxsub_neither', save=True)
dust_comp_agn_sub(sfrm_gsw2.av_neither_agn[sfrm_gsw2.bpt_neither_sn_filt] ,  sfrm_gsw2.av_sub_neither[sfrm_gsw2.bpt_neither_sn_filt],filename='_fluxsub_neither_snfilt', save=True)
'''


def plotssfrm(bgx,bgy,save=True,filename='', title=None, leg=True):
    '''
    for doing ssfrmass diagram with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()

    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -14) &( bgy < -8) & (bgx<12.5)&(bgx > 7.5) )[0]
    nx = 5/0.01
    ny = 6.0/0.01
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylabel(r'log(sSFR)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}})$',fontsize=20)
    plt.xlim([7.5,12.5])
    plt.ylim([-14,-8])

    if title:
        plt.title(title,fontsize=30)
    if leg:
        plt.legend(fontsize=15,frameon = False, loc=3, bbox_to_anchor=(-0.04, .05))    
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/ssfr_mass'+filename+'.png',dpi=250)
        fig.savefig('plots/sfrmatch/pdf/ssfr_mass'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/ssfr_mass'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:

        plt.show()
'''


plotssfrm(sfrm_gsw2.mass, sfrm_gsw2.ssfr, save=True,filename='_fluxsub_bpt')
plotssfrm(sfrm_gsw2.mass[sfrm_gsw2.bpt_sn_filt], sfrm_gsw2.ssfr[sfrm_gsw2.bpt_sn_filt], save=True,filename='_fluxsub_bpt_snfilt')
plotssfrm(sfrm_gsw2.mass_plus, sfrm_gsw2.ssfr_plus, save=True,filename= '_fluxsub_plus', )
plotssfrm(sfrm_gsw2.mass_plus[sfrm_gsw2.bpt_plus_sn_filt], sfrm_gsw2.ssfr_plus[sfrm_gsw2.bpt_plus_sn_filt], save=True,filename='_fluxsub_plus_snfilt')
plotssfrm(sfrm_gsw2.mass_neither, sfrm_gsw2.ssfr_neither, save=True, filename='_fluxsub_neither')
plotssfrm(sfrm_gsw2.mass_neither[sfrm_gsw2.bpt_neither_sn_filt], sfrm_gsw2.ssfr_neither[sfrm_gsw2.bpt_neither_sn_filt], save=True,filename='_fluxsub_neither_filt')


'''





def plotoiiimass(bgx, bgy, save=False, filename='',
                 title=None, alph=0.1, leg=True):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''

    if save:
        fig = plt.figure()

    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > 36) &( bgy < 44) & (bgx<12)&(bgx > 9) )[0]
    nx = 3/0.03
    ny = 8/0.08
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylabel(r'log(L$_\mathrm{[OIII]}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([36,44])
    plt.xlim([9,12])
    if leg:
        plt.legend(fontsize=15, frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/OIIILum_Mass_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/sfrmatch/pdf/OIIILum_Mass_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/OIIILum_Mass_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiimass(sfrm_gsw2.mass,np.log10(sfrm_gsw2.oiiilum_sub_dered), save=True, leg=False, filename='_fluxsub_bpt')
plotoiiimass(sfrm_gsw2.mass[sfrm_gsw2.bpt_sn_filt],np.log10(sfrm_gsw2.oiiilum_sub_dered[sfrm_gsw2.bpt_sn_filt]), save=True, leg=False, filename='_fluxsub_bpt_snfilt')


plotoiiimass(sfrm_gsw2.mass_plus,np.log10(sfrm_gsw2.oiiilum_sub_plus_dered), leg=False, save=True, filename='_fluxsub_plus')
plotoiiimass(sfrm_gsw2.mass_plus[sfrm_gsw2.bpt_plus_sn_filt],np.log10(sfrm_gsw2.oiiilum_sub_plus_dered[sfrm_gsw2.bpt_plus_sn_filt]), save=True, leg=False, filename='_fluxsub_plus_snfilt')

plotoiiimass(sfrm_gsw2.mass_neither,np.log10(sfrm_gsw2.oiiilum_sub_neither_dered),leg=False,save=True, filename='_fluxsub_neither')
plotoiiimass(sfrm_gsw2.mass_neither[sfrm_gsw2.bpt_neither_sn_filt],np.log10(sfrm_gsw2.oiiilum_sub_neither_dered[sfrm_gsw2.bpt_neither_sn_filt]),leg=False,save=True, filename='_fluxsub_neither_snfilt')

#BPT SF BG

'''




def plotoiiimassedd(bgx, bgy, save=True, filename='',
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[]):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''

    if save:
        fig = plt.figure()
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > 25) &( bgy < 34) & (bgx<12.5)&(bgx > 8) )[0]
    nx = 4.5/0.01
    ny = 9/0.01
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')

    plt.ylabel(r'log(L$_\mathrm{[OIII]}$/M$_{\mathrm{*}}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    #plt.ylim([37,43])
    #plt.xlim([8,12])

    plt.legend(fontsize=15, frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/OIIILum_MassEdd_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILum_MassEdd_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILum_MassEdd_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiimassedd(EL_m1.mass, np.log10(EL_m1.oiiilum)-EL_m1.mass,
        save=True,alph=0.1)

'''
 


def plotbpt_sii(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=False,filename='',title=None,alph=0.1,ccode=[],ccodegsw=[],cont=None,ccodename='X-ray Fraction',levs=[0,0.2]):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > -1.2) &( bgy < 1.2) & (bgx<1)&(bgx > -2) )[0]
        nx = 3/0.01
        ny = 2.4/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline2_agn),np.log10(yline2_agn),'k--')#,label='AGN Line')
    plt.plot(np.log10(xline2_linersy2),np.log10(yline2_linersy2),c='k',ls='-.')#,label='LINER, Seyfert 2')

    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        cbar = plt.colorbar()

        cbar.set_label(ccodename,fontsize=20)
        cbar.ax.tick_params(labelsize=20)
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    plt.text(.1,-.5,'LINER',fontsize=15)
    plt.text(-1.0,1.0,'Seyfert',fontsize=15)
    plt.text(-1.0,-1,'HII',fontsize=15)

    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([SII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([-1.2,1.2])
    plt.xlim([-1.2,0.5])
    #plt.legend(fontsize=10,loc=4)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:

        fig.savefig('plots/xmm3/png/diagnostic/SII_OIII_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/SII_OIII_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/SII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()


'''
plotbpt_sii(xmm3eldiagmed_xrfilt.siiha,xmm3eldiagmed_xrfilt.oiiihb[xmm3eldiagmed_xrfilt.vo87_1_filt],
         nonagn_3xmmm_xrfiltvo87_1,  agn_3xmmm_xrfiltvo87_1,
         EL_m1.siiha,EL_m1.oiiihb[EL_m1.vo87_1_filt],save=False)
plotbpt_sii(xmm3eldiagmed_xrfilt.siiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,  agn_3xmmm_xrfilt,
         EL_m1.siiha,EL_m1.oiiihb,save=True, ccode=contaminations_xmm3_2, filename='xrfrac')

'''
def plotbpt_oi(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=False,filename='',title=None,alph=0.1,ccode=[],ccodegsw=[],cont=None,ccodename='X-ray Fraction',levs=[0,0.2]):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > -1.2) &( bgy < 1.2) & (bgx<0)&(bgx > -2.2) )[0]
        nx = 2.2/0.01
        ny = 2.4/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline3_linersy2),np.log10(yline3_linersy2),c='k',ls='-.')
    plt.plot(np.log10(xline3_agn), np.log10(yline3_agn),'k--')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        cbar = plt.colorbar()

        cbar.set_label(ccodename,fontsize=20)
        cbar.ax.tick_params(labelsize=20)
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    plt.text(-.5,-0.7,'LINER',fontsize=15)
    plt.text(-1.8,1.0,'Seyfert',fontsize=15)
    plt.text(-2,-1.1,'HII',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([OI]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([-1.2,1.2])
    plt.xlim([-2.2,0])
    #plt.legend(fontsize=15,frameon = False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:

        fig.savefig('plots/xmm3/png/diagnostic/OI_OIII_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OI_OIII_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OI_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()

'''
plotbpt_oi(xmm3eldiagmed_xrfilt.oiha,xmm3eldiagmed_xrfilt.oiiihb[xmm3eldiagmed_xrfilt.vo87_2_filt],
         nonagn_3xmmm_xrfiltvo87_2,  agn_3xmmm_xrfiltvo87_2,
         EL_m1.oiha,EL_m1.oiiihb[EL_m1.vo87_2_filt],save=True)
plotbpt_oi(xmm3eldiagmed_xrfilt.oiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt,
         EL_m1.oiha,EL_m1.oiiihb,save=True)
plotbpt_oi(xmm3eldiagmed_xrfilt.oiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt,
         EL_m1.oiha,EL_m1.oiiihb,save=True, filename='xrfrac', ccode=contaminations_xmm3_2)


'''

def plot_mex(xvals, yvals,nonagnfilt,agnfilt, bgx,bgy,save=True, filename='',title=None):
    #plt.scatter(np.log10(xvals3_bpt[valid_bpt][::50]), np.log10(yvals_bpt[valid_bpt][::50]),color='g',marker='.',alpha=0.15,label='SDSS DR7')
    plt.plot(xline_mex,ylineup_mex,'k--')
    plt.plot(xline_mex,ylinedown_mex,'k-.')
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')

    plt.scatter(bgx,np.log10(bgy),color='gray',marker='.',alpha=0.1,edgecolors='none',label='SDSS DR7')
    plt.scatter(xvals[nonagnfilt],np.log10(yvals[nonagnfilt]),
                 marker='^',color='b', label='X-Ray AGN (BPT-HII)')
    plt.scatter(xvals[agnfilt],np.log10(yvals[agnfilt]),
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)')
    plt.xlim([8,12])
    plt.ylim([-1.5,1.5])
    plt.legend(fontsize=10)
    plt.xlabel(r'log(M$_{*}$/M$_{\odot}$)',fontsize=20)
    plt.text(11.3,1.1,'AGN')
    #plt.text(-.1,-1.5,'Comp')
    plt.text(9.5,-1,'HII')
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    #plt.title('Mass-Excitation')
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()

    if save:
        plt.savefig('plots/xmm3/png/diagnostic/Mex_OIII'+filename+'.png',dpi=250)
        plt.savefig('plots/xmm3/pdf/diagnostic/Mex_OIII'+filename+'.pdf',dpi=250)
        plt.close()
    else:
        plt.show()
'''
plot_mex(xmm3eldiagmed_xrfilt.mass, xmm3eldiagmed_xrfilt.oiiihb,nonagn_3xmmm_xrfilt, 
         agn_3xmmm_xrfilt, EL_m1.mass, EL_m1.oiiihb)
'''


'''
Below are X-ray plots

'''


lsfrrelat = {'soft': [r'SFR/M$_{*} = 1.39\cdot 10^{-40}$ L$_{\rm x}/$M$_{*}$', r'SFR = $1.39\cdot 10^{-40}$ L$_{\rm x}$',logsfrsoft],
             'hard': [r'SFR/M$_{*} = 1.26\cdot 10^{-40} $L$_{\rm x}$/M$_{*}$', r'SFR = $1.26\cdot 10^{-40}$ L$_{\rm x}$',logsfrhard],
             'full': [r'SFR/M$_{*} = 0.66\cdot 10^{-40}$ L$_{\rm x}$/M$_{*}$',r'SFR = $0.66\cdot 10^{-40}$ L$_{\rm x}$', logsfrfull]  }

def plot_lxmsfrm(xraysfr, label, save=False, filtagn=[], filtnonagn=[],filename='',weakem=False,scat=False, nofilt=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    #lum/m  vs sfr/m
    #plt.title(r'Stripe 82X Galaxies, $z<0.3$, Soft X-Ray Luminosity')
    if len(filtagn) != 0 or len(filtnonagn) != 0:
        plt.scatter(xraysfr.lum_mass[make_m1[halp_filt_s82]][nonagn][filtnonagn],
                    xraysfr.sfr_mass[make_m1[halp_filt_s82]][nonagn][filtnonagn],
                    marker='^', color='b',label='BPT-HII')
        plt.scatter(xraysfr.lum_mass[make_m1[halp_filt_s82]][agn][filtagn],
                    xraysfr.sfr_mass[make_m1[halp_filt_s82]][agn][filtagn],
                    marker='o',facecolors='none',color='k',
                    label='BPT-AGN')
    elif nofilt:
        plt.scatter(xraysfr.lum_mass_val_filt, xraysfr.sfr_mass_val_filt, marker='x', color='k')
    else:
        plt.scatter(xraysfr.lum_mass_val_filtnoagn,xraysfr.sfr_mass_val_filtnoagn,
                    marker='^', color='b',label='BPT-HII',zorder=2)
        plt.scatter(xraysfr.lum_mass_val_filtagn,xraysfr.sfr_mass_val_filtagn,
                    marker='o',facecolors='none',color='k',label='BPT-AGN',zorder=1)
    if weakem:
        plt.scatter(weakem.lum_mass_val_filt, weakem.sfr_mass_val_filt, marker='x', color='gray', label=r'Weak Emission', zorder=0, alpha=0.8)
    #GMM
    '''
    gmm_lxm_sfr = mixture.GaussianMixture(n_components=2, covariance_type='tied')
    X_lxm_sfr = np.vstack([xraysfr.lum_mass_val_filtnoagn,xraysfr.sfr_mass_val_filtnoagn]).transpose()
    gmm_lxm_sfr.fit(X_lxm_sfr)
    sfrmass_grid =  np.linspace(np.min(xraysfr.sfr_mass),np.max(xraysfr.sfr_mass),50)
    lxmass_grid =  np.linspace(np.min(xraysfr.lum_mass),np.max(xraysfr.lum_mass),50)
    X_lxmass, Y_sfrmass = np.meshgrid(lxmass_grid,sfrmass_grid)
    XX_lxmass = np.array([X_lxmass.ravel(),Y_sfrmass.ravel()]).T
    Z_mass = -gmm_lxm_sfr.score_samples(XX_lxmass).reshape(X_lxmass.shape)
    CS = plt.contour(X_lxmass, Y_sfrmass, Z_mass, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
    '''
    avg_log_mass = np.mean( xraysfr.mass_val_filt)
    plt.plot(loglum_arr-avg_log_mass,lsfrrelat[label][2]-avg_log_mass,'k--',label=lsfrrelat[label][0],zorder=3)
    if scat:
        plt.fill_between(loglum_arr-avg_log_mass+scat,lsfrrelat[label][2]-avg_log_mass,lsfrrelat[label][2]-avg_log_mass-20, color='gray', zorder=4,alpha=0.2,linewidth=0)#,label=lsfrrelat[label][0])

    plt.xlabel(r'log(L$_{\rm x}$/M$_{\rm *}$)',fontsize=20)
    print('avg_log_mass',avg_log_mass)
    plt.ylabel(r'log(SFR/M$_{\rm *}$)',fontsize=20)
    plt.xlim([27,34])
    plt.ylim([-13.5,-6.5])
    plt.text(32.4,-12.1,'X-ray AGN', fontsize=15,rotation=45)

    plt.legend(frameon=False,fontsize=15,loc=2,bbox_to_anchor = (-0.02, 1.01))
    ax.set(adjustable='box-forced', aspect='equal')

    plt.tight_layout()
    if save:
        fig.savefig("plots/xmm3/png/xraylum/"+label +filename+ "_lxm_vs_sfrm.png",dpi=250)
        fig.savefig("plots/xmm3/pdf/xraylum/"+label + filename+"_lxm_vs_sfrm.pdf",dpi=250,format='pdf')
        fig.savefig("plots/xmm3/eps/xraylum/"+label + filename+"_lxm_vs_sfrm.eps",format='eps',dpi=250)
        plt.close(fig)
    else:
        plt.show()

def plot_lxsfr(xraysfr, label, save=False, filtagn=[], filtnonagn=[],filename='',weakem=False,scat=False, nofilt=False, fibssfr=[]):
    '''
    Plots star-formation rate versus X-ray luminosity and
    color-codes a region in bottom right where X-ray AGN are.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    if nofilt:
        plt.scatter(xraysfr.lum_val_filt, xraysfr.sfr_val_filt, marker='x', color='k')
    else:
        plt.scatter(xraysfr.lum_val_filtnoagn,xraysfr.sfr_val_filtnoagn,
                    marker='^', color='b',label='BPT-HII',zorder=3)
        plt.scatter(xraysfr.lum_val_filtagn,xraysfr.sfr_val_filtagn,
                    marker='o',facecolors='none',color='k',label='BPT-AGN',zorder=2)
    if weakem:
        plt.scatter(weakem.lum_val_filt,weakem.sfr_val_filt,marker='x',color='gray',label=r'Weak Emission',zorder=1,alpha=0.8)

    plt.plot(loglum_arr,lsfrrelat[label][2],'k--',label=lsfrrelat[label][1],zorder=3)
    if scat:
        plt.fill_between(loglum_arr+scat,lsfrrelat[label][2],y2=lsfrrelat[label][2]-20,color='gray', zorder=0,alpha=0.5,linewidth=0)
    plt.xlabel(r'log(L$_{\rm x}$)',fontsize=20)
    plt.ylabel(r'log(SFR)',fontsize=20)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.legend(frameon=False,fontsize=15,loc=2,bbox_to_anchor = (-0.00, 0.95))
    plt.text(42.8,-1.2,'X-ray AGN', fontsize=15,rotation=45)
    plt.xlim([37.5,44.5])
    plt.ylim([-2.5,4.5])
    ax.set(adjustable='box-forced', aspect='equal')
    plt.tight_layout() 
    if save:
        plt.savefig("plots/xmm3/png/xraylum/"+label+filename+"_lx_vs_sfr.png",dpi=250)
        fig.set_rasterized(True)

        plt.savefig("plots/xmm3/pdf/xraylum/"+label+filename+"_lx_vs_sfr.pdf",dpi=250)
        plt.savefig("plots/xmm3/eps/xraylum/"+label +filename+"_lx_vs_sfr.eps",format='eps',dpi=250)
        plt.close(fig)
    else:
        plt.show()
'''
plot_lxsfr(fullxray_xmm_dr7,'full',save=True, nofilt=True, filename='nofiltall_shade', scat = 0.6)
'''
def plot_lxfibsfr(xraysfr, label, save=False, filtagn=[], filtnonagn=[],filename='',weakem=False,scat=False, nofilt=False, fibssfr=[]):
        
    if len(fibssfr) !=0:
        fig = plt.figure()
        plt.ylim([-4,2.0])
        plt.xlim([38,44.5])
        if weakem:
            plt.scatter(weakem.lum_val_filt,EL_3xmm.weakfibssfr,marker='x',color='gray',label=r'Weak Emission',zorder=1,alpha=0.8)
            
        if nofilt:
            plt.scatter(xraysfr.lum_mass_val_filt, EL_3xmm.fibssfr, marker='x', color='k')
            plt.scatter(xraysfr.lum[make_m1_3xmm][EL_3xmm.not_halp_filt], EL_3xmm.weakfibssfr, marker='x', color='k')

        else:
            plt.scatter(xraysfr.lum_mass_val_filtnoagn_xrayagn,fibssfr[nonagn_3xmmm_xrfilt],
                    marker='^', color='b',label='BPT-HII',zorder=3)
            plt.scatter(xraysfr.lum_mass_val_filtagn_xrayagn, fibssfr[agn_3xmmm_xrfilt],
                    marker='o',facecolors='none',color='k',label='BPT-AGN',zorder=2)
        #plt.plot(loglum_arr,lsfrrelat[label][2],'k--',label=lsfrrelat[label][1],zorder=3)
        #if scat:
        #    plt.fill_between(loglum_arr+scat,lsfrrelat[label][2],y2=lsfrrelat[label][2]-20,color='gray', zorder=0,alpha=0.2,linewidth=0)
        plt.xlabel(r'log(L$_{\rm x}$)',fontsize=20)
        plt.ylabel(r'log(sSFR$_{fib}$)',fontsize=20)
        #plt.legend(frameon=False,fontsize=15,loc=2)
        plt.tight_layout()
        if save:
            plt.savefig("plots/xmm3/png/xraylum/"+label+filename+"_lx_vs_fibsfr.png",dpi=250)
            plt.savefig("plots/xmm3/pdf/xraylum/"+label+filename+"_lx_vs_fibsfr.pdf",dpi=250)
            plt.savefig("plots/xmm3/eps/xraylum/"+label +filename+"_lx_vs_fibsfr.eps",format='eps',dpi=250)
            plt.close(fig)
        else:
            plt.show()
            
        fig = plt.figure()
        
        
        '''
        Distance
        '''
        plt.ylim([-57,-52])
        plt.xlim([-14,-9])
        if weakem:
            plt.scatter(weakem.lum_val_filt-2*np.log10(EL_3xmm.weakdistances),EL_3xmm.weakfibsfr-2*np.log10(EL_3xmm.weakdistances),marker='x',color='gray',label=r'Weak Emission',zorder=1,alpha=0.8)
            
        if nofilt:
            plt.scatter(xraysfr.lum_val_filt-2*np.log10(EL_3xmm.distances), EL_3xmm.fibsfr-2*np.log10(EL_3xmm.distances), marker='x', color='k')
            plt.scatter(xraysfr.lum[make_m1_3xmm][EL_3xmm.not_halp_filt]-2*np.log10(EL_3xmm.weakdistances), EL_3xmm.weakfibsfr-2*np.log10(EL_3xmm.weakdistances), marker='x', color='k')

        else:
            plt.scatter(xraysfr.lum_val_filtnoagn_xrayagn-2*np.log10(xmm3eldiagmed_xrfilt.distances[nonagn_3xmmm_xrfilt]),fibssfr[nonagn_3xmmm_xrfilt]-2*np.log10(xmm3eldiagmed_xrfilt.distances[nonagn_3xmmm_xrfilt]),
                    marker='^', color='b',label='BPT-HII',zorder=3)
            plt.scatter(xraysfr.lum_val_filtagn_xrayagn-2*np.log10(xmm3eldiagmed_xrfilt.distances[agn_3xmmm_xrfilt]), fibssfr[agn_3xmmm_xrfilt]-2*np.log10(xmm3eldiagmed_xrfilt.distances[agn_3xmmm_xrfilt]),
                    marker='o',facecolors='none',color='k',label='BPT-AGN',zorder=2)
        if scat:
            plt.fill_between(loglum_arr+scat,lsfrrelat[label][2],y2=lsfrrelat[label][2]-20,color='gray', zorder=0,alpha=0.2,linewidth=0)
        #plt.plot(loglum_arr,lsfrrelat[label][2],'k--',label=lsfrrelat[label][1],zorder=3)

        plt.xlabel(r'log(L$_{\rm x}$)-2$\cdot$log(d)',fontsize=20)  
        plt.ylabel(r'log(SFR$_{fib}$)-2$\cdot$log(d)',fontsize=20)
        #plt.legend(frameon=False,fontsize=15,loc=2)
        plt.tight_layout()
        if save:
            plt.savefig("plots/xmm3/png/xraylum/"+label+filename+"_lxd_vs_fibsfrd.png",dpi=250)
            plt.savefig("plots/xmm3/pdf/xraylum/"+label+filename+"_lxd_vs_fibsfrd.pdf",dpi=250)
            plt.savefig("plots/xmm3/eps/xraylum/"+label +filename+"_lxd_vs_fibsfrd.eps",format='eps',dpi=250)
            plt.close(fig)
        else:
            plt.show()
'''
plot_lxsfr(fullxray_xmm_all,'full',save=False, scat=0.6)

plot_lxsfr(fullxray_xmm,'full',save=False, weakem = fullxray_xmm_no, filename='weakem_shade', scat=0.6)

plot_lxsfr(fullxray_xmm,'full',save=False, nofilt=True, filename='nofiltall_shade', scat = 0.6, fibsfr= xmm3eldiagmed_xrfilt.fibsfr)

plot_lxsfr(fullxray_xmm,'full',save=True, filename='weakem_shade', scat = 0.6, fibsfr= xmm3eldiagmed_xrfilt.fibsfr,weakem = fullxray_xmm_no)
plot_lxsfr(fullxray_xmm,'full',save=True, filename='shade', fibsfr= xmm3eldiagmed_xrfilt.fibsfr, scat=0.6)

plot_lxsfr(fullxray_xmm,'full',save=True, filename='weakem_shade', scat = 0.6, fibsfr= xmm3eldiagmed_xrfilt.fibsfrgsw,weakem = fullxray_xmm_no)
plot_lxfibsfr(fullxray_xmm,'full',save=False, filename='shade', fibssfr= xmm3eldiagmed_xrfilt.fibsfrgsw, scat=0.6)


plot_lxsfr(softxray_xmm,'soft',save=False)
plot_lxsfr(hardxray_xmm,'hard',save=False)

plot_lxsfr(fullxray_xmm,'full',save=False,scat=0.3)
plot_lxsfr(softxray_xmm,'soft',save=False,scat=0.3)
plot_lxsfr(hardxray_xmm,'hard',save=False,scat=0.3)

plot_lxsfr(fullxray_xmm,'full',save=False,scat=0.6)
plot_lxsfr(softxray_xmm,'soft',save=False,scat=0.6)
plot_lxsfr(hardxray_xmm,'hard',save=False,scat=0.6)


plot_lxsfr(fullxray_xmm,'full',scat=0.6, save=True)

plot_lxsfr(softxray_xmm,'soft',save=False,scat=0.6, nofilt=True)
plot_lxsfr(hardxray_xmm,'hard',save=False,scat=0.6, nofilt=True)


plot_lxsfr(fullxray_xmm,'full',save=True)
plot_lxsfr(softxray_xmm,'soft',save=True)
plot_lxsfr(hardxray_xmm,'hard',save=True)
'''
def plotlx_z(xraysfr, save=False,filename=''):
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.scatter(xraysfr.z_filt, xraysfr.lum_filt,color='black', marker='x')
    plt.xlabel('z')
    plt.xlim([0,0.3])
    plt.ylim([39,46])
    plt.ylabel(r'L$_{\mathrm{X}}$')
    plt.tight_layout()
    if save:
       plt.savefig("plots/xmm3/png/xraylum/"+filename+"_lxz.png",dpi=250)
       plt.savefig("plots/xmm3/pdf/xraylum/"+filename+"_lxz.pdf",dpi=250)
       plt.savefig("plots/xmm3/eps/xraylum/"+filename+"_lxz.eps",format='eps',dpi=250)
       plt.close()
    else:
       plt.show()    
'''
plotlx_z(fullxray_xmm_dr7, save=True, filename='full')

'''
def plothardfull(hard,full,save=False):
    plt.scatter(hard.lum,full.lum,marker='o')
    plt.ylabel(r'L$_{0.5-10\ keV }$')
    plt.xlabel(r'L$_{2-10\ keV}$')
    #plt.legend()
    plt.ylim([40.5,43.5])
    plt.xlim([40.5,43.5])
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    if save:
        plt.savefig("plots/xmm3/png/xraylum/hard_vs_full.png",dpi=250)
        plt.savefig("plots/xmm3/pdf/xraylum/hard_vs_full.pdf",dpi=250,format='pdf')
        plt.savefig("plots/xmm3/eps/xraylum/hard_vs_full.eps",dpi=250,format='eps')

#plothardfull(hardxray,fullxray)
#plothardfull(hardxray,fullxray_no)
sdssquery = np.loadtxt('sdssquery_out.csv', delimiter=',', unpack=True, skiprows=1)
ra, dec =sdssquery[0], sdssquery[1]
hbetafwhm = sdssquery[2]* 2*np.sqrt(2*np.log(2))
oiiifwhm = sdssquery[4] * 2*np.sqrt(2*np.log(2))

def plotfwhmbalmerforbid(eldiag,gswbalm, gswforb, agnfilt,nonagnfilt,save=False, filename=''):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    realvalxragn = np.where((eldiag.balmerfwhm[agnfilt] <1170)& (eldiag.forbiddenfwhm[agnfilt] <1170) &(eldiag.balmerfwhm[agnfilt]>2.4) &(eldiag.forbiddenfwhm[agnfilt]>2.4))[0]
    
    realvalxrhii = np.where((eldiag.balmerfwhm[nonagnfilt] <1170)& (eldiag.forbiddenfwhm[nonagnfilt] <1170) &(eldiag.balmerfwhm[nonagnfilt]>2.4) &(eldiag.forbiddenfwhm[nonagnfilt]>2.4))[0]
    #[realvalxragn][realvalxragn]
    plt.scatter(eldiag.balmerfwhm[nonagnfilt],eldiag.forbiddenfwhm[nonagnfilt],marker='^',color='b',label='BPT-HII', zorder=11)

    plt.scatter(eldiag.balmerfwhm[agnfilt],eldiag.forbiddenfwhm[agnfilt],marker='o',label='BPT-AGN',color='k',facecolor='none', zorder=10)
    #[realvalxrhii][realvalxrhii]
    realvalgsw = np.where((gswbalm <1170)& (gswforb <1170) &(gswbalm>2.4) &(gswforb>2.4))[0]
    
    yarr = np.arange(0,1200)
    #plt.plot(yarr*1.1, yarr,)
    #plt.plot(yarr*1.2, yarr,)

    #plt.plot(yarr*1.3, yarr,'k-.',label=r'30\%')
    #[realvalgsw][realvalgsw]
    plot2dhist(gswbalm, gswforb, 200,200)
    plt.xlabel(r'FWHM$_{\mathrm{Balmer}} (\mathrm{km\ s}^{-1}$)')
    plt.ylabel(r'FWHM$_{\mathrm{Forbidden}}(\mathrm{km\ s}^{-1})$')
    #plt.legend()
    #plt.ylim([40.5,43.5])
    #plt.xlim([40.5,43.5])
    plt.xlim([0,1100])
    plt.ylim([0,1100])
    ax.set(adjustable='box-forced', aspect='equal')
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.legend(frameon=False,fontsize=15,loc=2,bbox_to_anchor=(0, 0.95))
    plt.tight_layout()
    if save:
        plt.savefig("plots/xmm3/png/diagnostic/fwhm_balmer_forbid"+filename+".png",dpi=250)
        plt.savefig("plots/xmm3/pdf/diagnostic/fwhm_balmer_forbid"+filename+".pdf",dpi=250,format='pdf')
        plt.savefig("plots/xmm3/eps/diagnostic/fwhm_balmer_forbid"+filename+".eps",dpi=250,format='eps')
        
'''
plotfwhmbalmerforbid(xmm3eldiagmed_xrfilt, EL_m1.balmerfwhm, EL_m1.forbiddenfwhm,
                     agn_3xmmm_xrfilt,
                     nonagn_3xmmm_xrfilt, save=True)
plotfwhmbalmerforbid(EL_qsos, EL_m1.balmerfwhm, EL_m1.forbiddenfwhm,
                     agn_qsos,
                     nonagn_qsos, save=False)
plotfwhmbalmerforbid(xmm3eldiagmed_xrfilt, EL_m1.balmerfwhm, EL_m1.forbiddenfwhm,
                     agn_3xmmm_xrfilt,
                     nonagn_3xmmm_xrfilt, save=True, filename='30')

realvalxragn = np.where((eldiag.balmerfwhm[agnfilt] <1170)& (eldiag.forbiddenfwhm[agnfilt] <1170) &(eldiag.balmerfwhm[agnfilt]>2.4) &(eldiag.forbiddenfwhm[agnfilt]>2.4))[0]
badfwhm = np.where((xmm3eldiagmed_xrfilt.balmerfwhm[nonagn_3xmmm_xrfilt] == 1177.41003418 ) |
        (xmm3eldiagmed_xrfilt.balmerfwhm[nonagn_3xmmm_xrfilt] ==2.35482001)|
        (xmm3eldiagmed_xrfilt.forbiddenfwhm[nonagn_3xmmm_xrfilt] == 1177.41003418)|
        (xmm3eldiagmed_xrfilt.forbiddenfwhm[nonagn_3xmmm_xrfilt] == 2.35482001))[0]    
    
'''




'''
def mass_bins(el_obj,start=9, step=0.5,frac=''):

    for i in range(len(mass_bins_sdss)):
        nonagn, agn = el_obj.get_bpt1_groups()
        fnam = 'massbin'+frac+str(round(i*step+start,2))+'-'+str(round(i*step+start+step,2))
        titl = str(round(i*step+start,2))+r'$<$ M $<$'+str(round(i*step+start+step,2))
        plotbpt(el_obj.niiha, el_obj.oiiihb, nonagn, agn, gsw_xvals1_bpt[valid_bpt2][mass_bins_sdss[i]],
                    gsw_yvals_bpt[valid_bpt2][mass_bins_sdss[i]],
                    filename=fnam,title=titl)
        
def z_bins(z_bins_sdss,z_bins_s82,start=0, step=.03):
    zbins = np.arange(0,0.3+step,step)
    inds = []
    for i in range(len(zbins)-1):
        bin_inds = np.where((z>zbins[i]) & (z <zbins[i+1]))[0]
        inds.append(bin_inds)
    for i in range(len(z_bins_sdss)):
        nonagn, agn = el_obj.get_bpt1_groups(),
                                      stripe_82y[halp_filt_s82][z_bins_s82[i]])

        fnam = 'zbin'+str(round(i*step+start,2))+'-'+str(round(i*step+start+step,2))
        titl = str(round(i*step+start,2))+r'$<$ z $<$'+str(round(i*step+start+step,2))

        plotbpt(EL_m2[halp_filt_s82][z_bins_s82[i]],
                    stripe_82y[halp_filt_s82][z_bins_s82[i]],
                    nonagn, agn,gsw_xvals1_bpt[valid_bpt][z_bins_sdss[i]],
                    gsw_yvals_bpt[valid_bpt][z_bins_sdss[i]],
                    filename=fnam,
                    title=titl)
#Bad versions ^^^
       



mass_bins_sdss  = get_mass_bins(all_sdss_avgmasses[spec_inds_allm1][valid_bpt2],step=0.5)
z_bins_sdss = get_z_bins(all_sdss_spec_z[spec_inds_allm1][valid_bpt2],step=0.05)
z_bins_s82 = get_z_bins(m1_z[halp_filt_s82],step=0.05)
EL_m2.neither_filt,
massfrac_bins_sdss  =get_mass_bins(all_sdss_massfrac[spec_inds_allm1][valid_bpt2],start =0, stop = 1,step=0.1)
'''

def mass_bins_plus(el_obj,start=9, step=0.5, stop=12):
    massbins = np.arange(start,stop+step,step)
    for i in range(len(massbins)-1):
        bin_inds = np.where((el_obj.mass >massbins[i]) &(el_obj.mass <massbins[i+1]) )[0]
        groups, nonagn, agn = el_obj.get_bpt1_groups(filt=el_obj.bpt_sn_filt[bin_inds])
        bin_inds_plus = np.where((el_obj.mass_plus> massbins[i]) &(el_obj.mass_plus<massbins[i+1]))[0]
        bin_inds_neit = np.where((el_obj.mass_neither > massbins[i]) &(el_obj.mass_neither < massbins[i+1]))[0]
        fnam = 'massbin'+str(round(i*step+start,2))+'-'+str(round(i*step+start+step,2))
        titl = str(round(i*step+start,2))+r'$<$ log(M$_{*}$) $<$'+str(round(i*step+start+step,2))
        plotbptplus(el_obj.niiha[bin_inds], el_obj.oiiihb[bin_inds], el_obj.niihaplus[bin_inds_plus], el_obj.neither_filt[bin_inds_neit], nonagn= nonagn, agn=agn, filename=fnam, labels=False, save=True)
'''
mass_bins_plus(EL_m2)
'''        
def massfrac_bins_plus(el_obj,start=0.0, step=0.1, stop=0.6):
    massfracbins = np.arange(start,stop+step,step)
    for i in range(len(massfracbins)-1):
        bin_inds = np.where((el_obj.massfracgsw >massfracbins[i]) &(el_obj.massfracgsw <massfracbins[i+1]) )[0]
        groups, nonagn, agn = el_obj.get_bpt1_groups(filt=el_obj.bpt_sn_filt[bin_inds])
        bin_inds_plus = np.where((el_obj.massfracgsw_plus> massfracbins[i]) &(el_obj.massfracgsw_plus<massfracbins[i+1]))[0]
        bin_inds_neit = np.where((el_obj.massfracgsw_neither > massfracbins[i]) &(el_obj.massfracgsw_neither < massfracbins[i+1]))[0]
        fnam = 'massfracbin'+str(round(i*step+start,2))+'-'+str(round(i*step+start+step,2))

        titl = str(round(i*step+start,2))+r'$<$ M_{frac} $<$'+str(round(i*step+start+step,2))
        
        plotbptplus(el_obj.niiha[bin_inds], el_obj.oiiihb[bin_inds], el_obj.niihaplus[bin_inds_plus], el_obj.neither_filt[bin_inds_neit], nonagn= nonagn, agn=agn, filename=fnam, labels=False, save=True)
'''
massfrac_bins_plus(EL_m2)
'''        
def z_bins_plus(el_obj,start=0,stop=0.3, step=.05):
    zbins = np.arange(start,stop+step,step)
    for i in range(len(zbins)-1):
        bin_inds = np.where((el_obj.z > zbins[i]) &(el_obj.z < zbins[i+1]) )[0]
        groups, nonagn, agn = el_obj.get_bpt1_groups(filt=el_obj.bpt_sn_filt[bin_inds])
        bin_inds_plus = np.where((el_obj.z_plus> zbins[i]) &(el_obj.z_plus< zbins[i+1]))[0]
        bin_inds_neit = np.where((el_obj.z_neither > zbins[i]) &(el_obj.z_neither < zbins[i+1]))[0]
        fnam = 'zbin'+str(round(i*step+start,2))+'-'+str(round(i*step+start+step,2))
        titl = str(round(i*step+start,2))+r'$<$ z $<$'+str(round(i*step+start+step,2))
        plotbptplus(el_obj.niiha[bin_inds], el_obj.oiiihb[bin_inds], el_obj.niihaplus[bin_inds_plus], el_obj.neither_filt[bin_inds_neit], nonagn= nonagn, agn=agn, filename=fnam, labels=False, save=True)
'''
z_bins_plus(EL_m2)
'''