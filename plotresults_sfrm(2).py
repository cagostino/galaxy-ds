#from matchgal_gsw2 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib
from matplotlib.colors import LogNorm
#from sklearn import mixture
from ast_func import *
import scipy
from sklearn.linear_model import LinearRegression
from xraysfr_obj import *
from demarcations import *
#import os
#os.environ['PATH']+=':~/texlive'
#from mpl_toolkits.basemap import Basemap
mydpi = 96
plt.rc('font',family='serif')
plt.rc('text',usetex=True)

class Plot:
    def __init__(self, name, minx=0, maxx=0, miny=0, maxy=0, 
                 xlabel='', ylabel='', nx=500, ny=500):
        self.name = name
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.nx = nx
        self.ny = ny
    def make2ddensplot(self,x,y, ccode='', ccodename='', xlabel='', ylabel=''):
        if len(xlabel)==0:
            xlabel=self.xlabel
        if len(ylabel)==0:
            ylabel=self.ylabel
        plot2dhist(x,y, self.nx, self.ny, minx=self.minx, maxx=self.maxx,
                   miny=self.miny, maxy=self.maxy, xlabel=xlabel, 
                   ylabel=ylabel, ccode=ccode, ccodename=ccodename)
          
class Hist:
    def __init__(self, name, minx=0, maxx=0, xlabel='', ylabel='', nbins=20):
        self.name = name
        self.minx=minx
        self.maxx=maxx
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.nbins=nbins

def reject_outliers(data, m=2):
    return np.array([abs(data - np.mean(data)) < m * np.std(data)])[0]

def plthist(bincenters, counts):
    plt.plot(bincenters, counts,color='k', drawstyle='steps-mid')

def plot2dhist(x,y,nx,ny, ccode= [], nan=False, data=True, fig=None, ax=None,
               bin_y=False, bin_stat_y = 'mean',  ybincolsty='r-',size_y_bin=0.05, 
               ccodename = '', ccodelim=[], linewid=2, label='', zorder=10,
               minx=0, maxx=0, miny=0, maxy=0, xlabel='', ylabel=''):
    if maxx == 0:
        minx = np.sort(x)[int(0.01*len(x))]
        maxx = np.sort(x)[int(0.99*len(x))]
        miny = np.sort(y)[int(0.01*len(y))]
        maxy = np.sort(y)[int(0.99*len(y))]        
    
    if ax and not fig:
        fig=ax
    #valid = np.where((x>minx) &(x<maxx)&(y>miny)&(y<maxy))[0]
    #x=np.copy(x[valid])
    #y=np.copy(y[valid])    
    if nan:
        fin = np.where((np.isfinite(x)) &(np.isfinite(y) ))[0]
        x = np.copy(x[fin])
        y =np.copy(y[fin])
    hist, xedges, yedges = np.histogram2d(x,y,bins = (int(nx),int(ny)))
    extent= [np.min(xedges),np.max(xedges),np.min(yedges),np.max(yedges)]
    plt.minorticks_on()

    if len(ccode)==0:        
        if data:
            if ax:
                ax.imshow((hist.transpose())**0.3, cmap='gray_r',extent=extent,origin='lower',
                          aspect='auto',alpha=0.9) 
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                
            else:
                plt.imshow((hist.transpose())**0.3, cmap='gray_r',extent=extent,origin='lower',
                   aspect='auto',alpha=0.9) 
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
        if bin_y:
            nbins = int((np.max(x)-np.min(x))/(size_y_bin))
            avg_y, xedges, binnum = scipy.stats.binned_statistic(x,y, statistic=bin_stat_y, bins = nbins,range=(np.min(x), np.max(x)))
            count_y, xedges, binnum = scipy.stats.binned_statistic(x,y,statistic='count', bins = nbins,range=(np.min(x), np.max(x)))
            good_y = np.where(count_y>5)
            xmid = (xedges[1:]+xedges[:-1])/2
            if ax:
                ax.plot(xmid[good_y], avg_y[good_y], ybincolsty, linewidth=linewid, label=label, zorder=zorder)
            else:
                plt.plot(xmid[good_y], avg_y[good_y],ybincolsty, linewidth=linewid, label=label, zorder=zorder)
            if ax: 
                return fig, ax, xmid, avg_y
            else:
                return xmid, avg_y
    else:
        ccode_avgs = np.zeros_like(hist)
        for i in range(len(xedges)-1):
            for j in range(len(yedges)-1):
                val_rang = np.where( (x>=xedges[i]) &(x<xedges[i+1]) &
                                     (y>=yedges[j]) & (y<yedges[j+1]))[0]
                if val_rang.size >= 20:
                    ccode_avgs[i,j] = np.nanmedian(ccode[val_rang])                    
                else:
                    ccode_avgs[i,j]= np.nan

        if ax:
            im = ax.imshow((ccode_avgs.transpose()), cmap='Greens',extent=extent,origin='lower',
                   aspect='auto',alpha=0.9) 
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if len(ccodelim) !=0:
                if len(ccodelim) ==2:
                    mn, mx = ccodelim
                else:
                    mn, mx = np.nanmin(ccode_avgs), np.nanmax(ccode_avgs)       
                fig.clim(mn, mx)
            #pcm = im.get_children()[2]
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(ccodename, fontsize=20)
            cbar.ax.tick_params(labelsize=20)
            return fig, ax, ccode_avgs
        else:
            plt.imshow((ccode_avgs.transpose()), cmap='Greens',extent=extent,origin='lower',
                   aspect='auto',alpha=0.9)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if len(ccodelim) !=0:
                if len(ccodelim) ==2:
                    mn, mx = ccodelim
                else:
                    mn, mx = np.nanmin(ccode_avgs), np.nanmax(ccode_avgs)       
                
                plt.clim(mn, mx)
            cbar = plt.colorbar()
            cbar.set_label(ccodename, fontsize=20)
            cbar.ax.tick_params(labelsize=20)
            return ccode_avgs
def plot3d(x,y,z):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, s=0.1, alpha=0.1)
    plt.show()
    return fig

def plot3panel(x1,x2,x3,y1,y2,y3, ccode1=[], ccode2=[], ccode3=[], save=False,
                   nobj=False,filename='',minx=0, maxx=0, miny=0, maxy=0,
                   bin_y=False, bin_stat_y = 'mean',  ybincolsty='r-',size_y_bin=0.05,
                   nx=300, ny=240, ccodename='', xlabel='', ylabel='', aspect='equal', nan=True):
    '''
    Makes a vertical 3  panel plot showing three different attributes
    '''
    fig = plt.figure(figsize=(8,8))
    allx = np.concatenate((x1,x2,x3))
    ally = np.concatenate((y1,y2,y3))
    if maxx == 0:
        minx = np.sort(allx)[int(0.01*len(allx))]
        maxx = np.sort(allx)[int(0.99*len(allx))]
        miny = np.sort(ally)[int(0.01*len(ally))]
        maxy = np.sort(ally)[int(0.99*len(ally))]            
    ax1 = fig.add_subplot(311)
    ax1.set_xlim([minx,maxx])
    ax1.set_ylim([miny,maxy])
    ax1.set(aspect='equal', adjustable='box')

    ax2 = fig.add_subplot(312, sharey = ax1, sharex=ax1)
    #ax1.set_xticklabels([''])
    #ax2.set_xticklabels([''])
    ax3 = fig.add_subplot(313, sharey = ax1, sharex=ax1)

    plot2dhist(x1, y1, nx, ny, ccode=ccode1, ax=ax1,
               fig=fig, ccodename=ccodename, 
               minx=minx, maxx=maxx, miny=miny, maxy=maxy,
               bin_y=bin_y, bin_stat_y = bin_stat_y,  ybincolsty=ybincolsty,size_y_bin=size_y_bin,nan=nan)
    plot2dhist(x2, y2, nx, ny, ccode=ccode2, ax=ax2,
               fig=fig, ccodename=ccodename, 
               minx=minx, maxx=maxx, miny=miny, maxy=maxy,
               bin_y=bin_y, bin_stat_y = bin_stat_y,  ybincolsty=ybincolsty,size_y_bin=size_y_bin,nan=nan)
    plot2dhist(x3, y3, nx, ny, ccode=ccode3, ax=ax3,
               fig=fig, ccodename=ccodename, 
               minx=minx, maxx=maxx, miny=miny, maxy=maxy,
               bin_y=bin_y, bin_stat_y = bin_stat_y,  ybincolsty=ybincolsty,size_y_bin=size_y_bin,nan=nan)    
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel(ylabel)
    ax3.set_ylabel(ylabel)
    plt.tight_layout()
    ax1.set(aspect=aspect, adjustable='box')
    ax2.set(aspect=aspect, adjustable='box')
    ax3.set(aspect=aspect, adjustable='box')

    plt.subplots_adjust(wspace=0, hspace=0)

    if save:
        fig.savefig('plots/sfrmatch/pdf/'+filename+'.pdf', bbox_inches='tight', dpi=250, format='pdf')
        fig.savefig('plots/sfrmatch/png/'+filename+'.png', bbox_inches='tight', dpi=250)
        plt.close(fig)
    return fig, ax1, ax2, ax3
def linregress_test(offset_x, offset_y, mass, sfr, av, bptx, bpty):
    X = np.vstack([mass, sfr, av, bptx, bpty])
    y = np.sqrt(offset_x**2+offset_y**2)
    reg = LinearRegression().fit(X, y)    
    
'''
linregress_test(sfrm_gsw2.offset_nii_halpha, sfrm_gsw2.offset_oiii_hbeta, sfrm_gsw2.mass_bpt, sfrm_gsw2.sfr_bpt, sfrm_gsw2.av_bpt_agn, sfrm_Gsw2.
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
    plt.tight_layout()
    if save:
        fig1.savefig('plots/sfrmatch/png/distmet/avg_sample_dists_'+ str(samp_num) +'.png', dpi=250,bbox_inches='tight')
        fig1.savefig('plots/sfrmatch/pdf/distmet/avg_sample_dists_'+ str(samp_num) +'.pdf', dpi=250,bbox_inches='tight')
        fig1.set_rasterized(True)
        fig1.savefig('plots/sfrmatch/eps/avg_sample_dists_'+ str(samp_num) +'.eps', dpi=150,format='eps')
        plt.close(fig1)
    fig2 = plt.figure()
    plt.scatter(np.log10(n_vals_used), sampl,marker='x', color='k')
    plt.xlabel(r'log(n$_{\mathrm{Avg.}}$)', fontsize=20)
    plt.ylabel('Avg. Sample Distance', fontsize=20)
    plt.tight_layout()
    if save:
        fig2.savefig('plots/sfrmatch/png/distmet/logavg_sample_dists_'+ str(samp_num) +'.png', dpi=250,bbox_inches='tight')
        fig2.savefig('plots/sfrmatch/pdf/distmet/logavg_sample_dists_'+ str(samp_num) +'.pdf', dpi=250,bbox_inches='tight')
        fig2.set_rasterized(True)
        fig2.savefig('plots/sfrmatch/eps/logavg_sample_dists_'+ str(samp_num) +'.eps', dpi=150,format='eps')
        plt.close(fig2)
def get_avg_dist(dists, save=False, filename=''):
    fig1 = plt.figure()
    plt.scatter(n_vals_used, np.mean(dists, axis=1),marker='x', color='k')    
    plt.xlabel(r'n$_{\mathrm{Avg.}}$', fontsize=20)
    plt.ylabel('Avg. Distance', fontsize=20)
    plt.tight_layout()
    if save:
        fig1.savefig('plots/sfrmatch/png/distmet/avg_dists.png', dpi=250,bbox_inches='tight')
        fig1.savefig('plots/sfrmatch/pdf/distmet/avg_dists.pdf', dpi=250,bbox_inches='tight')
        fig1.set_rasterized(True)
        fig1.savefig('plots/sfrmatch/eps/avg_dists.eps', dpi=150,format='eps')
        plt.close(fig1)
    fig2 = plt.figure()
    plt.scatter(np.log10(n_vals_used), np.mean(dists, axis=1),marker='x', color='k')
    plt.xlabel(r'log(n$_{\mathrm{Avg.}}$)', fontsize=20)
    plt.ylabel('Avg. Distance', fontsize=20)
    plt.tight_layout()    
    if save:
        fig2.savefig('plots/sfrmatch/png/distmet/logavg_dists.png', dpi=250,bbox_inches='tight')
        fig2.savefig('plots/sfrmatch/pdf/distmet/logavg_dists.pdf', dpi=250,bbox_inches='tight')
        fig2.set_rasterized(True)
        fig2.savefig('plots/sfrmatch/eps/logavg_dists.eps', dpi=150,format='eps')
        plt.close(fig2)
def get_med_dist(dists, save=False, filename=''):
    fig1 = plt.figure()
    plt.scatter(n_vals_used, np.median(dists, axis=1),marker='x', color='k')    
    plt.xlabel(r'n$_{\mathrm{Avg.}}$', fontsize=20)
    plt.ylabel('Median Distance', fontsize=20)
    plt.tight_layout()
    if save:
        fig1.savefig('plots/sfrmatch/png/distmet/med_dists.png', dpi=250,bbox_inches='tight')
        fig1.savefig('plots/sfrmatch/pdf/distmet/med_dists.pdf', dpi=250,bbox_inches='tight')
        fig1.set_rasterized(True)
        fig1.savefig('plots/sfrmatch/eps/distmet/med_dists.eps', dpi=150,format='eps')
        plt.close(fig1)
    fig2 = plt.figure()
    plt.scatter(np.log10(n_vals_used), np.median(dists, axis=1),marker='x', color='k')
    plt.xlabel(r'log(n$_{\mathrm{Avg.}}$)', fontsize=20)
    plt.ylabel('Median Distance', fontsize=20)
    plt.tight_layout()    
    if save:
        fig2.savefig('plots/sfrmatch/png/distmet/logmed_dists.png', dpi=250,bbox_inches='tight')
        fig2.savefig('plots/sfrmatch/pdf/distmet/logmed_dists.pdf', dpi=250,bbox_inches='tight')
        fig2.set_rasterized(True)
        fig2.savefig('plots/sfrmatch/eps/distmet/logmed_dists.eps', dpi=150,format='eps')
        plt.close(fig2)        


def plotnclass(n_class,matchtyp, save=False, filename=''):
    fig1 = plt.figure()
    
    plt.scatter(n_vals_used, n_class,marker='x', color='k')    
    plt.xlabel(r'n$_{\mathrm{Avg.}}$', fontsize=20)
    plt.ylabel(r'n$_{\mathrm{Classifiable}}$, '+matchtyp, fontsize=20)
    plt.tight_layout()
    if save:
        fig1.savefig('plots/sfrmatch/png/distmet/n_classifiable_'+filename+'.png', dpi=250,bbox_inches='tight')
        fig1.savefig('plots/sfrmatch/pdf/distmet/n_classifiable_'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig1.set_rasterized(True)
        fig1.savefig('plots/sfrmatch/eps/distmet/n_classifiable_'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig1)
    fig2 = plt.figure()
    plt.scatter(np.log10(n_vals_used), n_class,marker='x', color='k')
    plt.xlabel(r'log(n$_{\mathrm{Avg.}}$)', fontsize=20)
    plt.ylabel(r'n$_{\mathrm{Classifiable}}$, '+matchtyp, fontsize=20)
    plt.tight_layout()    
    if save:
        fig2.savefig('plots/sfrmatch/png/distmet/logn_classifiable_'+filename+'.png', dpi=250,bbox_inches='tight')
        fig2.savefig('plots/sfrmatch/pdf/distmet/logn_classifiable_'+filename+'.pdf', dpi=250,bbox_inches='tight')
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


plt.hist(-sfrm_gsw2.fibsfr_match_sing+sfrm_gsw2.fibsfr_sing, bins=100, histtype='step', range=(-5,5), label='GSW Fib. SFR.')
plt.hist(-sfrm_gsw2.fibsfr_mpa_match_sing+sfrm_gsw2.fibsfr_mpa_sing, bins=100, histtype='step', range=(-5,5), label='MPA/JHU Fib. SFR')
plt.ylabel('Count')
plt.xlabel('AGN Fiber SFR-Match Fiber SFR')
plt.legend()
plt.tight_layout()


plt.hist(-sfrm_gsw2.ssfr_match_sing+sfrm_gsw2.ssfr_sing, bins=100, histtype='step', range=(-5,5), label='GSW sSFR.')
plt.hist(-sfrm_gsw2.fibssfr_mpa_match_sing+sfrm_gsw2.fibssfr_mpa_sing, bins=100, histtype='step', range=(-5,5), label='MPA/JHU Fib. sSFR')
plt.ylabel('Count')
plt.xlabel('AGN sSFR-Match sSFR')
plt.legend()
plt.tight_layout()
'''

  
#%%        disps
disp_sfr = Plot('disp_sfr',minx=-3.5, maxx=3.5, miny=0, maxy=1.5, xlabel='log(SFR)')
disp_ssfr = Plot('disp_sfr')
disp_mass = Plot('disp_mass')
disp_fibmass = Plot('disp_fibmass')
disp_av = Plot('disp_av')

dispy_sfr = Plot('dispy_sfr')
dispy_ssfr = Plot('dispy_ssfr')
dispy_mass = Plot('dispy_mass')
dispy_fibmass = Plot('dispy_fibmass')

dispx_sfr = Plot('dispx_sfr')
dispx_ssfr = Plot('dispx_ssfr')
dispx_mass = Plot('dispx_mass')
dispx_fibmass = Plot('dispx_fibmass')
dist_disp = Plot('dist_disp')
'''    plt.xlabel('Displacements', fontsize=20)
    plt.ylabel('Match Distances', fontsize=20)
    plt.xlim([-0.1, 1.5])
    plt.ylim([-0.1, 0.5])
'''
#%% properties
fwhms = Plot('FWHMs')
'''
    plt.xlabel(r'FWHM$_{\mathrm{Balmer}} (\mathrm{km\ s}^{-1}$)')
    plt.ylabel(r'FWHM$_{\mathrm{Forbidden}}(\mathrm{km\ s}^{-1})$')
    #plt.legend()
    #plt.ylim([40.5,43.5])
    #plt.xlim([40.5,43.5])
    plt.xlim([0,1100])
    plt.ylim([0,1100])
'''
fluxcomp = Plot('fluxcomp')
'''
    plt.xlabel(line+' Flux, AGN', fontsize=20)
    plt.ylabel(line+' Flux, ' + ' Match', fontsize=20)
'''
d4000_ssfr = Plot('d4000_ssfr')
mass_z = Plot('mass_z')
oiiidisp_comb =Plot('oiiidisp_comb')
'''
    #        marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    plt.ylabel(r'log(L$_\mathrm{[OIII]}/\sigma^{4}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([7.5,15.5])
    plt.xlim([8,12.5])
'''
oiiimass_comb =Plot('oiiidisp_comb')
'''
    plt.ylabel(r'log(L$_\mathrm{[OIII]}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([36,44])
    plt.xlim([9,12])
'''
oiiimass_edd =Plot('oiiidisp_comb')
'''
    plt.ylabel(r'log(L$_\mathrm{[OIII]}$/M$_{\mathrm{*}}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    #plt.ylim([37,43])
    #plt.xlim([8,12])
'''
ssfrm =  Plot('ssfrm', minx=7.5, maxx=12.5, miny=-14, maxy=-7.5,xlabel=r'log(M$_{\mathrm{*}})$',ylabel=r'log(sSFR)', nx=500,ny=600)
U_OH  = Plot('U_OH')
'''
    plt.ylabel(r'log([OIII]/[OII])',fontsize=20)
    plt.xlabel(r'log([NII]/[OII])',fontsize=20)
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
'''
sfrcomp = Plot('sfrcomp')
'''
    plt.ylabel(r'log(SFR$_{\rm Match}$)',fontsize=20)
    plt.xlabel(r'log(SFR$_{\rm AGN}$)',fontsize=20)
'''
avcomp = Plot('avcomp')
'''
    plt.ylabel(r'log(A$_{\rm V, GSW, Match}$)',fontsize=20)
    plt.xlabel(r'log(A$_{\rm V, GSW, AGN}$)',fontsize=20)
'''
fibmasscomp = Plot('fibmasscomp')
'''
    plt.ylabel(r'log(M$_{\rm *, Fib., Match}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\rm *, Fib., AGN}$)',fontsize=20)
'''
masscomp = Plot('masscomp')
'''
    plt.ylabel(r'log(M$_{\rm *, Match}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\rm *, AGN}$)',fontsize=20)
'''
zcomp = Plot('zcomp')
'''
    plt.ylabel(r'z$_{\rm Match}$',fontsize=20)
    plt.xlabel(r'z$_{\rm AGN}$',fontsize=20)
'''
#%% dust
dust_oiii = Plot('dustoiii')
'''
    plt.ylim([-6, 6])
    plt.xlim([36, 44])
    plt.plot([36,44],[-6,6], c='k',ls='-.')
    plt.ylabel(r'A$_{\rm V, AGN-match}$',fontsize=20)
    plt.xlabel(r'L$_{\mathrm{[OIII]}}$',fontsize=20)
'''
dustcomp_sf = Plot('dustcomp_sf')
'''
    plt.ylim([-6, 6])
    plt.xlim([-6, 6])
    plt.plot([-6,6],[-6,6], c='k',ls='-.')
    plt.ylabel(r'A$_{\rm V, AGN-match}$',fontsize=20)
    plt.xlabel(r'A$_{\rm V, match}$',fontsize=20)

'''
dustcomp_agn = Plot('dustcomp_agn')
'''
    plt.ylim([-6, 6])
    plt.xlim([-6, 6])
    plt.xlabel(r'A$_{\rm V, AGN-match}$',fontsize=20)
    plt.ylabel(r'A$_{\rm V, AGN}$',fontsize=20)
'''
dustcomp_line = Plot('dustcomp_line_sn')
'''
    plt.ylim([-6, 6])
    plt.xlim([0, 150])
    plt.xlabel(line+ r' S/N',fontsize=20)
    plt.ylabel(r'A$_{\rm V, AGN}$',fontsize=20)
    plt.plot([-6,6],[-6,6], c='k',ls='-.')
'''
dustcomp_agnsub_line = Plot('dustcomp_agnsub_linesn')
'''
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([-6, 6])
    plt.xlim([0, 150])
    plt.xlabel(line+ r' S/N',fontsize=20)
    plt.ylabel(r'A$_{\rm V, AGN-Match}$',fontsize=20)
'''
dustcomp_gsw_dec = Plot('dustcomp_gsw_dec')
'''
    plt.ylim([-6, 6])
    plt.xlim([-6, 6])
    plt.xlabel(r'A$_{\rm V, GSW}$',fontsize=20)
    plt.ylabel(r'A$_{\rm V, Balmer\ Dec.}$',fontsize=20)
'''
#%% bpt
bpt =  Plot('bpt',  minx=-2, maxx=1, miny=-1.2, maxy = 1.2, xlabel = r'log([NII]/H$\rm \alpha$)',ylabel=r'log([OIII]/H$\rm \beta$)', nx=300,ny=240)
bptdiffs =  Plot('bpt',  minx=-2, maxx=1, miny=-1.2, maxy = 1.2, xlabel = r'log([NII]/H$\rm \alpha$)',ylabel=r'log([OIII]/H$\rm \beta$)', nx=300,ny=240)
bptdiffsgrid =  Plot('bpt',  minx=-2, maxx=1, miny=-1.2, maxy = 1.2, xlabel = r'log([NII]/H$\rm \alpha$)',ylabel=r'log([OIII]/H$\rm \beta$)', nx=300,ny=240)
bptnormal =  Plot('bpt',  minx=-2, maxx=1, miny=-1.2, maxy = 1.2, xlabel = r'log([NII]/H$\rm \alpha$)',ylabel=r'log([OIII]/H$\rm \beta$)', nx=300,ny=240)
bptplus =  Plot('bpt',  minx=-2, maxx=1, miny=-1.2, maxy = 1.2, xlabel = r'log([NII]/H$\rm \alpha$)',ylabel=r'log([OIII]/H$\rm \beta$)', nx=300,ny=240)
bpt_sii = Plot('bpt_sii') 
bpt_oii = Plot('bpt_oii')
bpt_mex = Plot('bpt_mex')

#%% hists
match_typs = {0:'BPT SF', 1:'BPT+ SF', 2:'Unclassifiable'}

disp_cell_hist = Hist('disps')
'''    plt.xlabel('Displacements', fontsize=20)
    plt.ylabel('Counts', fontsize=20)
'''
disp_type_hist = Hist('disp type')
'''plt.ylim([0, np.max(hst)+1*np.max(hst)/6])
    plt.xlim([-0.05,2])
    plt.xlabel('Displacements', fontsize=20)
    plt.ylabel('Counts', fontsize=20)
'''

disp_all_hist = Hist('Disp all')
'''    plt.xlabel('Displacements', fontsize=20)
    plt.ylabel('Counts', fontsize=20)
'''
'''
hist_displacements_cell(sfrm_gsw2.bpt_set[427][-3],sfrm_gsw2.bpt_set[427][-2], sfrm_gsw2.bpt_set[427][2:4], save= True, filename='bpt_example_1')

hist_displacements_cell(sfrm_gsw2.plus_set[427][-3],sfrm_gsw2.plus_set[427][-2], sfrm_gsw2.plus_set[427][2:4], save= False, filename='plus_example_1')

hist_displacements_cell(sfrm_gsw2.neither_set[427][-3],sfrm_gsw2.neither_set[427][-2], sfrm_gsw2.neither_set[427][2:4], save= True, filename='neither_example_1')



hist_displacements_by_type(sfrm_gsw2.bpt_set, bins=40, rang=[0,2], filename='bpt', save=True)
hist_displacements_by_type(sfrm_gsw2.plus_set, bins=40, rang=[0,2], filename='plus', save=True)
hist_displacements_by_type(sfrm_gsw2.neither_set, bins=40, rang=[0,2], filename='unclassified', save=True)
hist_displacements_all(sfrm_gsw2.bpt_set,sfrm_gsw2.plus_set, sfrm_gsw2.neither_set, bins=40, rang=[0,2], filename='all_agns', save=True)

hist_displacements_by_type(sfrm_gsw2.bpt_set, bins=40, filename='bpt')
hist_displacements_by_type(sfrm_gsw2.plus_set, bins=40, filename='plus')
hist_displacements_by_type(sfrm_gsw2.neither_set, bins=40, filename='unclassified')
hist_displacements_all(sfrm_gsw2.bpt_set,sfrm_gsw2.plus_set, sfrm_gsw2.neither_set, bins=40, rang=[0,5], filename='all_agns')
'''
sf_diffs_hist = Hist('sf')
'''    plt.xlabel('Distance to ' + match_typs[matchtyp], fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    
i=0
histsf_diffs(sfrm_gsw2.dists_best[i], sfrm_gsw2.best_types[i], filename =str(i)+'_avgn'+str(sfrm_gsw2.n_avg), save=True)
'''
flux_diffs_hist = Hist('flux_diffs')
'''        plt.xlabel(line +r'$_{\mathrm{AGN}}$-'+line+r'$_{\mathrm{match}}$'+', '+ typ, fontsize=20)
'''
sfr_diffs_hist = Hist('sfr_diffs')
'''    plt.xlabel('Distance to closest match',fontsize=20)
    plt.ylabel(r'N$_{matches}$',fontsize=20)    
    hist_sfrdiffs(sfrm_gsw2.mindists_best, 'Combined', ymax=30000, col='k', ran=(0,1), bins=40, save=True, filename='combined')    
    hist_sfrdiffs(sfrm_gsw2.mindists[0], 'BPT+', col='r',  ymax=30000,ran=(0,1), bins=40, save=False, filename='bptplus')    
    hist_sfrdiffs(sfrm_gsw2.mindists[1],'NII/Ha', col='g', ymax=30000, ran=(0,1), bins=40, save=False, filename='niihalp')    
    hist_sfrdiffs(sfrm_gsw2.mindists[2],'Unclassified', col='b',  ymax=30000, ran=(0,1), bins=40, save=False, filename='unclass')    
    hist_sfrdiffs(sfrm_gsw2.mindistsagn_best, 'AGN Combined', ymax=40000, col='k', ran=(0,1), bins=40, save=True, filename='combined_agn')
    hist_sfrdiffs(sfrm_gsw2.mindists_agn[0], 'BPT AGN', ymax=40000, col='r', ran=(0,1), bins=40, save=True, filename='agn')
    hist_sfrdiffs(sfrm_gsw2.mindists_agn[1], 'NII/Ha AGN', ymax=40000, col='g', ran=(0,1), bins=40, save=True, filename='agnplus')
'''    
distrat =Hist('distrat')
'''plt.xlabel(r'$(d_{Match}/d_{AGN})^2$', fontsize=20)
plotdistrat(sfrm_gsw2.bptdistrat, 'BPT',save=True, filename='bpt')
plotdistrat(sfrm_gsw2.bptdistrat_plus, 'BPT Plus', save=True, filename='bptplus')


plotdistrat(sfrm_gsw2.bptdistrat_neither, 'Unclassifiable', save=True,filename='unclassifiable')

'''
massfrac_hist = Hist('massfrac')
'''    plt.xlabel('Mass Fraction')
    plt.ylabel('Counts')
'''
xrayhists = Hist('xray')
#plt.rc('text',usetex=False)

'''
dispy_vs_sfr(sfrm_gsw2.offset_oiii_hbeta,  sfrm_gsw2.sfr_bpt, filename='_bpt', save=True)
dispy_vs_sfr(sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.sfr_plus, filename='_plus', save=True)
dispy_vs_sfr(sfrm_gsw2.offset_oiii_hbeta_neither,sfrm_gsw2.sfr_neither, filename='_unclassified', save=True)



plot3panel( sfrm_gsw2.sfr_sing, 
           sfrm_gsw2.sfr_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.sfr_sing[sfrm_gsw2.low_ssfr_obj],
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2), 
            np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.high_ssfr_obj], 
            np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.low_ssfr_obj], 
            filename='diagnostic/sfr_disp', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=0, maxy=1.5, nx=250, ny=250, xlabel='log(SFR)', ylabel='Displacement')
plot3panel( sfrm_gsw2.sfr_match_sing, 
           sfrm_gsw2.sfr_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.sfr_match_sing[sfrm_gsw2.low_ssfr_obj],
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2), 
            np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.high_ssfr_obj], 
            np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.low_ssfr_obj], 
            filename='diagnostic/sfr_match_disp', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=0, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{GSW, Match}}$)', ylabel='Displacement')
plot3panel( sfrm_gsw2.ssfr_sing, 
           sfrm_gsw2.ssfr_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.ssfr_sing[sfrm_gsw2.low_ssfr_obj],
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2), 
            np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.high_ssfr_obj], 
            np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.low_ssfr_obj], 
            filename='diagnostic/ssfr_disp', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=0, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{GSW}}$)', ylabel='Displacement')

plot3panel( sfrm_gsw2.ssfr_match_sing, 
           sfrm_gsw2.ssfr_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.ssfr_match_sing[sfrm_gsw2.low_ssfr_obj],
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2), 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.high_ssfr_obj], 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.low_ssfr_obj], 
           filename='diagnostic/ssfr_match_disp', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=0, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{GSW, Match}}$)', ylabel='Displacement')


plot3panel( sfrm_gsw2.fibsfr_sing, 
           sfrm_gsw2.fibsfr_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_sing[sfrm_gsw2.low_ssfr_obj],
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2), 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.high_ssfr_obj], 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.low_ssfr_obj], 
           filename='diagnostic/sfr_fib_disp', save=False, nobj=False,
           bin_y=True,minx=-3.5, maxx=3.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, GSW}}$)', ylabel='Displacement')             

plot3panel( sfrm_gsw2.fibsfr_match_sing, 
           sfrm_gsw2.fibsfr_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_match_sing[sfrm_gsw2.low_ssfr_obj],
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2), 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.high_ssfr_obj], 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.low_ssfr_obj], 
           filename='diagnostic/sfr_fib_match_disp', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=0, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, GSW, Match}}$)', ylabel='Displacement')             

plot3panel( sfrm_gsw2.fibsfr_mpa_sing, 
           sfrm_gsw2.fibsfr_mpa_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_mpa_sing[sfrm_gsw2.low_ssfr_obj],
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2), 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.high_ssfr_obj], 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.low_ssfr_obj],
           filename='diagnostic/sfr_fib_mj_disp', save=True,
           bin_y=True,minx=-3.5, maxx=3.5,  miny=0, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, MJ}}$)', ylabel='Displacement')             
plot3panel( sfrm_gsw2.fibsfr_mpa_match_sing, 
           sfrm_gsw2.fibsfr_mpa_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_mpa_match_sing[sfrm_gsw2.low_ssfr_obj],
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2), 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.high_ssfr_obj], 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.low_ssfr_obj], 
           filename='diagnostic/sfr_fib_mj_match_disp', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=0, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, MJ, Match}}$)', ylabel='Displacement')             
plot3panel( sfrm_gsw2.fibssfr_mpa_sing, 
           sfrm_gsw2.fibssfr_mpa_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibssfr_mpa_sing[sfrm_gsw2.low_ssfr_obj],
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2), 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.high_ssfr_obj], 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.low_ssfr_obj], 
           filename='diagnostic/ssfr_fib_mj_disp', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=0, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{fib, MJ}}$)', ylabel='Displacement')             
plot3panel( sfrm_gsw2.fibssfr_mpa_match_sing, 
           sfrm_gsw2.fibssfr_mpa_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibssfr_mpa_match_sing[sfrm_gsw2.low_ssfr_obj],
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2), 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.high_ssfr_obj], 
           np.sqrt((sfrm_gsw2.offset_oiii_hbeta_sing)**2+(sfrm_gsw2.offset_nii_halpha_sing)**2)[sfrm_gsw2.low_ssfr_obj], 
           filename='diagnostic/ssfr_fib_mj_match_disp', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=0, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{fib, MJ, Match}}$)', ylabel='Displacement')             




plot3panel( sfrm_gsw2.sfr_sing, 
           sfrm_gsw2.sfr_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.sfr_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_oiii_hbeta_sing, 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/sfr_dispy', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR)', ylabel='Displacement Y')
plot3panel( sfrm_gsw2.sfr_match_sing, 
           sfrm_gsw2.sfr_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.sfr_match_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_oiii_hbeta_sing, 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/sfr_match_dispy', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{GSW, Match}}$)', ylabel='Displacement Y')
plot3panel( sfrm_gsw2.ssfr_sing, 
           sfrm_gsw2.ssfr_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.ssfr_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_oiii_hbeta_sing, 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/ssfr_dispy', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{GSW}}$)', ylabel='Displacement Y')

plot3panel( sfrm_gsw2.ssfr_match_sing, 
           sfrm_gsw2.ssfr_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.ssfr_match_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_oiii_hbeta_sing, 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/ssfr_match_dispy', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{GSW, Match}}$)', ylabel='Displacement Y')


plot3panel( sfrm_gsw2.fibsfr_sing, 
           sfrm_gsw2.fibsfr_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_oiii_hbeta_sing, 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/sfr_fib_dispy', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, GSW}}$)', ylabel='Displacement Y')             

plot3panel( sfrm_gsw2.fibsfr_match_sing, 
           sfrm_gsw2.fibsfr_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_match_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_oiii_hbeta_sing, 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/sfr_fib_match_dispy', save=True,
           bin_y=True,minx=-3.5, maxx=3.5,  miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, GSW, Match}}$)', ylabel='Displacement Y')             

plot3panel( sfrm_gsw2.fibsfr_mpa_sing, 
           sfrm_gsw2.fibsfr_mpa_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_mpa_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_oiii_hbeta_sing, 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/sfr_fib_mj_dispy', save=True,
           bin_y=True,minx=-3.5, maxx=3.5,  miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, MJ}}$)', ylabel='Displacement Y')             
plot3panel( sfrm_gsw2.fibsfr_mpa_match_sing, 
           sfrm_gsw2.fibsfr_mpa_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_mpa_match_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_oiii_hbeta_sing, 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/sfr_fib_mj_match_dispy', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, MJ, Match}}$)', ylabel='Displacement Y')             
plot3panel( sfrm_gsw2.fibssfr_mpa_sing, 
           sfrm_gsw2.fibssfr_mpa_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibssfr_mpa_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_oiii_hbeta_sing, 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/ssfr_fib_mj_dispy', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{fib, MJ}}$)', ylabel='Displacement Y')             
plot3panel( sfrm_gsw2.fibssfr_mpa_match_sing, 
           sfrm_gsw2.fibssfr_mpa_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibssfr_mpa_match_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_oiii_hbeta_sing, 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/ssfr_fib_mj_match_dispy', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{fib, MJ, Match}}$)', ylabel='Displacement Y')             



plot3panel( sfrm_gsw2.halpfibsfr_match_sing[sfrm_gsw2.ha_match_sing], 
           sfrm_gsw2.halpfibsfr_match_sing[sfrm_gsw2.high_ssfr_ha_match_sing], 
           sfrm_gsw2.halpfibsfr_match_sing[sfrm_gsw2.low_ssfr_ha_match_sing],
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.ha_match_sing], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_ha_match_sing], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_ha_match_sing], 
           filename='diagnostic/sfr_fibha_dispy', save=True,
           bin_y=True,minx=-3.5, maxx=3.5,  miny=-1.5, maxy=1.5, nx=250,
           ny=250, xlabel=r'log(SFR$_{\mathrm{H} \alpha}$)',  ylabel='Displacement Y')             



plot3panel( sfrm_gsw2.halpfibssfr_match_sing[sfrm_gsw2.ha_match_sing], 
           sfrm_gsw2.halpfibssfr_match_sing[sfrm_gsw2.high_ssfr_ha_match_sing], 
           sfrm_gsw2.halpfibssfr_match_sing[sfrm_gsw2.low_ssfr_ha_match_sing],
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.ha_match_sing], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_ha_match_sing], 
           sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_ha_match_sing], 
           filename='diagnostic/ssfr_fibha_dispy', save=True,
           bin_y=True,minx=-14, maxx=-7.5,  miny=-1.5, maxy=1.5, nx=250,
           ny=250, xlabel=r'log(sSFR$_{\mathrm{H} \alpha}$)',  ylabel='Displacement Y')             



plot3panel( sfrm_gsw2.sfr_sing, 
           sfrm_gsw2.sfr_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.sfr_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_nii_halpha_sing, 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/sfr_dispx', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR)', ylabel='Displacement X')
plot3panel( sfrm_gsw2.sfr_match_sing, 
           sfrm_gsw2.sfr_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.sfr_match_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_nii_halpha_sing, 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/sfr_match_dispx', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{GSW, Match}}$)', ylabel='Displacement X')
plot3panel( sfrm_gsw2.ssfr_sing, 
           sfrm_gsw2.ssfr_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.ssfr_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_nii_halpha_sing, 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/ssfr_dispx', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{GSW}}$)', ylabel='Displacement X')

plot3panel( sfrm_gsw2.ssfr_match_sing, 
           sfrm_gsw2.ssfr_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.ssfr_match_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_nii_halpha_sing, 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/ssfr_match_dispx', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{GSW, Match}}$)', ylabel='Displacement X')


plot3panel( sfrm_gsw2.fibsfr_sing, 
           sfrm_gsw2.fibsfr_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_nii_halpha_sing, 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/sfr_fib_dispx', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, GSW}}$)', ylabel='Displacement X')             

plot3panel( sfrm_gsw2.fibsfr_match_sing, 
           sfrm_gsw2.fibsfr_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_match_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_nii_halpha_sing, 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/sfr_fib_match_dispx', save=True,
           bin_y=True,minx=-3.5, maxx=3.5,  miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, GSW, Match}}$)', ylabel='Displacement X')             
plot3panel( sfrm_gsw2.halpfibsfr_match_sing[sfrm_gsw2.ha_match_sing], 
           sfrm_gsw2.halpfibsfr_match_sing[sfrm_gsw2.high_ssfr_ha_match_sing], 
           sfrm_gsw2.halpfibsfr_match_sing[sfrm_gsw2.low_ssfr_ha_match_sing],
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.ha_match_sing], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_ha_match_sing], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_ha_match_sing], 
           filename='diagnostic/sfr_fibha_dispx', save=False,
           bin_y=True,minx=-3.5, maxx=3.5,  miny=-1.5, maxy=1.5, nx=250,
           ny=250, xlabel=r'log(SFR$_{\mathrm{H} \alpha}$)', ylabel='Displacement X')             



plot3panel( sfrm_gsw2.fibsfr_mpa_sing, 
           sfrm_gsw2.fibsfr_mpa_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_mpa_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_nii_halpha_sing, 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/sfr_fib_mj_dispx', save=True,
           bin_y=True,minx=-3.5, maxx=3.5,  miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, MJ}}$)', ylabel='Displacement X')             
plot3panel( sfrm_gsw2.fibsfr_mpa_match_sing, 
           sfrm_gsw2.fibsfr_mpa_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibsfr_mpa_match_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_nii_halpha_sing, 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/sfr_fib_mj_match_dispx', save=True,
           bin_y=True,minx=-3.5, maxx=3.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(SFR$_{\mathrm{fib, MJ, Match}}$)', ylabel='Displacement X')             
plot3panel( sfrm_gsw2.fibssfr_mpa_sing, 
           sfrm_gsw2.fibssfr_mpa_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibssfr_mpa_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_nii_halpha_sing, 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/ssfr_fib_mj_dispx', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{fib, MJ}}$)', ylabel='Displacement X')             
plot3panel( sfrm_gsw2.fibssfr_mpa_match_sing, 
           sfrm_gsw2.fibssfr_mpa_match_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.fibssfr_mpa_match_sing[sfrm_gsw2.low_ssfr_obj],
           sfrm_gsw2.offset_nii_halpha_sing, 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/ssfr_fib_mj_match_dispx', save=True,
           bin_y=True,minx=-14, maxx=-7.5, miny=-1.5, maxy=1.5, nx=250, ny=250, xlabel='log(sSFR$_{\mathrm{fib, MJ, Match}}$)', ylabel='Displacement X')             


plot3panel( sfrm_gsw2.halpfibssfr_match_sing[sfrm_gsw2.ha_match_sing], 
           sfrm_gsw2.halpfibssfr_match_sing[sfrm_gsw2.high_ssfr_ha_match_sing], 
           sfrm_gsw2.halpfibssfr_match_sing[sfrm_gsw2.low_ssfr_ha_match_sing],
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.ha_match_sing], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_ha_match_sing], 
           sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_ha_match_sing], 
           filename='diagnostic/ssfr_fibha_dispx', save=False,
           bin_y=True,minx=-14, maxx=-7.5,  miny=-1.5, maxy=1.5, nx=250,
           ny=250, xlabel=r'log(SFR$_{\mathrm{H} \alpha}$)', ylabel='Displacement X')             


plot3panel( np.log10(sfrm_gsw2.siiflux_sub_sing/sfrm_gsw2.oiiflux_sub_sing)[sfrm_gsw2.oii_sii_sub_sing], 
           np.log10(sfrm_gsw2.siiflux_sub_sing/sfrm_gsw2.oiiflux_sub_sing)[sfrm_gsw2.high_ssfr_oii_sii_sub_sing], 
           np.log10(sfrm_gsw2.siiflux_sub_sing/sfrm_gsw2.oiiflux_sub_sing)[sfrm_gsw2.low_ssfr_oii_sii_sub_sing],
           np.log10(sfrm_gsw2.halpflux_sub_sing/sfrm_gsw2.hbetaflux_sub_sing)[sfrm_gsw2.oii_sii_sub_sing], 
           np.log10(sfrm_gsw2.halpflux_sub_sing/sfrm_gsw2.hbetaflux_sub_sing)[sfrm_gsw2.high_ssfr_oii_sii_sub_sing], 
           np.log10(sfrm_gsw2.halpflux_sub_sing/sfrm_gsw2.hbetaflux_sub_sing)[sfrm_gsw2.low_ssfr_oii_sii_sub_sing],
           filename='diagnostic/sii_oii_halp_hbeta_dust', save=False,
           bin_y=True,bin_stat_y='median',minx=-2, maxx=2, miny=-1, maxy=2, nx=250, ny=250, 
           xlabel='log([SII]/[OII])', ylabel=r'log(H$\alpha$/H$\beta$)',nan=True)             




#oiiilum/mass
plot3panel( sfrm_gsw2.mass_sing, 
           sfrm_gsw2.mass_sing[sfrm_gsw2.high_ssfr_obj], 
           sfrm_gsw2.mass_sing[sfrm_gsw2.low_ssfr_obj],
           (np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing), 
           (np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.high_ssfr_obj], 
           (np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/oiiilum_mass', save=False,
           bin_y=True,minx=8, maxx=12, miny=25, maxy=35, nx=250, ny=250, xlabel=r'log(M$_{*}$)', ylabel=r'log(L$_{\mathrm{[OIII]}}$/M$_{*}$')             

'''

def plotbpt(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=False,filename='',labels=True, leg=True,title=None, minx=-2, maxx=1, miny=-1.2, maxy=1.2):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > miny) &( bgy < maxy) & (bgx<maxx)&(bgx > minx) )[0]
    nx = 3/0.01
    ny = 2.4/0.01
    
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
    
    plt.plot(np.log10(xline1_kauffmann),np.log10(yline1_kauffmann),c='k',ls='-.')#,label='kauffmann Line')
    #plt.plot([-0.4, -0.4], [np.log10(yline1_kauffmann_plus[-1]), miny-0.1], c='k',ls='-.')
    if len(list(nonagnfilt))!=0:
        plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
    if len(list(agnfilt)) !=0:
        plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    if labels:
        plt.text(0.6,0.75,'AGN', fontsize=15)
        plt.text(-1.15,-0.3,'SF',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([miny-0.1,maxy])
    plt.xlim([minx-0.1, maxx+0.1])
    if leg:
        plt.legend(fontsize=15,frameon=False,loc=3,bbox_to_anchor=(-0.02, -0.02))
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/NII_OIII_scat'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/NII_OIII_scat'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        #fig.savefig('plots/xmm3/eps/diagnostic/NII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
        
'''
plotbpt(sfrm_gsw2.log_nii_halpha_sing[xmm3_xr_to_sfrm_bpt], 
        sfrm_gsw2.log_oiii_hbeta_sing[xmm3_xr_to_sfrm_bpt], 
        [], np.arange(len(xmm3_xr_to_sfrm_bpt)), EL_m2.niiha, EL_m2.oiiihb)
plotbpt(sfrm_gsw2.log_nii_halpha_sing[xmm3_xr_to_sfrm_plus], 
        sfrm_gsw2.log_oiii_hbeta_sing[xmm3_xr_to_sfrm_plus], 
        [], np.arange(len(xmm3_xr_to_sfrm_plus)), EL_m2.niiha, EL_m2.oiiihb)
plotbpt(sfrm_gsw2.log_nii_halpha_sing[xmm3_xr_to_sfrm_neither],
        sfrm_gsw2.log_oiii_hbeta_sing[xmm3_xr_to_sfrm_neither], 
        [], np.arange(len(xmm3_xr_to_sfrm_neither)), EL_m2.niiha, EL_m2.oiiihb)

plotbpt(sfrm_gsw2.log_nii_halpha_sub_sing[xmm3_xr_to_sfrm_bpt], 
        sfrm_gsw2.log_oiii_hbeta_sub_sing[xmm3_xr_to_sfrm_bpt], 
        [], np.arange(len(xmm3_xr_to_sfrm_bpt)), EL_m2.niiha, EL_m2.oiiihb)
plotbpt(sfrm_gsw2.log_nii_halpha_sub_sing[xmm3_xr_to_sfrm_plus], 
        sfrm_gsw2.log_oiii_hbeta_sub_sing[xmm3_xr_to_sfrm_plus], 
        [], np.arange(len(xmm3_xr_to_sfrm_plus)), EL_m2.niiha, EL_m2.oiiihb)
plotbpt(sfrm_gsw2.log_nii_halpha_sub_sing[xmm3_xr_to_sfrm_neither],
        sfrm_gsw2.log_oiii_hbeta_sub_sing[xmm3_xr_to_sfrm_neither], 
        [], np.arange(len(xmm3_xr_to_sfrm_neither)), EL_m2.niiha, EL_m2.oiiihb)
'''
def plotbptdiffs(x1,y1,x2,y2,save=False,filename='',labels=True, title=None, minx=-2, maxx=1.5, miny=-1.2, maxy=2, nobj=False):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(x1) & (np.isfinite(y1))) &
                (y1 > miny) &( y1 < maxy) & (x1<maxx)&(x1 > minx) )[0]
    nx = 3/0.01
    ny = 2.4/0.01
    dx  = x2-x1
    dy = y2-y1
    for i in range(len(x1))[3::5000]:
        plt.arrow(x1[i], y1[i], dx[i], dy[i], head_width = 0.05, alpha=0.6, length_includes_head=True , color='red', edgecolor='none', facecolor='none')
    
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffmann_plus),np.log10(yline1_kauffmann_plus),c='k',ls='-.')#,label='kauffmann Line')
    plt.plot([-0.4, -0.4], [np.log10(yline1_kauffmann_plus[-1]), miny-0.1], c='k',ls='-.')
    if nobj:
        plt.text(-1.7, -1, r"N$_{\mathrm{obj}}$: "+str(len(valbg)), fontsize=20)
    if labels:
        plt.text(0.6,0.75,'AGN', fontsize=15)
        plt.text(-1.15,-0.3,'SF',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([miny-0.1,maxy])
    plt.xlim([minx-0.1, maxx+0.1])
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/NII_OIII_diffs'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/NII_OIII_diffs'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/NII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''

plotbptdiffs(sfrm_gsw2.log_nii_halpha_sing, 
             sfrm_gsw2.log_oiii_hbeta_sing,
             sfrm_gsw2.log_nii_halpha_sub_sing,
             sfrm_gsw2.log_oiii_hbeta_sub_sing,
             save=False)


'''
def plotbptdiffsgrid(grid_arr, fig=None, ax=None, save=False,filename='',labels=True, title=None, minx=-2, maxx=1.5, miny=-1.2, maxy=2,nobj=False, avg = True):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if not fig and not ax:
        fig = plt.figure()
        ax = fig.add_subplot()
    nx = 3/0.01
    ny = 2.4/0.01
    if avg:
        x_ind = 4
        y_ind = 5
    else:
        x_ind = 6
        y_ind = 7
    for i in range(len(grid_arr)):
        if grid_arr[i][x_ind]==0 and grid_arr[i][y_ind]==0:
            plt.arrow(grid_arr[i][2], grid_arr[i][3], 0,0, color='red')
        else:
            plt.arrow(grid_arr[i][2], grid_arr[i][3], grid_arr[i][x_ind], grid_arr[i][y_ind], color='red', length_includes_head=True, head_width=0.02)
    #plt.quiver(x, y, offset_x, offset_y, color='red', alpha=0.7, scale_units='xy', angles='xy', scale=1)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffmann_plus),np.log10(yline1_kauffmann_plus),c='k',ls='-.')#,label='kauffmann Line')
    plt.plot([-0.4, -0.4], [np.log10(yline1_kauffmann_plus[-1]), miny-0.1], c='k',ls='-.')
    if nobj:
        plt.text(-1.7, -1, r"N$_{\mathrm{obj}}$: "+str(len(valbg)), fontsize=20)
    if labels:
        plt.text(0.6,0.75,'AGN', fontsize=15)
        plt.text(-1.15,-0.3,'SF',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([miny-0.1,maxy])
    plt.xlim([minx-0.1, maxx+0.1])
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/bptdiffs/NII_OIII_diffsgrid_'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/bptdiffs/NII_OIII_diffsgrid_'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/NII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''

plotbptdiffsgrid(sfrm_gsw2.bpt_set,
             save=True, filename='bpt')

plotbptdiffsgrid(sfrm_gsw2.plus_set,
             save=True, filename='plus')

plotbptdiffsgrid(sfrm_gsw2.neither_set,
             save=True, filename= 'unclassified')

fig = plt.figure(figsize=(8,8))
ax1= fig.add_subplot(311)

plotbptdiffsgrid(sfrm_gsw2.sing_set,
             save=False, filename= 'sing', fig=fig, ax=ax1)
ax2= fig.add_subplot(312)

plotbptdiffsgrid(sfrm_gsw2.low_set, save=False, filename='low', fig=fig, ax=ax2)
ax3= fig.add_subplot(313)
ax1.set_xlabel('')
ax2.set_xlabel('')


plotbptdiffsgrid(sfrm_gsw2.high_set, save=False, filename= 'high', fig=fig, ax=ax3)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)



plotbptdiffsgrid(sfrm_gsw2.bpt_set, save=True, filename='med_bpt', avg=False)
plotbptdiffsgrid(sfrm_gsw2.plus_set, save=True, filename='med_plus', avg=False)
plotbptdiffsgrid(sfrm_gsw2.neither_set, save=True, filename='med_unclassified', avg=False)


plotbptdiffsgrid(sfrm_gsw2.mesh_x, 
             sfrm_gsw2.mesh_y,
             sfrm_gsw2.med_offset_plus_x,
             sfrm_gsw2.med_offset_plus_y,
             save=True, filename='plus_med')

plotbptdiffsgrid(sfrm_gsw2.mesh_x, 
             sfrm_gsw2.mesh_y,
             sfrm_gsw2.med_offset_neither_x,
             sfrm_gsw2.med_offset_neither_y,
             save=True, filename= 'unclassified_med')


'''
def plotbptnormal(bgx,bgy,ccode= [], fig=None, ax=None, save=False,filename='',labels=True, title=None, minx=-2, maxx=1.5, miny=-1.2, maxy=2, nx=300, ny=240, ccodename='', ccodelim=[], nobj=True):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > miny) &( bgy < maxy) & (bgx<maxx)&(bgx > minx) )[0]
    #nx = 3/0.01
    #ny = 2.4/0.01
    if len(ccode) !=0:    
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ccode=ccode[valbg], ccodename=ccodename, ccodelim = ccodelim,ax=ax, fig=fig)
    else:
        plot2dhist(bgx[valbg], bgy[valbg], nx,ny, ax=ax, fig=fig)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    ax.plot(np.log10(xline1_kauffmann_plus),np.log10(yline1_kauffmann_plus),c='k',ls='-.')#,label='kauffmann Line')
    ax.plot([-0.4, -0.4], [np.log10(yline1_kauffmann_plus[-1]), miny-0.1], c='k',ls='-.')
    if nobj:
        ax.text(-1.85, -1, r"N$_{\mathrm{obj}}$: "+str(len(valbg)), fontsize=12)
    if labels:
        ax.text(0.6,0.75,'AGN', fontsize=15)
        ax.text(-1.15,-0.3,'SF',fontsize=15)
    ax.set_ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    ax.set_xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([miny-0.1,maxy])
    plt.xlim([minx-0.1, maxx+0.1])
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/bpt/NII_OIII_scat'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/bpt/NII_OIII_scat'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/NII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
def plot3panel_bpt(x1,x2,x3,y1,y2,y3,ccode1=[], ccode2=[], ccode3=[], save=False,
                   nobj=False,filename='',minx=-2, maxx=1.5, miny=-1.3, maxy=2,
                   nx=300, ny=240, ccodename='', bptfunc = plotbptnormal, 
                   ssfr_labels=True, liner_sy2_labels=False, d4000_labels=False):
    fig = plt.figure(figsize=(8,8))
  
    ax1 = fig.add_subplot(311)
    if bptfunc == plotbptnormal:
        ax1.set_xlim([minx,maxx])
        ax1.set_ylim([miny,maxy])
    ax1.set(aspect='equal', adjustable='box')
    
    ax2 = fig.add_subplot(312, sharey = ax1, sharex=ax1)
    ax3 = fig.add_subplot(313, sharey = ax1, sharex=ax1)
    ax2.set(adjustable='box', aspect='equal')
    ax3.set(adjustable='box', aspect='equal')
    bptfunc(x1, y1, ccode=ccode1, save=False, nobj=nobj, ax=ax1,fig=fig, minx=minx, maxx=maxx, miny=miny, maxy=maxy, nx=nx, ny=ny, ccodename=ccodename)
    bptfunc(x2, y2, ccode=ccode2, save=False, nobj=nobj, ax=ax2,fig=fig, minx=minx, maxx=maxx, miny=miny, maxy=maxy, nx=nx, ny=ny, ccodename=ccodename)
    if ssfr_labels:
        ax2.text(-1.85, 1.5, r'$\Delta$log(sSFR)$>-0.7$', fontsize=15)
        ax3.text(-1.85, 1.5, r'$\Delta$log(sSFR)$\leq-0.7$', fontsize=15)
    elif liner_sy2_labels:
        ax2.text(-1.85, 1.5, r'Seyfert 2', fontsize=15)
        ax3.text(-1.85, 1.5, r'LINER', fontsize=15)
    elif d4000_labels:
        ax2.text(-1.85, 1.5, r'D$_{4000}<$1.6', fontsize=15)
        ax3.text(-1.85, 1.5, r'D$_{4000}\geq$1.6', fontsize=15)
        
    bptfunc(x3, y3, ccode=ccode3, save=False, nobj=nobj, ax=ax3,fig=fig, minx=minx, maxx=maxx, miny=miny, maxy=maxy, nx=nx, ny=ny, ccodename=ccodename)    
    #ax3.set_ylabel('')
    #ax1.set_ylabel('')
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    plt.subplots_adjust(wspace=0, hspace=0)
    if save:
        fig.savefig('plots/sfrmatch/pdf/'+filename+'.pdf', bbox_inches='tight', dpi=250, format='pdf')
        fig.savefig('plots/sfrmatch/png/'+filename+'.png', bbox_inches='tight', dpi=250)
        plt.close(fig)
'''
#bpt pre-sub
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_presub_3panel', nobj=True,save=False)    
#ssfr color code
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.ssfr_sing,ccode2=sfrm_gsw2.ssfr_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.ssfr_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_ssfr', nobj=True, nx=20, ny=25,save=True, ccodename='log(sSFR)')


plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.ssfr_sing,
               ccode2=sfrm_gsw2.ssfr_sing[sfrm_gsw2.high_ssfr_obj], 
               ccode3=sfrm_gsw2.ssfr_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_ssfr', nobj=True, nx=20, ny=25,save=True, ccodename='log(sSFR)')


plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.ssfr_match_sing,ccode2=sfrm_gsw2.ssfr_match_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.ssfr_match_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_ssfr', nobj=False, nx=20, ny=25,save=True, ccodename='log(sSFR)')
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.delta_ssfr_sing,ccode2=sfrm_gsw2.delta_ssfr_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.delta_ssfr_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_delta_ssfr', nobj=True, nx=20, ny=25,save=False, ccodename=r'$\Delta$log(sSFR)')
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.delta_ssfr_match_sing,ccode2=sfrm_gsw2.delta_ssfr_match_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.delta_ssfr_match_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_delta_ssfr', nobj=False, nx=20, ny=25,save=False, ccodename=r'$\Delta$log(sSFR)')

#sfr color-code
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.sfr_sing,ccode2=sfrm_gsw2.sfr_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.sfr_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_sfr', nobj=True, nx=20, ny=25,save=True, ccodename='log(SFR)')
#fibsfr mpa
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.fibsfr_mpa_sing,ccode2=sfrm_gsw2.fibsfr_mpa_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.fibsfr_mpa_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_fibsfr_mpa', nobj=True, nx=20, ny=25,save=True, ccodename=r'log(SFR$_{\mathrm{Fib., MJ}}$)')
#fibssfr mpa
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.fibssfr_mpa_sing,ccode2=sfrm_gsw2.fibssfr_mpa_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.fibssfr_mpa_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_fibssfr_mpa', nobj=True, nx=20, ny=25,save=True, ccodename='log(sSFR$_{\mathrm{Fib., MJ}}$)')
#halpfibsfr
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.ha_match_sing],sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_ha_match_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_ha_match_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.ha_match_sing], sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_ha_match_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_ha_match_sing], 
               ccode1=sfrm_gsw2.halpfibsfr_match_sing[sfrm_gsw2.ha_match_sing],ccode2=sfrm_gsw2.halpfibsfr_match_sing[sfrm_gsw2.high_ssfr_ha_match_sing], ccode3=sfrm_gsw2.halpfibsfr_match_sing[sfrm_gsw2.low_ssfr_ha_match_sing],
               filename='diagnostic/BPT_presub_3panel_halpsfr',  nx=20, ny=25,nobj=True, save=True, ccodename=r'log(SFR$_{\mathrm{H}\alpha}$)')

#halpfibssfr

plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.ha_match_sing],sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_ha_match_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_ha_match_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.ha_match_sing], sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_ha_match_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_ha_match_sing], 
               ccode1=sfrm_gsw2.halpfibssfr_match_sing[sfrm_gsw2.ha_match_sing],ccode2=sfrm_gsw2.halpfibssfr_match_sing[sfrm_gsw2.high_ssfr_ha_match_sing], ccode3=sfrm_gsw2.halpfibssfr_match_sing[sfrm_gsw2.low_ssfr_ha_match_sing],
               filename='diagnostic/BPT_presub_3panel_halpssfr',  nx=20, ny=25,nobj=True, save=True, ccodename=r'log(sSFR$_{\mathrm{H}\alpha}$)')


#fibsfr gsw
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.fibsfr_sing,ccode2=sfrm_gsw2.fibsfr_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.fibsfr_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_fibsfr_gsw', nobj=True, nx=20, ny=25,save=True, ccodename='log(SFR$_{\mathrm{Fib., GSW}}$)')
#fibsfr gsw match
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.fibsfr_match_sing,ccode2=sfrm_gsw2.fibsfr_match_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.fibsfr_match_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_fibsfr_gsw_match', nobj=True, nx=20, ny=25,save=False, ccodename='log(SFR$_{\mathrm{Fib., GSW, Match}}$)')
#fib sfr mpa match
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.fibsfr_mpa_match_sing,ccode2=sfrm_gsw2.fibsfr_mpa_match_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.fibsfr_mpa_match_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_fibsfr_mpa_match', nobj=True, nx=20, ny=25,save=True, ccodename='log(SFR$_{\mathrm{Fib., MJ, Match}}$)')
#fib ssfr mpa match
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               ccode1=sfrm_gsw2.fibssfr_mpa_match_sing,ccode2=sfrm_gsw2.fibssfr_mpa_match_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.fibssfr_mpa_match_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_fibssfr_mpa_match', nobj=True, nx=20, ny=25,save=False, ccodename='log(sSFR$_{\mathrm{Fib., MJ, Match}}$)')

#bpt sub
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, 
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], 
               filename='diagnostic/BPT_sub_3panel', save=False)    
#bpt sub mass restricted
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.mass_filt],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_mass_filt_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_mass_filt_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.mass_filt], 
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_mass_filt_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_mass_filt_obj], 
               filename='diagnostic/BPT_sub_3panel_mass_restricted', save=False, nx=100, ny=150)    
#bpt sub sy2 linear
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.sii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.sy2_sub_filt],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.liner_sub_filt],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.sii_sub_sing], 
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.sy2_sub_filt],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.liner_sub_filt], 
               filename='diagnostic/BPT_sub_3panel_sy2_liner', save=True, liner_sy2_labels=True, ssfr_labels=False, nobj=True)    
#bpt presub sy2 liner
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.sii_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.sy2_filt],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.liner_filt],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.sii_sing], 
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.sy2_filt],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.liner_filt], 
               filename='diagnostic/BPT_presub_3panel_sy2_liner', save=False, liner_sy2_labels=True, ssfr_labels=False, nobj=True)    


#bpt colorcoded by delta log ssfr
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_delta_ssfr_ccode',
               ccode1=sfrm_gsw2.delta_ssfr_sing, ccode2=sfrm_gsw2.delta_ssfr_sing[sfrm_gsw2.high_ssfr_obj], 
               ccode3=sfrm_gsw2.delta_ssfr_sing[sfrm_gsw2.low_ssfr_obj],
               nx=20, ny=25, ccodename=r'$\Delta$log(sSFR)', save=True)


plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_delta_ssfr_ccode',
               ccode1=sfrm_gsw2.delta_ssfr_sing, ccode2=sfrm_gsw2.delta_ssfr_sing[sfrm_gsw2.high_ssfr_obj], 
               ccode3=sfrm_gsw2.delta_ssfr_sing[sfrm_gsw2.low_ssfr_obj], nobj=False,
               nx=20, ny=25, ccodename=r'$\Delta$log(sSFR)', save=True)

plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_delta_ssfr_match_ccode',
               ccode1=sfrm_gsw2.delta_ssfr_match_sing, ccode2=sfrm_gsw2.delta_ssfr_match_sing[sfrm_gsw2.high_ssfr_obj], 
               ccode3=sfrm_gsw2.delta_ssfr_match_sing[sfrm_gsw2.low_ssfr_obj], nobj=False,
               nx=20, ny=25, ccodename=r'$\Delta$log(sSFR$_{\mathrm{Match}}$)', save=True)


plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_delta_ssfr_ccode',
               ccode1=sfrm_gsw2.fibssfr_mpa_match_sing, ccode2=sfrm_gsw2.fibssfr_mpa_match_sing[sfrm_gsw2.high_ssfr_obj], 
               ccode3=sfrm_gsw2.fibssfr_mpa_match_sing[sfrm_gsw2.low_ssfr_obj],
               nx=20, ny=25, ccodename=r'log(sSFR$_{\mathrm{fib, MJ, Match}}$)', save=False)


plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_sfr_ccode',
               ccode1=sfrm_gsw2.sfr_sing, ccode2=sfrm_gsw2.sfr_sing[sfrm_gsw2.high_ssfr_obj], 
               ccode3=sfrm_gsw2.sfr_sing[sfrm_gsw2.low_ssfr_obj],
               nx=20, ny=25, ccodename=r'log(SFR)', save=True)



plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sii_sing],sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_oii_sii_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_oii_sii_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sii_sing], sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_oii_sii_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_oii_sii_sing], filename='diagnostic/BPT_sub_3panel_sfr_ccode',
               ccode1=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing)[sfrm_gsw2.oii_sii_sing],
               ccode2=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing)[sfrm_gsw2.high_ssfr_oii_sii_sing], 
               ccode3=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing)[sfrm_gsw2.low_ssfr_oii_sii_sing],
               nx=20, ny=25, ccodename=r'log([OI]/[SII])', save=False)


plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_sfr_ccode',
               ccode1=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing), ccode2=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing)[sfrm_gsw2.high_ssfr_obj], 
               ccode3=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing)[sfrm_gsw2.low_ssfr_obj],
               nx=20, ny=25, ccodename=r'log([OI]/[SII])', save=False)

#[oIII]/[OII] sub
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_oii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_oii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_oii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_oii_sub_sing], filename='diagnostic/BPT_sub_3panel_U_ccode',
               ccode1=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.oii_sub_sing], 
               ccode2=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.high_ssfr_oii_sub_sing], 
               ccode3=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.low_ssfr_oii_sub_sing],
               nx=20, ny=25, ccodename=r'log([OIII]/[OII])', save=True)
#[OIII]/[OII] presub
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.oii_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_oii_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_oii_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.oii_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_oii_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_oii_sing], filename='diagnostic/BPT_presub_3panel_U_ccode',
               ccode1=np.log10(sfrm_gsw2.oiiiflux_sub_sing/sfrm_gsw2.oiiflux_sub_sing)[sfrm_gsw2.oii_sing], 
               ccode2=np.log10(sfrm_gsw2.oiiiflux_sub_sing/sfrm_gsw2.oiiflux_sub_sing)[sfrm_gsw2.high_ssfr_oii_sing], 
               ccode3=np.log10(sfrm_gsw2.oiiiflux_sub_sing/sfrm_gsw2.oiiflux_sub_sing)[sfrm_gsw2.low_ssfr_oii_sing],
               nx=20, ny=25, ccodename=r'log([OIII]/[OII])', save=True)
#[oIII]/[OII] sub mass restrict
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sub_mass_filt_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_oii_sub_mass_filt_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_oii_sub_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sub_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_oii_sub_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_oii_sub_mass_filt_sing],
               filename='diagnostic/BPT_sub_3panel_U_ccode_mass_restricted',
               ccode1=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.oii_sub_mass_filt_sing], 
               ccode2=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.high_ssfr_oii_sub_mass_filt_sing], 
               ccode3=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.low_ssfr_oii_sub_mass_filt_sing],
               nx=20, ny=25, ccodename=r'log([OIII]/[OII])', save=True, nobj=True)
#[OIII]/[OII] presub mass restrict
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.oii_mass_filt_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_oii_mass_filt_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_oii_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.oii_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_oii_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_oii_mass_filt_sing], 
               filename='diagnostic/BPT_presub_3panel_U_ccode_mass_restricted',
               ccode1=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.oii_mass_filt_sing], 
               ccode2=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.high_ssfr_oii_mass_filt_sing], 
               ccode3=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.low_ssfr_oii_mass_filt_sing],
               nx=20, ny=25, ccodename=r'log([OIII]/[OII])', save=True,nobj=True)
#[oIII]/[OII] sub linersy2
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sub_sy2_sub_filt_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sub_liner_sub_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sub_sy2_sub_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sub_liner_sub_filt_sing],
               filename='diagnostic/BPT_sub_3panel_U_ccode_sy2_liner',
               ccode1=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.oii_sub_sing], 
               ccode2=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.oii_sub_sy2_sub_filt_sing], 
               ccode3=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.oii_sub_liner_sub_filt_sing],
               nx=20, ny=25, ccodename=r'log([OIII]/[OII])', save=True, nobj=True, liner_sy2_labels=True, ssfr_labels=False)
#[OIII]/[OII] presub linersy2
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.oii_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.oii_sy2_filt_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.oii_liner_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.oii_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.oii_sy2_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.oii_liner_filt_sing], 
               filename='diagnostic/BPT_presub_3panel_U_ccode_sy2_liner',
               ccode1=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.oii_sing], 
               ccode2=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.oii_sy2_filt_sing], 
               ccode3=np.log10(sfrm_gsw2.oiiiflux_sub_dered_sing/sfrm_gsw2.oiiflux_sub_dered_sing)[sfrm_gsw2.oii_liner_filt_sing],
               nx=20, ny=25, ccodename=r'log([OIII]/[OII])', save=True,nobj=True, liner_sy2_labels=True, ssfr_labels=False)

#[OI]/[SII]
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oi_sii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_oi_sii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_oi_sii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oi_sii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_oi_sii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_oi_sii_sub_sing], filename='diagnostic/BPT_sub_3panel_oi_sii_ccode',
               ccode1=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing)[sfrm_gsw2.oi_sii_sub_sing], 
               ccode2=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing)[sfrm_gsw2.high_ssfr_oi_sii_sub_sing], 
               ccode3=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing)[sfrm_gsw2.low_ssfr_oi_sii_sub_sing],
               nx=20, ny=25, ccodename=r'log([OI]/[SII])', save=False)



#l oiii
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_delta_ssfr_ccode',
               ccode1=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing), ccode2=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)[sfrm_gsw2.high_ssfr_obj], 
               ccode3=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)[sfrm_gsw2.low_ssfr_obj],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$)', save=False)
#loiii mass sub
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, 
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_oiiilum_mass_sub_ccode',
               ccode1=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)-sfrm_gsw2.mass_sing, 
               ccode2=(np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.high_ssfr_obj], 
               ccode3=(np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.low_ssfr_obj],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$/M$_{*}$)', save=True)
#loiii mass presub
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing,
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing, 
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], 
               filename='diagnostic/BPT_presub_3panel_oiiilum_mass_ccode',
               ccode1=np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing, 
               ccode2=(np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.high_ssfr_obj], 
               ccode3=(np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.low_ssfr_obj],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$/M$_{*}$)', save=True)


#l oiii mass restrict
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.mass_filt],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_mass_filt_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_mass_filt_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.mass_filt], 
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_mass_filt_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_mass_filt_obj], 
               filename='diagnostic/BPT_sub_3panel_oiiilum_ccode_mass_restricted',
               ccode1=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)[sfrm_gsw2.mass_filt],
               ccode2=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)[sfrm_gsw2.high_ssfr_mass_filt_obj], 
               ccode3=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)[sfrm_gsw2.low_ssfr_mass_filt_obj],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$)', save=False, nobj=True)
#l oiii presub liner sy2
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.sii_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.sy2_filt],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.liner_filt],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.sii_sing], 
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.sy2_filt],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.liner_filt], 
               filename='diagnostic/BPT_presub_3panel_oiiilum_ccode_sy2_liner',
               ccode1=np.log10(sfrm_gsw2.oiiilum_dered_sing)[sfrm_gsw2.sii_sing],
               ccode2=np.log10(sfrm_gsw2.oiiilum_dered_sing)[sfrm_gsw2.sy2_filt], 
               ccode3=np.log10(sfrm_gsw2.oiiilum_dered_sing)[sfrm_gsw2.liner_filt],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$)', save=True, liner_sy2_labels=True, ssfr_labels=False, nobj=True)
#l oiii  liner sy2
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.sii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.sy2_sub_filt],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.liner_sub_filt],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.sii_sub_sing], 
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.sy2_sub_filt],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.liner_sub_filt], 
               filename='diagnostic/BPT_sub_3panel_oiiilum_ccode_sy2_liner',
               ccode1=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)[sfrm_gsw2.sii_sub_sing],
               ccode2=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)[sfrm_gsw2.sy2_sub_filt], 
               ccode3=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)[sfrm_gsw2.liner_sub_filt],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$)', save=True, liner_sy2_labels=True, ssfr_labels=False)
#l oiii presub mass restrict
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.mass_filt],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_mass_filt_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_mass_filt_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.mass_filt], 
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_mass_filt_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_mass_filt_obj], 
               filename='diagnostic/BPT_presub_3panel_oiiilum_ccode_mass_restricted',
               ccode1=np.log10(sfrm_gsw2.oiiilum_dered_sing)[sfrm_gsw2.mass_filt],
               ccode2=np.log10(sfrm_gsw2.oiiilum_dered_sing)[sfrm_gsw2.high_ssfr_mass_filt_obj], 
               ccode3=np.log10(sfrm_gsw2.oiiilum_dered_sing)[sfrm_gsw2.low_ssfr_mass_filt_obj],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$)', save=True)
#loiii mass sub mass restrict
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.mass_filt],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_mass_filt_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_mass_filt_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.mass_filt], 
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_mass_filt_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_mass_filt_obj],
               filename='diagnostic/BPT_sub_3panel_oiiilum_mass_sub_ccode_mass_restricted',
               ccode1=(np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.mass_filt], 
               ccode2=(np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.high_ssfr_mass_filt_obj], 
               ccode3=(np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.low_ssfr_mass_filt_obj],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$/M$_{*}$)', save=False)
#loiii mass presub mass restrict
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.mass_filt],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_mass_filt_obj],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_mass_filt_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.mass_filt], 
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_mass_filt_obj],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_mass_filt_obj], 
               filename='diagnostic/BPT_presub_3panel_oiiilum_mass_ccode_mass_restricted',
               ccode1=(np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.mass_filt], 
               ccode2=(np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.high_ssfr_mass_filt_obj], 
               ccode3=(np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.low_ssfr_mass_filt_obj],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$/M$_{*}$)', save=True)

#loiii mass sub sy2 liner
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.sii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.sy2_sub_filt],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.liner_sub_filt],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.sii_sub_sing], 
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.sy2_sub_filt],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.liner_sub_filt],
               filename='diagnostic/BPT_sub_3panel_oiiilum_mass_sub_ccode_sy2_liner',
               ccode1=(np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.sii_sub_sing], 
               ccode2=(np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.sy2_sub_filt], 
               ccode3=(np.log10(sfrm_gsw2.oiiilum_sub_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.liner_sub_filt],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$/M$_{*}$)', save=True, liner_sy2_labels=True, ssfr_labels=False, nobj=True)
#loiii mass presub sy2 liner
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.sii_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.sy2_filt],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.liner_filt],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.sii_sing], 
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.sy2_filt],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.liner_filt], 
               filename='diagnostic/BPT_presub_3panel_oiiilum_mass_ccode_sy2_liner',
               ccode1=(np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.sii_sing], 
               ccode2=(np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.sy2_filt], 
               ccode3=(np.log10(sfrm_gsw2.oiiilum_dered_sing)-sfrm_gsw2.mass_sing)[sfrm_gsw2.liner_filt],
               nx=20, ny=25, ccodename=r'log(L$_{\mathrm{[OIII]}}$/M$_{*}$)', save=True, liner_sy2_labels=True, ssfr_labels=False, nobj=True)

#logoh sub
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_oii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_oii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_oii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_oii_sub_sing], filename='diagnostic/BPT_sub_3panel_logoh_ccode',
               ccode1=sfrm_gsw2.logoh_sub_sing[sfrm_gsw2.oii_sub_sing], 
               ccode2=sfrm_gsw2.logoh_sub_sing[sfrm_gsw2.high_ssfr_oii_sub_sing], 
               ccode3=sfrm_gsw2.logoh_sub_sing[sfrm_gsw2.low_ssfr_oii_sub_sing],
               nx=20, ny=25, ccodename=r'log(O/H)+12', save=True)
#logoh presub

plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.oii_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_oii_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_oii_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.oii_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_oii_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_oii_sing], filename='diagnostic/BPT_3panel_logoh_ccode',
               ccode1=sfrm_gsw2.logoh_sing[sfrm_gsw2.oii_sing], 
               ccode2=sfrm_gsw2.logoh_sing[sfrm_gsw2.high_ssfr_oii_sing], 
               ccode3=sfrm_gsw2.logoh_sing[sfrm_gsw2.low_ssfr_oii_sing],
               nx=20, ny=25, ccodename=r'log(O/H)+12', save=True)


#logoh sub mass restricted
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sub_mass_filt_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_oii_sub_mass_filt_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_oii_sub_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sub_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_oii_sub_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_oii_sub_mass_filt_sing], 
               filename='diagnostic/BPT_sub_3panel_logoh_ccode_mass_restricted',
               ccode1=sfrm_gsw2.logoh_sub_sing[sfrm_gsw2.oii_sub_mass_filt_sing], 
               ccode2=sfrm_gsw2.logoh_sub_sing[sfrm_gsw2.high_ssfr_oii_sub_mass_filt_sing], 
               ccode3=sfrm_gsw2.logoh_sub_sing[sfrm_gsw2.low_ssfr_oii_sub_mass_filt_sing],
               nx=20, ny=25, ccodename=r'log(O/H)+12', save=True, nobj=True)
#logoh presub mass restricted

plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.oii_mass_filt_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_oii_mass_filt_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_oii_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.oii_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_oii_mass_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_oii_mass_filt_sing],
               filename='diagnostic/BPT_presub_3panel_logoh_ccode_mass_restricted',
               ccode1=sfrm_gsw2.logoh_sing[sfrm_gsw2.oii_mass_filt_sing], 
               ccode2=sfrm_gsw2.logoh_sing[sfrm_gsw2.high_ssfr_oii_mass_filt_sing], 
               ccode3=sfrm_gsw2.logoh_sing[sfrm_gsw2.low_ssfr_oii_mass_filt_sing],
               nx=20, ny=25, ccodename=r'log(O/H)+12', save=True, nobj=True)


#logoh sub liner sy2
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sub_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sub_sy2_sub_filt_sing],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.oii_sub_liner_sub_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sub_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sub_sy2_sub_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.oii_sub_liner_sub_filt_sing], 
               filename='diagnostic/BPT_sub_3panel_logoh_ccode_sy2_liner',
               ccode1=sfrm_gsw2.logoh_sub_sing[sfrm_gsw2.oii_sub_sing], 
               ccode2=sfrm_gsw2.logoh_sub_sing[sfrm_gsw2.oii_sub_sy2_sub_filt_sing], 
               ccode3=sfrm_gsw2.logoh_sub_sing[sfrm_gsw2.oii_sub_liner_sub_filt_sing],
               nx=20, ny=25, ccodename=r'log(O/H)+12', save=True, nobj=True, liner_sy2_labels=True, ssfr_labels=False)
#logoh presub liner sy2

plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.oii_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.oii_sy2_filt_sing],
               sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.oii_liner_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.oii_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.oii_sy2_filt_sing],
               sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.oii_liner_filt_sing],
               filename='diagnostic/BPT_presub_3panel_logoh_ccode_sy2_liner',
               ccode1=sfrm_gsw2.logoh_sing[sfrm_gsw2.oii_sing], 
               ccode2=sfrm_gsw2.logoh_sing[sfrm_gsw2.oii_sy2_filt_sing], 
               ccode3=sfrm_gsw2.logoh_sing[sfrm_gsw2.oii_liner_filt_sing],
               nx=20, ny=25, ccodename=r'log(O/H)+12', save=False, nobj=True, liner_sy2_labels=True, ssfr_labels=False)

#av
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing,
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_balmerdecrement_ccode',
               ccode1=sfrm_gsw2.av_sub_sing, 
               ccode2=sfrm_gsw2.av_sub_sing[sfrm_gsw2.high_ssfr_obj], 
               ccode3=sfrm_gsw2.av_sub_sing[sfrm_gsw2.low_ssfr_obj],
               nx=20, ny=25, ccodename=r'A$_{\mathrm{V, Balmer Dec.}}$', save=True)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing,save=False, nobj=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj], sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj],save=False, nobj=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj], sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],save=False, nobj=False)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt]),filename='_test', save=False, nobj=False)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][agn_gsw_bptplus]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][agn_gsw_bptplus]),filename='_agns_presub', save=False)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sing, sfrm_gsw2.log_oiii_hbeta_sing)
plotbptnormal(EL_m2.niiha[agn_gsw_bptplus] , EL_m2.oiiihb[agn_gsw_bptplus],
              filename='', save=False, nx=200, ny= 200)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sing ,  sfrm_gsw2.log_oiii_hbeta_sing, 
              filename='_sfr', save=False, nx=200, ny=200)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=np.log10(sfrm_gsw2.oiiilum_sub_dered_bpt), filename='_oiiilum_dered_bpt', save=False, nx=75, ny=60)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,filename='_fluxsub_highsn_plus', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing ,  sfrm_gsw2.log_oiii_hbeta_sub_sing,filename='_fluxsub_highsn_all_agns', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing_agns_sf ,  sfrm_gsw2.log_oiii_hbeta_sub_sing_agns_sf,filename='_fluxsub_highsn_all_agns_all_sfs', save=True)


'''
nii_bound=0.4
def plotbptplus(bgx, bgy, bgxhist, unclass, nonagn=[], agn=[], save=False,filename='',labels=True, title=None,nii_bound=nii_bound, minx=-2, maxx=1, miny=-1.2, maxy=1.2):
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
    plt.plot(np.log10(xline1_kauffmann_plus),np.log10(yline1_kauffmann_plus),c='k',ls='-.')#,label='kauffmann Line')
    plt.plot([-0.4, -0.4], [np.log10(yline1_kauffmann_plus[-1]), miny-0.1], c='k',ls='-.')
    #ax1.plot(np.log10(xline1_kauffmann),np.log10(yline1_kauffmann),c='k',ls='-.')#,label='kauffmann Line')
    #ax1.plot(np.log10(xline1_kauffmann),np.log10(yline_stasinska),c='k',ls='-.')#,label='kauffmann Line')

    ax1.set_ylim([miny-0.1,maxy])
    ax1.set_xlim([minx-0.1, maxx+0.1])

    #ax1.axvline(x=nii_bound,color='k', alpha=0.8, linewidth=1, ls='-')
    #ax1.axvline(x=nii_bound+0.05,color='k', alpha=0.8, linewidth=1, ls='-')
    
    plt.text(-1.8, -0.4, r"N$_{\mathrm{obj}}$: "+str(len(bgx))+'('+str(round(100*len(bgx)/(len(bgxhist_finite)+len(bgx) +len(unclass))))+'\%)', fontsize=15)
    if len(nonagn) != 0 and len(agn) != 0:
        plt.text(-1.8, -1, r"N$_{\mathrm{AGN}}$: "+str(len(agn)) +'('+str(round(100*len(agn)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)
        plt.text(-1.8, -0.7, r"N$_{\mathrm{SF}}$: "+str(len(nonagn))+'('+str(round(100*len(nonagn)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)
        
    if labels:
        ax1.text(0.6,0.75,'AGN', fontsize=15)
        ax1.text(-1.15,-0.3,'SF',fontsize=15)
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
    ax2.set_xlim([minx-0.1, maxx+0.1])
    ax2.set_xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    ax2.set_ylabel(r'Fraction',fontsize=20)
    #ax2.set_xticks(np.arange(0.1, 1, 0.1), minor = True)
    ax2.axvline(x=nii_bound, color='k',ls='-.' )
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
        fig.savefig('./plots/sfrmatch/png/diagnostic/NII_OIII_scat'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('./plots/sfrmatch/pdf/diagnostic/NII_OIII_scat'+filename+'.pdf',dpi=250,bbox_inches='tight')
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
            filename='bptplus_sn2', save=True, labels=False)

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
         [],
         [],
         EL_m2.niiha,EL_m2.oiiihb,save=False)






plotbpt(EL_3xmm.niiha,EL_3xmm.oiiihb,
         [],
         [],
         EL_m2.niiha,EL_m2.oiiihb,save=False)
plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][1], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][1], color='b', marker='*', label='XR2')
plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][2], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][2], marker='^', color='g', label='XR3')
plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][9], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][9],label='XR11', marker='o', color='r')
plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][31], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][31], label='XR35', marker='>', color='orange')
plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][45], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][45],label='XR47', marker='s', color='purple')
plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][32], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][32],label='XR36', marker='d', color='k')
plt.legend(loc=4)



plotbpt(EL_3xmm.niiha,EL_3xmm.oiiihb,
         [],
         [],
         EL_m2.niiha,EL_m2.oiiihb,save=False)

plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][41], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][41], label='XR43', marker='s', color='k')
plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][58], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][58], label='XR60', marker='d', color='orange')
plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][61], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][61], label='XR65', marker='>', color='r')
plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][29], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][29], label='XR33', marker='*', color='k')
plt.scatter(xmm3eldiagmed_xrfilt_all.niiha[nonagn_3xmmm_all_xrfilt][46], xmm3eldiagmed_xrfilt_all.oiiihb[nonagn_3xmmm_all_xrfilt][46], label='XR48', marker='^', color='lightblue')
plt.legend(loc=4)
'''

def plotd4000_ssfr(bgx, bgy, save=False, filename='', title=None, leg=False, data=False, ybincolsty='r-', ax=None, linewid=2, label='', zorder=10):
    '''
    for doing d4000 against ssfr with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > 0) &( bgy < 3.5) & (bgx<-8)&(bgx > -14) )[0]
    nx = 0.3/0.001
    ny = 4/0.01
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny,ax=ax,bin_stat_y='mean', bin_y=True, size_y_bin=0.1, data=data, ybincolsty=ybincolsty, linewid=linewid,label=label, zorder=zorder)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([0.5,3])
    plt.xlim([-14,-8])
    plt.xlabel(r'log(sSFR)',fontsize=20)
    plt.ylabel(r'D$_{4000}$',fontsize=20)

    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()

    if save:
        ax.set(adjustable='box', aspect='equal')

        fig.savefig('plots/sfrmatch/png/diagnostic/d4000_ssfr'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/d4000_ssfr'+filename+'.pdf',dpi=250,bbox_inches='tight', format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/d4000_ssfr'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)

'''
plotd4000_ssfr(EL_m2.ssfr[agn_gsw_bptplus],EL_m2.d4000[agn_gsw_bptplus],save=True, filename= '_agn')    
plotd4000_ssfr(EL_m2.ssfr[nonagn_gsw_bptplus],EL_m2.d4000[nonagn_gsw_bptplus],save=True, filename= '_sf')
plotd4000_ssfr(EL_m2.ssfr_plus[nonagn_gsw_bptplusnii],EL_m2.d4000_plus[nonagn_gsw_bptplusnii],save=True, filename= '_sf_bptplusnii')
plotd4000_ssfr(EL_m2.ssfr_plus[agn_gsw_bptplusnii],EL_m2.d4000_plus[agn_gsw_bptplusnii],save=True, filename= '_agn_bptplusnii')
plotd4000_ssfr(EL_m2.ssfr_neither,EL_m2.d4000_neither,save=True, filename= '_neither')

save=False
massfracmin=0
massfracmax =0.1
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax1.set_xlim([-14,-8])
ax1.set_ylim([0.5,3])
ax1.set(aspect='equal', adjustable='box')

ax2 = fig.add_subplot(412, sharey = ax1, sharex=ax1)
ax3 = fig.add_subplot(413, sharey = ax1, sharex=ax1)
ax4 = fig.add_subplot(414, sharey = ax1)#, sharex=ax1)
ax2.set(adjustable='box', aspect='equal')
ax3.set(adjustable='box', aspect='equal')
ax4.set(adjustable='box', aspect='equal')
ax1.set_xticklabels([''])
ax4.set_xlim([-14,-8])
ax3.set_ylabel(r'D$_{4000}$')
ax1.set_ylabel(r'D$_{4000}$')
ax2.set_ylabel(r'D$_{4000}$')

plt.subplots_adjust(wspace=0, hspace=0)

massfracbin_bpt_agn = np.where((EL_m2.massfracgsw[agn_gsw_bptplus] >massfracmin)&(EL_m2.massfracgsw[agn_gsw_bptplus]<=massfracmax))[0]
massfracbin_bpt_sf = np.where((EL_m2.massfracgsw[nonagn_gsw_bptplus] >massfracmin)&(EL_m2.massfracgsw[nonagn_gsw_bptplus]<=massfracmax))[0]

massfracbin_plus_agn = np.where((EL_m2.massfracgsw_plus[agn_gsw_bptplusnii] >massfracmin)&(EL_m2.massfracgsw_plus[agn_gsw_bptplusnii]<=massfracmax))[0]
massfracbin_plus_sf = np.where((EL_m2.massfracgsw_plus[nonagn_gsw_bptplusnii] >massfracmin)&(EL_m2.massfracgsw_plus[nonagn_gsw_bptplusnii]<=massfracmax))[0]

massfracbin_neither = np.where((EL_m2.massfracgsw_neither >massfracmin)&(EL_m2.massfracgsw_neither<=massfracmax))[0]

ax=ax1
plotd4000_ssfr(EL_m2.ssfr[nonagn_gsw_bptplus][massfracbin_bpt_sf],EL_m2.d4000[nonagn_gsw_bptplus][massfracbin_bpt_sf],save=save, filename= '_sf_'+str(massfracmin)+'_'+str(massfracmax),ax=ax, ybincolsty='b--',  linewid=3, label='BPT SF')
plotd4000_ssfr(EL_m2.ssfr[agn_gsw_bptplus][massfracbin_bpt_agn],EL_m2.d4000[agn_gsw_bptplus][massfracbin_bpt_agn],save=save, filename= '_agn_'+str(massfracmin)+'_'+str(massfracmax), ax=ax, ybincolsty='k:', linewid=3, label='BPT AGN')    
plotd4000_ssfr(EL_m2.ssfr_neither[massfracbin_neither],EL_m2.d4000_neither[massfracbin_neither],save=save, filename= '_neither_'+str(massfracmin)+'_'+str(massfracmax),zorder=0, ax=ax, label='Unclassifiable', ybincolsty='r-', linewid=3) 
ax1.text(-13,1,r"0$<\frac{M_{*,\mathrm{fib}}}{M_{*, \mathrm {tot} }} <0.1$", fontsize=15)

massfracmin+=0.1
massfracmax+=0.1

massfracbin_bpt_agn = np.where((EL_m2.massfracgsw[agn_gsw_bptplus] >massfracmin)&(EL_m2.massfracgsw[agn_gsw_bptplus]<=massfracmax))[0]
massfracbin_bpt_sf = np.where((EL_m2.massfracgsw[nonagn_gsw_bptplus] >massfracmin)&(EL_m2.massfracgsw[nonagn_gsw_bptplus]<=massfracmax))[0]

massfracbin_plus_agn = np.where((EL_m2.massfracgsw_plus[agn_gsw_bptplusnii] >massfracmin)&(EL_m2.massfracgsw_plus[agn_gsw_bptplusnii]<=massfracmax))[0]
massfracbin_plus_sf = np.where((EL_m2.massfracgsw_plus[nonagn_gsw_bptplusnii] >massfracmin)&(EL_m2.massfracgsw_plus[nonagn_gsw_bptplusnii]<=massfracmax))[0]

massfracbin_neither = np.where((EL_m2.massfracgsw_neither >massfracmin)&(EL_m2.massfracgsw_neither<=massfracmax))[0]

ax=ax2
plotd4000_ssfr(EL_m2.ssfr[nonagn_gsw_bptplus][massfracbin_bpt_sf],EL_m2.d4000[nonagn_gsw_bptplus][massfracbin_bpt_sf],save=save, filename= '_sf_'+str(massfracmin)+'_'+str(massfracmax),ax=ax, ybincolsty='b--',  linewid=3, label='BPT SF')
plotd4000_ssfr(EL_m2.ssfr[agn_gsw_bptplus][massfracbin_bpt_agn],EL_m2.d4000[agn_gsw_bptplus][massfracbin_bpt_agn],save=save, filename= '_agn_'+str(massfracmin)+'_'+str(massfracmax), ax=ax, ybincolsty='k:', linewid=3, label='BPT AGN')    
plotd4000_ssfr(EL_m2.ssfr_neither[massfracbin_neither],EL_m2.d4000_neither[massfracbin_neither],save=save, filename= '_neither_'+str(massfracmin)+'_'+str(massfracmax),zorder=0, ax=ax, label='Unclassifiable', ybincolsty='r-', linewid=3) 
ax2.text(-13,1,r"0.1$<\frac{M_{*,\mathrm{fib}}}{M_{*, \mathrm {tot} }} <0.2$", fontsize=15)

massfracmin+=0.1
massfracmax+=0.1

massfracbin_bpt_agn = np.where((EL_m2.massfracgsw[agn_gsw_bptplus] >massfracmin)&(EL_m2.massfracgsw[agn_gsw_bptplus]<=massfracmax))[0]
massfracbin_bpt_sf = np.where((EL_m2.massfracgsw[nonagn_gsw_bptplus] >massfracmin)&(EL_m2.massfracgsw[nonagn_gsw_bptplus]<=massfracmax))[0]

massfracbin_plus_agn = np.where((EL_m2.massfracgsw_plus[agn_gsw_bptplusnii] >massfracmin)&(EL_m2.massfracgsw_plus[agn_gsw_bptplusnii]<=massfracmax))[0]
massfracbin_plus_sf = np.where((EL_m2.massfracgsw_plus[nonagn_gsw_bptplusnii] >massfracmin)&(EL_m2.massfracgsw_plus[nonagn_gsw_bptplusnii]<=massfracmax))[0]

massfracbin_neither = np.where((EL_m2.massfracgsw_neither >massfracmin)&(EL_m2.massfracgsw_neither<=massfracmax))[0]

ax=ax3
plotd4000_ssfr(EL_m2.ssfr[nonagn_gsw_bptplus][massfracbin_bpt_sf],EL_m2.d4000[nonagn_gsw_bptplus][massfracbin_bpt_sf],save=save, filename= '_sf_'+str(massfracmin)+'_'+str(massfracmax),ax=ax, ybincolsty='b--',  linewid=3, label='BPT SF')
plotd4000_ssfr(EL_m2.ssfr[agn_gsw_bptplus][massfracbin_bpt_agn],EL_m2.d4000[agn_gsw_bptplus][massfracbin_bpt_agn],save=save, filename= '_agn_'+str(massfracmin)+'_'+str(massfracmax), ax=ax, ybincolsty='k:', linewid=3, label='BPT AGN')    
plotd4000_ssfr(EL_m2.ssfr_neither[massfracbin_neither],EL_m2.d4000_neither[massfracbin_neither],save=save, filename= '_neither_'+str(massfracmin)+'_'+str(massfracmax),zorder=0, ax=ax, label='Unclassifiable', ybincolsty='r-', linewid=3) 
ax3.text(-13,1,r"0.2$<\frac{M_{*,\mathrm{fib}}}{M_{*, \mathrm {tot} }} <0.3$", fontsize=15)

massfracmin+=0.1
massfracmax+=0.1

massfracbin_bpt_agn = np.where((EL_m2.massfracgsw[agn_gsw_bptplus] >massfracmin)&(EL_m2.massfracgsw[agn_gsw_bptplus]<=massfracmax))[0]
massfracbin_bpt_sf = np.where((EL_m2.massfracgsw[nonagn_gsw_bptplus] >massfracmin)&(EL_m2.massfracgsw[nonagn_gsw_bptplus]<=massfracmax))[0]

massfracbin_plus_agn = np.where((EL_m2.massfracgsw_plus[agn_gsw_bptplusnii] >massfracmin)&(EL_m2.massfracgsw_plus[agn_gsw_bptplusnii]<=massfracmax))[0]
massfracbin_plus_sf = np.where((EL_m2.massfracgsw_plus[nonagn_gsw_bptplusnii] >massfracmin)&(EL_m2.massfracgsw_plus[nonagn_gsw_bptplusnii]<=massfracmax))[0]

massfracbin_neither = np.where((EL_m2.massfracgsw_neither >massfracmin)&(EL_m2.massfracgsw_neither<=massfracmax))[0]

ax=ax4
plotd4000_ssfr(EL_m2.ssfr[nonagn_gsw_bptplus][massfracbin_bpt_sf],EL_m2.d4000[nonagn_gsw_bptplus][massfracbin_bpt_sf],save=save, filename= '_sf_'+str(massfracmin)+'_'+str(massfracmax),ax=ax, ybincolsty='b--',  linewid=3, label='BPT SF')
plotd4000_ssfr(EL_m2.ssfr[agn_gsw_bptplus][massfracbin_bpt_agn],EL_m2.d4000[agn_gsw_bptplus][massfracbin_bpt_agn],save=save, filename= '_agn_'+str(massfracmin)+'_'+str(massfracmax), ax=ax, ybincolsty='k:', linewid=3, label='BPT AGN')    
plotd4000_ssfr(EL_m2.ssfr_neither[massfracbin_neither],EL_m2.d4000_neither[massfracbin_neither],save=save, filename= '_neither_'+str(massfracmin)+'_'+str(massfracmax),zorder=0, ax=ax, label='Unclassifiable', ybincolsty='r-', linewid=3) 
ax4.text(-13,1,r"0.3$<\frac{M_{*,\mathrm{fib}}}{M_{*, \mathrm {tot} }} <0.4$", fontsize=15)



'''
def mass_z(bgx,bgy,save=True,filename='',title=None, leg=False):
    '''
    for doing mass against z with sdss galaxies scatter plotted
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/mass_z'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/mass_z'+filename+'.pdf',dpi=250,bbox_inches='tight', format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/mass_z'+filename+'.eps',dpi=150,format='eps')
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



def plotoiiidispcomb(xvals, yvals, nonagnfilt, agnfilt, bgx1, bgy1, bgx2, bgy2, save=True, filename='',
                 title=None,leg=True):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
   
    fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    ax1 = fig.add_subplot(121)
    valbg = np.where((np.isfinite(bgx1) & (np.isfinite(bgy1))) &
            (bgy1 > 7.5) &( bgy1 < 15.5) & (bgx1<12.5)&(bgx1 > 8) )[0]
    nx = 4.5/0.01
    ny = 8/0.02
    plot2dhist(bgx1[valbg],bgy1[valbg],nx,ny)
    #ax1.set_title('BPT SF Background',fontsize=20)
    #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    if len(list(nonagnfilt))!=0:
        plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
            marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
    #if len(list(agnfilt)) !=0:
    #    plt.scatter(xvals[agnfilt],yvals[agnfilt],
    #        marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    plt.ylabel(r'log(L$_\mathrm{[OIII]}/\sigma^{4}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([7.5,15.5])
    plt.xlim([8,12.5])
    ax2 = fig.add_subplot(122, sharey = ax1, sharex=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    #plt.ylabel(r'log(L$_\mathrm{[OIII]/SFR}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    valbg = np.where((np.isfinite(bgx2) & (np.isfinite(bgy2))) &
            (bgy2 > 7.5) &( bgy2 < 15.5) & (bgx2<12.5)&(bgx2 > 8) )[0]
    nx = 4.5/0.045
    ny = 8/0.08
    plot2dhist(bgx2[valbg],bgy2[valbg],nx,ny)
    #ax2.set_title('BPT-AGN Background',fontsize=20)
    #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.subplots_adjust(wspace=0, hspace=0)

    #if len(list(nonagnfilt))!=0:
    #    plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
    #        marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
    if len(list(agnfilt)) !=0:
        plt.scatter(xvals[agnfilt],yvals[agnfilt],
            marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    if leg:
        plt.legend(fontsize=15, frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/OIIILumdisp_scat'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILumdisp_scat'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILumdisp_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    return fig
'''
fig = plotoiiidispcomb(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum/(xmm3eldiagmed_xrfilt.vdisp*1e5)**4),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m2.mass[nonagn_gsw], np.log10(EL_m2.oiiilum[nonagn_gsw]/(EL_m2.vdisp[nonagn_gsw]*1e5)**4),EL_m2.mass[agn_gsw], np.log10(EL_m2.oiiilum[agn_gsw]/(EL_m2.vdisp[agn_gsw]*1e5)**4),
        save=False,leg=False)
'''
def plotssfrm(bgx,bgy,ccode=[], ccodename='', save=False,filename='', ax=None, fig=None,title=None, leg=False,  nx=500, ny=600):
    '''
    for doing ssfrmass diagram with sdss galaxies scatter plotted
    '''
    if not ax and not fig:
        fig = plt.figure()
        ax=fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -14) &( bgy < -8) & (bgx<12.5)&(bgx > 7.5) )[0]
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ccode=ccode, ccodename=ccodename, ax=ax, fig=fig)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    ax.set_ylabel(r'log(sSFR)',fontsize=20)
    ax.set_xlabel(r'log(M$_{\mathrm{*}})$',fontsize=20)
    ax.set_xlim([7.5,12.5])
    ax.set_ylim([-14,-7.5])

    if title:
        plt.title(title,fontsize=30)
    if leg:
        plt.legend(fontsize=15,frameon = False, loc=3)#, bbox_to_anchor=(-0.04, .05))    
        
    plt.tight_layout()
    #ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/ssfr_mass'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/ssfr_mass'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/ssfr_mass'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:

        plt.show()
def plot3panel_ssfrm(x1,x2,x3,y1,y2,y3,ccode1=[], ccode2=[], ccode3=[], save=False,
                   filename='',minx=-2, maxx=1.5, miny=-1.3, maxy=2,
                   nx=300, ny=240, ccodename='', ms=True):
    fig = plt.figure(figsize=(8,8))
  
    ax1 = fig.add_subplot(311)
    ax1.set_xlim([minx,maxx])
    ax1.set_ylim([miny,maxy])
    #ax1.set(aspect='equal', adjustable='box')
    ax2 = fig.add_subplot(312, sharey = ax1, sharex=ax1)
    ax3 = fig.add_subplot(313, sharey = ax1, sharex=ax1)
    #ax2.set(adjustable='box', aspect='equal')
    #ax3.set(adjustable='box', aspect='equal')
    plotssfrm(x1, y1, ccode=ccode1, save=False, ax=ax1,fig=fig, nx=nx, ny=ny, ccodename=ccodename)
    
    #ax2.text(-1.85, 1.5, r'$\Delta$log(sSFR)$>-0.7$', fontsize=15)
    plotssfrm(x2, y2, ccode=ccode2, save=False,  ax=ax2,fig=fig,  nx=nx, ny=ny, ccodename=ccodename)

    #ax3.text(-1.85, 1.5, r'$\Delta$log(sSFR)$\leq-0.7$', fontsize=15)
    plotssfrm(x3, y3, ccode=ccode3, save=False,  ax=ax3,fig=fig, nx=nx, ny=ny, ccodename=ccodename)    
    if ms:
        m_ssfr = -0.4597
        b_ssfr = -5.2976
        x=np.arange(7.5, 12.5, 0.1)
        ax1.plot(x,x*m_ssfr+b_ssfr,'k', label='Main Sequence')
        ax1.plot(x,x*m_ssfr+b_ssfr-0.7,'k-.', label='Main Sequence Cutoff')
        ax2.plot(x,x*m_ssfr+b_ssfr,'k', label='Main Sequence')
        ax2.plot(x,x*m_ssfr+b_ssfr-0.7,'k-.', label='Main Sequence Cutoff')
        ax3.plot(x,x*m_ssfr+b_ssfr,'k', label='Main Sequence')
        ax3.plot(x,x*m_ssfr+b_ssfr-0.7,'k-.', label='Main Sequence Cutoff')

    #ax1.set_xticklabels([''])
    #ax2.set_xticklabels([''])
    #ax3.set_xticklabels(['7','8','9','10','11','12'])

    #ax3.set_ylabel('')
    #ax1.set_ylabel('')
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    plt.subplots_adjust(wspace=0, hspace=0)
    if save:
        fig.savefig('plots/sfrmatch/pdf/'+filename+'.pdf', bbox_inches='tight', dpi=250, format='pdf')
        fig.savefig('plots/sfrmatch/png/'+filename+'.png', bbox_inches='tight', dpi=250)
        
'''
plot3panel_ssfrm(sfrm_gsw2.mass_sing_sing,sfrm_gsw2.mass_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.mass_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.ssfr_sing, sfrm_gsw2.ssfr_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.ssfr_sing_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/ssfrm_3panel', nobj=False,save=True)    

plot3panel_ssfrm(sfrm_gsw2.mass_bpt,sfrm_gsw2.mass_plus,
               sfrm_gsw2.mass_neither,
               sfrm_gsw2.ssfr_bpt, sfrm_gsw2.ssfr_plus,
               sfrm_gsw2.ssfr_neither, ccode1= sfrm_gsw2.z_bpt,ccode2=sfrm_gsw2.z_plus, ccode3=sfrm_gsw2.z_neither,
               filename='diagnostic/ssfrm_by_group_z', save=True, ms=False, nx=20, ny=25, ccodename='z')    

plotssfrm(bpt_sfrm_gsw2.mass, bpt_sfrm_gsw2.ssfr, save=False,filename='_fluxsub_bpt')
plotssfrm(sfrm_gsw2.mass_bpt[sfrm_gsw2.bpt_sn_filt], sfrm_gsw2.ssfr_bpt[sfrm_gsw2.bpt_sn_filt], save=False,filename='_fluxsub_bpt_snfilt',nx=200, ny=300)
plotssfrm(sfrm_gsw2.mass_plus, sfrm_gsw2.ssfr_plus, save=False,filename= '_fluxsub_plus', nx=200, ny=300)
plotssfrm(sfrm_gsw2.mass_plus[sfrm_gsw2.bpt_plus_sn_filt], sfrm_gsw2.ssfr_plus[sfrm_gsw2.bpt_plus_sn_filt], save=True,filename='_fluxsub_plus_snfilt')
plotssfrm(sfrm_gsw2.mass_neither, sfrm_gsw2.ssfr_neither, save=True, filename='_fluxsub_neither')
plotssfrm(sfrm_gsw2.mass_neither[sfrm_gsw2.bpt_neither_sn_filt], sfrm_gsw2.ssfr_neither[sfrm_gsw2.bpt_neither_sn_filt], save=True,filename='_fluxsub_neither_filt')

x= np.arange(7.5,12.5,0.1)

plt.plot(x,x*m_ssfr+b_ssfr,'k', label='Main Sequence')
plt.plot(x,x*m_ssfr+b_ssfr-0.7,'k-.', label='Main Sequence Cutoff')
plotssfrm(sfrm_gsw2.mass_sing, sfrm_gsw2.ssfr_sing, save=False,filename='_fluxsub_bpt_snfilt')

plotssfrm(m2Cat_GSW_3xmm.allmass[mpa_spec_allm2.make_prac], (m2Cat_GSW_3xmm.allsfr-m2Cat_GSW_3xmm.allmass)[mpa_spec_allm2.make_prac], save=False, leg=True)




fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(121)plot
ax2 = fig.add_subplot(122)
plotssfrm(m2Cat_GSW_3xmm.allmass[mpa_spec_allm2.make_prac], (m2Cat_GSW_3xmm.allsfr-m2Cat_GSW_3xmm.allmass)[mpa_spec_allm2.make_prac], save=False, ax=ax1)
ax1.plot(x,x*m_ssfr+b_ssfr,'k', label='Main Sequence')
ax1.plot(x,x*m_ssfr+b_ssfr-0.7,'k-.', label='Main Sequence Cutoff')
ax1.legend(loc=1)
ax1.set_aspect(1/2)
plotssfrm(sfrm_gsw2.mass_sing, sfrm_gsw2.ssfr_sing, save=False,filename='_fluxsub_bpt_snfilt', ax=ax2)
ax2.plot(x,x*m_ssfr+b_ssfr,'k', label='Main Sequence')
ax2.plot(x,x*m_ssfr+b_ssfr-0.7,'k-.', label='Main Sequence Cutoff')
ax2.set_aspect(1/2)
ax1.text(8, -13., 'All GSW-M2', fontsize=15)
ax2.text(8, -13., 'BPT AGN', fontsize=15)

ax2.set_ylabel('')
ax2.set_yticklabels('')
plt.subplots_adjust(wspace=0, hspace=0)


fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plotssfrm(m2Cat_GSW_3xmm.allmass[mpa_spec_allm2.make_prac], (m2Cat_GSW_3xmm.allsfr-m2Cat_GSW_3xmm.allmass)[mpa_spec_allm2.make_prac], save=False, ax=ax1)
ax1.plot(x,x*m_ssfr+b_ssfr,'k', label='Main Sequence')
ax1.plot(x,x*m_ssfr+b_ssfr-0.7,'k-.', label='Main Sequence Cutoff')

ax1.legend(loc=3)
ax1.set_aspect(1/2)

plotssfrm(sfrm_gsw2.mass_sing, sfrm_gsw2.ssfr_sing, save=False,filename='_fluxsub_bpt_snfilt', ax=ax2)
ax2.plot(x,x*m_ssfr+b_ssfr,'k', label='Main Sequence')
ax2.plot(x,x*m_ssfr+b_ssfr-0.7,'k-.', label='Main Sequence Cutoff')
ax2.set_aspect(1/2)

ax1.text(11.1, -8.7, 'All GSW-M2', fontsize=15)
ax2.text(11.2, -8.7, 'BPT AGN', fontsize=15)

ax1.set_xlabel('')
ax1.set_xticklabels('')
plt.subplots_adjust(wspace=0, hspace=0)
'''






def plotbpt_sii(bgx,bgy,ccode= [], save=False,filename='',labels=True, title=None,
                minx=-1.2, maxx=0.75, miny=-1.2, maxy=2, nx=300, ny=240, 
                ccodename='', ccodelim=[], nobj=True, ax= None, fig=None):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > miny) &( bgy < maxy) & (bgx<maxx)&(bgx > minx) )[0]
    if len(ccode) !=0:    
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ccode=ccode[valbg], ccodename=ccodename, ccodelim = ccodelim,ax=ax, fig=fig)
    else:
        plot2dhist(bgx[valbg], bgy[valbg], nx,ny, ax=ax, fig=fig)
    ax.plot(np.log10(xline2_agn),np.log10(yline2_agn),'k--')#,label='AGN Line')
    ax.plot(np.log10(xline2_linersy2),np.log10(yline2_linersy2),c='k',ls='-.')#,label='LINER, Seyfert 2')
    if labels:       
        ax.text(.4,-.5,'LINER',fontsize=15)
        ax.text(-1.5,1.1,'Seyfert',fontsize=15)
        ax.text(-1.4,-1,'SF',fontsize=15)

    ax.set_ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    ax.set_xlabel(r'log([SII]/H$\rm \alpha$)',fontsize=20)
    if title:
        plt.title(title,fontsize=30)
    ax.set(adjustable='box', aspect='equal')
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/SII_OIII_scat'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/SII_OIII_scat'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/SII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()


'''
#postsub
plotbpt_sii(np.log10(sfrm_gsw2.siiflux_sub/sfrm_gsw2.halpflux_sub)[sfrm_gsw2.good_oiii_hb_sii_ha_sub_bpt],  
              sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.good_oiii_hb_sii_ha_sub_bpt],nobj=False,
              filename='_fluxsub_bpt', save=True, nx=100, ny=100)
plotbpt_sii(np.log10(sfrm_gsw2.siiflux_sub_plus/sfrm_gsw2.halpflux_sub_plus)[sfrm_gsw2.good_oiii_hb_sii_ha_sub_plus],  
              sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.good_oiii_hb_sii_ha_sub_plus],nobj=False,
              filename='_fluxsub_plus', save=True, nx=100, ny=100)
plotbpt_sii(np.log10(sfrm_gsw2.siiflux_sub_neither/sfrm_gsw2.halpflux_sub_neither)[sfrm_gsw2.good_oiii_hb_sii_ha_sub_neither],  
              sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.good_oiii_hb_sii_ha_sub_neither],nobj=False,
              filename='_fluxsub_neither', save=True)

#presub
plotbpt_sii(np.log10(sfrm_gsw2.siiflux/sfrm_gsw2.halpflux)[sfrm_gsw2.good_oiii_hb_sii_ha_bpt],  
              sfrm_gsw2.log_oiii_hbeta[sfrm_gsw2.good_oiii_hb_sii_ha_bpt],nobj=False,
              filename='_bpt', save=True, nx=100, ny=100)
plotbpt_sii(np.log10(sfrm_gsw2.siiflux_plus/sfrm_gsw2.halpflux_plus)[sfrm_gsw2.good_oiii_hb_sii_ha_plus],  
              sfrm_gsw2.log_oiii_hbeta_plus[sfrm_gsw2.good_oiii_hb_sii_ha_plus],nobj=False,
              filename='_plus', save=True, nx=100, ny=100)
plotbpt_sii(np.log10(sfrm_gsw2.siiflux_neither/sfrm_gsw2.halpflux_neither)[sfrm_gsw2.good_oiii_hb_sii_ha_neither],  
              sfrm_gsw2.log_oiii_hbeta_neither[sfrm_gsw2.good_oiii_hb_sii_ha_neither],nobj=False,
              filename='neither', save=True, nx=100, ny=100)

'''
def plotbpt_oi(bgx,bgy,save=False,filename='',
               title=None,alph=0.1,ccode=[],ccodegsw=[],cont=None,
               ccodename='', minx =-2.2, maxx=0, miny=-1.2, maxy=2, nx=300, ny=240, nobj=False, ax= None, fig=None):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > miny) &( bgy < maxy) & (bgx<maxx)&(bgx > minx) )[0]
    if len(ccode) !=0:    
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ccode=ccode[valbg], ccodename=ccodename, ccodelim = ccodelim,ax=ax, fig=fig)
    else:
        plot2dhist(bgx[valbg], bgy[valbg], nx,ny, ax=ax, fig=fig)
    if save:
        fig = plt.figure()
    ax.plot(np.log10(xline3_linersy2),np.log10(yline3_linersy2),c='k',ls='-.')
    ax.plot(np.log10(xline3_agn), np.log10(yline3_agn),'k--')
    
    ax.text(-.3,-0.7,'LINER',fontsize=15)
    ax.text(-2,1.1,'Seyfert',fontsize=15)
    ax.text(-2.2,-1.1,'SF',fontsize=15)
    ax.set_ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    ax.set_xlabel(r'log([OI]/H$\rm \alpha$)',fontsize=20)
    ax.set_ylim([miny-0.1,maxy])
    ax.set_xlim([minx-0.1, maxx+0.1])
    ax.set(adjustable='box', aspect='equal')
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:

        fig.savefig('plots/sfrmatch/png/diagnostic/OI_OIII_scat'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/png/diagnostic/OI_OIII_scat'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/png/diagnostic/OI_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()

'''
plotbpt_oi(xmm3eldiagmed_xrfilt.oiha,xmm3eldiagmed_xrfilt.oiiihb[xmm3eldiagmed_xrfilt.vo87_2_filt],
         nonagn_3xmmm_xrfiltvo87_2,  agn_3xmmm_xrfiltvo87_2,
         EL_m2.oiha,EL_m2.oiiihb[EL_m2.vo87_2_filt],save=True)
plotbpt_oi(xmm3eldiagmed_xrfilt.oiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt,
         EL_m2.oiha,EL_m2.oiiihb,save=True)
plotbpt_oi(xmm3eldiagmed_xrfilt.oiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt,
         EL_m2.oiha,EL_m2.oiiihb,save=True, filename='xrfrac', ccode=contaminations_xmm3_2)


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
    plt.text(9.5,-1,'SF')
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    #plt.title('Mass-Excitation')
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()

    if save:
        plt.savefig('plots/xmm3/png/diagnostic/Mex_OIII'+filename+'.png',dpi=250,bbox_inches='tight')
        plt.savefig('plots/xmm3/pdf/diagnostic/Mex_OIII'+filename+'.pdf',dpi=250,bbox_inches='tight')
        plt.close()
    else:
        plt.show()
'''
plot_mex(xmm3eldiagmed_xrfilt.mass, xmm3eldiagmed_xrfilt.oiiihb,nonagn_3xmmm_xrfilt, 
         agn_3xmmm_xrfilt, EL_m2.mass, EL_m2.oiiihb)
'''


'''
Below are X-ray plots

'''
# %% X-ray

lxmsfrm = Plot('lxmsfrm')
lxsfr = Plot('lxsfr')
'''
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
    plt.text(42.8,-2.2,'X-ray AGN', fontsize=15,rotation=45)
    plt.xlim([37.5,44.5])
    plt.ylim([-2.5,4.5])
'''
lxoiii = Plot('lxoiii')
'''    plt.xlabel(r'L$_{\mathrm{X}}$', fontsize=20)
    plt.ylabel(r'L$_{\mathrm{[OIII]}}$', fontsize=20)
    plt.ylim([37, 45])
    plt.xlim([38,46])
    plt.plot([38,46], [37,45], color='k')
    ax.set(adjustable='box', aspect='equal')
'''
lxsfr_oiii = Plot('lxsfr_oiii')
'''    plt.xlabel(r'L$_{\mathrm{X}}$-L$_{\mathrm{X,SFR}}$', fontsize=20)
    plt.ylabel(r'L$_{\mathrm{[OIII]}}$', fontsize=20)
    plt.ylim([37, 45])
    plt.xlim([-2,8])
    plt.plot([38,46], [37,45], color='k')
'''
lxfibsfr =Plot('lxfibsfr')

lx_z = Plot('lx_z')
'''
    plt.xlabel('z')
    plt.xlim([0,0.3])
    plt.ylim([39,46])
    plt.ylabel(r'L$_{\mathrm{X}}$')
'''
lx_hardfull = Plot('lx_hardfull')
'''
    plt.ylabel(r'L$_{0.5-10\ keV }$')
    plt.xlabel(r'L$_{2-10\ keV}$')
    #plt.legend()
    plt.ylim([40.5,43.5])
    plt.xlim([40.5,43.5])
'''
lsfrrelat = {'soft': [r'SFR/M$_{*} = 1.39\cdot 10^{-40}$ L$_{\rm x}/$M$_{*}$', r'SFR = $1.39\cdot 10^{-40}$ L$_{\rm x}$',logsfrsoft],
             'hard': [r'SFR/M$_{*} = 1.26\cdot 10^{-40} $L$_{\rm x}$/M$_{*}$', r'SFR = $1.26\cdot 10^{-40}$ L$_{\rm x}$',logsfrhard],
             'full': [r'SFR/M$_{*} = 0.66\cdot 10^{-40}$ L$_{\rm x}$/M$_{*}$',r'SFR = $0.66\cdot 10^{-40}$ L$_{\rm x}$', logsfrfull]  }





def plot_lxsfr(xraysfr, label, save=False, filtagn=[], filtnonagn=[],filename='',weakem=False,scat=False, nofilt=False, fibssfr=[]):
    '''
    Plots star-formation rate versus X-ray luminosity and
    color-codes a region in bottom right where X-ray AGN are.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
    plt.legend(frameon=False,fontsize=15,loc=2,bbox_to_anchor = (-0.00, 0.95))
    plt.text(42.8,-2.2,'X-ray AGN', fontsize=15,rotation=45)
    plt.xlim([37.5,44.5])
    plt.ylim([-2.5,4.5])
    ax.set(adjustable='box', aspect='equal')
    plt.tight_layout() 
    if save:
        plt.savefig("plots/xmm3/png/xraylum/"+label+filename+"_lx_vs_sfr.png",dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)

        plt.savefig("plots/xmm3/pdf/xraylum/"+label+filename+"_lx_vs_sfr.pdf",dpi=250,bbox_inches='tight')
        plt.savefig("plots/xmm3/eps/xraylum/"+label +filename+"_lx_vs_sfr.eps",format='eps',dpi=250,bbox_inches='tight')
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

'''
plotlx_z(fullxray_xmm_dr7, save=True, filename='full')

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
