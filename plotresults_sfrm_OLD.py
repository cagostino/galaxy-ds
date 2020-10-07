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
    def __init__(self, name, minx=0, maxx=0, miny=0, maxy=0, xlabel='', ylabel='', nx=500, ny=500):
        self.name = name
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.nx = nx
        self.ny = ny
    def make2ddensplot(self,x,y):
        plot2dhist(x,y, self.nx, self.ny, minx=self.minx, maxx=self.maxx, miny=self.miny, maxy=self.maxy, xlabel=self.xlabel, ylabel=self.ylabel)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)                
ssfrm =  Plot('ssfrm', minx=7.5, maxx=12.5, miny=-14, maxy=-7.5,xlabel=r'log(M$_{\mathrm{*}})$',ylabel=r'log(sSFR)', nx=500,ny=600)
bpt =  Plot('ssfrm',  minx=-2, maxx=1, miny=-1.2, maxy = 1.2, xlabel = r'log([NII]/H$\rm \alpha$)',ylabel=r'log([OIII]/H$\rm \beta$)', nx=300,ny=240)
disp_sfr = Plot('disp_sfr')

#plt.rc('text',usetex=False)
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
    if nan:
        fin = np.where((np.isfinite(x)) &(np.isfinite(y) ))[0]
        x= np.copy(x[fin])
        y=np.copy(y[fin])
    hist, xedges, yedges = np.histogram2d(x,y,bins = (int(nx),int(ny)))
    extent= [np.min(xedges),np.max(xedges),np.min(yedges),np.max(yedges)]
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
def plot3d(x,y,z):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, s=0.1, alpha=0.1)
    plt.show()
def hist_displacements_cell(disp_x, disp_y, coordpair, save = False, bins = 20, filename = ''):
    avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    disps = np.sqrt(disp_x**2+disp_y**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(disps, bins=bins, histtype='step', color='k',
             label='Avg = '+str(round(avg_disp,2))+ 
             ' Med = '+str(round(med_disp,2))+' ,(x, y) = ('+str(round(coordpair[0],2))+
             ' , '+str(round(coordpair[1],2)) +')' )
    hst, _bns = np.histogram(disps, bins=bins)
    plt.ylim([0, np.max(hst)+1*np.max(hst)/6])
    plt.xlabel('Displacements', fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.legend(frameon=False,fontsize=12)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    if save:
        plt.savefig('plots/sfrmatch/png/diagnostic/hist_displacements_'+ filename+'.png', dpi=250,bbox_inches='tight')
        plt.savefig('plots/sfrmatch/pdf/diagnostic/hist_displacements_'+ filename+'.pdf', dpi=250,bbox_inches='tight')
        #plt.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/histsf_diffs_'+filename+'.eps', dpi=150,format='eps')
        plt.close()
def hist_displacements_by_type(disps_set, save = False, bins = 20, filename = '', rang=[]):
    disps_x = np.array([])
    disps_y = np.array([])
    for i in range(len(disps_set)):
        disps_x = np.append(disps_x, disps_set[i][8])
        disps_y = np.append(disps_y, disps_set[i][9])
    disps = np.sqrt(disps_x**2+disps_y**2)        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if len(rang)!=0:
        valid = np.where(np.isfinite(disps) &(disps<rang[1]))[0]
        avg_disp =np.sqrt(np.mean(disps_x[valid])**2+np.mean(disps_y[valid])**2)
        med_disp = np.sqrt(np.median(disps_x[valid])**2+np.median(disps_y[valid])**2)
        plt.hist(disps[valid], bins=bins, histtype='step', color='k',
             label='Average = '+str(round(avg_disp,2))+ 
             ', Median = '+str(round(med_disp,2)), range=rang )
        hst, _bns = np.histogram(disps[valid], bins=bins, range=rang)

    else:
        valid = np.where(np.isfinite(disps))[0]
        avg_disp =np.sqrt(np.mean(disps_x[valid])**2+np.mean(disps_y[valid])**2)
        med_disp = np.sqrt(np.median(disps_x[valid])**2+np.median(disps_y[valid])**2)
        plt.hist(disps[valid], bins=bins, histtype='step', color='k',
             label='Average = '+str(round(avg_disp,2))+ 
             ', Median = '+str(round(med_disp,2)) )
        hst, _bns = np.histogram(disps[valid], bins=bins)
    print(avg_disp, med_disp)
    
    plt.ylim([0, np.max(hst)+1*np.max(hst)/6])
    plt.xlim([-0.05,2])
    plt.xlabel('Displacements', fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.legend(frameon=False,fontsize=15)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    if save:
        plt.savefig('plots/sfrmatch/png/distmet/hist_displacements_'+ filename+'.png', dpi=250,bbox_inches='tight')
        plt.savefig('plots/sfrmatch/pdf/distmet/hist_displacements_'+ filename+'.pdf', dpi=250,bbox_inches='tight')
        #plt.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/histsf_diffs_'+filename+'.eps', dpi=150,format='eps')
        plt.close()
def hist_displacements_all(disps_set1, disps_set2, disps_set3, save = False, bins = 20, filename = '', rang=[]):
    disps_x = np.array([])
    disps_y = np.array([])
    for i in range(len(disps_set1)):
        disps_x = np.append(disps_x, disps_set1[i][8])
        disps_y = np.append(disps_y, disps_set1[i][9])
    for i in range(len(disps_set2)):
        disps_x = np.append(disps_x, disps_set2[i][8])
        disps_y = np.append(disps_y, disps_set2[i][9])
    for i in range(len(disps_set3)):
        disps_x = np.append(disps_x, disps_set3[i][8])
        disps_y = np.append(disps_y, disps_set3[i][9])
    disps = np.sqrt(disps_x**2+disps_y**2)        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if len(rang)!=0:
        valid = np.where(np.isfinite(disps) &(disps<rang[1]))[0]
        avg_disp =np.sqrt(np.mean(disps_x[valid])**2+np.mean(disps_y[valid])**2)
        med_disp = np.sqrt(np.median(disps_x[valid])**2+np.median(disps_y[valid])**2)
        plt.hist(disps[valid], bins=bins, histtype='step', color='k',
             label='Average = '+str(round(avg_disp,2))+ 
             ', Median = '+str(round(med_disp,2)), range=rang )
        hst, _bns = np.histogram(disps[valid], bins=bins, range=rang)
    else:
        valid = np.where(np.isfinite(disps))[0]
        avg_disp =np.sqrt(np.mean(disps_x[valid])**2+np.mean(disps_y[valid])**2)
        med_disp = np.sqrt(np.median(disps_x[valid])**2+np.median(disps_y[valid])**2)
        plt.hist(disps[valid], bins=bins, histtype='step', color='k',
             label='Average = '+str(round(avg_disp,2))+ 
             ', Median = '+str(round(med_disp,2)) )
        hst, _bns = np.histogram(disps[valid], bins=bins)
    print(avg_disp, med_disp)
    plt.ylim([0, np.max(hst)+1*np.max(hst)/6])
    plt.xlim([-0.05,2])
    plt.xlabel('Displacements', fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.legend(frameon=False,fontsize=15)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        plt.savefig('plots/sfrmatch/png/distmet/hist_displacements_'+ filename+'.png', dpi=250,bbox_inches='tight')
        plt.savefig('plots/sfrmatch/pdf/distmet/hist_displacements_'+ filename+'.pdf', dpi=250,bbox_inches='tight')
        #plt.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/histsf_diffs_'+filename+'.eps', dpi=150,format='eps')
        plt.close()
        
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

def disp_vs_av(disp_x, disp_y, av, save=False, filename='', nx = 50, ny=50 ): 
    plt.xlim([-6, 6])
    plt.ylim([0, 2])
    plt.xlabel(r'A$_{\mathrm{V}}$', fontsize=20)
    plt.ylabel('Displacement', fontsize=20)
    ax.set(adjustable='box', aspect='equal')
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/disp_vs_av' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/disp_vs_av'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/disp_vs_av'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
disp_vs_av(sfrm_gsw2.offset_nii_halpha, sfrm_gsw2.offset_oiii_hbeta, sfrm_gsw2.av_bpt_agn, filename='_bpt', save=False)
disp_vs_av(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.av_plus_agn, filename='_plus', save=False)
disp_vs_av(sfrm_gsw2.offset_nii_halpha_neither, sfrm_gsw2.offset_oiii_hbeta_neither, sfrm_gsw2.av_neither_agn, filename='_unclassified', save=False)
disp_vs_av(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.sfr_sing, filename='_all', save=True)
'''

def disp_vs_mass(disp_x, disp_y, mass, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disps = np.sqrt(disp_x**2+disp_y**2)
    val = np.where((disps <2) &(mass >8) &(mass<12))[0]
    
    plot2dhist(mass[val], disps[val], nx,ny, bin_y=False, bin_stat_y='mean', nan=True)
    plt.xlim([8, 12])
    plt.ylim([0, 2])
    plt.xlabel(r'log(M$_{\mathrm{*}}$)', fontsize=20)
    plt.ylabel('Displacement', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/disp_vs_mass' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/disp_vs_mass'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/disp_vs_mass'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
disp_vs_mass(sfrm_gsw2.offset_nii_halpha, sfrm_gsw2.offset_oiii_hbeta, sfrm_gsw2.mass_bpt, filename='_bpt', save=False)
disp_vs_mass(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.mass_plus, filename='_plus', save=False)
disp_vs_mass(sfrm_gsw2.offset_nii_halpha_neither, sfrm_gsw2.offset_oiii_hbeta_neither, sfrm_gsw2.mass_neither, filename='_unclassified', save=False)

disp_vs_mass(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.sfr_sing, filename='_all', save=True)
'''

def disp_vs_fibmass(disp_x, disp_y, mass, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disps = np.sqrt(disp_x**2+disp_y**2)
    val = np.where((disps <2) &(mass >8) &(mass<12))[0]
    
    plot2dhist(mass[val], disps[val], nx,ny, bin_y=False, bin_stat_y='mean', nan=True)
    plt.xlim([8, 12])
    plt.ylim([0, 2])
    plt.xlabel(r'log(M$_{\mathrm{*}}$)', fontsize=20)
    plt.ylabel('Displacement', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/disp_vs_fibmass' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/disp_vs_fibmass'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/disp_vs_fibmass'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)

'''
disp_vs_fibmass(sfrm_gsw2.offset_nii_halpha, sfrm_gsw2.offset_oiii_hbeta, sfrm_gsw2.fibmass_bpt, filename='_bpt', save=False)
disp_vs_fibmass(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.fibmass_plus, filename='_plus', save=False)
disp_vs_fibmass(sfrm_gsw2.offset_nii_halpha_neither, sfrm_gsw2.offset_oiii_hbeta_neither, sfrm_gsw2.fibmass_neither, filename='_unclassified', save=False)

disp_vs_fibmass(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.sfr_sing, filename='_all', save=True)
'''

def linregress_test(offset_x, offset_y, mass, sfr, av, bptx, bpty):
    X = np.vstack([mass, sfr, av, bptx, bpty])
    y = np.sqrt(offset_x**2+offset_y**2)
    reg = LinearRegression().fit(X, y)
'''
linregress_test(sfrm_gsw2.offset_nii_halpha, sfrm_gsw2.offset_oiii_hbeta, sfrm_gsw2.mass_bpt, sfrm_gsw2.sfr_bpt, sfrm_gsw2.av_bpt_agn, sfrm_Gsw2.
'''
def disp_vs_sfr(disp_x, disp_y, sfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disps = np.sqrt(disp_x**2+disp_y**2)
    val = np.where((disps <2) &(sfr <3) &(sfr>-3))[0]
    
    plot2dhist(sfr[val], disps[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    plt.xlim([-3, 3])
    plt.ylim([-1, 1])
    plt.xlabel('log(SFR), ', fontsize=20)
    plt.ylabel('Displacement', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/disp_vs_sfr' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/disp_vs_sfr'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/disp_vs_sfr'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
disp_vs_sfr(sfrm_gsw2.offset_nii_halpha, sfrm_gsw2.offset_oiii_hbeta, sfrm_gsw2.sfr_bpt, filename='_bpt', save=True)
disp_vs_sfr(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.sfr_plus, filename='_plus', save=True)
disp_vs_sfr(sfrm_gsw2.offset_nii_halpha_neither, sfrm_gsw2.offset_oiii_hbeta_neither, sfrm_gsw2.sfr_neither, filename='_unclassified', save=True)

disp_vs_sfr(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.sfr_sing, filename='_all', save=True)

disp_vs_sfr(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.sfr_sing, filename='_all', save=False)
disp_vs_sfr(sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj], sfrm_gsw2.sfr_sing[sfrm_gsw2.high_ssfr_obj], filename='_highssfr', save=False)
disp_vs_sfr(sfrm_gsw2.offset_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj], sfrm_gsw2.offset_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj], sfrm_gsw2.sfr_sing[sfrm_gsw2.low_ssfr_obj], filename='_lowssfr', save=False)


'''

def disp_vs_fibsfr(disp_x, disp_y, fibsfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disps = np.sqrt(disp_x**2+disp_y**2)
    val = np.where((disps <2) &(fibsfr <3) &(fibsfr>-4))[0]
    plot2dhist(fibsfr[val], disps[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    plt.xlim([-4, 3])
    plt.ylim([-1, 1])
    plt.xlabel(r'log(SFR$_{\mathrm{fib}}$), ', fontsize=20)
    plt.ylabel('Displacement', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')        
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/disp_vs_fibsfr' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/disp_vs_fibsfr'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/disp_vs_fibsfr'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
disp_vs_fibsfr(sfrm_gsw2.offset_nii_halpha, sfrm_gsw2.offset_oiii_hbeta, sfrm_gsw2.fibsfr_bpt, filename='_bpt', save=True)
disp_vs_fibsfr(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.fibsfr_plus, filename='_plus', save=True)
disp_vs_fibsfr(sfrm_gsw2.offset_nii_halpha_neither, sfrm_gsw2.offset_oiii_hbeta_neither, sfrm_gsw2.fibsfr_neither, filename='_unclassified', save=True)


'''

def disp_vs_fibsfr_match(disp_x, disp_y, fibsfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disps = np.sqrt(disp_x**2+disp_y**2)
    val = np.where((disps <2) &(fibsfr <3) &(fibsfr>-4))[0]
    
    plot2dhist(fibsfr[val], disps[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    plt.xlim([-4, 3])
    plt.ylim([-1, 1])
    plt.xlabel(r'log(SFR$_{\mathrm{fib}}$), ', fontsize=20)
    plt.ylabel('Displacement', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/disp_vs_fibsfr_match' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/disp_vs_fibsfr_match'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/disp_vs_fibsfr_match'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
disp_vs_fibsfr_match(sfrm_gsw2.offset_nii_halpha, sfrm_gsw2.offset_oiii_hbeta, sfrm_gsw2.fibsfr_match_bpt, filename='_bpt', save=True)
disp_vs_fibsfr_match(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.fibsfr_match_plus, filename='_plus', save=True)
disp_vs_fibsfr_match(sfrm_gsw2.offset_nii_halpha_neither, sfrm_gsw2.offset_oiii_hbeta_neither, sfrm_gsw2.fibsfr_match_neither, filename='_unclassified', save=True)


'''
def disp_vs_ssfr(disp_x, disp_y, ssfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disps = np.sqrt(disp_x**2+disp_y**2)
    val = np.where((disps <2) &(ssfr <-8) &(ssfr>-14))[0]
    plot2dhist(ssfr[val], disps[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    #plt.xlim([-0.1, 1.5])
    #plt.ylim([-0.1, 0.5])
    plt.xlim([-14,-8])
    plt.ylim([-1,1])
    plt.xlabel('log(sSFR), ', fontsize=20)
    plt.ylabel('Displacement', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/disp_vs_ssfr' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/disp_vs_ssfr'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/disp_vs_ssfr'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
disp_vs_ssfr(sfrm_gsw2.offset_nii_halpha, sfrm_gsw2.offset_oiii_hbeta, sfrm_gsw2.ssfr_bpt, filename='_bpt', save=True)
disp_vs_ssfr(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.ssfr_plus, filename='_plus', save=True)
disp_vs_ssfr(sfrm_gsw2.offset_nii_halpha_neither, sfrm_gsw2.offset_oiii_hbeta_neither, sfrm_gsw2.ssfr_neither, filename='_unclassified', save=True)

disp_vs_ssfr(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)

'''
def dispx_vs_ssfr(disp_x, ssfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    val = np.where((disp_x <2) &(ssfr <-8) &(ssfr>-14))[0]
    plot2dhist(ssfr[val], disp_x[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    #plt.xlim([-0.1, 1.5])
    #plt.ylim([-0.1, 0.5])
    plt.xlim([-14,-8])
    plt.ylim([-1,1])
    plt.xlabel('log(sSFR), ', fontsize=20)
    plt.ylabel('Displacement X', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/dispx_vs_ssfr' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/dispx_vs_ssfr'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/dispx_vs_ssfr'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
dispx_vs_ssfr(sfrm_gsw2.offset_nii_halpha,  sfrm_gsw2.ssfr_bpt, filename='_bpt', save=True)
dispx_vs_ssfr(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.ssfr_plus, filename='_plus', save=True)
dispx_vs_ssfr(sfrm_gsw2.offset_nii_halpha_neither,sfrm_gsw2.ssfr_neither, filename='_unclassified', save=True)

dispx_vs_ssfr(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)

'''
def dispx_vs_sfr(disp_x, sfr, save=False, filename='', nx = 50, ny=50, fig=None, ax=None ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    if not fig and not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    val = np.where((disp_x <2) &(sfr <3) &(sfr>-3))[0]
    plot2dhist(sfr[val], disp_x[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    ax.set_xlim([-3,3])
    ax.set_ylim([-1,1])
    ax.set_xlabel('log(SFR), ', fontsize=20)
    ax.set_ylabel('Displacement X', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/dispx_vs_sfr' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/dispx_vs_sfr'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/dispx_vs_sfr'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
dispx_vs_sfr(sfrm_gsw2.offset_nii_halpha,  sfrm_gsw2.sfr_bpt, filename='_bpt', save=True)
dispx_vs_sfr(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.sfr_plus, filename='_plus', save=True)
dispx_vs_sfr(sfrm_gsw2.offset_nii_halpha_neither,sfrm_gsw2.sfr_neither, filename='_unclassified', save=True)

dispx_vs_sfr(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)

'''

def dispx_vs_fibsfr(disp_x, fibsfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    val = np.where((disp_x <2) &(fibsfr <3) &(fibsfr>-4))[0]
    plot2dhist(fibsfr[val], disp_x[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    #plt.xlim([-0.1, 1.5])
    #plt.ylim([-0.1, 0.5])
    plt.xlim([-4,3])
    plt.ylim([-1,1])
    plt.xlabel(r'log(SFR$_{\mathrm{fib}}$)', fontsize=20)
    plt.ylabel('Displacement X', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/dispx_vs_fibsfr' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/dispx_vs_fibsfr'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/dispx_vs_fibsfr'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
dispx_vs_fibsfr(sfrm_gsw2.offset_nii_halpha,  sfrm_gsw2.fibsfr_bpt, filename='_bpt', save=True)
dispx_vs_fibsfr(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.fibsfr_plus, filename='_plus', save=True)
dispx_vs_fibsfr(sfrm_gsw2.offset_nii_halpha_neither,sfrm_gsw2.fibsfr_neither, filename='_unclassified', save=True)

dispx_vs_sfr(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)

'''
def dispx_vs_fibsfr_match(disp_x, fibsfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    val = np.where((disp_x <2) &(fibsfr <3) &(fibsfr>-4))[0]
    plot2dhist(fibsfr[val], disp_x[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    #plt.xlim([-0.1, 1.5])
    #plt.ylim([-0.1, 0.5])
    plt.xlim([-4,3])
    plt.ylim([-1,1])
    plt.xlabel(r'log(SFR$_{\mathrm{fib}}$)', fontsize=20)
    plt.ylabel('Displacement X', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/dispx_vs_fibsfr_match' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/dispx_vs_fibsfr_match'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/dispx_vs_fibsfr_match'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
dispx_vs_fibsfr_match(sfrm_gsw2.offset_nii_halpha,  sfrm_gsw2.fibsfr_match_bpt, filename='_bpt', save=True)
dispx_vs_fibsfr_match(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.fibsfr_match_plus, filename='_plus', save=True)
dispx_vs_fibsfr_match(sfrm_gsw2.offset_nii_halpha_neither,sfrm_gsw2.fibsfr_match_neither, filename='_unclassified', save=True)


dispx_vs_fibsfr_match(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.sfr_sing, filename='_all', save=False)

'''
def dispx_vs_mass(disp_x, mass, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    val = np.where((disp_x <2) &(mass >8) &(mass<12))[0]
    plot2dhist(mass[val], disp_x[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    #plt.xlim([-0.1, 1.5])
    #plt.ylim([-0.1, 0.5])
    plt.xlim([8,12])
    plt.ylim([-1,1])
    plt.xlabel(r'log(M$_{\mathrm{*}}$), ', fontsize=20)
    plt.ylabel('Displacement X', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/dispx_vs_mass' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/dispx_vs_mass'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/dispx_vs_mass'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
dispx_vs_mass(sfrm_gsw2.offset_nii_halpha,  sfrm_gsw2.mass_bpt, filename='_bpt', save=False)
dispx_vs_mass(sfrm_gsw2.offset_nii_halpha_plus, sfrm_gsw2.mass_plus, filename='_plus', save=False)
dispx_vs_mass(sfrm_gsw2.offset_nii_halpha_neither,sfrm_gsw2.mass_neither, filename='_unclassified', save=False)

dispx_vs_mass(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.sfr_sing, filename='_all', save=True)

'''
def dispy_vs_ssfr(disp_y, ssfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    val = np.where((disp_y <2) &(ssfr <-8) &(ssfr>-14))[0]
    plot2dhist(ssfr[val], disp_y[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    #plt.xlim([-0.1, 1.5])
    #plt.ylim([-0.1, 0.5])
    plt.xlim([-14,-8])
    plt.ylim([-1,1])
    plt.xlabel('log(sSFR), ', fontsize=20)
    plt.ylabel('Displacement Y', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/dispy_vs_ssfr' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/dispy_vs_ssfr'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/dispy_vs_ssfr'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
dispy_vs_ssfr(sfrm_gsw2.offset_oiii_hbeta,  sfrm_gsw2.ssfr_bpt, filename='_bpt', save=True)
dispy_vs_ssfr(sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.ssfr_plus, filename='_plus', save=True)
dispy_vs_ssfr(sfrm_gsw2.offset_oiii_hbeta_neither,sfrm_gsw2.ssfr_neither, filename='_unclassified', save=True)

#fiber sfr mpa

dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)
#fiber ssfr mpa
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)

#fiber sfr match mpa
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)

#fiber ssfr match mpa
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)
dispy_vs_ssfr( sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.ssfr_sing, filename='_all', save=True)

#fiber sfr gsw

#fiber sfr gsw match
'''



def dispy_vs_sfr(disp_y, sfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    val = np.where((disp_y <2) &(sfr <3) &(sfr>-3))[0]
    plot2dhist(sfr[val], disp_y[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    #plt.xlim([-0.1, 1.5])
    #plt.ylim([-0.1, 0.5])
    plt.xlim([-3,3])
    plt.ylim([-1,1])
    plt.xlabel('log(SFR), ', fontsize=20)
    plt.ylabel('Displacement Y', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/dispy_vs_sfr' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/dispy_vs_sfr'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/dispy_vs_sfr'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
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
           filename='diagnostic/sfr_fib_disp', save=True,
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




plot3panel( np.log10(sfrm_gsw2.siiflux_sub_sing/sfrm_gsw2.oiiflux_sub_sing), 
           np.log10(sfrm_gsw2.siiflux_sub_sing/sfrm_gsw2.oiiflux_sub_sing)[sfrm_gsw2.high_ssfr_obj], 
           np.log10(sfrm_gsw2.siiflux_sub_sing/sfrm_gsw2.oiiflux_sub_sing)[sfrm_gsw2.low_ssfr_obj],
           np.log10(sfrm_gsw2.halpflux_sub_sing/sfrm_gsw2.hbetaflux_sub_sing), 
           np.log10(sfrm_gsw2.halpflux_sub_sing/sfrm_gsw2.hbetaflux_sub_sing)[sfrm_gsw2.high_ssfr_obj], 
           np.log10(sfrm_gsw2.halpflux_sub_sing/sfrm_gsw2.hbetaflux_sub_sing)[sfrm_gsw2.low_ssfr_obj],filename='diagnostic/ssfr_fib_mj_match_dispx', save=False,
           bin_y=True,minx=-3, maxx=3, miny=-3, maxy=3, nx=250, ny=250, xlabel='log([SII]/[OII])', ylabel=r'log(H$\alpha$/H$\beta$',nan=True)             

'''


def dispy_vs_fibsfr(disp_y, fibsfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    val = np.where((disp_y <2) &(fibsfr <3) &(fibsfr>-4))[0]
    plot2dhist(fibsfr[val], disp_y[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    #plt.xlim([-0.1, 1.5])
    #plt.ylim([-0.1, 0.5])
    plt.xlim([-4,3])
    plt.ylim([-1,1])
    plt.xlabel(r'log(SFR$_{\mathrm{fib}}$), ', fontsize=20)
    plt.ylabel('Displacement Y', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/dispy_vs_fibsfr' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/dispy_vs_fibsfr'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/dispy_vs_fibsfr'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
dispy_vs_fibsfr(sfrm_gsw2.offset_oiii_hbeta,  sfrm_gsw2.fibsfr_bpt, filename='_bpt', save=True)
dispy_vs_fibsfr(sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.fibsfr_plus, filename='_plus', save=True)
dispy_vs_fibsfr(sfrm_gsw2.offset_oiii_hbeta_neither,sfrm_gsw2.fibsfr_neither, filename='_unclassified', save=True)

dispy_vs_fibsfr(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.sfr_sing, filename='_all', save=True)

'''

def dispy_vs_fibsfr_match(disp_y, fibsfr, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    val = np.where((disp_y <2) &(fibsfr <3) &(fibsfr>-4))[0]
    plot2dhist(fibsfr[val], disp_y[val], nx,ny, bin_y=True, bin_stat_y='mean', nan=True)
    #plt.xlim([-0.1, 1.5])
    #plt.ylim([-0.1, 0.5])
    plt.xlim([-4,3])
    plt.ylim([-1,1])
    plt.xlabel(r'log(SFR$_{\mathrm{fib}}$), ', fontsize=20)
    plt.ylabel('Displacement Y', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/dispy_vs_fibsfr_match' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/dispy_vs_fibsfr_match'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/dispy_vs_fibsfr_match'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
dispy_vs_fibsfr_match(sfrm_gsw2.offset_oiii_hbeta,  sfrm_gsw2.fibsfr_match_bpt, filename='_bpt', save=True)
dispy_vs_fibsfr_match(sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.fibsfr_match_plus, filename='_plus', save=True)
dispy_vs_fibsfr_match(sfrm_gsw2.offset_oiii_hbeta_neither,sfrm_gsw2.fibsfr_match_neither, filename='_unclassified', save=True)

dispy_vs_fibsfr_match(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.sfr_sing, filename='_all', save=True)

'''


def dispy_vs_mass(disp_y, mass, save=False, filename='', nx = 50, ny=50 ): 
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    val = np.where((disp_y <2) &(mass >8) &(mass<12))[0]
    plot2dhist(mass[val], disp_y[val], nx,ny, bin_y=False, bin_stat_y='mean', nan=True)
    #plt.xlim([-0.1, 1.5])
    #plt.ylim([-0.1, 0.5])
    plt.xlim([8,12])
    plt.ylim([-2,2])
    plt.xlabel(r'log(M$_{\mathrm{*}}$), ', fontsize=20)
    plt.ylabel('Displacement Y', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/dispy_vs_mass' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/dispy_vs_mass'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/dispy_vs_mass'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
  
'''
dispy_vs_mass(sfrm_gsw2.offset_oiii_hbeta,  sfrm_gsw2.mass_bpt, filename='_bpt', save=False)
dispy_vs_mass(sfrm_gsw2.offset_oiii_hbeta_plus, sfrm_gsw2.mass_plus, filename='_plus', save=False)
dispy_vs_mass(sfrm_gsw2.offset_oiii_hbeta_neither,sfrm_gsw2.mass_neither, filename='_unclassified', save=False)

dispy_vs_mass(sfrm_gsw2.offset_nii_halpha_sing, sfrm_gsw2.offset_oiii_hbeta_sing, sfrm_gsw2.sfr_sing, filename='_all', save=True)

'''

def dist_vs_disp_cell(disp_x, disp_y, dist, coordpair, typ, save=False, filename='', nx = 50, ny=50 ):
    #avg_disp =np.sqrt(np.mean(disp_x)**2+np.mean(disp_y)**2)
    #med_disp = np.sqrt(np.median(disp_x)**2+np.median(disp_y)**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disps = np.sqrt(disp_x**2+disp_y**2)

    plot2dhist(disps, dist, nx,ny, bin_y=True, bin_stat_y='mean')
    plt.title('(x, y) = ('+str(round(coordpair[0],2))+' , '+str(round(coordpair[1],2)) +')', fontsize=20)
    plt.xlim([-0.1, 1.5])
    plt.ylim([-0.1, 0.5])
    plt.xlabel('Displacements, '+typ, fontsize=20)
    plt.ylabel('Match Distances', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/distmet/disps_dists_cell_xy_'+str(round(coordpair[0],2))+'_'+str(round(coordpair[1],2))+'_' +filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/distmet/disps_dists_cell_xy_'+str(round(coordpair[0],2))+'_'+str(round(coordpair[1],2))+'_' +filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/distmet/disps_dists_cell_xy_'+str(round(coordpair[0],2))+'_'+str(round(coordpair[1],2))+'_' +filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
    
def dist_vs_disp_all(disps_set, typ, save=False, filename='', nx=200, ny=200):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disps_x = np.array([])
    disps_y = np.array([])
    dists =  np.array([])
    for i in range(len(disps_set)):
        disps_x = np.append(disps_x, disps_set[i][8])
        disps_y = np.append(disps_y, disps_set[i][9])
        dists = np.append(dists, disps_set[i][10])
    disps = np.sqrt(disps_x**2+disps_y**2)
    print(np.where(disps<0.001)[0].size/disps.size)
    fin = np.where(disps >0)[0]
    plt.title(typ, fontsize=20)
    plot2dhist(disps[fin], dists[fin], nx,ny, bin_y=True, bin_stat_y='mean')
    plt.xlabel('Displacements', fontsize=20)
    plt.ylabel('Match Distances', fontsize=20)
    plt.xlim([-0.1, 1.5])
    plt.ylim([-0.1, 0.5])
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/distmet/disps_dists_'+ filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/distmet/disps_dists_'+ filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/distmet/disps_dists_'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)
    
'''

i=445# 455,398,408,445
dist_vs_disp_cell(sfrm_gsw2.bpt_set[i][8],sfrm_gsw2.bpt_set[i][9], sfrm_gsw2.bpt_set[i][10],
                  sfrm_gsw2.bpt_set[i][2:4],  'BPT', save=False, filename='bpt', 
                  nx = 15, ny = 15)
dist_vs_disp_cell(sfrm_gsw2.plus_set[i][8],sfrm_gsw2.plus_set[i][9], sfrm_gsw2.plus_set[i][10],
                  sfrm_gsw2.plus_set[i][2:4],  'BPT+', save=False, filename='plus', 
                  nx = 15, ny = 15)
dist_vs_disp_cell(sfrm_gsw2.neither_set[i][8],sfrm_gsw2.neither_set[i][9], sfrm_gsw2.neither_set[i][10],
                  sfrm_gsw2.neither_set[i][2:4],  'Unclassified', save=True, filename='unclassified',
                  nx = 15, ny = 15)

dist_vs_disp_cell(sfrm_gsw2.bpt_set[428][8],sfrm_gsw2.bpt_set[428][9], sfrm_gsw2.bpt_set[428][10],sfrm_gsw2.bpt_set[428][2:4],  'BPT', save=False, filename='bpt_example_1', nx = 15, ny = 15)
dist_vs_disp_cell(sfrm_gsw2.plus_set[428][8],sfrm_gsw2.plus_set[428][9], sfrm_gsw2.plus_set[428][10],sfrm_gsw2.plus_set[428][2:4],  'BPT+', save=False, filename='plus_example_1', nx = 15, ny = 15)
dist_vs_disp_cell(sfrm_gsw2.neither_set[428][8],sfrm_gsw2.neither_set[428][9], sfrm_gsw2.neither_set[428][10],sfrm_gsw2.neither_set[428][2:4],  'Unclassified', save=False, filename='bpt_example_1', nx = 15, ny = 15)


dist_vs_disp_all(sfrm_gsw2.bpt_set,'BPT', save=True, filename='bpt')

dist_vs_disp_all(sfrm_gsw2.plus_set,'BPT+', save=True, filename='plus')

dist_vs_disp_all(sfrm_gsw2.neither_set, 'Unclassified', save=False, filename='unclassified')


'''


def histsf_diffs(diffs, matchtyp, save=False, filename='', bins=20, ran=(0,1)):
    fig = plt.figure()
    ax= fig.add_subplot(111)
    plt.hist(diffs, color='k', histtype='step', bins=bins, range=ran)
    plt.xlabel('Distance to ' + match_typs[matchtyp], fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/distmet/histsf_diffs_'+ filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/distmet/histsf_diffs_'+ filename+'.pdf', dpi=250,bbox_inches='tight')
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




def plotfluxcomp(agn_flux, matches_flux, line, matchtyp, filename='', save=False, nx=250, ny=250):
    nagn_bigger = np.where(agn_flux > matches_flux)[0].size
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print('agn bigger: ', nagn_bigger)
    print('match bigger: ', matches_flux.size-nagn_bigger)
    mnx = np.percentile(agn_flux, 1)
    mxx = np.percentile(agn_flux, 99)
    mny = np.percentile(matches_flux, 1)
    mxy = np.percentile(matches_flux, 99)
    
    mxval = np.max([mxx,mxy])
    valbg = np.where((np.isfinite(agn_flux) & (np.isfinite(matches_flux))) &
            (matches_flux > mny) &( matches_flux < mxy) & (agn_flux< mxx)&(agn_flux > -1.5e-15) )[0]    
    plot2dhist(agn_flux[valbg], matches_flux[valbg], nx, ny)
    plt.plot([0,mxval],[0,mxval],'b--')
    #plt.text(mxval/8, mxval-mxval/5, r'N$_{\mathrm{above}}$ = '+str(matches_flux.size-nagn_bigger) + '(' +str(round((matches_flux.size-nagn_bigger)/matches_flux.size,3)*100)[0:4]+'\%)', fontsize=15)
    
    plt.axvline(x=0, color='r', ls=':')
    #plt.axhline(y=0, color='r', ls=':')
    plt.xlabel(line+' Flux, AGN', fontsize=20)
    plt.ylabel(line+' Flux, ' + ' Match', fontsize=20)
    
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/fluxcomp/fluxcomp_'+ filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/fluxcomp/fluxcomp_'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/fluxcomp_'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)



'''
plotfluxcomp(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.neither_agn],
             sfrm_gsw2.eldiag.oiiiflux_neither[sfrm_gsw2.neither_matches]*sfrm_gsw2.bptdistrat_neither,
             '[OIII]', 'Unclassified', filename='unclass_oiii', save=False, nx = 40, ny = 40)
plotfluxcomp(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.neither_agn], 
             sfrm_gsw2.eldiag.hbetaflux_neither[sfrm_gsw2.neither_matches]*sfrm_gsw2.bptdistrat_neither,r'H$\beta$', 
             'Unclassified', filename='unclass_hbeta', save=False, nx=40, ny=40)
plotfluxcomp(sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.neither_agn],
             sfrm_gsw2.eldiag.halpflux_neither[sfrm_gsw2.neither_matches]*sfrm_gsw2.bptdistrat_neither,r'H$\alpha$', 
             'Unclassified', filename='unclass_halp', save=False, nx=40, ny=40)
plotfluxcomp(sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.neither_agn], 
             sfrm_gsw2.eldiag.niiflux_neither[sfrm_gsw2.neither_matches]*sfrm_gsw2.bptdistrat_neither,'[NII]',
             'Unclassified', filename='unclass_nii', save=False, nx=40, ny=40)

plotfluxcomp(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.agns_plus], 
             sfrm_gsw2.eldiag.oiiiflux_plus[sfrm_gsw2.sfs_plus]*sfrm_gsw2.bptdistrat_plus,'[OIII]',
             'NII/Ha SF', filename='plus_oiii', save=True)
plotfluxcomp(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.agns_plus], 
             sfrm_gsw2.eldiag.hbetaflux_plus[sfrm_gsw2.sfs_plus]*sfrm_gsw2.bptdistrat_plus,
             r'H$\beta$', 'NII/Ha SF', filename='plus_hbeta', save=True)
plotfluxcomp(sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.agns_plus], sfrm_gsw2.eldiag.halpflux_plus[sfrm_gsw2.sfs_plus]*sfrm_gsw2.bptdistrat_plus,r'H$\alpha$', 'NII/Ha SF', filename='plus_halp', save=False)
plotfluxcomp(sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.agns_plus], sfrm_gsw2.eldiag.niiflux_plus[sfrm_gsw2.sfs_plus]*sfrm_gsw2.bptdistrat_plus,'[NII]', 'NII/Ha SF', filename='plus_nii', save=False)

plotfluxcomp(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.agns],
             sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat,'[OIII]',
             'BPT SF', filename='bpt_oiii', save=True)
plotfluxcomp(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.agns],
             sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat,r'H$\beta$', 
             'BPT SF', filename='bpt_hbeta', save=True)
plotfluxcomp(sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat,r'H$\alpha$', 'BPT SF', filename='plus_halp', save=False)
plotfluxcomp(sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat,'[NII]', 'BPT SF', filename='plus_nii', save=False)

i=445
plotfluxcomp(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.agns][sfrm_gsw2.bpt_set[i][-1]], 
             (sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat)[sfrm_gsw2.bpt_set[i][-1]],
             '[OIII]', 'BPT SF', filename='bpt_oiii_xy_'+str(round(sfrm_gsw2.bpt_set[i][2],2))+'_'+str(round(sfrm_gsw2.bpt_set[i][3],2)),
             save=True, nx=40, ny=40)
plotfluxcomp(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.agns][sfrm_gsw2.bpt_set[i][-1]], 
             (sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat)[sfrm_gsw2.bpt_set[i][-1]]
             ,r'H$\beta$', 'BPT SF', filename='bpt_hbeta_xy_'+str(round(sfrm_gsw2.bpt_set[i][2],2))+'_'+str(round(sfrm_gsw2.bpt_set[i][3],2)), 
             save=True, nx=40, ny=40)

plotfluxcomp(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.agns_plus][sfrm_gsw2.plus_set[i][-1]], 
             (sfrm_gsw2.eldiag.oiiiflux_plus[sfrm_gsw2.sfs_plus]*sfrm_gsw2.bptdistrat_plus)[sfrm_gsw2.plus_set[i][-1]],
             '[OIII]', 'NII/Ha SF', filename='plus_oiii_xy_'+str(round(sfrm_gsw2.plus_set[i][2],2))+'_'+str(round(sfrm_gsw2.plus_set[i][3],2)),
             save=True, nx=40, ny=40)
plotfluxcomp(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.agns_plus][sfrm_gsw2.plus_set[i][-1]], 
             (sfrm_gsw2.eldiag.hbetaflux_plus[sfrm_gsw2.sfs_plus]*sfrm_gsw2.bptdistrat_plus)[sfrm_gsw2.plus_set[i][-1]]
             ,r'H$\beta$', 'NII/Ha SF', filename='plus_hbeta_xy_'+str(round(sfrm_gsw2.plus_set[i][2],2))+'_'+str(round(sfrm_gsw2.plus_set[i][3],2)), 
             save=True, nx=40, ny=40)

plotfluxcomp(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.neither_agn][sfrm_gsw2.neither_set[i][-1]], 
             (sfrm_gsw2.eldiag.oiiiflux_neither[sfrm_gsw2.neither_matches]*sfrm_gsw2.bptdistrat_neither)[sfrm_gsw2.neither_set[i][-1]],
             '[OIII]', 'Unclassified', filename='unclassified_oiii_xy_'+str(round(sfrm_gsw2.neither_set[i][2],2))+'_'+str(round(sfrm_gsw2.neither_set[i][3],2)),
             save=True, nx=40, ny=40)
plotfluxcomp(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.neither_agn][sfrm_gsw2.neither_set[i][-1]], 
             (sfrm_gsw2.eldiag.hbetaflux_neither[sfrm_gsw2.neither_matches]*sfrm_gsw2.bptdistrat_neither)[sfrm_gsw2.neither_set[i][-1]]
             ,r'H$\beta$', 'Unclassified', filename='unclassified_hbeta_xy_'+str(round(sfrm_gsw2.neither_set[i][2],2))+'_'+str(round(sfrm_gsw2.neither_set[i][3],2)), 
             save=True, nx=40, ny=40)



plotfluxcomp(sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat,r'H$\alpha$', 'BPT SF', filename='plus_halp', save=False)
plotfluxcomp(sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat,'[NII]', 'BPT SF', filename='plus_nii', save=False)




n_pos_unclass = np.where((sfrm_gsw2.niiflux_sub_sn_neither>0)&
         (sfrm_gsw2.oiiiflux_sub_sn_neither>0) &
         (sfrm_gsw2.halpflux_sub_sn_neither>0) &
         (sfrm_gsw2.hbetaflux_sub_sn_neither>0))[0].size

perc_pos_unclass = n_pos_unclass /sfrm_gsw2.neither_agn.size

n_pos_niiha = np.where((sfrm_gsw2.niiflux_sub_sn_plus>0)&
         (sfrm_gsw2.oiiiflux_sub_sn_plus>0) &
         (sfrm_gsw2.halpflux_sub_sn_plus>0) &
         (sfrm_gsw2.hbetaflux_sub_sn_plus>0))[0].size
racmin)&(EL_m2.massfracgsw_plus[nonagn_gsw_bptplusnii]<=massfracmax))[0]

perc_pos_niiha = n_pos_niiha /sfrm_gsw2.agns_plus.size

n_pos_bpt = np.where((sfrm_gsw2.niiflux_sub_sn>0)&
         (sfrm_gsw2.oiiiflux_sub_sn>0) &
         (sfrm_gsw2.halpflux_sub_sn>0) &
         (sfrm_gsw2.hbetaflux_sub_sn>0))[0].size
perc_pos_bpt = n_pos_bpt /sfrm_gsw2.agns.size
                       
'''

def histflux_diffs(flux1, flux2, line, typ, col='k', ran=[], bins=30, save = False, filename='', log=False):
    fig = plt.figure()
    ax =fig.add_subplot(111)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    
    if log:
        val = np.where(flux1-flux2 >0)[0]
        plt.hist(np.log10(flux1-flux2)[val], bins=bins, color=col, histtype='step')
        plt.xlabel('log('+line +r'$_{\mathrm{AGN}}$-'+line+r'$_{\mathrm{match}}$)'+', '+ typ, fontsize=20)

    else:
        plt.xlabel(line +r'$_{\mathrm{AGN}}$-'+line+r'$_{\mathrm{match}}$'+', '+ typ, fontsize=20)

        plt.hist(flux1-flux2, bins=bins, color=col, histtype='step')
    plt.ylabel('Counts', fontsize=20)
    
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    
    if save:
        fig.savefig('plots/sfrmatch/png/flux_diffs_'+filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/flux_diffs_'+filename+'.pdf', dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        #fig.savefig('plots/sfrmatch/eps/match_dists_'+filename+'.eps', dpi=150,format='eps')
        plt.close(fig)

    else:
        plt.show()
        
        
'''
histflux_diffs(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.neither_agn], sfrm_gsw2.eldiag.oiiiflux_neither[sfrm_gsw2.neither_matches]*sfrm_gsw2.bptdistrat_neither,'[OIII]', 'Unclassified', bins=50,log=True,filename='unclass_oiii', save=False)
histflux_diffs(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.neither_agn], sfrm_gsw2.eldiag.hbetaflux_neither[sfrm_gsw2.neither_matches]*sfrm_gsw2.bptdistrat_neither,r'H$\beta$', 'Unclassified', bins=50,log=True, filename='unclass_hbeta', save=False)
histflux_diffs(sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.neither_agn], sfrm_gsw2.eldiag.halpflux_neither[sfrm_gsw2.neither_matches]*sfrm_gsw2.bptdistrat_neither,r'H$\alpha$', 'Unclassified', bins=50,log=True, filename='unclass_halp', save=False)
histflux_diffs(sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.neither_agn], sfrm_gsw2.eldiag.niiflux_neither[sfrm_gsw2.neither_matches]*sfrm_gsw2.bptdistrat_neither,'[NII]', 'Unclassified', bins=50,log=True, filename='unclass_nii', save=False)

histflux_diffs(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.agns_plus], sfrm_gsw2.eldiag.oiiiflux_plus[sfrm_gsw2.sfs_plus]*sfrm_gsw2.bptdistrat_plus,'[OIII]', 'NII/Ha SF', bins=50, log=True,filename='plus_oiii', save=False)
histflux_diffs(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.agns_plus], sfrm_gsw2.eldiag.hbetaflux_plus[sfrm_gsw2.sfs_plus]*sfrm_gsw2.bptdistrat_plus,r'H$\beta$', 'NII/Ha SF', bins=50,log=True, filename='plus_hbeta', save=False)
histflux_diffs(sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.agns_plus], sfrm_gsw2.eldiag.halpflux_plus[sfrm_gsw2.sfs_plus]*sfrm_gsw2.bptdistrat_plus,r'H$\alpha$', 'NII/Ha SF', bins=50,log=True, filename='plus_halp', save=False)
histflux_diffs(sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.agns_plus], sfrm_gsw2.eldiag.niiflux_plus[sfrm_gsw2.sfs_plus]*sfrm_gsw2.bptdistrat_plus,'[NII]', 'NII/Ha SF', bins=50,log=True, filename='plus_nii', save=False)

histflux_diffs(sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.oiiiflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat,'[OIII]', 'BPT SF', filename='bpt_oiii', bins=50,log=True, save=False)
histflux_diffs(sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.hbetaflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat,r'H$\beta$', 'BPT SF', bins=50,log=True, filename='bpt_hbeta', save=False)
histflux_diffs(sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.halpflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat,r'H$\alpha$', 'BPT SF', bins=50,log=True, filename='plus_halp', save=False)
histflux_diffs(sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.agns], sfrm_gsw2.eldiag.niiflux[sfrm_gsw2.sfs]*sfrm_gsw2.bptdistrat,'[NII]', 'BPT SF', bins=50,log=True, filename='plus_nii', save=False)
'''

def plthist(bincenters, counts):
    plt.plot(bincenters, counts,color='k', drawstyle='steps-mid')
def hist_sfrdiffs(diffs, lab, ymax=16000, col='k', ran=[], bins=30, save = False, filename=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
    ax.set(adjustable='box', aspect='equal')
    if save:
        fig.savefig('plots/sfrmatch/png/match_dists_'+filename+'.png', dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/match_dists_'+filename+'.pdf', dpi=250,bbox_inches='tight')
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

def plotdistrat(dists, label, rang=(0.5, 1.5), bins=40, save = False, filename=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    
    plt.hist(dists, range=rang, histtype='step', bins=bins)    
    plt.xlabel(r'$(d_{Match}/d_{AGN})^2$', fontsize=20)
    plt.title(label, fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    if save:
        fig.savefig('plots/sfrmatch/png/distrat_'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/distrat_'+filename+'.pdf',dpi=250,bbox_inches='tight')
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    val = np.where((massfrac >0)&(massfrac<2))[0]
    plt.hist(massfrac[val],label=lab)
    plt.xlabel('Mass Fraction')
    plt.ylabel('Counts')
    plt.xlim([0,1])
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

#massfrachist(all_sdss_massfrac)
def xrayhists(prop,propname,nbins=20):
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
    ax.set(adjustable='box', aspect='equal')


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

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
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
def plotbptdiffs(x1,y1,x2,y2,save=False,filename='',labels=True, title=None, minx=-2, maxx=1, miny=-1.2, maxy=1.2, nobj=False):
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
             save=True)


'''
def plotbptdiffsgrid(grid_arr, save=False,filename='',labels=True, title=None, minx=-2, maxx=1, miny=-1.2, maxy=1.2, nobj=False, avg = True):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
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

plotbptdiffsgrid(sfrm_gsw2.sing_set,
             save=True, filename= 'sing')

plotbptdiffsgrid(sfrm_gsw2.low_set, save=False, filename='low')
plotbptdiffsgrid(sfrm_gsw2.high_set, save=False, filename= high)



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
                   nx=300, ny=240, ccodename=''):
    fig = plt.figure(figsize=(8,8))
  
    ax1 = fig.add_subplot(311)
    ax1.set_xlim([minx,maxx])
    ax1.set_ylim([miny,maxy])
    ax1.set(aspect='equal', adjustable='box')
    
    ax2 = fig.add_subplot(312, sharey = ax1, sharex=ax1)
    ax3 = fig.add_subplot(313, sharey = ax1, sharex=ax1)
    ax2.set(adjustable='box', aspect='equal')
    ax3.set(adjustable='box', aspect='equal')
    plotbptnormal(x1, y1, ccode=ccode1, save=False, nobj=nobj, ax=ax1,fig=fig, minx=minx, maxx=maxx, miny=miny, maxy=maxy, nx=nx, ny=ny, ccodename=ccodename)
    ax2.text(-1.85, 1.5, r'$\Delta$log(sSFR)$>-0.7$', fontsize=15)
    plotbptnormal(x2, y2, ccode=ccode2, save=False, nobj=nobj, ax=ax2,fig=fig, minx=minx, maxx=maxx, miny=miny, maxy=maxy, nx=nx, ny=ny, ccodename=ccodename)
    ax3.text(-1.85, 1.5, r'$\Delta$log(sSFR)$\leq-0.7$', fontsize=15)
    plotbptnormal(x3, y3, ccode=ccode3, save=False, nobj=nobj, ax=ax3,fig=fig, minx=minx, maxx=maxx, miny=miny, maxy=maxy, nx=nx, ny=ny, ccodename=ccodename)    
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
               ccode1=sfrm_gsw2.ssfr_match_sing,ccode2=sfrm_gsw2.ssfr_match_sing[sfrm_gsw2.high_ssfr_obj], ccode3=sfrm_gsw2.ssfr_match_sing[sfrm_gsw2.low_ssfr_obj],
               filename='diagnostic/BPT_presub_3panel_ssfr', nobj=True, nx=20, ny=25,save=True, ccodename='log(sSFR)')
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
plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel', save=True)    

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
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_sfr_ccode',
               ccode1=sfrm_gsw2.sfr_sing, ccode2=sfrm_gsw2.sfr_sing[sfrm_gsw2.high_ssfr_obj], 
               ccode3=sfrm_gsw2.sfr_sing[sfrm_gsw2.low_ssfr_obj],
               nx=20, ny=25, ccodename=r'log(SFR)', save=True)



plot3panel_bpt(sfrm_gsw2.log_nii_halpha_sub_sing,sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],
               sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj], filename='diagnostic/BPT_sub_3panel_sfr_ccode',
               ccode1=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing), ccode2=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing)[sfrm_gsw2.high_ssfr_obj], 
               ccode3=np.log10(sfrm_gsw2.oiflux_sub_sing/sfrm_gsw2.siiflux_sub_sing)[sfrm_gsw2.low_ssfr_obj],
               nx=20, ny=25, ccodename=r'log([OI]/[SII])', save=False)

plot2dhist(sfrm_gsw2.siiflux/sfrm_gsw2.oiiflux, sfrm_gsw2.halpflux/sfrm_gsw2.hbetaflux, nx=200, ny=200, nan=True)


fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(311)
ax1.set_xlim([-2,1.5])
ax1.set_ylim([-1.2,2])
ax1.set(aspect='equal', adjustable='box')

ax2 = fig.add_subplot(312, sharey = ax1, sharex=ax1)
ax3 = fig.add_subplot(313, sharey = ax1, sharex=ax1)
ax2.set(adjustable='box', aspect='equal')
ax3.set(adjustable='box', aspect='equal')

#ax1.set_xticklabels([''])

ax2.set_ylabel(r'log([OIII]/H$\rm \beta$)')

plotbptnormal(sfrm_gsw2.log_nii_halpha_sing, sfrm_gsw2.log_oiii_hbeta_sing,save=False, nobj=False,ax=ax1)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.high_ssfr_obj], sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.high_ssfr_obj],save=False, nobj=False,ax=ax2)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.low_ssfr_obj], sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.low_ssfr_obj],save=False, nobj=False,ax=ax3)

ax3.set_ylabel('')
ax1.set_ylabel('')
ax1.set_xlabel('')
ax2.set_xlabel('')


plt.subplots_adjust(wspace=0, hspace=0)


plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing, sfrm_gsw2.log_oiii_hbeta_sub_sing,save=False, nobj=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.low_ssfr_obj], sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.low_ssfr_obj],save=False, nobj=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing[sfrm_gsw2.high_ssfr_obj], sfrm_gsw2.log_oiii_hbeta_sub_sing[sfrm_gsw2.high_ssfr_obj],save=False, nobj=False)






plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt]),filename='_test', save=False, nobj=False)

plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][agn_gsw_bptplus]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][agn_gsw_bptplus]),filename='_agns_presub', save=False)


#OIII parts
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=np.log10(sfrm_gsw2.oiiilum_sub_bpt), 
              filename='_oiiilum_bpt', save=True, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$', ccodelim = np.log10(sfrm_gsw2.oiiilum_sub_sing))
plotbracmin)&(EL_m2.massfracgsw_plus[nonagn_gsw_bptplusnii]<=massfracmax))[0]
ptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=np.log10(sfrm_gsw2.oiiilum_sub_dered_bpt), 
              filename='_oiiilum_dered_bpt', save=True, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$', ccodelim=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing))
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=np.log10(sfrm_gsw2.oiiilum_sub_plus), 
              filename='_oiiilum_plus', save=True, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$', ccodelim = np.log10(sfrm_gsw2.oiiilum_sub_sing))
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=np.log10(sfrm_gsw2.oiiilum_sub_dered_plus), 
              filename='_oiiilum_dered_plus', save=True, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$', ccodelim=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing))
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=np.log10(sfrm_gsw2.oiiilum_sub_neither), 
              filename='_oiiilum_neither', save=True, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$', ccodelim = np.log10(sfrm_gsw2.oiiilum_sub_sing))
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=np.log10(sfrm_gsw2.oiiilum_sub_dered_neither), 
              filename='_oiiilum_dered_neither', save=True, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$', ccodelim=np.log10(sfrm_gsw2.oiiilum_sub_dered_sing))


plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=np.log10(sfrm_gsw2.ssfr_neither), 
              filename='_ssfr', save=False, nx=25, ny=20, ccodename=r'log(sSFR)')


plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=np.log10(sfrm_gsw2.oiiilum_sub_bpt), 
              filename='_oiiilum_bpt', save=True, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=np.log10(sfrm_gsw2.oiiilum_sub_dered_bpt), 
              filename='_oiiilum_dered_bpt', save=True, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=np.log10(sfrm_gsw2.oiiilum_sub_plus), 
              filename='_oiiilum_plus', save=True, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=np.log10(sfrm_gsw2.oiiilum_sub_dered_plus), 
              filename='_oiiilum_dered_plus', save=True, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=np.log10(sfrm_gsw2.oiiilum_sub_neither), 
              filename='_oiiilum_neither', save=False, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=np.log10(sfrm_gsw2.oiiilum_sub_dered_neither), 
              filename='_oiiilum_dered_neither', save=False, nx=25, ny=20, ccodename=r'L$_{\mathrm{[OIII]}}$')

#sfr
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=sfrm_gsw2.sfr_bpt, 
              filename='_sfr_bpt', save=True, nx=25, ny=20, ccodename=r'log(SFR)', ccodelim = sfrm_gsw2.sfr_sing)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=sfrm_gsw2.sfr_plus, 
              filename='_sfr_plus', save=True, nx=25, ny=20, ccodename=r'log(SFR)', ccodelim = sfrm_gsw2.sfr_sing)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=sfrm_gsw2.sfr_neither, 
              filename='_sfr_neither', save=True, nx=25, ny=20, ccodename=r'log(SFR)', ccodelim = sfrm_gsw2.sfr_sing)
#sfr no ccodelim
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=sfrm_gsw2.sfr_bpt, 
              filename='_sfr_bpt_descaled', save=True, nx=25, ny=20, ccodename=r'log(SFR)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=sfrm_gsw2.sfr_plus, 
              filename='_sfr_plus_descaled', save=True, nx=25, ny=20, ccodename=r'log(SFR)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=sfrm_gsw2.sfr_neither, 
              filename='_sfr_neither_descaled', save=True, nx=25, ny=20, ccodename=r'log(SFR)')

#fibsfr
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=sfrm_gsw2.fibsfr_bpt, 
              filename='_fibsfr_bpt', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)', ccodelim = sfrm_gsw2.fibsfr_sing)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=sfrm_gsw2.fibsfr_plus, 
              filename='_fibsfr_plus', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)', ccodelim = sfrm_gsw2.fibsfr_sing)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=sfrm_gsw2.fibsfr_neither, 
              filename='_fibsfr_neither', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)', ccodelim = sfrm_gsw2.fibsfr_sing)
#fibsfr no ccodelim
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=sfrm_gsw2.fibsfr_bpt, 
              filename='_fibsfr_bpt_descaled', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=sfrm_gsw2.fibsfr_plus, 
              filename='_fibsfr_plus_descaled', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=sfrm_gsw2.fibsfr_neither, 
              filename='_fibsfr_neither_descaled', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)')


#fibsfr match
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=sfrm_gsw2.fibsfr_match_bpt, 
              filename='_fibsfr_match_bpt', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)', ccodelim = sfrm_gsw2.fibsfr_match_sing)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=sfrm_gsw2.fibsfr_match_plus, 
              filename='_fibsfr_match_plus', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)', ccodelim = sfrm_gsw2.fibsfr_match_sing)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=sfrm_gsw2.fibsfr_match_neither, 
              filename='_fibsfr_match_neither', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)', ccodelim = sfrm_gsw2.fibsfr_match_sing)
#fibsfr match no ccodelim
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=sfrm_gsw2.fibsfr_match_bpt, 
              filename='_fibsfr_match_bpt_descaled', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=sfrm_gsw2.fibsfr_match_plus, 
              filename='_fibsfr_match_plus_descaled', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=sfrm_gsw2.fibsfr_match_neither, 
              filename='_fibsfr_match_neither_descaled', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{fib}$)')

#ssfr
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=sfrm_gsw2.ssfr_bpt, 
              filename='_ssfr_bpt', save=False, nx=25, ny=20, ccodename=r'log(sSFR)', ccodelim = sfrm_gsw2.ssfr_sing)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing ,  sfrm_gsw2.log_oiii_hbeta_sub_sing,ccode=sfrm_gsw2.ssfr_sing, 
              filename='_ssfr_bpt', save=False, nx=25, ny=20, ccodename=r'log(sSFR)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=sfrm_gsw2.ssfr_plus, 
              filename='_ssfr_plus', save=True, nx=25, ny=20, ccodename=r'log(sSFR)', ccodelim = sfrm_gsw2.ssfr_sing)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=sfrm_gsw2.ssfr_neither, 
              filename='_ssfr_neither', save=True, nx=25, ny=20, ccodename=r'log(sSFR)', ccodelim = sfrm_gsw2.ssfr_sing)


#ssfr no ccodelim
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=sfrm_gsw2.ssfr_bpt, 
              filename='_ssfr_bpt_descaled', save=True, nx=25, ny=20, ccodename=r'log(sSFR)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,ccode=sfrm_gsw2.ssfr_plus, 
              filename='_ssfr_plus_descaled', save=True, nx=25, ny=20, ccodename=r'log(sSFR)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,ccode=sfrm_gsw2.ssfr_neither, 
              filename='_ssfr_neither_descaled', save=True, nx=25, ny=20, ccodename=r'log(sSFR)')


#ssfr presub
plotbptnormal(sfrm_gsw2.log_nii_halpha_sing ,  sfrm_gsw2.log_oiii_hbeta_sing,ccode=sfrm_gsw2.ssfr_sing, 
              filename='_ssfr', save=False, nx=50, ny=50, ccodename=r'log(sSFR)')
plotbptnormal(sfrm_gsw2.log_nii_halpha, sfrm_gsw2.log_oiii_hbeta, ccode=sfrm_gsw2.ssfr_bpt,
              nx=30, ny=30, ccodename=r'log(sSFR)', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_plus, sfrm_gsw2.log_oiii_hbeta_plus, ccode=sfrm_gsw2.ssfr_plus,
              nx=10, ny=10, ccodename=r'log(sSFR)', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_neither, sfrm_gsw2.log_oiii_hbeta_neither, ccode=sfrm_gsw2.ssfr_neither,
              nx=30, ny=30, ccodename=r'log(sSFR)', save=False)

#sfr presub
plotbptnormal(sfrm_gsw2.log_nii_halpha_sing ,  sfrm_gsw2.log_oiii_hbeta_sing,ccode=sfrm_gsw2.sfr_sing, 
              filename='_sfr', save=False, nx=30, ny=30, ccodename=r'log(SFR)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sing ,  sfrm_gsw2.log_oiii_hbeta_sing,ccode=sfrm_gsw2.fibsfr_sing, 
              filename='_sfr', save=False, nx=10, ny=10, ccodename=r'log(SFR)')

plotbptnormal(EL_m2.niiha[agn_gsw_bptplus[sfrm_gsw2.agn_ind_mapping]] ,  EL_m2.oiiihb[agn_gsw_bptplus[sfrm_gsw2.agn_ind_mapping]],
              filename='', save=False, ccode=EL_m2.ssfr[agn_gsw_bptplus[sfrm_gsw2.agn_ind_mapping]], nx=30, ny= 30)
plotbptnormal(EL_m2.niiha[agn_gsw_bptplus] ,  EL_m2.oiiihb[pplagn_gsw_bptplus],
              filename='', save=False, ccode=EL_m2.ssfr[agn_gsw_bptplus], nx=50, ny=50)

plotbptnormal(EL_m2.niiha[sfrm_gsw2.agns] ,  EL_m2.oiiihb[sfrm_gsw2.agns],
              filename='', save=False, ccode=EL_m2.ssfr[sfrm_gsw2.agns], nx=30, ny= 30)

plotbptnormal(EL_m2.niiha[sfrm_gsw2.agns_plus] ,  EL_m2.oiiihb[sfrm_gsw2.agns_plus],
              filename='', save=False, ccode=EL_m2.ssfr[sfrm_gsw2.agns_plus], nx=30, ny= 30)

plotbptnormal(EL_m2.niiha[sfrm_gsw2.neither_agn] ,  EL_m2.oiiihb[sfrm_gsw2.neither_agn],
              filename='', save=False, ccode=EL_m2.ssfr[sfrm_gsw2.neither_agn], nx=30, ny= 30)
plotbptnormal(EL_m2.niiha[agn_gsw_bptplus] , EL_m2.oiiihb[agn_gsw_bptplus],
              filename='', save=False, ccode=EL_m2.sfr[agn_gsw_bptplus], nx=30, ny= 30)
plotbptnormal(EL_m2.niiha[agn_gsw_bptplus] , EL_m2.oiiihb[agn_gsw_bptplus],
              filename='', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sing, sfrm_gsw2.log_oiii_hbeta_sing)

plotbptnormal(EL_m2.niiha[agn_gsw_bptplus] , EL_m2.oiiihb[agn_gsw_bptplus],
              filename='', save=False, nx=200, ny= 200)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sing ,  sfrm_gsw2.log_oiii_hbeta_sing, 
              filename='_sfr', save=False, nx=200, ny=200)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,ccode=np.log10(sfrm_gsw2.oiiilum_sub_dered_bpt), filename='_oiiilum_dered_bpt', save=False, nx=75, ny=60)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,filename='_fluxsub_highsn_plus', save=True)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,filename='_fluxsub_highsn_neither', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing ,  sfrm_gsw2.log_oiii_hbeta_sub_sing,filename='_fluxsub_highsn_all_agns', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing_agns_sf ,  sfrm_gsw2.log_oiii_hbeta_sub_sing_agns_sf,filename='_fluxsub_highsn_all_agns_all_sfs', save=True)

n=0
plt.scatter(EL_m2.niiha[agn_gsw_bptplus[sfrm_gsw2.agn_plus_dist_inds]][n*100:(1+n)*100],EL_m2.oiiihb[agn_gsw_bptplus[sfrm_gsw2.agn_plus_dist_inds]][n*100:(1+n)*100], 
            c= EL_m2.sfr[agn_gsw_bptplus[sfrm_gsw2.agn_plus_dist_inds]][n*100:(1+n)*100])
plt.figure()
plt.scatter(sfrm_gsw2.log_nii_halpha_plus[n*100:(1+n)*100],sfrm_gsw2.log_oiii_hbeta_plus[n*100:(1+n)*100], c= sfrm_gsw2.sfr_plus[n*100:(1+n)*100])


n=0
plt.scatter(EL_m2.niiha[agn_gsw_bptplus[sfrm_gsw2.agn_neither_dist_inds]][n*2000:(1+n)*2000],EL_m2.oiiihb[agn_gsw_bptplus[sfrm_gsw2.agn_neither_dist_inds]][n*2000:(1+n)*2000], 
            c= EL_m2.sfr[agn_gsw_bptplus[sfrm_gsw2.agn_neither_dist_inds]][n*2000:(1+n)*2000])
plt.colorbar()
plt.figure()
plt.scatter(sfrm_gsw2.log_nii_halpha_neither[n*2000:(1+n)*2000],sfrm_gsw2.log_oiii_hbeta_neither[n*2000:(1+n)*2000], c= sfrm_gsw2.sfr_neither[n*2000:(1+n)*2000])
plt.colorbar()

n=13
plt.scatter(EL_m2.niiha[agn_gsw_bptplus[sfrm_gsw2.agn_dist_inds]][n*2000:(1+n)*2000],EL_m2.oiiihb[agn_gsw_bptplus[sfrm_gsw2.agn_dist_inds]][n*2000:(1+n)*2000], 
            c= EL_m2.sfr[agn_gsw_bptplus[sfrm_gsw2.agn_dist_inds]][n*2000:(1+n)*2000])
plt.colorbar()
plt.figure()
plt.scatter(sfrm_gsw2.log_nii_halpha[n*2000:(1+n)*2000],sfrm_gsw2.log_oiii_hbeta[n*2000:(1+n)*2000], c= sfrm_gsw2.sfr_bpt[n*2000:(1+n)*2000])
plt.colorbar()


plotbptnormal(sfrm_gsw2.log_nii_halpha_sub ,  sfrm_gsw2.log_oiii_hbeta_sub,filename='_fluxsub_highsn_bpt', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus ,  sfrm_gsw2.log_oiii_hbeta_sub_plus,filename='_fluxsub_highsn_plus', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither ,  sfrm_gsw2.log_oiii_hbeta_sub_neither,filename='_fluxsub_highsn_neither', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing ,  sfrm_gsw2.log_oiii_hbeta_sub_sing,filename='_fluxsub_highsn_all_agns', save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing_agns_sf ,  sfrm_gsw2.log_oiii_hbeta_sub_sing_agns_sf,filename='_fluxsub_highsn_all_agns_all_sfs', save=True)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing_agns_sf_match ,  sfrm_gsw2.log_oiii_hbeta_sub_sing_agns_sf_match,filename='_fluxsub_highsn_all_agns_sf_matches', save=True)




plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.bpt_neither_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.bpt_neither_sn_filt],filename='fluxsub_old_neither', save=True)


plotbptnormal(sfrm_gsw2.log_nii_halpha_sub[sfrm_gsw2.bpt_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.bpt_sn_filt],filename='fluxsub_bpt_snfilt2', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.bpt_plus_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.bpt_plus_sn_filt],filename='fluxsub_plus_snfilt0', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.bpt_neither_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.bpt_neither_sn_filt],filename='fluxsub_neither_snfilt0', save=False, minx=-3.0, maxx=2.0, miny=-2, maxy=2)


sfrm_gsw2.getsfrmatch(agn_gsw_bptplus, nonagn_gsw_bptplus, nonagn_gsw_bptplusnii, agn_gsw_bptplusnii, load=True)
sfrm_gsw2.subtract_elflux(sncut=1)


for nn in n_vals_used[1:]:
    sfrm_gsw2.getsfrmatch_avg(agn_gsw_bptplus, nonagn_gsw_bptplus, nonagn_gsw_bptplusnii, agn_gsw_bptplusnii, load=True, n_avg=nn)
    sfrm_gsw2.subtract_elflux_avg(nonagn_gsw_bptplus, nonagn_gsw_bptplusnii,sncut=1)
    
    plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_avg[sfrm_gsw2.bpt_sn_filt_avg] ,  sfrm_gsw2.log_oiii_hbeta_sub_avg[sfrm_gsw2.bpt_sn_filt_avg],filename='fluxsub_bpt_snfilt1_avg'+str(sfrm_gsw2.n_avg),
                  save=True, minx=-3.0, maxx=2.0, miny=-2, maxy=2)


plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus_avg[sfrm_gsw2.bpt_plus_sn_filt_avg] ,  sfrm_gsw2.log_oiii_hbeta_sub_plus_avg[sfrm_gsw2.bpt_plus_sn_filt_avg],filename='fluxsub_plus_snfilt2_avg'+str(sfrm_gsw2.n_avg),
              save=True, minx=-3.0, maxx=2.0, miny=-2, maxy=2)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither_avg[sfrm_gsw2.bpt_neither_sn_filt_avg] ,  sfrm_gsw2.log_oiii_hbeta_sub_neither_avg[sfrm_gsw2.bpt_neither_sn_filt_avg],filename='fluxsub_neither_snfilt2_avg'+str(sfrm_gsw2.n_avg),
              save=False, minx=-3.0, maxx=2.0, miny=-2, maxy=2)


#post sub
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub[sfrm_gsw2.bpt_sn_filt_intermed] ,  sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.bpt_sn_filt_intermed],filename='fluxsub_bpt_snfilt_0-1', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.bpt_plus_sn_filt_intermed] ,  sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.bpt_plus_sn_filt_intermed],filename='fluxsub_plus_snfilt_0-1', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.bpt_neither_sn_filt_intermed] ,  sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.bpt_neither_sn_filt_intermed],filename='fluxsub_neither_snfilt_0-1', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing ,  sfrm_gsw2.log_oiii_hbeta_sub_sing,filename='fluxsub_highsn_bpt_sing', 
              save=False)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing_agns_sf_match ,  
              sfrm_gsw2.log_oiii_hbeta_sub_sing_agns_sf_match, 
              filename='fluxsub_highsn_bpt_sing', 
              save=False)

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_sing_agns_sf ,  
              sfrm_gsw2.log_oiii_hbeta_sub_sing_agns_sf, 
              filename='fluxsub_highsn_bpt_sing', 
              save=False)


plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.bpt_plus_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.bpt_plus_sn_filt],filename='fluxsub_plus_snfilt0', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.bpt_neither_sn_filt] ,  sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.bpt_neither_sn_filt],filename='fluxsub_neither_snfilt0', save=False, minx=-3.0, maxx=2.0, miny=-2, maxy=2)

#high_sn things

plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns]),filename='_highsn_agns', save=False)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]),filename='_highsn_agns_plus', save=False)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.neither_agn]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.neither_agn]),filename='_highsn_neither_agns', save=False)

#before subtraction
plotbptnormal(sfrm_gsw2.log_nii_halpha , sfrm_gsw2.log_oiii_hbeta, filename='_agns', save=False)

plotbptnormal(sfrm_gsw2.log_nii_halpha_plus , sfrm_gsw2.log_oiii_hbeta_plus, filename='_agns_plus', save=False)
plotbptnormal( sfrm_gsw2.log_nii_halpha_neither,sfrm_gsw2.log_oiii_hbeta_neither, filename='_neither_agns', save=False)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sing ,  sfrm_gsw2.log_oiii_hbeta_sing,filename='_all_agns', 
              save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_full_bpt ,  sfrm_gsw2.log_oiii_hbeta_full_bpt,filename='_all_agns_all_sfs', 
              save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_agns_sf ,  sfrm_gsw2.log_oiii_hbeta_agns_sf,filename='_all_agns_sf_matches', 
              save=True)
plotbptnormal(sfrm_gsw2.log_nii_halpha_sf_match ,  sfrm_gsw2.log_oiii_hbeta_sf_match,filename='_sf_matches', 
              save=True)





plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_selfmatch]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_selfmatch]),filename='_agns_selfmatch', save=False)

plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agnsplus_selfmatch]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agnsplus_selfmatch]),filename='_agnsplus_selfmatch', save=False)

plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.sfs]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.sfs]),filename='_old_sfs', save=True)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.sfs]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.sfs]),filename='_highsn_sfs', save=True)

plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_snlr_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_snlr_filt]),filename='sn_lr3', save=True)
    




#[SII]/Ha sub
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub[sfrm_gsw2.good_oiii_hb_sii_ha_sub_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.good_oiii_hb_sii_ha_sub_bpt] ,
              ccode=np.log10(sfrm_gsw2.siiflux_sub/sfrm_gsw2.halpflux_sub)[sfrm_gsw2.good_oiii_hb_sii_ha_sub_bpt] , 
              filename='_fluxsub_sii_ha_bpt', save=True, nx=25, ny=20, ccodename=r'log([SII]/H$\alpha$)') 
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.good_oiii_hb_sii_ha_sub_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.good_oiii_hb_sii_ha_sub_plus] ,
              ccode=np.log10(sfrm_gsw2.siiflux_sub_plus/sfrm_gsw2.halpflux_sub_plus)[sfrm_gsw2.good_oiii_hb_sii_ha_sub_plus] , 
              filename='_fluxsub_sii_ha_plus', save=True, nx=25, ny=20, ccodename=r'log([SII]/H$\alpha$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.good_oiii_hb_sii_ha_sub_neither] ,  
              sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.good_oiii_hb_sii_ha_sub_neither] ,
              ccode=np.log10(sfrm_gsw2.siiflux_sub_neither/sfrm_gsw2.halpflux_sub_neither)[sfrm_gsw2.good_oiii_hb_sii_ha_sub_neither] , 
              filename='_fluxsub_sii_ha_neither', save=True, nx=25, ny=20, ccodename=r'log([SII]/H$\alpha$)') 
#[SII]/Ha presub
plotbptnormal(sfrm_gsw2.log_nii_halpha[sfrm_gsw2.good_oiii_hb_sii_ha_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta[sfrm_gsw2.good_oiii_hb_sii_ha_bpt] ,
              ccode=np.log10(sfrm_gsw2.siiflux/sfrm_gsw2.halpflux)[sfrm_gsw2.good_oiii_hb_sii_ha_bpt] , 
              filename='_sii_ha_bpt', save=True, nx=25, ny=20, ccodename=r'log([SII]/H$\alpha$)') 
plotbptnormal(sfrm_gsw2.log_nii_halpha_plus[sfrm_gsw2.good_oiii_hb_sii_ha_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_plus[sfrm_gsw2.good_oiii_hb_sii_ha_plus] ,
              ccode=np.log10(sfrm_gsw2.siiflux_plus/sfrm_gsw2.halpflux_plus)[sfrm_gsw2.good_oiii_hb_sii_ha_plus] , 
              filename='_sii_ha_plus', save=True, nx=25, ny=20, ccodename=r'log([SII]/H$\alpha$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_neither[sfrm_gsw2.good_oiii_hb_sii_ha_neither] ,  
              sfrm_gsw2.log_oiii_hbeta_neither[sfrm_gsw2.good_oiii_hb_sii_ha_neither] ,
              ccode=np.log10(sfrm_gsw2.siiflux_neither/sfrm_gsw2.halpflux_neither)[sfrm_gsw2.good_oiii_hb_sii_ha_neither] , 
              filename='_sii_ha_neither', save=True, nx=25, ny=20, ccodename=r'log([SII]/H$\alpha$)') 

#[OIII]/[OII] sub
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub[sfrm_gsw2.good_oiii_oii_nii_sub_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.good_oiii_oii_nii_sub_bpt] ,
              ccode=np.log10(sfrm_gsw2.oiiiflux_sub_dered_bpt/sfrm_gsw2.oiiflux_sub_dered_bpt)[sfrm_gsw2.good_oiii_oii_nii_sub_bpt] , 
              filename='_fluxsub_oiii_oii_bpt', save=True, nx=25, ny=20, ccodename=r'log([OIII]/[OII])') 
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.good_oiii_oii_nii_sub_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.good_oiii_oii_nii_sub_plus] ,
              ccode=np.log10(sfrm_gsw2.oiiiflux_sub_dered_plus/sfrm_gsw2.oiiflux_sub_dered_plus)[sfrm_gsw2.good_oiii_oii_nii_sub_plus] , 
              filename='_fluxsub_oiii_oii_plus', save=True, nx=25, ny=20, ccodename=r'log([OIII]/[OII])')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.good_oiii_oii_nii_sub_neither] ,  
              sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.good_oiii_oii_nii_sub_neither] ,
              ccode=np.log10(sfrm_gsw2.oiiiflux_sub_dered_neither/sfrm_gsw2.oiiflux_sub_dered_neither)[sfrm_gsw2.good_oiii_oii_nii_sub_neither] , 
              filename='_fluxsub_oiii_oii_neither', save=True, nx=25, ny=20, ccodename=r'log([OIII]/[OII])') 
#[OIII]/[OII] presub

plotbptnormal(sfrm_gsw2.log_nii_halpha[sfrm_gsw2.good_oiii_oii_nii_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta[sfrm_gsw2.good_oiii_oii_nii_bpt] ,
              ccode=np.log10(sfrm_gsw2.oiiiflux_dered_bpt/sfrm_gsw2.oiiflux_dered_bpt)[sfrm_gsw2.good_oiii_oii_nii_bpt] , 
              filename='__oiii_oii_bpt', save=True, nx=25, ny=20, ccodename=r'log([OIII]/[OII])') 
plotbptnormal(sfrm_gsw2.log_nii_halpha_plus[sfrm_gsw2.good_oiii_oii_nii_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_plus[sfrm_gsw2.good_oiii_oii_nii_plus] ,
              ccode=np.log10(sfrm_gsw2.oiiiflux_dered_plus/sfrm_gsw2.oiiflux_dered_plus)[sfrm_gsw2.good_oiii_oii_nii_plus] , 
              filename='_oiii_oii_plus', save=True, nx=25, ny=20, ccodename=r'log([OIII]/[OII])')
plotbptnormal(sfrm_gsw2.log_nii_halpha_neither[sfrm_gsw2.good_oiii_oii_nii_neither] ,  
              sfrm_gsw2.log_oiii_hbeta_neither[sfrm_gsw2.good_oiii_oii_nii_neither] ,
              ccode=np.log10(sfrm_gsw2.oiiiflux_dered_neither/sfrm_gsw2.oiiflux_dered_neither)[sfrm_gsw2.good_oiii_oii_nii_neither] , 
              filename='_oiii_oii_neither', save=True, nx=25, ny=20, ccodename=r'log([OIII]/[OII])') 

#[NII]/[OII] sub
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub[sfrm_gsw2.good_oiii_oii_nii_sub_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.good_oiii_oii_nii_sub_bpt] ,
              ccode=np.log10(sfrm_gsw2.niiflux_sub_dered_bpt/sfrm_gsw2.oiiflux_sub_dered_bpt)[sfrm_gsw2.good_oiii_oii_nii_sub_bpt] , 
              filename='_fluxsub_nii_oii_bpt', save=True, nx=25, ny=20, ccodename=r'log([NII]/[OII])') 
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.good_oiii_oii_nii_sub_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.good_oiii_oii_nii_sub_plus] ,
              ccode=np.log10(sfrm_gsw2.niiflux_sub_dered_plus/sfrm_gsw2.oiiflux_sub_dered_plus)[sfrm_gsw2.good_oiii_oii_nii_sub_plus] , 
              filename='_fluxsub_nii_oii_plus', save=True, nx=25, ny=20, ccodename=r'log([NII]/[OII])')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.good_oiii_oii_nii_sub_neither] ,  
              sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.good_oiii_oii_nii_sub_neither] ,
              ccode=np.log10(sfrm_gsw2.niiflux_sub_dered_neither/sfrm_gsw2.oiiflux_sub_dered_neither)[sfrm_gsw2.good_oiii_oii_nii_sub_neither] , 
              filename='_fluxsub_nii_oii_neither', save=True, nx=25, ny=20, ccodename=r'log([NII]/[OII])') 
#[NII]/[OII] presub
plotbptnormal(sfrm_gsw2.log_nii_halpha[sfrm_gsw2.good_oiii_oii_nii_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta[sfrm_gsw2.good_oiii_oii_nii_bpt] ,
              ccode=np.log10(sfrm_gsw2.niiflux_dered_bpt/sfrm_gsw2.oiiflux_dered_bpt)[sfrm_gsw2.good_oiii_oii_nii_bpt] , 
              filename='_nii_oii_bpt', save=True, nx=25, ny=20, ccodename=r'log([NII]/[OII])') 
plotbptnormal(sfrm_gsw2.log_nii_halpha_plus[sfrm_gsw2.good_oiii_oii_nii_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_plus[sfrm_gsw2.good_oiii_oii_nii_plus] ,
              ccode=np.log10(sfrm_gsw2.niiflux_dered_plus/sfrm_gsw2.oiiflux_dered_plus)[sfrm_gsw2.good_oiii_oii_nii_plus] , 
              filename='_nii_oii_plus', save=True, nx=25, ny=20, ccodename=r'log([NII]/[OII])')
plotbptnormal(sfrm_gsw2.log_nii_halpha_neither[sfrm_gsw2.good_oiii_oii_nii_neither] ,  
              sfrm_gsw2.log_oiii_hbeta_neither[sfrm_gsw2.good_oiii_oii_nii_neither] ,
              ccode=np.log10(sfrm_gsw2.niiflux_dered_neither/sfrm_gsw2.oiiflux_dered_neither)[sfrm_gsw2.good_oiii_oii_nii_neither] , 
              filename='_nii_oii_neither', save=True, nx=25, ny=20, ccodename=r'log([NII]/[OII])') 

#[OI]/Ha
plotbptnormal(sfrm_gsw2.log_nii_halpha[sfrm_gsw2.good_oi_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta[sfrm_gsw2.good_oi_bpt] ,
              ccode=np.log10(sfrm_gsw2.oiflux[sfrm_gsw2.good_oi_bpt]/sfrm_gsw2.halpflux[sfrm_gsw2.good_oi_bpt]) , 
              filename='_oi_ha_bpt', save=True, nx=25, ny=20, ccodename=r'log([OI]/H$\alpha$)')

plotbptnormal(sfrm_gsw2.log_nii_halpha_plus[sfrm_gsw2.good_oi_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_plus[sfrm_gsw2.good_oi_plus] ,
              ccode=np.log10(sfrm_gsw2.oiflux_plus[sfrm_gsw2.good_oi_plus]/sfrm_gsw2.halpflux_plus[sfrm_gsw2.good_oi_plus]) , 
              filename='_oi_ha_plus', save=True, nx=25, ny=20, ccodename=r'log([OI]/H$\alpha$)')

plotbptnormal(sfrm_gsw2.log_nii_halpha_neither[sfrm_gsw2.good_oi_neither],  
              sfrm_gsw2.log_oiii_hbeta_neither[sfrm_gsw2.good_oi_neither],
              ccode=np.log10(sfrm_gsw2.oiflux_neither[sfrm_gsw2.good_oi_neither]/sfrm_gsw2.halpflux_neither[sfrm_gsw2.good_oi_neither]) , 
              filename='_oi_ha_neither', save=True, nx=25, ny=20, ccodename=r'log([OI]/H$\alpha$)')
#[OI]/Ha sub
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub[sfrm_gsw2.good_oi_sub_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.good_oi_sub_bpt] ,
              ccode=np.log10(sfrm_gsw2.oiflux_sub[sfrm_gsw2.good_oi_sub_bpt]/sfrm_gsw2.halpflux_sub[sfrm_gsw2.good_oi_sub_bpt]) , 
              filename='_fluxsub_oi_ha_sub_bpt', save=True, nx=25, ny=20, ccodename=r'log([OI]/H$\alpha$)')

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.good_oi_sub_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.good_oi_sub_plus] ,
              ccode=np.log10(sfrm_gsw2.oiflux_sub_plus[sfrm_gsw2.good_oi_sub_plus]/sfrm_gsw2.halpflux_sub_plus[sfrm_gsw2.good_oi_sub_plus]) , 
              filename='_flubsub_oi_ha_plus', save=1, nx=25, ny=20, ccodename=r'log([OI]/H$\alpha$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.good_oi_sub_neither],  
              sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.good_oi_sub_neither],
              ccode=np.log10(sfrm_gsw2.oiflux_sub_neither[sfrm_gsw2.good_oi_sub_neither]/sfrm_gsw2.halpflux_sub_neither[sfrm_gsw2.good_oi_sub_neither]) , 
              filename='_fluxsub_oi_ha_neither', save=1, nx=25, ny=20, ccodename=r'log([OI]/H$\alpha$)')



#[OI] lum
plotbptnormal(sfrm_gsw2.log_nii_halpha[sfrm_gsw2.good_oi_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta[sfrm_gsw2.good_oi_bpt] ,
              ccode=np.log10(sfrm_gsw2.oilum_dered_bpt[sfrm_gsw2.good_oi_bpt]) , 
              filename='_oi_lum_bpt', save=True, nx=25, ny=20, ccodename=r'log(L$_{\mathrm{[OI]}}$)')

plotbptnormal(sfrm_gsw2.log_nii_halpha_plus[sfrm_gsw2.good_oi_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_plus[sfrm_gsw2.good_oi_plus] ,
              ccode=np.log10(sfrm_gsw2.oilum_dered_plus[sfrm_gsw2.good_oi_plus]) , 
              filename='_oi_lum_plus', save=True, nx=25, ny=20, ccodename=r'log(L$_{\mathrm{[OI]}}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_neither[sfrm_gsw2.good_oi_neither],  
              sfrm_gsw2.log_oiii_hbeta_neither[sfrm_gsw2.good_oi_neither],
              ccode=np.log10(sfrm_gsw2.oilum_dered_neither[sfrm_gsw2.good_oi_neither]) , 
              filename='_oi_lum_neither', save=1, nx=25, ny=20, ccodename=r'log(L$_{\mathrm{[OI]}}$)')
#[OI] lum sub
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub[sfrm_gsw2.good_oi_sub_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta_sub[sfrm_gsw2.good_oi_sub_bpt] ,
              ccode=np.log10(sfrm_gsw2.oilum_sub_dered_bpt[sfrm_gsw2.good_oi_sub_bpt]) , 
              filename='_fluxsub_oi_lum_sub_bpt', save=True, nx=25, ny=20, ccodename=r'log(L$_{\mathrm{[OI]}}$)')

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus[sfrm_gsw2.good_oi_sub_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_sub_plus[sfrm_gsw2.good_oi_sub_plus] ,
              ccode=np.log10(sfrm_gsw2.oilum_sub_dered_plus[sfrm_gsw2.good_oi_sub_plus]) , 
              filename='_fluxsub_oi_lum_sub_plus', save=True, nx=25, ny=20, ccodename=r'log(L$_{\mathrm{[OI]}}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither[sfrm_gsw2.good_oi_sub_neither],  
              sfrm_gsw2.log_oiii_hbeta_sub_neither[sfrm_gsw2.good_oi_sub_neither],
              ccode=np.log10(sfrm_gsw2.oilum_sub_dered_neither[sfrm_gsw2.good_oi_sub_neither]) , 
              filename='_fluxsub_oi_lum_sub_neither', save=1, nx=25, ny=20, ccodename=r'log(L$_{\mathrm{[OI]}}$)')

#balmer dec

plotbptnormal(sfrm_gsw2.log_nii_halpha[sfrm_gsw2.good_oiii_oii_nii_bpt] ,  
              sfrm_gsw2.log_oiii_hbeta[sfrm_gsw2.good_oiii_oii_nii_bpt] ,
              ccode=sfrm_gsw2.av_sub_bpt_agn ,
              filename='_av_oii_bpt', save=False, nx=25, ny=20, ccodename=r'log([NII]/[OII])') 
plotbptnormal(sfrm_gsw2.log_nii_halpha_plus[sfrm_gsw2.good_oiii_oii_nii_plus] ,  
              sfrm_gsw2.log_oiii_hbeta_plus[sfrm_gsw2.good_oiii_oii_nii_plus] ,
              ccode=np.log10(sfrm_gsw2.niiflux_dered_plus/sfrm_gsw2.oiiflux_dered_plus)[sfrm_gsw2.good_oiii_oii_nii_plus] , 
              filename='_nii_oii_plus', save=True, nx=25, ny=20, ccodename=r'log([NII]/[OII])')
plotbptnormal(sfrm_gsw2.log_nii_halpha_neither[sfrm_gsw2.good_oiii_oii_nii_neither] ,  
              sfrm_gsw2.log_oiii_hbeta_neither[sfrm_gsw2.good_oiii_oii_nii_neither] ,
              ccode=np.log10(sfrm_gsw2.niiflux_dered_neither/sfrm_gsw2.oiiflux_dered_neither)[sfrm_gsw2.good_oiii_oii_nii_neither] , 
              filename='_nii_oii_neither', save=True, nx=25, ny=20, ccodename=r'log([NII]/[OII])') 


#match dists
plotbptnormal(sfrm_gsw2.log_nii_halpha_sing[sfrm_gsw2.agn_selfplus_dist_inds] ,  
              sfrm_gsw2.log_oiii_hbeta_sing[sfrm_gsw2.agn_selfplus_dist_inds] ,
              ccode=sfrm_gsw2.mindistsagn_best[sfrm_gsw2.agn_selfplus_dist_inds] , 
              filename='_agnplus_matchdist', save=False, nx=25, ny=20, ccodename=r'AGN+ Match Dist.')

#mpa fib ssfr

plotbptnormal(sfrm_gsw2.log_nii_halpha , 
              sfrm_gsw2.log_oiii_hbeta ,
              ccode=sfrm_gsw2.fibssfr_mpa_bpt , 
              filename='_mpa_fib_ssfr_bpt', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{\mathrm{Fib,MPA}}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_plus,  
              sfrm_gsw2.log_oiii_hbeta_plus ,
              ccode=sfrm_gsw2.fibssfr_mpa_plus , 
              filename='_mpa_fib_ssfr_plus', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{\mathrm{Fib,MPA}}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_neither,  
              sfrm_gsw2.log_oiii_hbeta_neither ,
              ccode=sfrm_gsw2.fibssfr_mpa_neither , 
              filename='_mpa_fib_ssfr_neither', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{\mathrm{Fib,MPA}}$)')
#mpa fib sfr
plotbptnormal(sfrm_gsw2.log_nii_halpha , 
              sfrm_gsw2.log_oiii_hbeta ,
              ccode=sfrm_gsw2.fibsfr_mpa_bpt , 
              filename='_mpa_fib_sfr_bpt', save=True, nx=25, ny=20, ccodename=r'log(SFR$_{\mathrm{Fib,MPA}}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_plus,  
              sfrm_gsw2.log_oiii_hbeta_plus ,
              ccode=sfrm_gsw2.fibsfr_mpa_plus , 
              filename='_mpa_fib_sfr_plus', save=True, nx=25, ny=20, ccodename=r'log(SFR$_{\mathrm{Fib,MPA}}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_neither,  
              sfrm_gsw2.log_oiii_hbeta_neither ,
              ccode=sfrm_gsw2.fibsfr_mpa_neither , 
              filename='_mpa_fib_sfr_neither', save=True, nx=25, ny=20, ccodename=r'log(SFR$_{\mathrm{Fib,MPA}}$)')

#mpa fib sub ssfr

plotbptnormal(sfrm_gsw2.log_nii_halpha_sub , 
              sfrm_gsw2.log_oiii_hbeta_sub ,
              ccode=sfrm_gsw2.fibssfr_mpa_bpt , 
              filename='_fluxsub_mpa_fib_ssfr_bpt', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{\mathrm{Fib,MPA}}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus,  
              sfrm_gsw2.log_oiii_hbeta_sub_plus ,
              ccode=sfrm_gsw2.fibssfr_mpa_plus , 
              filename='_fluxsub_mpa_fib_ssfr_plus', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{\mathrm{Fib,MPA}}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither,  
              sfrm_gsw2.log_oiii_hbeta_sub_neither ,
              ccode=sfrm_gsw2.fibssfr_mpa_neither , 
              filename='_fluxsub_mpa_fib_ssfr_neither', save=True, nx=25, ny=20, ccodename=r'log(sSFR$_{\mathrm{Fib,MPA}}$)')
#mpa fib sfr
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub , 
              sfrm_gsw2.log_oiii_hbeta_sub ,
              ccode=sfrm_gsw2.fibsfr_mpa_bpt , 
              filename='_fluxsub_mpa_fib_sfr_bpt', save=True, nx=25, ny=20, ccodename=r'log(SFR$_{\mathrm{Fib,MPA}}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_plus,  
              sfrm_gsw2.log_oiii_hbeta_sub_plus ,
              ccode=sfrm_gsw2.fibsfr_mpa_plus , 
              filename='_fluxsub_mpa_fib_sfr_plus', save=True, nx=25, ny=20, ccodename=r'log(SFR$_{\mathrm{Fib,MPA}}$)')
plotbptnormal(sfrm_gsw2.log_nii_halpha_sub_neither,  
              sfrm_gsw2.log_oiii_hbeta_sub_neither ,
              ccode=sfrm_gsw2.fibsfr_mpa_neither , 
              filename='_fluxsub_mpa_fib_sfr_neither', save=True, nx=25, ny=20, ccodename=r'log(SFR$_{\mathrm{Fib,MPA}}$)')

#halplum

plotbptnormal(sfrm_gsw2.log_nii_halpha , 
              sfrm_gsw2.log_oiii_hbeta ,
              ccode=sfrm_gsw2.fibssfr_mpa_bpt , 
              filename='_mpa_fib_ssfr_bpt', save=True, nx=25, ny=20, ccodename=r'log(SFR$_{\mathrm{Fib,MPA}}$)')

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

def dust_oiiilum_sub(bgx,bgy,save=True,filename='',title=None, leg=False):
    '''
    for doing A_V for AGN versus OIII lum
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -6) &( bgy < 6) & (bgx<44)&(bgx > 36) )[0]
    nx = 8/0.04
    ny = 12/0.06
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, bin_y=True, bin_stat_y='median')
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([-6, 6])
    plt.xlim([36, 44])
    plt.plot([36,44],[-6,6], c='k',ls='-.')
    plt.ylabel(r'A$_{\rm V, AGN-match}$',fontsize=20)
    plt.xlabel(r'L$_{\mathrm{[OIII]}}$',fontsize=20)

    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/dustcomp/dustagn_oiiilum'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/dustcomp/dustagn_oiiilum'+filename+'.pdf',dpi=250,bbox_inches='tight', format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/dustcomp/dustagn_oiiilum '+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''
dust_oiiilum_sub(np.log10(sfrm_gsw2.oiiilum_sub_bpt),  sfrm_gsw2.av_sub_bpt_agn,save=True, filename= '_fluxsub_bpt')
dust_oiiilum_sub(np.log10(sfrm_gsw2.oiiilum_bpt),  sfrm_gsw2.av_sub_bpt_agn,save=True, filename= '_presub_bpt')

dust_oiiilum_sub(np.log10(sfrm_gsw2.oiiilum_sub_plus),  sfrm_gsw2.av_sub_plus_agn,save=True, filename= '_fluxsub_plus')
dust_oiiilum_sub(np.log10(sfrm_gsw2.oiiilum_plus),  sfrm_gsw2.av_sub_plus_agn,save=True, filename= '_presub_plus')

dust_oiiilum_sub(np.log10(sfrm_gsw2.oiiilum_sub_neither),  sfrm_gsw2.av_sub_neither_agn,save=True, filename= '_fluxsub_neither')
dust_oiiilum_sub(np.log10(sfrm_gsw2.oiiilum_neither),  sfrm_gsw2.av_sub_neither_agn,save=True, filename= '_presub_neither')

dust_oiiilum_sub(np.log10(sfrm_gsw2.oiiilum_dered_bpt),  sfrm_gsw2.av_sub_bpt_agn,save=False, filename= '_fluxsub_bpt')

plt.errorbar(sfrm_gsw2.av_bpt_sf_bncenters[sfrm_gsw2.av_bpt_sf_valbns],  7.23*np.log10(sfrm_gsw2.bootstrapped_halphbeta/3.1), 
             yerr= 7.23*sfrm_gsw2.bootstrapped_halphbeta_err/(sfrm_gsw2.bootstrapped_halphbeta*np.log(10)), color='k', capsize=2)
            av = 7.23*np.log10((ha/hb) / dec_rat) # A(V) mag

dust_comp_sf_sub( sfrm_gsw2.av_plus_sf,sfrm_gsw2.av_sub_plus_agn, save=False, filename= '_fluxsub_plus')
dust_comp_sf_sub(  sfrm_gsw2.av_neither_sf ,sfrm_gsw2.av_sub_neither_agn,save=False, filename='_fluxsub_neither')

dust_comp_sf_sub( sfrm_gsw2.av_bpt_sf[sfrm_gsw2.halphbeta_sub_sn_filt_bpt],sfrm_gsw2.av_sub_bpt_agn[sfrm_gsw2.halphbeta_sub_sn_filt_bpt],save=True, filename= '_fluxsub_bpt_sn4')
dust_comp_sf_sub( sfrm_gsw2.av_plus_sf[sfrm_gsw2.halphbeta_sub_sn_filt_plus], sfrm_gsw2.av_sub_plus_agn[sfrm_gsw2.halphbeta_sub_sn_filt_plus], save=True, filename= '_fluxsub_plus_sn4')
dust_comp_sf_sub( sfrm_gsw2.av_neither_sf[sfrm_gsw2.halphbeta_sub_sn_filt_neither], sfrm_gsw2.av_sub_neither_agn[sfrm_gsw2.halphbeta_sub_sn_filt_neither], save=True, filename='_fluxsub_neither_sn4')

dust_comp_sf_sub( sfrm_gsw2.av_bpt_sf[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_bpt],sfrm_gsw2.av_sub_bpt_agn[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_bpt],save=True, filename= '_fluxsub_bothaxes_filt_bpt_sn10')
dust_comp_sf_sub( sfrm_gsw2.av_plus_sf[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_plus], sfrm_gsw2.av_sub_plus_agn[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_plus], save=True, filename= '_fluxsub_bothaxes_filt_plus_sn10')
dust_comp_sf_sub( sfrm_gsw2.av_neither_sf[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_neither], sfrm_gsw2.av_sub_neither_agn[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_neither], save=True, filename='_fluxsub_bothaxes_filt_neither_sn10')

'''


def dust_comp_sf_sub(bgx,bgy,save=True,filename='',title=None, leg=False):
    '''
    for doing A_V for SF versus AGN
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -6) &( bgy < 6) & (bgx<6)&(bgx > -6) )[0]
    nx = 12/0.05
    ny = 12/0.05
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, bin_y=True, bin_stat_y='median')
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([-6, 6])
    plt.xlim([-6, 6])
    plt.plot([-6,6],[-6,6], c='k',ls='-.')
    plt.ylabel(r'A$_{\rm V, AGN-match}$',fontsize=20)
    plt.xlabel(r'A$_{\rm V, match}$',fontsize=20)

    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/dustcomp/dustcomp_sf_sub'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/dustcomp/dustcomp_sf_sub'+filename+'.pdf',dpi=250,bbox_inches='tight', format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/dustcomp/dustcomp_sf_sub'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''
dust_comp_sf_sub(sfrm_gsw2.av_bpt_sf[sfrm_gsw2.filt_by_av_bpt],  sfrm_gsw2.av_sub_bpt_agn[sfrm_gsw2.filt_by_av_bpt],save=False, filename= '_fluxsub_bpt')
plt.errorbar(sfrm_gsw2.av_bpt_sf_bncenters[sfrm_gsw2.av_bpt_sf_valbns],  7.23*np.log10(sfrm_gsw2.bootstrapped_halphbeta/3.1), 
             yerr= 7.23*sfrm_gsw2.bootstrapped_halphbeta_err/(sfrm_gsw2.bootstrapped_halphbeta*np.log(10)), color='k', capsize=3, capthick=1, zorder=10)
plt.savefig('plots/sfrmatch/png/dustcomp/dustcomp_sf_sub_fluxsub_bpt_bootstrap0.3_sn10.png', dpi=250,bbox_inches='tight')


dust_comp_sf_sub(sfrm_gsw2.av_plus_sf[sfrm_gsw2.filt_by_av_plus],  sfrm_gsw2.av_sub_plus_agn[sfrm_gsw2.filt_by_av_plus],save=False, filename= '_fluxsub_plus')
plt.errorbar(sfrm_gsw2.av_plus_sf_bncenters[sfrm_gsw2.av_plus_sf_valbns],  7.23*np.log10(sfrm_gsw2.bootstrapped_halphbeta_plus/3.1), 
             yerr= 7.23*sfrm_gsw2.bootstrapped_halphbeta_err_plus/(sfrm_gsw2.bootstrapped_halphbeta_plus*np.log(10)), color='k', capsize=3, capthick=1, zorder=10)
plt.savefig('plots/sfrmatch/png/dustcomp/dustcomp_sf_sub_fluxsub_plus_bootstrap0.3_sn10.png', dpi=250,bbox_inches='tight')

dust_comp_sf_sub(sfrm_gsw2.av_neither_sf[sfrm_gsw2.filt_by_av_neither],  sfrm_gsw2.av_sub_neither_agn[sfrm_gsw2.filt_by_av_neither],save=False, filename= '_fluxsub_neither')
plt.errorbar(sfrm_gsw2.av_neither_sf_bncenters[sfrm_gsw2.av_neither_sf_valbns],  7.23*np.log10(sfrm_gsw2.bootstrapped_halphbeta_neither/3.1), 
             yerr= 7.23*sfrm_gsw2.bootstrapped_halphbeta_err_neither/(sfrm_gsw2.bootstrapped_halphbeta_neither*np.log(10)), color='k', capsize=3, capthick=1, zorder=10)
plt.savefig('plots/sfrmatch/png/dustcomp/dustcomp_sf_sub_fluxsub_neither_bootstrap0.3_sn10.png', dpi=250,bbox_inches='tight')


dust_comp_sf_sub( sfrm_gsw2.av_plus_sf,sfrm_gsw2.av_sub_plus_agn, save=False, filename= '_fluxsub_plus')
dust_comp_sf_sub(  sfrm_gsw2.av_neither_sf ,sfrm_gsw2.av_sub_neither_agn,save=False, filename='_fluxsub_neither')

dust_comp_sf_sub( sfrm_gsw2.av_bpt_sf[sfrm_gsw2.halphbeta_sub_sn_filt_bpt],sfrm_gsw2.av_sub_bpt_agn[sfrm_gsw2.halphbeta_sub_sn_filt_bpt],save=True, filename= '_fluxsub_bpt_sn4')
dust_comp_sf_sub( sfrm_gsw2.av_plus_sf[sfrm_gsw2.halphbeta_sub_sn_filt_plus], sfrm_gsw2.av_sub_plus_agn[sfrm_gsw2.halphbeta_sub_sn_filt_plus], save=True, filename= '_fluxsub_plus_sn4')
dust_comp_sf_sub( sfrm_gsw2.av_neither_sf[sfrm_gsw2.halphbeta_sub_sn_filt_neither], sfrm_gsw2.av_sub_neither_agn[sfrm_gsw2.halphbeta_sub_sn_filt_neither], save=True, filename='_fluxsub_neither_sn4')

dust_comp_sf_sub( sfrm_gsw2.av_bpt_sf[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_bpt],sfrm_gsw2.av_sub_bpt_agn[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_bpt],save=True, filename= '_fluxsub_bothaxes_filt_bpt_sn10')
dust_comp_sf_sub( sfrm_gsw2.av_plus_sf[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_plus], sfrm_gsw2.av_sub_plus_agn[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_plus], save=True, filename= '_fluxsub_bothaxes_filt_plus_sn10')
dust_comp_sf_sub( sfrm_gsw2.av_neither_sf[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_neither], sfrm_gsw2.av_sub_neither_agn[sfrm_gsw2.halphbeta_sub_and_match_sn_filt_neither], save=True, filename='_fluxsub_bothaxes_filt_neither_sn10')

'''


def dust_comp_agn_sub(bgx,bgy,save=True,filename='',title=None, leg=False):
    '''
    for doing A_V for SF versus AGN
    '''
    fig = plt.figure()
    ax= fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -6) &( bgy < 6) & (bgx<6)&(bgx > -6) )[0]
    nx = 12/0.1
    ny = 12/0.1
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, bin_y=True, bin_stat_y='median', ax=ax)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    plt.plot([-6,6],[-6,6], c='k',ls='-.')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([-6, 6])
    plt.xlim([-6, 6])
    plt.xlabel(r'A$_{\rm V, AGN-match}$',fontsize=20)
    plt.ylabel(r'A$_{\rm V, AGN}$',fontsize=20)

    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/dustcomp/dustcomp_agn_sub'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/dustcomp/dustcomp_agn_sub'+filename+'.pdf',dpi=250,bbox_inches='tight', format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/dustcomp/dustcomp_agn_sub'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''
dust_comp_agn_sub(sfrm_gsw2.av_sub_bpt_agn,sfrm_gsw2.av_bpt_agn, save=True, filename= '_fluxsub_bpt')
dust_comp_agn_sub(sfrm_gsw2.av_sub_plus_agn, sfrm_gsw2.av_plus_agn, save=True, filename= '_fluxsub_plus')
dust_comp_agn_sub(sfrm_gsw2.av_sub_neither_agn, sfrm_gsw2.av_neither_agn, save=True, filename='_fluxsub_neither')
'''



def dust_comp_vs_line_sn(bgx,bgy,line, save=True,filename='',title=None, leg=False):
    '''
    for doing A_V for SF versus AGN
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -6) &( bgy < 6) & (bgx<150)&(bgx > -1) )[0]
    nx = 7.5/0.1
    ny = 6/0.1
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, bin_y=False, bin_stat_y='median', ax=ax)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([-6, 6])
    plt.xlim([0, 150])
    plt.xlabel(line+ r' S/N',fontsize=20)
    plt.ylabel(r'A$_{\rm V, AGN}$',fontsize=20)
    plt.plot([-6,6],[-6,6], c='k',ls='-.')

    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/dustcomp/dust_vs_line_sn_'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/dustcomp/dust_vs_line_sn_'+filename+'.pdf',dpi=250,bbox_inches='tight', format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/dustcomp/dust_vs_line_sn_'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''
dust_comp_vs_line_sn(sfrm_gsw2.halp_sn, sfrm_gsw2.av_bpt_agn,r'H$\alpha$',save=True, filename= 'halp_bpt')
dust_comp_vs_line_sn(sfrm_gsw2.hbeta_sn, sfrm_gsw2.av_bpt_agn,r'H$\beta$',save=True, filename= 'hbeta_bpt')
dust_comp_vs_line_sn(sfrm_gsw2.halphbeta_sn, sfrm_gsw2.av_bpt_agn,r'$\frac{\mathrm{H}_{\alpha}}{\mathrm{H}_{\beta}}$',save=True, filename= 'halphbeta_bpt')

dust_comp_vs_line_sn(sfrm_gsw2.halp_sn_plus, sfrm_gsw2.av_plus_agn,r'H$\alpha$',save=True, filename= 'halp_plus')
dust_comp_vs_line_sn(sfrm_gsw2.hbeta_sn_plus, sfrm_gsw2.av_plus_agn,r'H$\beta$',save=True, filename= 'hbeta_plus')
dust_comp_vs_line_sn(sfrm_gsw2.halphbeta_sn_plus, sfrm_gsw2.av_plus_agn,r'H$\alpha$/H$\beta$',save=True, filename= 'halphbeta_plus')

dust_comp_vs_line_sn(sfrm_gsw2.halp_sn_neither, sfrm_gsw2.av_neither_agn,r'H$\alpha$',save=True, filename= 'halp_neither')
dust_comp_vs_line_sn(sfrm_gsw2.hbeta_sn_neither, sfrm_gsw2.av_neither_agn,r'H$\beta$',save=True, filename= 'hbeta_neither')
dust_comp_vs_line_sn(sfrm_gsw2.halphbeta_sn_neither, sfrm_gsw2.av_neither_agn,r'$\frac{\mathrm{H}_{\alpha}}{\mathrm{H}_{\beta}}$',save=True, filename= 'halphbeta_neither')


'''

def dust_comp_agnsub_vs_line_sn(bgx,bgy,line, save=True,filename='',title=None, leg=False):
    '''
    for doing A_V for SF versus AGN
    '''
    fig = plt.figure()
    ax= fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -6) &( bgy < 6) & (bgx<150)&(bgx > -1) )[0]
    nx = 7.5/0.1
    ny = 6/0.1
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, bin_y=False, bin_stat_y='median', ax=ax)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([-6, 6])
    plt.xlim([0, 150])
    plt.xlabel(line+ r' S/N',fontsize=20)
    plt.ylabel(r'A$_{\rm V, AGN-Match}$',fontsize=20)

    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')
    if save:
        fig.savefig('plots/sfrmatch/png/dustcomp/dust_vs_line_sn_'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/dustcomp/dust_vs_line_sn_'+filename+'.pdf',dpi=250,bbox_inches='tight', format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/dustcomp/dust_vs_line_sn_'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''

dust_comp_agnsub_vs_line_sn(sfrm_gsw2.halpflux_sub_sn, sfrm_gsw2.av_sub_bpt_agn,r'H$\alpha$',save=True, filename= 'fluxsub_halp_bpt')
dust_comp_agnsub_vs_line_sn(sfrm_gsw2.hbetaflux_sub_sn, sfrm_gsw2.av_sub_bpt_agn,r'H$\beta$',save=True, filename= 'fluxsub_hbeta_bpt')
dust_comp_agnsub_vs_line_sn(sfrm_gsw2.halphbeta_sub_sn, sfrm_gsw2.av_bpt_agn,r'$\frac{\mathrm{H}_{\alpha}}{\mathrm{H}_{\beta}}$',save=True, filename= 'fluxsub_halphbeta_bpt')

dust_comp_agnsub_vs_line_sn(sfrm_gsw2.halpflux_sub_sn_plus, sfrm_gsw2.av_sub_plus_agn,r'H$\alpha$',save=True, filename= 'fluxsub_halp_plus')
dust_comp_agnsub_vs_line_sn(sfrm_gsw2.hbetaflux_sub_sn_plus, sfrm_gsw2.av_sub_plus_agn,r'H$\beta$',save=True, filename= 'fluxsub_hbeta_plus')
dust_comp_agnsub_vs_line_sn(sfrm_gsw2.halphbeta_sub_sn_plus, sfrm_gsw2.av_sub_plus_agn,r'H$\alpha$/H$\beta$',save=True, filename= 'fluxsub_halphbeta_plus')

dust_comp_agnsub_vs_line_sn(sfrm_gsw2.halpflux_sub_sn_neither, sfrm_gsw2.av_sub_neither_agn,r'H$\alpha$',save=True, filename= 'fluxsub_halp_neither')
dust_comp_agnsub_vs_line_sn(sfrm_gsw2.hbetaflux_sub_sn_neither, sfrm_gsw2.av_sub_neither_agn,r'H$\beta$',save=True, filename= 'fluxsub_hbeta_neither')
dust_comp_agnsub_vs_line_sn(sfrm_gsw2.halphbeta_sub_sn_neither, sfrm_gsw2.av_sub_neither_agn,r'$\frac{\mathrm{H}_{\alpha}}{\mathrm{H}_{\beta}}$',save=True, filename= 'fluxsub_halphbeta_neither')

'''



def dust_comp_gsw_vs_dec(bgx,bgy,save=True,filename='',title=None, leg=False):
    '''
    for doing A_V for GSW versus balmer decrement
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -6) &( bgy < 6) & (bgx<6)&(bgx > -6) )[0]
    nx = 12/0.1
    ny = 12/0.1
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, bin_y=True, bin_stat_y='median', ax=ax)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([-6, 6])
    plt.xlim([-6, 6])
    plt.xlabel(r'A$_{\rm V, GSW}$',fontsize=20)
    plt.ylabel(r'A$_{\rm V, Balmer\ Dec.}$',fontsize=20)

    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/dustcomp/dustcomp_gsw_vs_dec'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/dustcomp/dustcomp_gsw_vs_dec'+filename+'.pdf',dpi=250,bbox_inches='tight', format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/dustcomp/dustcomp_gsw_vs_dec'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''
dust_comp_gsw_vs_dec(sfrm_gsw2.av_gsw_bpt_agn,sfrm_gsw2.av_bpt_agn, save=True, filename= '_bpt')
dust_comp_gsw_vs_dec(sfrm_gsw2.av_gsw_plus_agn, sfrm_gsw2.av_plus_agn, save=True, filename= '_plus')
dust_comp_gsw_vs_dec(sfrm_gsw2.av_gsw_neither_agn, sfrm_gsw2.av_neither_agn, save=True, filename='_neither')
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

plt.plot(x,x*m_ssfr+b_ssfr,'k', label='Main Sequence')
plt.plot(x,x*m_ssfr+b_ssfr-0.7,'k-.', label='Main Sequence Cutoff')
plotssfrm(m2Cat_GSW_3xmm.allmass[mpa_spec_allm2.make_prac], (mdisp2Cat_GSW_3xmm.allsfr-m2Cat_GSW_3xmm.allmass)[mpa_spec_allm2.make_prac], save=False, leg=True)


'''

def plotU_OH(bgx,bgy,save=True,filename='', title=None, leg=False, ccode=[], nx=50, ny=50):
    '''
    for doing U ([OIII]/[OII]) vs. O/H ([NII]/[OII]) diagram with sdss galaxies scatter plotted
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -1.5) &( bgy < 1.5) & (bgx<1.5)&(bgx > -1.5) )[0]
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ccode=ccode)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylabel(r'log([OIII]/[OII])',fontsize=20)
    plt.xlabel(r'log([NII]/[OII])',fontsize=20)
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])

    if title:
        plt.title(title,fontsize=30)
    if leg:
        plt.legend(fontsize=15,frameon = False, loc=3, bbox_to_anchor=(-0.04, .05))    
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/diagnostic/oiii_oii_nii_oii'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/diagnostic/oiii_oii_nii_oii'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/diagnostic/oiii_oii_nii_oii'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:

        plt.show()
'''


plotU_OH(np.log10(sfrm_gsw2.niiflux_sub_dered_bpt/sfrm_gsw2.oiiflux_sub_dered_bpt)[sfrm_gsw2.good_oiii_oii_nii_sub_bpt], 
         np.log10(sfrm_gsw2.oiiiflux_sub_dered_bpt/sfrm_gsw2.oiiflux_sub_dered_bpt)[sfrm_gsw2.good_oiii_oii_nii_sub_bpt], 
         save=True,filename='_fluxsub_bpt')
plotU_OH(np.log10(sfrm_gsw2.niiflux_sub_dered_plus/sfrm_gsw2.oiiflux_sub_dered_plus)[sfrm_gsw2.good_oiii_oii_nii_sub_plus], 
         np.log10(sfrm_gsw2.oiiiflux_sub_dered_plus/sfrm_gsw2.oiiflux_sub_dered_plus)[sfrm_gsw2.good_oiii_oii_nii_sub_plus], 
         save=True,filename='_fluxsub_plus')
plotU_OH(np.log10(sfrm_gsw2.niiflux_sub_dered_neither/sfrm_gsw2.oiiflux_sub_dered_neither)[sfrm_gsw2.good_oiii_oii_nii_sub_neither], 
         np.log10(sfrm_gsw2.oiiiflux_sub_dered_neither/sfrm_gsw2.oiiflux_sub_dered_neither)[sfrm_gsw2.good_oiii_oii_nii_sub_neither], nx=50,ny=50,
         save=True,filename='_fluxsub_neither')

plotU_OH(np.log10(sfrm_gsw2.niiflux_dered_bpt/sfrm_gsw2.oiiflux_dered_bpt)[sfrm_gsw2.good_oiii_oii_nii_bpt], 
         np.log10(sfrm_gsw2.oiiiflux_dered_bpt/sfrm_gsw2.oiiflux_dered_bpt)[sfrm_gsw2.good_oiii_oii_nii_bpt], 
         save=True,filename='_bpt')
plotU_OH(np.log10(sfrm_gsw2.niiflux_dered_plus/sfrm_gsw2.oiiflux_dered_plus)[sfrm_gsw2.good_oiii_oii_nii_plus], 
         np.log10(sfrm_gsw2.oiiiflux_dered_plus/sfrm_gsw2.oiiflux_dered_plus)[sfrm_gsw2.good_oiii_oii_nii_plus], 
         save=True,filename='_plus')
plotU_OH(np.log10(sfrm_gsw2.niiflux_dered_neither/sfrm_gsw2.oiiflux_dered_neither)[sfrm_gsw2.good_oiii_oii_nii_neither], 
         np.log10(sfrm_gsw2.oiiiflux_dered_neither/sfrm_gsw2.oiiflux_dered_neither)[sfrm_gsw2.good_oiii_oii_nii_neither], nx=50,ny=50,
         save=True,filename='_neither')




'''

def plot_sfrcomp(bgx,bgy,save=True,filename='', title=None, leg=False):
    '''
    for doing sfr comparison between match and object
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -3) &( bgy < 3) & (bgx<3)&(bgx > -3) )[0]
    nx = 6/0.06
    ny = 6.0/0.06
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ax=ax)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    ax.plot([-3,3], [-3,3], ls='--', linewidth=1, color='k', alpha=0.7)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylabel(r'log(SFR$_{\rm Match}$)',fontsize=20)
    plt.xlabel(r'log(SFR$_{\rm AGN}$)',fontsize=20)
    plt.xlim([-3,3])
    plt.ylim([-3,3])


    if title:
        plt.title(title,fontsize=30)
    if leg:
        plt.legend(fontsize=15,frameon = False, loc=3, bbox_to_anchor=(-0.04, .05))    
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/distmet/sfr_comp'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/distmet/sfr_comp'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/distmet/sfr_comp'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:

        plt.show()
'''
plot_sfrcomp(sfrm_gsw2.sfr_bpt, sfrm_gsw2.sfr_match_bpt, save=True,filename='_bpt')
plot_sfrcomp(sfrm_gsw2.sfr_plus, sfrm_gsw2.sfr_match_plus, save=True,filename='_plus')
plot_sfrcomp(sfrm_gsw2.sfr_neither, sfrm_gsw2.sfr_match_neither, save=True,filename='_neither')
plot_sfrcomp(sfrm_gsw2.sfr_sing, sfrm_gsw2.sfr_match_sing, save=True,filename='_sing')

'''

def plot_avcomp(bgx,bgy,save=True,filename='', title=None, leg=False):
    '''
    for doing sfr comparison between match and object
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -0) &( bgy < 3) & (bgx<3)&(bgx > -0) )[0]
    nx = 2/0.02
    ny = 2.0/0.02
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ax=ax)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    ax.plot([0,2], [0,2], ls='--', linewidth=1, color='k', alpha=0.7)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylabel(r'log(A$_{\rm V, GSW, Match}$)',fontsize=20)
    plt.xlabel(r'log(A$_{\rm V, GSW, AGN}$)',fontsize=20)
    plt.xlim([-0,2])
    plt.ylim([-0,2])


    if title:
        plt.title(title,fontsize=30)
    if leg:
        plt.legend(fontsize=15,frameon = False, loc=3, bbox_to_anchor=(-0.04, .05))    
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/distmet/av_comp'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/distmet/av_comp'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/distmet/av_comp'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:

        plt.show()
'''
plot_avcomp(sfrm_gsw2.av_gsw_bpt_agn, sfrm_gsw2.av_gsw_bpt_sf, save=True,filename='_bpt')
plot_avcomp(sfrm_gsw2.av_gsw_plus_agn, sfrm_gsw2.av_gsw_plus_sf, save=True,filename='_plus')
plot_avcomp(sfrm_gsw2.av_gsw_neither_agn, sfrm_gsw2.av_gsw_neither_sf, save=True,filename='_neither')
plot_avcomp(sfrm_gsw2.av_gsw_sing_agn, sfrm_gsw2.av_gsw_sing_sf, save=True,filename='_sing')

'''



def plot_masscomp(bgx,bgy,save=True,filename='', title=None, leg=False):
    '''
    for doing sfr comparison between match and object
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > 9) &( bgy < 12) & (bgx<12)&(bgx > 9) )[0]
    nx = 3/0.03
    ny = 3.0/0.03
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ax=ax)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    ax.plot([0,12], [0,12], ls='--', linewidth=1, color='k', alpha=0.7)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylabel(r'log(M$_{\rm *, Match}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\rm *, AGN}$)',fontsize=20)
    plt.xlim([9,12])
    plt.ylim([9,12])


    if title:
        plt.title(title,fontsize=30)
    if leg:
        plt.legend(fontsize=15,frameon = False, loc=3, bbox_to_anchor=(-0.04, .05))    
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/distmet/mass_comp'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/distmet/mass_comp'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/distmet/mass_comp'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:

        plt.show()
'''
plot_masscomp(sfrm_gsw2.mass_bpt, sfrm_gsw2.mass_match_bpt, save=True,filename='_bpt')
plot_masscomp(sfrm_gsw2.mass_plus, sfrm_gsw2.mass_match_plus, save=True,filename='_plus')
plot_masscomp(sfrm_gsw2.mass_neither, sfrm_gsw2.mass_match_neither, save=True,filename='_neither')
plot_masscomp(sfrm_gsw2.mass_sing, sfrm_gsw2.mass_match_sing, save=True,filename='_sing')

'''
def plot_fibmasscomp(bgx,bgy,save=True,filename='', title=None, leg=False):
    '''
    for doing sfr comparison between match and object
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > 8) &( bgy < 12) & (bgx<12)&(bgx > 8) )[0]
    nx = 4/0.04
    ny = 4.0/0.04
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ax=ax)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    ax.plot([0,12], [0,12], ls='--', linewidth=1, color='k', alpha=0.7)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylabel(r'log(M$_{\rm *, Fib., Match}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\rm *, Fib., AGN}$)',fontsize=20)
    plt.xlim([8,12])
    plt.ylim([8,12])


    if title:
        plt.title(title,fontsize=30)
    if leg:
        plt.legend(fontsize=15,frameon = False, loc=3, bbox_to_anchor=(-0.04, .05))    
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/distmet/fibmass_comp'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/distmet/fibmass_comp'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/distmet/fibmass_comp'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:

        plt.show()
'''
plot_fibmasscomp(sfrm_gsw2.fibmass_bpt, sfrm_gsw2.fibmass_match_bpt, save=True,filename='_bpt')
plot_fibmasscomp(sfrm_gsw2.fibmass_plus, sfrm_gsw2.fibmass_match_plus, save=True,filename='_plus')
plot_fibmasscomp(sfrm_gsw2.fibmass_neither, sfrm_gsw2.fibmass_match_neither, save=True,filename='_neither')
plot_fibmasscomp(sfrm_gsw2.fibmass_sing, sfrm_gsw2.fibmass_match_sing, save=True,filename='_sing')

'''
def plot_zcomp(bgx,bgy,save=True,filename='', title=None, leg=False):
    '''
    for doing sfr comparison between match and object
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > 0.00 )&( bgy < 0.3) & (bgx>0.00)&(bgx < 0.3) )[0]
    nx = 100
    ny = 100
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny, ax=ax)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    ax.plot([0,2], [0,2], ls='--', linewidth=1, color='k', alpha=0.7)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylabel(r'z$_{\rm Match}$',fontsize=20)
    plt.xlabel(r'z$_{\rm AGN}$',fontsize=20)
    plt.xlim([0, 0.3])
    plt.ylim([0, 0.3])


    if title:
        plt.title(title,fontsize=30)
    if leg:
        plt.legend(fontsize=15,frameon = False, loc=3, bbox_to_anchor=(-0.04, .05))    
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    if save:
        fig.savefig('plots/sfrmatch/png/distmet/z_comp'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/distmet/z_comp'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/distmet/z_comp'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:

        plt.show()
'''
plot_zcomp(sfrm_gsw2.z_bpt, sfrm_gsw2.z_match_bpt, save=True,filename='_bpt')
plot_zcomp(sfrm_gsw2.z_plus, sfrm_gsw2.z_match_plus, save=True,filename='_plus')
plot_zcomp(sfrm_gsw2.z_neither, sfrm_gsw2.z_match_neither, save=True,filename='_neither')
plot_zcomp(sfrm_gsw2.z_sing, sfrm_gsw2.z_match_sing, save=True,filename='_sing')

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
        fig.savefig('plots/sfrmatch/png/OIIILum_Mass_scat'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/sfrmatch/pdf/OIIILum_Mass_scat'+filename+'.pdf',dpi=250,bbox_inches='tight')
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
        fig.savefig('plots/xmm3/png/diagnostic/OIIILum_MassEdd_scat'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILum_MassEdd_scat'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILum_MassEdd_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiimassedd(EL_m2.mass, np.log10(EL_m2.oiiilum)-EL_m2.mass,
        save=True,alph=0.1)

'''
 

def plotbpt_sii(bgx,bgy,ccode= [], save=False,filename='',labels=True, title=None, minx=-1.2, maxx=0.75, miny=-1.2, maxy=2, nx=300, ny=240, ccodename='', ccodelim=[], nobj=True):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > miny) &( bgy < maxy) & (bgx<maxx)&(bgx > minx) )[0]
    if len(ccode) !=0:    
        plot2dhist(bgx[valbg], bgy[valbg], nx,ny, ccode=ccode[valbg], ccodename=ccodename, ccodelim = ccodelim)
    else:
        plot2dhist(bgx[valbg], bgy[valbg], nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    if nobj:
        plt.text(-1.7, -1, r"N$_{\mathrm{obj}}$: "+str(len(valbg)), fontsize=20)
    #if labels:
    #    plt.text(0.6,0.75,'AGN', fontsize=15)
    #    plt.text(-1.15,-0.3,'SF',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([miny-0.1,maxy])
    plt.xlim([minx-0.1, maxx+0.1])
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    ax.set(adjustable='box', aspect='equal')

    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline2_agn),np.log10(yline2_agn),'k--')#,label='AGN Line')
    plt.plot(np.log10(xline2_linersy2),np.log10(yline2_linersy2),c='k',ls='-.')#,label='LINER, Seyfert 2')
    if labels:       
        plt.text(.1,-.5,'LINER',fontsize=15)
        plt.text(-1.0,1.0,'Seyfert',fontsize=15)
        plt.text(-1.0,-1,'SF',fontsize=15)

    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([SII]/H$\rm \alpha$)',fontsize=20)

    #plt.legend(fontsize=10,loc=4)
    if title:
        plt.title(title,fontsize=30)
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
    plt.text(-2,-1.1,'SF',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([OI]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([-1.2,1.2])
    plt.xlim([-2.2,0])
    #plt.legend(fontsize=15,frameon = False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:

        fig.savefig('plots/xmm3/png/diagnostic/OI_OIII_scat'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/xmm3/pdf/diagnostic/OI_OIII_scat'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OI_OIII_scat'+filename+'.eps',dpi=150,format='eps')
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
    ax.set(adjustable='box', aspect='equal')

    plt.tight_layout()
    if save:
        fig.savefig("plots/xmm3/png/xraylum/"+label +filename+ "_lxm_vs_sfrm.png",dpi=250,bbox_inches='tight')
        fig.savefig("plots/xmm3/pdf/xraylum/"+label + filename+"_lxm_vs_sfrm.pdf",dpi=250,bbox_inches='tight',format='pdf')
        fig.savefig("plots/xmm3/eps/xraylum/"+label + filename+"_lxm_vs_sfrm.eps",format='eps',dpi=250,bbox_inches='tight')
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
plot_lxsfr(fullxray_xmm_dr7,'full',save=False, nofilt=True, filename='nofiltall_shade', scat = 0.6)

'''
def plotlx_oiii(lx, oiii, save=False, filename='',scat=False):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.scatter(lx, oiii, marker='x', color='k')
    plt.xlabel(r'L$_{\mathrm{X}}$', fontsize=20)
    plt.ylabel(r'L$_{\mathrm{[OIII]}}$', fontsize=20)
    plt.ylim([37, 45])
    plt.xlim([38,46])
    plt.plot([38,46], [37,45], color='k')
    ax.set(adjustable='box', aspect='equal')
    plt.tight_layout()
def plotlxsfr_oiii(lx_sfr, oiii, save=False, filename='',scat=False):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.scatter(lx_sfr, oiii, marker='x', color='k')
    plt.xlabel(r'L$_{\mathrm{X}}$-L$_{\mathrm{X,SFR}}$', fontsize=20)
    plt.ylabel(r'L$_{\mathrm{[OIII]}}$', fontsize=20)
    plt.ylim([37, 45])
    plt.xlim([-2,8])
    plt.plot([38,46], [37,45], color='k')
    ax.set(adjustable='box', aspect='equal')
    plt.tight_layout()
'''
plotlx_oiii(fullxray_xmm_all_bptplus.lum_val_filtagn_xrayagn[xmm3all_xrind_bpt], np.log10(sfrm_gsw2.oiiilum_bpt[xmm3all_xr_to_sfrm_bpt]))
plotlxsfr_oiii(fullxray_xmm_all_bptplus.lum_val_filtagn_xrayagn[xmm3all_xrind_bpt]-fullxray_xmm_all_bptplus.lxsfr[xmm3all_xrind_bpt], np.log10(sfrm_gsw2.oiiilum_bpt[xmm3all_xr_to_sfrm_bpt]))
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
            plt.savefig("plots/xmm3/png/xraylum/"+label+filename+"_lx_vs_fibsfr.png",dpi=250,bbox_inches='tight')
            plt.savefig("plots/xmm3/pdf/xraylum/"+label+filename+"_lx_vs_fibsfr.pdf",dpi=250,bbox_inches='tight')
            plt.savefig("plots/xmm3/eps/xraylum/"+label +filename+"_lx_vs_fibsfr.eps",format='eps',dpi=250,bbox_inches='tight')
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
            plt.savefig("plots/xmm3/png/xraylum/"+label+filename+"_lxd_vs_fibsfrd.png",dpi=250,bbox_inches='tight')
            plt.savefig("plots/xmm3/pdf/xraylum/"+label+filename+"_lxd_vs_fibsfrd.pdf",dpi=250,bbox_inches='tight')
            plt.savefig("plots/xmm3/eps/xraylum/"+label +filename+"_lxd_vs_fibsfrd.eps",format='eps',dpi=250,bbox_inches='tight')
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
       plt.savefig("plots/xmm3/png/xraylum/"+filename+"_lxz.png",dpi=250,bbox_inches='tight')
       plt.savefig("plots/xmm3/pdf/xraylum/"+filename+"_lxz.pdf",dpi=250,bbox_inches='tight')
       plt.savefig("plots/xmm3/eps/xraylum/"+filename+"_lxz.eps",format='eps',dpi=250,bbox_inches='tight')
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
        plt.savefig("plots/xmm3/png/xraylum/hard_vs_full.png",dpi=250,bbox_inches='tight')
        plt.savefig("plots/xmm3/pdf/xraylum/hard_vs_full.pdf",dpi=250,bbox_inches='tight',format='pdf')
        plt.savefig("plots/xmm3/eps/xraylum/hard_vs_full.eps",dpi=250,bbox_inches='tight',format='eps')

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
    ax.set(adjustable='box', aspect='equal')
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
    plt.legend(frameon=False,fontsize=15,loc=2,bbox_to_anchor=(0, 0.95))
    plt.tight_layout()
    if save:
        plt.savefig("plots/xmm3/png/diagnostic/fwhm_balmer_forbid"+filename+".png",dpi=250,bbox_inches='tight')
        plt.savefig("plots/xmm3/pdf/diagnostic/fwhm_balmer_forbid"+filename+".pdf",dpi=250,bbox_inches='tight',format='pdf')
        plt.savefig("plots/xmm3/eps/diagnostic/fwhm_balmer_forbid"+filename+".eps",dpi=250,bbox_inches='tight',format='eps')
        
'''
plotfwhmbalmerforbid(xmm3eldiagmed_xrfilt, EL_m2.balmerfwhm, EL_m2.forbiddenfwhm,
                     agn_3xmmm_xrfilt,
                     nonagn_3xmmm_xrfilt, save=True)
plotfwhmbalmerforbid(EL_qsos, EL_m2.balmerfwhm, EL_m2.forbiddenfwhm,
                     agn_qsos,
                     nonagn_qsos, save=False)
plotfwhmbalmerforbid(xmm3eldiagmed_xrfilt, EL_m2.balmerfwhm, EL_m2.forbiddenfwhm,
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



lsfrrelat = {'soft': [r'SFR/M$_{*} = 1.39\cdot 10^{-40}$ L$_{\rm x}/$M$_{*}$', r'SFR = $1.39\cdot 10^{-40}$ L$_{\rm x}$',logsfrsoft],
             'hard': [r'SFR/M$_{*} = 1.26\cdot 10^{-40} $L$_{\rm x}$/M$_{*}$', r'SFR = $1.26\cdot 10^{-40}$ L$_{\rm x}$',logsfrhard],
             'full': [r'SFR/M$_{*} = 0.66\cdot 10^{-40}$ L$_{\rm x}$/M$_{*}$',r'SFR = $0.66\cdot 10^{-40}$ L$_{\rm x}$', logsfrfull]  }

def plot_lxmsfrm(xraysfr, label, save=False, filtagn=[], filtnonagn=[],filename='',weakem=False,scat=False, nofilt=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
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
    ax.set(adjustable='box', aspect='equal')

    plt.tight_layout()
    if save:
        fig.savefig("plots/xmm3/png/xraylum/"+label +filename+ "_lxm_vs_sfrm.png",dpi=250,bbox_inches='tight')
        fig.savefig("plots/xmm3/pdf/xraylum/"+label + filename+"_lxm_vs_sfrm.pdf",dpi=250,bbox_inches='tight',format='pdf')
        fig.savefig("plots/xmm3/eps/xraylum/"+label + filename+"_lxm_vs_sfrm.eps",format='eps',dpi=250,bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plotoiiimasscomb(xvals, yvals, nonagnfilt, agnfilt, bgx1, bgy1, bgx2, bgy2, save=True, filename='',
                 title=None,leg=True):
    '''
    for doing oiii lum versus mass sdss galaxies scatter plotted
    '''
   
    fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    ax1 = fig.add_subplot(121)
    valbg = np.where((np.isfinite(bgx1) & (np.isfinite(bgy1))) &
            (bgy1 > 37) &( bgy1 < 43) & (bgx1<12)&(bgx1 > 8) )[0]
    nx = 4/0.01
    ny = 6/0.01
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
    plt.ylabel(r'log(L$_\mathrm{[OIII]}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([37,43])
    plt.xlim([8,12])
    ax2 = fig.add_subplot(122, sharey = ax1, sharex=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    #plt.ylabel(r'log(L$_\mathrm{[OIII]/SFR}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    valbg = np.where((np.isfinite(bgx2) & (np.isfinite(bgy2))) &
            (bgy2 > 37) &( bgy2 < 43) & (bgx2<12)&(bgx2 > 8) )[0]
    nx = 4/0.01
    ny = 6/0.01
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
        fig.savefig('plots/xmm3/png/diagnostic/OIIILum_Mass_scat'+filename+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILum_Mass_scat'+filename+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILum_Mass_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    return fig
'''
fig = plotoiiimasscomb(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m2.mass[nonagn_gsw], np.log10(EL_m2.oiiilum[nonagn_gsw]),EL_m2.mass[agn_gsw], np.log10(EL_m2.oiiilum[agn_gsw]),
        save=False,leg=False)
fig = plotoiiimasscomb(xmm3eldiagmed_xrfilt_all.mass,np.log10(xmm3eldiagmed_xrfilt_all.oiiilum),
          nonagn_3xmmm_all_xrfilt,
          agn_3xmmm_all_xrfilt,
          EL_m2.mass[nonagn_gsw], np.log10(EL_m2.oiiilum[nonagn_gsw]),EL_m2.mass[agn_gsw], np.log10(EL_m2.oiiilum[agn_gsw]),
        save=False,leg=False)

'''
def plot_lxsfr(xraysfr, label, save=False, filtagn=[], filtnonagn=[],filename='',weakem=False,scat=False, nofilt=False, fibssfr=[]):
    '''
    Plots star-formation rate versus X-ray luminosity and
    color-codes a region in bottom right where X-ray AGN are.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    if nofilt:
        plt.scatter(xraysfr.lum_val_filt, xraysfr.sfr_val_filt, marker='x', color='k')
    else:
        plt.scatter(xraysfr.lum_val_filtnoagn,xraysfr.sfr_val_filtnoagn,
                    marker='^', color='b',label='BPT-HII',zorder=3)
        plt.scatter(xraysfr.lum_val_filtagn,xraysfr.sfr_val_filtagn,
                    marker='o',facecolors='none',color='k',label='BPT-AGN',zorder=2)
    if weakem:
        plt.scatter(weakem.lum_val_filt,weakem.sfr_val_filt,marker='x',color='red',label=r'Weak Emission',zorder=1,alpha=0.8)

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
            plt.savefig("plots/xmm3/png/xraylum/"+label+filename+"_lx_vs_fibsfr.png",dpi=250,bbox_inches='tight')
            plt.savefig("plots/xmm3/pdf/xraylum/"+label+filename+"_lx_vs_fibsfr.pdf",dpi=250,bbox_inches='tight')
            plt.savefig("plots/xmm3/eps/xraylum/"+label +filename+"_lx_vs_fibsfr.eps",format='eps',dpi=250,bbox_inches='tight')
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
            plt.savefig("plots/xmm3/png/xraylum/"+label+filename+"_lxd_vs_fibsfrd.png",dpi=250,bbox_inches='tight')
            plt.savefig("plots/xmm3/pdf/xraylum/"+label+filename+"_lxd_vs_fibsfrd.pdf",dpi=250,bbox_inches='tight')
            plt.savefig("plots/xmm3/eps/xraylum/"+label +filename+"_lxd_vs_fibsfrd.eps",format='eps',dpi=250,bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
'''
plot_lxsfr(fullxray_xmm_all,'full',save=False, scat=0.6)

plot_lxsfr(fullxray_xmm,'full',save=False, weakem = fullxray_xmm_no, filename='weakem_shade', scat=0.6)

plot_lxsfr(fullxray_xmm,'full',save=False, nofilt=True, filename='nofiltall_shade', scat = 0.6, fibsfr= xmm3eldiagmed_xrfilt.fibsfr)

plot_lxsfr(fullxray_xmm_all,'full',save=False, nofilt=True, filename='nofiltall_shade', scat = 0.6)


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


plot_lxsfr(fullxray_xmm,'full',scat=0.6, save=False, nofilt=True)

plot_lxsfr(softxray_xmm,'soft',save=False,scat=0.6, nofilt=True)
plot_lxsfr(hardxray_xmm,'hard',save=False,scat=0.6, nofilt=True)


plot_lxsfr(fullxray_xmm,'full',save=True)
plot_lxsfr(softxray_xmm,'soft',save=True)
plot_lxsfr(hardxray_xmm,'hard',save=True)
'''
def plotlx_z(xraysfr, save=False,filename=''):
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.scatter(xraysfr.z_filt, xraysfr.lum_filt,color='black', marker='x')
    plt.xlabel('z')
    plt.xlim([0,0.3])
    plt.ylim([39,46])
    plt.ylabel(r'L$_{\mathrm{X}}$')
    plt.tight_layout()
    if save:
       plt.savefig("plots/xmm3/png/xraylum/"+filename+"_lxz.png",dpi=250,bbox_inches='tight')
       plt.savefig("plots/xmm3/pdf/xraylum/"+filename+"_lxz.pdf",dpi=250,bbox_inches='tight')
       plt.savefig("plots/xmm3/eps/xraylum/"+filename+"_lxz.eps",format='eps',dpi=250,bbox_inches='tight')
       plt.close()
    else:
       plt.show()    
'''
plotlx_z(fullxray_xmm_dr7, save=True, filename='full')

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
        fig1.savefig('plots/sfrmatch/png/distmet/avg_sample_dists_'+ str(samp_num) +'.png', dpi=250,bbox_inches='tight')
        fig1.savefig('plots/sfrmatch/pdf/distmet/avg_sample_dists_'+ str(samp_num) +'.pdf', dpi=250,bbox_inches='tight')
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
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
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
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
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
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
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
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
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
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
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
    plt.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    plt.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    plt.minorticks_on()
    plt.tick_params(direction='in',axis='both',which='both')
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


'''
