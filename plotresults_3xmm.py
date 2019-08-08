#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from matchgal_gsw2 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib
from matplotlib.colors import LogNorm
from sklearn import mixture
from ast_func import *
#from mpl_toolkits.basemap import Basemap
mydpi = 96
plt.rc('font',family='serif')
plt.rc('text',usetex=True)
#plt.rc('text',usetex=False)
def plot2dhist(x,y,nx,ny, nan=False, ax=None):
    if nan:
        xfin = np.where(np.isfinite(x))[0]
        yfin = np.where(np.isfinite(y))[0]
        x= np.copy(x[xfin])
        y=np.copy(y[yfin])
    hist, xedges, yedges = np.histogram2d(x,y,bins = (nx,ny))
    extent= [np.min(xedges),np.max(xedges),np.min(yedges),np.max(yedges)]
    print(extent)
    if ax:
        ax.imshow((hist.transpose())**0.3, cmap='gray_r',extent=extent,origin='lower',
               aspect='auto',alpha=0.9) 
    else:
        plt.imshow((hist.transpose())**0.3, cmap='gray_r',extent=extent,origin='lower',
               aspect='auto',alpha=0.9) 
def plthist(bincenters, counts):
    plt.plot(bincenters, counts,color='k', drawstyle='steps-mid')

def plot_xmmtime(time,name,filename='',save=False):
    nbins = np.arange(3.0,5.4,0.2)
    fig = plt.figure()
    plt.hist(time,bins = nbins,histtype='step')
    plt.xlabel(name)
    plt.ylabel('Counts')
    plt.axvline(x=4.5,ls='--')
    plt.axvline(x=4.1,ls='--')
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/exptimes/'+filename+'_exptime.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/exptimes/'+filename+'_exptime.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/exptimes/'+filename+'_exptime.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()

#plot_xmmtime(np.log10(x3.medtimes),r'log(t$_{\mathrm{exp}}$)[s]', save=True)
#plot_xmmtime(np.log10(x3.alltimes),r'log(t$_{\mathrm{exp}}$)')
#plot_xmmtime(np.log10(x3.deeptimes),r'log(t$_{\mathrm{exp}}$)',nbins=55)

def reject_outliers(data, m=2):
    return np.array([abs(data - np.mean(data)) < m * np.std(data)])[0]

def pairgdiff(x,y,z,m,dist,ind,ids,ngals,mind):
    finit  = np.where(((np.isfinite(x[ind])) & (np.isfinite(y[ind])) &
                       (x[ind] < np.nanmedian(x[ind])*10) &
                       (y[ind] <np.nanmedian(y[ind])*10)) )[0]

    mat = np.vstack([x[ind][finit],y[ind][finit],z[ind][finit],m[ind][finit],dist[ind][finit]])
    cols = ['BPTx','BPTy','z',r'log(M$_{\odot}$)','all']
    titl= 'Distance for SDSS'+str(ids[ind])+r', $n_{gal} = $'+str(ngals[ind])+','+'mindist = '+str(round(mind[ind],2))
    pairs(mat,cols,title=titl)
'''
pairgdiff(gbptxd82,gbptyd82,gzd82,gmassd82,gdiffs82,nonagn[0],s82ids,nrx,mindx)
pairgdiff(gbptxd82,gbptyd82,gzd82,gmassd82,gdiffs82,nonagn[1],s82ids,nrx,mindx)
pairgdiff(gbptxd82,gbptyd82,gzd82,gmassd82,gdiffs82,nonagn[2],s82ids,nrx,mindx)
pairgdiff(gbptxd82,gbptyd82,gzd82,gmassd82,gdiffs82,nonagn[3],s82ids,nrx,mindx)
pairgdiff(gbptxd82,gbptyd82,gzd82,gmassd82,gdiffs82,nonagn[4],s82ids,nrx,mindx)
pairgdiff(gbptxd82,gbptyd82,gzd82,gmassd82,gdiffs82,nonagn[5],s82ids,nrx,mindx)
pairgdiff(gbptxd82,gbptyd82,gzd82,gmassd82,gdiffs82,nonagn[6],s82ids,nrx,mindx)
pairgdiff(gbptxd82,gbptyd82,gzd82,gmassd82,gdiffs82,nonagn[7],s82ids,nrx,mindx)
logbptsx = np.log10(bpt82x)
logbptsy = np.log10(bpt82y)
fins82 = np.where((np.isfinite(logbptsx)) & (np.isfinite(logbptsy)))

#pairs(np.vstack([logbptsx[fins82],logbptsy[fins82],z82[fins82],mass82[fins82]]),['BPTx','BPTy','z',r'log(M$_{\odot}$)'])
logbptgx = np.log10(bptgswx[stripe82])
logbptgy = np.log10(bptgswy[stripe82])

fing = np.where((np.isfinite(logbptgx)) & (np.isfinite(logbptgy)))
'''
#pairs(np.vstack([logbptgx[fing],logbptgy[fing],zgsw[stripe82][fing],massgsw[stripe82][fing]]),['BPTx','BPTy','z',r'log(M$_{\odot}$)'])
def pairs(data_mat,cols,title=''):
    df = pd.DataFrame(np.transpose(data_mat),columns = cols)
    axes = pd.plotting.scatter_matrix(df,alpha=0.2,hist_kwds={'bins':50})
    plt.suptitle(title)
    #plt.tight_layout()

def ratiogdiff(frac,agnfilt, nonagnfilt,filename='',save=False):
    bins = np.copy(s82gdiff.bins)
    fig = plt.figure()
    for i in agnfilt:
        plt.plot(bins,frac[i],'k--',linewidth=0.5)
    if len(agnfilt)!=0:
        plt.plot(bins,frac[agnfilt[0]],'k--',label='BPT-AGN',linewidth=0.5)
    for i in nonagnfilt:
        plt.plot(bins,frac[i],'b-.',linewidth=1)
    if len(nonagnfilt) !=0:
        plt.plot(bins,frac[nonagnfilt[0]],'b-.',label='BPT-HII',linewidth=1)
    plt.xlabel('Distance',fontsize=20)
    plt.ylabel(r'N$_{\mathrm{X}}$/N$_{\mathrm{GSW}}$',fontsize=20)
    plt.ylim([0,1.05])
    plt.xlim([0,5])
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.axvline(x=2.5,ls='-.',color='gray', alpha=0.5, zorder=0, label='d=2.5')
    #plt.axvline(x=3,ls='-.', label='d=3')
    
    plt.tight_layout()
    plt.legend(frameon=False,fontsize=15)
    if save:
        fig.savefig('plots/xmm3/png/distmet/ratios/ratio_nx_ngsw_'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/distmet/ratios/ratio_nx_ngsw_'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/distmet/ratios/ratio_nx_ngsw_'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #plt.title('Ratios fo')
'''
ratiogdiff(xmm3gdiff.xrgswfracs,[],nonagn_3xmmm_xrfilt,filename='bpthii')
ratiogdiff(xmm3gdiff.xrgswfracs,agn_3xmmm_xrfilt,[],filename='agn')
ratiogdiff(xmm3gdiff.xrgswfracs,agn_3xmmm_xrfilt,nonagn_3xmmm_xrfilt,filename='all')

ratiogdiff(xmm3gdiff.xrgswfracs,[],nonagn_3xmmm_xrfilt,filename='bpthii', save=True)
ratiogdiff(xmm3gdiff.xrgswfracs,agn_3xmmm_xrfilt,[],filename='agn', save=True)
ratiogdiff(xmm3gdiff.xrgswfracs,agn_3xmmm_xrfilt,nonagn_3xmmm_xrfilt,filename='all',save=True)
'''

def plotcumxrgswrat(fracxr, fracgsw, gals,filename='',save=False):
    bins = np.copy(xmm3gdiff.bins)
    fig = plt.figure()
    for i in gals:
        plt.plot(bins, fracxr[i],'k-.')
        plt.plot(bins, fracgsw[i],'b--')
    plt.plot(bins, fracxr[i],'k-.',label='3XMM Fraction')
    plt.plot(bins, fracgsw[i],'b--',label='GSW Fraction')
    plt.legend(frameon=False,fontsize=15)
    plt.ylim([0,1])
    plt.xlim([0,10])
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.xlabel('Distance',fontsize=20)
    plt.ylabel('Cumulative Ratio',fontsize=20)
    
    plt.tight_layout()
    
    if save:
        fig.savefig('plots/xmm3/png/distmet/ratios/cumulative_ratios_'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/distmet/ratios/cumulative_ratios_'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/distmet/ratios/cumulative_ratios_'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotcumxrgswrat(xmm3gdiff.xrfrac,xmm3gdiff.gswfrac,agn_3xmmm_xrfilt,filename='agn')
plotcumxrgswrat(xmm3gdiff.xrfrac,xmm3gdiff.gswfrac,range(len(xmm3gdiff.xrfrac)),filename='all')
plotcumxrgswrat(xmm3gdiff.xrfrac,xmm3gdiff.gswfrac,nonagn_3xmmm_xrfilt,filename='bpthii')

plotcumxrgswrat(xmm3gdiff.xrfrac,xmm3gdiff.gswfrac,agn_3xmmm_xrfilt,filename='agn',save=True)
plotcumxrgswrat(xmm3gdiff.xrfrac,xmm3gdiff.gswfrac,range(len(xmm3gdiff.xrfrac)),filename='all',save=True)
plotcumxrgswrat(xmm3gdiff.xrfrac,xmm3gdiff.gswfrac,nonagn_3xmmm_xrfilt,filename='bpthii', save=True)

'''
def plotgdiffhist(gdiffs, s82diffs, sdssid,ngal,nearestx=None):
    relevgsw = np.where((gdiffs <10)&(gdiffs >0))[0] #only want relevant cases in hist
    relevs82 = np.where((s82diffs<10)& (s82diffs>0))[0]
    binsgsw = np.copy(xmm3gdiff.bins)
    bins82 = np.copy(xmm3gdiff.bins)
    plt.hist(gdiffs[relevgsw],bins=binsgsw,histtype='step',label='GSW Galaxies')
    plt.hist(s82diffs[relevs82],bins=bins82,label='3XMM Galaxies')
    plt.ylabel('Counts')
    plt.xlabel('Distance')
    plt.title('Distances for SDSS'+str(sdssid),fontsize=20)
    plt.xlim([0,10])
    '''
    if nearestx:
        plt.axvline(x=nearestx,ls=':',label='Nearest X-Ray Detection: Distance = '+str(round(nearestx,2)))
    '''
    plt.legend(frameon=False,fontsize=15)

    plt.tight_layout()
    plt.show()
'''
i=2
plotgdiffhist(xmm3gdiff.dists_filt[nonagn_3xmmm_xrfilt[i]],xmm3gdiff.alls82dists[nonagn_3xmmm_xrfilt[i]],
              xmm3ids[nonagn_3xmmm_xrfilt[i]],xmm3gdiff.nrx[nonagn_3xmmm_xrfilt[i]],
              nearestx=xmm3gdiff.mindx[nonagn_3xmmm_xrfilt[i]])
i=40
plotgdiffhist(xmm3gdiff.dists_filt[agn_3xmmm_xrfilt[i]],
              xmm3gdiff.alls82dists[agn_3xmmm_xrfilt[i]],
              xmm3ids[agn_3xmmm_xrfilt[i]],
              xmm3gdiff.nrx[agn_3xmmm_xrfilt[i]],
              nearestx=xmm3gdiff.mindx[agn_3xmmm_xrfilt[i]])

'''
def getallgdiffhistsbpthii(gdiff,ids, nonagn):
    for i in range(len(nonagn)):
        plotgdiffhist(gdiff.dists_filt[nonagn[i]], gdiff.alls82dists[nonagn[i]], ids[nonagn[i]], gdiff.nrx[nonagn[i]], nearestx=gdiff.mindx[nonagn[i]])
        plt.savefig('plots/xmm3/png/distmet/bpt_hii_dists/bpthII'+str(i)+'.png')
        plt.savefig('plots/xmm3/eps/distmet/bpt_hii_dists/bpthII'+str(i)+'.eps')
        plt.savefig('plots/xmm3/pdf/distmet/bpt_hii_dists/bpthII'+str(i)+'.pdf')

        plt.close()
#getallgdiffhistsbpthii(xmm3gdiff, xmm3ids, nonagn_3xmmm_xrfilt)
def getallgdiffhistsbptagn(gdiff, ids, agn):
    for i in range(len(agn)):
        plotgdiffhist(gdiff.dists_filt[agn[i]], gdiff.alls82dists[agn[i]], ids[agn[i]], gdiff.nrx[agn[i]], nearestx=gdiff.mindx[agn[i]])
        plt.savefig('plots/xmm3/png/distmet/bpt_agn_dists/agn'+str(i)+'.png')
        plt.savefig('plots/xmm3/eps/distmet/bpt_agn_dists/agn'+str(i)+'.eps')
        plt.savefig('plots/xmm3/pdf/distmet/bpt_agn_dists/agn'+str(i)+'.pdf')
        plt.close()
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
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.legend()
    plt.tight_layout()

def plotbpt(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=False,filename='',labels=True,
            leg=True,title=None,alph=0.1,ccode=[],ccodegsw=[],cont=None,ccodename='X-ray AGN Fraction',ccodecolor='Blues',levs=[0,0.2], fibssfr=False):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues',marker='.',label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > -1.2) &( bgy < 1.2) & (bgx<1)&(bgx > -2) )[0]
        nx = 3/0.01
        ny = 2.4/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        if fibssfr:
            finnonagn = np.where(np.isfinite(ccode[nonagnfilt]) &(ccode[nonagnfilt]<-5))[0]
            finagn = np.where(np.isfinite(ccode[agnfilt])&(ccode[agnfilt]<-5))[0]
            fin = np.where(np.isfinite(ccode)&(ccode<-5))[0]
            mn, mx = ccode[fin].min(), ccode[fin].max()       
            sc =plt.scatter(xvals[nonagnfilt][finnonagn], yvals[nonagnfilt][finnonagn], c=ccode[nonagnfilt][finnonagn], s=20, cmap=ccodecolor,marker='^',label='X-Ray AGN (BPT-HII)')
            plt.clim(mn, mx)

            sc =plt.scatter(xvals[agnfilt][finagn], yvals[agnfilt][finagn], c=ccode[agnfilt][finagn], s=20, cmap=ccodecolor,marker='o',label='X-Ray AGN (BPT-AGN)')
            plt.clim(mn, mx)
        else:
            #finnonagn = np.where(np.isfinite(ccode[nonagnfilt]) &(ccode[nonagnfilt]<-5))[0]
            #finagn = np.where(np.isfinite(ccode[agnfilt])&(ccode[agnfilt]<-5))[0]
            #fin = np.where(np.isfinite(ccode)&(ccode<-5))[0]
            mn, mx = ccode.min(), ccode.max()       
            sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap=ccodecolor,marker='^',label='X-Ray AGN (BPT-HII)')
            plt.clim(mn, mx)

            sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap=ccodecolor,marker='o',label='X-Ray AGN (BPT-AGN)')
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
    if cont:
        CS = plt.contour(cont.meshx, cont.meshy, cont.grid,cmap=ccodecolor,levels=levs,vmin=contaminations_xmm3.min(), vmax= contaminations_xmm3.max())
        norm= matplotlib.colors.Normalize(vmin=contaminations_xmm3_3.min(), vmax=contaminations_xmm3_3.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap = CS.cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm)

        cbar.set_label(ccodename,fontsize=20)
        '''
        cbar.ax.tick_params(labelsize=20)
        plt.text(-1,-0.05,'0.02',fontsize=15)
        plt.text(-0.76,-0.52,'0.04',fontsize=15)
        plt.text(-0.28,-0.46,'0.06',fontsize=15)
        plt.text(-0.2,-0.17,'0.08',fontsize=15)
        plt.text(0.30,-0.187,'0.10',fontsize=15)
        '''
    if labels:
        plt.text(0.6,0.75,'AGN', fontsize=15)
        plt.text(-1.15,-0.3,'HII',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([-1.3,1.2])
    plt.xlim([-2.1,1])
    if leg:
        plt.legend(fontsize=15,frameon=False,loc=3,bbox_to_anchor=(-0.02, -0.02))
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/NII_OIII_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/NII_OIII_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/NII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()

def plotbptnormal(bgx,bgy,save=False,filename='',labels=True, title=None):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > -1.2) &( bgy < 1.2) & (bgx<1)&(bgx > -2) )[0]
    nx = 3/0.01
    ny = 2.4/0.01
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    plt.text(-1.7, -1, r"N$_{\mathrm{obj}}$: "+str(len(bgx)), fontsize=20)
    if labels:
        plt.text(0.6,0.75,'AGN', fontsize=15)
        plt.text(-1.15,-0.3,'HII',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([-1.3,1.2])
    plt.xlim([-2.1,1])
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/sfrmatch/png/NII_OIII_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/sfrmatch/pdf/NII_OIII_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/sfrmatch/eps/NII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt]),filename='sn2', save=False)


plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns]),filename='agns_plus', save=False)
plotbptnormal(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.sfs]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.sfs]),filename='sfs_plus', save=False)

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
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.tick_params(axis='both', which='minor', labelsize=15)
    ax1.tick_params(direction='in',axis='both',which='both')
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
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='minor', labelsize=15)
    ax2.tick_params(direction='in',axis='both',which='both')
    ax2.set_xlim([-2.1,1])
    ax2.set_xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    ax2.set_ylabel(r'Fraction',fontsize=20)
    ax2.set_xticks(np.arange(0.1, 1, 0.1), minor = True)
    ax2.axvline(x=nii_bound,ls='-',linewidth=1, color='k', alpha=0.8 )
    #ax2.axvline(x=nii_bound+0.05,ls='-',linewidth=1, color='k', alpha=0.8 )
    
    nii_agn = np.where(bgxhist_finite >nii_bound)[0]
    nii_sf = np.where(bgxhist_finite <nii_bound)[0]
    #axs[1].set_xlim([-1.3,1.2])
    plt.text(-1.8, 6.8/8 * np.max(cnts)/len(bgxhist_finite), r"N$_{\mathrm{obj}}$: " + str(len(bgxhist_finite)) +'('+str(round(100*len(bgxhist_finite)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)
    plt.text(-1.8, 4.8/8 * np.max(cnts)/len(bgxhist_finite), r"N$_{\mathrm{AGN}}$: "+str(len(nii_agn))+'('+str(round(100*len(nii_agn)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)
    plt.text(-1.8, 2.8/8 * np.max(cnts)/len(bgxhist_finite), r"N$_{\mathrm{SF}}$: "+str(len(nii_sf))+'('+str(round(100*len(nii_sf)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)
    if len(unclass) != 0:
        plt.text(-1.8, 0.8/8 * np.max(cnts)/len(bgxhist_finite), r"N$_{\mathrm{unclass}}$: "+str(len(unclass)) +'('+str(round(100*len(unclass)/(len(bgxhist_finite)+len(bgx)+len(unclass))))+'\%)', fontsize=15)

    #ax2.set_aspect(10)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if save:
        fig.savefig('./plots/sfrmatch/png/NII_OIII_scat'+filename+'.png',dpi=250)
        fig.savefig('./plots/sfrmatch/pdf/NII_OIII_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('./plots/sfrmatch/eps/NII_OIII_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    
'''

plotbptplus(np.log10(EL_m2.xvals1_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]) , np.log10(EL_m2.yvals_bpt[EL_m2.bpt_sn_filt][sfrm_gsw2.agns_plus]),
            np.log10(EL_m2.xvals1_bpt[EL_m2.halp_nii_filt][sfrm_gsw2.sfs_plus]), [], nonagn=[], agn= [],
            filename='bptplus_sfrmatch', save=False, labels=False)

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

plotbpt(EL_qsos.niiha,EL_qsos.oiiihb,
         nonagn_qsos,
         agn_qsos,
         EL_m1.niiha,EL_m1.oiiihb,save=True, leg=False, filename='qsos')
plotbpt(EL_3xmm.niiha,EL_3xmm.oiiihb,
         [],
         [],
         EL_m1.niiha,EL_m1.oiiihb,save=False, ccodegsw= EL_m1.ssfr, labels=False,leg=False)

oiiigswlumfilt = np.where((np.log10(EL_m1.oiiilum)<43) & (np.log10(EL_m1.oiiilum)>37))[0]
plotbpt(EL_3xmm.niiha,EL_3xmm.oiiihb,
         [],
         [],
         EL_m1.niiha[oiiigswlumfilt],EL_m1.oiiihb[oiiigswlumfilt],save=False,leg=False, ccodegsw= np.log10(EL_m1.oiiilum[oiiigswlumfilt]), labels=False)
plotbpt(EL_3xmm.niiha,EL_3xmm.oiiihb,
         [],
         [],
         EL_m1.niiha[oiiigswlumfilt],EL_m1.oiiihb[oiiigswlumfilt],save=False,leg=False, ccodegsw= np.log10(EL_m1.oiiilum[oiiigswlumfilt]/EL_m1.vdisp[oiiigswlumfilt]**4), labels=False)

#xray SF
plotbpt(xmm3eldiagmed_xrsffilt.niiha,xmm3eldiagmed_xrsffilt.oiiihb,
         nonagn_3xmmm_xrsffilt,
         agn_3xmmm_xrsffilt,
         EL_m1.niiha,EL_m1.oiiihb,save=False, filename='xray-sf')



#xrayAGN
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=False, filename='xrfilt')
plotbpt(xmm3eldiagmed_xrfilt_all.niiha,xmm3eldiagmed_xrfilt_all.oiiihb,
         nonagn_3xmmm_all_xrfilt,
         agn_3xmmm_all_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=False, filename='xrfilt')
#low sSFR
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt[lowssfr],
         [],
         EL_m1.niiha,EL_m1.oiiihb,save=False, filename='lowssfr')

#contaminations

plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=True, filename='xrfrac3', leg=False,ccode=contaminations_xmm3_3)

plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=True, filename='xrfrac2', leg=False,ccode=contaminations_xmm3_2)

#THESE HAVE TO BE COLOR CODED BY THE MIN MAX OF d=3
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=False, filename='xrfrac25', leg=False,ccode=contaminations_xmm3_25)
#ssfr
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=True, ccodename='log(sSFR)',filename='ssfrcode',
         ccodecolor='Reds', leg=False,ccode=xmm3eldiagmed_xrfilt.ssfr)
#fibssfr
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=True, ccodename=r'log(sSFR$_{\mathrm{fib}}$)',filename='ssfrfibcode',
         ccodecolor='Reds', leg=False,ccode=xmm3eldiagmed_xrfilt.fibssfr, fibssfr=True)
#fibsfr
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb, save=True, ccodename=r'log(SFR$_{\mathrm{fib}}$)',filename='sfrfibcode',
         ccodecolor='Reds', leg=False,ccode=xmm3eldiagmed_xrfilt.fibsfr_uncorr, fibssfr=False)

#Lx
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb, save=True, ccodename=r'log(L$_{\mathrm{X}}$)',filename='lxcode',
         ccodecolor='Reds', leg=False,ccode=fullxray_xmm.lum_val_filt_xrayagn, fibssfr=False)
#lx-mass
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb, save=True, ccodename=r'log(L$_{\mathrm{X}}$/M$_{\mathrm{*}}$)',filename='lxmasscode',
         ccodecolor='Reds', leg=False,ccode=fullxray_xmm.lum_mass_val_filt_xrayagn, fibssfr=False)

#sfr
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=True, ccodename='log(SFR)',ccodecolor='Reds', 
         filename='sfrcode', leg=False,ccode=xmm3eldiagmed_xrfilt.sfr)

plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=True, ccodename='log([OIII]/sigma4)',ccodecolor='Greens', 
         filename='oiiisig', leg=False,ccode=np.log10(xmm3eldiagmed_xrfilt.oiiilum/(xmm3eldiagmed_xrfilt.vdisp*1e5)**4))
#contour
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb, [], [],
         EL_m1.niiha,EL_m1.oiiihb,save=False, cont=cont, levs=[0.02,0.04, 0.06,0.08],filename='xrfrac_cont')
        

plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb, [], [],
         EL_m1.niiha,EL_m1.
         oiiihb,save=False, levs=[0.02,0.03,0.04, 0.06,0.08,0.10],filename='xrfrac_contmore')
        
plt.imshow(cont.grid, vmin=contaminations_xmm3.min(), vmax=contaminations_xmm3.max(), extent=[cont.meshx.min(), cont.meshx.max(), cont.meshy.min(), cont.meshy.max()], origin='lower',cmap='Blues')
plt.scatter(xmm3eldiagmed_xrfilt.niiha, xmm3eldiagmed_xrfilt.oiiihb,marker='o',color='orange')

plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb, [], [],
         EL_m1.niiha,EL_m1.oiiihb,save=False, levs=[0.02,0.03,0.04, 0.06,0.08,0.10],filename='xrfrac_contmore')
plt.scatter(cont.meshx, cont.meshy, c=cont.grid, cmap='Blues', s= 25)        
plt.colorbar().set_label('X-ray Fraction')
plt.tight_layout()


#mass
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=True,filename='mass', ccode=xmm3eldiagmed_xrfilt.mass, ccodecolor='Reds',ccodename='Mass')
#z
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,filename='z',save=True, ccode=xmm3eldiagmed_xrfilt.z, ccodecolor='Reds', ccodename='z')
#oiiilum
plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=True, filename='oiilum',
         ccode=np.log10(xmm3eldiagmed_xrfilt.oiiilum),ccodecolor='Greens', ccodename=r'L$_{[OIII]}$')

plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=False, filename='x-raylum',
         ccode=np.log10(fullxray_xmm.lum_val_filt_xrayagn),ccodecolor='Reds', ccodename=r'L$_{x}$')

plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=False, filename='ccodehalplum',
         ccode=np.log10(xmm3eldiagmed_xrfilt.halplum),ccodecolor='Reds', ccodename=r'L$_{H\alpha}$',leg=False)


plotbpt(xmm3eldiagmed_xrfilt.niiha,xmm3eldiagmed_xrfilt.oiiihb,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.niiha,EL_m1.oiiihb,save=True, filename='ccodeoiiihalplum',
         ccode=np.log10(xmm3eldiagmed_xrfilt.oiiilum)-np.log10(xmm3eldiagmed_xrfilt.halplum),ccodecolor='Reds', ccodename=r'L$_{[OIII]}$/L$_{H\alpha}$',leg=False)
'''

def mass_z(xvals,yvals,nonagnfilt, agnfilt, bgx,bgy,save=True,filename='',title=None,alph=0.1,ccode=[],ccodegsw=[], leg=False, weakel=False):
    '''
    for doing mass against z with sdss galaxies scatter plotted
    '''



    if save:
        fig = plt.figure()
    counts1,xbins1,ybins1= np.histogram2d(bgx, bgy, bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > 8) &( bgy < 12.5) & (bgx<0.3)&(bgx > 0) )[0]
        nx = 0.3/0.001
        ny = 4/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.ylim([8, 12.5])
    plt.xlim([0,0.3])
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt)) !=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    
    plt.ylabel(r'log(M$_{\rm *}$)',fontsize=20)
    plt.xlabel(r'z',fontsize=20)
    if weakel:
        plt.scatter(weakel.z_filt_xrayagn, weakel.mass_filt_xrayagn, color='red', marker='+', s=75, label='Weak Emission Line X-ray AGN')
    if leg:
        plt.legend(fontsize=15,loc=4,bbox_to_anchor=(1, 0.05),frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/mass_z'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/mass_z'+filename+'.pdf',dpi=250, format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/mass_z'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''
#general


mass_z(xmm3eldiagmed_xrfilt.z,xmm3eldiagmed_xrfilt.mass,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.z,EL_m1.mass,save=True)


mass_z(xmm3eldiagmed_xrfilt.z,xmm3eldiagmed_xrfilt.mass,
         [],
         [], 
         EL_m1.z,EL_m1.mass,save=True, filename='weakel', leg=True, weakel=fullxray_xmm_no)
mass_z(xmm3eldiagmed_xrfilt.z,xmm3eldiagmed_xrfilt.mass,
         nonagn_3xmmm_xrfilt,
         [], 
         EL_m1.z,EL_m1.mass,save=True, filename='weakelbpthii', leg=True, weakel=fullxray_xmm_no)




'''
def massfrac_z(xvals,yvals,nonagnfilt, agnfilt, bgx,bgy,save=True,filename='',title=None,alph=0.1,ccode=[],ccodegsw=[]):
    '''
    for doing mass against z with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
    counts1,xbins1,ybins1= np.histogram2d(bgx, bgy, bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc =plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > 0) &( bgy < 0.8) & (bgx<0.3)&(bgx > 0) )[0]
        nx = 0.3/0.001
        ny = 0.8/0.001
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    if len(ccode) != 0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt)) !=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)

    plt.ylabel(r'Mass Fraction',fontsize=20)
    plt.xlabel(r'z',fontsize=20)
    plt.ylim([0, 0.8])
    plt.xlim([0,0.3])
    plt.legend(fontsize=15, frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/massfrac_z'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/massfrac_z'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/massfrac_z'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi))
'''

massfrac_z(xmm3eldiagmed_xrfilt.z,xmm3eldiagmed_xrfilt.massfrac,
         nonagn_3xmmm_xrfilt,
         agn_3xmmm_xrfilt,
         EL_m1.z,EL_m1.massfrac,save=True)
'''
def plotssfrm(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=True,filename='',
              title=None, alph=0.1, ccode=[], ccodegsw=[],leg=True, weakel=False):
    '''
    for doing ssfrmass diagram with sdss galaxies scatter plotted
    '''

    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > -14) &( bgy < -8) & (bgx<12.5)&(bgx > 7.5) )[0]
        nx = 5/0.01
        ny = 6.0/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')


    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()
        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT-HII)',zorder=10)
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT-AGN)',zorder=9)
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20,zorder=11)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20,zorder=10)
    plt.ylabel(r'log(sSFR)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}})$',fontsize=20)
    plt.xlim([7.5,12.5])
    plt.ylim([-14,-8])

    if title:
        plt.title(title,fontsize=30)
    if weakel:
        plt.scatter(weakel.mass_filt_xrayagn,weakel.sfr_mass_val_filt_xrayagn,color='red',marker='+', s=75, label='Weak Emission Line X-ray AGN' )
    if leg:
        plt.legend(fontsize=15,frameon = False, loc=3, bbox_to_anchor=(-0.04, .05))    
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/ssfr_mass'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/ssfr_mass'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/ssfr_mass'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:

        plt.show()
'''
plotssfrm(xmm3eldiagmed_xrfilt.mass,xmm3eldiagmed_xrfilt.ssfr,
          nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt,
          EL_m1.mass, EL_m1.ssfr,   save=True, alph=0.1)


plotssfrm(xmm3eldiagmed_xrfilt.mass,xmm3eldiagmed_xrfilt.ssfr,
          nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt,
          EL_m1.mass, EL_m1.ssfr,   save=False,alph=0.1, weakha_xxray = EL_3xmm.weakmass,
          weakha_yxray = EL_3xmm.weakssfr )
plotssfrm(xmm3eldiagmed_xrfilt.mass,xmm3eldiagmed_xrfilt.ssfr,
          [], [],
          EL_m1.mass, EL_m1.ssfr,   save=True, filename='weakel',
          weakel = fullxray_xmm_no )
plotssfrm(xmm3eldiagmed_xrfilt.mass,xmm3eldiagmed_xrfilt.ssfr,
          nonagn_3xmmm_xrfilt, [],
          EL_m1.mass, EL_m1.ssfr,   save=True,
          weakel = fullxray_xmm_no, filename='weakelbpthii' )

'''

def plotssfrtbt(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=True,filename='',
              title=None, alph=0.1, ccode=[], leg=True, weakel=False):
    '''
    for doing ssfr-tbt diagram with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > 8.5) &( bgy < 13) & (bgx<0.25)&(bgx > -2) )[0]
    nx = 2.25/0.025
    ny = 4.5/0.05
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)


    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()
        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT-HII)',zorder=10)
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT-AGN)',zorder=9)
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20,zorder=11)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20,zorder=10)
    plt.ylabel(r'-log(sSFR)',fontsize=20)
    plt.xlabel(r'log([NeIII]/[OII])',fontsize=20)
    plt.xlim([-2,0.25])
    plt.ylim([8.5,13])

    if title:
        plt.title(title,fontsize=30)
    if weakel:
        plt.scatter(weakel.mass_filt_xrayagn,weakel.sfr_mass_val_filt_xrayagn,color='red',marker='+', s=75, label='Weak Emission Line X-ray AGN' )
    if leg:
        plt.legend(fontsize=15,frameon = False, loc=3, bbox_to_anchor=(-0.04, .05))    
    plt.tight_layout()
    if save:
        fig.savefig('plots/tbt/ssfr_tbt'+filename+'.png',dpi=250)
        fig.savefig('plots/tbt/ssfr_tbt'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/tbt/ssfr_tbt'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:

        plt.show()
        
'''

valtbt= np.where( (EL_m1.neiiioii[EL_m1.tbt_filt] > -2) & (EL_m1.neiiioii[EL_m1.tbt_filt] < 1))[0]

plotssfrtbt(xmm3eldiagmed_xrfilt.neiiioii[xmm3eldiagmed_xrfilt.tbt_filt],-xmm3eldiagmed_xrfilt.ssfr[xmm3eldiagmed_xrfilt.tbt_filt],
            nonagn_3xmmm_xrfilttbt, agn_3xmmm_xrfilttbt, EL_m1.neiiioii[EL_m1.tbt_filt], -EL_m1.ssfr[EL_m1.tbt_filt], 
            leg=False,save=False)
plotssfrtbt(xmm3eldiagmed_xrfilt.neiiioii[xmm3eldiagmed_xrfilt.tbt_filt],-xmm3eldiagmed_xrfilt.ssfr[xmm3eldiagmed_xrfilt.tbt_filt],
            nonagn_3xmmm_xrfilttbt, [], EL_m1.neiiioii[EL_m1.tbt_filt][nonagn_gswtbt], -EL_m1.ssfr[EL_m1.tbt_filt][nonagn_gswtbt], 
            leg=False,save=False)

plotssfrtbt(xmm3eldiagmed_xrfilt.neiiioii[xmm3eldiagmed_xrfilt.tbt_filt],-xmm3eldiagmed_xrfilt.ssfr[xmm3eldiagmed_xrfilt.tbt_filt],
            [], agn_3xmmm_xrfilttbt, EL_m1.neiiioii[EL_m1.tbt_filt][agn_gswtbt], -EL_m1.ssfr[EL_m1.tbt_filt][agn_gswtbt], 
            leg=False,save=False)

plot2dhist(EL_m1.neiiioii[EL_m1.tbt_filt][valtbt], -EL_m1.ssfr[EL_m1.tbt_filt][valtbt], 100, 100, nan=True)
[-1.9565441608428955, 0.095269307494163513, 7.931, 13.209]

plt.xlim([-2,0.25])
Out[21]: (-2, 0.25)

plt.scatter(xmm3eldiagmed_xrfilt.neiiioii[xmm3eldiagmed_xrfilt.tbt_filt][nonagn_3xmmm_xrfilttbt],-xmm3eldiagmed_xrfilt.ssfr[xmm3eldiagmed_xrfilt.tbt_filt][nonagn_3xmmm_xrfilttbt],marker='^', color='blue')
Out[22]: <matplotlib.collections.PathCollection at 0x7f5215e76d30>

plt.scatter(xmm3eldiagmed_xrfilt.neiiioii[xmm3eldiagmed_xrfilt.tbt_filt][agn_3xmmm_xrfilttbt],-xmm3eldiagmed_xrfilt.ssfr[xmm3eldiagmed_xrfilt.tbt_filt][agn_3xmmm_xrfilttbt],marker='o', facecolor='none', color='black')
Out[23]: <matplotlib.collections.PathCollection at 0x7f521b75ebe0>

plt.xlabel('log([NeIII]/[OII])')
Out[24]: <matplotlib.text.Text at 0x7f5263f16c18>

plt.ylabel('-log(sSFR)')
Out[25]: <matplotlib.text.Text at 0x7f52247cccc0>
'''

def plotfibssfrm(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=True,filename='',
              title=None, alph=0.1, ccode=[], ccodegsw=[], weakha_x = [], weakha_y=[],
              weakha_xxray=[], weakha_yxray=[]):
    '''
    for doing ssfrmass diagram with sdss galaxies scatter plotted
    '''
    nonzero_ssfr_agn = np.where(yvals[agnfilt] >-20)
    nonzero_ssfr_hii = np.where(yvals[nonagnfilt] >-20)
    print('BPT-HII mass avg:',np.mean(xvals[nonagnfilt][nonzero_ssfr_hii]))
    print('BPT-HII mass std:',np.std(xvals[nonagnfilt][nonzero_ssfr_hii]))

    print('BPT-HII fib ssfr frac avg:',np.mean(yvals[nonagnfilt][nonzero_ssfr_hii]))
    print('BPT-HII fib ssfr frac std:',np.std(yvals[nonagnfilt][nonzero_ssfr_hii]))

    print('BPT-AGN mass avg:',np.mean(xvals[agnfilt][nonzero_ssfr_agn]))
    print('BPT-AGN mass std:',np.std(xvals[agnfilt][nonzero_ssfr_agn]))

    print('BPT-AGN fib ssfr frac avg:',np.mean(yvals[agnfilt][nonzero_ssfr_agn]))
    print('BPT-AGN fib ssfr frac std:',np.std(yvals[agnfilt][nonzero_ssfr_agn]))
    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > -14) &( bgy < -8) & (bgx<12.5)&(bgx > 7.5) )[0]
        nx = 5.0/0.01
        ny = 6.0/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, edgecolors='none' )
        plt.scatter(0,0,color='indianred', marker='.', edgecolors='none', label=r'SDSS DR7 (Weak Lines)')
    if len(weakha_xxray) !=0 and len(weakha_yxray) !=0:
        plt.scatter(weakha_xxray, weakha_yxray,color='k',marker='x' , label=r'X-ray AGN (Weak Emission)', alpha=0.6, zorder=2)

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()
        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20,zorder=10)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20,zorder=11)
    plt.ylabel(r'log(sSFR$_{\mathrm{fiber}}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}})$',fontsize=20)
    plt.xlim([7.5,12.5])
    plt.ylim([-14,-8])
    plt.legend(fontsize=15,frameon = False, loc=3, bbox_to_anchor=(-0.04, .04))
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/fibssfr_mass'+filename+'.png', dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/fibssfr_mass'+filename+'.pdf', dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/fibssfr_mass'+filename+'.eps', dpi=150, format='eps')
        plt.close(fig)
    else:

        plt.show()
'''
plotfibssfrm(xmm3eldiagmed_xrfilt.mass,xmm3eldiagmed_xrfilt.fibsfr-xmm3eldiagmed_xrfilt.fibmass,
          nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt,
          EL_m1.mass, EL_m1.fibsfr - EL_m1.fibmass,   save=False, alph=0.1)
plotfibssfrm(xmm3eldiagmed_xrfilt.mass,xmm3eldiagmed_xrfilt.fibsfrgsw-xmm3eldiagmed_xrfilt.mass,
          nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt,
          EL_m1.mass, EL_m1.fibsfrgsw - EL_m1.mass,   save=False, alph=0.1)


plotfibssfrm(xmm3eldiagmed_xrfilt.mass,xmm3eldiagmed_xrfilt.fibssfr,
          nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt,
          EL_m1.mass, EL_m1.fibssfr,   save=False, alph=0.1, ccode=xmm3eldiagmed_xrfilt.z)


plotfibssfrm(xmm3eldiagmed_xrfilt.mass,xmm3eldiagmed_xrfilt.ssfr,
          nonagn_3xmmm_xrfilt, agn_3xmmm_xrfilt,
          EL_m1.mass, EL_m1.ssfr,   save=False,alph=0.1, weakha_xxray = EL_3xmm.weakmass,
          weakha_yxray = EL_3xmm.weakssfr )


'''


def plotoiiimass(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, filename='',
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[], ccodename='X-ray Fraction', ccodecolor='Blues',leg=True):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    nonzero_oiii_agn = np.where(yvals[agnfilt] >0)
    nonzero_oiii_hii = np.where(yvals[nonagnfilt] >0)
    print('BPT-HII mass avg:',np.mean(xvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII mass std:',np.std(xvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-HII oiii frac avg:',np.mean(yvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII oiii frac std:',np.std(yvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-AGN mass avg:',np.mean(xvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN mass std:',np.std(xvals[agnfilt][nonzero_oiii_agn]))

    print('BPT-AGN oiii frac avg:',np.mean(yvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN oiii frac std:',np.std(yvals[agnfilt][nonzero_oiii_agn]))
    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > 37) &( bgy < 43) & (bgx<12)&(bgx > 8) )[0]
        nx = 4/0.01
        ny = 6/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, label=r'SDSS DR7 (Weak H$\alpha$)')
    if len(weakha_xs82) !=0 and len(weakha_ys82) !=0:
        plt.scatter(weakha_xs82,weakha_ys82,color='k',marker='x' , label=r'X-ray AGN (Weak H$\alpha$)')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap=ccodecolor,marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap=ccodecolor,marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label(ccodename)
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    plt.ylabel(r'log(L$_\mathrm{[OIII]}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([37,43])
    plt.xlim([8,12])
    if leg:
        plt.legend(fontsize=15, frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/OIIILum_Mass_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILum_Mass_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILum_Mass_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiimass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass, np.log10(EL_m1.oiiilum),
        save=True,alph=0.1)
#BPT SF BG
plotoiiimass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass[nonagn_gsw], np.log10(EL_m1.oiiilum[nonagn_gsw]),
        save=True,alph=0.1, filename='nonagn_bg')
#BPT-AGN bg
plotoiiimass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass[agn_gsw], np.log10(EL_m1.oiiilum[agn_gsw]),
        save=True,alph=0.1, filename='agn_bg')
plotoiiimass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass, np.log10(EL_m1.oiiilum), ccode=xmm3eldiagmed_xrfilt.ssfr,ccodename='log(sSFR)',
          ccodecolor='Reds',leg=False,
        save=True,filename='ccode_ssfr',alph=0.1)
plotoiiimass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass, np.log10(EL_m1.oiiilum), ccode=np.log10(xmm3eldiagmed_xrfilt.halplum),ccodename=r'log(L$_{H\alpha}$)',
          ccodecolor='Reds',leg=False,
        save=True,filename='ccodehalplum',alph=0.1)

plotoiiimass(xmm3eldiagmed.mass,np.log10(xmm3eldiagmed.oiiilum),
          nonagn_3xmmm[fullxray_xmm.validnoagn][fullxray_xmm.likelyagnbpthii],
          agn_3xmmm[fullxray_xmm.validagn][fullxray_xmm.likelyagnbptagn],
          EL_m1.mass, np.log10(EL_m1.oiiilum),
        save=False,alph=0.1,filename='_weakha_bg_inc',weakha_x= EL_m1_oiii.mass,
        weakha_y = np.log10(EL_m1_oiii.oiiilum) )
'''


def plotoiiihalpmass(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, filename='',
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[], ccodename='X-ray Fraction', ccodecolor='Blues',leg=True):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    nonzero_oiii_agn = np.where(yvals[agnfilt] >0)
    nonzero_oiii_hii = np.where(yvals[nonagnfilt] >0)
    print('BPT-HII mass avg:',np.mean(xvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII mass std:',np.std(xvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-HII oiii frac avg:',np.mean(yvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII oiii frac std:',np.std(yvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-AGN mass avg:',np.mean(xvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN mass std:',np.std(xvals[agnfilt][nonzero_oiii_agn]))

    print('BPT-AGN oiii frac avg:',np.mean(yvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN oiii frac std:',np.std(yvals[agnfilt][nonzero_oiii_agn]))
    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > -3) &( bgy < 3) & (bgx<12)&(bgx > 8) )[0]
        nx = 4/0.01
        ny = 6/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, label=r'SDSS DR7 (Weak H$\alpha$)')
    if len(weakha_xs82) !=0 and len(weakha_ys82) !=0:
        plt.scatter(weakha_xs82,weakha_ys82,color='k',marker='x' , label=r'X-ray AGN (Weak H$\alpha$)')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap=ccodecolor,marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap=ccodecolor,marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label(ccodename)
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$/L$_{H\alpha}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([-3,3])
    plt.xlim([8,12])
    if leg:
        plt.legend(fontsize=15, frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/OIIIhalpLum_Mass_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIIhalpLum_Mass_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIIhalpLum_Mass_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiihalpmass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum)-np.log10(xmm3eldiagmed_xrfilt.halplum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass, np.log10(EL_m1.oiiilum)-np.log10(EL_m1.halplum),
        save=True,alph=0.1)
#BPT SF BG
plotoiiihalpmass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum)-np.log10(xmm3eldiagmed_xrfilt.halplum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass[nonagn_gsw], np.log10(EL_m1.oiiilum[nonagn_gsw])-np.log10(EL_m1.halplum[nonagn_gsw]),
        save=True,alph=0.1, filename='nonagn_bg')
#BPT-AGN bg
plotoiiihalpmass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum)-np.log10(xmm3eldiagmed_xrfilt.halplum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass[agn_gsw], np.log10(EL_m1.oiiilum[agn_gsw])-np.log10(EL_m1.halplum[agn_gsw]),
        save=True,alph=0.1, filename='agn_bg')

'''



def plotoiiisfrmass(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, filename='',
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[], ccodename='X-ray Fraction', ccodecolor='Blues',leg=True):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    nonzero_oiii_agn = np.where(yvals[agnfilt] >0)
    nonzero_oiii_hii = np.where(yvals[nonagnfilt] >0)
    print('BPT-HII mass avg:',np.mean(xvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII mass std:',np.std(xvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-HII oiiisfr frac avg:',np.mean(yvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII oiiisfr frac std:',np.std(yvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-AGN mass avg:',np.mean(xvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN mass std:',np.std(xvals[agnfilt][nonzero_oiii_agn]))

    print('BPT-AGN oiiisfr frac avg:',np.mean(yvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN oiiisfr frac std:',np.std(yvals[agnfilt][nonzero_oiii_agn]))
    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > 37) &( bgy < 43) & (bgx<12)&(bgx > 8) )[0]
        nx = 4/0.01
        ny = 6/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, label=r'SDSS DR7 (Weak H$\alpha$)')
    if len(weakha_xs82) !=0 and len(weakha_ys82) !=0:
        plt.scatter(weakha_xs82,weakha_ys82,color='k',marker='x' , label=r'X-ray AGN (Weak H$\alpha$)')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap=ccodecolor,marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap=ccodecolor,marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label(ccodename)
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    plt.ylabel(r'log(L$_\mathrm{[OIII]/SFR}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([30,45])
    plt.xlim([8,12])
    if leg:
        plt.legend(fontsize=15, frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/OIIILum_Mass_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILum_Mass_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILum_Mass_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiisfrmass(xmm3eldiagmed_xrfilt.mass,xmm3eldiagmed_xrfilt.oiiilumfiboh,
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass, np.log10(EL_m1.oiiilum),
        save=False,alph=0.1)

'''




def plotoiiimasscomb(xvals, yvals, nonagnfilt, agnfilt, bgx1, bgy1, bgx2, bgy2, save=True, filename='',
                 title=None,leg=True):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
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
        fig.savefig('plots/xmm3/png/diagnostic/OIIILum_Mass_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILum_Mass_scat'+filename+'.pdf',dpi=250)
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
          EL_m1.mass[nonagn_gsw], np.log10(EL_m1.oiiilum[nonagn_gsw]),EL_m1.mass[agn_gsw], np.log10(EL_m1.oiiilum[agn_gsw]),
        save=False,leg=False)

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
    nx = 4.5/0.01
    ny = 8/0.02
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
        fig.savefig('plots/xmm3/png/diagnostic/OIIILumdisp_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILumdisp_scat'+filename+'.pdf',dpi=250)
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
          EL_m1.mass[nonagn_gsw], np.log10(EL_m1.oiiilum[nonagn_gsw]/(EL_m1.vdisp[nonagn_gsw]*1e5)**4),EL_m1.mass[agn_gsw], np.log10(EL_m1.oiiilum[agn_gsw]/(EL_m1.vdisp[agn_gsw]*1e5)**4),
        save=False,leg=False)

plotoiiidispmass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum/(xmm3eldiagmed_xrfilt.vdisp*1e5)**4),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass, np.log10(EL_m1.oiiilum/(EL_m1.vdisp*1e5)**4),
        save=True,alph=0.1)
'''


def plotoiiissfr(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, filename='',
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[], ccodename='X-ray Fraction', ccodecolor='Blues',leg=True,m10=False):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    nonzero_oiii_agn = np.where(yvals[agnfilt] >0)
    nonzero_oiii_hii = np.where(yvals[nonagnfilt] >0)
    print('BPT-HII mass avg:',np.mean(xvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII mass std:',np.std(xvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-HII oiii frac avg:',np.mean(yvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII oiii frac std:',np.std(yvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-AGN mass avg:',np.mean(xvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN mass std:',np.std(xvals[agnfilt][nonzero_oiii_agn]))

    print('BPT-AGN oiii frac avg:',np.mean(yvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN oiii frac std:',np.std(yvals[agnfilt][nonzero_oiii_agn]))
    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > 37) &( bgy < 43) & (bgx<-8)&(bgx > -14) )[0]
        nx = 6/0.01
        ny = 6/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, label=r'SDSS DR7 (Weak H$\alpha$)')
    if len(weakha_xs82) !=0 and len(weakha_ys82) !=0:
        plt.scatter(weakha_xs82,weakha_ys82,color='k',marker='x' , label=r'X-ray AGN (Weak H$\alpha$)')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    #plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()
        
        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap=ccodecolor,marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap=ccodecolor,marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label(ccodename)
    else:
        m10filt = np.where(xmm3eldiagmed_xrfilt.mass >10)[0]
        m10nonagn = np.where(xmm3eldiagmed_xrfilt.mass[nonagnfilt]>10)[0]
        m10agn = np.where(xmm3eldiagmed_xrfilt.mass[agnfilt]>10)[0]
        if m10:

            if len(list(nonagnfilt))!=0:
                plt.scatter(xvals[nonagnfilt][m10nonagn],yvals[nonagnfilt][m10nonagn],
                      marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
            if len(list(agnfilt)) !=0:
                plt.scatter(xvals[agnfilt][m10agn],yvals[agnfilt][m10agn],
                      marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
        else:
            if len(list(nonagnfilt))!=0:
                plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                      marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
            if len(list(agnfilt)) !=0:
                plt.scatter(xvals[agnfilt],yvals[agnfilt],
                      marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    plt.ylabel(r'log(L$_\mathrm{[OIII]}$)',fontsize=20)
    plt.xlabel(r'log(sSFR)',fontsize=20)
    plt.ylim([37,43])
    plt.xlim([-14,-8])
    if leg:
        plt.legend(fontsize=15, frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/OIIILum_ssfr_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILum_ssfr_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILum_ssfr_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiissfr(xmm3eldiagmed_xrfilt.ssfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.ssfr, np.log10(EL_m1.oiiilum),
        save=True,alph=0.1, leg=False)
#BPT SF BG
plotoiiissfr(xmm3eldiagmed_xrfilt.ssfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.ssfr[nonagn_gsw], np.log10(EL_m1.oiiilum)[nonagn_gsw],
        save=True,alph=0.1, leg=False, filename='nonagn_bg')
#BPT-AGN BG
plotoiiissfr(xmm3eldiagmed_xrfilt.ssfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.ssfr[agn_gsw], np.log10(EL_m1.oiiilum)[agn_gsw],
        save=True,alph=0.1, filename='agn_bg',leg=False)


        save=True,alph=0.1, leg=False)
plotoiiissfr(xmm3eldiagmed_xrfilt.ssfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.ssfr, np.log10(EL_m1.oiiilum),
        save=True,alph=0.1, leg=False, m10=True, filename='m10')
plotoiiissfr(xmm3eldiagmed_xrfilt.ssfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.ssfr, np.log10(EL_m1.oiiilum), ccode=xmm3eldiagmed_xrfilt.mass,ccodename='Mass',
          ccodecolor='Reds',leg=False,
        save=True,filename='ccode_mass',alph=0.1)
'''

def plotoiiisfr(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, filename='',
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[], ccodename='X-ray Fraction', ccodecolor='Blues',leg=True,m10=False):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    nonzero_oiii_agn = np.where(yvals[agnfilt] >0)
    nonzero_oiii_hii = np.where(yvals[nonagnfilt] >0)
    print('BPT-HII mass avg:',np.mean(xvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII mass std:',np.std(xvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-HII oiii frac avg:',np.mean(yvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII oiii frac std:',np.std(yvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-AGN mass avg:',np.mean(xvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN mass std:',np.std(xvals[agnfilt][nonzero_oiii_agn]))

    print('BPT-AGN oiii frac avg:',np.mean(yvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN oiii frac std:',np.std(yvals[agnfilt][nonzero_oiii_agn]))
    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > 37) &( bgy < 43) & (bgx>-2.5)&(bgx < 4.5) )[0]
        nx = 6/0.01
        ny = 6/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, label=r'SDSS DR7 (Weak H$\alpha$)')
    if len(weakha_xs82) !=0 and len(weakha_ys82) !=0:
        plt.scatter(weakha_xs82,weakha_ys82,color='k',marker='x' , label=r'X-ray AGN (Weak H$\alpha$)')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    #plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()
        
        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap=ccodecolor,marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap=ccodecolor,marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label(ccodename)
    else:
        m10filt = np.where(xmm3eldiagmed_xrfilt.mass >10)[0]
        m10nonagn = np.where(xmm3eldiagmed_xrfilt.mass[nonagnfilt]>10)[0]
        m10agn = np.where(xmm3eldiagmed_xrfilt.mass[agnfilt]>10)[0]
        if m10:

            if len(list(nonagnfilt))!=0:
                plt.scatter(xvals[nonagnfilt][m10nonagn],yvals[nonagnfilt][m10nonagn],
                      marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
            if len(list(agnfilt)) !=0:
                plt.scatter(xvals[agnfilt][m10agn],yvals[agnfilt][m10agn],
                      marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
        else:
            if len(list(nonagnfilt))!=0:
                plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                      marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
            if len(list(agnfilt)) !=0:
                plt.scatter(xvals[agnfilt],yvals[agnfilt],
                      marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    plt.ylabel(r'log(L$_\mathrm{[OIII]}$)',fontsize=20)
    plt.xlabel(r'log(Fiber SFR)',fontsize=20)
    plt.ylim([37,43])
    plt.xlim([-4.5,4.5])
    if leg:
        plt.legend(fontsize=15, frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/OIIILum_sfr_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILum_sfr_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILum_sfr_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiisfr(xmm3eldiagmed_xrfilt.fibsfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.fibsfr, np.log10(EL_m1.oiiilum),
        save=False,alph=0.1, leg=False)
#BPT SF BG
plotoiiisfr(xmm3eldiagmed_xrfilt.sfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.sfr[nonagn_gsw], np.log10(EL_m1.oiiilum)[nonagn_gsw],
        save=True,alph=0.1,filename='nonagn_bg', leg=False)
#BPT-AGN BG
plotoiiisfr(xmm3eldiagmed_xrfilt.sfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.sfr[agn_gsw], np.log10(EL_m1.oiiilum)[agn_gsw],
        save=True,alph=0.1, leg=False, filename='agn_bg')


        save=True,alph=0.1, leg=False)
plotoiiisfr(xmm3eldiagmed_xrfilt.ssfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.ssfr, np.log10(EL_m1.oiiilum),
        save=True,alph=0.1, leg=False, m10=True, filename='m10')
plotoiiisfr(xmm3eldiagmed_xrfilt.ssfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.ssfr, np.log10(EL_m1.oiiilum), ccode=xmm3eldiagmed_xrfilt.mass,ccodename='Mass',
          ccodecolor='Reds',leg=False,
        save=True,filename='ccode_mass',alph=0.1)
'''

def plotlxissfr(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, filename='',
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[], ccodename='X-ray Fraction', ccodecolor='Blues',leg=True,m10=False):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    nonzero_oiii_agn = np.where(yvals[agnfilt] >0)
    nonzero_oiii_hii = np.where(yvals[nonagnfilt] >0)
    print('BPT-HII mass avg:',np.mean(xvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII mass std:',np.std(xvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-HII oiii frac avg:',np.mean(yvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII oiii frac std:',np.std(yvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-AGN mass avg:',np.mean(xvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN mass std:',np.std(xvals[agnfilt][nonzero_oiii_agn]))

    print('BPT-AGN oiii frac avg:',np.mean(yvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN oiii frac std:',np.std(yvals[agnfilt][nonzero_oiii_agn]))
    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > 37) &( bgy < 43) & (bgx<-8)&(bgx > -14) )[0]
        nx = 6/0.01
        ny = 6/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, label=r'SDSS DR7 (Weak H$\alpha$)')
    if len(weakha_xs82) !=0 and len(weakha_ys82) !=0:
        plt.scatter(weakha_xs82,weakha_ys82,color='k',marker='x' , label=r'X-ray AGN (Weak H$\alpha$)')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    #plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()
        
        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap=ccodecolor,marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap=ccodecolor,marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label(ccodename)
    else:
        m10filt = np.where(xmm3eldiagmed_xrfilt.mass >10)[0]
        m10nonagn = np.where(xmm3eldiagmed_xrfilt.mass[nonagnfilt]>10)[0]
        m10agn = np.where(xmm3eldiagmed_xrfilt.mass[agnfilt]>10)[0]
        if m10:

            if len(list(nonagnfilt))!=0:
                plt.scatter(xvals[nonagnfilt][m10nonagn],yvals[nonagnfilt][m10nonagn],
                      marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
            if len(list(agnfilt)) !=0:
                plt.scatter(xvals[agnfilt][m10agn],yvals[agnfilt][m10agn],
                      marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
        else:
            if len(list(nonagnfilt))!=0:
                plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                      marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
            if len(list(agnfilt)) !=0:
                plt.scatter(xvals[agnfilt],yvals[agnfilt],
                      marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    plt.ylabel(r'log(L$_\mathrm{[OIII]}$)',fontsize=20)
    plt.xlabel(r'log(sSFR)',fontsize=20)
    plt.ylim([37,43])
    plt.xlim([-14,-8])
    if leg:
        plt.legend(fontsize=15, frameon=False)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/OIIILum_ssfr_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILum_ssfr_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILum_ssfr_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotlxissfr(xmm3eldiagmed_xrfilt.ssfr,fullxray_xmm.lums,
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.ssfr, np.log10(EL_m1.oiiilum),
        save=True,alph=0.1, leg=False)
plotlxissfr(xmm3eldiagmed_xrfilt.ssfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.ssfr, np.log10(EL_m1.oiiilum),
        save=True,alph=0.1, leg=False, m10=True, filename='m10')
plotlxissfr(xmm3eldiagmed_xrfilt.ssfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.ssfr, np.log10(EL_m1.oiiilum), ccode=xmm3eldiagmed_xrfilt.mass,ccodename='Mass',
          ccodecolor='Reds',leg=False,
        save=True,filename='ccode_mass',alph=0.1)

'''


def plotoiiimassedd(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, filename='',
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[]):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    nonzero_oiii_agn = np.where(yvals[agnfilt] >0)
    nonzero_oiii_hii = np.where(yvals[nonagnfilt] >0)
    print('BPT-HII mass avg:',np.mean(xvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII mass std:',np.std(xvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-HII oiii frac avg:',np.mean(yvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII oiii frac std:',np.std(yvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-AGN mass avg:',np.mean(xvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN mass std:',np.std(xvals[agnfilt][nonzero_oiii_agn]))

    print('BPT-AGN oiii frac avg:',np.mean(yvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN oiii frac std:',np.std(yvals[agnfilt][nonzero_oiii_agn]))
    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > 25) &( bgy < 34) & (bgx<12.5)&(bgx > 8) )[0]
        nx = 4.5/0.01
        ny = 9/0.01
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, label=r'SDSS DR7 (Weak H$\alpha$)')
    if len(weakha_xs82) !=0 and len(weakha_ys82) !=0:
        plt.scatter(weakha_xs82,weakha_ys82,color='k',marker='x' , label=r'X-ray AGN (Weak H$\alpha$)')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    #plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    if cont:
        #extent = [np.min(cont.rangex),np.max(cont.rangex),np.min(cont.rangey), np.max(cont.rangey)]
        #plt.imshow(cont.grid, extent=extent,origin='lower')
        '''
        levs=[0.02,0.04,0.07,0.11,0.13]
        locs = np.vstack([posx,posy]).transpose()
        CS = plt.contour(cont.meshx, cont.meshy, cont.grid,cmap='Blues',levels=levs)
        norm= matplotlib.colors.Normalize(vmin=contaminations.min(), vmax=contaminations.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap = CS.cmap)
        sm.set_array([])
        plt.colorbar(sm).set_label('Xray Fraction')
        plt.text(-0.19,-0.379,'0.02',fontsize=15)
        plt.text(0.005,-0.21,'0.04',fontsize=15)
        plt.text(0.18,-0.0267,'0.07',fontsize=15)
        plt.text(0.294,0.12,'0.11',fontsize=15)
        plt.text(0.294,0.2768,'0.13',fontsize=15)
        '''
        levs = [0.02,0.04,0.07,0.11,0.15]

        locs = np.vstack([posx,posy]).transpose()

        CS = plt.contour(cont.meshx, cont.meshy, cont.grid,cmap='Blues',levels=levs)

        norm= matplotlib.colors.Normalize(vmin=contaminations.min(), vmax=contaminations.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap = CS.cmap)
        sm.set_array([])
        #plt.clim(contaminations.min(),contaminations.max())

        plt.colorbar(sm).set_label('Xray Fraction')

        plt.text(-0.27,-0.56,'0.02',fontsize=15)
        plt.text(-0.033,-0.349,'0.04',fontsize=15)
        plt.text(0.18,-0.12,'0.07',fontsize=15)
        plt.text(0.28,0.09,'0.11',fontsize=15)
        plt.text(0.325,0.371,'0.15',fontsize=15)

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
plotoiiimassedd(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum)-xmm3eldiagmed_xrfilt.mass,
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass, np.log10(EL_m1.oiiilum)-EL_m1.mass,
        save=True,alph=0.1)
plotoiiimass(s82eldiag.mass,np.log10(s82eldiag.oiiilum),nonagn,agn,
             EL_m1.mass, np.log10(EL_m1.oiiilum),save=True,alph=0.1)
plotoiiimass(s82eldiag.mass,np.log10(s82eldiag.oiiilum),nonagn,agn,
             EL_m1.mass[agn_gsw], np.log10(EL_m1.oiiilum[agn_gsw]),save=True,alph=0.1,filename='_bptagn_bg'),#weakha_xs82=s82eldiag_oiii.mass, weakha_ys82=np.log10(s82eldiag_oiii.oiiilum))
plotoiiimass(s82eldiag.mass,np.log10(s82eldiag.oiiilum),nonagn,agn,
             EL_m1.mass[nonagn_gsw], np.log10(EL_m1.oiiilum[nonagn_gsw]),
             save=True,alph=0.1,filename='_bpthii_bg')#,weakha_xs82=s82eldiag_oiii.mass, weakha_ys82=np.log10(s82eldiag_oiii.oiiilum))
plotoiiimass(s82eldiag.mass,np.log10(s82eldiag.oiiilum),nonagn,agn,EL_m1.mass, np.log10(EL_m1.oiiilum),
             save=True,alph=0.1,filename='_weakha_bg_inc',weakha_x= EL_m1_oiii.mass,weakha_y = np.log10(EL_m1_oiii.oiiilum) ,
             weakha_xs82=s82eldiag_oiii.mass, weakha_ys82=np.log10(s82eldiag_oiii.oiiilum))
plotoiiimass(xmm3eldiagmed.mass,np.log10(xmm3eldiagmed.oiiilum),
          nonagn_3xmmm[fullxray_xmm.validnoagn][fullxray_xmm.likelyagnbpthii],
          agn_3xmmm[fullxray_xmm.validagn][fullxray_xmm.likelyagnbptagn],
          EL_m1.mass, np.log10(EL_m1.oiiilum),
        save=False,alph=0.1,filename='_weakha_bg_inc',weakha_x= EL_m1_oiii.mass,
        weakha_y = np.log10(EL_m1_oiii.oiiilum) )
'''
def plotoiiidispmass(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, filename='',
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[]):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    nonzero_oiii_agn = np.where(yvals[agnfilt] >0 )[0]
    nonzero_oiii_hii = np.where((yvals[nonagnfilt] >0) &(np.isfinite(yvals[nonagnfilt])))[0]
    print('BPT-HII mass avg:',np.mean(xvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII mass std:',np.std(xvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-HII oiii/sigma^4 frac avg:',np.nanmean(yvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII oiii/sigma^4 frac std:',np.nanstd(yvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-AGN mass avg:',np.mean(xvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN mass std:',np.std(xvals[agnfilt][nonzero_oiii_agn]))

    print('BPT-AGN oiii/sigma^4 frac avg:',np.mean(yvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN oiii/sigma^4 frac std:',np.std(yvals[agnfilt][nonzero_oiii_agn]))
    fig = plt.figure()

    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > 7.5) &( bgy < 15.5) & (bgx<12.5)&(bgx > 8) )[0]
        nx = 4.5/0.01
        ny = 8/0.02
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, label=r'SDSS DR7 (Weak H$\alpha$)')
    if len(weakha_xs82) !=0 and len(weakha_ys82) !=0:
        plt.scatter(weakha_xs82,weakha_ys82,color='k',marker='x' , label=r'X-ray AGN (Weak H$\alpha$)')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    if cont:

        levs = [0.02,0.04,0.07,0.11,0.15]

        locs = np.vstack([posx,posy]).transpose()

        CS = plt.contour(cont.meshx, cont.meshy, cont.grid,cmap='Blues',levels=levs)

        norm= matplotlib.colors.Normalize(vmin=contaminations.min(), vmax=contaminations.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap = CS.cmap)
        sm.set_array([])
        #plt.clim(contaminations.min(),contaminations.max())

        plt.colorbar(sm).set_label('Xray Fraction')

        plt.text(-0.27,-0.56,'0.02',fontsize=15)
        plt.text(-0.033,-0.349,'0.04',fontsize=15)
        plt.text(0.18,-0.12,'0.07',fontsize=15)
        plt.text(0.28,0.09,'0.11',fontsize=15)
        plt.text(0.325,0.371,'0.15',fontsize=15)

    plt.ylabel(r'log(L$_\mathrm{[OIII]}/\sigma_{*}^4$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([7.5,15.5])
    plt.xlim([8,12])

    plt.legend(fontsize=15,frameon=False, loc=3, bbox_to_anchor = (0.0,0.03))
    if title:
        plt.title(title,fontsize=30)

    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/OIIILumdisp_Mass_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILumdisp_Mass_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILumdisp_Mass_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiidispmass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum/(xmm3eldiagmed_xrfilt.vdisp*1e5)**4),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass, np.log10(EL_m1.oiiilum/(EL_m1.vdisp*1e5)**4),
        save=True,alph=0.1)

#BPT SF
plotoiiidispmass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum/(xmm3eldiagmed_xrfilt.vdisp*1e5)**4),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass[nonagn_gsw], np.log10(EL_m1.oiiilum/(EL_m1.vdisp*1e5)**4)[nonagn_gsw],
        save=True,alph=0.1,filename='agn_bg')
#BPT-AGN
plotoiiidispmass(xmm3eldiagmed_xrfilt.mass,np.log10(xmm3eldiagmed_xrfilt.oiiilum/(xmm3eldiagmed_xrfilt.vdisp*1e5)**4),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.mass[agn_gsw], np.log10(EL_m1.oiiilum/(EL_m1.vdisp*1e5)**4)[agn_gsw],
        save=True,alph=0.1,filename='nonagn_bg')

'''
def plotoiiidispfibsfr(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, filename='',
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[], leg=False):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    nonzero_oiii_agn = np.where(yvals[agnfilt] >0 )[0]
    nonzero_oiii_hii = np.where((yvals[nonagnfilt] >0) &(np.isfinite(yvals[nonagnfilt])))[0]
    print('BPT-HII mass avg:',np.mean(xvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII mass std:',np.std(xvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-HII oiii/sigma^4 frac avg:',np.nanmean(yvals[nonagnfilt][nonzero_oiii_hii]))
    print('BPT-HII oiii/sigma^4 frac std:',np.nanstd(yvals[nonagnfilt][nonzero_oiii_hii]))

    print('BPT-AGN mass avg:',np.mean(xvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN mass std:',np.std(xvals[agnfilt][nonzero_oiii_agn]))

    print('BPT-AGN oiii/sigma^4 frac avg:',np.mean(yvals[agnfilt][nonzero_oiii_agn]))
    print('BPT-AGN oiii/sigma^4 frac std:',np.std(yvals[agnfilt][nonzero_oiii_agn]))
    fig = plt.figure()

    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
        valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
                (bgy > 7.5) &( bgy < 15.5) & (bgx<4)&(bgx > -4) )[0]
        nx = 8/0.02
        ny = 8/0.02
        plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
        #plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, label=r'SDSS DR7 (Weak H$\alpha$)')
    if len(weakha_xs82) !=0 and len(weakha_ys82) !=0:
        plt.scatter(weakha_xs82,weakha_ys82,color='k',marker='x' , label=r'X-ray AGN (Weak H$\alpha$)')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT-HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT-AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
    if cont:

        levs = [0.02,0.04,0.07,0.11,0.15]

        locs = np.vstack([posx,posy]).transpose()

        CS = plt.contour(cont.meshx, cont.meshy, cont.grid,cmap='Blues',levels=levs)

        norm= matplotlib.colors.Normalize(vmin=contaminations.min(), vmax=contaminations.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap = CS.cmap)
        sm.set_array([])
        #plt.clim(contaminations.min(),contaminations.max())

        plt.colorbar(sm).set_label('Xray Fraction')

        plt.text(-0.27,-0.56,'0.02',fontsize=15)
        plt.text(-0.033,-0.349,'0.04',fontsize=15)
        plt.text(0.18,-0.12,'0.07',fontsize=15)
        plt.text(0.28,0.09,'0.11',fontsize=15)
        plt.text(0.325,0.371,'0.15',fontsize=15)

    plt.ylabel(r'log(L$_\mathrm{[OIII]}/\sigma_{*}^4$)',fontsize=20)
    plt.xlabel(r'log(Fiber SFR)',fontsize=20)
    plt.ylim([7.5,15.5])
    plt.xlim([-4,4])
    if leg:
        plt.legend(fontsize=15,frameon=False, loc=3, bbox_to_anchor = (0.0,0.03))
    if title:
        plt.title(title,fontsize=30)

    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/diagnostic/OIIILumdisp_fibsfr_scat'+filename+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/diagnostic/OIIILumdisp_fibsfr_scat'+filename+'.pdf',dpi=250)
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/diagnostic/OIIILumdisp_fibsfr_scat'+filename+'.eps',dpi=150,format='eps')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiidispfibsfr(xmm3eldiagmed_xrfilt.fibsfr,np.log10(xmm3eldiagmed_xrfilt.oiiilum/(xmm3eldiagmed_xrfilt.vdisp*1e5)**4),
          nonagn_3xmmm_xrfilt,
          agn_3xmmm_xrfilt,
          EL_m1.fibsfr, np.log10(EL_m1.oiiilum/(EL_m1.vdisp*1e5)**4),
        save=False,alph=0.1)
       
'''
def plotmasscomp(mass_sdss,massgsw):
    plt.scatter(mass_sdss, massgsw)
    plt.xlabel('Mass SDSS')
    plt.ylabel('Mass GSW')
    plt.tight_layout()
    plt.show()

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

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
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

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
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
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
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

#use_mass_bins(mass_bins_sdss,mass_bins_s82,start=8, step=1.0)
#use_mass_bins(mass_bins_sdss,mass_bins_s82,start=8, step=0.5)
#use_mass_bins(massfrac_bins_sdss,massfrac_bins_s82,start=0, step=0.1,frac='frac')

#use_z_bins(z_bins_sdss,z_bins_s82,start=0, step=.05)
#use_z_bins(z_bins_sdss,z_bins_s82,start=0, step=.1)

#use_mass_bins(mass_bins_sdss,mass_bins_s82,start=8, step=0.5)
#ad=np.where((all_sdss_spec_z[spec_inds_allm1][valid_bpt2]<=.3)&(all_sdss_spec_z[spec_inds_allm1][valid_bpt2]>0 ) )[0]
#ad=np.where((all_sdss_spec_z<=.3)&(all_sdss_spec_z>0 ) )[0]

'''
Below are X-ray plots

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

def plot_ra_dec_map_tog():
    '''made for plotting all of the catalogs together '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="aitoff")
    #plt.scatter(ra_x1, dec_x1,c='k')
    plt.scatter(np.radians(conv_ra(ra_a1)+120),np.radians(dec_a1),s=1,c='b',label='GSWLC All Sky')
    plt.scatter(np.radians(conv_ra(ra_m1)+120), np.radians(dec_m1),s=1,c='g',label='GSWLC Medium')
    plt.scatter(np.radians(conv_ra(ra_d1)+120),np.radians(dec_d1),s=1,c='r',label='GSWLC Deep')
    plt.scatter(np.radians(conv_ra(ra_filt_c)+120),np.radians(dec_filt_c),c='k',label='Chandra (Stripe 82X), 0.0 < z <0.3',marker='x')
    plt.scatter(np.radians(conv_ra(ra_filt_x)+120),np.radians(dec_filt_x),c='c',label='XMM Newton (Stripe 82X: AO10), 0.0 < z <0.3',marker='x')
    plt.scatter(np.radians(conv_ra(ra_filt_x2)+120),np.radians(dec_filt_x2),c='m',label='XMM Newton (Stripe 82X: AO13), 0.0 < z <0.3)',marker='x')
    plt.xlabel('RA')
    ax.set_xticklabels(np.arange(270,-91,-30))
    plt.ylabel('Dec')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_stripe82_filt():
    ''' plot the filtered section of stripe 82'''

    fig = plt.figure(figsize=(1920/mydpi,1080/mydpi))
   	#ax = fig.add_subplot(111, projection="aitoff")
    plt.scatter(conv_ra(m1Cat_GSW.allra), m1Cat_GSW.alldec,s=1,c='g',label='GSWLC Medium',marker=',',alpha=.1)
    #ax.set_xticklabels(np.arange(270,-91,-30))
    plt.scatter(conv_ra(Chandra.ra),Chandra.dec,c='k',label=r'Chandra (Stripe 82X), 0.0 $<$ z $<$ 0.3',marker=',',s=1,alpha=.3)
    plt.scatter(conv_ra(XMM1.ra),XMM1.dec,c='c',label=r'XMM Newton (Stripe 82X: AO10), 0.0 $<$ z $<$ 0.3',marker=',',s=1,alpha=.3)
    plt.scatter(conv_ra(XMM2.ra),XMM2.dec,c='m',label=r'XMM Newton (Stripe 82X: AO13), 0.0 $<$ z $<$ 0.3)',marker=',',s=1,alpha=.3)
    plt.xlim([-60,60])

    plt.ylim([-1.2,1.2])
    plt.xlabel('RA')

    plt.ylabel('Dec')
    plt.legend(fontsize=20,markerscale=10)
    plt.tight_layout()
    fig.savefig('plots/xmm3/pdf/sky/gswlc_ra_dec_medsky_xray_flats82.pdf',bbox_inches='tight',dpi=5000,format='pdf')
    fig.savefig('plots/xmm3/png/sky/gswlc_ra_dec_medsky_xray_flats82.png',bbox_inches='tight',dpi=500,format='png')
    plt.close(fig)#plt.show()
#plot_stripe82_filt()
def plot_ra_dec_map_sep(xmm3, gsw, save=False,filename='',filt=[], agnfilt=[], nonagnfilt=[]):
    '''spherical'''
    #medium depth
    fig = plt.figure(figsize=(1920/mydpi,1080/mydpi))
    ax = fig.add_subplot(111, projection="aitoff")
    plt.scatter(np.radians(conv_ra(gsw.allra) +120), np.radians(gsw.alldec),s=15,c='g',label='GSWLC-M',marker='.',alpha=.15)
    plt.xlabel('RA',fontsize=40)
    st = -np.pi+np.pi/6
    pi = np.pi
    posx = [st-pi/18,st+pi/2-pi/18,st+pi-pi/36, st+3*pi/2]
    posy = -pi/6
    labels=[270,180,90,0]
    for i in range(len(posx)):
        plt.text(posx[i],posy,labels[i],fontsize=40)
    ax.set_xticks([st,st+pi/2,st+pi, st+3*pi/2])
    ax.set_xticklabels([],[])
    ax.set_yticks([-np.radians(75),-pi/3,-pi/6, 0, pi/6,pi/3,np.radians(75)])
    plt.axvline(x=-np.pi)
    #ax.text(-np.pi,0,'-270',fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=40)

    #ax.set_yticklabels(fontsize=40)
    if len(filt) != 0:
        #plt.scatter(np.radians(conv_ra(xmm3.matchra[filt][agnfilt])+120),np.radians(xmm3.matchdec[filt][agnfilt]),c='k',label='3XMM-DR6 BPT-AGN',marker='o',facecolors='none',edgecolors='k',alpha=1,s=15)
        plt.scatter(np.radians(conv_ra(xmm3.matchra[filt][nonagnfilt])+120),np.radians(xmm3.matchdec[filt][nonagnfilt]),c='b',label='3XMM-DR6 BPT-HII',marker='^',alpha=1,s=15)

    else:
        plt.scatter(np.radians(conv_ra(xmm3.matchra[filt])+120),np.radians(xmm3.matchdec[filt]),c='k',label='3XMM-DR6',marker='x',facecolors='none',edgecolors='k',alpha=1,s=15)

    plt.ylabel('Dec',fontsize=40)
    plt.legend(fontsize=30,markerscale=5)
    plt.grid(True)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/pdf/sky/gswlc_ra_dec_'+filename+'sky_xray.pdf',bbox_inches='tight',dpi=5000,format='pdf')
        fig.savefig('plots/xmm3/png/sky/gswlc_ra_dec_'+filename+'sky_xray.png',bbox_inches='tight',dpi=500,format='png')
        plt.close(fig)
    else:
        plt.show()

'''
plot_ra_dec_map_sep(m1Cat_GSW_3xmm, m1Cat_GSW, save=False,filename='med', filt=make_m1_3xmm[halp_filt_3xmm_med][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr],
                    nonagnfilt = nonagn_3xmmm_xrfilt, agnfilt=agn_3xmmm_xrfilt)
plot_ra_dec_map_sep(m1Cat_GSW_3xmm_all, m1Cat_GSW, save=False,filename='med', filt=make_m1_3xmm_all[halp_filt_3xmm_med_all][fullxray_xmm_all.valid][fullxray_xmm_all.likelyagn_xr],
                    nonagnfilt = nonagn_3xmmm_all_xrfilt, agnfilt=agn_3xmmm_all_xrfilt)
plot_ra_dec_map_sep(d1Cat_GSW_3xmm, d1Cat_GSW, save=True,filename='deep')
plot_ra_dec_map_sep(a1Cat_GSW_3xmm, a1Cat_GSW, save=True,filename='all')

'''
def plot_ra_dec_map_flat(xmm3, gsw, save=False,filename='',filt=[], agnfilt=[], nonagnfilt=[]):
    #medium depth
    fig = plt.figure(figsize=(1920/mydpi,1080/mydpi))
    ax = fig.add_subplot(111)
    #plt.scatter(conv_ra(gsw.allra)*-1, gsw.alldec,s=15,c='g',label='GSWLC-M',marker='.',alpha=.15)
    plt.xlabel('RA',fontsize=40)
    '''
    st = -np.pi+np.pi/6
    pi = np.pi
    posx = [st-pi/18,st+pi/2-pi/18,st+pi-pi/36, st+3*pi/2]
    posy = -pi/6
    labels=[270,180,90,0]
    for i in range(len(posx)):
        plt.text(posx[i],posy,labels[i],fontsize=40)
    ax.set_xticks([st,st+pi/2,st+pi, st+3*pi/2])
    ax.set_xticklabels([],[])
    ax.set_yticks([-np.radians(75),-pi/3,-pi/6, 0, pi/6,pi/3,np.radians(75)])
    plt.axvline(x=-np.pi)
    '''
    #ax.text(-np.pi,0,'-270',fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=40)

    #ax.set_yticklabels(fontsize=40)
    if len(filt) != 0:
        plt.scatter(conv_ra(xmm3.matchra[filt][agnfilt])*-1,xmm3.matchdec[filt][agnfilt],label='S82X BPT-AGN',marker='o',facecolors='none',edgecolors='k',alpha=1,s=15)
        plt.scatter(conv_ra(xmm3.matchra[filt][nonagnfilt])*-1,xmm3.matchdec[filt][nonagnfilt],c='b',label='S82X BPT-HII',marker='^',facecolors='blue',alpha=1,s=15)

    else:
        plt.scatter(conv_ra(xmm3.matchra[filt])*-1,xmm3.matchdec[filt],c='k',label='3XMM-DR6',marker='x',facecolors='none',edgecolors='k',alpha=1,s=15)
    plt.xlim([300, -60])
    plt.ylim([-30, 90])
    plt.ylabel('Dec',fontsize=40)
    plt.legend(fontsize=30,markerscale=5)
    plt.grid(True)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/pdf/sky/gswlc_ra_dec_'+filename+'sky_xray.pdf',bbox_inches='tight',dpi=5000,format='pdf')
        fig.savefig('plots/xmm3/png/sky/gswlc_ra_dec_'+filename+'sky_xray.png',bbox_inches='tight',dpi=500,format='png')
        plt.close(fig)
    else:
        plt.show()
'''
plot_ra_dec_map_flat(m1Cat_GSW_3xmm, m1Cat_GSW, save=False,filename='med', filt=make_m1_3xmm[halp_filt_3xmm_med][fullxray_xmm.valid][fullxray_xmm.likelyagn_xr],
                    nonagnfilt = nonagn_3xmmm_xrfilt, agnfilt=agn_3xmmm_xrfilt)
plot_ra_dec_map_flat(m1Cat_GSW, m1Cat_GSW, save=False,filename='med', filt=make_m1[halp_filt_s82],
                    nonagnfilt = nonagn, agnfilt=agn)
'''
def plot_ra_dec_map_sep_flat(xmm, gsw, save=False,filename=''):
    '''flats'''
    fig = plt.figure(figsize=(1920/mydpi,1080/mydpi))
    plt.scatter(conv_ra(gsw.allra)*-1, gsw.alldec,s=1,c='g',label='GSWLC Medium',marker=',',alpha=.1)
    plt.xlabel('RA')
    #ax.set_xticklabels(np.arange(270,-91,-30))
    plt.scatter(conv_ra(xmm.matchra)*-1,xmm.matchdec,c='k',label='XMM3',marker=',',s=1,alpha=.3)
    plt.xlim(270,-90)
    plt.ylabel('Dec')
    plt.legend(fontsize=20,markerscale=10)
    plt.grid(True)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/pdf/sky/gswlc_ra_dec_'+filename+'sky_xray_flat.pdf',bbox_inches='tight',dpi=5000,format='pdf')
        fig.savefig('plots/xmm3/png/sky/gswlc_ra_dec_'+filename+'sky_xray_flat.png',bbox_inches='tight',dpi=500,format='png')
        plt.close(fig)
    else:
        plt.show()



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