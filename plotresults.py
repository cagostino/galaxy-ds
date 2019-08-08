#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matchgal import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib

mydpi = 96
plt.rc('font',family='serif')
plt.rc('text',usetex=True)
#plt.rc('text',usetex=False)

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
    
def ratiogdiff(frac,agnfilt, nonagnfilt,fname='',save=False):
    bins = np.copy(s82gdiff.bins)
    fig = plt.figure()
    for i in agnfilt:
        plt.plot(bins,frac[i],'k--')
    if len(agnfilt)!=0:
        plt.plot(bins,frac[agnfilt[0]],'k--',label='BPT AGN')
    for i in nonagnfilt:
        plt.plot(bins,frac[i],'b-.')
    if len(nonagnfilt) !=0:
        plt.plot(bins,frac[nonagnfilt[0]],'b-.',label='BPT HII')
    plt.xlabel('Distance',fontsize=20)
    plt.ylabel(r'N$_{\mathrm{X}}$/N$_{\mathrm{GSW}}$',fontsize=20)
    plt.ylim([0,1])
    plt.xlim([0,2])
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    plt.legend(frameon=False,fontsize=15)  
    if save:
        fig.savefig('plots/s82x/png/distmet/ratios/ratio_nx_ngsw_'+fname+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/s82x/pdf/distmet/ratios/ratio_nx_ngsw_'+fname+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/s82x/eps/distmet/ratios/ratio_nx_ngsw_'+fname+'.eps',dpi=150,format='eps',bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    #plt.title('Ratios fo')
'''
ratiogdiff(s82gdiff.xrgswfracs,[],nonagn,fname='bpthii')
ratiogdiff(s82gdiff.xrgswfracs,agn,[],fname='agn')
ratiogdiff(s82gdiff.xrgswfracs,agn,nonagn,fname='all')

ratiogdiff(s82gdiff.xrgswfracs,[],nonagn,fname='bpthiinew', save=True)
ratiogdiff(s82gdiff.xrgswfracs,agn,[],fname='agnnew', save=True)
ratiogdiff(s82gdiff.xrgswfracs,agn,nonagn,fname='allnew',save=True)
'''
    
def plotcumxrgswrat(fracxr, fracgsw, gals,fname='',save=False):
    bins = np.copy(s82gdiff.bins)
    fig = plt.figure()
    for i in gals:
        plt.plot(bins, fracxr[i],'k-.')
        plt.plot(bins, fracgsw[i],'b--')
    plt.plot(bins, fracxr[i],'k-.',label='S82 Fraction')
    plt.plot(bins, fracgsw[i],'b--',label='GSW Fraction')
    plt.legend(frameon=False,fontsize=15)
    plt.ylim([0,1])
    plt.xlim([0,5])
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.xlabel('Distance',fontsize=20)
    plt.ylabel('Cumulative Ratio',fontsize=20)
    plt.tight_layout()
    if save:
        fig.savefig('plots/s82x/png/distmet/ratios/cumulative_ratios_'+fname+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/s82x/pdf/distmet/ratios/cumulative_ratios_'+fname+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/s82x/eps/distmet/ratios/cumulative_ratios_'+fname+'.eps',dpi=150,format='eps',bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
'''
plotcumxrgswrat(s82gdiff.xrfrac,s82gdiff.gswfrac,agn,fname='agn')
plotcumxrgswrat(s82gdiff.xrfrac,s82gdiff.gswfrac,range(44),fname='all')
plotcumxrgswrat(s82gdiff.xrfrac,s82gdiff.gswfrac,nonagn,fname='bpthii')

plotcumxrgswrat(s82gdiff.xrfrac,s82gdiff.gswfrac,agn,fname='agnnew',save=True)
plotcumxrgswrat(s82gdiff.xrfrac,s82gdiff.gswfrac,range(44),fname='allnew',save=True)
plotcumxrgswrat(s82gdiff.xrfrac,s82gdiff.gswfrac,nonagn,fname='bpthiinew',save=True)
'''   
def plotgdiffhist(gdiffs, s82diffs, sdssid,ngal,nearestx=None):
    relevgsw = np.where((gdiffs <10)&(gdiffs >0))[0] #only want relevant cases in hist
    relevs82 = np.where((s82diffs<10)& (s82diffs>0))[0]
    binsgsw = np.copy(s82gdiff.bins)
    bins82 = np.copy(s82gdiff.bins)
    plt.hist(gdiffs[relevgsw],bins=binsgsw,histtype='step',label='GSW Galaxies')
    plt.hist(s82diffs[relevs82],bins=bins82,label='S82X Galaxies')
    plt.ylabel('Counts')
    plt.xlabel('Distance')
    plt.title('Distances for SDSS'+str(sdssid),fontsize=20)
    plt.xlim([0,5])
    '''
    if nearestx:
        plt.axvline(x=nearestx,ls=':',label='Nearest X-Ray Detection: Distance = '+str(round(nearestx,2)))
    '''
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()
'''
plotgdiffhist(s82gdiff.dists_filt[nonagn[0]],s82gdiff.alls82dists[nonagn[0]],s82ids[nonagn[0]],s82gdiff.nrx[nonagn[0]], nearestx=s82gdiff.mindx[nonagn[0]])
plotgdiffhist(s82gdiff.dists_filt[nonagn[1]],s82gdiff.alls82dists[nonagn[1]],s82ids[nonagn[1]],s82gdiff.nrx[nonagn[1]], nearestx=s82gdiff.mindx[nonagn[1]])
plotgdiffhist(s82gdiff.dists_filt[nonagn[2]],s82gdiff.alls82dists[nonagn[2]],s82ids[nonagn[2]],s82gdiff.nrx[nonagn[2]], nearestx=s82gdiff.mindx[nonagn[2]])
plotgdiffhist(s82gdiff.dists_filt[nonagn[3]],s82gdiff.alls82dists[nonagn[3]],s82ids[nonagn[3]],s82gdiff.nrx[nonagn[3]], nearestx=s82gdiff.mindx[nonagn[3]])
plotgdiffhist(s82gdiff.dists_filt[nonagn[4]],s82gdiff.alls82dists[nonagn[4]],s82ids[nonagn[4]],s82gdiff.nrx[nonagn[4]], nearestx=s82gdiff.mindx[nonagn[4]])
'''    
def getallgdiffhistsbpthii():
    for i in range(len(nonagn)):
        plotgdiffhist(s82gdiff.dists_filt[nonagn[i]], s82gdiff.alls82dists[nonagn[i]], s82ids[nonagn[i]], s82gdiff.nrx[nonagn[i]], nearestx=s82gdiff.mindx[nonagn[i]])
        plt.savefig('plots/s82x/png/distmet/bpt_hii_dists/bpthII'+str(i)+'.png',bbox_inches='tight')
        plt.savefig('plots/s82x/eps/distmet/bpt_hii_dists/bpthII'+str(i)+'.eps',bbox_inches='tight')
        plt.savefig('plots/s82x/pdf/distmet/bpt_hii_dists/bpthII'+str(i)+'.pdf',bbox_inches='tight')
        
        plt.close()
#getallgdiffhistsbpthii()
def getallgdiffhistsbptagn():
    for i in range(len(agn)):
        plotgdiffhist(s82gdiff.dists_filt[agn[i]],s82gdiff.alls82dists[agn[i]],s82ids[agn[i]],s82gdiff.nrx[agn[i]], nearestx=s82gdiff.mindx[agn[i]])
        plt.savefig('plots/s82x/png/distmet/bpt_agn_dists/agn'+str(i)+'.png',bbox_inches='tight')
        plt.savefig('plots/s82x/eps/distmet/bpt_agn_dists/agn'+str(i)+'.eps',bbox_inches='tight')
        plt.savefig('plots/s82x/pdf/distmet/bpt_agn_dists/agn'+str(i)+'.pdf',bbox_inches='tight')
        plt.close()        
#getallgdiffhistsbptagn()
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
def plot2dhist(x,y,nx,ny):
    hist,xedges, yedges = np.histogram2d(x,y,bins = (nx,ny))
    extent= [np.min(xedges),np.max(xedges),np.min(yedges),np.max(yedges)]
    plt.imshow((hist.transpose())**0.3, cmap='gray_r',extent=extent,origin='lower',alpha=0.9)
    
def plotbpt(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=False,fname='',title=None,alph=0.1,ccode=[],ccodegsw=[],cont=None,ccodename='X-ray Fraction',levs=[0,0.2]):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    fig= plt.figure()
    ax = fig.add_subplot(111)
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    

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
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT AGN)')
        plt.clim(mn, mx)
        cbar = plt.colorbar()

        cbar.set_label(ccodename,fontsize=20)
        cbar.ax.tick_params(labelsize=20)
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT AGN)',s=20)
    if cont:
        #extent = [np.min(cont.rangex),np.max(cont.rangex),np.min(cont.rangey), np.max(cont.rangey)]
        #plt.imshow(cont.grid, extent=extent,origin='lower')
        
        #locs = np.vstack([posx,posy]).transpose()

        CS = plt.contour(cont.meshx, cont.meshy, cont.grid,cmap='Blues',levels=levs,vmin=contaminations.min(), vmax= contaminations.max())
        norm= matplotlib.colors.Normalize(vmin=contaminations.min(), vmax=contaminations.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap = CS.cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm)

        cbar.set_label(ccodename,fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        plt.text(-0.245,-0.40,'0.01',fontsize=15)
        plt.text(-0.129,-0.27,'0.02',fontsize=15)
        plt.text(0.0167,-0.14,'0.04',fontsize=15)
        plt.text(0.2014,-0.0088,'0.08',fontsize=15)
        plt.text(0.325,0.189,'0.12',fontsize=15)
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
        '''
    plt.text(0.6,0.75,'AGN',fontsize=15)
    #plt.text(-.1,-1.5,'Comp')
    plt.text(-1.15,-0.5,'HII',fontsize=15)
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([NII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([-1.2,1.2])
    plt.xlim([-2,1])
    
    plt.legend(fontsize=10)
    if title:
        plt.title(title,fontsize=30)
    ax.set(adjustable='box-forced', aspect='equal')    
    
    plt.tight_layout()
    if save:
        fig.savefig('plots/s82x/png/diagnostic/NII_OIII_scat'+fname+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/s82x/pdf/diagnostic/NII_OIII_scat'+fname+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/s82x/eps/diagnostic/NII_OIII_scat'+fname+'.eps',dpi=150,format='eps',bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi))
#blues = cm.Blues(np.linspace(np.min(contaminations),np.max(contaminations),len(s82eldiag.niiha)))

'''
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb,
         nonagn, agn,gsweldiag.niiha,gsweldiag.oiiihb,save=False)
plotbpt(xmm3eldiagmed.niiha,xmm3eldiagmed.oiiihb,
         nonagn_3xmm, agn_3xmm,gsweldiag.niiha,gsweldiag.oiiihb,save=False)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb,
         nonagn, agn,gsweldiag.niiha[agn_gsw],gsweldiag.oiiihb[agn_gsw],fname='bptagn_bg',save=True)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb,
         nonagn, agn,gsweldiag.niiha[nonagn_gsw],gsweldiag.oiiihb[nonagn_gsw],fn0ame='bpthii_bg',save=True )
#color coding for 
#contamination
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, nonagn,agn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=True, fname='xrfracnew',alph=0.1,ccode=contaminations) 
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [], [],
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=True, fname = 'xrfrac_cont10x10new', alph=0.1, cont=s82gdiff, levs = [0.01 ,0.02, 0.04, 0.08, 0.12])
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [], [],
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=True, fname='xrfrac_cont16x16',alph=0.1,cont=s82gdiff)
#82 contam
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, nonagn, agn,
        gsweldiagcov.niiha, gsweldiagcov.oiiihb,
        save=False, fname='xrfrac',alph=1,ccode=contaminations,cont=s82gdiff)
#z
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, nonagn, agn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=True, fname='zccode',alph=0.1,ccode=s82eldiag.z,ccodename='z')
#mass
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, nonagn, agn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=True, fname='massccode',alph=0.1,ccode=s82eldiag.mass,ccodename=r'log(M$_{*}$)')
#sfr
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, nonagn, agn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=False, fname='disthresh_hii0',alph=0.1,ccode=fullxray.sfr[make_m1][halp_filt_s82])
#sSFR
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, nonagn, agn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=False, fname='disthresh_hii0',alph=0.1,ccode=fullxray.sfr_mass[make_m1][halp_filt_s82])
#full xray
plotbpt(s82eldiag.niiha[fullxray.valid],s82eldiag.oiiihb[fullxray.valid], fullxray.validnoagn, fullxray.validagn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=False, fname='disthresh_hii0',alph=0.1,ccode=fullxray.lum[make_m1][halp_filt_s82][fullxray.valid])
#full xray-mass
plotbpt(s82eldiag.niiha[fullxray.valid],s82eldiag.oiiihb[fullxray.valid], nonagn, agn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=False, fname='disthresh_hii0',alph=0.1,ccode=fullxray.lum_mass[make_m1][halp_filt_s82][fullxray.valid])
#soft xray
plotbpt(s82eldiag.niiha[softxray.valid],s82eldiag.oiiihb[softxray.valid], nonagn, agn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=False, fname='disthresh_hii0',alph=0.1,ccode=softxray.lum[make_m1][halp_filt_s82][softxray.valid])
#soft xray - mass
plotbpt(s82eldiag.niiha[softxray.valid],s82eldiag.oiiihb[softxray.valid], nonagn, agn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=False, fname='disthresh_hii0',alph=0.1,ccode=softxray.lum_mass[make_m1][halp_filt_s82][softxray.valid])
#hard xray
plotbpt(s82eldiag.niiha[hardxray.valid],s82eldiag.oiiihb[hardxray.valid], nonagn, agn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=False, fname='disthresh_hii0',alph=0.1,ccode=hardxray.lum_val[make_m1][halp_filt_s82][hardxray.valid])
#hard xray-mass
plotbpt(s82eldiag.niiha[hardxray.valid],s82eldiag.oiiihb[hardxray.valid], nonagn, agn,
        gsweldiag.niiha, gsweldiag.oiiihb,
        save=False, fname='disthresh_hii0',alph=0.1,ccode=hardxray.lum_mass_val[make_m1][halp_filt_s82][hardxray.valid])

#below is for doing distance threshold

plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [nonagn[0]], [],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[nonagn[0]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[nonagn[0]]],
        save=False, fname='_dist2bpthii0',alph=1,ccodegsw =gdiffs82[nonagn[0]][passinggals[nonagn[0]]] )
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [nonagn[0]], [],
        gsweldiag.niiha[stripe82[covered_gsw]], gsweldiag.oiiihb[stripe82[covered_gsw]],
        save=False, fname='_dist2bpthii0',alph=1,ccodegsw =gdiffs82[nonagn[0]] )

plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [nonagn[1]], [],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[nonagn[1]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[nonagn[1]]],
        save=True, fname='_dist2bpthii1',alph=1)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [nonagn[2]], [],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[nonagn[2]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[nonagn[2]]],
        save=True, fname='_dist2bpthii2',alph=1)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [nonagn[3]], [],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[nonagn[3]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[nonagn[3]]],
        save=True, fname='_dist2bpthii3',alph=1)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [nonagn[4]], [],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[nonagn[4]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[nonagn[4]]],
        save=True, fname='_dist2bpthii4',alph=1)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [nonagn[5]], [],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[nonagn[5]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[nonagn[5]]],
        save=True, fname='_dist2bpthii5',alph=1)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [nonagn[6]], [],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[nonagn[6]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[nonagn[6]]],
        save=True, fname='_dist2bpthii6',alph=1)

plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [nonagn[7]], [],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[nonagn[7]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[nonagn[7]]],
        save=True, fname='_dist2bpthii7',alph=1)

plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [], [agn[0]],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[agn[0]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[agn[0]]],
        save=True, fname='_dist2bptagn0',alph=1)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [], [agn[10]],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[agn[10]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[agn[10]]],
        save=True, fname='_dist2bptagn10',alph=1)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [], [agn[20]],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[agn[20]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[agn[20]]],
        save=True, fname='_dist2bptagn20',alph=1)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [], [agn[30]],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[agn[30]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[agn[30]]],
        save=True, fname='_dist2bptagn30',alph=1)
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [], [agn[37]],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[agn[37]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[agn[37]]],
        save=True, fname='_dist2bptagn37',alph=1)
n=15
plotbpt(s82eldiag.niiha,s82eldiag.oiiihb, [], [agn[n]],
        gsweldiag.niiha[stripe82[covered_gsw]][passinggals[agn[n]]], gsweldiag.oiiihb[stripe82[covered_gsw]][passinggals[agn[n]]],
        save=True, fname='_dist2bptagn'+str(n),alph=1)


'''

def mass_z(xvals,yvals,nonagnfilt, agnfilt, bgx,bgy,save=True,fname='',title=None,alph=0.1,ccode=[],ccodegsw=[]):
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
    
        plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagn], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agn], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt)) !=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT AGN)',s=20)

    plt.ylabel(r'log(M$_{\rm *}$)',fontsize=20)
    plt.xlabel(r'z',fontsize=20)
    plt.ylim([8, 12])
    plt.xlim([0,0.3])
    plt.legend(fontsize=10)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/s82x/png/diagnostic/mass_z'+fname+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/s82x/pdf/diagnostic/mass_z'+fname+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/s82x/eps/diagnostic/mass_z'+fname+'.eps',dpi=150,format='eps',bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi)
'''
#general
mass_z(s82eldiag.z,s82eldiag.mass, nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=True, alph=0.1)

n=37 #from n= 0 to 39?
mass_z(s82eldiag.z,s82eldiag.mass, [], [agn[n]],
        gsweldiag.z[stripe82[covered_gsw]][passinggals[agn[n]]], gsweldiag.mass[stripe82[covered_gsw]][passinggals[agn[n]]],
        save=True, fname='_dist2bptagn'+str(n),alph=1)
n=7 #from n=0 to 7
mass_z(s82eldiag.z,s82eldiag.mass, [nonagn[n]], [],
        gsweldiag.z[stripe82[covered_gsw]][passinggals[nonagn[n]]], gsweldiag.mass[stripe82[covered_gsw]][passinggals[nonagn[n]]],
        save=True, fname='_dist2bpthii'+str(n),alph=1)


#contaminations
mass_z(s82eldiag.z,s82eldiag.mass, nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=True, fname='xrfrac',alph=0.1,ccode=contaminations)
#nii/ha
mass_z(s82eldiag.z,s82eldiag.mass, nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=False, fname='disthresh_agn'+str(n),alph=0.1,ccode=s82eldiag.niiha)
#oiii/hb
mass_z(s82eldiag.z,s82eldiag.mass, nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=False, fname='disthresh_agn'+str(n),alph=0.1,ccode=s82eldiag.oiiihb)
#sfr
mass_z(s82eldiag.z,s82eldiag.mass, nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=False, fname='disthresh_agn'+str(n),alph=0.1,ccode=fullxray.sfr_val[make_m1][halp_filt_s82])
#sSFR
mass_z(s82eldiag.z,s82eldiag.mass, nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=False, fname='disthresh_agn'+str(n),alph=0.1,ccode=fullxray.sfr_mass_val[make_m1][halp_filt_s82])
#full xray
mass_z(s82eldiag.z[fullxray.valid],s82eldiag.mass[fullxray.valid], nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=False, fname='disthresh_agn'+str(n),alph=0.1,ccode=fullxray.lum_val[make_m1][halp_filt_s82][fullxray.valid])
#full xray-mass
mass_z(s82eldiag.z[fullxray.valid],s82eldiag.mass[fullxray.valid], nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=False, fname='disthresh_agn'+str(n),alph=0.1,ccode=fullxray.lum_mass_val[make_m1][halp_filt_s82][fullxray.valid])
#soft xray
mass_z(s82eldiag.z[softxray.valid],s82eldiag.mass[softxray.valid], nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=False, fname='disthresh_agn'+str(n),alph=0.1,ccode=softxray.lum_val[make_m1][halp_filt_s82][softxray.valid])
#soft xray-mass
mass_z(s82eldiag.z[softxray.valid],s82eldiag.mass[softxray.valid], nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=False, fname='disthresh_agn'+str(n),alph=0.1,ccode=softxray.lum_mass_val[make_m1][halp_filt_s82][softxray.valid])
#hard xray
mass_z(s82eldiag.z[hardxray.valid],s82eldiag.mass[hardxray.valid], nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=False, fname='disthresh_agn'+str(n),alph=0.1,ccode=hardxray.lum_val[make_m1][halp_filt_s82][hardxray.valid])
#hard xray-mass
mass_z(s82eldiag.z[hardxray.valid],s82eldiag.mass[hardxray.valid], nonagn, agn,
        gsweldiag.z, gsweldiag.mass,
        save=False, fname='disthresh_agn'+str(n),alph=0.1,ccode=hardxray.lum_mass_val[make_m1][halp_filt_s82][hardxray.valid])

'''
def massfrac_z(xvals,yvals,nonagnfilt, agnfilt, bgx,bgy,save=True,fname='',title=None,alph=0.1,ccode=[],ccodegsw=[]):
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
        plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')    
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()

        sc =plt.scatter(xvals[nonagn], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agn], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt)) !=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT AGN)',s=20)

    plt.ylabel(r'Mass Fraction',fontsize=20)
    plt.xlabel(r'z',fontsize=20)
    plt.ylim([0, 0.8])
    plt.xlim([0,0.3])
    plt.legend(fontsize=10)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/s82x/png/diagnostic/massfrac_z'+fname+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/s82x/pdf/diagnostic/massfrac_z'+fname+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/s82x/eps/diagnostic/massfrac_z'+fname+'.eps',dpi=150,format='eps',bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    #figsize=(960/mydpi,540/mydpi))
'''
massfrac_z(s82eldiag.z,s82eldiag.massfrac, nonagn, agn,
        gsweldiag.z, gsweldiag.massfrac,
        save=True, alph=0.1)
#contaminations
massfrac_z(s82eldiag.z,s82eldiag.massfrac, nonagn, agn,
        gsweldiag.z, gsweldiag.massfrac,
        save=True, fname='xrfrac',alph=0.1,ccode=contaminations)
massfrac_z(s82eldiag.z,s82eldiag.massfrac, nonagn, agn,
        gsweldiag.z, gsweldiag.massfrac,
        save=False, fname='xrfrac',alph=0.1,ccodegsw=gsweldiag.mass)
'''
def plotssfrm(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=True,fname='',
              title=None,alph=0.1,ccode=[],ccodegsw=[],cont=None,weakha_x = [], weakha_y=[],
              weakha_xs82=[], weakha_ys82=[]):
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
        plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    if len(weakha_x) !=0 and len(weakha_y) !=0:
        plt.scatter(weakha_x,weakha_y,color='indianred',marker='.',alpha=alph, label=r'SDSS DR7 (Weak H$\alpha$)')
    if len(weakha_xs82) !=0 and len(weakha_ys82) !=0:
        plt.scatter(weakha_xs82,weakha_ys82,color='k',marker='x' , label=r'X-ray AGN (Weak H$\alpha$)')    
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    if len(ccode) !=0: #for the contamination
        mn, mx = ccode.min(), ccode.max()
        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT AGN)',s=20)
    if cont:
        #extent = [np.min(cont.rangex),np.max(cont.rangex),np.min(cont.rangey), np.max(cont.rangey)]
        #plt.imshow(cont.grid, extent=extent,origin='lower')
        levs =[0.015,0.045,0.075,0.12,0.16]#np.log10((np.logspace(mn+0.014,mx-0.025,6)))
        print(levs)
        posx = np.array([-0.14,0.03,0.24,0.31,0.32])
        posy = np.array([-0.6,-0.35,-0.14,0.16,0.31])
        locs = np.vstack([posx,posy]).transpose()
        CS = plt.contour(cont.meshx, cont.meshy, cont.grid,colors='k',levels=levs)
        plt.text(-0.27,-0.56,'0.015',fontsize=15)
        plt.text(-0.033,-0.349,'0.045',fontsize=15)
        plt.text(0.18,-0.12,'0.075',fontsize=15)
        plt.text(0.325,0.11,'0.12',fontsize=15)
        plt.text(0.325,0.371,'0.16',fontsize=15)
    plt.ylabel(r'sSFR',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}})$')
    plt.xlim([8,12])
    plt.ylim([-14,-8,])
    plt.legend(fontsize=10)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/s82x/png/diagnostic/ssfr_mass'+fname+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/s82x/pdf/diagnostic/ssfr_mass'+fname+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/s82x/eps/diagnostic/ssfr_mass'+fname+'.eps',dpi=150,format='eps',bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
'''
plotssfrm(fullxray.mass_filt,fullxray.sfr_mass_filt,nonagn,agn,m1Cat_GSW.allmass[make_allm1][halp_filt][valid_bpt],
        m1Cat_GSW.allsfr[make_allm1][halp_filt][valid_bpt]-m1Cat_GSW.allmass[make_allm1][halp_filt][valid_bpt],
        save=True,alph=0.1)
plotssfrm(fullxray.mass_filt,fullxray.sfr_mass_filt,nonagn,agn,m1Cat_GSW.allmass[make_allm1][halp_filt][valid_bpt],
        m1Cat_GSW.allsfr[make_allm1][halp_filt][valid_bpt]-m1Cat_GSW.allmass[make_allm1][halp_filt][valid_bpt],
        save=True, fname='xrfrac',alph=0.1,ccode=contaminations)
plotssfrm(fullxray.mass_filt,fullxray.sfr_mass_filt,nonagn,agn,gsweldiag.mass,
        gsweldiag.sfr-gsweldiag.mass, save=True,fname='weakha_bg_and_agn_inc',alph=0.1,weakha_x =gsweldiag_no.mass,
        weakha_y = gsweldiag_no.sfr-gsweldiag_no.mass, weakha_xs82 = s82eldiag_no.mass, 
        weakha_ys82 = s82eldiag_no.sfr-s82eldiag_no.mass)
plotssfrm(fullxray.mass_filt,fullxray.sfr_mass_filt,nonagn,agn,gsweldiag.mass[nonagn_gsw],
        gsweldiag.sfr[nonagn_gsw]-gsweldiag.mass[nonagn_gsw], save=True,fname='bpthii_bg',alph=0.1)
plotssfrm(fullxray.mass_filt,fullxray.sfr_mass_filt,nonagn,agn,gsweldiag.mass[agn_gsw],
        gsweldiag.sfr[agn_gsw]-gsweldiag.mass[agn_gsw], save=True,fname='bptagn_bg',alph=0.1)
'''
def plotoiiimass(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, fname='', 
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[]):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
    
        plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
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

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT AGN)',s=20)
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
        
    plt.ylabel(r'log(L$_\mathrm{[OIII]}$)',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([37,43])
    plt.xlim([8,13])
    
    plt.legend(fontsize=10)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/s82x/png/diagnostic/OIIILum_Mass_scat'+fname+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/s82x/pdf/diagnostic/OIIILum_Mass_scat'+fname+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/s82x/eps/diagnostic/OIIILum_Mass_scat'+fname+'.eps',dpi=150,format='eps',bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiimass(s82eldiag.mass,np.log10(s82eldiag.oiiilum/1e17),nonagn,agn,
             gsweldiag.mass, np.log10(gsweldiag.oiiilum/1e17),save=True,alph=0.1)
plotoiiimass(s82eldiag.mass,np.log10(s82eldiag.oiiilum/1e17),nonagn,agn,
             gsweldiag.mass[agn_gsw], np.log10(gsweldiag.oiiilum[agn_gsw]/1e17),save=True,alph=0.1,fname='_bptagn_bg'),#weakha_xs82=s82eldiag_oiii.mass, weakha_ys82=np.log10(s82eldiag_oiii.oiiilum/1e17))
plotoiiimass(s82eldiag.mass,np.log10(s82eldiag.oiiilum/1e17),nonagn,agn,
             gsweldiag.mass[nonagn_gsw], np.log10(gsweldiag.oiiilum[nonagn_gsw]/1e17),
             save=True,alph=0.1,fname='_bpthii_bg')#,weakha_xs82=s82eldiag_oiii.mass, weakha_ys82=np.log10(s82eldiag_oiii.oiiilum/1e17))
plotoiiimass(s82eldiag.mass,np.log10(s82eldiag.oiiilum/1e17),nonagn,agn,gsweldiag.mass, np.log10(gsweldiag.oiiilum/1e17),
             save=True,alph=0.1,fname='_weakha_bg_inc',weakha_x= gsweldiag_oiii.mass,weakha_y = np.log10(gsweldiag_oiii.oiiilum/1e17) ,
             weakha_xs82=s82eldiag_oiii.mass, weakha_ys82=np.log10(s82eldiag_oiii.oiiilum/1e17))

'''        
def plotoiiidispmass(xvals, yvals, nonagnfilt, agnfilt, bgx, bgy, save=True, fname='', 
                 title=None, alph=0.1, ccode=[], ccodegsw=[], cont=None, weakha_x = [], weakha_y = [],
                 weakha_xs82=[],weakha_ys82=[]):
    '''
    for doing bpt diagram with sdss galaxies scatter plotted
    '''
    if save:
        fig = plt.figure()
    #counts1,xbins1,ybins1= np.histogram2d(np.log10(bgx), np.log10(bgy), bins=(50,50))#,cmap='gray',cmin=2)#marker='.',alpha=0.15,label='SDSS DR7')
    if len(ccodegsw)!=0:
        val = np.where(np.isfinite(ccodegsw))[0]
        sc = plt.scatter(bgx[val], bgy[val],c=ccodegsw[val],cmap='Blues_r',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
        plt.colorbar(sc)
    else:
    
        plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
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

        sc =plt.scatter(xvals[nonagnfilt], yvals[nonagnfilt], c=ccode[nonagnfilt], s=20, cmap='Blues',marker='^',label='X-Ray AGN (BPT HII)')
        plt.clim(mn,mx)
        sc =plt.scatter(xvals[agnfilt], yvals[agnfilt], c=ccode[agnfilt], s=20, cmap='Blues',marker='o',label='X-Ray AGN (BPT AGN)')
        plt.clim(mn, mx)
        plt.colorbar().set_label('Xray Fraction')
    else:
        if len(list(nonagnfilt))!=0:
            plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT HII)',s=20)
        if len(list(agnfilt)) !=0:
            plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT AGN)',s=20)
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
        
    plt.ylabel(r'log(L$_\mathrm{[OIII]})/\sigma_{*}^4$',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}}$)',fontsize=20)
    plt.ylim([28,36])
    plt.xlim([8,13])
    
    plt.legend(fontsize=10)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/s82x/png/diagnostic/OIIILumdisp_Mass_scat'+fname+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/s82x/pdf/diagnostic/OIIILumdisp_Mass_scat'+fname+'.pdf',dpi=250,bbox_inches='tight')
        fig.set_rasterized(True)
        fig.savefig('plots/s82x/eps/diagnostic/OIIILumdisp_Mass_scat'+fname+'.eps',dpi=150,format='eps',bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
'''
plotoiiidispmass(s82eldiag.mass,np.log10(s82eldiag.oiiilum/1e17/s82eldiag.vdisp**4),nonagn,agn,
             gsweldiag.mass, np.log10(gsweldiag.oiiilum/1e17/gsweldiag.vdisp**4),save=True,alph=0.1)
plotoiiidispmass(s82eldiag.mass,np.log10(s82eldiag.oiiilum/1e17/s82eldiag.vdisp**4),nonagn,agn,
             gsweldiag.mass[agn_gsw], np.log10(gsweldiag.oiiilum[agn_gsw]/1e17/gsweldiag.vdisp[agn_gsw]**4),
             save=True,alph=0.1,fname='_bptagn_bg')
plotoiiidispmass(s82eldiag.mass,np.log10(s82eldiag.oiiilum/1e17/s82eldiag.vdisp**4),nonagn,agn,
             gsweldiag.mass[nonagn_gsw], np.log10(gsweldiag.oiiilum[nonagn_gsw]/1e17/gsweldiag.vdisp[nonagn_gsw]**4),
             save=True,alph=0.1,fname='_bpthii_bg')
plotoiiidispmass(s82eldiag.mass,np.log10(s82eldiag.oiiilum/1e17/s82eldiag.vdisp**4),nonagn,agn,gsweldiag.mass, np.log10(gsweldiag.oiiilum/1e17/gsweldiag.vdisp**4),
             save=True,alph=0.1,fname='_weakha_bg_inc',weakha_x= gsweldiag_oiii.mass,weakha_y = np.log10(gsweldiag_oiii.oiiilum/1e17/gsweldiag_oiii.vdisp**4) ,
             weakha_xs82=s82eldiag_oiii.mass, weakha_ys82=np.log10(s82eldiag_oiii.oiiilum/1e17/s82eldiag_oiii.vdisp**4))
plotoiiidispmass(fullxray_no.mass_filt,fullxray_no.sfr_mass_filt,nonagn_no,agn_no,m1Cat_GSW.allmass[make_allm1][not_halp_filt][valid_bpt_no],
        m1Cat_GSW.allsfr[make_allm1][not_halp_filt][valid_bpt_no]-m1Cat_GSW.allmass[make_allm1][not_halp_filt][valid_bpt_no],
        save=False, fname='xrfrac',alph=0.1)
'''        
def plotmasscomp(mass_sdss,massgsw):
    plt.scatter(mass_sdss, massgsw)
    plt.xlabel('Mass SDSS')
    plt.ylabel('Mass GSW')
    plt.tight_layout()
    plt.show()
def plot_ssfr_mass(xraysfr, label='',save =False):
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')   
    plt.scatter(m1Cat_GSW.allmass[make_allm1][halp_filt][valid_bpt], 
                m1Cat_GSW.allsfr[make_allm1][halp_filt][valid_bpt]-m1Cat_GSW.allmass[make_allm1][halp_filt][valid_bpt]
                ,marker='o', edgecolors='none', color='gray', alpha=0.1,label='GSWLC')
    plt.xlim([8,12])
    plt.ylim([-14,-8,])
    plt.scatter(xraysfr.mass_filtnoagn,xraysfr.sfr_mass_val_filtnoagn,marker='^', color='b',label='BPT HII')
    plt.scatter(xraysfr.mass_filtagn,xraysfr.sfr_mass_val_filtagn,marker='o',facecolors='none',color='k',label='BPT AGN')
    plt.ylabel(r'sSFR',fontsize=20)
    plt.xlabel(r'log(M$_{\mathrm{*}})$')
    plt.legend(fontsize=10)    
    plt.tight_layout()
    if save:
        plt.savefig("plots/s82x/png/diagnostic/ssfr_mass"+label+".png",dpi=250,bbox_inches='tight')
        plt.savefig("plots/s82x/pdf/diagnostic/ssfr_mass"+label+".pdf",dpi=250,format='pdf',bbox_inches='tight')
        plt.savefig("plots/s82x/eps/diagnostic/ssfr_mass"+label+".eps",dpi=250,format='eps',bbox_inches='tight')    
#plot_ssfr_mass(fullxray)

def plotbpt_sii(xvals, yvals, nonagnfilt, agnfilt, bgx,bgy,save=False,fname='',
                title=None,alph=0.1,ccode=[],ccodegsw=[]):
    fig = plt.figure()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')
    plt.tick_params(direction='in',axis='both',which='both')
    
    plt.plot(np.log10(xline2_agn),np.log10(yline2_agn),'k--')#,label='AGN Line')
    plt.plot(np.log10(xline2_linersy2),np.log10(yline2_linersy2),c='k',ls='-.')#,label='LINER, Seyfert 2')
    plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT HII)',s=20)
    plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT AGN)',s=20)
    plt.text(.4,-.5,'LINER')
    plt.text(-0.8,1.0,'Seyfert')
    plt.text(-0.9,-1,'HII')

    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([SII]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([-1.2,1.2])
    plt.xlim([-1.2,0.75])
    plt.legend(fontsize=10)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        
        fig.savefig('plots/s82x/png/diagnostic/SII_OIII_scat'+fname+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/s82x/pdf/diagnostic/SII_OIII_scat'+fname+'.pdf',dpi=250,bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
'''
plotbpt_sii(s82eldiag.siiha,s82eldiag.oiiihb,
        nonagn, agn,gsweldiag.siiha,gsweldiag.oiiihb,save=False)
plotbpt_sii(s82eldiag.siiha,s82eldiag.oiiihb,
        nonagn, agn,gsweldiag.siiha[agn_gsw],gsweldiag.oiiihb[agn_gsw],save=True,fname='bptagn_bg')
plotbpt_sii(s82eldiag.siiha,s82eldiag.oiiihb,
        nonagn, agn,gsweldiag.siiha[nonagn_gsw],gsweldiag.oiiihb[nonagn_gsw],save=True,fname='bpthii_bg')
'''
def plotbpt_oi(xvals,yvals,nonagnfilt, agnfilt, bgx,bgy,save=False, fname='',title=None,alph=0.1,ccode=[],ccodegsw=[]):
    fig = plt.figure()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)

    plt.scatter(bgx, bgy,color='gray',marker='.',alpha=alph,label='SDSS DR7',edgecolors='none')

    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline3_linersy2),np.log10(yline3_linersy2),c='k',ls='-.')
    plt.plot(np.log10(xline3_agn), np.log10(yline3_agn),'k--')
    plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT HII)',s=20)
    plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT AGN)',s=20)
    plt.text(-.4,-0.5,'LINER')
    plt.text(-1.5,1.0,'Seyfert')
    plt.text(-2,-0.95,'HII')
    plt.ylabel(r'log([OIII]/H$\rm \beta$)',fontsize=20)
    plt.xlabel(r'log([OI]/H$\rm \alpha$)',fontsize=20)
    plt.ylim([-1.2,1.2])
    plt.xlim([-2.2,0])
    plt.legend(fontsize=10)
    if title:
        plt.title(title,fontsize=30)
    plt.tight_layout()
    if save:
        fig.savefig('plots/s82x/png/diagnostic/OI_OIII_scat'+fname+'.png',dpi=250,bbox_inches='tight')
        fig.savefig('plots/s82x/pdf/diagnostic/OI_OIII_scat'+fname+'.pdf',dpi=250,bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
'''   
plotbpt_oi(s82eldiag.oiha,s82eldiag.oiiihb, nonagn, agn, gsweldiag.oiha,gsweldiag.oiiihb,save=True)
plotbpt_oi(s82eldiag.oiha,s82eldiag.oiiihb,
        nonagn, agn,gsweldiag.oiha[agn_gsw],gsweldiag.oiiihb[agn_gsw],save=True,fname='bptagn_bg')
plotbpt_oi(s82eldiag.oiha,s82eldiag.oiiihb,
        nonagn, agn,gsweldiag.oiha[nonagn_gsw],gsweldiag.oiiihb[nonagn_gsw],save=True,fname='bpthii_bg')
'''  
def plot_mex(xvals, yvals,nonagnfilt,agnfilt, bgx,bgy,save=True, fname='',title=None):
    #plt.scatter(np.log10(xvals3_bpt[valid_bpt][::50]), np.log10(yvals_bpt[valid_bpt][::50]),color='g',marker='.',alpha=0.15,label='SDSS DR7')
    plt.plot(xline_mex,ylineup_mex,'k--')
    plt.plot(xline_mex,ylinedown_mex,'k-.')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')

    plt.scatter(bgx,np.log10(bgy),color='gray',marker='.',alpha=0.1,edgecolors='none',label='SDSS DR7')
    plt.scatter(xvals[nonagnfilt],np.log10(yvals[nonagnfilt]),
                 marker='^',color='b', label='X-Ray AGN (BPT HII)')
    plt.scatter(xvals[agnfilt],np.log10(yvals[agnfilt]),
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT AGN)')
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
        plt.savefig('plots/s82x/png/diagnostic/Mex_OIII'+fname+'.png',dpi=250,bbox_inches='tight')
        plt.savefig('plots/s82x/pdf/diagnostic/Mex_OIII'+fname+'.pdf',dpi=250,bbox_inches='tight')
        plt.close()
    else:
        plt.show()
'''
plot_mex(s82eldiag.mx, s82eldiag.oiiihb,nonagn, agn, gsweldiag.mx, gsweldiag.oiiihb)
'''
def get_mass_bins(mass,start=8,stop=12,step=1):
    massbins = np.arange(start,stop+step,step)
    inds= []
    for i in range(len(massbins)-1):
        bin_inds = np.where((mass >massbins[i]) &(mass <massbins[i+1]) )[0]
        inds.append(bin_inds)
    return inds
def get_z_bins(z,step=.05):
    zbins = np.arange(0,0.3+step,step)
    inds = []
    for i in range(len(zbins)-1):
        bin_inds = np.where((z>zbins[i]) & (z <zbins[i+1]))[0]
        inds.append(bin_inds)
    return inds
'''
mass_bins_sdss  =get_mass_bins(all_sdss_avgmasses[spec_inds_allm1][valid_bpt2],step=0.5)
mass_bins_s82  =get_mass_bins(mass_exc_stripe[halp_filt_s82],step=0.5)
z_bins_sdss = get_z_bins(all_sdss_spec_z[spec_inds_allm1][valid_bpt2],step=0.05)
z_bins_s82 = get_z_bins(m1_z[halp_filt_s82],step=0.05)

massfrac_bins_sdss  =get_mass_bins(all_sdss_massfrac[spec_inds_allm1][valid_bpt2],start =0, stop = 1,step=0.1)
massfrac_bins_s82  =get_mass_bins(s82frac,step=0.1,stop=1, start=0)
'''
def use_mass_bins(mass_bins_sdss,mass_bins_s82,start=8, step=1,frac=''):
    for i in range(len(mass_bins_sdss)):
        nonagn, agn = get_bpt1_groups(stripe_82x_1[halp_filt_s82][mass_bins_s82[i]],
                                      stripe_82y[halp_filt_s82][mass_bins_s82[i]])

        fnam = 'massbin'+frac+str(round(i*step+start,2))+'-'+str(round(i*step+start+step,2))
        titl = str(round(i*step+start,2))+r'$<$ M $<$'+str(round(i*step+start+step,2))        
        plotbpt(stripe_82x_1[halp_filt_s82][mass_bins_s82[i]],
                    stripe_82y[halp_filt_s82][mass_bins_s82[i]],
                    nonagn, agn,gsw_xvals1_bpt[valid_bpt2][mass_bins_sdss[i]],
                    gsw_yvals_bpt[valid_bpt2][mass_bins_sdss[i]],
                    fname=fnam,title=titl)
        plotbpt_sii(stripe_82x_2[halp_filt_s82][mass_bins_s82[i]],
                    stripe_82y[halp_filt_s82][mass_bins_s82[i]],
                    nonagn, agn,gsw_xvals2_bpt[valid_bpt2][mass_bins_sdss[i]],
                    gsw_yvals_bpt[valid_bpt2][mass_bins_sdss[i]],
                    fname=fnam,title=titl)
        plotbpt_oi(stripe_82x_3[halp_filt_s82][mass_bins_s82[i]],
                    stripe_82y[halp_filt_s82][mass_bins_s82[i]],
                    nonagn, agn,gsw_xvals3_bpt[valid_bpt2][mass_bins_sdss[i]],
                    gsw_yvals_bpt[valid_bpt2][mass_bins_sdss[i]],
                    fname=fnam,title=titl)
        if frac!='':
            plot_mex(mass_exc_stripe[halp_filt_s82][mass_bins_s82[i]],
                    stripe_82y[halp_filt_s82][mass_bins_s82[i]],nonagn, agn, 
                    all_sdss_avgmasses[spec_inds_allm1][valid_bpt][mass_bins_sdss[i]],
                    gsw_yvals_bpt[valid_bpt][mass_bins_sdss[i]],
                    fname=fnam,title=titl)
def use_z_bins(z_bins_sdss,z_bins_s82,start=0, step=.03):
    for i in range(len(z_bins_sdss)):
        nonagn, agn = get_bpt1_groups(stripe_82x_1[halp_filt_s82][z_bins_s82[i]],
                                      stripe_82y[halp_filt_s82][z_bins_s82[i]])

        fnam = 'zbin'+str(round(i*step+start,2))+'-'+str(round(i*step+start+step,2))
        titl = str(round(i*step+start,2))+r'$<$ z $<$'+str(round(i*step+start+step,2))
        
        plotbpt(stripe_82x_1[halp_filt_s82][z_bins_s82[i]],
                    stripe_82y[halp_filt_s82][z_bins_s82[i]],
                    nonagn, agn,gsw_xvals1_bpt[valid_bpt][z_bins_sdss[i]],
                    gsw_yvals_bpt[valid_bpt][z_bins_sdss[i]],
                    fname=fnam,
                    title=titl)
        plotbpt_sii(stripe_82x_2[halp_filt_s82][z_bins_s82[i]],
                    stripe_82y[halp_filt_s82][z_bins_s82[i]],
                    nonagn, agn,gsw_xvals2_bpt[valid_bpt][z_bins_sdss[i]],
                    gsw_yvals_bpt[valid_bpt][z_bins_sdss[i]],
                    fname=fnam,
                    title=titl)
        plotbpt_oi(stripe_82x_3[halp_filt_s82][z_bins_s82[i]],
                    stripe_82y[halp_filt_s82][z_bins_s82[i]],
                    nonagn, agn,gsw_xvals3_bpt[valid_bpt][z_bins_sdss[i]],
                    gsw_yvals_bpt[valid_bpt][z_bins_sdss[i]],
                    fname=fnam,
                    title=titl)
        plot_mex(mass_exc_stripe[halp_filt_s82][z_bins_s82[i]],
                    stripe_82y[halp_filt_s82][z_bins_s82[i]],nonagn, agn, 
                    all_sdss_avgmasses[spec_inds_allm1][valid_bpt2][z_bins_sdss[i]],
                    gsw_yvals_bpt[valid_bpt2][z_bins_sdss[i]],
                    fname=fnam,
                    title=titl)
        
#use_mass_bins(mass_bins_sdss,mass_bins_s82,start=8, step=1.0)
#use_mass_bins(mass_bins_sdss,mass_bins_s82,start=8, step=0.5)
#use_mass_bins(massfrac_bins_sdss,massfrac_bins_s82,start=0, step=0.1,frac='frac')

#use_z_bins(z_bins_sdss,z_bins_s82,start=0, step=.05)
#use_z_bins(z_bins_sdss,z_bins_s82,start=0, step=.1)

#use_mass_bins(mass_bins_sdss,mass_bins_s82,start=8, step=0.5)
#ad=np.where((all_sdss_spec_z[spec_inds_allm1][valid_bpt2]<=.3)&(all_sdss_spec_z[spec_inds_allm1][valid_bpt2]>0 ) )[0]
#ad=np.where((all_sdss_spec_z<=.3)&(all_sdss_spec_z>0 ) )[0]
lsfrrelat = {'soft': [r'SFR/M$_{*} = 1.39\cdot 10^{-40}$ L$_{\rm x}/$M$_{*}$', r'SFR = $1.39\cdot 10^{-40}$ L$_{\rm x}$',logsfrsoft],
             'hard': [r'SFR/M$_{*} = 1.26\cdot 10^{-40} $L$_{\rm x}$/M$_{*}$', r'SFR = $1.26\cdot 10^{-40}$ L$_{\rm x}$',logsfrhard],
             'full': [r'SFR/M$_{*} = 0.66\cdot 10^{-40}$ L$_{\rm x}$/M$_{*}$',r'SFR = $0.66\cdot 10^{-40}$ L$_{\rm x}$', logsfrfull]  }

def plot_lxsfr(xraysfr, label, save=False, filtagn=[], filtnonagn=[],fname='',weakha=False):
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
                    marker='^', color='b',label='BPT HII')
        plt.scatter(xraysfr.lum_mass[make_m1[halp_filt_s82]][agn][filtagn],
                    xraysfr.sfr_mass[make_m1[halp_filt_s82]][agn][filtagn],
                    marker='o',facecolors='none',color='k',
                    label='BPT AGN')
    elif weakha:
        plt.scatter(xraysfr.lum_mass_val_filt,xraysfr.sfr_mass_val_filt,marker='x',color='k',label=r'Weak H$\alpha$')
    else:
        plt.scatter(xraysfr.lum_mass_val_filtnoagn,xraysfr.sfr_mass_val_filtnoagn,
                    marker='^', color='b',label='BPT HII')
        plt.scatter(xraysfr.lum_mass_val_filtagn,xraysfr.sfr_mass_val_filtagn,
                    marker='o',facecolors='none',color='k',label='BPT AGN')
   
    avg_log_mass = np.mean( xraysfr.mass_val_filt)
    plt.plot(loglum_arr-avg_log_mass,lsfrrelat[label][2]-avg_log_mass,'k--',label=lsfrrelat[label][0])        
    plt.xlabel(r'log(L$_{\rm x}$/M$_{*}$)',fontsize=20)
    print('avg_log_mass',avg_log_mass)   
    plt.ylabel(r'log(SFR/M$_{*}$)',fontsize=20)
    plt.xlim([27,34])
    plt.ylim([-13.5,-6.5])
    plt.legend(frameon=False,fontsize=15,loc=2)
    ax.set(adjustable='box-forced', aspect='equal')    
    plt.tight_layout()
    if save:
        fig.savefig("plots/s82x/png/xraylum/"+label +fname+ "_lxm_vs_sfrm.png",dpi=250,bbox_inches='tight')
        fig.savefig("plots/s82x/pdf/xraylum/"+label + fname+"_lxm_vs_sfrm.pdf",dpi=250,format='pdf',bbox_inches='tight')
        fig.savefig("plots/s82x/eps/xraylum/"+label + fname+"_lxm_vs_sfrm.eps",format='eps',dpi=250,bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # lum  vs sfr
    #plt.title('Stripe 82X Galaxies, z<0.3, Soft X-Ray Luminosity')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    if len(filtagn) != 0 or len(filtnonagn) != 0:
        plt.scatter(xraysfr.lum[make_m1[halp_filt_s82]][nonagn][filtnonagn],
                    xraysfr.sfr[make_m1[halp_filt_s82]][nonagn][filtnonagn],
                    marker='^', color='b',label='BPT HII')
        plt.scatter(xraysfr.lum[make_m1[halp_filt_s82]][agn][filtagn],
                    xraysfr.sfr[make_m1[halp_filt_s82]][agn][filtagn],
                    marker='o',facecolors='none',color='k',
                    label='BPT AGN')
    elif weakha:
        plt.scatter(xraysfr.lum_val_filt,xraysfr.sfr_val_filt,marker='x',color='k',label=r'Weak H$\alpha$')    
    else:
        plt.scatter(xraysfr.lum_val_filtnoagn,xraysfr.sfr_val_filtnoagn,
                    marker='^', color='b',label='BPT HII')
        plt.scatter(xraysfr.lum_val_filtagn,xraysfr.sfr_val_filtagn,
                    marker='o',facecolors='none',color='k',label='BPT AGN')

    plt.plot(loglum_arr,lsfrrelat[label][2],'k--',label=lsfrrelat[label][1])
    #plt.plot(softl_val[softmess], softsfr_val[softmess])
    plt.xlabel(r'log(L$_{\rm x}$)',fontsize=20)
    plt.ylabel(r'log(SFR)',fontsize=20)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.legend(frameon=False,fontsize=15,loc=2)
    plt.xlim([37.5,44.5])
    plt.ylim([-2.5,4.5])
    ax.set(adjustable='box-forced', aspect='equal')    

    plt.tight_layout() 
    
    if save:
        plt.savefig("plots/s82x/png/xraylum/"+label+fname+"_lx_vs_sfr.png",dpi=250,bbox_inches='tight')
        plt.savefig("plots/s82x/pdf/xraylum/"+label+fname+"_lx_vs_sfr.pdf",dpi=250,bbox_inches='tight')
        plt.savefig("plots/s82x/eps/xraylum/"+label +fname+"_lxm_vs_sfrm.eps",format='eps',dpi=250,bbox_inches='tight')

        plt.close(fig)
    else:
        plt.show()
'''
plot_lxsfr(softxray,'soft',filtnonagn = softxray.filtsoftonlynonagn,filtagn = softxray.filtsoftonlyagn,fname='only',save=True)
plot_lxsfr(softxray,'soft',filtnonagn = softxray.filtothersnonagn,filtagn = softxray.filtothersagn,fname='compl',save=True)

plot_lxsfr(fullxray,'full',save=True)
plot_lxsfr(softxray,'soft',save=True)
plot_lxsfr(hardxray,'hard',save=True)

plot_lxsfr(fullxray,'full',save=False)
plot_lxsfr(softxray,'soft',save=False)
plot_lxsfr(hardxray,'hard',save=False)

plot_lxsfr(fullxray_no,'full',save=True,fname='weakha',weakha=True)
plot_lxsfr(softxray_no,'soft',save=True,fname='weakha',weakha=True)
plot_lxsfr(hardxray_no,'hard',save=True,fname='weakha',weakha=True)

plot_lxsfr(fullxray_no,'full',save=False,fname='weakha',weakha=True)
plot_lxsfr(softxray_no,'soft',save=False,fname='weakha',weakha=True)
plot_lxsfr(hardxray_no,'hard',save=False,fname='weakha',weakha=True)
'''
xbongs = {'full':r'log(L$_{\mathrm{0.5-10\ keV}}$)','soft':r'log(L$_{\mathrm{0.5-2\ keV}}$)','hard':r'log(L$_{\mathrm{2-10\ keV}}$)'}
def plotxbong_lx_z(xraysfr,label):
    plt.scatter(xraysfr.lum_val_filtnoagn,z82[nonagn][xraysfr.validnoagn],marker='o',facecolor='none',edgecolor='k',label='BPT HII')
    plt.scatter(xraysfr.lum_val_filtagn,z82[agn][xraysfr.validagn],marker='x', color='k',label='BPT AGN')
    plt.ylabel('z')
    plt.xlabel(xbongs[label])
    plt.legend()
    plt.tight_layout()
def plothardfull(hard,full,save=False):
    plt.scatter(hard.lum,full.lum,marker='o')
    plt.ylabel(r'L$_{0.5-10\ keV }$')
    plt.xlabel(r'L$_{2-10\ keV}$')
    #plt.legend()
    plt.ylim([40.5,43.5])
    plt.xlim([40.5,43.5])
    plt.tight_layout()
    if save:
        plt.savefig("plots/s82x/png/xraylum/hard_vs_full.png",dpi=250)
        plt.savefig("plots/s82x/pdf/xraylum/hard_vs_full.pdf",dpi=250,format='pdf')
        plt.savefig("plots/s82x/eps/xraylum/hard_vs_full.eps",dpi=250,format='eps')

#plothardfull(hardxray,fullxray)
#plothardfull(hardxray,fullxray_no)
        
def plotfwhmbalmerforbid(eldiag,agnfilt,nonagnfilt,save=False):
    plt.scatter(eldiag.balmerfwhm[nonagnfilt],eldiag.forbiddenfwhm[nonagnfilt],marker='^',color='b',label='BPT HII')
    plt.scatter(eldiag.balmerfwhm[agnfilt],eldiag.forbiddenfwhm[agnfilt],marker='o',label='BPT AGN',color='k',facecolor='none')

    plt.xlabel(r'FWHM$_{Balmer} (km\ s^{-1}$)')
    plt.ylabel(r'FWHM$_{Forbidden}(km\ s^{-1})$')
    #plt.legend()
    #plt.ylim([40.5,43.5])
    #plt.xlim([40.5,43.5])
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("plots/s82x/png/xraylum/hard_vs_full.png",dpi=250)
        plt.savefig("plots/s82x/pdf/xraylum/hard_vs_full.pdf",dpi=250,format='pdf')
        plt.savefig("plots/s82x/eps/xraylum/hard_vs_full.eps",dpi=250,format='eps')
 
#plothardfull(hardxray,fullxray)
#plothardfull(hardxray,fullxray_no)

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
    fig.savefig('plots/s82x/pdf/sky/gswlc_ra_dec_medsky_xray_flats82.pdf',bbox_inches='tight',dpi=5000,format='pdf')
    fig.savefig('plots/s82x/png/sky/gswlc_ra_dec_medsky_xray_flats82.png',bbox_inches='tight',dpi=500,format='png')
    plt.close(fig)#plt.show()
#plot_stripe82_filt()
def plot_ra_dec_map_sep():
    '''spherical'''
    #medium depth
    fig = plt.figure(figsize=(1920/mydpi,1080/mydpi))
    ax = fig.add_subplot(111, projection="aitoff")
    plt.scatter(np.radians(conv_ra(m1Cat_GSW.allra) +120), np.radians(m1Cat_GSW.alldec),s=1,c='g',label='GSWLC Medium',marker=',',alpha=.25)
    plt.xlabel('RA')
    ax.set_xticklabels(np.arange(270,-91,-30))
    plt.scatter(np.radians(conv_ra(Chandra.ra)+120),np.radians(Chandra.dec),c='k',label='Chandra (Stripe 82X), 0.0 $<$ z $<$ 0.3',marker='s',facecolors='none',edgecolors='k',alpha=.5,s=1)
    plt.scatter(np.radians(conv_ra(XMM1.ra)+120),np.radians(XMM1.dec),c='c',label=r'XMM Newton (Stripe 82X: AO10), 0.0 $<$ z $<$ 0.3', marker='D', facecolors='none',edgecolors='c',alpha=.5,s=1)
    plt.scatter(np.radians(conv_ra(XMM2.ra)+120),np.radians(XMM2.dec),c='m',label=r'XMM Newton (Stripe 82X: AO13), 0.0 $<$ z $<$ 0.3)', facecolors='none', marker='*',edgecolors='m',alpha=.5,s=1)
    
    plt.ylabel('Dec')
    plt.legend(fontsize=20,markerscale=10)
    plt.grid(True)
    fig.savefig('plots/s82x/pdf/sky/gswlc_ra_dec_medsky_xray.pdf',bbox_inches='tight',dpi=5000,format='pdf')
    fig.savefig('plots/s82x/png/sky/gswlc_ra_dec_medsky_xray.png',bbox_inches='tight',dpi=500,format='png')

    #plt.show()


    plt.close(fig)
    #deep 


def plot_ra_dec_map_sep_flat():
    '''flats'''


    #medium depth
    fig = plt.figure(figsize=(1920/mydpi,1080/mydpi))

    plt.scatter(conv_ra(m1Cat_GSW.allra)*-1, m1Cat_GSW.alldec,s=1,c='g',label='GSWLC Medium',marker=',',alpha=.1)
    plt.xlabel('RA')
    #ax.set_xticklabels(np.arange(270,-91,-30))
    plt.scatter(conv_ra(Chandra.ra)*-1,Chandra.dec,c='k',label='Chandra (Stripe 82X), 0.0 $<$ z $<$ 0.3',marker=',',s=1,alpha=.3)
    plt.scatter(conv_ra(XMM1.ra)*-1,XMM1.dec,c='c',label=r'XMM Newton (Stripe 82X: AO10), 0.0 $<$ z $<$ 0.3',marker=',',s=1,alpha=.3)
    plt.scatter(conv_ra(XMM2.ra)*-1,XMM2.dec,c='m',label=r'XMM Newton (Stripe 82X: AO13), 0.0 $<$ z $<$ 0.3)',marker=',',s=1,alpha=.3)
    plt.xlim(270,-90)
    
    #plt.ylim(-90,90)
    plt.ylabel('Dec')
    plt.legend(fontsize=20,markerscale=10)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig('plots/s82x/pdf/sky/gswlc_ra_dec_medsky_xray_flat.pdf',bbox_inches='tight',dpi=5000,format='pdf')
    fig.savefig('plots/s82x/png/sky/gswlc_ra_dec_medsky_xray_flat.png',bbox_inches='tight',dpi=500,format='png')
    #plt.show()
    
    plt.close(fig)
    #deep 
