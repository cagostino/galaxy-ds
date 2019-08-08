import numpy as np
import matplotlib.pyplot as plt
from plot2dhist import plot2dhist
from demarcations import *
plt.rc('font',family='serif')
plt.rc('text',usetex=True)
nii_ha_xmm3, oiii_hb_xmm3 = np.loadtxt('xrayagnbptvals.txt',unpack=True)
nii_ha_gsw, oiii_hb_gsw = np.loadtxt('gswbptvals.txt',unpack=True)

xraybptagn_filt = np.int64(np.loadtxt('xraybptagnfilt.txt'))
xraybpthii_filt = np.int64(np.loadtxt('xraybpthiifilt.txt'))


def plotbpt(xvals,yvals, nonagnfilt, agnfilt, bgx,bgy,save=False,filename='',labels=True,
            leg=True,title=None):
    '''
    for doing bpt diagram with gsw background
    '''
    if save:
        fig = plt.figure()
    valbg = np.where((np.isfinite(bgx) & (np.isfinite(bgy))) &
            (bgy > -1.2) &( bgy < 1.2) & (bgx<1)&(bgx > -2) )[0]
    nx = 3/0.01
    ny = 2.4/0.01
    plot2dhist(bgx[valbg],bgy[valbg],nx,ny)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.plot(np.log10(xline1_kauffman),np.log10(yline1_kauffman),c='k',ls='-.')#,label='Kauffman Line')
    plt.scatter(xvals[nonagnfilt],yvals[nonagnfilt],
                marker='^',color='b',label='X-Ray AGN (BPT-HII)',s=20)
    plt.scatter(xvals[agnfilt],yvals[agnfilt],
                marker='o',facecolors='none',color='k',label='X-Ray AGN (BPT-AGN)',s=20)
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
        
'''
plotbpt(nii_ha_xmm3,oiii_hb_xmm3,xraybpthii_filt, xraybptagn_filt, nii_ha_gsw,oiii_hb_gsw)
'''