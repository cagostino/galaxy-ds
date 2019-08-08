import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='serif')
plt.rc('text',usetex=True)

lum, sfr = np.loadtxt('lum_sfr.txt', unpack=True)
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
lsfrrelat = {'soft': [r'SFR/M$_{*} = 1.39\cdot 10^{-40}$ L$_{\rm x}/$M$_{*}$', r'SFR = $1.39\cdot 10^{-40}$ L$_{\rm x}$',logsfrsoft],
             'hard': [r'SFR/M$_{*} = 1.26\cdot 10^{-40} $L$_{\rm x}$/M$_{*}$', r'SFR = $1.26\cdot 10^{-40}$ L$_{\rm x}$',logsfrhard],
             'full': [r'SFR/M$_{*} = 0.66\cdot 10^{-40}$ L$_{\rm x}$/M$_{*}$',r'SFR = $0.66\cdot 10^{-40}$ L$_{\rm x}$', logsfrfull]  }

def plot_lxsfr(lum, sfr, label, save=False,filename='',scat=False):
    '''
    Plots star-formation rate versus X-ray luminosity and
    color-codes a region in bottom right where X-ray AGN are.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.scatter(lum, sfr, marker='x', color='k')
    
    plt.plot(loglum_arr,lsfrrelat[label][2],'k--',label=lsfrrelat[label][1],zorder=3)
    if scat:
        plt.fill_between(loglum_arr+scat,lsfrrelat[label][2],y2=lsfrrelat[label][2]-20,color='gray', zorder=0,alpha=0.2,linewidth=0)
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
        plt.savefig("plots/xmm3/pdf/xraylum/"+label+filename+"_lx_vs_sfr.pdf",dpi=250)
        plt.savefig("plots/xmm3/eps/xraylum/"+label +filename+"_lx_vs_sfr.eps",format='eps',dpi=250)
        plt.close(fig)
    else:
        plt.show()
        
'''
plot_lxsfr(lum, sfr, 'full', scat=0.6)
'''