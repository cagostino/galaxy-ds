import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='serif')
plt.rc('text',usetex=True)
logexptimes = np.loadtxt('xmm_logexptimes.txt')

def plot_xmmtime(time,name,filename='',save=False):
    '''
    Used to plot a histogram of the xmm times
    '''
    nbins = np.arange(3.0,5.4,0.2)
    fig = plt.figure()
    plt.hist(time,bins = nbins,histtype='step')
    plt.xlabel(name, fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
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

