import numpy as np
import matplotlib.pyplot as plt

def plothist(prop,propname, nbins=20, rang=[]):
    if len(rang)==0:
        rang =[np.min(prop),np.max(prop)]
    plt.hist(prop,histtype='step',bins=nbins, range=rang)
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
