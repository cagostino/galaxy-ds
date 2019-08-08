import numpy as np
import matplotlib.pyplot as plt

def plot2dhist(x,y,nx,ny):
    hist, xedges, yedges = np.histogram2d(x,y,bins = (nx,ny))
    extent= [np.min(xedges),np.max(xedges),np.min(yedges),np.max(yedges)]
    print(extent)
    plt.imshow((hist.transpose())**0.3, cmap='gray_r',extent=extent,origin='lower',
               aspect='auto',alpha=0.9)