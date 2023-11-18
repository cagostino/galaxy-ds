from astropy.wcs import WCS
import astropy.io.fits as pf
import numpy as np
import matplotlib.pyplot as plt

class AstImage:
    def __init__(self,filnam):
        self.hdu = pf.open(filnam)[0]
        self.im = self.hdu.data
        self.wcs = WCS(self.hdu.header)
    def plotim(self, spec=False):
        fig = plt.figure()
        if not spec:
            fig.add_subplot(111,projection=self.wcs )

            plt.xlabel('RA')
            plt.ylabel('Dec')
        plt.imshow(self.im,origin='lower', cmap='gray_r',vmin = np.median(self.im)*5,vmax =np.median(self.im)*10)
