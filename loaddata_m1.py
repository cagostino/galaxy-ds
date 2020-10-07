catfold='catalogs/'
import numpy as np
print('loading GSW')
m1 = np.loadtxt(catfold+"GSWLC-M1.dat", unpack = True, usecols=(0,1), dtype=np.int64)
redshift_m1 =np.loadtxt(catfold+"GSWLC-M1.dat", unpack = True, usecols=(7,))
allm1 = np.loadtxt(catfold+"GSWLC-M1.dat", unpack = True, usecols=(5, 6, 11, 9, 2, 4, 3, 19, 17, 18))
#for getting r mags for doing x-ray duplicate removal
m1_photcatids = np.loadtxt(catfold+'gs_mis_sdss_phot.dat', unpack=True,usecols=(6,),dtype=np.int64)
m1_modelrflux = np.loadtxt(catfold+'gs_mis_sdss_phot.dat', unpack=True, usecols=(41,))
ind2_m1phot = np.loadtxt(catfold+'photmatchinginds.txt', dtype=np.int64)