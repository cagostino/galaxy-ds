catfold='catalogs/'
import numpy as np
print('loading GSW')
m2 = np.loadtxt(catfold+"GSWLC-M2.dat", unpack = True, usecols=(0,1), dtype=np.int64)
redshift_m2 = np.loadtxt(catfold+"GSWLC-M2.dat", unpack = True, usecols=(7,))
header=  ['']
allm2 = np.loadtxt(catfold+"GSWLC-M2.dat", unpack = True, usecols=(5, 6, 11, 9, 2, 4, 3, 19, 17, 18,13, 12, 10))

a2 = np.loadtxt(catfold+"GSWLC-A2.dat", unpack = True, usecols=(0,1), dtype=np.int64)
redshift_a2 = np.loadtxt(catfold+"GSWLC-A2.dat", unpack = True, usecols=(7,))
alla2 = np.loadtxt(catfold+"GSWLC-A2.dat", unpack = True, usecols=(5, 6, 11, 9, 2, 4, 3, 19, 17, 18,13, 12, 10))

x2 = np.loadtxt(catfold+"GSWLC-X2.dat", unpack = True, usecols=(0,1), dtype=np.int64)
redshift_x2 = np.loadtxt(catfold+"GSWLC-X2.dat", unpack = True, usecols=(7,))
allx2 = np.loadtxt(catfold+"GSWLC-X2.dat", unpack = True, usecols=(5, 6, 11, 9, 2, 4, 3, 19, 17, 18,13, 12, 10))


#for getting r mags for doing x-ray duplicate removal
m1_photcatids = np.loadtxt(catfold+'gs_mis_sdss_phot.dat', unpack=True,usecols=(6,),dtype=np.int64)
m1_modelrflux = np.loadtxt(catfold+'gs_mis_sdss_phot.dat', unpack=True, usecols=(41,))
ind2_m1phot = np.loadtxt(catfold+'photmatchinginds.txt', dtype=np.int64)
sigma1_m = np.loadtxt(catfold+'sigma1_mis.dat', dtype=np.float64, usecols=(2), unpack=True)
env_nyu_m = np.loadtxt(catfold+'envir_nyu_mis.dat', dtype=np.float64, usecols=(0), unpack=False)
env_bald_m = np.loadtxt(catfold+'baldry_mis.dat', dtype=np.float64, usecols=(4), unpack=True)
irx_m = np.loadtxt(catfold+'irexcess_mis.dat', dtype=np.float64, usecols=(0), unpack=True)

nuv, nuverr, fuv, fuverr = np.loadtxt(catfold+'gs_mis_galex_br.dat', usecols =(14,15,18,19),unpack=True)

axisrat = 1-np.loadtxt(catfold+'simard_ellip_mis.dat', dtype=np.float64, usecols=(1), unpack=True)

allm2 = np.vstack((allm2, sigma1_m,env_nyu_m, env_bald_m, irx_m, axisrat, allm2[0]*0-999, allm2[0]*0-999)) #nuv[ind2_m1phot], fuv[ind2_m1phot]))
allx2 = np.vstack((allx2, allx2[0]*0 - 999,allx2[0]*0 - 999,allx2[0]*0 - 999, allx2[0]*0-999, allx2[0]*0-999,allx2[0]*0-999,allx2[0]*0-999))
