import numpy as np
import astropy.io.fits as pf
manga = pf.open('catalogs/drpall-v2_1_2.fits')
galras, galdecs = np.loadtxt('observing_sample_supp.txt', unpack=True,delimiter=',', usecols = (2,3))
galids = np.loadtxt('observing_sample_supp.txt', unpack=True,delimiter=',', usecols = (1))

mat  = []
for i in range(len(galids)):
    print(galras[i])
    print(galdecs[i])
    mt = np.where((abs(manga[1].data.field('objra') -galras[i]) < 0.01)&
            (abs(manga[1].data.field('objdec') - galdecs[i]) <0.01))[0]
    print(mt)
    if len(mt)>0:
        print(manga[1].data.field('objra')[mt])
        print(manga[1].data.field('objdec')[mt])



