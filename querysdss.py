from astroquery.sdss import SDSS

import numpy as np
ra,dec = np.loadtxt('sdssobjlookupradec2.txt', unpack=True, delimiter=',')
for i in range(len(ra)):
    query = "select ra dec Sigma_Hb_4861 Sigma_Hb_4861_Err Sigma_OIII_5006 Sigma_OIII_5006_Err FROM emissionLinesPort where ra ="+str(ra[i])+' and dec ='+str(dec[i])
    result = SDSS.query_sql(query)
    print(result)