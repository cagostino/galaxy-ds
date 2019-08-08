
#ra_allfields, dec_allfields = concatradecs(mosaics)
def findmatch(ragsw,decgsw,ra_all,dec_all):
    matchedgals = []
    matchedras = []
    matcheddecs = []
    for i in range(len(ragsw)):
        ra_gal,dec_gal = ragsw[i],decgsw[i]
        intpixdist= mosaics[0].get_intpixdist()
        radists = (ra_gal-ra_all)**2
        decdist = (dec_gal - dec_all)**2
        dists = np.sqrt(radists + decdist)
        matched = np.where(dists <= intpixdist)[0]

        if matched.size >0:
            matchedgals.append(i)
            matchedras.append(ra_gal)
            matcheddecs.append(dec_gal)
            print(matched)
    return matchedgals, matchedras, matcheddecs

def findmatchbyfield(ragsw,decgsw,fields):
    matchedgals = np.array([])
    matchedfields = []
    gswinds = np.arange(len(ragsw))
    for i in range(len(fields)):
        if i%20 == 0:
            print(i,time.ctime())

        field = fields[i]
        intpixdist = field.get_intpixdist()
        ra_cent, dec_cent = field.ra_cent, field.dec_cent
        #extent of the field
        maxfielddist = ((field.dec.max() - field.dec.min())**2+((field.ra.max()-field.ra.min())*np.cos(np.radians((field.dec.max()+field.dec.min())/2)) )**2)**(1./2)
        if maxfielddist >350:
            print(field.ra.max(), field.ra.min())
            maxfielddist = 5
        print('maxfield dist', maxfielddist)
        distfromfield = (((ragsw - ra_cent)*np.cos((decgsw+dec_cent)/2))**2+(decgsw-dec_cent)**2)**(1./2.)
        nearbygals = np.where(distfromfield <maxfielddist*2)[0]
        #if there is nothing nearby don't bother with calculating
        if len(nearbygals) ==0 :
            continue

        dists = (((ragsw[nearbygals] - field.ra[field.dil][:, None])*np.cos(np.radians((decgsw[nearbygals]+field.dec[field.dil][:, None])/2)) )**2+(decgsw[nearbygals]-field.dec[field.dil][:, None])**2)**(1./2.)
        matched = np.where(dists <intpixdist)
        #the first index will be the index in the field.ra/dec
        #second index will correspond to GSW
        print(np.unique(matched[1]))
        matchedgswgals = gswinds[nearbygals[np.unique(matched[1])]]
        if len(matchedgswgals) >0:
            matchedgals = np.append(matchedgals,matchedgswgals)
            matchedfields.append(i)
    unmatchedgals = []
    matchedgals = np.int64(np.unique(matchedgals))
    for i in range(len(ragsw)):
        if gswinds[i] not in matchedgals:
            unmatchedgals.append(gswinds[i])
    return matchedgals, unmatchedgals, matchedfields