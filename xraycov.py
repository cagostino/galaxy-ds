import glob
import time
from astropy.wcs import WCS
import astropy.io.fits as pf
import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import roberts,sobel, scharr, prewitt
#from astimage import *

plt.rc('font',family='serif')
plt.rc('text',usetex=True)

def conv_ra(ra):
    '''
    convert ra for plotting spherical aitoff projection.
    '''
    copra = np.copy(ra)
    for i in range(len(ra)):
        if copra[i] >270:
            copra[i] =-360+copra[i]
    return (copra)*(-1)
sampleobsids = np.loadtxt('catalogs/sampleobstimes.txt', dtype=np.str)

ragsw,decgsw = np.loadtxt('catalogs/gsw2galsbpt.txt') #galaxies with BPT SNR > 2
'''
-*8000*.ftz are the full-field images
-Need them for the three diff. X-ray cameras
'''

'''
print('M1')
xmm3_m1 = glob.glob('xray_imgs/xmm3/m1/*/pps/*8000*.FTZ')
uniq_m1, uniq_ind_m1=np.unique([xmm3_m1[i].split('/')[3] for i in range(len(xmm3_m1))],return_index=True)
xmm3_m1 = np.array(xmm3_m1) 
print('M2')
xmm3_m2 = glob.glob('xray_imgs/xmm3/m2/*/pps/*8000*.FTZ')
uniq_m2, uniq_ind_m2=np.unique([xmm3_m2[i].split('/')[3] for i in range(len(xmm3_m2))],return_index=True)
xmm3_m2 = np.array(xmm3_m2) 
print('pn')
xmm3_pn = glob.glob('xray_imgs/xmm3/pn/*/pps/*8000*.FTZ')
uniq_pn, uniq_ind_pn=np.unique([xmm3_pn[i].split('/')[3] for i in range(len(xmm3_pn))],return_index=True)
xmm3_pn = np.array(xmm3_pn)  
allimglists = np.hstack([xmm3_m1, xmm3_m2, xmm3_pn])
'''

#loading from file
allimglists = np.loadtxt('xrayimglists.txt', dtype=np.str)
def getxraycov_byimg(imglist, racomp, deccomp):
    '''
    Determines if a field contains galaxies for given ra/dec.
    '''
    gswinds = np.arange(len(racomp))
    ra_centarr = []
    dec_centarr = []
    ra_missarr = []
    dec_missarr = []
    matchedgals = np.array([])
    matchedfields = []
    unmatchedfields = []
    for i, img in enumerate(imglist):
        mos = Mosaic(img)
        obsid = img.split('/')[3]         
        if i%100==0:
            print(i)
        matchedgalfield, matchedfield = findmatch1field(racomp, deccomp, mos)
        if matchedfield:
            print('n gals:', len(matchedgalfield))
            matchedgals = np.append(matchedgals,matchedgalfield)
            matchedfields.append(i)
            ra_centarr.append(mos.ra_cent)
            dec_centarr.append(mos.dec_cent)
        else:
            unmatchedfields.append(i)
            ra_missarr.append(mos.ra_cent)
            dec_missarr.append(mos.dec_cent)
        del mos
                
        #elif obsid in sampleobsids:
        #    print('CHECK')
        #    print(i, img)
            
        
    unmatchedgals = []
    matchedgals = np.int64(np.unique(matchedgals))
    for j in range(len(racomp)):
        if gswinds[j] not in matchedgals:
            unmatchedgals.append(gswinds[j])
    return matchedgals, np.array(matchedfields), unmatchedgals, np.array(ra_centarr), np.array(dec_centarr), unmatchedfields, np.array(ra_missarr), np.array(dec_missarr)
class Mosaic:
    '''
    Class for opening an X-ray image and turning it into a binary mask.
    '''
    def __init__(self, filnam):
        self.struct =np.array([[0,0,1,0,0],
                               [0,1,1,1,0],
                               [0,0,1,0,0]]) #kernel for dilation
        self.obsid = filnam.split('/')[3]
        self.hdul = pf.open(filnam)
        self.hdu = pf.open(filnam)[0]
        self.hdul.close() #closing necessary for freeing memory
        self.wcs = WCS(self.hdu.header)
        self.dil = nd.binary_dilation(self.hdu.data,iterations=10,structure=self.struct)
        #converting all nonzero elements to 1s
        self.nonzero = np.where(self.hdu.data !=0)
        self.bin_img = np.copy(self.hdu.data)
        self.bin_img[self.nonzero[0],self.nonzero[1]] = 1
        self.edges = feature.canny(self.dil,sigma=2)
        self.dil = nd.binary_erosion(self.dil,iterations=10)
        self.ny,self.nx = self.hdu.data.shape
        self.x = np.arange(self.nx)
        self.y = np.arange(self.ny)
        self.header = self.hdu.header
        self.X,self.Y = np.meshgrid(self.x,self.y)
        self.ra, self.dec = self.wcs.all_pix2world(self.X,self.Y,0)
        self.ra_cent = np.mean(self.ra) #self.header['RA_OBJ']
        self.dec_cent = np.mean(self.dec) #self.header['DEC_OBJ']
        self.coords = np.dstack((self.ra,self.dec))
    def get_intpixdist(self):
        return np.sqrt( ( (self.ra[0][0]-self.ra[0][1])*np.cos(np.radians((self.dec[0][0]+self.dec[0][1])/2)) )**2 
                       +(self.dec[0][0]-self.dec[0][1])**2)
def concatradecs(fields):
    '''
    To create arrays of all valid RAs, Decs
    Uses lots of memory.. not advised.
    '''
    ras = np.array([])
    decs = np.array([])
    for field in fields:
        ras= np.append(ras,field.ra[field.dil])
        decs = np.append(decs,field.dec[field.dil])
    return ras, decs

def findmatch1field(ra,dec,field):
    '''
    For a list of RA/dec, find if/how many galaxies are contained within a 
    certain field of observations
    '''
    matchedgals = np.array([])
    matchedfield = False
    galinds = np.arange(len(ra))
    intpixdist = field.get_intpixdist()
    ra_cent, dec_cent = field.ra_cent, field.dec_cent
    #extent of the field
    maxfielddist = ((field.dec.max() - field.dec.min())**2. +
                    ((field.ra.max()-field.ra.min())*np.cos( np.radians((field.dec.max()+field.dec.min())/2) ))**2)**(1./2)
    if maxfielddist >15:
        #if it's near the turnover, just set it to be 10
        #sort of overkill since FOV for XMM-Newton is <1 deg.
        print(field.ra.max(), field.ra.min())
        nearbygals=np.where( (ra>350)| (ra<10))[0]
    else:
        print('maxfield dist', maxfielddist, ra_cent, dec_cent)
        distfromfield = (((ra - ra_cent)*np.cos(np.radians((dec+dec_cent)/2)))**2+(dec-dec_cent)**2)**(1./2.)
        nearbygals = np.where(distfromfield <maxfielddist*5)[0]
    matchedgswgals = np.array([])

    #if there is nothing nearby don't bother with calculating
    if len(nearbygals) !=0 :
        #dists = ((ragsw[nearbygals] - field.ra[field.dil][:, None])**2+(decgsw[nearbygals]-field.dec[field.dil][:, None])**2)**(1./2.)
        for i in range(len(nearbygals)):
            dists = (((ra[nearbygals][i] - field.ra[field.dil])*np.cos(np.radians((dec[nearbygals][i]+field.dec[field.dil])/2) ))**2 +
                          (dec[nearbygals][i]-field.dec[field.dil])**2)**(1./2.)
        #        dists = (((ra[nearbygals] - field.ra[field.dil][:, None])*np.cos(np.radians((dec[nearbygals]+field.dec[field.dil][:, None])/2) ))**2 +
        #                  (dec[nearbygals]-field.dec[field.dil][:, None])**2)**(1./2.)
            matched = np.where(dists < intpixdist+7./3600.)[0]
            #print(dists.shape)
            #print(np.unique(matched[1]))
            if len(matched)>0:
                matchedgswgals = np.append(matchedgswgals, galinds[nearbygals[i]])
                
        #the first index will be the index in the field.ra/dec
        #second index will correspond to GSW
    if len(matchedgswgals) >0:
        matchedgals = np.append(matchedgals,matchedgswgals)
        matchedfield = True
    matchedgals = np.int64(np.unique(matchedgals))
    return matchedgals, matchedfield
def getxrcov(imglist, ra, dec, fname):
    '''
    Runs the X-ray coverage and saves out the results
    '''
    matched_ = getxraycov_byimg(imglist, ra, dec)
    matched_gals, matched_fields,unmatched_gals,racent, deccent, unmatched_fields, ramiss, decmiss = matched_
    np.savetxt('catalogs/xraycov/matched_gals_'+fname+'_xrcovg_fields_set.txt', matched_gals)
    np.savetxt('catalogs/xraycov/unmatched_gals_'+fname+'_xrcovg_fields_set.txt', unmatched_gals)#
    np.savetxt('catalogs/xraycov/matchedfields_'+fname+'_xrcovg_fields_set.txt', matched_fields)
    np.savetxt('catalogs/xraycov/racent_'+fname+'_xrcovg_fields_set.txt', racent)#
    np.savetxt('catalogs/xraycov/deccent_' +fname+'_xrcovg_fields_set.txt', deccent)
    np.savetxt('catalogs/xraycov/unmatchedfields_'+fname+'_xrcovg_fields_set.txt', unmatched_fields)
    np.savetxt('catalogs/xraycov/ramiss_'+fname+'_xrcovg_fields_set.txt', ramiss)#
    np.savetxt('catalogs/xraycov/decmiss_' +fname+'_xrcovg_fields_set.txt', decmiss)


getxrcov(allimglists, ragsw, decgsw, 'gsw2xmm3') #to run the main parts
    
'''
Below: If such X-ray coverage is already done, easier to load certain information for plotting
'''
matched_gals =np.int64(np.loadtxt('catalogs/xraycov/matched_gals_gsw2xmm3_xrcovg_fields_set.txt'))
unmatched_gals = np.int64(np.loadtxt('catalogs/xraycov/unmatched_gals_gsw2xmm3_xrcovg_fields_set.txt'))
matched_fields = np.int64(np.loadtxt('catalogs/xraycov/matchedfields_gsw2xmm3_xrcovg_fields_set.txt'))
unmatched_fields = np.int64(np.loadtxt('catalogs/xraycov/unmatchedfields_gsw2xmm3_xrcovg_fields_set.txt'))

racent = np.float64(np.loadtxt('./catalogs/xraycov/racent_gsw2xmm3_xrcovg_fields_set.txt'))
deccent = np.float64(np.loadtxt('./catalogs/xraycov/deccent_gsw2xmm3_xrcovg_fields_set.txt'))

ramiss = np.float64(np.loadtxt('./catalogs/xraycov/ramiss_gsw2xmm3_xrcovg_fields_set.txt'))
decmiss = np.float64(np.loadtxt('./catalogs/xraycov/decmiss_gsw2xmm3_xrcovg_fields_set.txt'))

mydpi=48
def plotcov( matched_fields, save=False,fname=''):
    '''
    Plotting an all-sky map.
    '''
    fig = plt.figure(figsize=(1920/mydpi,1080/mydpi))
    ax = fig.add_subplot(111, projection="aitoff")

    plt.tick_params(axis='both', which='major', labelsize=50)
    plt.tick_params(axis='both', which='minor', labelsize=45)
    plt.grid(True)
    rags60 = np.where(ragsw>300)[0]
    ra_gsw = np.copy(ragsw)
    ra_gsw[rags60] = ra_gsw[rags60]-360
    for img in allimglists[matched_fields]:
        print(img)
        mos = Mosaic(img)
        plt.scatter(np.radians(conv_ra(mos.ra[mos.dil][::1000])+120), np.radians(mos.dec[mos.dil][::1000]),c='gray',s=3,marker='.')
        del mos      
    plt.scatter(np.radians(conv_ra(racent)+120),np.radians(deccent),c='gray',marker='.',s=1,alpha=1, label='3XMM Fields')
    plt.scatter(np.radians(conv_ra(ra_gsw[unmatched_gals])+120),np.radians(decgsw[unmatched_gals]),c='magenta',marker='.',s=2,alpha=0.2,label='GSWLC-M1 Not Covered by 3XMM',zorder=0)
    plt.scatter(np.radians(conv_ra(ra_gsw[matched_gals])+120),np.radians(decgsw[matched_gals]),c='black',marker='.',s=2,alpha=1,label='GSWLC-M1 Covered by 3XMM',zorder=len(matched_fields)+5)
    st = -np.pi+np.pi/6
    pi = np.pi
    posx = [st-pi/18,st+pi/2-pi/18,st+pi-pi/36, st+3*pi/2]
    posy = -pi/6
    labels=[r'$270^{\circ}$',r'$180^{\circ}$',r'$90^{\circ}',r'$0^{\circ}$']
    for i in range(len(posx)):
        plt.text(posx[i],posy,labels[i],fontsize=40)
    plt.xlabel('RA [deg]',fontsize=50)
    plt.ylabel('Dec [deg]',fontsize=50)
    ax.set_xticks([st,st+pi/2,st+pi, st+3*pi/2])
    ax.set_xticklabels([],[])
    ax.set_yticks([-np.radians(75),-pi/3,-pi/6, 0, pi/6,pi/3,np.radians(75)])
    ax.set_yticklabels(['',r'$-60^{\circ}$',r'$-30^{\circ}$','',r'$+30^{\circ}$',r'$+60^{\circ}$'])
    plt.legend(markerscale=10, fontsize=35, loc =8)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/xraycovg/covgmap_xmm_chand_GSW'+fname+'.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/xraycovg/covgmap_xmm_chand_GSW'+fname+'.pdf',dpi=250,format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/xraycovg/covgmap_xmm_chand_GSW'+fname+'.eps',format='eps')
        plt.close(fig)
def plotstuff(im, save=False):
    '''
    Showing a single observation, its binary mask version, and its edges.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111,projection=im.wcs)
    plt.imshow(im.bin_img,origin='lower', cmap='hot',vmin = np.median(im.hdu.data)*5)
    ax.tick_params(axis='both', which='major', pad=20)
    ax.coords[0].set_ticklabel(size=20, fontweight='ultralight', usetex=False)
    ax.coords[1].set_ticklabel(usetex=False, size=20, fontweight='ultralight')
    ax.set_xlabel('RA[deg]',fontsize=20)
    ax.set_ylabel('Dec[deg]',fontsize=20)
    fig.tight_layout()
    fig.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/xraycovg/field.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/xraycovg/field.pdf',dpi=250, format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/xraycovg/field.eps',format='eps')
        plt.close(fig)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection=im.wcs)
    plt.imshow(im.dil,origin='lower', cmap='hot')
    ax.coords[0].set_ticklabel(size=20)
    ax.coords[1].set_ticklabel(size=20)
    locs, labels=plt.xticks()
    ax.set_xlabel('RA[deg]',fontsize=20)
    ax.set_ylabel('Dec[deg]',fontsize=20)
    plt.tight_layout()
    if save:
        fig.savefig('plots/xmm3/png/xraycovg/field_dilat.png',dpi=250)
        fig.savefig('plots/xmm3/pdf/xraycovg/field_dilat.pdf',dpi=250, format='pdf')
        fig.set_rasterized(True)
        fig.savefig('plots/xmm3/eps/xraycovg/field_dilat.eps',format='eps')
        plt.close(fig)
    fig = plt.figure()
    fig.add_subplot(111,projection=im.wcs)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    plt.imshow(im.edges,origin='lower', cmap='hot')
    plt.xlabel('RA',fontsize=20)
    plt.ylabel('Dec',fontsize=20)
    plt.tight_layout()
#plotstuff(allimglists[0])
#plotcov(mosaics, matched_fields, save=False)

