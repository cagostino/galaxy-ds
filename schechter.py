import numpy as np

from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import Distance
from scipy.optimize import curve_fit
from astropy.stats import bootstrap
from scipy.odr import ODR, Model, Data, RealData
import cosmolopy as cpy
from cosmolopy import fidcosmo
from cosmolopy import magnitudes as mag
unit_flx = u.erg/u.s/u.cm/u.cm
app_oiii_sub, abs_oiii_sub = mag.magnitude_AB(sfrm_gsw2.allagn_df.z, sfrm_gsw2.allagn_df.oiiiflux_corr_sub, 5007*(sfrm_gsw2.allagn_df.z) +5007, **fidcosmo)
app_oiii_presub, abs_oiii_presub = mag.magnitude_AB(sfrm_gsw2.allagn_df.z, sfrm_gsw2.allagn_df.oiiiflux_corr, 5007*(sfrm_gsw2.allagn_df.z) +5007, **fidcosmo)

def schechter_mag(mags, alpha, phi_star, m_star):
    first = 0.4*np.log(10)*phi_star
    second = (10**(0.4*(m_star-mags)))**(alpha+1)
    third = np.exp(-10**(0.4*(m_star-mags)))
    return np.log10(first*second*third)
def pow_mag(mags, alpha, phi_star, m_star):
    first = 0.4*np.log(10)*phi_star
    second = (10**(0.4*(m_star-mags)))**(alpha+1)
    third = np.exp(-10**(0.4*(m_star-mags)))
    return np.log10(first*second*third)


def schechter_lum (lums, alpha, phi_star, lum_star):
    first = phi_star/lum_star
    second = (lums/lum_star)**(alpha)
    third = np.exp(-lums/lum_star)
    return first*second*third
def comp_lum_func(lum, distances, bins =[], zmin=0.01, zmax=0.3, nbins=-1, maglim=17.77):
    lum = np.copy(np.array(lum))
    nums = []
    lum_sorted = np.sort(lum)
    if nbins ==-1:
        nbins=15
    if len(bins) ==0:
        bins = np.round(np.linspace(lum_sorted[np.int(0.02*len(lum))],lum_sorted[np.int(0.98*len(lum))], nbins),1)
    bincenters = []
    densities = []
    completeness_vals = []
    compl_factors = []
    distmax_z = 10*10**(np.float64(Distance(z=zmax).distmod/5))/1e6 #Mpc
    distmin = 10*10**(np.float64(Distance(z=zmin).distmod)/5)/1e6 # in Mpc
    
    for i in range(len(bins)-1):            
        binmin = bins[i]
        binmax = bins[i+1]
        bincenter= (binmin+binmax)/2.
        bincenters.append(bincenter)
        distmodmax = maglim-bincenter
        distmax = 10*10**(distmodmax/5)/1e6 #in Mpc
        #print('distmax', distmax)
        if distmax > distmax_z:
            #if the maximum distance is beyond the redshift cut, correct for that
            distmax = distmax_z

        #print('distmin', distmin)
        binned_gals = np.where((lum>binmin) & (lum<binmax ))[0]
        count = binned_gals.size
        distmod_avg = np.array(distances)[binned_gals]
        dist_avg = np.mean(10*10**(distmod_avg/5)/1e6) #in Mpc
        #print('distavg', dist_avg)
        Vmax = 4*np.pi*(distmax-distmin)**3/3
        V = 4*np.pi*(dist_avg-distmin)**3/3
        nums.append(count)
        densities.append(count/(Vmax))
        
        v_over_vmax = V/Vmax
        compl_factor = 0.5/v_over_vmax
        completeness_vals.append(v_over_vmax)
        compl_factors.append(compl_factor)
    return bins,np.array(bincenters), np.array(nums), np.array(densities), np.array(completeness_vals), np.array(compl_factors)

def fit_schecht(mags, dens, alpha_0=-1.2, weights=[]):
    '''
    For fitting a schechter functiton
    '''
    m_star_0 = np.mean(mags)
    phi_star_0 = 10**np.mean(dens)
    nans = np.where((np.isinf(dens)) | (np.isnan(dens)))[0]
    print(phi_star_0)
    if len(nans) !=0:
        mags = np.copy(np.delete(mags,nans))
        dens = np.copy(np.delete(dens, nans))
    #pdb.set_trace()
    if len(weights)!=0:
        if len(nans) !=0:
            weights = np.copy(np.delete(weights, nans))
        popt, pcov = curve_fit(schechter_mag, mags, dens,sigma=weights, p0 = (alpha_0, phi_star_0, m_star_0), maxfev=20000)

    else:
        popt, pcov = curve_fit(schechter_mag, mags, dens, p0 = (alpha_0, phi_star_0, m_star_0))
    alpha, phi_star, m_star = popt
    mags_fit = np.linspace(np.min(mags)-0.1,np.max(mags)+0.1, 100)
    schecht_fit = schechter_mag(mags_fit, alpha, phi_star, m_star)
    print('alpha: ',alpha)
    print('phi*: ',phi_star)
    print('m*: ',m_star)
    
    return alpha, phi_star, m_star, schecht_fit, mags_fit
lum_fn_sub = comp_lum_func(np.array(abs_oiii_sub), np.array(sfrm_gsw2.allagn_df.z), maglim=20.4)
lum_fn_presub = comp_lum_func(np.array(abs_oiii_presub), np.array(sfrm_gsw2.allagn_df.z), maglim=20.4)
fit_sch_sub = fit_schecht(lum_fn_sub[1], np.log10(lum_fn_sub[3]))
fit_sch_presub = fit_schecht(lum_fn_presub[1], np.log10(lum_fn_presub[3]))
def bootstrap_lum_fn( lum, dist, n_resamp = 10000, zmin=0.01, zmax = 0.3, bins = [], nbins=15, maglim=20.32):
    '''
    Bootstrapping lum function by way of indices to get resampled 
    lums and distances
    '''
    inds_lum = np.arange(len(lum))
    bootstr_samps = np.int64(bootstrap(inds_lum, n_resamp))
    
    nums_fns = np.zeros(shape=(n_resamp, nbins-1))
    dens_fns = np.zeros(shape=(n_resamp, nbins-1))
    completeness_fns =  np.zeros(shape=(n_resamp, nbins-1))
    compl_factors_fns = np.zeros(shape=(n_resamp, nbins-1))
    
    for i, samp in enumerate(bootstr_samps):
        lumfn = comp_lum_func(lum[samp], dist[samp], zmin = zmin, zmax = zmax, bins = bins, nbins= nbins, maglim=maglim)
   
        bincenters_abs_r = lumfn[1]
        nums_abs_r = lumfn[2]
        nums_fns[i, :] = nums_abs_r
        dens_abs_r = lumfn[3]
        dens_fns[i, :] = dens_abs_r
        
        completeness_abs_r = lumfn[4] 
        completeness_fns[i,:] = completeness_abs_r
        
        compl_factor = lumfn[5]
        compl_factors_fns[i, :] = compl_factor
    
    dens_fns_means = np.mean(dens_fns, axis=0)
    dens_fns_stds = np.std(dens_fns, axis=0)
    dens_fns_errs = dens_fns_stds/np.sqrt(n_resamp)
    completeness_fns_means = np.mean(completeness_fns, axis=0)
    compl_factors_means = np.mean(compl_factors_fns, axis=0)
    try:
        schecht = fit_schecht(bincenters_abs_r, np.log10(dens_fns_means), weights = compl_factors_means*0.434*dens_fns_stds/dens_fns_means )
        alpha, phi_star, m_star, schecht_fit, mags_fit = schecht
        
        return bincenters_abs_r, dens_fns_means, dens_fns_stds, dens_fns_errs, compl_factors_means, completeness_fns_means,  alpha, phi_star, m_star, schecht_fit, mags_fit
    except :
        return bincenters_abs_r, dens_fns_means, dens_fns_stds, dens_fns_errs, compl_factors_means, completeness_fns_means, -99, -99, -99, -99, -99
    #return schecht
bootstr_sub_fn = bootstrap_lum_fn(np.array(abs_oiii_sub), np.array(sfrm_gsw2.allagn_df.z), maglim=20.4)
bootstr_presub_fn = bootstrap_lum_fn(np.array(abs_oiii_presub), np.array(sfrm_gsw2.allagn_df.z), maglim=20.4)

def get_z_bins(self, nbins=-1): 
    zbins = np.array([0.01, 0.10666, 0.20333, 0.3])
    bincenters = []
    dens = []
    dens_std = []
    dens_errs = []        
    compl = []
    compl_factors = []
    alphas = []
    phi_stars = []
    m_stars = []
    schecht_fits = []
    mags_fits = []
    if nbins==-1:
        nbins=self.nbins_lf
    for i in range(len(zbins)-1):
        zmn = zbins[i]
        zmx = zbins[i+1]
        print('zmn, zmax: ', zmn, zmx)
        zfilt = np.where( (self.z_filt_kcorr < zmx) &(self.z_filt_kcorr>=zmn))[0]
        magsort = np.sort(self.abs_petromag_r_dered_kcorr[zfilt])
        print(nbins)
        bins_in = np.linspace(np.min(magsort[2:]), np.max(magsort[:-2]), nbins)
        print(bins_in)
        lumfn = self.bootstrap_lum_fn(self.abs_petromag_r_dered_kcorr[zfilt], self.distmod[zfilt], zmin= zmn, zmax = zmx, bins=bins_in, nbins=nbins)
        

        bincenters_abs_r = lumfn[0]
        bincenters.append(bincenters_abs_r)

        dens_abs_r = lumfn[1]
        dens.append(dens_abs_r)

        dens_std_r = lumfn[2]
        dens_std.append(dens_std_r)
        
        dens_errs_r = lumfn[3]
        dens_errs.append(dens_errs_r)
       
        compl_factor = lumfn[4]
        compl_factors.append(compl_factor)

        
        completeness_abs_r = lumfn[5] 
        compl.append(completeness_abs_r)
        
        alpha_r = lumfn[6]
        phi_star_r = lumfn[7]
        m_star_r = lumfn[8]
        schecht_fit_r = lumfn[9]
        mags_fit_r = lumfn[10]
            
        alphas.append(alpha_r)
        phi_stars.append(phi_star_r)
        m_stars.append(m_star_r)
        schecht_fits.append(schecht_fit_r)
        mags_fits.append(mags_fit_r)
        
    return  bincenters, dens, dens_std, compl,compl_factors, alphas, phi_stars, m_stars, schecht_fits, mags_fits

def plotlumfn(x, y, xlab, ylab, title, sym, save=False, mags=False, magfit=[],fit=[], alpha=0, phi_star=0,m_star=0, lab='', errsy=[]):
    '''
    If magfit /fit are designated, need to also input alpha, phi_star, m_star
    '''
    plt.scatter(x,y, label=lab, marker=sym, edgecolor='k', facecolor='none')
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params(direction='in',axis='both',which='both')
    if mags:
        plt.gca().invert_xaxis()
    if type(fit) != int and len(fit) !=0 and alpha !=0:
        plt.plot(magfit, fit,'k--', label=r'Fit: $\alpha = $'+str(round(alpha,2))+r', $M^{*}$ = '+str(round(m_star,2))+r', log($\Phi^{*}$) =' + str(round(phi_star,2)))    
    plt.legend(frameon=False, fontsize=14)
    if len(errsy) !=0:
        plt.errorbar(x, y, yerr=errsy, fmt='none', capsize=4, ecolor='k', elinewidth=1, capthick=1)
    plt.tight_layout()
    if save:
        plt.savefig('figures/'+title+'.png', dpi=250)
        plt.close()
        
        
plotlumfn(lum_fn_sub[1], np.log10(lum_fn_sub[3]), r'Abs. $[OIII]$', 
  r'log(N/V$_{\mathrm{max}}$)','./oiiilum_fn', 'o',
  alpha=fit_sch_sub[0], m_star=fit_sch_sub[2], phi_star = np.log10(fit_sch_sub[1]),
  lab='AGN', mags=True, fit=fit_sch_sub[-2], magfit=fit_sch_sub[-1],save=False)
plotlumfn(lum_fn_presub[1], np.log10(lum_fn_presub[3]), r'Abs. $[OIII]$', 
  r'log(N/V$_{\mathrm{max}}$)','./oiiilum_fn', 'o',
  alpha=fit_sch_presub[0], m_star=fit_sch_presub[2], phi_star = np.log10(fit_sch_presub[1]),
  lab='AGN', mags=True, fit=fit_sch_presub[-2], magfit=fit_sch_presub[-1],save=False)



        
plotlumfn(bootstr_sub_fn[0], np.log10(bootstr_sub_fn[1]), r'Abs. $[OIII]$', 
  r'log(N/V$_{\mathrm{max}}$)','./oiiilum_fn', 'o',
  alpha=bootstr_sub_fn[6], m_star=bootstr_sub_fn[8], phi_star = np.log10(bootstr_sub_fn[7]),
  lab='AGN', mags=True, fit=bootstr_sub_fn[9], magfit=bootstr_sub_fn[10],save=False)
plotlumfn(lum_fn_presub[1], np.log10(lum_fn_presub[3]), r'Abs. $[OIII]$', 
  r'log(N/V$_{\mathrm{max}}$)','./oiiilum_fn', 'o',
  alpha=fit_sch_presub[0], m_star=fit_sch_presub[2], phi_star = np.log10(fit_sch_presub[1]),
  lab='AGN', mags=True, fit=fit_sch_presub[-2], magfit=fit_sch_presub[-1],save=False)
