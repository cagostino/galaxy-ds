"""Cosmological utilities: luminosity, flux, SFR, sky distances.

Ported from original ast_func.py (commit 83572ef).
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from constants import COSMO, DELTA_SSFR_SLOPE, DELTA_SSFR_INTERCEPT


def get_lum_from_flux(flux, z):
    """Convert flux to luminosity using luminosity distance.

    Parameters
    ----------
    flux : array-like
        Flux in erg/s/cm^2.
    z : array-like
        Redshift.

    Returns
    -------
    lum : ndarray
        Luminosity in erg/s.
    """
    distances = np.array(COSMO.luminosity_distance(z)) * 3.086e24  # Mpc → cm
    return 4 * np.pi * distances ** 2 * flux


def get_flux_from_lum(lum, z):
    """Convert luminosity to flux."""
    distances = np.array(COSMO.luminosity_distance(z)) * 3.086e24
    return lum / (4 * np.pi * distances ** 2)


def halpha_to_sfr(halpha_lum):
    """H-alpha luminosity to fiber SFR (Kennicutt 1998, corrected).

    log(SFR) = log(7.9e-42) + log(L_Ha) - 0.24
    """
    return np.log10(7.9e-42) + np.log10(halpha_lum) - 0.24


def delta_ssfr(mass, ssfr):
    """Distance from the star-forming main sequence (Agostino+19).

    delta_sSFR = sSFR - (m * mass + b)
    """
    return ssfr - (mass * DELTA_SSFR_SLOPE + DELTA_SSFR_INTERCEPT)


def sky_separation(ra1, dec1, ra2, dec2):
    """Angular separation between two sky positions (degrees in, arcsec out)."""
    c1 = SkyCoord(ra1, dec1, unit='deg', frame='icrs')
    c2 = SkyCoord(ra2, dec2, unit='deg', frame='icrs')
    return c1.separation(c2).arcsec
