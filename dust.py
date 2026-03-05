"""Dust extinction and correction functions.

Ported from original ELObj.py (commit 83572ef, lines 16-109).
Uses CCM89 extinction law via the `extinction` package.
"""

import numpy as np
import extinction as ext

from constants import (
    BALMER_DEC_SF, BALMER_DEC_AGN,
    AV_HYBRID_SLOPE, AV_HYBRID_INTERCEPT,
    AV_HYBRID_SLOPE_SUB, AV_HYBRID_INTERCEPT_SUB,
)


def compute_av(ha, hb, agn=False, zeroed=False):
    """Compute A_V from the Balmer decrement (Cardelli law).

    Parameters
    ----------
    ha, hb : array-like
        H-alpha and H-beta fluxes.
    agn : bool
        Use AGN intrinsic ratio (3.1) instead of SF (2.86).
    zeroed : bool
        Clip A_V to [0, 3].

    Returns
    -------
    av : ndarray
    """
    ha = np.asarray(ha, dtype=np.float64)
    hb = np.asarray(hb, dtype=np.float64)
    dec_rat = BALMER_DEC_AGN if agn else BALMER_DEC_SF
    av = 7.23 * np.log10((ha / hb) / dec_rat)
    if zeroed:
        av = np.clip(av, 0.0, 3.0)
    return av


def dustcorrect(flux, av, wavelength):
    """De-redden flux using CCM89.

    Parameters
    ----------
    flux : array-like
        Observed flux.
    av : array-like
        A_V in magnitudes.
    wavelength : float
        Line wavelength in Angstroms.

    Returns
    -------
    corrected_flux : ndarray
    """
    fact = ext.ccm89(np.array([wavelength]), 1.0, 3.1)
    return np.asarray(flux * 10 ** (0.4 * av * fact))


def redden(flux, av, wavelength):
    """Apply reddening (inverse of dustcorrect)."""
    fact = ext.ccm89(np.array([wavelength]), 1.0, 3.1)
    return np.asarray(flux / (10 ** (0.4 * av * fact)))


def correct_av_hybrid(av_balmer, hb_sn, av_gsw, sub=False):
    """Vectorized hybrid A_V correction (replaces old Python for-loop).

    When Hbeta S/N >= 10 and 0 <= A_V(Balmer) <= 3, use Balmer A_V.
    Otherwise, predict A_V from GSWLC SED A_V using a linear fit.

    Parameters
    ----------
    av_balmer : ndarray
        A_V from Balmer decrement.
    hb_sn : ndarray
        H-beta signal-to-noise ratio.
    av_gsw : ndarray
        A_V from GSWLC SED fitting.
    sub : bool
        Use sub-sample coefficients.

    Returns
    -------
    av_corrected : ndarray
    """
    av_balmer = np.asarray(av_balmer, dtype=np.float64)
    hb_sn = np.asarray(hb_sn, dtype=np.float64)
    av_gsw = np.asarray(av_gsw, dtype=np.float64)

    if sub:
        slope, intercept = AV_HYBRID_SLOPE_SUB, AV_HYBRID_INTERCEPT_SUB
    else:
        slope, intercept = AV_HYBRID_SLOPE, AV_HYBRID_INTERCEPT

    # Default: use predicted from SED A_V
    av_pred = np.clip(slope * av_gsw + intercept, 0.0, 3.0)

    # Use Balmer A_V where S/N is good
    use_balmer = hb_sn >= 10
    av_out = np.where(use_balmer, av_balmer, av_pred)

    # Clip Balmer values that fell outside [0, 3]
    bad_balmer = use_balmer & ((av_balmer < 0) | (av_balmer > 3))
    av_out = np.where(bad_balmer, np.clip(av_balmer, 0.0, 3.0), av_out)

    return av_out
