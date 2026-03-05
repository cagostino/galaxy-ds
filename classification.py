"""BPT/WHAN demarcation lines and galaxy classification.

Absorbs old demarcations.py + classification functions from ELObj.py (commit 83572ef).
All classification functions are vectorized with numpy (no Python for-loops).
"""

import numpy as np
from constants import NII_HA_AGN_CUT

# ---------------------------------------------------------------------------
# Demarcation line functions
# ---------------------------------------------------------------------------

def y1_kauffmann(x):
    """Kauffmann+03 BPT1 SF/composite boundary."""
    return 10 ** (0.61 / (x - 0.05) + 1.3)

def y1_kewley(x):
    """Kewley+01 BPT1 composite/AGN boundary."""
    return 10 ** (0.61 / (x - 0.47) + 1.19)

def y1_schawinski(x):
    """Schawinski+07 Sy2/LINER boundary on BPT1."""
    return 10 ** (1.05 * x + 0.45)

def y2_agn(x):
    """Kewley+06 BPT2 (SII/Ha vs OIII/Hb) SF/AGN boundary."""
    return 10 ** (0.72 / (x - 0.32) + 1.30)

def y2_linersy2(x):
    """Kewley+06 BPT2 Sy2/LINER boundary."""
    return 10 ** (1.89 * x + 0.76)

def y3_agn(x):
    """Kewley+06 BPT3 (OI/Ha vs OIII/Hb) SF/AGN boundary."""
    return 10 ** (0.73 / (x + 0.59) + 1.33)

def y3_linersy2(x):
    """Kewley+06 BPT3 Sy2/LINER boundary."""
    return 10 ** (1.18 * x + 1.30)

def ooo_agn(x):
    return 10 ** (-1.701 * x - 2.163)

def ooo_linersy2(x):
    return 10 ** (1.0 * x + 0.7)

def mex_main(x):
    return 0.375 / (x - 10.5) + 1.14

def mex_upper(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.where(x < 10,
                   mex_main(x),
                   410.24 - 109.333 * x + 9.71731 * x**2 - 0.288244 * x**3)
    return out

def mex_lower(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.where(x < 10,
                   mex_main(x),
                   352.066 - 93.8249 * x + 8.32651 * x**2 - 0.246416 * x**3)
    return out

def y_stasinska(x):
    return 10 ** ((-30.787 + 1.1358 * x + 0.27297 * x**2) * np.tanh(5.7409 * x) - 31.093)


# ---------------------------------------------------------------------------
# Pre-computed demarcation arrays (for plotting)
# ---------------------------------------------------------------------------

# BPT1 — x in log(NII/Ha), converted back to linear for plotting
_xlog_ke = np.log10(np.logspace(-2.5, 1, num=100))
_xlog_ka = np.log10(np.logspace(np.log10(0.007), 0, num=100))
_xlog_ka_plus = np.log10(np.logspace(np.log10(0.007), NII_HA_AGN_CUT, num=100))

DEMLINES = {}
DEMLINES['bpt1_kewley_y'] = y1_kewley(_xlog_ke)
DEMLINES['bpt1_kewley_x'] = 10 ** _xlog_ke
DEMLINES['bpt1_kauffmann_y'] = y1_kauffmann(_xlog_ka)
DEMLINES['bpt1_kauffmann_x'] = 10 ** _xlog_ka
DEMLINES['bpt1_kauffmann_plus_y'] = y1_kauffmann(_xlog_ka_plus)
DEMLINES['bpt1_kauffmann_plus_x'] = 10 ** _xlog_ka_plus
DEMLINES['bpt1_stasinska_y'] = y_stasinska(_xlog_ka)
DEMLINES['bpt1_stasinska_x'] = 10 ** _xlog_ka

# BPT2 — x in log(SII/Ha)
_xlog_b2a = np.log10(np.logspace(-1.5, 0.1, num=100))
_xlog_b2l = np.log10(np.logspace(-0.3, 0.2, num=100))
DEMLINES['bpt2_agn_y'] = y2_agn(_xlog_b2a)
DEMLINES['bpt2_agn_x'] = 10 ** _xlog_b2a
DEMLINES['bpt2_linersy2_y'] = y2_linersy2(_xlog_b2l)
DEMLINES['bpt2_linersy2_x'] = 10 ** _xlog_b2l

# BPT3 — x in log(OI/Ha)
_xlog_b3a = np.log10(np.logspace(-2.5, -0.8, num=100))
_xlog_b3l = np.log10(np.logspace(-1.1, -0.2, num=100))
DEMLINES['bpt3_agn_y'] = y3_agn(_xlog_b3a)
DEMLINES['bpt3_agn_x'] = 10 ** _xlog_b3a
DEMLINES['bpt3_linersy2_y'] = y3_linersy2(_xlog_b3l)
DEMLINES['bpt3_linersy2_x'] = 10 ** _xlog_b3l

# MEX
DEMLINES['mex_x'] = np.linspace(8, 12, 100)
DEMLINES['mex_upper_y'] = mex_upper(DEMLINES['mex_x'])
DEMLINES['mex_lower_y'] = mex_lower(DEMLINES['mex_x'])


# ---------------------------------------------------------------------------
# Distance metrics (Agostino+19)
# ---------------------------------------------------------------------------

_AGN_BRANCH_VEC = np.array([-1.1643, -0.1543])
_PERP_AGN_BRANCH_VEC = np.array([0.86, 1.5])


def get_perpdist(x, y):
    """Perpendicular distance from the AGN branch."""
    top = np.abs(_PERP_AGN_BRANCH_VEC[0] * x + y + _PERP_AGN_BRANCH_VEC[1])
    bot = np.sqrt(_PERP_AGN_BRANCH_VEC[0] ** 2 + 1)
    return top / bot


def get_pardist(x, y):
    """Parallel distance along the AGN branch."""
    top = _AGN_BRANCH_VEC[0] * x + y + _AGN_BRANCH_VEC[1]
    bot = np.sqrt(_AGN_BRANCH_VEC[0] ** 2 + 1)
    return top / bot


def get_thom_dist(x, y):
    """Projected distance from the Kewley+13 mixing sequence endpoint."""
    slope = (-0.408 - 0.979) / (-0.466 - 0.003)
    ke_x, ke_y = -0.22321, 0.310
    top = -slope * (x - ke_x) - ke_y + y
    bot = np.sqrt(slope ** 2 + 1)
    perp = np.abs(top) / bot
    d_obj = np.sqrt((x - ke_x) ** 2 + (y - ke_y) ** 2)
    theta = np.arcsin(np.clip(perp / d_obj, -1, 1))
    proj_d = d_obj * np.cos(theta)
    slope_inv = -0.327
    top_inv = -slope_inv * (x - ke_x) - ke_y + y
    return np.sign(top_inv) * proj_d


# ---------------------------------------------------------------------------
# Vectorized classification functions
# ---------------------------------------------------------------------------

def get_bpt1_groups(niiha, oiiihb):
    """BPT1 classification: HII vs AGN using Kauffmann+03.

    Parameters
    ----------
    niiha : ndarray
        log10(NII/Halpha).
    oiiihb : ndarray
        log10(OIII/Hbeta).

    Returns
    -------
    groups : ndarray of str
        'HII', 'AGN', or 'NO'.
    """
    x = np.asarray(niiha, dtype=np.float64)
    y = np.asarray(oiiihb, dtype=np.float64)
    groups = np.full(len(x), 'NO', dtype='U6')

    valid = np.isfinite(x) & np.isfinite(y)
    below_ka = y < np.log10(y1_kauffmann(x))
    sf_mask = valid & (x < 0) & below_ka
    groups[sf_mask] = 'HII'
    groups[valid & ~sf_mask] = 'AGN'
    groups[~valid] = 'NO'
    return groups


def get_bpt1_groups_ke01(niiha, oiiihb):
    """BPT1 with Kewley+01 + Schawinski: HII / COMP / Sy2 / LINER.

    Parameters
    ----------
    niiha, oiiihb : ndarray
        log10 line ratios.

    Returns
    -------
    groups : ndarray of str
    """
    x = np.asarray(niiha, dtype=np.float64)
    y = np.asarray(oiiihb, dtype=np.float64)
    n = len(x)
    groups = np.full(n, 'LINER', dtype='U6')

    ka_line = np.log10(y1_kauffmann(x))
    ke_line = np.log10(y1_kewley(x))
    sch_line = np.log10(y1_schawinski(x))

    # SF: below Kauffmann and x < 0
    sf = (x < 0) & (y < ka_line)
    groups[sf] = 'HII'

    # Composite: between Kauffmann and Kewley
    comp = ~sf & (((x < 0) & (y >= ka_line) & (y < ke_line)) |
                  ((x >= 0) & (x < 0.43) & (y < ke_line)))
    groups[comp] = 'COMP'

    # Sy2: above Schawinski line and above Kewley (or x >= 0.43)
    above_ke = (y >= ke_line) | (x >= 0.43)
    sy2 = ~sf & ~comp & above_ke & (y > sch_line)
    groups[sy2] = 'Sy2'
    # LINER is the default for everything else above Kewley

    return groups


def get_bptplus_groups(niiha, oiiihb, nii_cut=NII_HA_AGN_CUT):
    """BPT+ classification: HII vs AGN using Kauffmann+03 + NII/Ha cut.

    Uses the Agostino+21 cut at log(NII/Ha) = -0.35 by default.

    Parameters
    ----------
    niiha, oiiihb : ndarray
        log10 line ratios.
    nii_cut : float
        NII/Ha boundary (default -0.35).

    Returns
    -------
    groups : ndarray of str
        'HII', 'AGN', or 'NO'.
    """
    x = np.asarray(niiha, dtype=np.float64)
    y = np.asarray(oiiihb, dtype=np.float64)
    groups = np.full(len(x), 'NO', dtype='U4')

    valid = np.isfinite(x) & np.isfinite(y)
    ka_line = np.log10(y1_kauffmann(x))
    sf = valid & (x < nii_cut) & (y < ka_line)
    groups[sf] = 'HII'
    groups[valid & ~sf] = 'AGN'
    return groups


def get_bptplus_niigroups(niiha, nii_cut=NII_HA_AGN_CUT):
    """NII/Ha-only classification (for objects without full BPT).

    Parameters
    ----------
    niiha : ndarray
        log10(NII/Halpha).
    nii_cut : float
        Boundary (default -0.35).

    Returns
    -------
    groups : ndarray of str
    """
    x = np.asarray(niiha, dtype=np.float64)
    groups = np.full(len(x), 'NO', dtype='U4')
    valid = np.isfinite(x)
    groups[valid & (x < nii_cut)] = 'HII'
    groups[valid & (x >= nii_cut)] = 'AGN'
    return groups


def get_bpt2_groups(siiha, oiiihb):
    """BPT2 (SII/Ha): HII / Seyfert / LINER."""
    x = np.asarray(siiha, dtype=np.float64)
    y = np.asarray(oiiihb, dtype=np.float64)
    groups = np.full(len(x), 'LINER', dtype='U7')

    agn_line = np.log10(y2_agn(x))
    ls_line = np.log10(y2_linersy2(x))

    sf = (x < 0.32) & (y < agn_line)
    groups[sf] = 'HII'
    groups[~sf & (y > ls_line)] = 'Seyfert'
    return groups


def get_bpt3_groups(oiha, oiiihb):
    """BPT3 (OI/Ha): HII / Seyfert / LINER."""
    x = np.asarray(oiha, dtype=np.float64)
    y = np.asarray(oiiihb, dtype=np.float64)
    groups = np.full(len(x), 'LINER', dtype='U7')

    valid = np.isfinite(x) & np.isfinite(y)
    agn_line = np.log10(y3_agn(x))
    ls_line = np.log10(y3_linersy2(x))

    sf = valid & (x < -0.59) & (y < agn_line)
    groups[sf] = 'HII'
    groups[valid & ~sf & (y > ls_line)] = 'Seyfert'
    groups[~valid] = 'NO'
    return groups


def get_whan_groups(niiha, halpha_eqw):
    """WHAN diagram: SF / sAGN / wAGN / RG / PG.

    Parameters
    ----------
    niiha : ndarray
        log10(NII/Halpha).
    halpha_eqw : ndarray
        H-alpha equivalent width (Angstroms).

    Returns
    -------
    groups : ndarray of str
    """
    x = np.asarray(niiha, dtype=np.float64)
    y = np.asarray(halpha_eqw, dtype=np.float64)
    groups = np.full(len(x), 'NO', dtype='U4')

    valid = np.isfinite(x)
    pg = valid & (y < 0.5)
    rg = valid & (y >= 0.5) & (y < 3)
    wagn = valid & (y >= 3) & (y < 6) & (x > -0.4)
    sagn = valid & (y >= 6) & (x > -0.4)
    sf = valid & (y >= 3) & (x <= -0.4)

    groups[pg] = 'PG'
    groups[rg] = 'RG'
    groups[wagn] = 'wAGN'
    groups[sagn] = 'sAGN'
    groups[sf] = 'SF'
    return groups


def get_ooo_groups(x_ratio, y_ratio):
    """OOO diagram classification: HII / Seyfert / LINER."""
    x = np.asarray(x_ratio, dtype=np.float64)
    y = np.asarray(y_ratio, dtype=np.float64)
    groups = np.full(len(x), 'LINER', dtype='U7')

    valid = np.isfinite(x) & np.isfinite(y)
    agn_line = np.log10(ooo_agn(x))
    ls_line = np.log10(ooo_linersy2(x))

    sf = valid & (y < agn_line)
    groups[sf] = 'HII'
    groups[valid & ~sf & (y > ls_line)] = 'Seyfert'
    groups[~valid] = 'NO'
    return groups
