"""Single source of truth for wavelengths, cosmology constants, and column definitions."""

import astropy.cosmology as apc

COSMO = apc.Planck15

# Correct wavelengths in Angstroms — from original ELObj.py (commit 83572ef)
LINE_WAVELENGTHS = {
    'H_ALPHA': 6563.0,
    'H_BETA': 4861.0,
    'OIII_5007': 5007.0,
    'OIII_4959': 4959.0,
    'OIII_4363': 4363.0,
    'OII_3726': 3727.0,
    'NII_6584': 6583.0,
    'OI_6300': 6300.0,
    'SII': 6724.0,
    'SII_6717': 6717.0,
    'SII_6731': 6731.0,
    'H_GAMMA': 4340.0,
    'H_DELTA': 4102.0,
    'NEIII_3869': 3869.0,
}

# Intrinsic Balmer decrements
BALMER_DEC_SF = 2.86
BALMER_DEC_AGN = 3.1

# Agostino+21 NII/Ha cut for least-contaminated SF sample
NII_HA_AGN_CUT = -0.35

# Old Stasinska value (kept for reference / backward compat)
NII_HA_AGN_CUT_OLD = -0.4

# Hybrid A_V correction coefficients (Agostino+19/21)
AV_HYBRID_SLOPE = 1.78281143
AV_HYBRID_INTERCEPT = 0.2085823945997035
AV_HYBRID_SLOPE_SUB = 1.43431072
AV_HYBRID_INTERCEPT_SUB = 0.34834144657321453

# X-ray SFR conversion factors (Ranalli+ 2003)
XRAY_SFR_FACTORS = {
    'soft': 1 / 1.39e-40,
    'hard': 1 / 1.26e-40,
    'full': 1 / 0.66e-40,
}

# Delta-sSFR main sequence fit (Agostino+19)
DELTA_SSFR_SLOPE = -0.4597
DELTA_SSFR_INTERCEPT = -5.2976

# GSWLC-M2 column names (24 whitespace-delimited columns)
# From official Table 1 (Salim et al. 2018) — no W1/W2 magnitudes in this file
GSWLC_M2_COLUMNS = [
    'ObjID', 'GLXID', 'plate', 'MJD', 'fiber_ID',
    'RA', 'Decl', 'z', 'chi2', 'logMstar', 'logMstar_err',
    'logSFR', 'logSFR_err', 'A_FUV', 'A_FUV_err',
    'A_B', 'A_B_err', 'av_gsw', 'av_gsw_err', 'flag_sed',
    'uv_survey', 'flag_uv', 'flag_midir', 'flag_mgs',
]
