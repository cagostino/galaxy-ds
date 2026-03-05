"""Emission line analysis pipeline.

Replaces the monolithic ELObj class (1216 lines) with composable functions
that operate on pandas/polars DataFrames.

Ported from original ELObj.py (commit 83572ef).
"""

import numpy as np
import pandas as pd

from constants import LINE_WAVELENGTHS, XRAY_SFR_FACTORS
from dust import compute_av, dustcorrect, correct_av_hybrid
from classification import (
    get_bpt1_groups, get_bpt1_groups_ke01, get_bptplus_groups,
    get_bptplus_niigroups, get_bpt2_groups, get_bpt3_groups,
    get_whan_groups, get_ooo_groups, get_thom_dist,
)
from cosmology import get_lum_from_flux, halpha_to_sfr, delta_ssfr

# Line names → SDSS MPA-JHU column name stems
LINE_FLUX_MAP = {
    'halp':     'H_ALPHA',
    'hbeta':    'H_BETA',
    'oiii':     'OIII_5007',
    'oiii4959': 'OIII_4959',
    'oiii4363': 'OIII_4363',
    'oii':      'OII_3726',
    'nii':      'NII_6584',
    'oi':       'OI_6300',
    'sii6717':  'SII_6717',
    'sii6731':  'SII_6731',
    'neiii':    'NEIII_3869',
    'hgamma':   'H_GAMMA',
    'hdelta':   'H_DELTA',
}


def compute_sn(df, sncut=2):
    """Compute signal-to-noise ratios and classifiability flags.

    Parameters
    ----------
    df : DataFrame
        Must have {LINE}_FLUX and {LINE}_FLUX_ERR columns.
    sncut : float
        S/N threshold.

    Returns
    -------
    df : DataFrame (modified in place)
        Adds *_SN columns and filter booleans.
    """
    for short, stem in LINE_FLUX_MAP.items():
        flux_col = f'{stem}_FLUX'
        err_col = f'{stem}_FLUX_ERR'
        sn_col = f'{short}_sn'
        if flux_col in df.columns and err_col in df.columns:
            df[sn_col] = df[flux_col] / df[err_col]

    # Combined SII
    if 'SII_6717_FLUX' in df.columns and 'SII_6731_FLUX' in df.columns:
        df['sii_flux'] = df['SII_6717_FLUX'] + df['SII_6731_FLUX']
        df['sii_flux_err'] = np.sqrt(df['SII_6717_FLUX_ERR']**2 + df['SII_6731_FLUX_ERR']**2)
        df['sii_sn'] = df['sii_flux'] / df['sii_flux_err']

    # Line ratio S/N
    if 'halp_sn' in df.columns and 'nii_sn' in df.columns:
        df['halpnii_sn'] = 1.0 / np.sqrt(1.0 / df['halp_sn']**2 + 1.0 / df['nii_sn']**2)
    if 'hbeta_sn' in df.columns and 'oiii_sn' in df.columns:
        df['hbetaoiii_sn'] = 1.0 / np.sqrt(1.0 / df['hbeta_sn']**2 + 1.0 / df['oiii_sn']**2)

    # BPT classifiability (S/N > sncut for all 4 lines)
    df['bpt_sn_filt_bool'] = (
        (df.get('halp_sn', 0) > sncut) &
        (df.get('hbeta_sn', 0) > sncut) &
        (df.get('oiii_sn', 0) > sncut) &
        (df.get('nii_sn', 0) > sncut)
    )

    # NII/Ha only classifiable (has Ha+NII but not full BPT)
    df['halp_nii_filt_bool'] = (
        (df.get('halp_sn', 0) > sncut) &
        (df.get('nii_sn', 0) > sncut) &
        ((df.get('oiii_sn', 0) <= sncut) | (df.get('hbeta_sn', 0) <= sncut))
    )

    # Neither classifiable
    df['neither_filt_bool'] = ~(df['bpt_sn_filt_bool'] | df['halp_nii_filt_bool'])

    return df


def compute_line_ratios(df):
    """Compute BPT line ratios (log scale).

    Adds: niiha, siiha, oiha, oiiihb, xvals1_bpt, xvals2_bpt, xvals3_bpt, yvals_bpt
    """
    ha = df['H_ALPHA_FLUX']
    hb = df['H_BETA_FLUX']
    oiii = df['OIII_5007_FLUX']
    nii = df['NII_6584_FLUX']

    df['xvals1_bpt'] = nii / ha
    df['yvals_bpt'] = oiii / hb
    df['niiha'] = np.log10(df['xvals1_bpt'])
    df['oiiihb'] = np.log10(df['yvals_bpt'])

    if 'sii_flux' in df.columns:
        df['xvals2_bpt'] = df['sii_flux'] / ha
        df['siiha'] = np.log10(df['xvals2_bpt'])

    if 'OI_6300_FLUX' in df.columns:
        df['xvals3_bpt'] = df['OI_6300_FLUX'] / ha
        df['oiha'] = np.log10(df['xvals3_bpt'])

    df['thom_dist'] = get_thom_dist(df['niiha'].values, df['oiiihb'].values)

    # Ji+20 projections
    if 'siiha' in df.columns:
        df['ji_p1'] = df['niiha'] * 0.63 + df['siiha'] * 0.51 + df['oiiihb'] * 0.59
        df['ji_p2'] = -df['niiha'] * 0.63 + df['siiha'] * 0.78
        df['ji_p3'] = -df['niiha'] * 0.46 - df['siiha'] * 0.37 + 0.81 * df['oiiihb']

    return df


def compute_av_columns(df, av_gsw_col='av_gsw'):
    """Compute A_V from Balmer decrement + hybrid correction.

    Adds: av_sf, av_agn, corrected_av, corrected_av_sf
    """
    ha = df['H_ALPHA_FLUX'].values
    hb = df['H_BETA_FLUX'].values

    df['av_sf'] = compute_av(ha, hb, agn=False)
    df['av_agn'] = compute_av(ha, hb, agn=True)

    hb_sn = df['hbeta_sn'].values if 'hbeta_sn' in df.columns else np.zeros(len(df))

    if av_gsw_col in df.columns:
        av_gsw = df[av_gsw_col].values
        df['corrected_av'] = correct_av_hybrid(df['av_agn'].values, hb_sn, av_gsw)
        df['corrected_av_sf'] = correct_av_hybrid(df['av_sf'].values, hb_sn, av_gsw)
    else:
        df['corrected_av'] = df['av_agn']
        df['corrected_av_sf'] = df['av_sf']

    return df


def apply_dust_correction(df, av_col='corrected_av', suffix='_corr'):
    """Dust-correct all emission line fluxes using LINE_WAVELENGTHS.

    Parameters
    ----------
    df : DataFrame
    av_col : str
        Column containing A_V values.
    suffix : str
        Suffix to add to corrected column names.
    """
    av = df[av_col].values

    # Map of output column → (flux column, wavelength)
    corrections = {
        f'halpflux{suffix}':    ('H_ALPHA_FLUX',   LINE_WAVELENGTHS['H_ALPHA']),
        f'hbetaflux{suffix}':   ('H_BETA_FLUX',    LINE_WAVELENGTHS['H_BETA']),
        f'oiiiflux{suffix}':    ('OIII_5007_FLUX',  LINE_WAVELENGTHS['OIII_5007']),
        f'oiii_err{suffix}':    ('OIII_5007_FLUX_ERR', LINE_WAVELENGTHS['OIII_5007']),
        f'oiiflux{suffix}':     ('OII_3726_FLUX',   LINE_WAVELENGTHS['OII_3726']),
        f'niiflux{suffix}':     ('NII_6584_FLUX',   LINE_WAVELENGTHS['NII_6584']),
        f'oiflux{suffix}':      ('OI_6300_FLUX',    LINE_WAVELENGTHS['OI_6300']),
        f'siiflux{suffix}':     ('sii_flux',        LINE_WAVELENGTHS['SII']),
        f'sii6717flux{suffix}': ('SII_6717_FLUX',   LINE_WAVELENGTHS['SII_6717']),
        f'sii6731flux{suffix}': ('SII_6731_FLUX',   LINE_WAVELENGTHS['SII_6731']),
    }

    for out_col, (flux_col, wl) in corrections.items():
        if flux_col in df.columns:
            df[out_col] = dustcorrect(df[flux_col].values, av, wl).ravel()

    return df


def compute_luminosities(df, z_col='z', av_col='corrected_av'):
    """Compute OIII and H-alpha luminosities from dust-corrected fluxes.

    Adds: oiiilum, halplum, oiiilum_uncorr, halplum_uncorr, halpfibsfr
    """
    z = df[z_col].values

    if 'oiiiflux_corr' in df.columns:
        oiii_lum = get_lum_from_flux(df['oiiiflux_corr'].values, z)
        df['oiiilum'] = np.log10(np.clip(oiii_lum, 1e-99, None))

        if 'oiii_err_corr' in df.columns:
            oiii_up = get_lum_from_flux(df['oiiiflux_corr'].values + df['oiii_err_corr'].values, z)
            oiii_down = get_lum_from_flux(df['oiiiflux_corr'].values - df['oiii_err_corr'].values, z)
            df['oiiilum_up'] = np.log10(np.clip(oiii_up, 1e-99, None))
            df['oiiilum_down'] = np.log10(np.clip(oiii_down, 1e-99, None))
            df['e_oiiilum_up'] = df['oiiilum_up'] - df['oiiilum']
            df['e_oiiilum_down'] = df['oiiilum'] - df['oiiilum_down']

    if 'halpflux_corr' in df.columns:
        halp_lum = get_lum_from_flux(df['halpflux_corr'].values, z)
        df['halplum'] = np.log10(np.clip(halp_lum, 1e-99, None))
        df['halpfibsfr'] = halpha_to_sfr(10 ** df['halplum'].values)

    # Uncorrected luminosities
    if 'OIII_5007_FLUX' in df.columns:
        df['oiiilum_uncorr'] = np.log10(np.clip(
            get_lum_from_flux(df['OIII_5007_FLUX'].values, z), 1e-99, None))
    if 'H_ALPHA_FLUX' in df.columns:
        df['halplum_uncorr'] = np.log10(np.clip(
            get_lum_from_flux(df['H_ALPHA_FLUX'].values, z), 1e-99, None))

    return df


def classify_bpt(df):
    """Apply all BPT classification schemes.

    Adds columns: bptgroups, bptplus_groups, bptplus_niigroups,
                   bpt1_ke01_groups, bpt2_groups, bpt3_groups, whan_groups
    """
    niiha = df['niiha'].values
    oiiihb = df['oiiihb'].values

    df['bptgroups'] = get_bpt1_groups(niiha, oiiihb)
    df['bptplus_groups'] = get_bptplus_groups(niiha, oiiihb)
    df['bptplus_niigroups'] = get_bptplus_niigroups(niiha)
    df['bpt1_ke01_groups'] = get_bpt1_groups_ke01(niiha, oiiihb)

    if 'siiha' in df.columns:
        df['bpt2_groups'] = get_bpt2_groups(df['siiha'].values, oiiihb)

    if 'oiha' in df.columns:
        df['bpt3_groups'] = get_bpt3_groups(df['oiha'].values, oiiihb)

    if 'H_ALPHA_EQW' in df.columns:
        df['whan_groups'] = get_whan_groups(niiha, df['H_ALPHA_EQW'].values)

    return df


def compute_derived_quantities(df):
    """Compute additional derived columns (mass fractions, sSFR, etc.)."""
    if 'logSFR' in df.columns and 'logMstar' in df.columns:
        df['ssfr'] = df['logSFR'] - df['logMstar']
        df['delta_ssfr'] = delta_ssfr(df['logMstar'].values, df['ssfr'].values)

    # Velocity dispersion derived
    if 'V_DISP' in df.columns:
        vdisp = df['V_DISP'].values
        df['mbh'] = np.log10(3 * (vdisp / 200) ** 4 * 1e8)
        df['edd_lum'] = np.log10(3e8 * 1.38e38 * (vdisp / 200) ** 4)

    return df


def build_emission_line_catalog(sdss_df, gsw_df=None, av_gsw_col='av_gsw',
                                z_col='z', sncut=2):
    """Full pipeline: S/N → ratios → A_V → dust correction → luminosities → BPT.

    Parameters
    ----------
    sdss_df : pandas DataFrame
        Merged SDSS + GSW data with emission line fluxes.
    gsw_df : polars DataFrame or None
        GSWLC data (merged in if provided).
    av_gsw_col : str
        Column name for GSWLC SED A_V.
    z_col : str
        Redshift column name.
    sncut : float
        S/N threshold for BPT classifiability.

    Returns
    -------
    pandas DataFrame with all derived columns.
    """
    df = sdss_df.copy()

    df = compute_sn(df, sncut=sncut)
    df = compute_line_ratios(df)
    df = compute_av_columns(df, av_gsw_col=av_gsw_col)

    # Dust correction with hybrid A_V (AGN assumption)
    df = apply_dust_correction(df, av_col='corrected_av', suffix='_corr')
    # Dust correction with SF assumption
    df = apply_dust_correction(df, av_col='corrected_av_sf', suffix='_corr_sf')
    # Dust correction with raw Balmer A_V (no hybrid)
    df = apply_dust_correction(df, av_col='av_sf', suffix='_corr_p1')

    df = compute_luminosities(df, z_col=z_col)
    df = compute_derived_quantities(df)
    df = classify_bpt(df)

    return df
