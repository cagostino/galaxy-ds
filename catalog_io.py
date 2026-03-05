"""Catalog I/O: load FITS, GSWLC, Agostino2021, SDSS MPA-JHU; coordinate matching.

Uses polars for fast I/O with parquet caching. Falls back to pandas where needed
(e.g., astropy FITS → pandas → polars conversion).
"""

import os
import numpy as np
import polars as pl
from pathlib import Path

CATFOLD = 'catalogs/'


# ---------------------------------------------------------------------------
# Generic loaders
# ---------------------------------------------------------------------------

def load_fits_to_polars(filename):
    """Load a FITS binary table to a polars DataFrame.

    Multi-dimensional columns are exploded into separate columns.
    """
    from astropy.table import Table

    t = Table.read(filename)
    # Explode multi-dim columns
    to_add = []
    to_remove = []
    for col in t.colnames:
        if t[col].ndim > 1:
            for i in range(t[col].shape[1]):
                to_add.append((f"{col}_{i}", t[col][:, i]))
            to_remove.append(col)
    for name, data in to_add:
        t[name] = data
    for col in to_remove:
        t.remove_column(col)

    pdf = t.to_pandas()
    return pl.from_pandas(pdf)


def load_tsv_to_polars(filename):
    return pl.read_csv(filename, separator='\t')


# ---------------------------------------------------------------------------
# GSWLC-M2
# ---------------------------------------------------------------------------

GSWLC_M2_COLUMNS = [
    'ObjID', 'GLXID', 'plate', 'MJD', 'fiber_ID',
    'RA', 'Decl', 'z', 'chi2', 'logMstar', 'logMstar_err',
    'logSFR', 'logSFR_err', 'A_FUV', 'A_FUV_err',
    'A_B', 'A_B_err', 'av_gsw', 'av_gsw_err', 'flag_sed',
    'uv_survey', 'flag_uv', 'flag_midir', 'flag_mgs',
]


def load_gswlc_m2(filename=None, cache=True):
    """Load GSWLC-M2 whitespace-delimited file → polars DataFrame.

    Parameters
    ----------
    filename : str or None
        Path to GSWLC-M2.dat. Defaults to catalogs/GSWLC-M2.dat.
    cache : bool
        If True, write/read a parquet cache for faster subsequent loads.

    Returns
    -------
    pl.DataFrame
    """
    if filename is None:
        filename = os.path.join(CATFOLD, 'GSWLC-M2.dat')

    parquet_path = Path(filename).with_suffix('.parquet')

    if cache and parquet_path.exists():
        return pl.read_parquet(str(parquet_path))

    # Read line-by-line to preserve 19-digit integer precision
    # (np.loadtxt would cast to float64, losing last ~3 digits of ObjID/GLXID)
    int_cols = {'ObjID', 'GLXID', 'plate', 'MJD', 'fiber_ID',
                'flag_sed', 'uv_survey', 'flag_uv', 'flag_midir', 'flag_mgs'}
    data = {col: [] for col in GSWLC_M2_COLUMNS}
    with open(filename) as f:
        for line in f:
            vals = line.split()
            for i, col in enumerate(GSWLC_M2_COLUMNS):
                if col in int_cols:
                    data[col].append(int(vals[i]))
                else:
                    data[col].append(float(vals[i]))

    df = pl.DataFrame({
        col: (pl.Series(col, data[col], dtype=pl.Int64) if col in int_cols
              else pl.Series(col, data[col], dtype=pl.Float64))
        for col in GSWLC_M2_COLUMNS
    })

    if cache:
        df.write_parquet(str(parquet_path))

    return df


# ---------------------------------------------------------------------------
# Agostino 2021 classifications
# ---------------------------------------------------------------------------

def load_agostino2021(filename=None):
    """Load Agostino+2021 emission-line classifications.

    Returns
    -------
    pl.DataFrame with columns: SDSS_ObjID, gen_el_class, sl_class1, sl_class2
    """
    if filename is None:
        filename = os.path.join(CATFOLD, 'agostino2021_table1.csv')
    return pl.read_csv(filename)


# ---------------------------------------------------------------------------
# SDSS MPA-JHU catalogs
# ---------------------------------------------------------------------------

def load_sdss_mpa_jhu(catfold=None, cache=True):
    """Load and merge the SDSS MPA-JHU DR7 FITS catalogs.

    Expects files in catfold:
        gal_info_dr7_v5_2.fit
        gal_line_dr7_v5_2.fit
        gal_indx_dr7_v5_2.fit
        gal_fiboh_dr7_v5_2.fits
        totlgm_dr7_v5_2b.fit
        fiblgm_dr7_v5_2.fit
        gal_fibsfr_dr7_v5_2.fits
        gal_fibspecsfr_dr7_v5_2.fits

    Returns
    -------
    pl.DataFrame
    """
    if catfold is None:
        catfold = CATFOLD

    parquet_path = os.path.join(catfold, 'sdss_mpa_jhu_merged.parquet')
    if cache and os.path.exists(parquet_path):
        return pl.read_parquet(parquet_path)

    files = {
        'gal_info': 'gal_info_dr7_v5_2.fit',
        'gal_line': 'gal_line_dr7_v5_2.fit',
        'gal_indx': 'gal_indx_dr7_v5_2.fit',
        'gal_fiboh': 'gal_fiboh_dr7_v5_2.fits',
        'totlgm': 'totlgm_dr7_v5_2b.fit',
        'fiblgm': 'fiblgm_dr7_v5_2.fit',
        'fibsfr': 'gal_fibsfr_dr7_v5_2.fits',
        'fibssfr': 'gal_fibspecsfr_dr7_v5_2.fits',
    }

    dfs = {}
    for key, fname in files.items():
        path = os.path.join(catfold, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing SDSS file: {path}")
        dfs[key] = load_fits_to_polars(path)

    # All MPA-JHU tables are row-aligned (same order), so horizontal concat
    # But we need to handle duplicate column names from the merge
    merged = dfs['gal_info']
    for key in ['gal_line', 'gal_indx', 'gal_fiboh', 'totlgm', 'fiblgm', 'fibsfr', 'fibssfr']:
        right = dfs[key]
        # Drop columns that already exist in merged
        existing = set(merged.columns)
        keep_cols = [c for c in right.columns if c not in existing]
        if keep_cols:
            merged = pl.concat([merged, right.select(keep_cols)], how='horizontal')

    if cache:
        merged.write_parquet(parquet_path)

    return merged


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------

def match_by_id(left, right, left_on, right_on, how='inner', suffix='_right'):
    """Join two polars DataFrames on ID columns.

    Parameters
    ----------
    left, right : pl.DataFrame
    left_on, right_on : str or list of str
    how : str
        Join type.
    suffix : str
        Suffix for duplicate columns from right.

    Returns
    -------
    pl.DataFrame
    """
    return left.join(right, left_on=left_on, right_on=right_on, how=how, suffix=suffix)


def match_by_coords(left, right, ra_left, dec_left, ra_right, dec_right,
                    max_sep_arcsec=3.0):
    """Cross-match two catalogs by sky coordinates.

    Parameters
    ----------
    left, right : pl.DataFrame
    ra_left, dec_left : str
        Column names for RA/Dec in left.
    ra_right, dec_right : str
        Column names for RA/Dec in right.
    max_sep_arcsec : float
        Maximum separation in arcseconds.

    Returns
    -------
    pl.DataFrame
        Matched rows with columns from both, plus 'sep_arcsec'.
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    c_left = SkyCoord(
        ra=left[ra_left].to_numpy() * u.deg,
        dec=left[dec_left].to_numpy() * u.deg,
    )
    c_right = SkyCoord(
        ra=right[ra_right].to_numpy() * u.deg,
        dec=right[dec_right].to_numpy() * u.deg,
    )

    idx, sep2d, _ = c_left.match_to_catalog_sky(c_right)
    sep_arcsec = sep2d.arcsec

    mask = sep_arcsec <= max_sep_arcsec
    left_matched = left.filter(pl.Series(mask))
    right_matched = right[idx[mask]]

    # Rename right columns that clash
    right_cols = {}
    for c in right_matched.columns:
        if c in left_matched.columns:
            right_cols[c] = f"{c}_right"
    if right_cols:
        right_matched = right_matched.rename(right_cols)

    result = pl.concat([left_matched, right_matched], how='horizontal')
    result = result.with_columns(pl.Series('sep_arcsec', sep_arcsec[mask]))
    return result


# ---------------------------------------------------------------------------
# Parquet persistence
# ---------------------------------------------------------------------------

def save_parquet(df, path):
    """Save a polars DataFrame to parquet."""
    if isinstance(df, pl.DataFrame):
        df.write_parquet(path)
    else:
        # pandas fallback
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            pl.from_pandas(df).write_parquet(path)


def load_parquet(path):
    """Load a parquet file to polars DataFrame."""
    return pl.read_parquet(path)
