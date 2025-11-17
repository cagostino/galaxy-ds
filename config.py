"""
Configuration management for galaxy data science pipeline.

This module provides centralized configuration for paths, constants,
and default parameters used throughout the pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any
import logging

# Base paths
BASE_DIR = Path(__file__).parent
CATALOG_DIR = BASE_DIR / 'catalogs'
OUTPUT_DIR = BASE_DIR / 'output'
LOG_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for directory in [CATALOG_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

# Database configuration
DATABASE_PATH = CATALOG_DIR / 'catalog_database.db'
DATABASE_CHUNK_SIZE = 10000  # Rows per insert batch

# Catalog file names (relative to CATALOG_DIR)
CATALOG_FILES = {
    # SDSS MPA/JHU catalogs
    'sdss_info': 'gal_info_dr7_v5_2.fit',
    'sdss_line': 'gal_line_dr7_v5_2.fit',
    'sdss_indx': 'gal_indx_dr7_v5_2.fit',
    'sdss_fiboh': 'gal_fiboh_dr7_v5_2.fits',
    'sdss_totmass': 'totlgm_dr7_v5_2b.fit',
    'sdss_fibmass': 'fiblgm_dr7_v5_2.fit',
    'sdss_fibsfr': 'gal_fibsfr_dr7_v5_2.fits',
    'sdss_fibssfr': 'gal_fibspecsfr_dr7_v5_2.fits',

    # GSWLC catalogs
    'gswlc_m2': 'GSWLC-M2.dat',
    'gswlc_a2': 'GSWLC-A2.dat',
    'gswlc_x2': 'GSWLC-X2.dat',
    'gswlc_d2': 'GSWLC-D2.dat',

    # X-ray catalogs
    'xmm_4xmm': '4xmm.fits',
    'xmm_4xmmobs': '4xmmobs.tsv',
    'xmm_3xmm': '3xmm.fits',
    'xmm_3xmmobs': '3xmmobs.fits',
    'chandra_csc': 'merged_csc.csv',

    # Additional GSWLC data files
    'gswlc_sigma1': 'sigma1_mis.dat',
    'gswlc_env_nyu': 'envir_nyu_mis.dat',
    'gswlc_baldry': 'baldry_mis.dat',
    'gswlc_irexcess': 'irexcess_mis.dat',
    'gswlc_simard': 'simard_ellip_mis.dat',
}

# Emission lines dictionary (wavelength in Angstroms)
EMISSION_LINES = {
    'H_ALPHA': 6562.8,
    'H_BETA': 4861.0,
    'H_GAMMA': 4340.0,
    'H_DELTA': 4101.0,
    'OI_6300': 6300.0,
    'OII_3726': 3727.0,
    'OII_3729': 3729.0,
    'OIII_4363': 4363.0,
    'OIII_4959': 4959.0,
    'OIII_5007': 5007.0,
    'NII_6548': 6548.0,
    'NII_6584': 6583.0,
    'SII_6717': 6717.0,
    'SII_6731': 6731.0,
    'NEIII_3869': 3869.0,
    'SII': 6724.0  # Combined doublet
}

# Analysis parameters
ANALYSIS_PARAMS = {
    # Signal-to-noise cuts
    'line_sncut': 2.0,
    'bpt_sncut': 2.0,

    # Coordinate matching
    'coord_match_threshold': 7.0,  # arcseconds

    # Dust correction models
    'dust_models': ['a19', 'a21'],
    'default_dust_model': 'a19',
    'default_dec_rat': 3.1,  # Declination ratio for AGN

    # Redshift range for analysis
    'z_min': 0.01,
    'z_max': 0.3,

    # X-ray analysis
    'xray_texp_min': 10000,  # seconds
    'xray_texp_max': 50000,  # seconds
    'xray_bands': ['soft', 'hard', 'full'],

    # GSWLC flags
    'gswlc_sedflag_good': 0,
    'gswlc_sedflag_qso': 1,
}

# Cosmology parameters (using Planck15 by default)
COSMOLOGY_PARAMS = {
    'H0': 67.7,  # km/s/Mpc
    'Om0': 0.307,
    'Ode0': 0.693,
    'Tcmb0': 2.725,  # K
}

# Logging configuration
LOG_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'filename': LOG_DIR / 'pipeline.log',
    'filemode': 'a',
}

# Merge column mappings
MERGE_COLUMNS = {
    'gswlc_sdss': {
        'left_on': ['plate', 'MJD', 'fiber_ID'],
        'right_on': ['PLATEID', 'MJD', 'FIBERID'],
    },
}

# SQL table schemas (for validation)
SQL_TABLE_SCHEMAS = {
    'gsw_m2_sdss': {
        'required_columns': ['ObjID', 'z', 'mass', 'sfr', 'bpt_sn_filt_bool', 'bptclass'],
        'index_columns': ['sdss_gsw_index', 'ObjID'],
    },
    'xmm4_gsw_sdss': {
        'required_columns': ['sourceids', 'fullflux', 'texp'],
        'index_columns': ['xmm_sdss_gsw_index'],
    },
}

# Cache settings
CACHE_CONFIG = {
    'enable_caching': True,
    'cache_dir': CATALOG_DIR,
    'force_recompute': False,  # Set to True to ignore all caches
}

# Diagnostic settings
DIAGNOSTIC_CONFIG = {
    'validation_report': OUTPUT_DIR / 'validation_report.txt',
    'discrepancy_csv': OUTPUT_DIR / 'pipeline_discrepancies.csv',
    'comparison_log': LOG_DIR / 'sql_vs_pandas.log',
}

# BPT diagram boundaries (for reference)
BPT_BOUNDARIES = {
    'kauffmann03': lambda n2ha: 0.61 / (n2ha - 0.05) + 1.3,
    'kewley01': lambda n2ha: 0.61 / (n2ha - 0.47) + 1.19,
}


def get_catalog_path(catalog_key: str) -> Path:
    """
    Get full path to a catalog file.

    Parameters
    ----------
    catalog_key : str
        Key from CATALOG_FILES dictionary

    Returns
    -------
    Path
        Full path to catalog file

    Raises
    ------
    KeyError
        If catalog_key not found
    FileNotFoundError
        If catalog file doesn't exist
    """
    if catalog_key not in CATALOG_FILES:
        raise KeyError(f"Unknown catalog key: {catalog_key}")

    path = CATALOG_DIR / CATALOG_FILES[catalog_key]

    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")

    return path


def get_param(param_key: str, default: Any = None) -> Any:
    """
    Get analysis parameter with optional default.

    Parameters
    ----------
    param_key : str
        Key from ANALYSIS_PARAMS dictionary
    default : Any, optional
        Default value if key not found

    Returns
    -------
    Any
        Parameter value
    """
    return ANALYSIS_PARAMS.get(param_key, default)


def update_params(new_params: Dict[str, Any]) -> None:
    """
    Update analysis parameters.

    Parameters
    ----------
    new_params : dict
        Dictionary of parameters to update
    """
    ANALYSIS_PARAMS.update(new_params)


def setup_logging() -> logging.Logger:
    """
    Setup logging configuration.

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logging.basicConfig(
        level=LOG_CONFIG['level'],
        format=LOG_CONFIG['format'],
        datefmt=LOG_CONFIG['datefmt'],
        handlers=[
            logging.FileHandler(LOG_CONFIG['filename']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# Create default logger
logger = setup_logging()


if __name__ == '__main__':
    # Print configuration summary
    print("="*80)
    print("GALAXY DATA SCIENCE PIPELINE - CONFIGURATION")
    print("="*80)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Catalog directory: {CATALOG_DIR}")
    print(f"Database: {DATABASE_PATH}")
    print(f"\nAvailable catalogs: {len(CATALOG_FILES)}")
    for key, filename in CATALOG_FILES.items():
        path = CATALOG_DIR / filename
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {key:20s} -> {filename}")

    print(f"\nEmission lines defined: {len(EMISSION_LINES)}")
    print(f"Analysis parameters: {len(ANALYSIS_PARAMS)}")
    print(f"Log file: {LOG_CONFIG['filename']}")
    print("\n" + "="*80)
