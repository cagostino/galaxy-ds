# Galaxy Data Science Pipeline

A comprehensive Python framework for multi-wavelength galaxy analysis, combining SDSS spectroscopy, GSWLC photometry, and X-ray observations (XMM-Newton, Chandra).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [SQL Refactor Status](#sql-refactor-status)
- [Diagnostic Tools](#diagnostic-tools)
- [Repository Structure](#repository-structure)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

This repository provides a unified framework for analyzing galaxy properties across multiple wavelengths. The pipeline handles:

- **Optical spectroscopy**: SDSS MPA/JHU emission line catalogs
- **UV/optical photometry**: GSWLC (GALEX-SDSS-WISE Legacy Catalog)
- **X-ray observations**: XMM-Newton 4XMM DR8, Chandra CSC
- **Radio data**: FIRST, NVSS

Key capabilities:
- BPT diagram classifications (SF, AGN, Composite, LINER)
- Dust correction with multiple models (Asano+2019, Asano+2021)
- Multi-catalog coordinate matching with sky separation thresholds
- X-ray luminosity calculations and AGN identification
- SQL database backend for efficient querying of large catalogs

## Features

### Data Loading & Transformation
- Unified `AstroTablePD` class for FITS, CSV, TSV files
- Automatic column name standardization
- Built-in validation and type checking
- Multi-dimensional FITS table handling

### Catalog Matching
- Sky coordinate matching with configurable separation thresholds
- Plate/MJD/Fiber matching for SDSS spectroscopy
- Caching system for expensive coordinate matches
- Comprehensive merge validation and logging

### Emission Line Analysis
- Signal-to-noise filtering for line detections
- BPT classification (Kauffmann+2003, Kewley+2001)
- Multiple dust correction models
- Line luminosity calculations with cosmology

### X-ray Analysis
- Flux to luminosity conversions
- AGN vs star formation separation
- Exposure time filtering
- Multiple X-ray bands (soft, hard, full)

### SQL Database Integration
- SQLite3 backend for multi-GB catalogs
- Indexed tables for fast querying
- Complex JOIN operations
- Batch insert with duplicate column handling

## Installation

### Requirements

```bash
# Core dependencies
python >= 3.8
numpy >= 1.20
pandas >= 1.3
astropy >= 4.3
scipy >= 1.7
matplotlib >= 3.4
scikit-learn >= 0.24

# Optional (for enhanced features)
chroptiks  # Custom plotting utilities
```

### Setup

```bash
# Clone repository
git clone https://github.com/cagostino/galaxy-ds.git
cd galaxy-ds

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from data_models import AstroTablePD; print('✓ Installation successful')"
```

### Data Setup

Place catalog files in `catalogs/` directory:

```
catalogs/
├── gal_info_dr7_v5_2.fit      # SDSS galaxy info
├── gal_line_dr7_v5_2.fit      # SDSS emission lines
├── gal_indx_dr7_v5_2.fit      # SDSS spectral indices
├── totlgm_dr7_v5_2b.fit       # Total galaxy masses
├── fiblgm_dr7_v5_2.fit        # Fiber aperture masses
├── GSWLC-M2.dat                # GSWLC main catalog
├── 4xmm.fits                   # XMM 4XMM catalog
├── 4xmmobs.tsv                 # XMM observation info
└── catalog_database.db         # SQLite database (auto-created)
```

## Quick Start

### Basic Pipeline Run

```python
from data_models import *
from data_utils import *

# Load SDSS spectroscopy
galinfo = Gal_Info('catalogs/gal_info_dr7_v5_2.fit')
galline = Gal_Line('catalogs/gal_line_dr7_v5_2.fit')

# Merge catalogs
merged_sdss = AstroTablePD(dataframe=iterative_merge([
    galinfo.data, galline.data
]))

# Apply BPT classification
merged_sdss.data = get_line_filters(merged_sdss.data)

# Count by class
print(merged_sdss.data['bptclass'].value_counts())
```

### Query Database

```python
from data_utils import DBConnector

db = DBConnector('catalogs/catalog_database.db')

# Get BPT-classified AGN
agn = db.query("""
    SELECT * FROM gsw_m2_sdss
    WHERE bpt_sn_filt_bool = 1
    AND bptclass = 'AGN'
""")

print(f"Found {len(agn)} AGN")
```

### X-ray Matching

```python
# Load X-ray catalog
xmm4obs = XMM4obs('catalogs/4xmmobs.tsv')
x4 = XMM('catalogs/4xmm.fits', xmm4obs)

# Match to optical catalog
matches = coordinate_matching_and_join(
    x4.data,
    optical_catalog,
    ra_dec_1=['RA_ICRS', 'DE_ICRS'],
    ra_dec_2=['RA', 'DEC'],
    dist_threshold=7  # arcseconds
)

print(f"Matched {len(matches)} X-ray sources to optical galaxies")
```

## Architecture

### Class Hierarchy

```
AstroTablePD (base class)
├── Gal_Info    - SDSS galaxy metadata
├── Gal_Line    - SDSS emission line measurements
├── Gal_Indx    - SDSS spectral indices
├── Gal_Fib     - SDSS fiber-specific properties
├── XMM         - XMM-Newton X-ray sources
├── XMM3obs/XMM4obs - XMM observation metadata
└── CSC         - Chandra Source Catalog
```

### Data Flow

```
FITS/CSV Files
    ↓
AstroTablePD.load()
    ↓
Data Cleaning & Validation
    ↓
Feature Engineering (BPT, dust correction)
    ↓
Catalog Matching (coordinate/ID based)
    ↓
SQLite Database
    ↓
SQL Queries & Analysis
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `data_models.py` | Data loading, class definitions, feature engineering |
| `data_utils.py` | Merging, matching, database operations |
| `ast_utils.py` | Astronomy calculations (luminosities, cosmology) |
| `demarcations.py` | BPT/emission line diagnostic boundaries |
| `load_data.py` | Main pipeline script |

## Data Pipeline

### Step 1: Load Individual Catalogs

```python
# Load all SDSS tables
galinfo = Gal_Info(catfold+'gal_info_dr7_v5_2.fit')
galline = Gal_Line(catfold+'gal_line_dr7_v5_2.fit')
galindx = Gal_Indx(catfold+'gal_indx_dr7_v5_2.fit')
# ... (fiber properties, masses, SFRs)

# Load GSWLC
m2_gsw = AstroTablePD(catfold+'GSWLC-M2.dat')
```

### Step 2: Merge SDSS Catalogs

```python
merged_sdss = AstroTablePD(dataframe=iterative_merge([
    galinfo.data, galline.data, galindx.data,
    galfiboh.data, galmass.data, fibsfr.data, fibssfr.data
]))
```

### Step 3: Apply Line Filters & BPT Classification

```python
merged_sdss.data = get_line_filters(merged_sdss.data)
# Creates: bpt_sn_filt_bool, bptclass, vo87_1_filt_bool, etc.
```

### Step 4: Dust Corrections

```python
merged_sdss.data = add_dust_corrected_fluxes_by_model(
    merged_sdss.data,
    galline.lines,
    model='a19',  # Asano+2019
    dec_rat=3.1
)
merged_sdss.data = get_dust_correction_quantities(merged_sdss.data, model='a19')
```

### Step 5: Match GSWLC to SDSS

```python
m2_sdss = match_and_merge(
    m2_gsw.data,
    merged_sdss.data,
    left_on=['plate', 'MJD', 'fiber_ID'],
    right_on=['PLATEID', 'MJD', 'FIBERID']
)
```

### Step 6: X-ray Matching

```python
xmm_matches = coordinate_matching_and_join(
    x4.data,
    m2_sdss_coords,
    ra_dec_1=['RA_ICRS', 'DE_ICRS'],
    ra_dec_2=['RA_span', 'dec_span'],
    dist_threshold=7
)
```

### Step 7: Database Insert

```python
insert_dataframe_to_table(
    m2_sdss,
    'gsw_m2_sdss',
    'catalogs/catalog_database.db'
)
```

## SQL Refactor Status

### Background

In November 2023, the pipeline was refactored from a pure pandas/numpy approach to use SQL database backend for better performance with large catalogs (>1M sources). However, this introduced discrepancies in selection criteria counts.

### Known Issues

⚠️ **Different counts for identical selection criteria between old and new systems**

**Possible causes:**
1. Boolean column storage (True/False vs 1/0)
2. NULL/NaN handling differences
3. Type coercion in SQL queries
4. Index alignment issues in pandas
5. Merge/join behavior differences

### Status: Under Investigation

We have created comprehensive diagnostic tools to identify root causes. See [Diagnostic Tools](#diagnostic-tools) section.

### What Works
✅ Data loading and FITS parsing
✅ Coordinate matching and caching
✅ BPT classification logic
✅ Dust correction calculations
✅ Database table creation and insertion

### What's Being Debugged
🔧 SQL query results vs pandas filter counts
🔧 Merge statistics validation
🔧 NULL handling in boolean columns

## Diagnostic Tools

To identify where SQL queries produce different results than pandas operations:

### 1. Full Pipeline Validation

```bash
python validate_sql_pipeline.py
```

**Output:** `validation_report.txt` with row counts at each step

### 2. Pipeline Comparison

```bash
python compare_pipelines.py
```

**Output:** `pipeline_discrepancies.csv` with count mismatches

### 3. SQL vs Pandas Testing

```bash
python test_sql_vs_pandas.py
```

**Output:** Direct comparison of query results with type/NaN analysis

### Reading Diagnostic Output

**Good (no issues):**
```
✓ Checkpoint 'm2_sdss_merged': 150000 rows, 250 columns
  Merge statistics:
    both: 145000 (96.67%)
    left_only: 3000 (2.00%)
✓ Match: both returned 145000 rows
```

**Problem detected:**
```
⚠ MISMATCH: Expected 145000, got 132000 (diff: -13000, -8.97%)
  Type mismatch in 'bpt_sn_filt_bool': SQL=int64, pandas=bool
  NaN counts differ: OIII_5007_FLUX: SQL=8000, pandas=12000
```

See `DIAGNOSTIC_TOOLS.md` for detailed usage guide.

## Repository Structure

```
galaxy-ds/
├── data_models.py           # Data loading classes & feature engineering
├── data_utils.py            # Merging, matching, database utilities
├── ast_utils.py             # Astronomy calculations & conversions
├── load_data.py             # Main pipeline script
├── demarcations.py          # BPT diagram boundaries
├── image_utils.py           # Image loading/plotting
├── matplotlibrc             # Matplotlib config
│
├── catalogs/                # Data files (not in repo)
│   ├── *.fit               # SDSS FITS tables
│   ├── *.dat               # GSWLC catalogs
│   └── catalog_database.db # SQLite database
│
├── dist_met_procedures/     # Distance metric calculations
│   ├── Gdiffs.py           # Galaxy difference metrics
│   └── sfrmatch.py         # SFR-based matching
│
├── observations/            # Observation-specific scripts
│   └── config_files/       # Analysis configurations
│
├── plotting_codes/          # Visualization scripts
│   ├── plotresults_sfrm.py
│   └── plotresults_3xmm.py
│
├── xray_data_analysis/      # X-ray specific tools
│   ├── xraycov.py          # X-ray coverage
│   ├── hiligt.py           # High-L_X galaxy tools
│   └── getxmmdata.py       # XMM data retrieval
│
├── compare_pipelines.py     # Diagnostic: compare counts
├── validate_sql_pipeline.py # Diagnostic: step-by-step validation
├── test_sql_vs_pandas.py    # Diagnostic: SQL vs pandas tests
│
├── DIAGNOSTIC_TOOLS.md      # Diagnostic usage guide
├── REFACTOR_ANALYSIS.md     # SQL refactor analysis
└── README.md                # This file
```

## Usage Examples

### Example 1: Select BPT-classified AGN with High OIII

```python
from data_utils import DBConnector

db = DBConnector('catalogs/catalog_database.db')

agn_high_oiii = db.query("""
    SELECT
        ObjID,
        z,
        OIII_5007_FLUX,
        H_BETA_FLUX,
        mass,
        sfr
    FROM gsw_m2_sdss
    WHERE bpt_sn_filt_bool = 1
        AND bptclass = 'AGN'
        AND OIII_5007_FLUX_SN > 5
        AND z BETWEEN 0.02 AND 0.1
""")

print(f"Selected {len(agn_high_oiii)} AGN")
```

### Example 2: X-ray Detected Star-Forming Galaxies

```python
xray_sf = db.query("""
    SELECT
        m2.*,
        xr.fulllumsrf,
        xr.texp
    FROM gsw_m2_sdss AS m2
    JOIN xmm4_gsw_sdss_sed0_combined AS xr
        ON m2.sdss_gsw_index = xr.sdss_gsw_index
    WHERE m2.bptclass = 'HII'
        AND xr.fulllumsrf > 40.0
        AND xr.texp > 10000
""")
```

### Example 3: Create Custom Catalog Match

```python
from data_utils import coordinate_matching

# Load your catalogs
catalog1 = pd.read_csv('catalog1.csv')
catalog2 = pd.read_csv('catalog2.csv')

# Match within 5 arcseconds
matches = coordinate_matching(
    catalog1,
    catalog2,
    ra_dec_1=('ra_col1', 'dec_col1'),
    ra_dec_2=('ra_col2', 'dec_col2'),
    dist_threshold=5,
    matches_filename='my_matches.csv'
)

print(f"Found {len(matches)} matches")
print(f"Mean separation: {matches['separation_arcsec'].mean():.2f}\"")
```

### Example 4: Batch Process Multiple Dust Models

```python
from data_models import add_dust_corrected_fluxes_by_model, get_dust_correction_quantities

models = ['a19', 'a21']
results = {}

for model in models:
    df_corrected = add_dust_corrected_fluxes_by_model(
        df.copy(),
        emission_lines,
        model=model
    )
    df_corrected = get_dust_correction_quantities(df_corrected, model=model)
    results[model] = df_corrected

# Compare metallicities
for model in models:
    median_oh = results[model]['log_oh'].median()
    print(f"{model}: median 12+log(O/H) = {median_oh:.2f}")
```

## Contributing

### Adding New Data Models

To add support for a new catalog:

1. **Create class in `data_models.py`:**
```python
class MyCatalog(AstroTablePD):
    def __init__(self, filename):
        super().__init__(filename)
        # Add derived columns
        self.data['my_derived_col'] = self.data['col1'] / self.data['col2']
```

2. **Load and insert to database:**
```python
mycat = MyCatalog('catalogs/mycat.fits')
insert_dataframe_to_table(mycat.data, 'mycat_table', 'catalogs/catalog_database.db')
```

3. **Add to pipeline** in `load_data.py`

### Coding Standards

- Use type hints for function parameters
- Add docstrings with parameter descriptions
- Log major operations with `logger.info()`
- Handle errors gracefully with try/except
- Write unit tests for new features

### Reporting Issues

When reporting discrepancies in counts:

1. Exact selection criteria (SQL query or pandas filter)
2. Expected count (from old pipeline/paper)
3. Actual count (from current pipeline)
4. Relevant log excerpts
5. Sample of affected rows

## Citation

If you use this code in published research, please cite:

```bibtex
@software{galaxy_ds,
  author = {Agostino, Christopher},
  title = {Galaxy Data Science Pipeline},
  year = {2023},
  url = {https://github.com/cagostino/galaxy-ds}
}
```

Related publications:
- SDSS MPA/JHU catalogs: [Brinchmann et al. 2004](https://ui.adsabs.harvard.edu/abs/2004MNRAS.351.1151B)
- GSWLC: [Salim et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJS..227....2S)
- XMM 4XMM: [Webb et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...641A.136W)
- BPT diagrams: [Kewley et al. 2006](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..961K)

## Related Packages

- [chroptiks](https://github.com/cagostino/chroptiks) - Advanced plotting utilities developed from this work
- [astropy](https://www.astropy.org/) - Core astronomy functionality
- [astroquery](https://astroquery.readthedocs.io/) - Query astronomical databases

## Support

- **Documentation:** See `DIAGNOSTIC_TOOLS.md` and `REFACTOR_ANALYSIS.md`
- **Issues:** [GitHub Issues](https://github.com/cagostino/galaxy-ds/issues)
- **Discussions:** [GitHub Discussions](https://github.com/cagostino/galaxy-ds/discussions)

## License

MIT License - see LICENSE file for details.

---

**Last Updated:** November 2025
**Python Version:** 3.8+
**Status:** Active development - SQL refactor under validation
