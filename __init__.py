"""galaxy-ds: Galaxy data science pipeline.

Refactored from known-good code (commit 83572ef, Oct 2021).
"""

from .constants import LINE_WAVELENGTHS, COSMO
from .dust import compute_av, dustcorrect, redden, correct_av_hybrid
from .classification import (
    get_bpt1_groups, get_bpt1_groups_ke01, get_bptplus_groups,
    get_bptplus_niigroups, get_bpt2_groups, get_bpt3_groups,
    get_whan_groups, get_ooo_groups,
)
from .cosmology import get_lum_from_flux, get_flux_from_lum, halpha_to_sfr
from .catalog_io import (
    load_gswlc_m2, load_agostino2021, load_sdss_mpa_jhu,
    match_by_id, match_by_coords, save_parquet, load_parquet,
)
from .emission_lines import build_emission_line_catalog
from .wise_analysis import (
    compute_wise_colors, kcorrect_w1w2, apply_kcorrection,
    reliability_completeness, scan_w1w2_cuts, bin_reliability_completeness,
)
