#!/usr/bin/env python3
"""
Validation script to test SQL pipeline and identify where counts diverge.
Compares each step of the pipeline to ensure data integrity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import local modules
from data_models import *
from data_utils import *


class ValidationCheckpoint:
    """Store validation checkpoints for comparison"""
    def __init__(self):
        self.checkpoints = {}

    def save(self, name, df, metadata=None):
        """Save a checkpoint"""
        checkpoint = {
            'shape': df.shape,
            'columns': list(df.columns),
            'index_name': df.index.name,
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'sample': df.head(10).to_dict() if len(df) > 0 else {},
            'metadata': metadata or {}
        }

        # Save statistics for numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            checkpoint['numerical_stats'] = df[num_cols].describe().to_dict()

        self.checkpoints[name] = checkpoint
        logger.info(f"✓ Checkpoint '{name}': {df.shape[0]} rows, {df.shape[1]} columns")

        return checkpoint

    def compare(self, name1, name2):
        """Compare two checkpoints"""
        if name1 not in self.checkpoints or name2 not in self.checkpoints:
            logger.error(f"Cannot compare: missing checkpoint(s)")
            return False

        cp1 = self.checkpoints[name1]
        cp2 = self.checkpoints[name2]

        logger.info(f"\nComparing '{name1}' vs '{name2}':")

        # Compare shapes
        if cp1['shape'] != cp2['shape']:
            logger.warning(f"  Shape mismatch: {cp1['shape']} vs {cp2['shape']}")
            return False

        # Compare columns
        cols1 = set(cp1['columns'])
        cols2 = set(cp2['columns'])
        if cols1 != cols2:
            logger.warning(f"  Column mismatch:")
            logger.warning(f"    Only in {name1}: {cols1 - cols2}")
            logger.warning(f"    Only in {name2}: {cols2 - cols1}")
            return False

        # Compare null counts
        null_diff = {}
        for col in cols1:
            if cp1['null_counts'].get(col, 0) != cp2['null_counts'].get(col, 0):
                null_diff[col] = (cp1['null_counts'].get(col, 0), cp2['null_counts'].get(col, 0))

        if null_diff:
            logger.warning(f"  Null count differences:")
            for col, (n1, n2) in null_diff.items():
                logger.warning(f"    {col}: {n1} vs {n2}")

        logger.info(f"  ✓ Checkpoints match")
        return True

    def export_report(self, filename='validation_report.txt'):
        """Export validation report"""
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")

            for name, cp in self.checkpoints.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"Checkpoint: {name}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Shape: {cp['shape']}\n")
                f.write(f"Columns: {', '.join(cp['columns'])}\n")

                # Null counts
                nulls = {k: v for k, v in cp['null_counts'].items() if v > 0}
                if nulls:
                    f.write(f"\nColumns with nulls:\n")
                    for col, count in nulls.items():
                        pct = (count / cp['shape'][0] * 100) if cp['shape'][0] > 0 else 0
                        f.write(f"  {col}: {count} ({pct:.2f}%)\n")

                # Metadata
                if cp['metadata']:
                    f.write(f"\nMetadata:\n")
                    for k, v in cp['metadata'].items():
                        f.write(f"  {k}: {v}\n")

        logger.info(f"Report exported to {filename}")


def validate_merge(left_df, right_df, merged_df, merge_on, name="merge"):
    """Validate a pandas merge operation"""
    logger.info(f"\nValidating {name}:")

    # Check that merge keys exist
    left_keys, right_keys = merge_on
    for k in left_keys:
        if k not in left_df.columns:
            logger.error(f"  Left merge key '{k}' not found in left DataFrame")
            return False
    for k in right_keys:
        if k not in right_df.columns:
            logger.error(f"  Right merge key '{k}' not found in right DataFrame")
            return False

    # Check merge indicator
    if '_merge' in merged_df.columns:
        merge_stats = merged_df['_merge'].value_counts()
        logger.info(f"  Merge statistics:")
        for status, count in merge_stats.items():
            pct = count / len(merged_df) * 100
            logger.info(f"    {status}: {count} ({pct:.2f}%)")

        # Check for unexpected merge results
        if 'left_only' in merge_stats and merge_stats['left_only'] > len(left_df) * 0.5:
            logger.warning(f"  ⚠ High number of left_only rows: {merge_stats['left_only']}")

        if 'right_only' in merge_stats and merge_stats['right_only'] > len(right_df) * 0.5:
            logger.warning(f"  ⚠ High number of right_only rows: {merge_stats['right_only']}")

    # Check for duplicate keys
    left_key_counts = left_df.groupby(left_keys).size()
    right_key_counts = right_df.groupby(right_keys).size()

    left_dups = (left_key_counts > 1).sum()
    right_dups = (right_key_counts > 1).sum()

    if left_dups > 0:
        logger.warning(f"  ⚠ Left DataFrame has {left_dups} duplicate key combinations")
    if right_dups > 0:
        logger.warning(f"  ⚠ Right DataFrame has {right_dups} duplicate key combinations")

    logger.info(f"  ✓ Merge validation complete")
    return True


def validate_selection_criteria(df, criteria_dict):
    """Validate selection criteria and report counts"""
    logger.info(f"\nSelection criteria validation:")

    results = {}
    for name, condition in criteria_dict.items():
        try:
            if callable(condition):
                mask = condition(df)
            elif isinstance(condition, str):
                mask = df.eval(condition)
            else:
                mask = condition

            count = mask.sum()
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            results[name] = count

            logger.info(f"  {name}: {count} / {len(df)} ({pct:.2f}%)")

        except Exception as e:
            logger.error(f"  Error applying '{name}': {e}")
            results[name] = None

    return results


def run_pipeline_validation():
    """Run full pipeline with validation checkpoints"""
    vp = ValidationCheckpoint()
    catfold = 'catalogs/'

    try:
        logger.info("="*80)
        logger.info("STARTING PIPELINE VALIDATION")
        logger.info("="*80)

        # Step 1: Load GSW catalog
        logger.info("\n--- Step 1: Loading GSW M2 catalog ---")
        m2_gsw = AstroTablePD(catfold+'GSWLC-M2.dat')
        vp.save('m2_gsw', m2_gsw.data, {'source': 'GSWLC-M2.dat'})

        # Step 2: Load SDSS catalogs
        logger.info("\n--- Step 2: Loading SDSS catalogs ---")
        galinfo  = Gal_Info(catfold+'gal_info_dr7_v5_2.fit')
        vp.save('galinfo', galinfo.data, {'source': 'gal_info_dr7_v5_2.fit'})

        galline = Gal_Line(catfold+'gal_line_dr7_v5_2.fit')
        vp.save('galline', galline.data, {'source': 'gal_line_dr7_v5_2.fit'})

        galindx = Gal_Indx(catfold+'gal_indx_dr7_v5_2.fit')
        vp.save('galindx', galindx.data, {'source': 'gal_indx_dr7_v5_2.fit'})

        # Step 3: Merge SDSS catalogs
        logger.info("\n--- Step 3: Merging SDSS catalogs ---")
        galfiboh = Gal_Fib(catfold+'gal_fiboh_dr7_v5_2.fits', 'fiboh')
        galmass = Gal_Fib(catfold+'totlgm_dr7_v5_2b.fit', 'mass')
        fibmass = Gal_Fib(catfold+'fiblgm_dr7_v5_2.fit', 'fibmass')
        fibsfr = Gal_Fib(catfold+'gal_fibsfr_dr7_v5_2.fits','fibsfr')
        fibssfr = Gal_Fib(catfold+'gal_fibspecsfr_dr7_v5_2.fits', 'fibssfr')

        merged_sdss = AstroTablePD(dataframe=iterative_merge([
            galinfo.data, galline.data, galindx.data,
            galfiboh.data, galmass.data, fibsfr.data, fibssfr.data
        ]))
        vp.save('merged_sdss_raw', merged_sdss.data)

        # Step 4: Apply line filters
        logger.info("\n--- Step 4: Applying line filters ---")
        merged_sdss.data = get_line_filters(merged_sdss.data)
        vp.save('merged_sdss_with_filters', merged_sdss.data)

        # Validate BPT classifications
        bpt_criteria = {
            'bpt_sn_filt_bool': lambda df: df['bpt_sn_filt_bool'],
            'HII (bptclass==HII)': lambda df: df['bptclass'] == 'HII',
            'AGN (bptclass==AGN)': lambda df: df['bptclass'] == 'AGN',
            'Composite': lambda df: df['bptclass'] == 'Composite',
        }
        bpt_counts = validate_selection_criteria(merged_sdss.data, bpt_criteria)

        # Step 5: Add dust corrections
        logger.info("\n--- Step 5: Adding dust corrections ---")
        merged_sdss.data = add_dust_corrected_fluxes_by_model(
            merged_sdss.data, galline.lines,
            modelfn=get_extinction, model='a19', dec_rat=3.1
        )
        merged_sdss.data = get_dust_correction_quantities(merged_sdss.data, model='a19')
        vp.save('merged_sdss_with_dust', merged_sdss.data)

        # Step 6: Merge GSW and SDSS
        logger.info("\n--- Step 6: Merging GSW and SDSS ---")
        left_id_columns = ['plate', 'MJD', 'fiber_ID']
        right_id_columns = ['PLATEID', 'MJD', 'FIBERID']

        m2_sdss = match_and_merge(
            m2_gsw.data, merged_sdss.data,
            left_on=left_id_columns, right_on=right_id_columns
        )
        vp.save('m2_sdss_merged', m2_sdss)

        # Validate merge
        validate_merge(
            m2_gsw.data, merged_sdss.data, m2_sdss,
            (left_id_columns, right_id_columns),
            name="GSW-SDSS merge"
        )

        # Check for column naming issues
        logger.info("\nChecking for RA/DEC columns:")
        ra_cols = [c for c in m2_sdss.columns if 'RA' in c.upper()]
        dec_cols = [c for c in m2_sdss.columns if 'DEC' in c.upper()]
        logger.info(f"  RA columns: {ra_cols}")
        logger.info(f"  DEC columns: {dec_cols}")

        # Export report
        vp.export_report('validation_report.txt')

        logger.info("\n" + "="*80)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*80)

        return vp

    except Exception as e:
        logger.error(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = run_pipeline_validation()

    if result:
        logger.info("\n✓ Validation completed successfully")
        logger.info("Check validation_report.txt for details")
        sys.exit(0)
    else:
        logger.error("\n❌ Validation failed")
        sys.exit(1)
