#!/usr/bin/env python3
"""
Diagnostic tool to compare old (summer 2023) vs new (SQL) pipeline outputs
and identify where selection criteria produce different counts.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineComparator:
    """Compare outputs between old pandas-based and new SQL-based pipelines"""

    def __init__(self, db_path='catalogs/catalog_database.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path) if Path(db_path).exists() else None
        self.discrepancies = []

    def compare_table_counts(self, table_name, expected_count=None):
        """Compare row counts for a given table"""
        if not self.conn:
            logger.error(f"No database connection for {table_name}")
            return None

        query = f"SELECT COUNT(*) FROM {table_name}"
        try:
            result = pd.read_sql_query(query, self.conn)
            actual_count = result.iloc[0, 0]

            logger.info(f"Table '{table_name}': {actual_count} rows")

            if expected_count is not None:
                diff = actual_count - expected_count
                if diff != 0:
                    self.discrepancies.append({
                        'table': table_name,
                        'expected': expected_count,
                        'actual': actual_count,
                        'diff': diff,
                        'pct_diff': (diff / expected_count * 100) if expected_count > 0 else float('inf')
                    })
                    logger.warning(f"  MISMATCH: Expected {expected_count}, got {actual_count} (diff: {diff}, {diff/expected_count*100:.2f}%)")

            return actual_count
        except Exception as e:
            logger.error(f"Error querying {table_name}: {e}")
            return None

    def compare_selection(self, table_name, condition, expected_count=None, label=""):
        """Compare counts for a specific selection criteria"""
        if not self.conn:
            return None

        query = f"SELECT COUNT(*) FROM {table_name} WHERE {condition}"
        try:
            result = pd.read_sql_query(query, self.conn)
            actual_count = result.iloc[0, 0]

            logger.info(f"Selection '{label}' on {table_name}: {actual_count} rows")
            logger.info(f"  Condition: {condition}")

            if expected_count is not None:
                diff = actual_count - expected_count
                if diff != 0:
                    self.discrepancies.append({
                        'table': table_name,
                        'condition': condition,
                        'label': label,
                        'expected': expected_count,
                        'actual': actual_count,
                        'diff': diff,
                        'pct_diff': (diff / expected_count * 100) if expected_count > 0 else float('inf')
                    })
                    logger.warning(f"  MISMATCH: Expected {expected_count}, got {actual_count} (diff: {diff})")

            return actual_count
        except Exception as e:
            logger.error(f"Error with selection '{label}': {e}")
            return None

    def compare_merge_stats(self, table_name):
        """Compare merge statistics (_merge column)"""
        if not self.conn:
            return None

        query = f"SELECT _merge, COUNT(*) as count FROM {table_name} GROUP BY _merge"
        try:
            result = pd.read_sql_query(query, self.conn)
            logger.info(f"Merge stats for {table_name}:")
            for _, row in result.iterrows():
                logger.info(f"  {row['_merge']}: {row['count']}")
            return result
        except Exception as e:
            logger.error(f"Error getting merge stats for {table_name}: {e}")
            return None

    def compare_dataframes(self, df1, df2, name1="df1", name2="df2"):
        """Compare two DataFrames and report differences"""
        logger.info(f"\nComparing {name1} vs {name2}:")
        logger.info(f"  {name1} shape: {df1.shape}")
        logger.info(f"  {name2} shape: {df2.shape}")

        # Compare columns
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        common = cols1 & cols2
        only1 = cols1 - cols2
        only2 = cols2 - cols1

        if only1:
            logger.warning(f"  Columns only in {name1}: {only1}")
        if only2:
            logger.warning(f"  Columns only in {name2}: {only2}")

        # For common columns, compare values
        for col in sorted(common):
            try:
                if df1[col].dtype in [np.float64, np.float32]:
                    # Numerical comparison with tolerance
                    diff_mask = ~np.isclose(df1[col], df2[col], rtol=1e-5, atol=1e-8, equal_nan=True)
                    num_diff = diff_mask.sum()
                    if num_diff > 0:
                        logger.warning(f"  Column '{col}': {num_diff} values differ")
                        # Show examples
                        diff_indices = np.where(diff_mask)[0][:5]
                        for idx in diff_indices:
                            logger.warning(f"    Row {idx}: {df1[col].iloc[idx]} vs {df2[col].iloc[idx]}")
                else:
                    # Exact comparison for non-numerical
                    num_diff = (df1[col] != df2[col]).sum()
                    if num_diff > 0:
                        logger.warning(f"  Column '{col}': {num_diff} values differ")
            except Exception as e:
                logger.error(f"  Error comparing column '{col}': {e}")

        return {
            'shape_match': df1.shape == df2.shape,
            'columns_match': only1 == set() and only2 == set(),
            'common_columns': len(common)
        }

    def trace_pipeline_step(self, step_name, df, selections=None):
        """Trace a pipeline step and log statistics"""
        logger.info(f"\n=== Pipeline Step: {step_name} ===")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Check for NaN/null values
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) > 0:
            logger.warning(f"Columns with null values:")
            for col, count in null_cols.items():
                logger.warning(f"  {col}: {count} nulls ({count/len(df)*100:.2f}%)")

        # Apply selections if provided
        if selections:
            for sel_name, sel_condition in selections.items():
                try:
                    count = sel_condition(df).sum()
                    logger.info(f"  Selection '{sel_name}': {count} rows ({count/len(df)*100:.2f}%)")
                except Exception as e:
                    logger.error(f"  Error applying selection '{sel_name}': {e}")

        return df

    def report_discrepancies(self):
        """Generate summary report of all discrepancies"""
        if not self.discrepancies:
            logger.info("\n✓ No discrepancies found!")
            return

        logger.warning(f"\n⚠ Found {len(self.discrepancies)} discrepancies:")
        logger.warning("=" * 80)

        for disc in self.discrepancies:
            logger.warning(f"\nTable: {disc.get('table', 'N/A')}")
            if 'label' in disc:
                logger.warning(f"Selection: {disc['label']}")
            if 'condition' in disc:
                logger.warning(f"Condition: {disc['condition']}")
            logger.warning(f"Expected: {disc['expected']}")
            logger.warning(f"Actual: {disc['actual']}")
            logger.warning(f"Difference: {disc['diff']} ({disc['pct_diff']:.2f}%)")

        # Save to file
        df_disc = pd.DataFrame(self.discrepancies)
        df_disc.to_csv('pipeline_discrepancies.csv', index=False)
        logger.info(f"\nDiscrepancies saved to pipeline_discrepancies.csv")


def check_bpt_classifications(df, label=""):
    """Check BPT classification counts"""
    logger.info(f"\n=== BPT Classifications {label} ===")

    # Check for required columns
    required = ['bpt_sn_filt_bool', 'bptclass']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        return

    # Count BPT classifiable
    bpt_classifiable = df['bpt_sn_filt_bool'].sum()
    logger.info(f"BPT classifiable (S/N cut): {bpt_classifiable} / {len(df)} ({bpt_classifiable/len(df)*100:.2f}%)")

    # Count by class
    if 'bptclass' in df.columns:
        class_counts = df['bptclass'].value_counts()
        logger.info(f"BPT classifications:")
        for cls, count in class_counts.items():
            logger.info(f"  {cls}: {count} ({count/len(df)*100:.2f}%)")

    return bpt_classifiable


if __name__ == "__main__":
    comp = PipelineComparator()

    logger.info("="*80)
    logger.info("PIPELINE COMPARISON DIAGNOSTICS")
    logger.info("="*80)

    # Check if database exists
    if not comp.conn:
        logger.error("Database not found. Run load_data.py first to create it.")
    else:
        # List all tables
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", comp.conn)
        logger.info(f"\nAvailable tables: {list(tables['name'])}")

        # Check main tables
        for table in ['sdss_combined', 'gsw_m2_sdss', 'xmm4_gsw_sdss',
                     'xmm4_gsw_sdss_sed0_combined', 'xmm4_gsw_sdss_sed1_combined']:
            comp.compare_table_counts(table)
            comp.compare_merge_stats(table)

    comp.report_discrepancies()
