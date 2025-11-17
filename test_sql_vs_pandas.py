#!/usr/bin/env python3
"""
Test SQL queries vs equivalent pandas operations to find discrepancies.
This is the key diagnostic to find where SQL gives different results.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sql_vs_pandas.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SQLvsPandasTester:
    """Compare SQL query results vs pandas operations"""

    def __init__(self, db_path='catalogs/catalog_database.db'):
        self.db_path = db_path
        if not Path(db_path).exists():
            logger.error(f"Database not found: {db_path}")
            self.conn = None
        else:
            self.conn = sqlite3.connect(db_path)
            logger.info(f"Connected to {db_path}")

    def test_select_where(self, table_name, condition, df=None):
        """
        Compare SQL SELECT WHERE vs pandas boolean indexing.

        Args:
            table_name: SQL table name
            condition: WHERE clause (SQL syntax) or pandas query string
            df: Optional pandas DataFrame to compare against
        """
        if not self.conn:
            return None

        logger.info(f"\n{'='*80}")
        logger.info(f"Testing: {table_name} WHERE {condition}")
        logger.info(f"{'='*80}")

        # SQL query
        sql_query = f"SELECT * FROM {table_name} WHERE {condition}"
        try:
            sql_result = pd.read_sql_query(sql_query, self.conn)
            sql_count = len(sql_result)
            logger.info(f"SQL result: {sql_count} rows")
        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return None

        # Pandas query (if DataFrame provided)
        if df is not None:
            try:
                # Convert SQL-style condition to pandas query if needed
                pandas_condition = condition.replace('==', '==').replace('=', '==')
                pandas_result = df.query(pandas_condition)
                pandas_count = len(pandas_result)
                logger.info(f"Pandas result: {pandas_count} rows")

                # Compare
                diff = sql_count - pandas_count
                if diff != 0:
                    logger.warning(f"❌ MISMATCH: {diff} row difference")
                    logger.warning(f"   SQL: {sql_count}, Pandas: {pandas_count}")

                    # Try to identify the issue
                    self._debug_mismatch(sql_result, pandas_result, condition)
                else:
                    logger.info(f"✓ Match: both returned {sql_count} rows")

                return {'sql': sql_count, 'pandas': pandas_count, 'diff': diff}

            except Exception as e:
                logger.error(f"Pandas query failed: {e}")
                return {'sql': sql_count, 'pandas': None, 'error': str(e)}

        return {'sql': sql_count}

    def _debug_mismatch(self, sql_df, pandas_df, condition):
        """Debug why SQL and pandas give different results"""
        logger.info("\nDebugging mismatch:")

        # Check for column presence
        sql_cols = set(sql_df.columns)
        pandas_cols = set(pandas_df.columns)

        if sql_cols != pandas_cols:
            logger.warning("  Column mismatch:")
            logger.warning(f"    Only in SQL: {sql_cols - pandas_cols}")
            logger.warning(f"    Only in pandas: {pandas_cols - sql_cols}")

        # Check data types
        common_cols = sql_cols & pandas_cols
        for col in common_cols:
            if sql_df[col].dtype != pandas_df[col].dtype:
                logger.warning(f"  Type mismatch in '{col}': SQL={sql_df[col].dtype}, pandas={pandas_df[col].dtype}")

        # Check for NaN handling differences
        logger.info("\n  NaN counts:")
        for col in common_cols:
            sql_nans = sql_df[col].isna().sum()
            pandas_nans = pandas_df[col].isna().sum()
            if sql_nans != pandas_nans:
                logger.warning(f"    {col}: SQL={sql_nans}, pandas={pandas_nans}")

    def test_join(self, left_table, right_table, join_on, join_type='inner'):
        """
        Test SQL JOIN vs pandas merge.

        Args:
            left_table: Name of left table
            right_table: Name of right table
            join_on: Dictionary mapping left columns to right columns
            join_type: 'inner', 'left', 'right', or 'outer'
        """
        if not self.conn:
            return None

        logger.info(f"\n{'='*80}")
        logger.info(f"Testing: {left_table} {join_type.upper()} JOIN {right_table}")
        logger.info(f"{'='*80}")

        # Build SQL JOIN
        left_cols = list(join_on.keys())
        right_cols = list(join_on.values())
        on_clause = " AND ".join([f"{left_table}.{l} = {right_table}.{r}"
                                  for l, r in zip(left_cols, right_cols)])

        sql_query = f"""
            SELECT {left_table}.*, {right_table}.*
            FROM {left_table}
            {join_type.upper()} JOIN {right_table}
            ON {on_clause}
        """

        try:
            sql_result = pd.read_sql_query(sql_query, self.conn)
            sql_count = len(sql_result)
            logger.info(f"SQL {join_type} join: {sql_count} rows")
        except Exception as e:
            logger.error(f"SQL join failed: {e}")
            return None

        # Load tables for pandas merge
        try:
            left_df = pd.read_sql_query(f"SELECT * FROM {left_table}", self.conn)
            right_df = pd.read_sql_query(f"SELECT * FROM {right_table}", self.conn)

            # Pandas merge
            pandas_result = pd.merge(
                left_df, right_df,
                left_on=left_cols,
                right_on=right_cols,
                how=join_type,
                indicator=True
            )
            pandas_count = len(pandas_result)
            logger.info(f"Pandas merge: {pandas_count} rows")

            # Check merge indicator
            if '_merge' in pandas_result.columns:
                merge_stats = pandas_result['_merge'].value_counts()
                logger.info("  Merge statistics:")
                for status, count in merge_stats.items():
                    logger.info(f"    {status}: {count}")

            # Compare
            diff = sql_count - pandas_count
            if diff != 0:
                logger.warning(f"❌ MISMATCH: {diff} row difference")
            else:
                logger.info(f"✓ Match: both returned {sql_count} rows")

            return {'sql': sql_count, 'pandas': pandas_count, 'diff': diff}

        except Exception as e:
            logger.error(f"Pandas merge failed: {e}")
            return None

    def test_aggregation(self, table_name, group_by, agg_col, agg_func='COUNT'):
        """Test SQL aggregation vs pandas groupby"""
        if not self.conn:
            return None

        logger.info(f"\n{'='*80}")
        logger.info(f"Testing: {agg_func}({agg_col}) GROUP BY {group_by}")
        logger.info(f"{'='*80}")

        # SQL aggregation
        sql_query = f"""
            SELECT {group_by}, {agg_func}({agg_col}) as agg_result
            FROM {table_name}
            GROUP BY {group_by}
        """

        try:
            sql_result = pd.read_sql_query(sql_query, self.conn)
            logger.info(f"SQL aggregation: {len(sql_result)} groups")

            # Load full table for pandas
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)

            # Pandas aggregation
            if agg_func.upper() == 'COUNT':
                pandas_result = df.groupby(group_by)[agg_col].count()
            elif agg_func.upper() == 'SUM':
                pandas_result = df.groupby(group_by)[agg_col].sum()
            elif agg_func.upper() == 'AVG' or agg_func.upper() == 'MEAN':
                pandas_result = df.groupby(group_by)[agg_col].mean()
            else:
                logger.error(f"Unsupported aggregation function: {agg_func}")
                return None

            logger.info(f"Pandas aggregation: {len(pandas_result)} groups")

            # Compare results
            if len(sql_result) != len(pandas_result):
                logger.warning(f"❌ Different number of groups")

            # Compare values
            diffs = []
            for _, row in sql_result.iterrows():
                group_val = row[group_by]
                sql_agg = row['agg_result']
                if group_val in pandas_result.index:
                    pandas_agg = pandas_result[group_val]
                    if not np.isclose(sql_agg, pandas_agg):
                        diffs.append((group_val, sql_agg, pandas_agg))

            if diffs:
                logger.warning(f"❌ {len(diffs)} groups have different aggregation results")
                for group, sql_val, pandas_val in diffs[:5]:
                    logger.warning(f"  {group}: SQL={sql_val}, pandas={pandas_val}")
            else:
                logger.info(f"✓ All aggregations match")

            return {'num_groups_sql': len(sql_result), 'num_groups_pandas': len(pandas_result), 'diffs': len(diffs)}

        except Exception as e:
            logger.error(f"Aggregation test failed: {e}")
            return None


def run_comprehensive_tests():
    """Run comprehensive SQL vs pandas tests"""
    tester = SQLvsPandasTester()

    if not tester.conn:
        logger.error("Cannot run tests without database connection")
        return

    logger.info("\n" + "="*80)
    logger.info("SQL vs PANDAS COMPREHENSIVE TESTING")
    logger.info("="*80)

    # Test 1: Basic SELECT with WHERE clause
    logger.info("\n### TEST 1: Basic WHERE clauses ###")

    # Load table for comparison
    try:
        gsw_m2_sdss = pd.read_sql_query("SELECT * FROM gsw_m2_sdss", tester.conn)
        logger.info(f"Loaded gsw_m2_sdss: {len(gsw_m2_sdss)} rows")

        # Test BPT filter
        tester.test_select_where(
            'gsw_m2_sdss',
            'bpt_sn_filt_bool == 1',
            df=gsw_m2_sdss
        )

        # Test flag_sed filter
        tester.test_select_where(
            'gsw_m2_sdss',
            'flag_sed == 0',
            df=gsw_m2_sdss
        )

        # Test combined filter
        tester.test_select_where(
            'gsw_m2_sdss',
            'bpt_sn_filt_bool == 1 AND flag_sed == 0',
            df=gsw_m2_sdss
        )

    except Exception as e:
        logger.error(f"Test 1 failed: {e}")

    # Test 2: JOIN operations
    logger.info("\n### TEST 2: JOIN operations ###")
    try:
        tester.test_join(
            'gsw_m2_sdss',
            'xmm4_gsw_m2_sdss',
            {'sdss_gsw_index': 'sdss_gsw_index'},
            join_type='inner'
        )
    except Exception as e:
        logger.error(f"Test 2 failed: {e}")

    # Test 3: Aggregations
    logger.info("\n### TEST 3: Aggregations ###")
    try:
        tester.test_aggregation(
            'gsw_m2_sdss',
            'bptclass',
            '*',
            'COUNT'
        )
    except Exception as e:
        logger.error(f"Test 3 failed: {e}")

    logger.info("\n" + "="*80)
    logger.info("TESTING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    run_comprehensive_tests()
