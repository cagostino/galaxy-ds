# Diagnostic Tools for SQL Refactor Issues

## Problem Statement

The SQL-based refactor produces **different counts** for the **same selection criteria** compared to the original pandas/numpy implementation from summer 2023.

## Root Cause Analysis Strategy

We need to find WHERE in the pipeline the counts diverge. The discrepancy could be in:

1. **Data loading** - Different numbers of rows loaded from FITS files
2. **Merging/Joining** - Different join results between pandas and SQL
3. **Filtering** - Different boolean mask results
4. **Null handling** - SQL vs pandas treat NaN/NULL differently
5. **Type coercion** - Implicit type conversions differ
6. **Index handling** - DataFrame indices may cause issues

## Diagnostic Scripts

### 1. `compare_pipelines.py`
**Purpose**: Compare outputs between old and new pipelines

**Usage**:
```python
python compare_pipelines.py
```

**What it does**:
- Checks row counts for all database tables
- Compares merge statistics (`_merge` column values)
- Reports discrepancies with percentages
- Exports `pipeline_discrepancies.csv`

**Use this when**: You want to see overall differences across all tables

---

### 2. `validate_sql_pipeline.py`
**Purpose**: Run pipeline with validation checkpoints at each step

**Usage**:
```python
python validate_sql_pipeline.py
```

**What it does**:
- Saves checkpoints after each major pipeline step
- Validates merge operations
- Checks for unexpected null values
- Reports selection criteria counts
- Exports `validation_report.txt`

**Use this when**: You want to trace the pipeline step-by-step to find where things go wrong

---

### 3. `test_sql_vs_pandas.py`
**Purpose**: Directly compare SQL queries vs equivalent pandas operations

**Usage**:
```python
python test_sql_vs_pandas.py
```

**What it does**:
- Runs identical operations in SQL and pandas
- Compares result counts and values
- Tests SELECT WHERE, JOINs, and aggregations
- Identifies data type mismatches
- Highlights NaN/NULL handling differences

**Use this when**: You want to pinpoint exact SQL query that gives wrong results

---

## Debugging Workflow

### Step 1: Run Full Pipeline Validation
```bash
python validate_sql_pipeline.py 2>&1 | tee validation_output.txt
```

This will show you:
- How many rows at each step
- Which columns have nulls
- Merge statistics
- Selection criteria counts

**Look for**: Unexpected drops in row counts or high null percentages

---

### Step 2: Compare Against Database
```bash
python compare_pipelines.py 2>&1 | tee comparison_output.txt
```

This checks what's actually in the database vs what you expect.

**Look for**: Tables with wrong row counts in `pipeline_discrepancies.csv`

---

### Step 3: Test Specific Queries
Edit `test_sql_vs_pandas.py` to add your specific selection criteria:

```python
# Example: Test your exact query that's giving wrong counts
tester.test_select_where(
    'gsw_m2_sdss',
    'bpt_sn_filt_bool == 1 AND OII_3726_FLUX_SN > 2',
    df=gsw_m2_sdss_pandas  # Pass the pandas DataFrame for comparison
)
```

---

## Common Issues and Solutions

### Issue 1: Boolean Columns Stored as Integers in SQL
**Symptom**: WHERE clauses with boolean conditions return different counts

**SQL**: `bpt_sn_filt_bool = 1` (integer comparison)
**Pandas**: `bpt_sn_filt_bool == True` (boolean comparison)

**Fix**: Ensure consistent boolean handling in `insert_dataframe_to_table()`

---

### Issue 2: NULL vs NaN Handling
**Symptom**: Counts differ when columns have missing values

**SQL**: `WHERE column IS NOT NULL`
**Pandas**: `df[df['column'].notna()]`

**Issue**: SQL excludes NULLs from comparisons automatically, pandas doesn't

**Fix**: Explicitly handle NaN in pandas operations or convert to SQL NULL properly

---

### Issue 3: Duplicate Handling in Merges
**Symptom**: Merged table has more/fewer rows than expected

**Cause**:
- One-to-many relationships create duplicates
- SQL JOIN vs pandas merge handle duplicates differently
- Missing merge keys treated differently

**Debug**:
```python
validate_merge(left_df, right_df, merged_df, (left_keys, right_keys))
```

**Fix**:
- Check for duplicate keys in source tables
- Ensure merge type (inner/left/outer) is correct
- Add explicit de-duplication if needed

---

### Issue 4: Type Coercion Differences
**Symptom**: Numeric filters give different results

**SQL**: Automatically converts types in comparisons
**Pandas**: Stricter type checking

**Debug**:
```python
# Check dtypes
print(df.dtypes)

# Check for mixed types
for col in df.columns:
    types = df[col].apply(type).unique()
    if len(types) > 1:
        print(f"{col} has mixed types: {types}")
```

**Fix**: Explicitly cast columns to correct types before operations

---

### Issue 5: Index Alignment in Pandas
**Symptom**: Operations on DataFrames give unexpected results

**Cause**: Pandas aligns on index automatically, SQL doesn't have this concept

**Fix**: Always `reset_index()` before SQL operations or merges

---

## Expected Outputs

After running diagnostics, you should have:

1. **validation_report.txt** - Full pipeline trace with row counts at each step
2. **pipeline_discrepancies.csv** - Table of all count mismatches
3. **validation.log** - Detailed step-by-step log
4. **sql_vs_pandas.log** - Direct SQL vs pandas comparisons
5. **data_pipeline.log** - Runtime execution log with all merge stats

---

## How to Read the Logs

### Good Output (No Issues):
```
✓ Checkpoint 'm2_sdss_merged': 150000 rows, 250 columns
✓ Merge validation complete
  Merge statistics:
    both: 145000 (96.67%)
    left_only: 3000 (2.00%)
    right_only: 2000 (1.33%)
✓ Match: both returned 145000 rows
```

### Bad Output (Issue Found):
```
⚠ MISMATCH: Expected 145000, got 132000 (diff: -13000, -8.97%)
❌ MISMATCH: 13000 row difference
   SQL: 132000, Pandas: 145000
  Type mismatch in 'bpt_sn_filt_bool': SQL=int64, pandas=bool
  NaN counts:
    OIII_5007_FLUX: SQL=8000, pandas=12000
```

The bad output tells you:
1. **What**: 13,000 missing rows
2. **Where**: In the SQL query result
3. **Why**: Type mismatch (boolean stored as int) + different NaN handling

---

## Quick Reference: Common Queries

### Count BPT-classifiable sources:
```sql
-- SQL
SELECT COUNT(*) FROM gsw_m2_sdss WHERE bpt_sn_filt_bool = 1;

-- Pandas
len(df[df['bpt_sn_filt_bool'] == True])
```

### Count by BPT class:
```sql
-- SQL
SELECT bptclass, COUNT(*) FROM gsw_m2_sdss GROUP BY bptclass;

-- Pandas
df['bptclass'].value_counts()
```

### X-ray matches with exposure time cut:
```sql
-- SQL
SELECT COUNT(*) FROM xmm4_gsw_sdss
WHERE texp > 12589 AND texp < 31622 AND flag_sed = 0;

-- Pandas
len(df[(df['texp'] > 12589) & (df['texp'] < 31622) & (df['flag_sed'] == 0)])
```

---

## Next Steps After Finding Issues

1. **Document the exact discrepancy** with line numbers and values
2. **Determine if it's a bug or intentional change**
3. **If bug**: Fix in `data_utils.py` or `data_models.py`
4. **If intentional**: Update documentation and expected counts
5. **Re-run validation** to confirm fix
6. **Add regression test** to prevent future issues

---

## Adding Custom Tests

Add your own tests to `test_sql_vs_pandas.py`:

```python
# Test your specific selection
def test_my_selection():
    tester = SQLvsPandasTester()

    # Load pandas version
    df = pd.read_sql_query("SELECT * FROM my_table", tester.conn)

    # Apply your selection in pandas
    pandas_result = df[(df['col1'] > 10) & (df['col2'] == 'value')]

    # Compare to SQL
    sql_result = tester.test_select_where(
        'my_table',
        'col1 > 10 AND col2 = "value"'
    )

    print(f"Pandas: {len(pandas_result)}, SQL: {sql_result['sql']}")
```

---

## Reporting Issues

When reporting a discrepancy, include:

1. **Selection criteria** (exact SQL query or pandas operation)
2. **Expected count** (from old pipeline)
3. **Actual count** (from SQL pipeline)
4. **Difference** (number and percentage)
5. **Relevant log excerpts** from diagnostic tools
6. **Data sample** showing affected rows

Example:
```
ISSUE: BPT AGN count mismatch

Selection: bpt_sn_filt_bool == 1 AND bptclass == 'AGN'
Expected: 45,231 (from August 2023 pipeline)
Actual: 41,089 (from SQL pipeline)
Diff: -4,142 (-9.16%)

Log excerpt:
  Type mismatch in 'bpt_sn_filt_bool': SQL=int64, pandas=bool
  4,142 rows have bpt_sn_filt_bool stored as 0/1 instead of True/False

Sample affected rows: [see rows 100-110 in validation_report.txt]
```
