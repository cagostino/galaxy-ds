"""WISE mid-infrared analysis for AGN selection.

Implements the analysis from Goel, Salim & Agostino (in prep):
W1-W2 color cuts, K-corrections, reliability/completeness.
"""

import numpy as np
import pandas as pd


def compute_wise_colors(df, w1_col='w1', w2_col='w2'):
    """Compute W1-W2 color and apply AGN selection flag.

    Parameters
    ----------
    df : DataFrame
        Must have W1 and W2 magnitude columns.
    w1_col, w2_col : str
        Column names.

    Returns
    -------
    df : DataFrame (modified in place)
        Adds: w1w2, w1w2_agn_flag
    """
    df['w1w2'] = df[w1_col] - df[w2_col]
    df['w1w2_agn_flag'] = df['w1w2'] > 0.5
    return df


def kcorrect_w1w2(w1w2, z):
    """K-correct W1-W2 color (Eq. 3 from the paper).

    For z <= 0.2: (W1-W2)_corr = (W1-W2) - 1.0 * z
    For z > 0.2:  (W1-W2)_corr = (W1-W2) - 0.2

    Parameters
    ----------
    w1w2 : array-like
        Observed W1-W2 color.
    z : array-like
        Redshift.

    Returns
    -------
    w1w2_corr : ndarray
    """
    w1w2 = np.asarray(w1w2, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    correction = np.where(z <= 0.2, 1.0 * z, 0.2)
    return w1w2 - correction


def apply_kcorrection(df, w1w2_col='w1w2', z_col='z'):
    """Apply K-correction to W1-W2 colors in a DataFrame.

    Adds: w1w2_kcorr, w1w2_kcorr_agn_flag
    """
    df['w1w2_kcorr'] = kcorrect_w1w2(df[w1w2_col].values, df[z_col].values)
    df['w1w2_kcorr_agn_flag'] = df['w1w2_kcorr'] > 0.5
    return df


def reliability_completeness(groups, w1w2, w1w2_cut=0.5, agn_classes=None):
    """Compute reliability and completeness for a W1-W2 cut.

    Reliability R = N(optical AGN & mid-IR AGN) / N(mid-IR AGN)
    Completeness C = N(optical AGN & mid-IR AGN) / N(optical AGN)

    Parameters
    ----------
    groups : array-like of str
        BPT classification labels.
    w1w2 : array-like
        W1-W2 color values.
    w1w2_cut : float
        Color cut threshold.
    agn_classes : list of str or None
        Which BPT classes count as "optical AGN". Defaults to ['AGN', 'Sy2', 'Seyfert'].

    Returns
    -------
    reliability : float
    completeness : float
    """
    if agn_classes is None:
        agn_classes = ['AGN', 'Sy2', 'Seyfert']

    groups = np.asarray(groups)
    w1w2 = np.asarray(w1w2, dtype=np.float64)

    opt_agn = np.isin(groups, agn_classes)
    mir_agn = w1w2 > w1w2_cut

    n_both = np.sum(opt_agn & mir_agn)
    n_mir = np.sum(mir_agn)
    n_opt = np.sum(opt_agn)

    reliability = n_both / n_mir if n_mir > 0 else 0.0
    completeness = n_both / n_opt if n_opt > 0 else 0.0

    return reliability, completeness


def scan_w1w2_cuts(groups, w1w2, cuts=None, agn_classes=None):
    """Compute R and C across a range of W1-W2 thresholds.

    Parameters
    ----------
    groups : array-like
        BPT classification labels.
    w1w2 : array-like
        W1-W2 color values.
    cuts : array-like or None
        Thresholds to evaluate. Defaults to np.arange(0.0, 1.51, 0.05).
    agn_classes : list or None

    Returns
    -------
    DataFrame with columns: cut, reliability, completeness
    """
    if cuts is None:
        cuts = np.arange(0.0, 1.51, 0.05)

    results = []
    for cut in cuts:
        r, c = reliability_completeness(groups, w1w2, w1w2_cut=cut,
                                        agn_classes=agn_classes)
        results.append({'cut': cut, 'reliability': r, 'completeness': c})

    return pd.DataFrame(results)


def bin_reliability_completeness(df, groups_col, w1w2_col, bin_col, edges,
                                 w1w2_cut=0.5, agn_classes=None):
    """Compute R and C in bins of another quantity (e.g., mass, redshift).

    Parameters
    ----------
    df : DataFrame
    groups_col : str
        Column with BPT class labels.
    w1w2_col : str
        Column with W1-W2 colors.
    bin_col : str
        Column to bin by.
    edges : array-like
        Bin edges.
    w1w2_cut : float
    agn_classes : list or None

    Returns
    -------
    DataFrame with: bin_center, reliability, completeness, n_total
    """
    results = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (df[bin_col] >= lo) & (df[bin_col] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        r, c = reliability_completeness(
            sub[groups_col].values, sub[w1w2_col].values,
            w1w2_cut=w1w2_cut, agn_classes=agn_classes,
        )
        results.append({
            'bin_center': (lo + hi) / 2,
            'bin_lo': lo,
            'bin_hi': hi,
            'reliability': r,
            'completeness': c,
            'n_total': len(sub),
        })

    return pd.DataFrame(results)
