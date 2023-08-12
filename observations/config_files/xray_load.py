import sys
sys.path.append("../..")

from scipy.stats import ks_2samp
from scipy.interpolate import interp1d
from plotresults_sfrm import *
from ELObj import *
from ast_func import *
from Fits_set import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bces.bces as BCES
import emcee



merged_xr_sy2_all = pd.read_csv('../../catalogs/merged_xr_sy2_all.csv')
merged_xr_liner2_all = pd.read_csv('../../catalogs/merged_xr_hliner_all.csv')
merged_xr_sf_all = pd.read_csv('../../catalogs/merged_xr_sliner_all.csv')
merged_xr_val_all = pd.read_csv('../../catalogs/merged_xr_val_all.csv')

merged_xr_sy2_all_combo = pd.read_csv(
    '../../catalogs/merged_xr_sy2_all_combo.csv')
merged_xr_liner2_all_combo = pd.read_csv(
    '../../catalogs/merged_xr_hliner_all_combo.csv')
merged_xr_sf_all_combo = pd.read_csv(
    '../../catalogs/merged_xr_sliner_all_combo.csv')
merged_xr_val_all_combo = pd.read_csv(
    '../../catalogs/merged_xr_val_all_combo.csv')

merged_xr_sy2_all_combo_allxr = pd.read_csv(
    '../../catalogs/merged_xr_sy2_all_combo_allxr.csv')
merged_xr_liner2_all_combo_allxr = pd.read_csv(
    '../../catalogs/merged_xr_hliner_all_combo_allxr.csv')
merged_xr_sf_all_combo_allxr = pd.read_csv(
    '../../catalogs/merged_xr_sliner_all_combo_allxr.csv')
merged_xr_val_all_combo_allxr = pd.read_csv(
    '../../catalogs/merged_xr_val_all_combo_allxr.csv')

merged_xr_sy2_all_woo1 = pd.read_csv(
    '../../catalogs/merged_xr_sy2_all_woo1.csv')
merged_xr_liner2_all_woo1 = pd.read_csv(
    '../../catalogs/merged_xr_hliner_all_woo1.csv')
merged_xr_sf_all_woo1 = pd.read_csv(
    '../../catalogs/merged_xr_sliner_all_woo1.csv')
merged_xr_val2_all = pd.read_csv('../../catalogs/merged_xr_val_all_woo1.csv')

merged_xr_all = pd.read_csv('../../catalogs/merged_xr_all.csv')

merged_xr_allliners_1 = pd.concat([merged_xr_sf_all, merged_xr_liner2_all])
merged_xr_allliners_2 = pd.concat(
    [merged_xr_sf_all_woo1, merged_xr_liner2_all_woo1])
merged_xr_allliners_combo = pd.concat(
    [merged_xr_sf_all_combo, merged_xr_liner2_all_combo])
merged_xr_allliners_combo_allxr = pd.concat(
    [merged_xr_sf_all_combo_allxr, merged_xr_liner2_all_combo_allxr])
high_sn_o3_xray_sample_all = pd.read_csv(
    '../../catalogs/xragn_high_sn_o3_sample.csv')

xragn_no_sn_cuts = pd.read_csv('../../catalogs/xragn_sample_no_sn_cuts.csv')
xragn_unclass_p1 = pd.read_csv(
    '../../catalogs/xragn_sample_unclass_p1_cuts.csv')
unclass_p1_highsn_o3 = xragn_unclass_p1.iloc[np.where(
    xragn_unclass_p1.oiiiflux_sn > 2)].copy()

xragn_unclass_p2 = pd.read_csv(
    '../../catalogs/xragn_sample_unclass_p2_cuts.csv')
unclass_p2_highsn_o3 = xragn_unclass_p2.iloc[np.where(
    xragn_unclass_p2.oiiiflux_sn > 2)].copy()

xragn_bpt_sf_df = pd.read_csv('../../catalogs/xragn_bptsf.csv')
xr_bpt_sf_df = pd.read_csv('../../catalogs/xr_bptsf.csv')
xr_bptplus_sf_df = pd.read_csv('../../catalogs/xr_bptplussf.csv')
xragn_bptplus_sf_df = pd.read_csv('../../catalogs/xragn_bptplussf.csv')

xr_nii_agn_df = pd.read_csv('../../catalogs/xr_bptniiagn.csv')
xr_nii_xragn_df = pd.read_csv('../../catalogs/xragn_bptplusnii.csv')

sternobj_df_spec_xr = pd.read_csv('../../catalogs/sternxr_match.csv')
x4_all_xray = pd.read_csv('../../catalogs/x4_xray_all_sample.csv')
x4_all_xragn = pd.read_csv('../../catalogs/x4_xragn_all_sample.csv')

xr_obj_info = xragn_bpt_sf_df.iloc[28]
xu_info = xragn_unclass_p1.iloc[[22, 23, 104, 210]]

pure_xr_agn_props = {'all': merged_xr_all, 'val': merged_xr_val_all,
                     'sliner': merged_xr_sf_all, 'hliner': merged_xr_liner2_all,
                     'sy2': merged_xr_sy2_all, 'val2_s2': merged_xr_val2_all,
                     'sliner_s2': merged_xr_sf_all_woo1, 'hliner_s2': merged_xr_liner2_all_woo1,
                     'sy2_s2': merged_xr_sy2_all_woo1,
                     'all_liners': merged_xr_allliners_1,
                     'all_liners_s2': merged_xr_allliners_2,
                     'combo_sy2': merged_xr_sy2_all_combo,
                     'combo_sliner': merged_xr_sf_all_combo,
                     'combo_hliner': merged_xr_liner2_all_combo,
                     'combo_all_liners': merged_xr_allliners_combo,
                     'combo_val': merged_xr_val_all_combo
                     }


xr_agn_props = {'o3all': high_sn_o3_xray_sample_all, 'xrall': xragn_no_sn_cuts,
                'x4all': x4_all_xray, 'x4agn': x4_all_xragn,
                'x4_fx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.fullflux_sn > 2) & (x4_all_xragn.z <= 0.07))].copy(),
                'x4_hx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.hardflux_sn > 2) & (x4_all_xragn.z <= 0.07))].copy(),
                'x4_fx_allz': x4_all_xragn.iloc[np.where((x4_all_xragn.fullflux_sn > 2) & (x4_all_xragn.z <= 0.3))].copy(),
                'x4_hx_allz': x4_all_xragn.iloc[np.where((x4_all_xragn.hardflux_sn > 2) & (x4_all_xragn.z <= 0.3))].copy(),
                'x4_sn2_o3_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 2) & (x4_all_xragn.z <= 0.07))].copy(),
                'x4_sn1_o3_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.07))].copy(),
                'x4_sn2_o3_fx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 2) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.fullflux_sn > 2))].copy(),
                'x4_sn2_o3_hx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 2) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.hardflux_sn > 2))].copy(),
                'x4_sn1_o3_fx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.fullflux_sn > 2))].copy(),
                'x4_sn1_o3_hx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.hardflux_sn > 2))].copy(),

                'x4_sn0_o3_fx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 0) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.fullflux_sn > 2))].copy(),
                'x4_sn0_o3_hx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 0) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.hardflux_sn > 2))].copy(),
                'x4_sn3_o3_fx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 3) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.fullflux_sn > 2))].copy(),
                'x4_sn3_o3_hx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 3) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.hardflux_sn > 2))].copy(),
                'x4_sn3_o3_fx_allz': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 3) & (x4_all_xragn.fullflux_sn > 2))].copy(),
                'x4_sn3_o3_hx_allz': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 3) & (x4_all_xragn.hardflux_sn > 2))].copy(),
                'x4_sn3_o3_hx_allz_noext': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 3) & (x4_all_xragn.hardflux_sn > 2)& (x4_all_xragn.ext == 0))].copy(),
                'x4_sn3_o3_hx_allz_ext': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 3) & (x4_all_xragn.hardflux_sn > 2)& (x4_all_xragn.ext != 0))].copy(),

                'x4_n2agn': xr_nii_agn_df.iloc[np.where((xr_nii_agn_df.hardflux_sn > 2))].copy(),
                'x4_n2agn_xragn': xr_nii_xragn_df.iloc[np.where((xr_nii_xragn_df.hardflux_sn > 2))].copy(),

                'x4_sn3_o3_hx_allz_bptagn': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 3) & (x4_all_xragn.hardflux_sn > 2)&
                                                                       (x4_all_xragn.bptplusgroups == 'AGN')& 
                                                                       (x4_all_xragn.niiflux_sn > 2)& 
                                                                       (x4_all_xragn.halpflux_sn > 2)&
                                                                       (x4_all_xragn.hbetaflux_sn > 2)
                                                                       )].copy(),
                'x4_sn2_o3_hx_allz_bptagn': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 2) & (x4_all_xragn.hardflux_sn > 2)&
                                                                       (x4_all_xragn.bptplusgroups == 'AGN')& 
                                                                       (x4_all_xragn.niiflux_sn > 2)& 
                                                                       (x4_all_xragn.halpflux_sn > 2)&
                                                                       (x4_all_xragn.hbetaflux_sn > 2)
                                                                       )].copy(),
                
                
                
                'x4_sn1_o3_fx_allz': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.3) & (x4_all_xragn.fullflux_sn > 2))].copy(),
                'x4_sn1_o3_hx_allz': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & 
                                                                (x4_all_xragn.z <= 0.3) & 
                                                                (x4_all_xragn.hardflux_sn > 2))].copy(),
                'x4_sn1_o3_hx_allz_bpt': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 2) & 
                                                                    (x4_all_xragn.z <= 0.3) & 
                                                                    (x4_all_xragn.hardflux_sn > 2)&
                                                                    (x4_all_xragn.niiflux_sn > 2)&
                                                                    (x4_all_xragn.hbetaflux_sn > 2)&
                                                                    (x4_all_xragn.halpflux_sn > 2)
                                                                     )].copy(),
                
                'x4_sn1_o3_hx_allz_bptnii': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & 
                                                                    (x4_all_xragn.z <= 0.3) & 
                                                                    (x4_all_xragn.hardflux_sn > 2)&
                                                                    (x4_all_xragn.niiflux_sn > 2)&
                                                                    (x4_all_xragn.hbetaflux_sn >0)&
                                                                    (x4_all_xragn.halpflux_sn > 2)
                                                                     )].copy(),


                'x4_sn1_o3_hx_allz_belowlxo3': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.3) &
                                                                          (x4_all_xragn.hardflux_sn > 2) &
                                                                          (x4_all_xragn.lo3_pred_fromlx > x4_all_xragn.oiiilum))].copy(),
                'hliner_sn1_o3_hx_allz_belowlxo3': merged_xr_liner2_all_combo.iloc[np.where((merged_xr_liner2_all_combo.oiiiflux_sn > 1) & (merged_xr_liner2_all_combo.z <= 0.3) &
                                                                          (merged_xr_liner2_all_combo.hardflux_sn > 2) &
                                                                          (merged_xr_liner2_all_combo.lo3_pred_fromlx > merged_xr_liner2_all_combo.oiiilum))].copy(),
                'sliner_sn1_o3_hx_allz_belowlxo3': merged_xr_sf_all_combo.iloc[np.where((merged_xr_sf_all_combo.oiiiflux_sn > 1) & (merged_xr_sf_all_combo.z <= 0.3) &
                                                                          (merged_xr_sf_all_combo.hardflux_sn > 2) &
                                                                          (merged_xr_sf_all_combo.lo3_pred_fromlx > merged_xr_sf_all_combo.oiiilum))].copy(),
                'sy2_sn1_o3_hx_allz_belowlxo3': merged_xr_sy2_all_combo.iloc[np.where((merged_xr_sy2_all_combo.oiiiflux_sn > 1) & (merged_xr_sy2_all_combo.z <= 0.3) &
                                                                          (merged_xr_sy2_all_combo.hardflux_sn > 2) &
                                                                          (merged_xr_sy2_all_combo.lo3_pred_fromlx > merged_xr_sy2_all_combo.oiiilum))].copy(),
                'all_liners_sn1_o3_hx_allz_belowlxo3': merged_xr_allliners_combo.iloc[np.where((merged_xr_allliners_combo.oiiiflux_sn > 1) & (merged_xr_allliners_combo.z <= 0.3) &
                                                                          (merged_xr_allliners_combo.hardflux_sn > 2) &
                                                                          (merged_xr_allliners_combo.lo3_pred_fromlx > merged_xr_allliners_combo.oiiilum))].copy(),
                'x4_sn1_o3_hx_allz_abovelxo3': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.3) &
                                                                          (x4_all_xragn.hardflux_sn > 2) &
                                                                          (x4_all_xragn.lo3_pred_fromlx < x4_all_xragn.oiiilum))].copy(),
                'hliner_sn1_o3_hx_allz_abovelxo3': merged_xr_liner2_all_combo.iloc[np.where((merged_xr_liner2_all_combo.oiiiflux_sn > 1) & (merged_xr_liner2_all_combo.z <= 0.3) &
                                                                          (merged_xr_liner2_all_combo.hardflux_sn > 2) &
                                                                          (merged_xr_liner2_all_combo.lo3_pred_fromlx < merged_xr_liner2_all_combo.oiiilum))].copy(),
                'sliner_sn1_o3_hx_allz_abovelxo3': merged_xr_sf_all_combo.iloc[np.where((merged_xr_sf_all_combo.oiiiflux_sn > 1) & (merged_xr_sf_all_combo.z <= 0.3) &
                                                                          (merged_xr_sf_all_combo.hardflux_sn > 2) &
                                                                          (merged_xr_sf_all_combo.lo3_pred_fromlx < merged_xr_sf_all_combo.oiiilum))].copy(),
                'sy2_sn1_o3_hx_allz_abovelxo3': merged_xr_sy2_all_combo.iloc[np.where((merged_xr_sy2_all_combo.oiiiflux_sn > 1) & (merged_xr_sy2_all_combo.z <= 0.3) &
                                                                          (merged_xr_sy2_all_combo.hardflux_sn > 2) &
                                                                          (merged_xr_sy2_all_combo.lo3_pred_fromlx < merged_xr_sy2_all_combo.oiiilum))].copy(),
                'all_liners_sn1_o3_hx_allz_abovelxo3': merged_xr_allliners_combo.iloc[np.where((merged_xr_allliners_combo.oiiiflux_sn > 1) & (merged_xr_allliners_combo.z <= 0.3) &
                                                                          (merged_xr_allliners_combo.hardflux_sn > 2) &
                                                                          (merged_xr_allliners_combo.lo3_pred_fromlx < merged_xr_allliners_combo.oiiilum))].copy(),
               
                'x4_sn1_o3_hx_z07_belowlxo3': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.07) &
                                                                         (x4_all_xragn.hardflux_sn > 2) &
                                                                         (x4_all_xragn.lo3_pred_fromlx > x4_all_xragn.oiiilum))].copy(),
                'x4_sn1_o3_hx_allz_belowlxo3_1dex': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.3) &
                                                                               (x4_all_xragn.hardflux_sn > 2) &
                                                                               (x4_all_xragn.lo3_pred_fromlx-1 > x4_all_xragn.oiiilum))].copy(),
                'x4_sn1_o3_hx_z07_belowlxo3_1dex': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.07) &
                                                                              (x4_all_xragn.hardflux_sn > 2) &
                                                                              (x4_all_xragn.lo3_pred_fromlx-1 > x4_all_xragn.oiiilum))].copy(),
                'x4_sn_lt_1_o3_hx_allz_belowlxo3': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn < 1) & (x4_all_xragn.z <= 0.3) &
                                                                              (x4_all_xragn.hardflux_sn > 2))].copy(),
                'x4_sn_lt_1_o3_hx_z07_belowlxo3': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn < 1) & (x4_all_xragn.z <= 0.07) &
                                                                             (x4_all_xragn.hardflux_sn > 2))].copy(),
                'x4_sn5_o3_fx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 5) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.fullflux_sn > 2))].copy(),
                'x4_sn5_o3_hx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 5) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.hardflux_sn > 2))].copy(),
                'x4_sn10_o3_fx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 10) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.fullflux_sn > 2))].copy(),
                'x4_sn10_o3_hx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 10) & (x4_all_xragn.z <= 0.07) & (x4_all_xragn.hardflux_sn > 2))].copy(),
                'x4_fx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.z <= 0.07) & (x4_all_xragn.fullflux_sn > 2))].copy(),
                'x4_hx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.z <= 0.07) & (x4_all_xragn.hardflux_sn > 2))].copy(),

                'all': merged_xr_all, 
                'all_hx': merged_xr_all.iloc[np.where(merged_xr_all.hardflux_sn>2)[0]],
                'val': merged_xr_val_all,
                'sliner': merged_xr_sf_all, 'hliner': merged_xr_liner2_all,

                'mid_sn_o3': high_sn_o3_xray_sample_all.iloc[np.where((high_sn_o3_xray_sample_all.oiiiflux_sn < 10) &
                                                                      (high_sn_o3_xray_sample_all.oiiiflux_sn > 2))].copy(),
                'low_sn_o3': xragn_no_sn_cuts.iloc[np.where(xragn_no_sn_cuts.oiiiflux_sn < 2)].copy(),

                'high_sn_o3_bd': high_sn_o3_xray_sample_all.iloc[np.where((high_sn_o3_xray_sample_all.hbetaflux_sn > 10) &
                                                                          (high_sn_o3_xray_sample_all.halpflux_sn > 10) &
                                                                          (high_sn_o3_xray_sample_all.oiiiflux_sn > 10))].copy(),
                'high_sn_lx_o3_bd': high_sn_o3_xray_sample_all.iloc[np.where((high_sn_o3_xray_sample_all.hbetaflux_sn > 10) &
                                                                             (high_sn_o3_xray_sample_all.halpflux_sn > 10) &
                                                                             (high_sn_o3_xray_sample_all.oiiiflux_sn > 10) &
                                                                             (high_sn_o3_xray_sample_all.fullflux_sn > 5))].copy(),
                'high_sn_o3_bd_no_sf': high_sn_o3_xray_sample_all.iloc[np.where((high_sn_o3_xray_sample_all.hbetaflux_sn > 10) &
                                                                                (high_sn_o3_xray_sample_all.halpflux_sn > 10) &
                                                                                (high_sn_o3_xray_sample_all.oiiiflux_sn > 10) &
                                                                                (high_sn_o3_xray_sample_all.bptplusniigroups != 'HII') &
                                                                                (high_sn_o3_xray_sample_all.bptplusgroups != 'HII'))].copy(),
                'high_sn_lx_o3_bd_no_sf': high_sn_o3_xray_sample_all.iloc[np.where((high_sn_o3_xray_sample_all.hbetaflux_sn > 10) &
                                                                          (high_sn_o3_xray_sample_all.halpflux_sn > 10) &
                                                                          (high_sn_o3_xray_sample_all.oiiiflux_sn > 10) &
                                                                          (high_sn_o3_xray_sample_all.fullflux_sn > 5) &
                                                                          (high_sn_o3_xray_sample_all.bptplusniigroups != 'HII') &
                                                                          (high_sn_o3_xray_sample_all.bptplusgroups != 'HII'))].copy(),

                'sy2': merged_xr_sy2_all, 'val2_s2': merged_xr_val2_all,
                'sliner_s2': merged_xr_sf_all_woo1, 'hliner_s2': merged_xr_liner2_all_woo1,
                'sy2_s2': merged_xr_sy2_all_woo1, 'all_liners': merged_xr_allliners_1,
                'all_liners_s2': merged_xr_allliners_2,
                'unclass_p1': xragn_unclass_p1,
                'unclass_p2': xragn_unclass_p2,
                'unclass_p1_hx_o3': xragn_unclass_p1.iloc[np.where((xragn_unclass_p1.hardflux_sn>2)&(xragn_unclass_p1.oiiiflux_sn>1))],
                'unclass_p2_hx_o3': xragn_unclass_p2.iloc[np.where((xragn_unclass_p2.hardflux_sn>2)&(xragn_unclass_p2.oiiiflux_sn>1))],
                'combo_sy2': merged_xr_sy2_all_combo,
                'combo_sliner': merged_xr_sf_all_combo,
                'combo_hliner': merged_xr_liner2_all_combo,
                'combo_all_liners': merged_xr_allliners_combo,
                
                'combo_val': merged_xr_val_all_combo,
                'combo_sy2_fx_z07': merged_xr_sy2_all_combo.iloc[np.where((merged_xr_sy2_all_combo.z <= 0.07) & (merged_xr_sy2_all_combo.fullflux_sn > 2))],
                'combo_sliner_fx_z07': merged_xr_sf_all_combo.iloc[np.where((merged_xr_sf_all_combo.z <= 0.07) & (merged_xr_sf_all_combo.fullflux_sn > 2))],
                'combo_hliner_fx_z07': merged_xr_liner2_all_combo.iloc[np.where((merged_xr_liner2_all_combo.z <= 0.07) & (merged_xr_liner2_all_combo.fullflux_sn > 2))],
                'combo_all_liners_fx_z07': merged_xr_allliners_combo.iloc[np.where((merged_xr_allliners_combo.z <= 0.07) & (merged_xr_allliners_combo.fullflux_sn > 2))],
                'combo_val_fx_z07': merged_xr_val_all_combo.iloc[np.where((merged_xr_val_all_combo.z <= 0.07) & (merged_xr_val_all_combo.fullflux_sn > 2))],

                'combo_sy2_hx_z07': merged_xr_sy2_all_combo.iloc[np.where((merged_xr_sy2_all_combo.z <= 0.07) & (merged_xr_sy2_all_combo.hardflux_sn > 2))],
                'combo_sliner_hx_z07': merged_xr_sf_all_combo.iloc[np.where((merged_xr_sf_all_combo.z <= 0.07) & (merged_xr_sf_all_combo.hardflux_sn > 2))],
                'combo_hliner_hx_z07': merged_xr_liner2_all_combo.iloc[np.where((merged_xr_liner2_all_combo.z <= 0.07) & (merged_xr_liner2_all_combo.hardflux_sn > 2))],
                'combo_all_liners_hx_z07': merged_xr_allliners_combo.iloc[np.where((merged_xr_allliners_combo.z <= 0.07) & (merged_xr_allliners_combo.hardflux_sn > 2))],
                'combo_val_hx_z07': merged_xr_val_all_combo.iloc[np.where((merged_xr_val_all_combo.z <= 0.07) & (merged_xr_val_all_combo.hardflux_sn > 2))],

                'combo_sy2_fx': merged_xr_sy2_all_combo.iloc[np.where((merged_xr_sy2_all_combo.z <= 0.3) & (merged_xr_sy2_all_combo.fullflux_sn > 2))],
                'combo_sliner_fx': merged_xr_sf_all_combo.iloc[np.where((merged_xr_sf_all_combo.z <= 0.3) & (merged_xr_sf_all_combo.fullflux_sn > 2))],
                'combo_hliner_fx': merged_xr_liner2_all_combo.iloc[np.where((merged_xr_liner2_all_combo.z <= 0.3) & (merged_xr_liner2_all_combo.fullflux_sn > 2))],
                'combo_all_liners_fx': merged_xr_allliners_combo.iloc[np.where((merged_xr_allliners_combo.z <= 0.3) & (merged_xr_allliners_combo.fullflux_sn > 2))],
                'combo_val_fx': merged_xr_val_all_combo.iloc[np.where((merged_xr_val_all_combo.z <= 0.3) & (merged_xr_val_all_combo.fullflux_sn > 2))],

                'combo_sy2_hx': merged_xr_sy2_all_combo.iloc[np.where((merged_xr_sy2_all_combo.z <= 0.3) & (merged_xr_sy2_all_combo.hardflux_sn > 2))],
                'combo_sliner_hx': merged_xr_sf_all_combo.iloc[np.where((merged_xr_sf_all_combo.z <= 0.3) & (merged_xr_sf_all_combo.hardflux_sn > 2))],
                'combo_hliner_hx': merged_xr_liner2_all_combo.iloc[np.where((merged_xr_liner2_all_combo.z <= 0.3) & (merged_xr_liner2_all_combo.hardflux_sn > 2))],
                'combo_all_liners_hx': merged_xr_allliners_combo.iloc[np.where((merged_xr_allliners_combo.z <= 0.3) & (merged_xr_allliners_combo.hardflux_sn > 2))],
                'combo_val_hx': merged_xr_val_all_combo.iloc[np.where((merged_xr_val_all_combo.z <= 0.3) & (merged_xr_val_all_combo.hardflux_sn > 2))],
                
                'combo_sy2_hx_allxr': merged_xr_sy2_all_combo_allxr.iloc[np.where((merged_xr_sy2_all_combo_allxr.z <= 0.3) & (merged_xr_sy2_all_combo_allxr.hardflux_sn > 2))],
                'combo_sliner_hx_allxr': merged_xr_sf_all_combo_allxr.iloc[np.where((merged_xr_sf_all_combo_allxr.z <= 0.3) & (merged_xr_sf_all_combo_allxr.hardflux_sn > 2))],
                'combo_hliner_hx_allxr': merged_xr_liner2_all_combo_allxr.iloc[np.where((merged_xr_liner2_all_combo_allxr.z <= 0.3) & (merged_xr_liner2_all_combo_allxr.hardflux_sn > 2))],
                'combo_all_liners_hx_allxr': merged_xr_allliners_combo_allxr.iloc[np.where((merged_xr_allliners_combo_allxr.z <= 0.3) & (merged_xr_allliners_combo_allxr.hardflux_sn > 2))],
                'combo_val_hx_allxr': merged_xr_val_all_combo_allxr.iloc[np.where((merged_xr_val_all_combo_allxr.z <= 0.3) & (merged_xr_val_all_combo_allxr.hardflux_sn > 2))],

                'combo_val_high_sn_o3_bd_bpt': merged_xr_val_all_combo.iloc[np.where((merged_xr_val_all_combo.hbetaflux_sn > 10) &
                                                                                     (merged_xr_val_all_combo.halpflux_sn > 10) &
                                                                                     (merged_xr_val_all_combo.oiiiflux_sn > 10) &
                                                                                     (merged_xr_val_all_combo.hbetaflux_sub_sn > 10) &
                                                                                     (merged_xr_val_all_combo.halpflux_sub_sn > 10) &
                                                                                     (merged_xr_val_all_combo.oiiiflux_sub_sn > 10))].copy(),
                'combo_val_high_sn_lx_o3_bd_bpt': merged_xr_val_all_combo.iloc[np.where((merged_xr_val_all_combo.hbetaflux_sn > 10) &
                                                                                        (merged_xr_val_all_combo.halpflux_sn > 10) &
                                                                                        (merged_xr_val_all_combo.oiiiflux_sn > 10) &
                                                                                        (merged_xr_val_all_combo.fullflux_sn > 5) &
                                                                                        (merged_xr_val_all_combo.hbetaflux_sub_sn > 10) &
                                                                                        (merged_xr_val_all_combo.halpflux_sub_sn > 10) &
                                                                                        (merged_xr_val_all_combo.oiiiflux_sub_sn > 10))].copy(),
                'unclass_p1_highsn_o3': unclass_p1_highsn_o3,
                'unclass_p2_highsn_o3': unclass_p2_highsn_o3,
                'bpt_sf': xragn_bpt_sf_df,
                'bpt_sf_allxr': xr_bpt_sf_df.iloc[np.where(xr_bpt_sf_df.hardflux_sn>2)[0]],
                'bptplus_sf_allxr': xr_bptplus_sf_df.iloc[np.where(xr_bptplus_sf_df.hardflux_sn>2)[0]],
                'bptplus_sf_allxragn': xragn_bptplus_sf_df.iloc[np.where(xragn_bptplus_sf_df.hardflux_sn>2)[0]],

                
                'high_sn_o3_bd_bpt_sf': xragn_bpt_sf_df.iloc[np.where((xragn_bpt_sf_df.hbetaflux_sn > 10) &
                                                                      (xragn_bpt_sf_df.halpflux_sn > 10) &
                                                                      (xragn_bpt_sf_df.oiiiflux_sn > 10))].copy(),
                'high_sn_lx_o3_bd_bpt_sf': xragn_bpt_sf_df.iloc[np.where((xragn_bpt_sf_df.hbetaflux_sn > 10) &
                                                                         (xragn_bpt_sf_df.halpflux_sn > 10) &
                                                                         (xragn_bpt_sf_df.oiiiflux_sn > 10) &
                                                                         (xragn_bpt_sf_df.fullflux_sn > 5))].copy()
                }

redshifts = [0,0.05, 0.1, 0.15, 0.2, 0.25, 0.35]
redshifts_ = np.array(redshifts)+0.025
nbins=  []
bpt_3sig = []

bpt_2sig = []

n2ha= []
bpt_2sig_n2 = []

bpt_2sig_n2_o3_1 = []

bpt_2sig_n2_o3_3 = []
o3_1 = []
o3_2 = []

for i in range(len(redshifts)):
    if i == len(redshifts)-1:
        zmin = 0
        zmax=0.3
    else:
        zmin = redshifts[i]
        zmax = redshifts[i+1]
        
    x4_sf = ( (xr_agn_props['x4_hx_allz'].bptplusgroups=='HII')& ((xr_agn_props['x4_hx_allz'].oiiiflux_sn > 2) & 
                                                   (xr_agn_props['x4_hx_allz'].niiflux_sn > 2)& (xr_agn_props['x4_hx_allz'].hbetaflux_sn > 2)& 
                                                   (xr_agn_props['x4_hx_allz'].halpflux_sn > 2)
                                                   )) | (((xr_agn_props['x4_hx_allz'].niiflux_sn > 2)&(((xr_agn_props['x4_hx_allz'].oiiiflux_sn <2)  |(xr_agn_props['x4_hx_allz'].hbetaflux_sn < 2)))&
                                                   (xr_agn_props['x4_hx_allz'].halpflux_sn > 2)&( xr_agn_props['x4_hx_allz'].bptplusniigroups=='HII')))
    x4_notsf = np.logical_not(x4_sf)
    x4_noext = xr_agn_props['x4_hx_allz'].iloc[np.where(( xr_agn_props['x4_hx_allz'].ext==0)& ( xr_agn_props['x4_hx_allz'].z<=zmax) &( xr_agn_props['x4_hx_allz'].z>zmin) & (x4_notsf)   )].copy()                                           
    '''x4_noext = xr_agn_props['x4_hx_allz'].iloc[np.where(( xr_agn_props['x4_hx_allz'].ext==0)& ( xr_agn_props['x4_hx_allz'].z<=z) &                                        
                                                  (((xr_agn_props['x4_hx_allz'].oiiiflux_sn > 2) & (xr_agn_props['x4_hx_allz'].z <= z) & 
                                                   (xr_agn_props['x4_hx_allz'].niiflux_sn > 2)& (xr_agn_props['x4_hx_allz'].hbetaflux_sn > 2)& 
                                                   (xr_agn_props['x4_hx_allz'].halpflux_sn > 2)&( xr_agn_props['x4_hx_allz'].bptplusgroups!='HII'))|
                                                   ((xr_agn_props['x4_hx_allz'].niiflux_sn > 2)&(((xr_agn_props['x4_hx_allz'].oiiiflux_sn <2)  |(xr_agn_props['x4_hx_allz'].hbetaflux_sn < 2)))&
                                                   (xr_agn_props['x4_hx_allz'].halpflux_sn > 2)&( xr_agn_props['x4_hx_allz'].bptplusniigroups!='HII'))|
                                                   ((xr_agn_props['x4_hx_allz'].niiflux_sn < 2)|(xr_agn_props['x4_hx_allz'].halpflux_sn < 2)))
                                                  )].copy()'''
                                              
    x4_noextsize = len(x4_noext)
    nbins.append(x4_noextsize)
    bpt_3sig.append( np.where((x4_noext.oiiiflux_sn > 3) & (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2)& (x4_noext.niiflux_sn > 3)&
                       (x4_noext.hbetaflux_sn > 3)& (x4_noext.halpflux_sn > 3)&
                       (x4_noext.bptplusgroups=='AGN'))[0].size/x4_noextsize)
    bpt_2sig.append( np.where((x4_noext.oiiiflux_sn > 2) & (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2)& (x4_noext.niiflux_sn > 2)&
                       (x4_noext.hbetaflux_sn > 2)& (x4_noext.halpflux_sn > 2)&
                       (x4_noext.bptplusgroups=='AGN'))[0].size/x4_noextsize)
    o3_1.append( np.where((x4_noext.oiiiflux_sn > 1) & (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2))[0].size/x4_noextsize)
    o3_2.append( np.where((x4_noext.oiiiflux_sn > 2) & (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2))[0].size/x4_noextsize)
    
    bpt_2sig_n2.append( np.where( (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2)& (x4_noext.niiflux_sn > 2)&
                       (((x4_noext.oiiiflux_sn >2)  &(x4_noext.hbetaflux_sn > 2))|
                       ((x4_noext.oiiiflux_sn <2)  |(x4_noext.hbetaflux_sn < 2)))&
                       (x4_noext.halpflux_sn > 2))[0].size/x4_noextsize)
    bpt_2sig_n2_o3_1.append( np.where( (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2)& (x4_noext.niiflux_sn > 2)&
                       
                       (((x4_noext.oiiiflux_sn >2)  &(x4_noext.hbetaflux_sn > 2))|
                       ((x4_noext.oiiiflux_sn >1)  &(x4_noext.hbetaflux_sn < 2)))&
                       (x4_noext.halpflux_sn > 2))[0].size/x4_noextsize)
    bpt_2sig_n2_o3_3.append( np.where( (x4_noext.z <= zmax) & (x4_noext.z > zmin) & 
                       (x4_noext.hardflux_sn > 2)& (x4_noext.niiflux_sn > 2)&
                       
                       (((x4_noext.oiiiflux_sn >2)  &(x4_noext.hbetaflux_sn > 2))|
                       ((x4_noext.oiiiflux_sn >3)  &(x4_noext.hbetaflux_sn < 2)))&
                       (x4_noext.halpflux_sn > 2))[0].size/x4_noextsize)
        
        
       
        
    

def plot_fiducial_pure_uncorr(xr_df, bin_y=True, percentiles=True, size_y_bin=0.5, color='k', fname='', save=False):
    xmid, avg_y, avg_quant, perc16, perc84 = scatter(xr_df.full_lxagn, xr_df.oiiilum_sub,
                                                     xerr=np.vstack([xr_df.e_full_xraylum_down,
                                                                     xr_df.e_full_xraylum_up]),
                                                     yerr=np.vstack([xr_df.e_oiiilum_down_sub,
                                                                     xr_df.e_oiiilum_up_sub]),
                                                     minx=37, maxx=46, miny=37, maxy=44, aspect='equal', alpha=1,
                                                     color=color, ecolor=color, bin_y=bin_y, size_y_bin=size_y_bin, percentiles=percentiles)
    plt.xticks([38, 40, 42, 44])
    plt.yticks([38, 40, 42, 44])

    plt.xlabel(r'log(L$_{\mathrm{X}}$)')
    plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('plots/lx_o3_uncorrected_sub'+fname+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/lx_o3_uncorrected_sub'+fname+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()
    return xmid, avg_y, avg_quant, perc16, perc84


def plot_fiducial_pure(xr_df, bin_y=True, percentiles=True, size_y_bin=0.5, color='k', fname='', save=False):
    xmid, avg_y, avg_quant, perc16, perc84 = scatter(xr_df.full_lxagn, xr_df.oiiilum_sub_dered,
                                                     xerr=np.vstack([xr_df.e_full_xraylum_down,
                                                                     xr_df.e_full_xraylum_up]),
                                                     yerr=np.vstack([xr_df.e_oiiilum_down_sub_dered,
                                                                     xr_df.e_oiiilum_up_sub_dered]),
                                                     minx=37, maxx=46, miny=37, maxy=44, aspect='equal', alpha=1,
                                                     color=color, ecolor=color, bin_y=bin_y, size_y_bin=size_y_bin, percentiles=percentiles)
    plt.xticks([38, 40, 42, 44])
    plt.yticks([38, 40, 42, 44])

    plt.xlabel(r'log(L$_{\mathrm{X}}$)')
    plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('plots/lx_o3_corrected_sub'+fname+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/lx_o3_corrected_sub'+fname+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()
    return xmid, avg_y, avg_quant, perc16, perc84


def plot_fiducial_uncorr(xr_df, bin_y=True, percentiles=True, size_y_bin=0.5, color='k', fname='', save=False):
    xmid, avg_y, avg_quant, perc16, perc84 = scatter(xr_df.full_lxagn, xr_df.oiiilum_uncorr,
                                                     xerr=np.vstack([xr_df.e_full_xraylum_down,
                                                                     xr_df.e_full_xraylum_up]),
                                                     yerr=np.vstack([xr_df.e_oiiilum_down_uncorr,
                                                                     xr_df.e_oiiilum_up_uncorr]),
                                                     minx=37, maxx=46, miny=37, maxy=44, aspect='equal', alpha=1,
                                                     color=color, ecolor=color, bin_y=bin_y, size_y_bin=size_y_bin, percentiles=percentiles)
    plt.xticks([38, 40, 42, 44])
    plt.yticks([38, 40, 42, 44])

    plt.xlabel(r'log(L$_{\mathrm{X}}$)')
    plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('plots/lx_o3_uncorrected_'+fname+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/lx_o3_uncorrected_'+fname+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()
    return xmid, avg_y, avg_quant, perc16, perc84


def plot_fiducial(xr_df, bin_y=True, percentiles=True, size_y_bin=0.5, color='k', fname='', save=False):
    xmid, avg_y, avg_quant, perc16, perc84 = scatter(xr_df.full_lxagn, xr_df.oiiilum,
                                                     xerr=np.vstack([xr_df.e_full_xraylum_down,
                                                                     xr_df.e_full_xraylum_up]),
                                                     yerr=np.vstack([xr_df.e_oiiilum_down,
                                                                     xr_df.e_oiiilum_up]),
                                                     minx=37, maxx=46, miny=37, maxy=44, aspect='equal', alpha=1,
                                                     color=color, ecolor=color, bin_y=bin_y, size_y_bin=size_y_bin, percentiles=percentiles)
    plt.xticks([38, 40, 42, 44])
    plt.yticks([38, 40, 42, 44])

    plt.xlabel(r'log(L$_{\mathrm{X}}$)')
    plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('plots/lx_o3_corrected_'+fname+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/lx_o3_corrected_'+fname+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()
    return xmid, avg_y, avg_quant, perc16, perc84


def plot_fwd_results(fwd_model, xr_df, filename):
    minks = np.argmin(fwd_model[-2])
    plt.plot(fwd_model[0], fwd_model[-2])
    plt.ylim([0, 1])
    plt.xlabel('Scatter Factor')
    plt.ylabel('KS-statistic')
    plt.tight_layout()
    plt.savefig('plots/scat_ks_' + filename+'.pdf',
                dpi=250, format='pdf', bbox_inches='tight')
    plt.savefig('plots/scat_ks_' + filename+'.png',
                dpi=250, format='png', bbox_inches='tight')
    plt.close()

    plothist(xr_df.oiiiflux, range=(-1e-13, 1e-13),
             bins=1000, cumulative=True, label='Real')
    plothist(fwd_model[1][minks], range=(-1e-13, 1e-13),
             bins=1000, cumulative=True, label='Simulated')

    plt.xlabel(r'F$_{\mathrm{[OIII]}}$')
    plt.ylabel(r'Cumulative Counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/cumulative_flux_' + filename+'.pdf',
                dpi=250, format='pdf', bbox_inches='tight')
    plt.savefig('plots/cumulative_flux_' + filename+'.png',
                dpi=250, format='png', bbox_inches='tight')
    plt.close()

    area_1 = plothist(xr_df.oiiilum-np.array(fwd_model[4][minks]), range=(-6, 6),
                      bins=30, normed=True, integrate=True)
    area_2 = plothist(np.array(fwd_model[2][minks])-np.array(fwd_model[4][minks]), range=(-6, 6),
                      bins=30, density=True, normed=True, integrate=True)
    plt.close()

    area_1 = plothist(xr_df.oiiilum-np.array(fwd_model[4][minks]), range=(-6, 6),
                      bins=30, normed=True, integrate=True, label='Real, area='+str(area_1)[0:4])
    area_2 = plothist(np.array(fwd_model[2][minks])-np.array(fwd_model[4][minks]), range=(-6, 6),
                      bins=30, density=True, normed=True, integrate=True, label='Simulated, area='+str(area_2)[0:4])

    plt.xlabel(r'$\Delta$log(L$_{\mathrm{[OIII]}}$)')
    plt.ylabel('Normalized Counts')

    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/delta_lo3_' + filename+'.pdf',
                dpi=250, format='pdf', bbox_inches='tight')
    plt.savefig('plots/delta_lo3_' + filename+'.png',
                dpi=250, format='png', bbox_inches='tight')
    plt.close()


def get_f_underlum(obs, nondetect, gauss):
    f_ul = 1-gauss.size/(obs.size+nondetect.size)
    return f_ul


'''


fig  = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(211)
ax1.set_aspect(2)
ax1.plot(xinp, gauss_ideal/np.max(gauss_ideal), '-.',label='Ideal Gaussian', color='b')
ax1.axvline(x=0.587, label=r'1 $\sigma$')
plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_offset, range=(0, 3), 
         bins=10, norm0=True, normed=True, label='X-ray AGN Candidates')
ax1.legend(fontsize=20)
ax2 = fig.add_subplot(212)
ax2.set_aspect(2)
ax2.plot(xinp, gauss_ideal/np.max(gauss_ideal), '-.', color='b')

#bncenters, cnt1, int_ = plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_offset.iloc[np.where(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].ext==0)[0]], range=(0, 3), bins=10, norm0=False, normed=False, label='Point Source')
plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_offset.iloc[np.where(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].ext==0)[0]], 
         range=(0, 3), bins=10, norm0=True, normed=True, label='Point Source', c='cyan')

bncenters,cnt2, int_ = plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_offset.iloc[np.where(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].ext!=0)[0]],
                                range=(0, 3), bins=10, normval=np.max(cnt1), normed=True, label='Extended', c='magenta')

plt.tight_layout()
ax2.legend(fontsize=20)
ax2.axvline(x=0.587)

ax1.set_xlabel('')
ax1.set_xticks([0.5, 1, 1.5, 2, 2.5])
ax2.set_ylim([0,1.05])
ax1.set_ylim([0,1.05])

ax1.set_xticklabels('')
ax2.set_xticks([0.5, 1, 1.5, 2, 2.5])

ax1.set_xlim([0,3])
ax2.set_xlim([0,3])


ax2.set_xlabel('$\Delta$log(L$_{\mathrm{[OIII]}}$)')

ax1.set_ylabel('Norm. Counts')
ax2.set_ylabel('Norm. Counts')

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig('plots/hist_delta_lo3.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hist_delta_lo3.png', bbox_inches='tight', dpi=250)



fig  = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.plot(x_panessa, geny_, '--',color='gray', linewidth=3)

scatter(xr_agn_props['x4_sn3_o3_hx_allz'].hard_xraylum, xr_agn_props['x4_sn3_o3_hx_allz'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=43.8, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True,
             color='k', edgecolor='none', fig=fig, ax=ax1)

ax1.plot(xliu_hard_dc[yavliu_hard_dc>0], yavliu_hard_dc[yavliu_hard_dc>0], color='c', linewidth=3, label='Type 1')
ax1.plot(xt2_hard, yavt2_hard, color='k', linewidth=3, label='Type 2')


ax1.plot(xliner_hard, yavliner_hard, color='orange', linewidth=3, label='LINER')
ax1.plot(xsy2_hard, yavsy2_hard, color='b', linewidth=3, label='Sy2')

scatter(xr_agn_props['combo_all_liners_hx'].hard_xraylum, xr_agn_props['combo_all_liners_hx'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=43.8, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True,
             color='orange', edgecolor='none', fig=fig, ax=ax1)
scatter(xr_agn_props['combo_sy2_hx'].hard_xraylum, xr_agn_props['combo_sy2_hx'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=43.8, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, 
             color='blue', edgecolor='none', fig=fig, ax=ax1)


ax1.set_xticks([38,40,42,44])
ax1.legend(fontsize=12)
#plt.yticks([38,40,42,44])
#plt.xlabel(r'log(L$_{\mathrm{X,\ 2-10\ keV}}$)')
ax1.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')

plt.tight_layout()
ax1.set_aspect('equal')


ax3.set_xlabel('log(L$_{\mathrm{X,\ 2-10\ keV}}$)')

ax2.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
ax3.set_ylabel(r'log(L$_{\mathrm{[OIII]}}$)')


scatter(EL_4xmm_all.EL_gsw_df.hard_xraylum.iloc[set_fullran], EL_4xmm_all.EL_gsw_df.oiiilum.iloc[set_fullran],
        minx=38, maxx=46, miny=38, maxy=43.8, label='X-ray AGN Candidates', s=5,edgecolor='k', facecolor='none', fig = fig, ax = ax2)

ax2.plot(x_panessa, geny_, '--',color='gray', linewidth=3, label='Panessa+06')
ax2.plot(x_panessa+0.587, geny_, 'r--', linewidth=3, label=r'Panessa+06 +1$\sigma$')
ax2.set_aspect('equal')
handles, labels = ax2.get_legend_handles_labels()
order = [2,1,0]
ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=12)


ax3.plot(x_panessa, geny_, '--',color='gray', linewidth=3)
ax3.plot(x_panessa+0.587, geny_, 'r--', linewidth=3)
scatter(EL_4xmm_all.EL_gsw_df.hard_xraylum.iloc[set_fullran_noext], EL_4xmm_all.EL_gsw_df.oiiilum.iloc[set_fullran_noext], 
        minx=38, maxx=46, miny=38, maxy=43.8, label='Point Source', facecolor='cyan', edgecolor='cyan', marker='s',s=10,fig = fig, ax = ax3)
scatter(EL_4xmm_all.EL_gsw_df.hard_xraylum.iloc[set_fullran_ext], EL_4xmm_all.EL_gsw_df.oiiilum.iloc[set_fullran_ext], 
        minx=38, maxx=46, miny=38, maxy=43.8, facecolor='none', 
        edgecolor='magenta',marker='^',color='none', label='Extended',s=15, fig = fig, ax = ax3)
ax3.legend(fontsize=12)

plt.tight_layout()

ax1.set_xlabel('')
ax3.set_xticks([38,40,42,44])
ax2.set_xticks([38,40,42,44])

ax1.set_xticklabels('')
ax2.set_xticklabels('')

plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig('plots/lxo3_full_selection_.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxo3_full_selection_.png', bbox_inches='tight', dpi=250)





xmido3,yavgo3err, av__, perc16o3err, perc84o3err= scatter(xragn_no_sn_cuts.oiiiflux, 
                                                    np.log10(xragn_no_sn_cuts.oiii_err),
                                                    aspect='auto', bin_y=True, 
                                                    percentiles=True, size_y_bin=1e-17, 
                                                    counting_thresh=0, minx=-1e-16, maxx=1e-13, 
                                                    miny=-18, maxy=-14)
xmido3,yavgo3err, av__, perc16o3err, perc84o3err= scatter(np.log10(xragn_no_sn_cuts.oiiiflux-np.min(xragn_no_sn_cuts.oiiiflux)*1.2), 
                                                    np.log10(xragn_no_sn_cuts.oiii_err),
                                                    aspect='auto', bin_y=True, 
                                                    percentiles=True, size_y_bin=0.2, 
                                                    counting_thresh=1, minx=-18, maxx=-12, 
                                                    miny=-18, maxy=-14)
xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['high_sn_o3_bd_no_sf'], 
                                                       fname='high_sn_o3_bd_no_sf', 
                                                       save=False)

plt.close()

a = fwd_model(xr_agn_props['high_sn_o3_bd_no_sf'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['high_sn_o3_bd_no_sf'], 'high_sn_o3_bd_no_sf')

xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['high_sn_lx_o3_bd_no_sf'], 
                                                       fname='high_sn_lx_o3_bd_no_sf', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['high_sn_lx_o3_bd_no_sf'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['high_sn_lx_o3_bd_no_sf'], 'high_sn_lx_o3_bd_no_sf')

xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['o3all'], 
                                                       fname='o3all', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['o3all'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['o3all'], 'o3all')

xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['xrall'], 
                                                       fname='xrall', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['xrall'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['xrall'], 'xrall')

xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['all'], 
                                                       fname='all', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['all'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['all'], 'all')


xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['combo_sy2'], 
                                                       fname='combo_sy2', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['combo_sy2'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)

plot_fwd_results(a, xr_agn_props['combo_sy2'], 'combo_sy2')



xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['combo_all_liners'], 
                                                       fname='combo_all_liners', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['combo_all_liners'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['combo_all_liners'], 'combo_all_liners')


xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['combo_sliner'], 
                                                       fname='combo_sliner', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['combo_sliner'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['combo_sliner'], 'combo_sliner')

xmid, avg_y, avg_quant, perc16, perc84 = plot_fiducial(xr_agn_props['combo_hliner'], 
                                                       fname='combo_hliner', 
                                                       save=False)
plt.close()

a = fwd_model(xr_agn_props['combo_hliner'], 
              xmid, avg_y, perc16, perc84, xmido3, 
              yavgo3err, perc16o3err, perc84o3err)
plot_fwd_results(a, xr_agn_props['combo_hliner'], 'combo_hliner')
'''


def flux_distros(lum_pred, dists, av):
    perturbs = np.abs(np.random.normal(size=len(dists), scale=0.58))
    lum_pert = lum_pred-perturbs
    flux_pert = getfluxfromlum(10**lum_pert, dists)
    redflux = redden(flux_pert, av, 5007.)
    return redflux


def bootstrapped_underlum_fracs(nboot=1000000):
    np.random.seed(13)
    f_uls_z07 = []
    f_uls_allz = []
    f_1s_z07 = []
    f_1s_allz = []

    n_z07_ul = len(xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3']) + \
        len(xr_agn_props['x4_sn_lt_1_o3_hx_z07_belowlxo3'])
    n_allz_ul = len(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']) + \
        len(xr_agn_props['x4_sn_lt_1_o3_hx_allz_belowlxo3'])

    n_z07_1 = len(xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3_1dex']) + \
        len(xr_agn_props['x4_sn_lt_1_o3_hx_z07_belowlxo3'])
    n_allz_1 = len(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3_1dex']) + \
        len(xr_agn_props['x4_sn_lt_1_o3_hx_allz_belowlxo3'])

    for i in range(nboot):
        gauss_z07 = np.random.normal(
            loc=0, scale=0.59, size=xr_agn_props['x4_hx_z07'].shape[0])
        gauss_allz = np.random.normal(
            loc=0, scale=0.59, size=xr_agn_props['x4_hx_allz'].shape[0])
        npos_z07 = np.where(gauss_z07 >= 0)[0].size
        npos_allz = np.where(gauss_allz >= 0)[0].size
        f_ul_z07 = 1-npos_z07/(n_z07_ul)
        f_ul_allz = 1-npos_allz/(n_allz_ul)
        f_uls_z07.append(f_ul_z07)
        f_uls_allz.append(f_ul_allz)

        n1_z07 = np.where(gauss_z07 >= 1)[0].size
        n1_allz = np.where(gauss_allz >= 1)[0].size
        f_1_z07 = 1-n1_z07/(n_z07_1)
        f_1_allz = 1-n1_allz/(n_allz_1)
        f_1s_z07.append(f_1_z07)
        f_1s_allz.append(f_1_allz)


def gaussian(x, sigma, mean, factor=1):
    return factor*np.exp(-(x-mean)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)


def integrate(x, y):
    int_ = scipy.integrate.simps(y, x=x)
    return int_


def mocksim(xrlum, z, av, f_ul,  sig2, sl2, int2, sl1=1.22, int1=7.55, sig1=0.587, n_samps=100):

    sampd = np.random.uniform(size=(n_samps, len(xrlum)))
    mocklums = np.zeros_like(sampd)
    xrlums = np.zeros_like(sampd)

    mockfluxes = np.zeros_like(sampd)
    for i in range(n_samps):
        regsampd = np.where(sampd[i] > f_ul)[0]
        mock_sim = np.where(sampd[i] <= f_ul)[0]
        mocklums[i, regsampd] = (xrlum[regsampd]+int1)/sl1 - \
            np.abs(np.random.normal(scale=0.58, size=len(regsampd)))
        mocklums[i, mock_sim] = (xrlum[mock_sim]+int2)/sl2 + \
            np.random.normal(scale=sig2, size=len(mock_sim))
        mockfluxes[i] = redden(getfluxfromlum(10**mocklums[i], z), av, 5007.)
    return mocklums, mockfluxes


def lnprior(p):
    # The parameters are stored as a vector of values, so unpack them
    ful, sig2, sl2, int2 = p
    # We're using only uniform priors, and only eps has a lower bound
    if sig2 <= 0.3 or sig2 > 0.8 or sl2 <= 3.8 or sl2 > 4.2 or ful < 0.05 or ful > 0.4 or int2 > 112 or int2 < 108:
        return -np.inf
    return 0


def lnlike(p, xrlum, z, av, oiiiflux):
    ful, sig2, sl2, int2 = p
    modellum, modelflux = mocksim(xrlum, z, av, ful, sig2, sl2, int2)
    # the likelihood is sum of the lot of normal distributions
    ks_ = ks_2samp(np.array(oiiiflux), modelflux.flatten())[0]
    return -ks_


def lnprob(p, xrlum, z, av, oiiiflux):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p,  xrlum, z, av, oiiiflux)


def getks(x1, x2):
    ks_stat = ks_2samp(x1, x2)[0]
    return ks_stat


'''
np.random.seed(13)

import scipy.optimize as opt
nll = lambda *args: -lnprob(*args)
result = opt.minimize(nll, [0.23,0.5, 4, 110],
                      args=(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['z']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['corrected_presub_av']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['oiiiflux']),
                                      ))

ndim, nwalkers = 4, 500
p0 = [result['x']+1e-2*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,
                                args=(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['z']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['corrected_presub_av']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['oiiiflux']),
                                      ))

pos,prob,state = sampler.run_mcmc(p0, 1000, progress=True)

offsets = np.arange(0.,5,0.1)
fuls = np.arange(0.1, 0.5, 0.01)
sigmas = np.arange(0.2, 0.6, 0.02)
sl2 = np.arange(3.,5, 0.1)
int2 = np.arange(60,150,1)
sampgrid = np.meshgrid( fuls, sigmas, sl2, int2)

fuls = sampgrid[0]
sigmas=sampgrid[1]
sl2 = sampgrid[2]
int2 = sampgrid[3]

ks_samps = np.copy(sampgrid[0])*-999
ks_sampslum = np.copy(sampgrid[0])*-999

pvals = np.copy(ks_samps)
sh_samp = sampgrid[0].shape
print(len(np.ravel(sampgrid[0])))

for i in range(len(np.ravel(sampgrid[0]))):
    if i %100==0:
        print(i)
    unraved = np.unravel_index(i, sampgrid[0].shape)
    modellums, modelflux = mocksim(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['hard_xraylum']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['z']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['corrected_presub_av']),
                                      fuls[unraved],   sigmas[unraved], sl2[unraved], int2[unraved], n_samps=5)
    ks_ = ks_2samp(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['oiiiflux']), np.ravel(modelflux))
    ks_lums = ks_2samp(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['oiiilum']), np.ravel(modellums))
    
    ks_samps[unraved] =ks_[0]
    ks_sampslum[unraved] =ks_lums[0]
    
    pvals[unraved] = ks_[1]
    
    
ks_ = ks_2samp(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiiflux), mockfluxes)
ks_samps_tot = ks_samps+ks_sampslum

ds = np.argsort(ks_samps.flatten())
ds = np.argsort(ks_sampslum.flatten())
ds = np.argsort(ks_samps_tot.flatten())
i=0
mocklums, mockfluxes = mocksim(np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['hard_xraylum']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['z']),
                                      np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['corrected_presub_av']),
                                      fuls.flatten()[ds[i]],  sigmas.flatten()[ds[i]], sl2.flatten()[ds[i]], int2.flatten()[ds[i]], 
                                      n_samps=100)
xray_lm = np.zeros_like(mocklums)

for k in range(mockfluxes.shape[0]):
    xray_lm[k] = np.array(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum'])
    

print(ks_samps.flatten()[ds[i]],ks_sampslum.flatten()[ds[i]],fuls.flatten()[ds[i]],  sigmas.flatten()[ds[i]], sl2.flatten()[ds[i]], int2.flatten()[ds[i]])
scatter(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiilum, minx=38, maxx=46, miny=38, maxy=46)
plt.figure()
scatter(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], mocklums, minx=38, maxx=46, miny=38, maxy=46)

plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiilum, range=(38,43), bins=15,  normed=True, label='Observed', linestyle='-')
plothist(mocklums, range=(38,43), bins=15,label='Simulated', normed=True)
plt.xlim([38,43])
plt.ylim([0,1])

plt.xlabel('log(L[OIII])')
plt.ylabel('Counts')
plt.legend()

plt.tight_layout()

plt.savefig('plots/lum_hist_bestfit_comboks.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lum_hist_bestfit_comboks.png', bbox_inches='tight', dpi=250)

plothist(np.log10(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiiflux), cumulative=True, normed=True, linestyle='-', reverse=True, label='Observed')
plothist(np.log10(mockfluxes), cumulative=True, normed=True, reverse=True, label='Simulated')

plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Counts')
plt.xlim([-13,-17])
plt.ylim([0,1])

plt.legend()
plt.tight_layout()

plt.savefig('plots/flux_hist_bestfit_fluxks.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/flux_hist_bestfit_fluxks.png', bbox_inches='tight', dpi=250)

plot2dhist(np.ravel(xray_lm), np.ravel(mocklums), minx=38, maxx=46, miny=38, maxy=46, setplotlims=True, lim=True, nx=200, ny=200)

scatter(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiilum, minx=38, maxx=46, miny=38, maxy=46)

plt.xlabel('log(Lx)')
plt.ylabel('log(L[OIII])')
plt.xlim([40,46])
plt.ylim([38,44])

plt.tight_layout()

plt.savefig('plots/lxlo3_bestfit_comboks.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxlo3_bestfit_comboks.png', bbox_inches='tight', dpi=250)




plot2dhist(np.ravel(xray_lm), np.ravel(mocklums), minx=38, maxx=46, miny=38, maxy=46, setplotlims=True, lim=True, nx=200, ny=200)

scatter(xr_agn_props['sy2_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], 
        xr_agn_props['sy2_sn1_o3_hx_allz_belowlxo3'].oiiilum, minx=38, maxx=46, miny=38, maxy=46, label='Sy2 below', edgecolor='b', facecolor='b')
scatter(xr_agn_props['hliner_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], 
        xr_agn_props['hliner_sn1_o3_hx_allz_belowlxo3'].oiiilum, minx=38, maxx=46, miny=38, maxy=46, label='H-LINERs below', edgecolor='cyan', facecolor='cyan')
scatter(xr_agn_props['sliner_sn1_o3_hx_allz_belowlxo3']['full_xraylum'], 
        xr_agn_props['sliner_sn1_o3_hx_allz_belowlxo3'].oiiilum, minx=38, maxx=46, miny=38, maxy=46, label='S-LINERs below', edgecolor='orange', facecolor='orange')

plt.plot(x_panessa, geny_, 'k--', linewidth=3, label='Panessa+06 without QSOs')
plt.xlabel('log(Lx)')
plt.ylabel('log(L[OIII])')
plt.xlim([40,46])
plt.ylim([38,44])
plt.legend()
plt.tight_layout()

plt.savefig('plots/lxlo3_all_groups_below_relation.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxlo3_all_groups_below_relation.png', bbox_inches='tight', dpi=250)



plot2dhist(np.ravel(xray_lm), np.ravel(mocklums), minx=38, maxx=46, miny=38, maxy=46, setplotlims=True, lim=True, nx=200, ny=200)

scatter(xr_agn_props['sy2_sn1_o3_hx_allz_abovelxo3']['full_xraylum'], xr_agn_props['sy2_sn1_o3_hx_allz_abovelxo3'].oiiilum, 
        minx=38, maxx=46, miny=38, maxy=46, label='Sy2 above', edgecolor='b', facecolor='b')
plt.plot(x_panessa, geny_, 'k--', linewidth=3, label='Panessa+07 without QSOs')
plt.xlabel('log(Lx)')
plt.ylabel('log(L[OIII])')
plt.xlim([40,46])
plt.ylim([38,44])

plt.tight_layout()

plt.savefig('plots/lxlo3_sy2_above_relation.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lxlo3_sy2_above_relation.png', bbox_inches='tight', dpi=250)


x = np.linspace(-5,5, 100000)
gauss_ideal = gaussian(x,0.587, 0,341*2)
hist_z07 = np.histogram(xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].lo3_offset, range=(0, 3),bins=15)[0]
hist_allz = np.histogram(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_offset, range=(0, 3.0),bins=15)[0]

+7.55)/(1.22)
fact= 2.0064796705359496
xinp = np.arange(0,3,0.2)+0.1

pnts_by_bin_g = []
gauss_bnc_allz = gaussian(xinp, 0.587,0, factor=341/2.5)
gauss_bnc_z07 = gaussian(xinp, 0.587,0, factor=124/2.5)

pois_g_z07 = np.random.poisson(lam=gauss_bnc_z07, size=(1000000, len(gauss_bnc_z07)))
pois_g_allz = np.random.poisson(lam=gauss_bnc_allz, size=(1000000, len(gauss_bnc_allz)))

pois_o_z07 = np.random.poisson(lam=hist_z07, size=(1000000, len(hist_z07)))
pois_o_allz = np.random.poisson(lam=hist_allz, size=(1000000, len(hist_allz)))

pois_nd_z07 = np.random.poisson(lam=5, size=(1000000))
pois_nd_allz = np.random.poisson(lam=37, size=(1000000))
pois_tot_allz = np.random.poisson(lam=638, size=(1000000))


x_ul2 = np.where(xinp>1)[0]



o_all_z =412
nd_all_z=37
g_all_z = 355

o_all_z =318
nd_all_z=12
g_all_z = 321


o_z07 = 147
nd_z07= 5
g_z07 = 124

o_all_z2 = 87
nd_all_z2= 37
g_all_z2 = 30

o_z072 = 28
nd_z072= 5
g_z072 = 11


f_ul07_g_pois = 1-(np.sum(pois_g_z07, axis=1))/(o_z07+nd_z07)
f_ulall_g_pois = 1-(np.sum(pois_g_allz, axis=1))/(o_all_z+nd_all_z)

f_ul207_g_pois = (o_z072+nd_z072 -np.sum(pois_g_z07[:, x_ul2], axis=1))/(o_z07+nd_z07)
f_ul2all_g_pois = (o_all_z2+nd_all_z2-np.sum(pois_g_allz[:, x_ul2], axis=1))/(o_all_z+nd_all_z)

        

f_ul07_o_nd_pois = 1-g_z07/(np.sum(pois_o_z07, axis=1)+pois_nd_z07)
f_ulall_o_nd_pois = 1-g_all_z/(np.sum(pois_o_allz, axis=1)+pois_nd_allz)


f_ul207_o_nd_pois = (np.sum(pois_o_z07[:, x_ul2], axis=1)+pois_nd_z07-g_z072)/(np.sum(pois_o_z07, axis=1)+pois_nd_z07)
f_ul2all_o_nd_pois = (np.sum(pois_o_allz[:, x_ul2], axis=1)+pois_nd_allz-g_all_z2)/(np.sum(pois_o_allz, axis=1)+pois_nd_allz)


plothist(f_ul07_g_pois, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul_z07_pois_g.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_z07_pois_g.png', bbox_inches='tight', dpi=250)


plothist(f_ulall_g_pois, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul_allz_pois_g.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_allz_pois_g.png', bbox_inches='tight', dpi=250)



        
plothist(f_ul207_g_pois, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_z07_pois_g.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_z07_pois_g.png', bbox_inches='tight', dpi=250)


plothist(f_ul2all_g_pois, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_allz_pois_g.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_allz_pois_g.png', bbox_inches='tight', dpi=250)


plothist(f_ul07_o_nd_pois, bins=20, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul_z07_pois_o_nd.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_z07_pois_o_nd.png', bbox_inches='tight', dpi=250)


plothist(f_ulall_o_nd_pois, bins=20, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul_allz_pois_o_nd.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_allz_pois_o_nd.png', bbox_inches='tight', dpi=250)



        
plothist(f_ul207_o_nd_pois, bins=20, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_z07_pois_o_nd.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_z07_pois_o_nd.png', bbox_inches='tight', dpi=250)


plothist(f_ul2all_o_nd_pois, bins=20, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_allz_pois_o_nd.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_allz_pois_o_nd.png', bbox_inches='tight', dpi=250)




fuls_z07 = []
fuls_allz = []




for i in range(10000000):
    rand07 = np.random.normal(loc=g_z07, scale=np.sqrt(g_z07))
    randall = np.random.normal(loc=g_all_z, scale=np.sqrt(g_all_z))
    ful07 = 1-(rand07)/(o_z07+nd_z07)
    fulall = 1-(randall)/(o_all_z+nd_all_z)
    fuls_z07.append(ful07)
    fuls_allz.append(fulall)
    
        



o_all_z2 = 87
nd_all_z2= 37
g_all_z2 = 30

o_z072 = 28
nd_z072= 5
g_z072 = 11


fuls_z072 = []
fuls_allz2 = []

for i in range(10000000):
    rand07 = np.random.normal(loc=g_z072, scale=np.sqrt(g_z072))
    randall = np.random.normal(loc=g_all_z2, scale=np.sqrt(g_all_z2))
    ful07 = (o_z072+nd_z072-rand07)/(o_z07+nd_z07)
    fulall = (o_all_z2+nd_all_z2 -randall)/(o_all_z+nd_all_z)
    fuls_z072.append(ful07)
    fuls_allz2.append(fulall)
    
        
plothist(fuls_z072, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_z07.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_z07.png', bbox_inches='tight', dpi=250)



  
        
plothist(fuls_allz2, bins=75, range=(0,0.5))
plt.xlabel(r'f$_{\mathrm{UL2}}$')
plt.ylabel(r'Counts')
plt.tight_layout()

plt.savefig('plots/f_ul2_allz.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul2_allz.png', bbox_inches='tight', dpi=250)



for i in range(5,50,5):
    norm_int = plothist(xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].lo3_offset, bins=i, range=(0,3), normed=True, norm0=True, integrate=True, label='Observed')
    gauss_int = plothist(gauss, bins=i, range=(0,3), normed=True, norm0=True, integrate=True, label='Gaussian')

    plt.legend()
    plt.xlabel('$\Delta$L$_{\mathrm{[OIII]}}$')
    plt.ylabel('Counts')
    plt.tight_layout()
    binwidth = str(round(3/i, 3))
    print(binwidth, norm_int, gauss_int, (norm_int-gauss_int)/gauss_int)    
    plt.savefig('plots/lo3_offset_hist_z07_binwid_'+binwidth+'.pdf', bbox_inches='tight', dpi=250, format='pdf')
    plt.savefig('plots/lo3_offset_hist_z07_binwid_'+binwidth+'.png', bbox_inches='tight', dpi=250)
    plt.close()

for i in range(5,50,5):
    norm_int = plothist(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_offset, bins=i, range=(0,3), normed=True, norm0=True, integrate=True, label='Observed')
    gauss_int = plothist(gauss, bins=i, range=(0,3), normed=True, norm0=True, integrate=True, label='Gaussian')
    
    plt.legend()
    plt.xlabel('$\Delta$L$_{\mathrm{[OIII]}}$')
    plt.ylabel('Counts')
    plt.tight_layout()
    binwidth = str(round(3/i, 3))
    print(binwidth, norm_int, gauss_int, (norm_int-gauss_int)/gauss_int)    
    plt.savefig('plots/lo3_offset_hist_binwid_'+binwidth+'.pdf', bbox_inches='tight', dpi=250, format='pdf')
    plt.savefig('plots/lo3_offset_hist_binwid_'+binwidth+'.png', bbox_inches='tight', dpi=250)
    plt.close()
    
flux_sim_allz = flux_distros(xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].lo3_pred_fromlx, xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].z, xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].corrected_presub_av)
flux_sim_z07 = flux_distros(xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].lo3_pred_fromlx, xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].z, xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].corrected_presub_av)
flux_real_allz = xr_agn_props['x4_sn1_o3_hx_allz_belowlxo3'].oiiiflux
flux_real_z07 = xr_agn_props['x4_sn1_o3_hx_z07_belowlxo3'].oiiiflux

plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=40, range=(-17,-13), label='Observed', normed=True)
plothist(np.log10(flux_sim_allz), cumulative=True, reverse=True, bins=40, range=(-17,-13), label='Simulated', normed=True)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/fo3_cum_dist_allz.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/fo3_cum_dist_allz.png', bbox_inches='tight', dpi=250)
plt.close()

plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=40, range=(-17,-13), label='Observed', normed=True)
plothist(np.log10(flux_sim_z07), cumulative=True, reverse=True, bins=40, range=(-17,-13), label='Simulated', normed=True)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/fo3_cum_dist_z07.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/fo3_cum_dist_z07.png', bbox_inches='tight', dpi=250)
plt.close()

  , cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.225, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_z07_scaled_15_-0.225.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_z07_scaled_15_-0.225.png', bbox_inches='tight', dpi=250)



bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.225, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_hist_z07_scaled_15_-0.225.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_hist_z07_scaled_15_-0.225.png', bbox_inches='tight', dpi=250)

bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.29, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_allz_scaled_15_-0.29.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_allz_scaled_15_-0.29.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.29, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_hist_allz_scaled_15_-0.29.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_hist_allz_scaled_15_-0.29.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.33, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_z07_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_z07_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)

bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.33, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_allz_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_allz_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.33, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_hist_allz_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_hist_allz_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_z07.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_z07.png', bbox_inches='tight', dpi=250)




bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.33, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_z07_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_z07_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.33, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_hist_z07_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_hist_z07_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Norm. Counts')
plt.legend()
plt.tight_layout()
plt.savefig('plots/f_cumulative_allz.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cumulative_allz.png', bbox_inches='tight', dpi=250)



bnc, cnts_real_allz, int = plothist(np.log10(flux_real_allz), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim_allz, int = plothist(np.log10(flux_sim_allz)-0.3, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel('Counts')
plt.legend()
plt.tight_layout()

plt.close()

plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])

bnc, cnts_real_allz, int = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim_allz, int = plothist(np.log10(flux_sim_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)

bnc, cnts_real_z07, int = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim_z07, int = plothist(np.log10(flux_sim_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)


bnc148 = np.where(bnc*10== -148)[0]
bnc15 = np.where(bnc== -15)[0]
bnc17 = np.where(bnc == -17)[0]

scaled_z07_148 = cnts_real_z07[bnc148]/cnts_sim_z07[bnc148]
scaled_allz_148 = cnts_real_allz[bnc148]/cnts_sim_allz[bnc148]
scaled_z07_15 = cnts_real_z07[bnc15]/cnts_sim_z07[bnc15]
scaled_allz_15 = cnts_real_allz[bnc15]/cnts_sim_allz[bnc15]

plt.plot(bnc, cnts_real_z07, 'k--', label='Observed')
plt.plot(bnc, cnts_sim_z07*scaled_z07_148, 'b--', label='Simulated')
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'Cumulative counts')
plt.legend()
plt.tight_layout()
plt.xlim([-13,-18])
plt.savefig('plots/f_cum_z07_scaled_14.8.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cum_z07_scaled_14.8.png', bbox_inches='tight', dpi=250)

plt.plot(bnc, cnts_real_z07, 'k--', label='Observed')
plt.plot(bnc, cnts_sim_z07*scaled_z07_15, 'b--', label='Simulated')
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'Cumulative counts')
plt.legend()
plt.tight_layout()
plt.xlim([-13,-18])
plt.savefig('plots/f_cum_z07_scaled_15.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cum_z07_scaled_15.png', bbox_inches='tight', dpi=250)

plt.plot(bnc, cnts_real_allz, 'k--', label='Observed')
plt.plot(bnc, cnts_sim_allz*scaled_allz_15, 'b--', label='Simulated')
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'Cumulative counts')
plt.legend()
plt.tight_layout()
plt.xlim([-13,-18])
plt.savefig('plots/f_cum_allz_scaled_15.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cum_allz_scaled_15.png', bbox_inches='tight', dpi=250)


plt.plot(bnc, cnts_real_allz, 'k--', label='Observed')
plt.plot(bnc, cnts_sim_allz*scaled_allz_148, 'b--', label='Simulated')
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'Cumulative counts')
plt.legend()
plt.tight_layout()
plt.xlim([-13,-18])
plt.savefig('plots/f_cum_allz_scaled_14.8.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_cum_allz_scaled_14.8.png', bbox_inches='tight', dpi=250)




bnc, cnts_real, int_ = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim, int_ = plothist(np.log10(flux_sim_allz)-0.33, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()
plt.plot(bnc[::-1], (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ulcum_allz_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ulcum_allz_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.33, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()

plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ul_allz_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_allz_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)




bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.33, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()
plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ulcum_z07_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ulcum_z07_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.33, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()

plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ul_z07_scaled_14.8_-0.33.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_z07_scaled_14.8_-0.33.png', bbox_inches='tight', dpi=250)





bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.225, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()
plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ulcum_z07_scaled_15_-0.225.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ulcum_z07_scaled_15_-0.225.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_z07), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_z07)-0.225, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()

plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])
plt.savefig('plots/f_ul_z07_scaled_15_-0.225.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_z07_scaled_15_-0.225.png', bbox_inches='tight', dpi=250)

bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.29, cumulative=True, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()
plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])

plt.savefig('plots/f_ulcum_allz_scaled_15_-0.29.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ulcum_allz_scaled_15_-0.29.png', bbox_inches='tight', dpi=250)


bnc, cnts_real = plothist(np.log10(flux_real_allz), cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Observed', normed=False)
bnc, cnts_sim = plothist(np.log10(flux_sim_allz)-0.29, cumulative=False, reverse=True, bins=25, range=(-17.9,-12.9), label='Simulated', normed=False)
plt.gca().invert_xaxis()
plt.close()

plt.plot(bnc, (cnts_real-cnts_sim)/cnts_real)
plt.gca().invert_xaxis()
plt.xlabel('log(F[OIII])')
plt.ylabel(r'f$_{\mathrm{UL}}$')
plt.legend()
plt.tight_layout()
plt.ylim([-1,1])
plt.xlim([-13,-18])
plt.savefig('plots/f_ul_allz_scaled_15_-0.29.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/f_ul_allz_scaled_15_-0.29.png', bbox_inches='tight', dpi=250)


'''


def fwd_model(xr_df, xmid, avg_y, perc16, perc84,
              xmido3err, avg_y_o3err, perc16_o3err, perc84_o3err):
    '''

    '''

    np.random.seed(13)
    z = xr_df.z
    av = xr_df.corrected_presub_av
    lx = xr_df.full_lxagn
    # get oiii from a given lx based on median relation
    avg_interp = interp1d(xmid, avg_y, fill_value='extrapolate')
    perc16_interp = interp1d(xmid, perc16, fill_value='extrapolate')
    perc84_interp = interp1d(xmid, perc84, fill_value='extrapolate')
    avgerr_interp = interp1d(xmido3err, avg_y_o3err, fill_value='extrapolate')
    perc16err_interp = interp1d(
        xmido3err, perc16_o3err, fill_value='extrapolate')
    perc84err_interp = interp1d(
        xmido3err, perc84_o3err, fill_value='extrapolate')

    fo3_allsim = []
    fo3_err_allsim = []
    lo3_allsim = []
    lo3_allfiducial = []
    ks_vals = []

    p_vals = []
    realcnts, realbins = np.histogram(
        xr_df.oiiiflux, bins=100, range=(-1e-16, 1e-13))
    real_dist = np.cumsum(realcnts)

    range_scatters = np.linspace(1, 10, 100)
    for scat in range_scatters:
        fo3_sim = []
        lo3_fiducial = []
        fo3_err_sim = []
        lo3_sim = []
        for i in range(len(lx)):
            lxi = lx.iloc[i]
            zi = z.iloc[i]
            avi = av.iloc[i]
            perc16_i = perc16_interp(lxi)
            perc84_i = perc84_interp(lxi)
            avg_i = avg_interp(lxi)
            lo3_fiducial.append(avg_i)
            factor = (perc84_i-avg_i)/((avg_i-perc16_i)*scat)
            unif_factor = 0.5*factor
            random_unif = np.random.uniform()
            gaussian_sample = np.random.normal()

            if random_unif < unif_factor:
                lo3_pert = avg_i+(perc84_i-avg_i)*np.abs(gaussian_sample)

            else:
                lo3_pert = avg_i-(avg_i-perc16_i)*np.abs(gaussian_sample)*scat
            fo3_pert = getfluxfromlum(10**lo3_pert, zi)
            fo3_pert_red = redden(fo3_pert, avi, 5007.0)

            logfo3_err = avgerr_interp(
                np.log10(fo3_pert_red-np.min(xragn_no_sn_cuts.oiiiflux)*1.2))

            fo3_err = 10**(logfo3_err)
            fo3_pert_red = fo3_pert_red+np.random.normal(scale=fo3_err)

            fo3_sim.append(fo3_pert_red[0])
            lo3_sim.append(lo3_pert)
            fo3_err_sim.append(fo3_err[0])
        simcnts, simbins = np.histogram(
            fo3_sim, bins=100, range=(-1e-16, 1e-13))
        sim_dist = np.cumsum(simcnts)

        ks = ks_2samp(sim_dist, real_dist)
        ks_vals.append(ks.statistic)
        p_vals.append(ks.pvalue)
        lo3_allfiducial.append(lo3_fiducial)
        fo3_allsim.append(fo3_sim)
        fo3_err_allsim.append(fo3_err_sim)
        lo3_allsim.append(lo3_sim)

    return range_scatters, fo3_allsim, lo3_allsim, fo3_err_allsim, lo3_allfiducial, ks_vals, p_vals
    # error_interp

    # get the empirical error,
    # use that error as a sigma for a gaussian to draw the error
    # add the drawn gaussian to the the flux

    # make cumulative distribution of simulated fluxes, not log
    # same for actual fluxes
    # ks teest for the two distributions
    # change the lower scatter, repeat experiement, check the ks test. repeat until minimum ks difference
    # plot offsets delta log lo3 for fiducial high snr model and for best model
    # make them have the same peak, compare areas, these are the underluminous agn
    # repeat everything with SNR>2 relation?
    # perturb lo3,


def plot_relations():
    for prop in xr_agn_props.keys():
        agn_prop = xr_agn_props[prop]
        a, b, aerr, berr, covab = BCES.bcesp(np.array(agn_prop.full_xraylum-42),
                                             np.array(np.nanmean(np.vstack([agn_prop.e_full_xraylum_down,
                                                                            agn_prop.e_full_xraylum_up]), axis=0)),
                                             np.array(agn_prop.oiiilum),
                                             np.array(np.nanmean(np.vstack([agn_prop.e_oiiilum_down,
                                                                           agn_prop.e_oiiilum_up]), axis=0)),
                                             np.zeros_like(agn_prop.full_xraylum))
        print(a, b)
        print(prop)
        scatter(agn_prop.hard_xraylum, agn_prop.oiiilum,
                # xerr=np.vstack([agn_prop.e_full_xraylum_down,
                #                agn_prop.e_full_xraylum_up]),
                # yerr=np.vstack([agn_prop.e_oiiilum_down,
                #                agn_prop.e_oiiilum_up]),
                percentiles=True,
                minx=37, maxx=46, miny=37, maxy=44, aspect='equal', alpha=1, ecolor='k', size_y_bin=0.5, bin_y=True)

        # plt.plot(np.arange(37,46,0.1), (np.arange(37,46,0.1)-42)*(a[3])+b[3],'k:',
        #         label='ODR: '+str(a[3])[:4]+r'$\cdot (x-42)$+'+str(b[3])[:4])
        plt.xticks([38, 40, 42, 44, 46])
        plt.yticks([38, 40, 42, 44])

        plt.xlabel(r'log(L$_{\mathrm{X}}$)')
        plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
        plt.legend()
        plt.savefig('plots/hx_o3_'+prop+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/hx_o3_'+prop+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()

    for prop in pure_xr_agn_props.keys():
        agn_prop = pure_xr_agn_props[prop]
        a, b, aerr, berr, covab = BCES.bcesp(np.array(agn_prop.full_lxagn-42),
                                             np.array(np.nanmean(np.vstack([agn_prop.e_full_xraylum_down,
                                                                           agn_prop.e_full_xraylum_up]), axis=0)),
                                             np.array(
                                                 agn_prop.oiiilum_sub_dered),
                                             np.array(np.nanmean(np.vstack([agn_prop.e_oiiilum_down_sub_dered,
                                                                            agn_prop.e_oiiilum_up_sub_dered]), axis=0)),
                                             np.zeros_like(agn_prop.full_lxagn))
        print(a, b)
        scatter(agn_prop.full_lxagn, agn_prop.oiiilum_sub_dered,
                xerr=np.vstack([agn_prop.e_full_xraylum_down,
                                agn_prop.e_full_xraylum_up]),
                yerr=np.vstack([agn_prop.e_oiiilum_down_sub_dered,
                                agn_prop.e_oiiilum_up_sub_dered]),
                minx=37, maxx=46, miny=37, maxy=46, aspect='equal', alpha=1, ecolor='k', bin_y=True)

        plt.plot(np.arange(37, 46, 0.1), (np.arange(37, 46, 0.1)-42)*(a[3])+b[3], 'k:',
                 label='ODR: '+str(a[3])[:4]+r'$\cdot (x-42)$+'+str(b[3])[:4])
        plt.xticks([38, 40, 42, 44, 46])
        plt.yticks([38, 40, 42, 44, 46])

        plt.xlabel(r'log(L$_{\mathrm{X}}$)')
        plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
        plt.legend()
        plt.savefig('plots/pure_lx_o3_'+prop+'.pdf',
                    bbox_inches='tight', dpi=250, format='pdf')
        plt.savefig('plots/pure_lx_o3_'+prop+'.png',
                    bbox_inches='tight', dpi=250, format='png')
        plt.close()


'''
prop='unclass_p1'
agn_prop= xr_agn_props[prop]

valid_samp = np.where((np.isfinite(agn_prop.full_xraylum))& 
                      (np.isfinite(agn_prop.oiiilum)) &
                      (np.isfinite(agn_prop.e_full_xraylum_down)) &
                      (np.isfinite(agn_prop.e_full_xraylum_up)) &
                      (np.isfinite(agn_prop.e_oiiilum_down)) & 
                      (np.isfinite(agn_prop.e_oiiilum_up)))[0]
                                                                           
a,b,aerr,berr,covab=BCES.bcesp(np.array(agn_prop.full_xraylum-42)[valid_samp],
                           np.array(np.nanmean(np.vstack([agn_prop.e_full_xraylum_down,
                                                          agn_prop.e_full_xraylum_up]),axis=0))[valid_samp],
                           np.array(agn_prop.oiiilum)[valid_samp],
                           np.array(np.nanmean(np.vstack([agn_prop.e_oiiilum_down, 
                                                          agn_prop.e_oiiilum_up]), axis=0))[valid_samp], 
                           np.zeros_like(agn_prop.full_xraylum)[valid_samp])
print(a,b)
print(prop)
scatter(agn_prop.full_xraylum[valid_samp],agn_prop.oiiilum[valid_samp], 
        xerr=np.vstack([agn_prop.e_full_xraylum_down[valid_samp], 
                        agn_prop.e_full_xraylum_up[valid_samp]]),
        yerr=np.vstack([agn_prop.e_oiiilum_down[valid_samp],
         
                        agn_prop.e_oiiilum_up[valid_samp]]),
                        
        minx=37, maxx=46, miny=37, maxy=46, aspect='equal',alpha=1, ecolor='k', bin_y=True, size_y_bin=0.5)    
plt.xticks([38,40,42,44,46])
plt.yticks([38,40,42,44,46])
plt.plot(np.arange(37,46,0.1), (np.arange(37,46,0.1)-42)*(a[3])+b[3],'k:', 
         label='ODR: '+str(a[3])[:4]+r'$\cdot (x-42)$+'+str(b[3])[:4])
plt.xlabel(r'log(L$_{\mathrm{X}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.savefig('plots/lx_o3_unclass_p1.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lx_o3_unclass_p1.png', bbox_inches='tight', dpi=250, format='png')
prop='unclass_p2'
agn_prop= xr_agn_props[prop]
a,b,aerr,berr,covab=BCES.bcesp(np.array(agn_prop.full_xraylum-42)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(np.nanmean(np.vstack([agn_prop.e_full_xraylum_down,
                                                          agn_prop.e_full_xraylum_up]),axis=0))[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(agn_prop.oiiilum)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(np.nanmean(np.vstack([agn_prop.e_oiiilum_down, 
                                                          agn_prop.e_oiiilum_up]), axis=0))[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]], 
                           np.zeros_like(agn_prop.full_xraylum)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]])
print(a,b)
print(prop)
scatter(agn_prop.full_xraylum[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],agn_prop.oiiilum[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]], 
        xerr=np.vstack([agn_prop.e_full_xraylum_down[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]], 
                        agn_prop.e_full_xraylum_up[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]]]),
        yerr=np.vstack([agn_prop.e_oiiilum_down[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]],
         
                        agn_prop.e_oiiilum_up[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]]]),
                        
        minx=37, maxx=46, miny=37, maxy=46, aspect='equal',alpha=1, ecolor='k', bin_y=True, size_y_bin=0.5)    
plt.xticks([38,40,42,44,46])
plt.yticks([38,40,42,44,46])
plt.plot(np.arange(37,46,0.1), (np.arange(37,46,0.1)-42)*(a[3])+b[3],'k:', 
         label='ODR: '+str(a[3])[:4]+r'$\cdot (x-42)$+'+str(b[3])[:4])
plt.xlabel(r'log(L$_{\mathrm{X}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.savefig('plots/lx_o3_unclass_p2.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lx_o3_unclass_p2.png', bbox_inches='tight', dpi=250, format='png')
prop='xrall'
agn_prop= xr_agn_props[prop]
a,b,aerr,berr,covab=BCES.bcesp(np.array(agn_prop.full_xraylum-42)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(np.nanmean(np.vstack([agn_prop.e_full_xraylum_down,
                                                          agn_prop.e_full_xraylum_up]),axis=0))[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(agn_prop.oiiilum)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],
                           np.array(np.nanmean(np.vstack([agn_prop.e_oiiilum_down, 
                                                          agn_prop.e_oiiilum_up]), axis=0))[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]], 
                           np.zeros_like(agn_prop.full_xraylum)[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]])
print(a,b)
print(prop)
scatter(agn_prop.full_xraylum[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum)))[0]],agn_prop.oiiilum[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]], 
        xerr=np.vstack([agn_prop.e_full_xraylum_down[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]], 
                        agn_prop.e_full_xraylum_up[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]]]),
        yerr=np.vstack([agn_prop.e_oiiilum_down[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]],
         
                        agn_prop.e_oiiilum_up[np.where((np.isfinite(agn_prop.full_xraylum))&(np.isfinite(agn_prop.oiiilum))&(agn_prop.oiii_err!=0))[0]]]),
                        
        minx=37, maxx=46, miny=37, maxy=46, aspect='equal',alpha=1, ecolor='k', bin_y=True, size_y_bin=0.5)    
plt.xticks([38,40,42,44,46])
plt.yticks([38,40,42,44,46])
plt.plot(np.arange(37,46,0.1), (np.arange(37,46,0.1)-42)*(a[3])+b[3],'k:', 
         label='ODR: '+str(a[3])[:4]+r'$\cdot (x-42)$+'+str(b[3])[:4])
plt.xlabel(r'log(L$_{\mathrm{X}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.savefig('plots/lx_o3_xrall.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/lx_o3_xrall.png', bbox_inches='tight', dpi=250, format='png')
'''


def lin_func(p, x):
    m, b = p
    return m*x+b-42


genx_ = np.arange(37, 46)
geny_ = np.arange(37, 46)

x_panessa = geny_*1.22-7.55

panessa_xraylum = np.array([40.79, 37.55, 42.84, 42.07, 42.83, 42.31, 42.58, 40.82, 41.85, 38.21,
                            39.95, 40.25, 42.62, 41.89, 40.79, 41.74, 38.86, 39.88, 42.29, 39.10,
                            37.88, 38.88, 41.18, 41.31, 41.29, 42.47, 39.87, 42.22, 40.87, 41.72,
                            39.81, 39.32, 39.65, 39.59, 39.43, 41.03, 40.22, 39.16, 38.89, 41.08,
                            40.91, 41.36, 43.25,  41.12, 41.47])
panessa_t1_xraylum = np.array([42.83, 40.25, 41.74, 42.29,
                               41.31, 42.47, 42.22, 39.81, 41.03,
                               40.22, 41.08, 41.36, 43.25])
panessa_t1_oiiilum = np.array([41.91, 38.56, 40.51, 40.50,
                               39.81, 41.47, 40.41, 38.28, 39.42,
                               38.71, 39.72, 39.03, 41.16])
panessa_oiiilum = np.array([39.04, 37.90, 41.91, 40.76, 41.91, 40.42, 40.94, 40.40, 39.92, 38.58,
                            38.96, 38.56, 40.07, 40.07, 39.90, 40.51, 37.99, 38.86, 40.50, 38.22,
                            38.79, 38.63, 40.50, 39.81, 38.74, 41.47, 38.46, 40.41, 39.07, 40.54,
                            38.28, 37.81, 39.04, 39.23, 38.69, 39.42, 38.71, 38.86, 38.58, 39.72,
                            39.90, 39.03, 41.16, 40.25, 40.21])

'''
bin_stat_y='median'
    
x_heckman = geny_+1.59
x_heckmant2 = geny_+0.57

y_berney_all = genx_*1.23-12
y_berney_t1 = genx_*1.1-6.5
y_berney_t2 = genx_*1.3-15
y_berney_sy = genx_*1.13-7.5
y_berney_liner = genx_*1.8-37
y_berney_comp = genx_-4



xliner_full, yavliner_full, _,liner_16_full, liner_84_full = scatter(xr_agn_props['combo_all_liners_fx'].full_xraylum, 
                                                                     xr_agn_props['combo_all_liners_fx'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)


xliner_hard, yavliner_hard, _,liner_16_hard,liner_84_hard = scatter(xr_agn_props['combo_all_liners_hx'].hard_xraylum,
                                                                    xr_agn_props['combo_all_liners_hx'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)

xliner_full07, yavliner_full07, _,liner_16_full07, liner_84_full07 = scatter(xr_agn_props['combo_all_liners_fx_z07'].full_xraylum, 
                                                                             xr_agn_props['combo_all_liners_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)


xliner_hard07, yavliner_hard07, _,liner_16_hard07,liner_84_hard07 = scatter(xr_agn_props['combo_all_liners_hx_z07'].hard_xraylum,
                                                                            xr_agn_props['combo_all_liners_hx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xliner_hard07edd, yavliner_hard07edd, _,liner_16_hard07edd,liner_84_hard07edd = scatter(xr_agn_props['combo_all_liners_hx_z07'].hard_xraylum-xr_agn_props['combo_all_liners_hx_z07'].mbh,
                                                                            xr_agn_props['combo_all_liners_hx_z07'].oiiilum-xr_agn_props['combo_all_liners_hx_z07'].mbh,
                                                                             aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,
             maxx=38, minx=31, miny=30, maxy=37)

xliner_hardedd, yavliner_hardedd, _,liner_16_hardedd,liner_84_hardedd = scatter(xr_agn_props['combo_all_liners_hx'].hard_xraylum-xr_agn_props['combo_all_liners_hx'].mbh,
                                                                    xr_agn_props['combo_all_liners_hx'].oiiilum-xr_agn_props['combo_all_liners_hx'].mbh,
                    maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)



xliu_hard_dc, yavliu_hard_dc, _, liu16_hard_dc, liu84_hard_dc = scatter(liuobj_xmm_hx_o3_dust_df['hardlums_rf'], liuobj_xmm_hx_o3_dust_df['lo3_corr'] ,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)
xliu_full_dc, yavliu_full_dc, _, liu16_full_dc, liu84_full_dc = scatter(liuobj_xmm_fx_o3_dust_df['fulllums_rf'], liuobj_xmm_fx_o3_dust_df['lo3_corr'] ,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xliu_hard_dc07, yavliu_hard_dc07, _, liu16_hard_dc07, liu84_hard_dc07 = scatter(liuobj_xmm_hx_o3_dust_z07_df['hardlums_rf'], liuobj_xmm_hx_o3_dust_z07_df['lo3_corr'] ,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)
xliu_full_dc07, yavliu_full_dc07, _, liu16_full_dc07, liu84_full_dc07 = scatter(liuobj_xmm_fx_o3_dust_z07_df['fulllums_rf'], liuobj_xmm_fx_o3_dust_z07_df['lo3_corr'] ,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xliu_hard_dc07edd, yavliu_hard_dc07edd, _, liu16_hard_dc07edd, liu84_hard_dc07edd = scatter(liuobj_xmm_hx_o3_dust_z07_df['hardlums_rf']-liuobj_xmm_hx_o3_dust_z07_df['logMBH'], 
                                                                                            liuobj_xmm_hx_o3_dust_z07_df['lo3_corr']-liuobj_xmm_hx_o3_dust_z07_df['logMBH'] ,
            maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xliu_hard_dcedd, yavliu_hard_dcedd, _, liu16_hard_dcedd, liu84_hard_dcedd = scatter(liuobj_xmm_hx_o3_dust_df['hardlums_rf']-liuobj_xmm_hx_o3_dust_df['logMBH'],
                                                                                    liuobj_xmm_hx_o3_dust_df['lo3_corr']-liuobj_xmm_hx_o3_dust_df['logMBH'] ,
             maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)


xsy2_full07, yavsy2_full07, _, sy2_16_full07, sy2_84_full07 = scatter(xr_agn_props['combo_sy2_fx_z07'].full_xraylum, xr_agn_props['combo_sy2_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xsy2_hard07, yavsy2_hard07, _, sy2_16_hard07, sy2_84_hard07 = scatter(xr_agn_props['combo_sy2_hx_z07'].hard_xraylum, xr_agn_props['combo_sy2_hx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xsy2_full, yavsy2_full, _, sy2_16_full, sy2_84_full = scatter(xr_agn_props['combo_sy2_fx'].full_xraylum, xr_agn_props['combo_sy2_fx'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xsy2_hard, yavsy2_hard, _, sy2_16_hard, sy2_84_hard = scatter(xr_agn_props['combo_sy2_hx'].hard_xraylum, xr_agn_props['combo_sy2_hx'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)
xsy2_hard07, yavsy2_hard07, _, sy2_16_hard07, sy2_84_hard07 = scatter(xr_agn_props['combo_sy2_hx_z07'].hard_xraylum, xr_agn_props['combo_sy2_hx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)

xsy2_hardedd, yavsy2_hardedd, _, sy2_16_hardedd, sy2_84_hardedd = scatter(xr_agn_props['combo_sy2_hx'].hard_xraylum-xr_agn_props['combo_sy2_hx'].mbh, 
                                                                          xr_agn_props['combo_sy2_hx'].oiiilum-xr_agn_props['combo_sy2_hx'].mbh,
            maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xsy2_hard07edd, yavsy2_hard07edd, _, sy2_16_hard07edd, sy2_84_hard07edd = scatter(xr_agn_props['combo_sy2_hx_z07'].hard_xraylum-xr_agn_props['combo_sy2_hx_z07'].mbh, 
                                                                                  xr_agn_props['combo_sy2_hx_z07'].oiiilum-xr_agn_props['combo_sy2_hx_z07'].mbh,
             maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)

xsy2_fulledd, yavsy2_fulledd, _, sy2_16_fulledd, sy2_84_fulledd = scatter(xr_agn_props['combo_sy2_fx'].full_xraylum-xr_agn_props['combo_sy2_fx'].mbh, 
                                                                          xr_agn_props['combo_sy2_fx'].oiiilum-xr_agn_props['combo_sy2_fx'].mbh,
            maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)
xsy2_full07edd, yavsy2_full07edd, _, sy2_16_full07edd, sy2_84_full07edd = scatter(xr_agn_props['combo_sy2_fx_z07'].full_xraylum-xr_agn_props['combo_sy2_fx_z07'].mbh, 
                                                                                  xr_agn_props['combo_sy2_fx_z07'].oiiilum-xr_agn_props['combo_sy2_fx_z07'].mbh,
             maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)



xt2_full07, yavt2_full07, _, t2_16_full07, t2_84_full07 = scatter(xr_agn_props['x4_sn3_o3_fx_z07'].full_xraylum, xr_agn_props['x4_sn3_o3_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)
xt2_hard07, yavt2_hard07, _, t2_16_hard07, t2_84_hard07 = scatter(xr_agn_props['x4_sn3_o3_hx_z07'].hard_xraylum, xr_agn_props['x4_sn3_o3_hx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)

xt2_full, yavt2_full, _, t2_16_full, t2_84_full = scatter(xr_agn_props['x4_sn3_o3_fx_allz'].full_xraylum, xr_agn_props['x4_sn3_o3_fx_allz'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)

xt2_hard, yavt2_hard, _, t2_16_hard, t2_84_hard = scatter(xr_agn_props['x4_sn3_o3_hx_allz_bptagn'].hard_xraylum, xr_agn_props['x4_sn3_o3_hx_allz_bptagn'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y,counting_thresh=10)

xt2_hard07edd, yavt2_hard07edd, _, t2_16_hard07edd, t2_84_hard07edd = scatter(xr_agn_props['x4_sn3_o3_hx_z07'].hard_xraylum-xr_agn_props['x4_sn3_o3_hx_z07'].mbh, 
                                                                              xr_agn_props['x4_sn3_o3_hx_z07'].oiiilum-xr_agn_props['x4_sn3_o3_hx_z07'].mbh,
               maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)

xt2_hardedd, yavt2_hardedd, _, t2_16_hardedd, t2_84_hardedd = scatter(xr_agn_props['x4_sn3_o3_hx_allz'].hard_xraylum-xr_agn_props['x4_sn3_o3_hx_allz'].mbh, 
                                                                      xr_agn_props['x4_sn3_o3_hx_allz'].oiiilum-xr_agn_props['x4_sn3_o3_hx_allz'].mbh,
               maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=True, size_y_bin=0.5, percentiles=True, bin_stat_y=bin_stat_y)




plt.plot(xliu_full07[yavliu_full07>0], yavliu_full07[yavliu_full07>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_full07, yavt2_full07, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_full07, yavsy2_full07, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_full07, yavliner_full07, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_fx_z07'].full_xraylum, xr_agn_props['x4_sn3_o3_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_fx_z07'].full_xraylum, xr_agn_props['combo_all_liners_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_fx_z07'].full_xraylum, xr_agn_props['combo_sy2_fx_z07'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([38,40,42,44])
#plt.yticks([38,40,42,44])

plt.xlabel(r'log(L$_{\mathrm{X,\ 0.5-10\ keV}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.tight_layout()




plt.plot(xliu_full_dc07[yavliu_full_dc07>0], yavliu_full_dc07[yavliu_full_dc07>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_full07, yavt2_full07, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_full07, yavsy2_full07, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_full07, yavliner_full07, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_fx_z07'].full_xraylum, xr_agn_props['x4_sn3_o3_fx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_fx_z07'].full_xraylum, xr_agn_props['combo_all_liners_fx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_fx_z07'].full_xraylum, xr_agn_props['combo_sy2_fx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([38,40,42,44])
plt.yticks([38,40,42,44])


plt.xlabel(r'log(L$_{\mathrm{X,\ 0.5-10\ keV}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/fx_o3_sn3_combined_objects_z07_t1dc_med.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/fx_o3_sn3_combined_objects_z07_t1dc_med.png', bbox_inches='tight', dpi=250, format='png')
plt.close()


plt.plot(xliu_hard_dc07[yavliu_hard_dc07>0], yavliu_hard_dc07[yavliu_hard_dc07>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_hard07, yavt2_hard07, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_hard07, yavsy2_hard07, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_hard07, yavliner_hard07, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_hx_z07'].hard_xraylum, xr_agn_props['x4_sn3_o3_hx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_hx_z07'].hard_xraylum, xr_agn_props['combo_all_liners_hx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_hx_z07'].hard_xraylum, xr_agn_props['combo_sy2_hx_z07'].oiiilum_sfsub_samir,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([38,40,42,44])
plt.yticks([38,40,42,44])
plt.plot(x_panessa, geny_, 'k--', linewidth=3, label='Panessa+07 without QSOs')

plt.xlabel(r'log(L$_{\mathrm{X,\ 2-10\ keV}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/hx_o3_sn3_combined_objects_z07_t1dc_med_p07_noqso.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hx_o3_sn3_combined_objects_z07_t1dc_med_p07_noqso.png', bbox_inches='tight', dpi=250, format='png')
plt.close()



plt.plot(xliu_hard_dc07edd[yavliu_hard_dc07edd>0], yavliu_hard_dc07edd[yavliu_hard_dc07edd>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_hard07edd, yavt2_hard07edd, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_hard07edd, yavsy2_hard07edd, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_hard07edd, yavliner_hard07edd, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_hx_z07'].hard_xraylum-xr_agn_props['x4_sn3_o3_hx_z07'].mbh, 
        xr_agn_props['x4_sn3_o3_hx_z07'].oiiilum-xr_agn_props['x4_sn3_o3_hx_z07'].mbh,
              maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_hx_z07'].hard_xraylum- xr_agn_props['combo_all_liners_hx_z07'].mbh,
        xr_agn_props['combo_all_liners_hx_z07'].oiiilum- xr_agn_props['combo_all_liners_hx_z07'].mbh,
              maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_hx_z07'].hard_xraylum-xr_agn_props['combo_sy2_hx_z07'].mbh, 
        xr_agn_props['combo_sy2_hx_z07'].oiiilum-xr_agn_props['combo_sy2_hx_z07'].mbh,
             maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([32,34,36,38])
plt.yticks([30,32,34,36])
#plt.plot(x_panessa, geny_, 'k--', linewidth=3, label='Panessa+07 without QSOs')

plt.xlabel(r'log(L$_{\mathrm{X,\ 2-10\ keV}}$/M$_{\mathrm{BH}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$/M$_{\mathrm{BH}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/hx_o3_mbh_sn3_combined_objects_z07_t1dc.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hx_o3_mbh_sn3_combined_objects_z07_t1dc.png', bbox_inches='tight', dpi=250, format='png')
plt.close()


plt.plot(xliu_hard_dcedd[yavliu_hard_dcedd>0], yavliu_hard_dcedd[yavliu_hard_dcedd>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_hardedd, yavt2_hardedd, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_hardedd, yavsy2_hardedd, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_hardedd, yavliner_hardedd, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_hx_allz'].hard_xraylum-xr_agn_props['x4_sn3_o3_hx_allz'].mbh, 
        xr_agn_props['x4_sn3_o3_hx_allz'].oiiilum-xr_agn_props['x4_sn3_o3_hx_allz'].mbh,
              maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_hx'].hard_xraylum- xr_agn_props['combo_all_liners_hx'].mbh,
        xr_agn_props['combo_all_liners_hx'].oiiilum- xr_agn_props['combo_all_liners_hx'].mbh,
              maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_hx'].hard_xraylum-xr_agn_props['combo_sy2_hx'].mbh, 
        xr_agn_props['combo_sy2_hx'].oiiilum-xr_agn_props['combo_sy2_hx'].mbh,
             maxx=38, minx=31, miny=30, maxy=37, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([32,34,36,38])
plt.yticks([30,32,34,36])
#plt.plot(x_panessa, geny_, 'k--', linewidth=3, label='Panessa+07 without QSOs')

plt.xlabel(r'log(L$_{\mathrm{X,\ 2-10\ keV}}$/M$_{\mathrm{BH}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$/M$_{\mathrm{BH}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/hx_o3_mbh_sn3_combined_objects_t1dc.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hx_o3_mbh_sn3_combined_objects_t1dc.png', bbox_inches='tight', dpi=250, format='png')
plt.close()


plt.plot(xliu_full_dc[yavliu_full_dc>0], yavliu_full_dc[yavliu_full_dc>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_full, yavt2_full, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_full, yavsy2_full, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_full, yavliner_full, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_fx_allz'].full_xraylum, xr_agn_props['x4_sn3_o3_fx_allz'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_fx'].full_xraylum, xr_agn_props['combo_all_liners_fx'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_fx'].full_xraylum, xr_agn_props['combo_sy2_fx'].oiiilum,
             maxx=46, minx=37, miny=37, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([38,40,42,44])
plt.yticks([38,40,42,44])

plt.xlabel(r'log(L$_{\mathrm{X,\ 0.5-10\ keV}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/fx_o3_sn3_combined_objects_allz_t1dc_med.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/fx_o3_sn3_combined_objects_allz_t1dc_med.png', bbox_inches='tight', dpi=250, format='png')
plt.close()



plt.plot(xliu_hard_dc[yavliu_hard_dc>0], yavliu_hard_dc[yavliu_hard_dc>0], color='c', linewidth=3, label='Type 1 Median')
plt.plot(xt2_hard, yavt2_hard, color='r', linewidth=3, label='Type 2 Median')
plt.plot(xsy2_hard, yavsy2_hard, color='b', linewidth=3, label='Sy2')
plt.plot(xliner_hard, yavliner_hard, color='orange', linewidth=3, label='LINER')
scatter(xr_agn_props['x4_sn3_o3_hx_allz'].hard_xraylum, xr_agn_props['x4_sn3_o3_hx_allz'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, label='Type 2 AGN', color='lightgray', edgecolor='none')
scatter(xr_agn_props['combo_all_liners_hx'].hard_xraylum, xr_agn_props['combo_all_liners_hx'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='orange', edgecolor='none')
scatter(xr_agn_props['combo_sy2_hx'].hard_xraylum, xr_agn_props['combo_sy2_hx'].oiiilum,
             maxx=46, minx=38, miny=38, maxy=44, aspect='equal', bin_y=False, size_y_bin=0.5, percentiles=True, color='blue', edgecolor='none')
plt.xticks([38,40,42,44])
plt.yticks([38,40,42,44])
plt.plot(x_panessa, geny_, 'k--', linewidth=3, label='Panessa+06')
plt.xlabel(r'log(L$_{\mathrm{X,\ 2-10\ keV}}$)')
plt.ylabel(r'log(L$_{\mathrm{[OIII]}}$)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/hx_o3_sn3_combined_objects_allz_t1dc_med_p07_noqso_sfsub.pdf', bbox_inches='tight', dpi=250, format='pdf')
plt.savefig('plots/hx_o3_sn3_combined_objects_allz_t1dc_med_p07_noqso_sfsub.png', bbox_inches='tight', dpi=250, format='png')
plt.close()

'''
