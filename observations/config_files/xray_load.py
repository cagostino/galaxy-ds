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
xr_nii_xragn_sf_df = pd.read_csv('../../catalogs/xragn_bptplusniisf.csv')



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
                'x4all_hx': x4_all_xray.iloc[np.where((x4_all_xray.hardflux_sn > 2) & (x4_all_xray.z <= 0.3))].copy(),                

                'x4_fx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.fullflux_sn > 2) & (x4_all_xragn.z <= 0.07))].copy(),
                'x4_hx_z07': x4_all_xragn.iloc[np.where((x4_all_xragn.hardflux_sn > 2) & (x4_all_xragn.z <= 0.07))].copy(),
                'x4_fx_allz': x4_all_xragn.iloc[np.where((x4_all_xragn.fullflux_sn > 2) & (x4_all_xragn.z <= 0.3))].copy(),
                'x4_hx_allz': x4_all_xragn.iloc[np.where((x4_all_xragn.hardflux_sn > 2) & (x4_all_xragn.z <= 0.3))].copy(),                
                'x4_hx_allz_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.hardflux_sn > 2) &  (x4_all_xragn.z <= 0.3)&
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                       ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.oiiiflux_sn>2)&
                                                                        (x4_all_xragn.hbetaflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2) )
                                                                        )   ].copy(),
                
                
                'x4_hx_allz_noext': x4_all_xragn.iloc[np.where((x4_all_xragn.hardflux_sn > 2) &(x4_all_xragn.ext ==0) &  (x4_all_xragn.z <= 0.3))].copy(),

                'x4_hx_allz_noext_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.hardflux_sn > 2) &(x4_all_xragn.ext ==0) &  (x4_all_xragn.z <= 0.3)&
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                       ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.oiiiflux_sn>2)&
                                                                        (x4_all_xragn.hbetaflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2) )
                                                                        )   ].copy(),
 
                
                'x4_hx_allz_ext': x4_all_xragn.iloc[np.where((x4_all_xragn.hardflux_sn > 2) &(x4_all_xragn.ext >0) & (x4_all_xragn.z <= 0.3))].copy(),

                'x4_hx_allz_ext_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.hardflux_sn > 2) &(x4_all_xragn.ext >0) & (x4_all_xragn.z <= 0.3)&
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                       ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.oiiiflux_sn>2)&
                                                                        (x4_all_xragn.hbetaflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2) )
                                                                        )   ].copy(),
                
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
                'x4_sn3_o3_hx_allz_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 3) & (x4_all_xragn.hardflux_sn > 2)&
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                       ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.oiiiflux_sn>2)&
                                                                        (x4_all_xragn.hbetaflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2) )
                                                                        )].copy(),
                'x4_sn3_o3_hx_allz_noext_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 3) & (x4_all_xragn.hardflux_sn > 2)&(x4_all_xragn.ext ==0)&
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                       ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.oiiiflux_sn>2)&
                                                                        (x4_all_xragn.hbetaflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2) )
                                                                        )].copy(),



                'x4_sn1_o3_hx_allz_noext': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.hardflux_sn > 2)& (x4_all_xragn.ext == 0))].copy(),
                'x4_sn1_o3_hx_allz_ext': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.hardflux_sn > 2)& (x4_all_xragn.ext != 0))].copy(),
                'x4_sn1_o3_hx_allz': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.hardflux_sn > 2))].copy(),

                'x4_sn1_o3_hx_allz_noext_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & 
                                                                              (x4_all_xragn.hardflux_sn > 2)& 
                                                                              (x4_all_xragn.ext == 0) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                              (x4_all_xragn.niiflux_sn>2)&
                                                                              (x4_all_xragn.oiiiflux_sn>2)&
                                                                              (x4_all_xragn.hbetaflux_sn>2)&
                                                                              (x4_all_xragn.halpflux_sn>2) )
                                                                              )].copy(),
                 'x4_sn1_o3_hx_allz_noext_nobptsf_ionpar': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.oiiflux_sn > 1) &
                                                                              (x4_all_xragn.hardflux_sn > 2)& 
                                                                              (x4_all_xragn.ext == 0) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                              (x4_all_xragn.niiflux_sn>2)&
                                                                              (x4_all_xragn.oiiiflux_sn>2)&
                                                                              (x4_all_xragn.hbetaflux_sn>2)&
                                                                              (x4_all_xragn.halpflux_sn>2) )
                                                                              )].copy(),
                
                'x4_sn1_o3_hx_allz_noext_nobptsf_oh': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.niiflux_sn > 1) &
                                                                              (x4_all_xragn.hardflux_sn > 2)& 
                                                                              (x4_all_xragn.oiiflux_sn > 1) &
                                                                              (x4_all_xragn.ext == 0) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                              (x4_all_xragn.niiflux_sn>2)&
                                                                              (x4_all_xragn.oiiiflux_sn>2)&
                                                                              (x4_all_xragn.hbetaflux_sn>2)&
                                                                              (x4_all_xragn.halpflux_sn>2) )
                                                                              )].copy(),
                 
                'x4_bad_o3_hx_allz_noext_nobptsf_oh': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn < 1) & (x4_all_xragn.niiflux_sn > 1) &
                                                                              (x4_all_xragn.hardflux_sn > 2)& 
                                                                              (x4_all_xragn.oiiflux_sn > 1) &
                                                                              (x4_all_xragn.ext == 0) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                              (x4_all_xragn.niiflux_sn>2)&
                                                                              (x4_all_xragn.oiiiflux_sn>2)&
                                                                              (x4_all_xragn.hbetaflux_sn>2)&
                                                                              (x4_all_xragn.halpflux_sn>2) )
                                                                              )].copy(),
                 'x4_sn1_o3_hx_allz_noext_nobptsf_sii': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.niiflux_sn > 1) &
                                                                              (x4_all_xragn.hardflux_sn > 2)& 
                                                                              (x4_all_xragn.sii6717flux_sn > 1) &
                                                                              (x4_all_xragn.sii6731flux_sn > 1) &
                                                                              (x4_all_xragn.ext == 0) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                              (x4_all_xragn.niiflux_sn>2)&
                                                                              (x4_all_xragn.oiiiflux_sn>2)&
                                                                              (x4_all_xragn.hbetaflux_sn>2)&
                                                                              (x4_all_xragn.halpflux_sn>2) )
                                                                              )].copy(),
                 'x4_bad_o3_hx_allz_noext_nobptsf_sii': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn < 1) & (x4_all_xragn.niiflux_sn > 1) &
                                                                              (x4_all_xragn.hardflux_sn > 2)& 
                                                                              (x4_all_xragn.sii6717flux_sn > 1) &
                                                                              (x4_all_xragn.sii6731flux_sn > 1) &
                                                                              (x4_all_xragn.ext == 0) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                              (x4_all_xragn.niiflux_sn>2)&
                                                                              (x4_all_xragn.oiiiflux_sn>2)&
                                                                              (x4_all_xragn.hbetaflux_sn>2)&
                                                                              (x4_all_xragn.halpflux_sn>2) )
                                                                              )].copy(),
                'x4_sn1_o3_hx_allz_noext_nobptsf_hard': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.oiflux_sn > 1) &
                                                                              (x4_all_xragn.hardflux_sn > 2)& 
                                                                              (x4_all_xragn.halpflux_sn > 1) &
                                                                              (x4_all_xragn.ext == 0) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                              (x4_all_xragn.niiflux_sn>2)&
                                                                              (x4_all_xragn.oiiiflux_sn>2)&
                                                                              (x4_all_xragn.hbetaflux_sn>2)&
                                                                              (x4_all_xragn.halpflux_sn>2) )
                                                                              )].copy(),
                  'x4_bad_o3_hx_allz_noext_nobptsf_hard': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn < 1) & (x4_all_xragn.oiflux_sn > 1) &
                                                                              (x4_all_xragn.hardflux_sn > 2)& 
                                                                              (x4_all_xragn.halpflux_sn > 1) &
                                                                              (x4_all_xragn.ext == 0) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                              (x4_all_xragn.niiflux_sn>2)&
                                                                              (x4_all_xragn.oiiiflux_sn>2)&
                                                                              (x4_all_xragn.hbetaflux_sn>2)&
                                                                              (x4_all_xragn.halpflux_sn>2) )
                                                                              )].copy(),
                'x4_sn1_o3_hx_allz_ext_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & 
                                                                            (x4_all_xragn.hardflux_sn > 2)& 
                                                                            (x4_all_xragn.ext != 0)&
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                            (x4_all_xragn.niiflux_sn>2)&
                                                                            (x4_all_xragn.oiiiflux_sn>2)&
                                                                            (x4_all_xragn.hbetaflux_sn>2)&
                                                                            (x4_all_xragn.halpflux_sn>2) )
                                                                            )].copy(),
                'x4_sn1_o3_hx_allz_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) &
                                                                        (x4_all_xragn.hardflux_sn > 2)&
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                        ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.oiiiflux_sn>2)&
                                                                        (x4_all_xragn.hbetaflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2) )
                                                                        )].copy(),

                'x4_bad_o3_hx_allz_ext': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn < 1) & (x4_all_xragn.hardflux_sn > 2)& (x4_all_xragn.ext != 0))].copy(),
                'x4_bad_o3_hx_allz_noext': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn < 1) & (x4_all_xragn.hardflux_sn > 2)& (x4_all_xragn.ext == 0))].copy(),



                'x4_bad_o3_hx_allz_ext_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn < 1) & (x4_all_xragn.hardflux_sn > 2)& (x4_all_xragn.ext != 0)&
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2))  )
                                                                            
                                                                            )].copy(),


                'x4_bad_o3_hx_allz_noext_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn < 1) &                                                                               
                                                                              (x4_all_xragn.hardflux_sn > 2)& (x4_all_xragn.ext == 0)&
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2))  )) ].copy(),

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
                'x4_sn1_o3_hx_allz_belowlxo3_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.3) &
                                                                          (x4_all_xragn.hardflux_sn > 2) &
                                                                          (x4_all_xragn.lo3_pred_fromlx > x4_all_xragn.oiiilum) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                         ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.oiiiflux_sn>2)&
                                                                        (x4_all_xragn.hbetaflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2) )
                                                                          )].copy(),
                'x4_sn1_o3_hx_allz_belowlxo3_noext_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.3) &
                                                                          (x4_all_xragn.hardflux_sn > 2) & (x4_all_xragn.ext ==0) &
                                                                          (x4_all_xragn.lo3_pred_fromlx > x4_all_xragn.oiiilum) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.oiiiflux_sn>2)&
                                                                        (x4_all_xragn.hbetaflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2) )
                                                                          )].copy(),

                'x4_sn1_o3_hx_allz_belowlxo3_ext_nobptsf': x4_all_xragn.iloc[np.where((x4_all_xragn.oiiiflux_sn > 1) & (x4_all_xragn.z <= 0.3) &
                                                                          (x4_all_xragn.hardflux_sn > 2) & (x4_all_xragn.ext >0) &
                                                                          (x4_all_xragn.lo3_pred_fromlx > x4_all_xragn.oiiilum) &
                                                                        ~(((x4_all_xragn.bptplusniigroups=='HII')|(x4_all_xragn.bptplusniigroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2)&
                                                                        ((x4_all_xragn.oiiiflux_sn<2)|
                                                                        (x4_all_xragn.hbetaflux_sn<2)) )&
                                                                              ~(((x4_all_xragn.bptplusgroups=='HII')|(x4_all_xragn.bptplusgroups=='MIX'))&
                                                                        (x4_all_xragn.niiflux_sn>2)&
                                                                        (x4_all_xragn.oiiiflux_sn>2)&
                                                                        (x4_all_xragn.hbetaflux_sn>2)&
                                                                        (x4_all_xragn.halpflux_sn>2) )
                                                                          )].copy(),

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
                'all_hx_ext': merged_xr_all.iloc[np.where((merged_xr_all.ext>0)&(merged_xr_all.hardflux_sn>2))[0]],
                'all_hx_noext': merged_xr_all.iloc[np.where((merged_xr_all.ext==0)&(merged_xr_all.hardflux_sn>2))[0]],
                
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
                'unclass_p1_hx': xragn_unclass_p1.iloc[np.where((xragn_unclass_p1.hardflux_sn>2))],
                'unclass_p2_hx': xragn_unclass_p2.iloc[np.where((xragn_unclass_p2.hardflux_sn>2))],

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

                'combo_sy2_hx_noext': merged_xr_sy2_all_combo.iloc[np.where((merged_xr_sy2_all_combo.z <= 0.3) & (merged_xr_sy2_all_combo.ext ==0) &(merged_xr_sy2_all_combo.hardflux_sn > 2))],
                'combo_sliner_hx_noext': merged_xr_sf_all_combo.iloc[np.where((merged_xr_sf_all_combo.z <= 0.3) &(merged_xr_sf_all_combo.ext==0) & (merged_xr_sf_all_combo.hardflux_sn > 2))],
                'combo_hliner_hx_noext': merged_xr_liner2_all_combo.iloc[np.where((merged_xr_liner2_all_combo.z <= 0.3) & (merged_xr_liner2_all_combo.ext==0) & (merged_xr_liner2_all_combo.hardflux_sn > 2))],
                'combo_all_liners_hx_noext': merged_xr_allliners_combo.iloc[np.where((merged_xr_allliners_combo.z <= 0.3) &(merged_xr_allliners_combo.ext ==0) & (merged_xr_allliners_combo.hardflux_sn > 2))],
                'combo_val_hx_noext': merged_xr_val_all_combo.iloc[np.where((merged_xr_val_all_combo.z <= 0.3) &(merged_xr_val_all_combo.ext ==0) & (merged_xr_val_all_combo.hardflux_sn > 2))],
                

                
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
                'nii_sf_xragn':xr_nii_xragn_sf_df.iloc[np.where((xr_nii_xragn_sf_df.hardflux_sn>2)&(xr_nii_xragn_sf_df.ext==0))],
                'bptplus_sf_allxr': xr_bptplus_sf_df.iloc[np.where(xr_bptplus_sf_df.hardflux_sn>2)[0]],
                
                'bptplus_sf_allxragn': xragn_bptplus_sf_df.iloc[np.where((xragn_bptplus_sf_df.hardflux_sn>2)&(xragn_bptplus_sf_df.ext==0))[0]],


                'high_sn_o3_bd_bpt_sf': xragn_bpt_sf_df.iloc[np.where((xragn_bpt_sf_df.hbetaflux_sn > 10) &
                                                                      (xragn_bpt_sf_df.halpflux_sn > 10) &
                                                                      (xragn_bpt_sf_df.oiiiflux_sn > 10))].copy(),
                'high_sn_lx_o3_bd_bpt_sf': xragn_bpt_sf_df.iloc[np.where((xragn_bpt_sf_df.hbetaflux_sn > 10) &
                                                                         (xragn_bpt_sf_df.halpflux_sn > 10) &
                                                                         (xragn_bpt_sf_df.oiiiflux_sn > 10) &
                                                                         (xragn_bpt_sf_df.fullflux_sn > 5))].copy()
                }

#cluster_info = pd.read_csv('x4_hx_allz_nobptsf_clusters.csv')
'''
xr_agn_props['x4_bad_o3_hx_allz_noext'] = xr_agn_props['x4_bad_o3_hx_allz_noext'].iloc[[1,2,3,4,5,7,8,9,11]]
xr_agn_props['x4_bad_o3_hx_allz_noext'].lo3_offset =np.random.random(9)*0.2+1.7


#xr_agn_props['nii_sf_xragn'].lo3_offset =np.random.random(9)*0.2+1.7
xr_agn_props['bptplus_sf_allxragn'].lo3_offset =xr_agn_props['bptplus_sf_allxragn'].lo3_offset+1
xr_agn_props['bptplus_sf_allxragn'].lo3_offset.iloc[np.where(xr_agn_props['bptplus_sf_allxragn'].lo3_offset>2)[0]] = 2

xr_agn_props['bptplus_sf_allxragn'].lo3_offset2 =xr_agn_props['bptplus_sf_allxragn'].lo3_pred_fromlx-xr_agn_props['bptplus_sf_allxragn'].oiiilum_sfsub_samir
xr_agn_props['bptplus_sf_allxragn'].lo3_offset2.iloc[np.where(~np.isfinite(xr_agn_props['bptplus_sf_allxragn'].lo3_offset2))] =xr_agn_props['bptplus_sf_allxragn'].lo3_offset


xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_oh'] = xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_oh'].iloc[[1]]
xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_oh'].lo3_offset =np.random.random(1)*0.2+1.7

xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_hard'] = xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_hard'].iloc[[1,2,4]]
xr_agn_props['x4_bad_o3_hx_allz_noext_nobptsf_hard'].lo3_offset =np.random.random(3)*0.2+1.7


xr_agn_props['x4_hx_allz_noext_nobptsf'][['ids', 'ra', 'dec',
                                        'hard_xraylum', 'hardflux_sn', 'full_xraylum','fullflux_sn', 
                                          'oiiilum',  'oiiiflux', 'oiii_err',
                                          'niiflux', 'nii_err',
                                          'halpflux', 'halp_err',
                                          'hbetaflux',  'hbeta_err',
                                          'halp_eqw',
                                          'balmerfwhm', 'forbiddenfwhm', 
                                          'av_gsw']]

o3_sn2_1 = Fits_set('o3_sn3_1237650796755222810_lx_42.8.fits')
o3_sn2_2 = Fits_set('o3_sn3_1237662263785357664_lx42.3.fits')
o3f1,o3l1 = o3_sn2_1.getcol(['flux', 'loglam'])
o3f2,o3l2 = o3_sn2_2.getcol(['flux', 'loglam'])

def sdssplot(wl, flux, wlmin = 3600, wlmax=9500, save=False, filename='', cont=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wl, flux, color='k', label='SDSS Spectrum')
    ax.set_xlabel('Wavelength (angstroms)', fontsize=20)
    ax.set_ylabel(r'Flux [$10^{-17}$ erg/cm$^2$/s/\AA]', fontsize=20)
    ax.set_xlim([wlmin, wlmax])
    ax.set_ylim([0, np.max(flux)+10])
    ax.tick_params(axis='both', which='major', labelsize=20, length=10, top=True, right=True)
    ax.tick_params(axis='both', which='minor', labelsize=15, length=5, top=True, right=True)
    ax.minorticks_on()
    ax.tick_params(direction='in',axis='both',which='both')
    plt.tight_layout()
    return ax
    #plt.legend(frameon=False,fontsize=12)
sdss1 = sdssplot(10**o3l1, o3f1)
plt.axvline(x=5760, ls='--')
inset = sdss1.inset_axes([0.65, 0.65, 0.35, 0.35])

#inset.set_tick_labels([])
inset.plot(10**o3l1[(10**o3l1<5780)&(10**o3l1>5740)],o3f1[(10**o3l1<5780)&(10**o3l1>5740)] )
inset.axvline(x=5760, ls='--')
plt.savefig('plots/o3_sn3_1_spec.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/o3_sn3_1_spec.png', dpi=250, bbox_inches='tight')
plt.close()

sdss2 = sdssplot(10**o3l2, o3f2)
plt.axvline(x=5400, ls='--')
inset = sdss2.inset_axes([0.55, 0.2, 0.35, 0.35])

#inset.set_tick_labels([])
inset.plot(10**o3l2[(10**o3l2<5430)&(10**o3l2>5370)],o3f2[(10**o3l2<5430)&(10**o3l2>5370)] )
inset.axvline(x=5400, ls='--')

plt.savefig('plots/o3_sn3_2_spec.pdf', dpi=250, bbox_inches='tight', format='pdf')
plt.savefig('plots/o3_sn3_2_spec.png', dpi=250, bbox_inches='tight')
plt.close()
'''

            