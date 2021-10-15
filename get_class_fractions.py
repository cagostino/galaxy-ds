#!/usr/bin/env python3
# -*- coding: utf-8 -*-



sy2_bpt1_groups , sy2_ke_sf, sy2_ke_comp, sy2_ke_sy2, sy2_ke_liner = get_bpt1_groups_ke01( sfrm_gsw2.fullagn_df.niiha_sub.iloc[sy2_1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[sy2_1] )
sliner_bpt1_groups, sliner_ke_sf, sliner_ke_comp, sliner_ke_sy2, sliner_ke_liner= get_bpt1_groups_ke01(  sfrm_gsw2.fullagn_df.niiha_sub.iloc[sf_1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[sf_1]  )
hliner_bpt1_groups, hliner_ke_sf, hliner_ke_comp,hliner_ke_sy2, hliner_ke_liner= get_bpt1_groups_ke01( sfrm_gsw2.fullagn_df.niiha_sub.iloc[liner2_1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[liner2_1]  )
val1_bpt1_groups, val1_ke_sf, val1_ke_comp,val1_ke_sy2, val1_ke_liner= get_bpt1_groups_ke01( sfrm_gsw2.fullagn_df.niiha_sub.iloc[val1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[val1]  )

sy2_bpt1_groups , sy2_ke_sf, sy2_ke_comp, sy2_ke_sy2, sy2_ke_liner = get_bpt1_groups_ke01( sfrm_gsw2.fullagn_df.niiha.iloc[sy2_1], sfrm_gsw2.fullagn_df.oiiihb.iloc[sy2_1] )
sliner_bpt1_groups, sliner_ke_sf, sliner_ke_comp, sliner_ke_sy2, sliner_ke_liner= get_bpt1_groups_ke01(  sfrm_gsw2.fullagn_df.niiha.iloc[sf_1], sfrm_gsw2.fullagn_df.oiiihb.iloc[sf_1]  )
hliner_bpt1_groups, hliner_ke_sf, hliner_ke_comp,hliner_ke_sy2, hliner_ke_liner= get_bpt1_groups_ke01( sfrm_gsw2.fullagn_df.niiha.iloc[liner2_1], sfrm_gsw2.fullagn_df.oiiihb.iloc[liner2_1]  )
val1_bpt1_groups, val1_ke_sf, val1_ke_comp,val1_ke_sy2, val1_ke_liner= get_bpt1_groups_ke01( sfrm_gsw2.fullagn_df.niiha.iloc[val1], sfrm_gsw2.fullagn_df.oiiihb.iloc[val1]  )


frac_sy2_ke_sf  = sy2_ke_sf.size/sy2_bpt1_groups.size
frac_sy2_ke_comp = sy2_ke_comp.size/sy2_bpt1_groups.size
frac_sy2_ke_sy2 = sy2_ke_sy2.size/sy2_bpt1_groups.size
frac_sy2_ke_liner = sy2_ke_liner.size/sy2_bpt1_groups.size


frac_sliner_ke_sf  =   sliner_ke_sf.size/sliner_bpt1_groups.size
frac_sliner_ke_comp = sliner_ke_comp.size/sliner_bpt1_groups.size
frac_sliner_ke_sy2 = sliner_ke_sy2.size/sliner_bpt1_groups.size
frac_sliner_ke_liner = sliner_ke_liner.size/sliner_bpt1_groups.size


frac_hliner_ke_sf = hliner_ke_sf.size/hliner_bpt1_groups.size
frac_hliner_ke_comp  = hliner_ke_comp.size/hliner_bpt1_groups.size
frac_hliner_ke_sy2 = hliner_ke_sy2.size/hliner_bpt1_groups.size
frac_hliner_ke_liner = hliner_ke_liner.size/hliner_bpt1_groups.size

frac_val1_ke_sf = val1_ke_sf.size/val1_bpt1_groups.size
frac_val1_ke_comp  = val1_ke_comp.size/val1_bpt1_groups.size
frac_val1_ke_sy2 = val1_ke_sy2.size/val1_bpt1_groups.size
frac_val1_ke_liner = val1_ke_liner.size/val1_bpt1_groups.size

print(frac_sy2_ke_sf,'&', frac_sy2_ke_comp, '&', frac_sy2_ke_sy2, '&', frac_sy2_ke_liner)
print(frac_sliner_ke_sf,'&', frac_sliner_ke_comp, '&', frac_sliner_ke_sy2, '&', frac_sliner_ke_liner)
print(frac_hliner_ke_sf,'&', frac_hliner_ke_comp, '&', frac_hliner_ke_sy2, '&', frac_hliner_ke_liner)
print(frac_val1_ke_sf,'&', frac_val1_ke_comp, '&', frac_val1_ke_sy2, '&', frac_val1_ke_liner)



sy2_bpt2_groups , sy2_s2_sf, sy2_s2_seyf, sy2_s2_liner= get_bpt2_groups( sfrm_gsw2.fullagn_df.siiha_sub.iloc[sy2_1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[sy2_1] )
sliner_bpt2_groups, sliner_s2_sf, sliner_s2_seyf, sliner_s2_liner= get_bpt2_groups(  sfrm_gsw2.fullagn_df.siiha_sub.iloc[sf_1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[sf_1]  )
hliner_bpt2_groups, hliner_s2_sf, hliner_s2_seyf, hliner_s2_liner= get_bpt2_groups( sfrm_gsw2.fullagn_df.siiha_sub.iloc[liner2_1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[liner2_1]  )
val1_bpt2_groups, val1_s2_sf, val1_s2_seyf,val1_s2_liner= get_bpt2_groups( sfrm_gsw2.fullagn_df.siiha_sub.iloc[val1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[val1]  )

sy2_bpt2_groups , sy2_s2_sf, sy2_s2_seyf, sy2_s2_liner= get_bpt2_groups( sfrm_gsw2.fullagn_df.siiha.iloc[sy2_1], sfrm_gsw2.fullagn_df.oiiihb.iloc[sy2_1] )
sliner_bpt2_groups, sliner_s2_sf, sliner_s2_seyf, sliner_s2_liner= get_bpt2_groups(  sfrm_gsw2.fullagn_df.siiha.iloc[sf_1], sfrm_gsw2.fullagn_df.oiiihb.iloc[sf_1]  )
hliner_bpt2_groups, hliner_s2_sf, hliner_s2_seyf, hliner_s2_liner= get_bpt2_groups( sfrm_gsw2.fullagn_df.siiha.iloc[liner2_1], sfrm_gsw2.fullagn_df.oiiihb.iloc[liner2_1]  )
val1_bpt2_groups, val1_s2_sf, val1_s2_seyf,val1_s2_liner= get_bpt2_groups( sfrm_gsw2.fullagn_df.siiha.iloc[val1], sfrm_gsw2.fullagn_df.oiiihb.iloc[val1]  )



frac_sy2_s2_sf  = sy2_s2_sf.size/sy2_bpt2_groups.size
frac_sy2_s2_seyf = sy2_s2_seyf.size/sy2_bpt2_groups.size
frac_sy2_s2_liner = sy2_s2_liner.size/sy2_bpt2_groups.size

frac_sliner_s2_sf  =   sliner_s2_sf.size/sliner_bpt2_groups.size
frac_sliner_s2_seyf = sliner_s2_seyf.size/sliner_bpt2_groups.size
frac_sliner_s2_liner = sliner_s2_liner.size/sliner_bpt2_groups.size

frac_hliner_s2_sf = hliner_s2_sf.size/hliner_bpt2_groups.size
frac_hliner_s2_seyf = hliner_s2_seyf.size/hliner_bpt2_groups.size
frac_hliner_s2_liner = hliner_s2_liner.size/hliner_bpt2_groups.size

frac_val1_s2_sf = val1_s2_sf.size/val1_bpt2_groups.size
frac_val1_s2_seyf  = val1_s2_seyf.size/val1_bpt2_groups.size
frac_val1_s2_liner = val1_s2_liner.size/val1_bpt2_groups.size

print(frac_sy2_s2_sf,'&', frac_sy2_s2_seyf, '&', frac_sy2_s2_liner)
print(frac_sliner_s2_sf,'&', frac_sliner_s2_seyf, '&', frac_sliner_s2_liner)
print(frac_hliner_s2_sf,'&', frac_hliner_s2_seyf, '&', frac_hliner_s2_liner)
print(frac_val1_s2_sf,'&', frac_val1_s2_seyf, '&', frac_val1_s2_liner)





sy2_bpt3_groups , sy2_o1_sf, sy2_o1_seyf, sy2_o1_liner= get_bpt3_groups( sfrm_gsw2.fullagn_df.oiha_sub.iloc[sy2_1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[sy2_1] )
sliner_bpt3_groups, sliner_o1_sf, sliner_o1_seyf, sliner_o1_liner= get_bpt3_groups(  sfrm_gsw2.fullagn_df.oiha_sub.iloc[sf_1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[sf_1]  )
hliner_bpt3_groups, hliner_o1_sf, hliner_o1_seyf, hliner_o1_liner= get_bpt3_groups( sfrm_gsw2.fullagn_df.oiha_sub.iloc[liner2_1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[liner2_1]  )
val1_bpt3_groups, val1_o1_sf, val1_o1_seyf,val1_o1_liner= get_bpt3_groups( sfrm_gsw2.fullagn_df.oiha_sub.iloc[val1], sfrm_gsw2.fullagn_df.oiiihb_sub.iloc[val1]  )


sy2_bpt3_groups , sy2_o1_sf, sy2_o1_seyf, sy2_o1_liner= get_bpt3_groups( sfrm_gsw2.fullagn_df.oiha.iloc[sy2_1], sfrm_gsw2.fullagn_df.oiiihb.iloc[sy2_1] )
sliner_bpt3_groups, sliner_o1_sf, sliner_o1_seyf, sliner_o1_liner= get_bpt3_groups(  sfrm_gsw2.fullagn_df.oiha.iloc[sf_1], sfrm_gsw2.fullagn_df.oiiihb.iloc[sf_1]  )
hliner_bpt3_groups, hliner_o1_sf, hliner_o1_seyf, hliner_o1_liner= get_bpt3_groups( sfrm_gsw2.fullagn_df.oiha.iloc[liner2_1], sfrm_gsw2.fullagn_df.oiiihb.iloc[liner2_1]  )
val1_bpt3_groups, val1_o1_sf, val1_o1_seyf,val1_o1_liner= get_bpt3_groups( sfrm_gsw2.fullagn_df.oiha.iloc[val1], sfrm_gsw2.fullagn_df.oiiihb.iloc[val1]  )


frac_sy2_o1_sf  = sy2_o1_sf.size/sy2_bpt3_groups.size
frac_sy2_o1_seyf = sy2_o1_seyf.size/sy2_bpt3_groups.size
frac_sy2_o1_liner = sy2_o1_liner.size/sy2_bpt3_groups.size

frac_sliner_o1_sf  =   sliner_o1_sf.size/sliner_bpt3_groups.size
frac_sliner_o1_seyf = sliner_o1_seyf.size/sliner_bpt3_groups.size
frac_sliner_o1_liner = sliner_o1_liner.size/sliner_bpt3_groups.size

frac_hliner_o1_sf = hliner_o1_sf.size/hliner_bpt3_groups.size
frac_hliner_o1_seyf = hliner_o1_seyf.size/hliner_bpt3_groups.size
frac_hliner_o1_liner = hliner_o1_liner.size/hliner_bpt3_groups.size

frac_val1_o1_sf = val1_o1_sf.size/val1_bpt3_groups.size
frac_val1_o1_seyf  = val1_o1_seyf.size/val1_bpt3_groups.size
frac_val1_o1_liner = val1_o1_liner.size/val1_bpt3_groups.size


print(frac_sy2_o1_sf,'&', frac_sy2_o1_seyf, '&', frac_sy2_o1_liner)
print(frac_sliner_o1_sf,'&', frac_sliner_o1_seyf, '&', frac_sliner_o1_liner)
print(frac_hliner_o1_sf,'&', frac_hliner_o1_seyf, '&', frac_hliner_o1_liner)
print(frac_val1_o1_sf,'&', frac_val1_o1_seyf, '&', frac_val1_o1_liner)





sy2_bptooo_groups , sy2_ooo_sf, sy2_ooo_seyf, sy2_ooo_liner= get_ooo_groups( sfrm_gsw2.fullagn_df.oiha_sub.iloc[sy2_1], sfrm_gsw2.fullagn_df.oiii_oii_sub.iloc[sy2_1] )
sliner_bptooo_groups, sliner_ooo_sf, sliner_ooo_seyf, sliner_ooo_liner= get_ooo_groups(  sfrm_gsw2.fullagn_df.oiha_sub.iloc[sf_1], sfrm_gsw2.fullagn_df.oiii_oii_sub.iloc[sf_1]  )
hliner_bptooo_groups, hliner_ooo_sf, hliner_ooo_seyf, hliner_ooo_liner= get_ooo_groups( sfrm_gsw2.fullagn_df.oiha_sub.iloc[liner2_1], sfrm_gsw2.fullagn_df.oiii_oii_sub.iloc[liner2_1]  )
val1_bptooo_groups, val1_ooo_sf, val1_ooo_seyf,val1_ooo_liner= get_ooo_groups( sfrm_gsw2.fullagn_df.oiha_sub.iloc[val1], sfrm_gsw2.fullagn_df.oiii_oii_sub.iloc[val1]  )

sy2_bptooo_groups , sy2_ooo_sf, sy2_ooo_seyf, sy2_ooo_liner= get_ooo_groups( sfrm_gsw2.fullagn_df.oiha.iloc[sy2_1], sfrm_gsw2.fullagn_df.oiii_oii.iloc[sy2_1] )
sliner_bptooo_groups, sliner_ooo_sf, sliner_ooo_seyf, sliner_ooo_liner= get_ooo_groups(  sfrm_gsw2.fullagn_df.oiha.iloc[sf_1], sfrm_gsw2.fullagn_df.oiii_oii.iloc[sf_1]  )
hliner_bptooo_groups, hliner_ooo_sf, hliner_ooo_seyf, hliner_ooo_liner= get_ooo_groups( sfrm_gsw2.fullagn_df.oiha.iloc[liner2_1], sfrm_gsw2.fullagn_df.oiii_oii.iloc[liner2_1]  )
val1_bptooo_groups, val1_ooo_sf, val1_ooo_seyf,val1_ooo_liner= get_ooo_groups( sfrm_gsw2.fullagn_df.oiha.iloc[val1], sfrm_gsw2.fullagn_df.oiii_oii.iloc[val1]  )




frac_sy2_ooo_sf  = sy2_ooo_sf.size/sy2_bptooo_groups.size
frac_sy2_ooo_seyf = sy2_ooo_seyf.size/sy2_bptooo_groups.size
frac_sy2_ooo_liner = sy2_ooo_liner.size/sy2_bptooo_groups.size

frac_sliner_ooo_sf  =   sliner_ooo_sf.size/sliner_bptooo_groups.size
frac_sliner_ooo_seyf = sliner_ooo_seyf.size/sliner_bptooo_groups.size
frac_sliner_ooo_liner = sliner_ooo_liner.size/sliner_bptooo_groups.size

frac_hliner_ooo_sf = hliner_ooo_sf.size/hliner_bptooo_groups.size
frac_hliner_ooo_seyf = hliner_ooo_seyf.size/hliner_bptooo_groups.size
frac_hliner_ooo_liner = hliner_ooo_liner.size/hliner_bptooo_groups.size

frac_val1_ooo_sf = val1_ooo_sf.size/val1_bptooo_groups.size
frac_val1_ooo_seyf  = val1_ooo_seyf.size/val1_bptooo_groups.size
frac_val1_ooo_liner = val1_ooo_liner.size/val1_bptooo_groups.size

print(frac_sy2_ooo_sf,'&', frac_sy2_ooo_seyf, '&', frac_sy2_ooo_liner)
print(frac_sliner_ooo_sf,'&', frac_sliner_ooo_seyf, '&', frac_sliner_ooo_liner)
print(frac_hliner_ooo_sf,'&', frac_hliner_ooo_seyf, '&', frac_hliner_ooo_liner)
print(frac_val1_ooo_sf,'&', frac_val1_ooo_seyf, '&', frac_val1_ooo_liner)





sy2_bptwhan_groups , sy2_whan_sf, sy2_whan_seyf, sy2_whan_liner,sy2_whan_rg,sy2_whan_pg= get_whan_groups( sfrm_gsw2.fullagn_df.niiha_sub.iloc[sy2_1], -sfrm_gsw2.fullagn_df.halp_eqw_sub.iloc[sy2_1] )
sliner_bptwhan_groups, sliner_whan_sf, sliner_whan_seyf, sliner_whan_liner, sliner_whan_rg,sliner_whan_pg= get_whan_groups(  sfrm_gsw2.fullagn_df.niiha_sub.iloc[sf_1], -sfrm_gsw2.fullagn_df.halp_eqw_sub.iloc[sf_1]  )
hliner_bptwhan_groups, hliner_whan_sf, hliner_whan_seyf, hliner_whan_liner,hliner_whan_rg,hliner_whan_pg= get_whan_groups( sfrm_gsw2.fullagn_df.niiha_sub.iloc[liner2_1], -sfrm_gsw2.fullagn_df.halp_eqw_sub.iloc[liner2_1]  )
val1_bptwhan_groups, val1_whan_sf, val1_whan_seyf,val1_whan_liner, val1_whan_rg, val1_whan_pg= get_whan_groups( sfrm_gsw2.fullagn_df.niiha_sub.iloc[val1], -sfrm_gsw2.fullagn_df.halp_eqw_sub.iloc[val1]  )


sy2_bptwhan_groups , sy2_whan_sf, sy2_whan_seyf, sy2_whan_liner,sy2_whan_rg,sy2_whan_pg= get_whan_groups( sfrm_gsw2.fullagn_df.niiha.iloc[sy2_1], -sfrm_gsw2.fullagn_df.halp_eqw.iloc[sy2_1] )
sliner_bptwhan_groups, sliner_whan_sf, sliner_whan_seyf, sliner_whan_liner, sliner_whan_rg,sliner_whan_pg= get_whan_groups(  sfrm_gsw2.fullagn_df.niiha.iloc[sf_1], -sfrm_gsw2.fullagn_df.halp_eqw.iloc[sf_1]  )
hliner_bptwhan_groups, hliner_whan_sf, hliner_whan_seyf, hliner_whan_liner,hliner_whan_rg,hliner_whan_pg= get_whan_groups( sfrm_gsw2.fullagn_df.niiha.iloc[liner2_1], -sfrm_gsw2.fullagn_df.halp_eqw.iloc[liner2_1]  )
val1_bptwhan_groups, val1_whan_sf, val1_whan_seyf,val1_whan_liner, val1_whan_rg, val1_whan_pg= get_whan_groups( sfrm_gsw2.fullagn_df.niiha.iloc[val1], -sfrm_gsw2.fullagn_df.halp_eqw.iloc[val1]  )



frac_sy2_whan_sf  = sy2_whan_sf.size/sy2_bptwhan_groups.size
frac_sy2_whan_seyf = sy2_whan_seyf.size/sy2_bptwhan_groups.size
frac_sy2_whan_liner = sy2_whan_liner.size/sy2_bptwhan_groups.size
frac_sy2_whan_rg = sy2_whan_rg.size/sy2_bptwhan_groups.size
frac_sy2_whan_pg = sy2_whan_pg.size/sy2_bptwhan_groups.size

frac_sliner_whan_sf  =   sliner_whan_sf.size/sliner_bptwhan_groups.size
frac_sliner_whan_seyf = sliner_whan_seyf.size/sliner_bptwhan_groups.size
frac_sliner_whan_liner = sliner_whan_liner.size/sliner_bptwhan_groups.size
frac_sliner_whan_rg = sliner_whan_rg.size/sliner_bptwhan_groups.size
frac_sliner_whan_pg = sliner_whan_pg.size/sliner_bptwhan_groups.size

frac_hliner_whan_sf = hliner_whan_sf.size/hliner_bptwhan_groups.size
frac_hliner_whan_seyf = hliner_whan_seyf.size/hliner_bptwhan_groups.size
frac_hliner_whan_liner = hliner_whan_liner.size/hliner_bptwhan_groups.size
frac_hliner_whan_rg = hliner_whan_rg.size/hliner_bptwhan_groups.size
frac_hliner_whan_pg = hliner_whan_pg.size/hliner_bptwhan_groups.size

frac_val1_whan_sf = val1_whan_sf.size/val1_bptwhan_groups.size
frac_val1_whan_seyf  = val1_whan_seyf.size/val1_bptwhan_groups.size
frac_val1_whan_liner = val1_whan_liner.size/val1_bptwhan_groups.size
frac_val1_whan_rg = val1_whan_rg.size/val1_bptwhan_groups.size
frac_val1_whan_pg = val1_whan_pg.size/val1_bptwhan_groups.size

print(frac_sy2_whan_sf,'&', frac_sy2_whan_seyf, '&', frac_sy2_whan_liner, '&',frac_sy2_whan_rg,'&',  frac_sy2_whan_pg)
print(frac_sliner_whan_sf,'&', frac_sliner_whan_seyf, '&', frac_sliner_whan_liner, '&',frac_sliner_whan_rg,'&',  frac_sliner_whan_pg)
print(frac_hliner_whan_sf,'&', frac_hliner_whan_seyf, '&', frac_hliner_whan_liner, '&',frac_hliner_whan_rg,'&',  frac_hliner_whan_pg)
print(frac_val1_whan_sf,'&', frac_val1_whan_seyf, '&', frac_val1_whan_liner, '&',frac_val1_whan_rg,'&',  frac_val1_whan_pg)

