#!/bin/bash

# Fitting CALIFA galaxies spaxel by spaxel with STARLIGHT.
# Natalia@UFSC - 07/Dec/2016

 
# Natalia@Corrego - 23/Dec/2016

# List of galaxies
# list_gals="$HOME/data/CALIFA/dust_sample/dust_all.txt"
# 
# IFS=$'\n'
# 
# for gal in $(cat $list_gals | head -1)
# do
    # Get only CALIFA ID
    
     id="xr31"



    # Fit cube with pycasso2/STARLIGHT 
    python ~/pycasso2/scripts/pycasso_starlight.py \
            --overwrite \
           ${id}_bco3n_imported.fits \
     --out ${id}_bco3n_starlighted.fits
    
    
    
    
    id="xu22"



    # Fit cube with pycasso2/STARLIGHT 
    python ~/pycasso2/scripts/pycasso_starlight.py \
            --overwrite \
           ${id}_bco3n_imported.fits \
     --out ${id}_bco3n_starlighted.fits
     id="xu23"



    # Fit cube with pycasso2/STARLIGHT 
    python ~/pycasso2/scripts/pycasso_starlight.py \
            --overwrite \
           ${id}_bco3n_imported.fits \
     --out ${id}_bco3n_starlighted.fits
     id="xu104"



    # Fit cube with pycasso2/STARLIGHT 
    python ~/pycasso2/scripts/pycasso_starlight.py \
            --overwrite \
           ${id}_bco3n_imported.fits \
     --out ${id}_bco3n_starlighted.fits
     id="xu210"



    # Fit cube with pycasso2/STARLIGHT 
    python ~/pycasso2/scripts/pycasso_starlight.py \
            --overwrite \
           ${id}_bco3n_imported.fits \
     --out ${id}_bco3n_starlighted.fits
    
    #     --use-error-flag
# done
# 
# unset IFS
# 
# echo '@@> Done. :) '

# EOF

# ++python ~/code/python/pycasso2/scripts/pycasso_import.py --cube-type califa \
# ++             ~/data/CALIFA/COMB_v2.2/K0073.COMB.rscube.fits.gz \
# ++       --out ~/data/CALIFA/COMB_v2.2_p2imported/K0073.COMB.rscube.fits

#++python ~/code/python/pycasso2/scripts/pycasso_starlight.py \
#++             ~/data/CALIFA/COMB_v2.2_p2imported/K0073.COMB.rscube.fits \
#++       --out ~/data/CALIFA/COMB_v2.2_p2starlight/K0073.COMB.rscube.fits
