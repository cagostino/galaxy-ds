#!/bin/bash

# Importing CALIFA galaxies with pycasso2.
# Natalia@Corrego - 23/Dec/2016

# List of galaxies
# list_gals="$HOME/data/CALIFA/dust_sample/dust_all.txt"
# 
# IFS=$'\n'
# 
# for gal in $(cat $list_gals | head -1)
# do
#     # Get only CALIFA ID
    id="xr31"

    # Import cube with pycasso2
python ~/pycasso2/scripts/pycasso_import.py \
    --cube-type muse --name ${id} --overwrite \
    ${id}_stamp.fits \
    --out ${id}_bco3n_imported.fits
     id="xu22"

    # Import cube with pycasso2
python ~/pycasso2/scripts/pycasso_import.py \
    --cube-type muse --name ${id} --overwrite \
    ${id}_stamp.fits \
    --out ${id}_bco3n_imported.fits
      id="xu23"

    # Import cube with pycasso2
python ~/pycasso2/scripts/pycasso_import.py \
    --cube-type muse --name ${id} --overwrite \
    ${id}_stamp.fits \
    --out ${id}_bco3n_imported.fits
       id="xu104"

    # Import cube with pycasso2
python ~/pycasso2/scripts/pycasso_import.py \
    --cube-type muse --name ${id} --overwrite \
    ${id}_stamp.fits \
    --out ${id}_bco3n_imported.fits
     
    id="xu210"

    # Import cube with pycasso2
python ~/pycasso2/scripts/pycasso_import.py \
    --cube-type muse --name ${id} --overwrite \
    ${id}_stamp.fits \
    --out ${id}_bco3n_imported.fits
 
# done
# 
# unset IFS
# 
# echo '@@> Done. :) '

# EOF

