# Galaxy Data Science Repository
### Overview

This repository hosts a variety of Python scripts and utilities for analyzing and visualizing astronomical data, with a specific focus on SDSS-based galaxy observations and analyses, though other extensions should be easy/straightforward with the frameworks provided. Specia focus has been placed on extracting quantities related to the emission line measurements.

Repository Structure
---dist_met_procedures/: Contains procedures for distance metric calculations.

---observations/: Hosts configuration files and scripts related to galaxy observations and analysis thereafter. Some of the scripts therein and procedures will be further integrated into these updated core scripts but remain as-is for now.


---plotting_codes/: Includes scripts for generating plots and visualizations of the data.  See also the [chroptiks plotting package](https://github.com/cagostino/chroptiks) that was a result of the work contained here.

---xray_data_analysis/: Contains codes specific to the procurement and analysis of X-ray data.

---ast_utils.py: Utility functions for initial database setup and table joins.

---data_models.py: Defines the data models used for handling and storing astronomical data. I tried to reduce the procedures to only model additional changes to the underlying data, otherwise the column names will be the same as in the original datasets. I have already set up the procedure for loading and merging of the 4XMM DR8 x-ray dataset with the SDSS MPA/JHU catalog and the GSWLC catalog. 

---data_utils.py: Utilities for managing and manipulating datasets, e.g. merging dataframes with fuzzy criteria like coordinate distances between two astronomical catalogs.

---demarcations.py: Scripts for defining demarcations on emission line diagnostic diagrams

---image_utils.py: Functions for image loading and plotting. Still sort of under construction and will be expanded when I incorporate some of the helper functions I made for doing analysis on MUSE data. 

---load_data.py: Scripts for loading and initial processing of datasets.

---matplotlibrc: Matplotlib configuration file for consistent plotting styles.


### Notes

Adding other datasets should follow these steps: 
---1. create a data model for the new dataset to transform columns when loading them in as necessary. Use the AstroTablePD class as a base as it should be able to handle fits tables, csvs, and tsvs.
---2. load the data in and insert it into the general database using `insert_dataframe_to_table` from `data_utils`. In my case, I have created an sqlite3 database in the same folder where I saved the initial datasets (`catalogs/catalog_database.db`). By storing the data in the db, one can make cuts on data and do various sql table joins more easily and faster than just doing so in pandas. It is important to have the tables indexed as well or the data loading/joining will be slow.
