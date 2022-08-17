# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:17:37 2021

@author: saulg
"""

import utils_01_satellite_data as usd
''''# GLADS
north_east_corner_lat = 89.875
south_west_corner_lat = -59.875
north_east_corner_lon = 179.875
south_west_corner_lon = -179.875
dx = 0.25
dy = 0.25

#PDSI
north_east_corner_lat = 77.5
south_west_corner_lat = -60.0
north_east_corner_lon = 180.0
south_west_corner_lon = -180.0
dx = 2.5
dy = 2.5'''

# Purpose of this script is to: 
# 1) Create a grid in the likeness of the netcdf we're interested in querying.
# 2) Using the grid, identify which cells are located within a shape file boundary.
# 3) Use the grids package, and to query the time series, for the variables and cells
#    the user has specified.
# 4) Save data as a pickle file.

# Set data location. If location does not exist, class initialization will create it.
data_root = './Datasets/'
ts_date_start = '1850-01-01'

# Class initialization, imports methods and creates data_root if it doesn't exist.
utils = usd.utils_netCDF(data_root)
# Create grid based on netcdf metadata. Inputs are NE_lat, SW_lat, NE_lon, SW_lon
# x resolution, and y resolution. Calculates the centroids.
grid = utils.netCDF_Grid_Creation(77.5, -60.0, 180.0, -180.0, 2.5, 2.5)
# Loads shape file and obtains bounding box coordinates.
bounds = utils.Shape_Boundary('./Aquifer Shapes/Escalante_Beryl.shp')
# Loop through grid determining if centroid is within shape boundary. Returns 
# boolean. Hyper-paramters include buffering and padding. Buffer is half cell size
# used to make sure approproate cells are captured.
cell_names = utils.Find_intercepting_cells(grid, bounds)
# Select intercepting cells from dictionary.
mask = utils.Cell_Mask(cell_names, grid)
# Create index from starting date until X weeks from present day.
dates = utils.Date_Index_Creation(ts_date_start)


# The grids_netCDF class is used to handle either the GLDAS or SC PDSI PM dataset
# The file structures are very different. Areas that have missing values, GLDAS
# will treat as NAN, while pdsi treats as mask 'None'. PDSI is a single variable file
# while GLDAS is a multi-variable file per month.

# File_String refers to the data being located with a single string, or list of strings
# Variable String, referes to the variables being within a string or text file.
parse = usd.grids_netCDF(File_String=True, Variable_String=True)
# Location of single netCDF file.
data_path = r'C:\Users\saulg\Desktop\Remote_Data\sc_PDSI_pm\pdsi_ramirez_hales_williams_1850-2020.nc4'
# Name of variables of interest within file.
variables = 'sc_PDSI_pm_filled'

# Parse_Data is a nested loop using the grids python package. For every cell, grabs
# every specified variable. Then parses the variable assigning it to the correct cell
# For PDSI use Mask, dates, data_path, and variable name.

#   Mask: dictionary with locations. 
#   dates: datetimeindex, used to index dataframe
#   data_path: Location of netcdf if it is a single file, else None
#   variable_name: Names of variables wanted parse.

Data = parse.Parse_Data(mask, dates, data_path = data_path, variable_name=variables)

# Validates that all data exists, removes nans and Nones from time series,
# as well as the location dataframe.
Data = parse.Validate_Data(mask, Data)

# Final Save.
utils.Save_Pickle(Data,'PDSI_Data', data_root)