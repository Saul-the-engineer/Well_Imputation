# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:17:37 2021

@author: saulg
"""

import utils_satellite_data as usd
''''# GLADS
north_east_corner_lat = 89.875
south_west_corner_lat = -59.875
north_east_corner_lon = 179.875
south_west_corner_lon = -179.875
dx = 0.25
dy = 0.25
'''

# Purpose of this script is to: 
# 1) Create a grid in the likeness of the netcdf we're interested in querying.
# 2) Using the grid, identify which cells are located within a shape file boundary.
# 3) Use the grids package, and to query the time series, for the variables and cells
#    the user has specified.
# 4) Save data as a pickle file.


# data_root is the location where data will be saved
data_root = './Datasets/'
# shape_location is the location to the Shape file
shape_location = './Aquifer Shapes/Escalante_Beryl.shp'
# gldas_root is the location of folder containing the tabulated GLDAS
gldas_root = r'C:\Users\saulg\Desktop\Remote_Data\Tabular GLDAS'
# variables_list_loc the text file containing the GLDAS variable names
variables_list_loc = r'C:\Users\saulg\OneDrive\Dissertation\Well Imputation\Master Code\Satellite Data Prep\variables_list.txt'

# Class initialization, imports methods and creates data_root if it doesn't exist.
utils = usd.utils_netCDF(data_root)
# Create grid based on netcdf metadata. Inputs are NE_lat, SW_lat, NE_lon, SW_lon
# x resolution, and y resolution. Calculates the centroids.
grid = utils.netCDF_Grid_Creation(89.875, -59.875, 179.875, -179.875, 0.25, 0.25)
# Loads shape file and obtains bounding box coordinates.
bounds = utils.Shape_Boundary(shape_location)

# Loop through grid determining if centroid is within shape boundary. Returns 
# boolean. Hyper-paramters include buffering and padding. Buffer is half cell size
# used to make sure approproate cells are captured.
cell_names = utils.Find_intercepting_cells(grid, bounds)
# Select intercepting cells from dictionary.
mask = utils.Cell_Mask(cell_names, grid)
# Construct list of cell names within shapefile
cell_names = list(cell_names.keys())
# Create GLDAS Parsing class
GLDAS_parse = usd.GLDAS_parse(gldas_root, cell_names)
# Open variable text file, load data, convert to list
variables_list = GLDAS_parse.Variable_List(variables_list_loc)
# Create subset of GLDAS variable columns based on mask
Variable_Dictionary = GLDAS_parse.Open_GLDAS(variables_list, mask)
# Split variables into cells and store them into a dictionary along with location
Data = GLDAS_parse.parse(Variable_Dictionary, mask)
# Save data
utils.Save_Pickle(Data,'GLDAS_Data', data_root)