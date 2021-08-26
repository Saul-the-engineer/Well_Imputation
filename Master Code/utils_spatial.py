# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:50:49 2021

@author: saulg
"""

import numpy as np
import os
import pickle
import gstools as gs  # "conda install -c conda-forge gstools"
import matplotlib.pyplot as plt
import netCDF4
import shapely.geometry
import fiona
from copy import deepcopy

class krigging_interpolation():
    # Establish where files are located. If location doesn't exist, it will 
    # create a new location.
    def __init__(self, data_root ='./Datasets', figures_root = './Figures Spatial'):
        # Data Root is the location where data will loaded from. Saved to class
        # in order to reference later in other functions.
        if os.path.isdir(data_root) is False:
            os.makedirs(data_root)
        self.data_root = data_root
        
        # Fquifer Root is the location to save figures.
        if os.path.isdir(figures_root) is False:
            os.makedirs(figures_root)
        self.figures_root = figures_root

    # Opens generic pickle file based on file path and loads data.
    def read_pickle(self, file, root):
        file = root + file + '.pickle'
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    # Saves generic pickle file based on file path and loads data.
    def Save_Pickle(self, Data, name:str):
        with open(self.Data_root + '/' + name + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=4)

    # Load shapefile boundary
    def Shape_Boundary(self, shape_file_path):
        self.shape_file_path = shape_file_path
        user_shape = fiona.open(shape_file_path)
        return user_shape
    
    def create_grid_polygon(self, polygon, num_cells=10):
        # create grid coordinates for kriging, make x and y steps the same
        # x_steps is the number of cells in the x-direction

        # Unpack shapfile boundary fiona uses south east corner (sec) and 
        # north west corner (nwc) to determine boundary
        polygon_boundary = polygon.bounds
        sec_lon = polygon_boundary[0]
        sec_lat = polygon_boundary[1]
        nwc_lon = polygon_boundary[2]       
        nwc_lat = polygon_boundary[3]
        
        # Determine length of aquifer used in variaogram as well as setting up grid
        poly_lon_len = abs(polygon_boundary[2] - polygon_boundary[0])
        poly_lat_len = abs(polygon_boundary[3] - polygon_boundary[1])
        
        # Determine cell size based on number of desired cells for square grid
        dx = poly_lon_len/float(num_cells)
        dy = poly_lat_len/float(num_cells)

        # Extent of grid
        grid_lat = np.arange(nwc_lat, sec_lat - dy, - dy).tolist()
        grid_long = np.arange(sec_lon, nwc_lon + dx,  dx).tolist()
        
        # Mask Creation: Create an array the shape of grid. 1 will be that the cell
        # is located inside shape. 0 is outside shape.
        mask_array = np.ones((num_cells+1, num_cells+1)) # Array creation
        polygon = next(iter(polygon))   # Used with multipolygon
        polygon_coordinates = polygon['geometry']['coordinates'][0][0] # Coordinates of first polygon
        polygon_object = shapely.geometry.Polygon(polygon_coordinates) # Create shapely Polygon
        
        # Loop through every point to see if point is in shape
        for i, lat in enumerate(grid_lat):
            for j, long in enumerate(grid_long):
                point_temp = shapely.geometry.Point(long, lat) 
                if not polygon_object.contains(point_temp): mask_array[i,j] = 0
        # Save mask to class to use in interpolation. Change 0s to NANs to make
        # Data visualization correct.
        self.mask_array = np.where(mask_array == 0, np.nan, 1)
        return grid_long, grid_lat


    def netcdf_setup(self, grid_long, grid_lat, timestamp, filename):
        # setup a netcdf file to store the time series of rasters
        # copied from other lab code - you probably don't need everything here
        
        file = netCDF4.Dataset(self.data_root + '/' + filename, 'w', format="NETCDF4")
        
        lon_len  = len(grid_long)  # size of grid in x_dir
        lat_len  = len(grid_lat)  # size of grid in y_dir
        
        time = file.createDimension("time", None) # time dimension - can extend e.g., size=0
        lat  = file.createDimension("lat", lat_len)  # create lat dimension in netcdf file of len lat_len
        lon  = file.createDimension("lon", lon_len)  # create lon dimension in netcdf file of len lon_len
        
        time      = file.createVariable("time", np.float64, ("time"))
        latitude  = file.createVariable("lat", np.float64, ("lat"))  # create latitude varilable
        longitude = file.createVariable("lon", np.float64, ("lon")) # create longitude varilbe
        tsvalue   = file.createVariable("tsvalue", np.float64, ('time', 'lat', 'lon'), fill_value=-9999)
        
        # Netcdf seems to flip lat/long for building grid
        latitude[:] = grid_lat[:] 
        longitude[:] = grid_long[:] 

    
        latitude.long_name = "Latitude"
        latitude.units = "degrees_north"
        latitude.axis = "Y"
        
        longitude.long_name = "Longitude"
        longitude.units = "degrees_east"
        longitude.axis = "X"
        
        timestamp = list(timestamp.to_pydatetime())
        units = 'days since 0001-01-01 00:00:00'
        calendar = 'standard'
        time[:] = netCDF4.date2num(dates = timestamp, units = units, calendar= calendar)
        time.axis = "T"
        time.units = units
        
        return file, tsvalue
    
    
    def fit_model_var(self, x_c, y_c, values):
        # Check this: https://www.youtube.com/watch?v=bRj3HnEa1Z4&ab_channel=GeostatsGuyLectures
        # fit the model varigrom to the experimental variogram
        # this will fit a model variogram to experiemental data
        # however, occasionaly , there isn't a good fit.
        # the current version specifies a vargiogram rather than fitting one
        # the other code is still here, just commented out.
        # For the hard-coded variogram, we assume the range is 1/8 the size of the
        # area we are interested in
    
        bin_num = 20  # number of bins in the experimental variogram
        # first get the coords and determine distances
        x_delta = abs(x_c.max() - x_c.min()) # distance across x coords
        y_delta = abs(y_c.max() - y_c.min())  # distance across y coords
        max_dist = np.sqrt(x_delta**2 + y_delta**2) / 4  # assume correlated over 1/4 of distance
    
        # setup bins for the variogram
        bins_c = np.linspace(0, max_dist, bin_num)  # bin edges in variogram, bin_num of bins
    
        # compute the experimental variogram
        bin_cent_c, gamma_c = gs.vario_estimate_unstructured((x_c, y_c), values, bins_c)
        # bin_center_c is the "lag" of the bin, gamma_c is the value
    
        # # if fitting the variogram, this gets rid of 0 and low values
        # gamma_c = smooth(gamma_c, 5) # smooth the experimental variogram to help fitting - moving window with 5 samples
    
        # #----------------------------------------
        # # fit the model variogram
        # # potential models are: Gaussian, Exponential, Matern, Rational, Stable, Linear,
        # #                       Circular, Spherical, and Intersection
        # fit_var = gs.Linear(dim=1)
    
        # fit_var.fit_variogram(bin_cent_c, gamma_c, nugget=False)  # fit the model variogram
    
    
        # # check to see of range of varigram is very small (problem with fit), it so, set it to a value
        # # also check to see if it is too long and set it to a value
        # if fit_var.len_scale < max_dist/3:  # check if too short
        #     fit_var.len_scale = max_dist/3  # set to value
        #     #print('too short, set len_scale to: ', max_dist/3)
        # elif fit_var.len_scale > max_dist*1.5: # check if too long
        #     fit_var.len_scale = max_dist       # set to value
        #     #print('too long, set len_scale to: ', max_dist)
        # # End of variogram fitting stuff
    
        data_var = np.var(values)
        data_std = np.std(values)
        fit_var = gs.Stable(dim=2, var=data_var, len_scale=max_dist/2, nugget=data_std)
        # this line creates a variogram that has a range of 1/8 the longest distance
        # across the data; above we divided the max_dist by 2, here we divide by 4 so
        # 1/8
        # the code commented out above "fits" the variogram to the actual data
        # here we just specifiy the range, with the sill equal to the
        # variation of the data, and the nugget equal to the standard deviation.
        # the nugget is probably too big using this approach
        # we could se the nugget to 0
    

        # plot the variogram to show fit and print out variogram paramters
        # should make it save to pdf file
        ax1 = fit_var.plot(x_max=max_dist)  # plot model variogram
        ax1.plot(bin_cent_c, gamma_c)  # plot experimental variogram
        plt.show()
        print(fit_var)  # print out model variogram parameters.
        return fit_var
    
    def krig_field_generate(self, var_fitted, x_c, y_c, values, grid_x, grid_y, plot=True):
        # use GSTools to krig  the well data, need coords and value for each well
        # use model variogram paramters generated by GSTools
        # fast - is faster the variogram fitting
        krig_map = gs.krige.Ordinary(var_fitted, cond_pos=[x_c, y_c], cond_val=values)
        krig_map.structured([grid_x, grid_y]) # krig_map.field is the numpy array of values
        test = krig_map.field
        krig_map.field = krig_map.field.T * self.mask_array
        if plot==True:
            test_map = deepcopy(krig_map)
            test_map.field = test_map.field.T
            test_map.plot()
            plt.scatter(x_c, y_c, c='r')
            plt.show()
        return krig_map
    
