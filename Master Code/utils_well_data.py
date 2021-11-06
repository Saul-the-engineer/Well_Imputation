# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:12:27 2020

@author: Saul Ramirez, PhD
"""
import os
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pandas import DatetimeIndex
from scipy import interpolate


# Support Script for Groundwater Imputation Tool 
# This script opens preprocessed pickle file containing 3 DataFrames: Centroid,
# Well Timeseries, and well location.
# High level does 3 things:
# 1: Extracts Well Time series, and drops any well without a minimum amount
#    of Data within a user specified window
# 2: Interpolates subset of data, removing gaps larger than user specified size
#    Gaps will be filled in using a machine learning algorithm. This step
#    is based on the assumption that groundwater changes are slow, and readings
#    will remain valid for a time period after the original reading.
# 3: Contains support functions of data augmentation such as Rolling-Window 
#    averages and plotting. 


class wellfunc():
    # Establish where files are located. If location doesn't exist, it will 
    # create a new location.
    def __init__(self, data_root ='./Datasets', aquifer_root ='./Aquifers Data', figures_root = './Figures Aquifer'):
        # Data Root is the location where data will be saved to. Saved to class
        # in order to reference later in other functions.
        if os.path.isdir(data_root) is False:
            os.makedirs(data_root)
        self.Data_root = data_root
        
        # Aquifer Root is the location of Raw Aquifer pickle file. The file
        # structure of the aquifer pickle is a python dictionary with 3 
        # DataFrames: Cetroid, Data, and Location
        if os.path.isdir(aquifer_root) is False:
            os.makedirs(aquifer_root)
            print('The Aquifer folder with data is empty')
        self.Aquifer_root = aquifer_root
        
        # Fquifer Root is the location to save figures.
        if os.path.isdir(figures_root) is False:
            os.makedirs(figures_root)
        self.figures_root = figures_root
        
    '''###################################
                Well Data Sampling
    '''###################################

    # Specifically opens raw aquifer pickle file.
    def read_well_pickle(self, well_file):
        self.wellfile = self.Aquifer_root + '/' + well_file + '.pickle'
        with open(self.wellfile, 'rb') as handle:
            wells = pickle.load(handle)
        return wells
    
    # Opens generic pickle file based on file path and loads data.
    def read_pickle(self, file, root):
        file = root + file + '.pickle'
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    # Saves generic pickle file based on file path and loads data.
    def Save_Pickle(self, Data, name:str, protocol:int = 3):
        with open(self.Data_root + name + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=protocol)
    
    # Extracts Well Time series, and drops any well without a minimum amount
    # of Data (MinExTotal) within a user specified window (Left, Right)
    def extractwelldata(self, wells, Left=1948, Right=2018, Min_Obs_Months=50, outlier = 3):
        # Validates that right-side window is greater than left-side window.
        assert Left < Right, 'Error: Left Cap Year is greater than Right Cap year.'
        # Validates that Aquifer pickle dataframe contains 
        assert Min_Obs_Months > 0, 'Error: Original Data dataframe must have at least 1 example.' 
        
        wells_dict = dict() #Create data dictionary, used to return data
        
        #Create caps as pandas time stamps to compare dates
        begtime = pd.Timestamp(dt.datetime(Left, 1, 1))# data before this date
        endtime = pd.Timestamp(dt.datetime(Right, 1, 1))# data after this date
        
        # Vectorized subsetting of well data. Mask wells with data between 
        # left cap (begtime) and right cap (endtime). Results in binary array
        # showing wells within time range containing specified number of points.
        mask = (wells['Data'].index > begtime) & (wells['Data'].index < endtime)
        well_subset = wells['Data'].loc[mask]
        
        # Creates subset of well data between caps. Determines number of unique
        # months, by determining the number unique (Year/Month) codings. Drop 
        # any column in subset that has less than MinExTotal non empty cells
        well_subset = well_subset.drop(well_subset.columns[well_subset.apply(
            lambda col: len(np.unique((col.dropna().index).strftime('%Y/%m')).tolist()) < Min_Obs_Months)], axis=1)

        # Remove outliers that larger than 3 standard deviations
        for col in well_subset:
            well_subset[col] = well_subset[col][np.abs(well_subset[col] - 
                    well_subset[col].mean()) <= (outlier*well_subset[col].std())]

        # Creates subset of well data between caps. Determines number of unique
        # months, by determining the number unique (Year/Month) codings. Drop 
        # any column in subset that has less than MinExTotal non empty cells
        well_subset = well_subset.drop(well_subset.columns[well_subset.apply(
            lambda col: len(np.unique((col.dropna().index).strftime('%Y/%m')).tolist()) < Min_Obs_Months)], axis=1)

        # Creating filtered Well Data DataFrame. This DataFrame will include 
        # all values for the wells selected in the pervious function- including
        # values outside of the caps range
        wells_dict['Data'] = wells['Data'][well_subset.columns]
        
        
        # Unpack dataframe well coorindates: Lat Long
        location_df = wells['Location']
        # Create new Locations dataframe with only wells that exist in the data.
        location_df = location_df.loc[wells_dict['Data'].columns]
        wells_dict['Location'] = location_df   
        
        
        # Unpack dataframe of well location centroid
        centroid = wells['Centroid']
        # Create new Centroid Data Frame
        centroid.loc['Latitude'][0]  = location_df['Latitude'].min()  + ((location_df['Latitude'].max()  - location_df['Latitude'].min())/2)
        centroid.loc['Longitude'][0] = location_df['Longitude'].min() + ((location_df['Longitude'].max() - location_df['Longitude'].min())/2)
        wells_dict['Centroid'] = centroid
        
        return wells_dict
        
    def interp_well(self, wells_df, gap_size = '365 days', pad = 90, spacing = '1MS'):
        # gaps bigger than this are set to nan
        # padding on either side of a measured point not set to nan, even if in a gap
        # spacing to interpolate to 1st day of the month (MS or month start)
        well_interp_df = pd.DataFrame()
        # create a time index to interpolate over - cover entire range
        interp_index: DatetimeIndex = pd.date_range(start=min(wells_df.index), freq=spacing, end=max(wells_df.index))
        # loop over each well, interpolate data using pchip
        for well in wells_df:
            temp_df = wells_df[well].dropna()  # available data for a well drops NaN
            x_index = temp_df.index.astype('int')  # dates for available data
            x_diff = temp_df.index.to_series().diff()  # data gap sizes
            fit2 = interpolate.pchip(x_index, temp_df)  # pchip fit to data
            ynew = fit2(interp_index.astype('int'))  # interpolated data on full range
            interp_df = pd.DataFrame(ynew, index=interp_index, columns=[well])
    
            # replace data in gaps of > 1 year with nans
            gaps = np.where(x_diff > gap_size)  # list of indexes where gaps are large
            
            # Can Probably optimize and remove this code
            # if padding plus start is larger than the gap size currently 1 year
            # Non nans will be replaced
            for g in gaps[0]:
                start = x_diff.index[g - 1] + dt.timedelta(days=pad)
                end = x_diff.index[g] - dt.timedelta(days=pad)
                interp_df[start:end] = np.nan
            beg_meas_date = x_diff.index[0]      # date of 1st measured point
            end_meas_date = temp_df.index[-1]    # date of last measured point
            mask1 = (interp_df.index < beg_meas_date)  # Check values before measured date
            interp_df[mask1] = np.nan  # blank out data before 1st measured point
            mask2 = (interp_df.index >= end_meas_date) # data from last measured point to end
            interp_df[mask2] = np.nan # blank data from last measured point
            
            # add the interp_df data to the full data frame
            well_interp_df = pd.concat([well_interp_df, interp_df], join="outer", axis=1, sort=False)
            
        # return a pd data frame with interpolated wells - gaps with nans
        return well_interp_df
        

    '''###################################
                Plotting Functions
    '''###################################
    def well_plot(self, combined_df, well_df, plot_wells):
        if plot_wells:
            # plot some of the wells just to look
            well_names = well_df.columns
            somewells = well_names[0:5] # first 5 wells
            well = well_names[-1]  # last well in the file

            combined_df.plot(y=somewells, style='.')            
            plt.title('Observations of first 5 Wells')
            plt.savefig(self.figures_root + '/' + '_First_Five_Wells_Observations')            
            plt.show()
        
            plt.figure(4)
            well_df.plot(y=somewells, style='-')
            plt.title('Example wells')
            plt.savefig(self.figures_root + '/' + '_First_Five_Wells_Interpolation')           
            plt.show()
            
            plt.figure(5)
            plt.plot(combined_df[well], '*')
            plt.plot(well_df[well])
            plt.title('Interpolation with Gaps and padding for well: ' + str(well))
            plt.savefig(self.figures_root + '/' + '_Interpolation_with_Gaps')       
            plt.show()
            
            plt.figure(6)
            plt.plot(well_df)
            plt.plot(combined_df[well_names], '-.')
            plt.title('Wells in Aquifer with Gaps and Padding')
            plt.savefig(self.figures_root + '/' + '_Interpolation_with_Gaps_Aquifer')            
            plt.show()
     