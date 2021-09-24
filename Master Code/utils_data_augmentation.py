# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 12:12:00 2021

@author: saulg
"""

import os
import pickle

class Data_Augmentation():
    # Establish where files are located. If location doesn't exist, it will 
    # create a new location.
    def __init__(self, data_root ='./Datasets', figures_root = False):
        # Data Root is the location where data will be saved to. Saved to class
        # in order to reference later in other functions.
        if os.path.isdir(data_root) is False:
            os.makedirs(data_root)
        self.Data_root = data_root
        
        # EEMD Root location is created in order to save figures.
        if figures_root and os.path.isdir(figures_root) is False:
            os.makedirs(figures_root)
        self.figures_root = figures_root

    # Opens generic pickle file based on file path and loads data.
    def read_pickle(self, file, root):
        file = root + file + '.pickle'
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    # Saves generic pickle file based on file path and loads data.
    def Save_Pickle(self, Data, name:str, protocol:int = 3):
        with open(self.Data_root + '/' + name + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=protocol)
    
    # Drops all missing values from row
    def DropNA(self, df):
        df.dropna(axis=0, inplace=True)
        return df
    
    # This loop adds the yearly, 3-year, 5-year, and 10-year rolling averages of each variable to the dataframe
    # rolling averages uses the preceding time to compute the last value, e.g., using the preceding 5 years of data to get todays
    # Causes the loss of X number of data points   
    def Rolling_Window(self, Feature_Data, Names, years=[1, 3, 5, 10]):
        for name in Names:
            for year in years:
                new = name + '_rw_yr_' + str(year).zfill(2)
                Feature_Data[new] = Feature_Data[name].rolling(year * 12).mean()
        return Feature_Data.dropna()

    # This loop adds straight offsets of original features in yearly increments
    def sat_offsets(self, Feature_Data, Names, offsets=[0.5,1,2,3]):
        for name in Names:
            for year in offsets:
                new = name + '_offset_' + str(year).zfill(3)
                Feature_Data[new] = Feature_Data[name].shift(int(year * 12))
        return Feature_Data
    
    # Deletes Features by name if desired
    # Must be input as list of strings where the strings are the name of the feautre
    def Delete_Data(self, Feature_Data, DeleteNames=[]):
        if len(DeleteNames) > 0:
            for i in range(len(DeleteNames)):
                del Feature_Data[str(DeleteNames[i])]          
        return Feature_Data
