# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:03:54 2021

@author: saulg
"""

import netCDF4 as nc
import numpy as np
import datetime as dt
import pandas as pd
import pickle
from tqdm import tqdm


def Data_List(root, name_text=None, data_root=None):
    Data_Location = []
    print('Creating Data List...')              
    data_list = open(root + '//' + name_text).readlines()
    data_list = [i.replace('\n','') for i in data_list]
    for i, file in enumerate(data_list):
        file = file[::-1]
        file = file.split('/')[0]
        file = file[::-1]
        Data_Location.append(data_root + '/' + file)
    return Data_Location

def Variable_List(variable_path:str = None):
    variables = open(variable_path,'r').readlines()
    Variables = [i.replace('\n','') for i in variables]
    print('Variable List Made.') 
    return Variables

# Saves generic pickle file based on file path and loads data.
def Save_Pickle(Data, save_root, name:str, protocol:int = 3):
    with open(save_root + '/' + name + '.pickle', 'wb') as handle:
        pickle.dump(Data, handle, protocol=protocol)

def Date_Index_Creation(date_start:str, week_buffer=16):
    date_end = dt.datetime.now() - dt.timedelta(weeks=week_buffer)
    dates = pd.date_range(start = date_start, end = date_end, freq = 'MS')
    return dates


root = r'C:\Users\saulg\OneDrive\Dissertation\Well Imputation\Master Code\Satellite Data Prep'
file_list = 'subset_GLDAS_NOAH025_M_2.0_20210628_013227.txt'
data_root = r'C:\Users\saulg\Desktop\Remote_Data\GLDAS'
file_list = Data_List(root, file_list, data_root)
variables_list = r'C:\Users\saulg\OneDrive\Dissertation\Well Imputation\Master Code\Satellite Data Prep\variables_list.txt'
variables_list = Variable_List(variables_list)
dates = Date_Index_Creation('1948-01-01')

dic = {v:[] for v in variables_list}
cell_name = ['Cell_' + str(i) for i in range(600*1440)]
print('Break Point')

for i, variable in enumerate(variables_list):
    for j, file_name in tqdm(enumerate(file_list)):
        ds = nc.Dataset(file_name)
        temp_var = ds.variables[variable]
        array = temp_var[:]
        array = np.squeeze(array, axis = 0).data
        array = np.flip(array, axis = 0)
        dic[variable].append(array)
        #print(str(j) + '/' + str(len(file_list)))
    temp_array1 = np.array(dic[variable])
    # Tests
    # test1 = temp_array1[:,300,900]
    # test2 = temp_array2[:, 300*1440 + 900]
    temp_df = pd.DataFrame(temp_array1.reshape(temp_array1.shape[0], 600*1440), index = dates[0:len(file_list)], columns = cell_name[:])
    #Save_Pickle(temp_df, r'C:\Users\saulg\Desktop\Remote_Data\Tabular GLDAS', str(variable))
    #temp_df.to_hdf(r'C:\Users\saulg\Desktop\New Folder' + '/' + str(variable), key=str(variable), mode='w')
    del temp_array1
    del temp_df
    del dic[variable]
    print(variable + ': ' + str(i+1) + '/' + str(len(variables_list)))
    break
