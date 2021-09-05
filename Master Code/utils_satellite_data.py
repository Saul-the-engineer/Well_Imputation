# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:03:10 2021

@author: saulg
"""
import numpy as np
import os
import fiona
import shapely
import datetime as dt
import pandas as pd
import pickle
import grids

class utils_netCDF():
    def __init__(self, data_root ='./Datasets'):
        # Data Root is the location where data will be saved to. Saved to class
        # in order to reference later in other functions.
        if os.path.isdir(data_root) is False:
            os.makedirs(data_root)
        self.Data_root = data_root
        pass
    
    def netCDF_Grid_Creation(self, north_east_corner_lat, south_west_corner_lat, 
                             north_east_corner_lon, south_west_corner_lon, dx, dy,):
        self.dx = dx
        self.dy = dy
        self.nec_lat = north_east_corner_lat
        self.swc_lat = south_west_corner_lat
        self.nec_lon = north_east_corner_lon
        self.swc_lon = south_west_corner_lon
        
        lat_range = np.arange(self.nec_lat, self.swc_lat - self.dy, -self.dy).tolist()
        lon_range = np.arange(self.swc_lon, self.nec_lon + self.dx,  self.dx).tolist()      
        grid_locations = dict()
        cell_int = -1
        for i, lat in enumerate(lat_range):
            for j, lon in enumerate(lon_range):
                cell_int += 1
                grid_locations['Cell_' + str(cell_int)] = {'Latitude':lat, 'Longitude':lon}
        return grid_locations
    
    def Shape_Boundary(self, shape_file_path):
        self.shape_file_path = shape_file_path
        user_shape = fiona.open(shape_file_path)
        user_shape_boundary = user_shape.bounds
        return user_shape_boundary
    
    def Find_intercepting_cells(self, grid_locations, bound_location, padding=True, buffer=None):
        self.padding = padding
        if self.padding == True and buffer == None: self.buffer=(self.dx/2, self.dy/2)
        cells = dict.fromkeys(grid_locations.keys(),[])
        boundary = bound_location
        
        if self.padding: new_shape = shapely.geometry.Polygon((
            [boundary[0] - self.buffer[0], boundary[1] - self.buffer[1]], 
            [boundary[0] - self.buffer[0], boundary[3] + self.buffer[1]], 
            [boundary[2] + self.buffer[0], boundary[3] + self.buffer[1]], 
            [boundary[2] + self.buffer[0], boundary[1] - self.buffer[1]]))
        
        else: new_shape = shapely.geometry.Polygon((
            [boundary[0], boundary[1]], [boundary[0], boundary[3]], 
            [boundary[2], boundary[3]], [boundary[2], boundary[1]]))
        
        print('Starting Cell Selection...')
        for i, c in enumerate(grid_locations):
            point = shapely.geometry.Point(grid_locations[c]['Longitude'], 
                                           grid_locations[c]['Latitude']) 
            if new_shape.contains(point): cells[c]= True
            else: del cells[c]
        print('Cells Found.')
        return cells
    
    def Cell_Mask(self, intercepting_cells, grid_locations):
        Mask = dict()
        for i, key in enumerate(intercepting_cells):
            Mask[key] = grid_locations[key]
        return Mask
    
    def Date_Index_Creation(self, date_start:str, week_buffer=16):
        date_end = dt.datetime.now() - dt.timedelta(weeks=week_buffer)
        dates = pd.date_range(start = date_start, end = date_end, freq = 'MS')
        return dates
    
    def Save_Pickle(self, Data, name:str, path:str, protocol:int = 3):
        with open(path + '/' + name + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol = protocol)
            
    def read_pickle(self, file, root):
        file = root + file + '.pickle'
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
        return data      

class grids_netCDF():
    def __init__(self, File_String=True, Variable_String=True, dim_order = None):
        self.File_String = File_String
        self.Variable_String = Variable_String
        if dim_order == None: self.dim_order = ('time', 'lat', 'lon')
        else: self.dim_order = dim_order
    
    def _Data_List(self, data_path = None, data_root=None, name_text=None):
        if self.File_String:
            Data_Location = [data_path]
        else:
            Data_Location = []
            print('Creating Data List...')              
            data_list = open(name_text).readlines()
            data_list = [i.replace('\n','') for i in data_list]
            for i, file in enumerate(data_list):
                file = file[::-1]
                file = file.split('/')[0]
                file = file[::-1]
                Data_Location.append(data_root + '/' + file)
        return Data_Location

    def _Variable_List(self, variable_name:str = None, variable_path:str = None):
        if self.Variable_String:
            Variables = [variable_name]
        else:
            variables = open(variable_path,'r').readlines()
            try:
                Variables = [i.replace('\n','') for i in variables]
            except:
                pass
            print('Variable List Made.') 
        return Variables
        
    def Parse_Data(self, Mask, dates, data_path = None, data_folder=None, file_list=None, variable_name=None, variables_list=None):
        Data = dict.fromkeys(Mask.keys(),[])
        df_temp = pd.DataFrame(index=list(Mask.keys()), columns =(['Longitude', 'Latitude']))
        for i, cell in enumerate(Mask):
            df_temp.loc[df_temp.index[i]] = Mask[cell]
        Data['Location'] = df_temp
 
        print('Loading netCDF location.')
        Data_Location = self._Data_List(data_path, data_folder, file_list)
        print('Creating Variables')
        Variables = self._Variable_List(variable_name, variables_list)
        print('Preparing to parse Data.')
        
        # Gets the all of the values for every variable
        Variable_Dictionary = dict()
        for j, var in enumerate(Variables):
            print('Working on ' + var + ' ' + str(j+1)+ '/' + str(len(Variables)))
            data_series = grids.TimeSeries(Data_Location, var, self.dim_order)
            coordinates_list = Data['Location']
            coordinates_list['Time'] = None
            coordinates_list = coordinates_list[['Time', 'Latitude', 'Longitude']]
            coordinates_list = [tuple(i) for i in coordinates_list[['Time', 'Latitude', 'Longitude']].values]
            df = data_series.multipoint(*coordinates_list, labels= Data['Location'].index.tolist())
            df.index = dates[0:len(df['datetime'])]
            df = df.drop(['datetime'], axis = 1)
            col_rename = df.columns.to_list()
            col_rename = [col_rename[i][col_rename[i].find('Cell'):] for i in range(len(col_rename))]
            df.columns = col_rename
            Variable_Dictionary[var] = df

        # Splits the variables and assigns to proper cell
        for i, cell in enumerate(Mask):
            print('Parsing ' + cell + ' ' + str(i+1)+ '/' + str(len(Mask)))
            feature_df = pd.DataFrame(index=dates)
            for j, var in enumerate(Variable_Dictionary):
                df = pd.DataFrame(Variable_Dictionary[var][cell].values, index = Variable_Dictionary[var].index, columns=[var])
                feature_df = pd.concat([feature_df, df], join="outer", axis=1)
            Data[cell] = feature_df
        return Data
    
    def Validate_Data(self, Mask, Data):
        print('Validating Data.')
        for i, cell in enumerate(Mask):
            try: 
                Validated_Data = pd.DataFrame(index=Data[cell].index)
                for j, var in enumerate(Data[cell].columns):
                    data_temp = Data[cell][var].to_frame()
                    data_temp = data_temp.astype(float)
                    data_temp = data_temp.dropna(axis=0, how='any')
                    if data_temp.empty:
                        del Data[cell]
                        Data['Location'].drop([cell],inplace=True, axis=0)
                        print('Removed Cell ' + cell)
                    else: Validated_Data=pd.concat([Validated_Data, data_temp], join="inner", axis=1)
                if not Validated_Data.empty: Data[cell] = Validated_Data
            except: 
                print('Error with cell: ' + cell + '. Cell Not Validated.')
                pass
        print('Data Validated.')
        return Data
    
    
    
    
    
    
    
    
    