# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 23:23:13 2021

@author: saulg
"""
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pickle
import os

class imputation():
    def __init__(self, data_root ='./Datasets', figures_root = './Figures Imputed'):
        # Data Path
        if os.path.isdir(data_root) is False:
            os.makedirs(data_root)
            print('The dataset folder with data is empty')
        self.data_root = data_root
        
        # Fquifer Root is the location to save figures.
        if os.path.isdir(figures_root) is False:
            os.makedirs(figures_root)
        self.figures_root = figures_root
        return
       
    def read_pickle(self, pickle_file, pickle_root):
        wellfile = pickle_root + pickle_file + '.pickle'
        with open(wellfile, 'rb') as handle:
            wells = pickle.load(handle)
        return wells
    
    def Save_Pickle(self, Data, name:str, path:str, protocol:int = 3):
        with open(path + '/' + name + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=protocol)
    
    def Data_Split(self, Data, well_name_temp, Shuffle=False):
        if Shuffle:
            # The frac keyword argument specifies the fraction of rows
            # to return in the random sample
             Data = Data.sample(frac=1)
        Y = Data[well_name_temp].to_frame()
        X = Data.drop(well_name_temp, axis=1)
        return Y, X
    
    def Data_Join(self, pd1, pd2, method='outer', axis=1):
        return pd.concat([pd1, pd2], join='outer', axis=1)
    
    def test_range_split(self, df, min_points = 1, Cut_left= None, 
                         gap_year=None, Random=True, seed_start = 42, max_tries = 15, max_gap = 5):
        self.gap_year = gap_year
        self.Cut_left = Cut_left
        attempt = 0
        Y_Test = []
        Y_Train = ['dummy variable']
        while attempt <= max_tries and len(Y_Test) <= min_points and len(Y_Train) >= min_points:
            if self.gap_year == None: gap_year = np.random.randint(1, max_gap+1)
            if self.Cut_left == None: Cut_left = None
            Cut_left, Cut_right = self.Define_Gap(df, Cut_left, 
                gap_year = gap_year, 
                seed = seed_start + attempt, 
                Random = Random)
            Y_Test = df[(df.index >= Cut_left) & 
                    (df.index < Cut_right)].dropna() 
            Y_Train = df[(df.index < Cut_left) |
                    (df.index >= Cut_right)].dropna() 
            attempt += 1
        if attempt == max_tries: return print('At least one of the wells has no points in the specified range')
        self.Cut_left, self.Cut_right= [Cut_left, Cut_right]
        return Y_Train, Y_Test
    
    def Define_Gap(self, df, Cut_left, gap_year, seed, Random=True):
        np.random.seed(seed)
        df = df.dropna()
        if Random and Cut_left == None:
            date_min = df.index[0].year
            if date_min < 1952: date_min = 1952
            _, Cut_Range_End = [date_min, df.index[-1].year]
            df_index = df[(df.index < dt.datetime(Cut_Range_End, 1, 1))]
            Cut_left = str(df.index[np.random.randint(0, len(df_index))])[0:4]
            Cut_right = int(Cut_left) + gap_year
        else:
            Cut_left = int(Cut_left)
            if Cut_left: Cut_right = Cut_left + gap_year    
            else: Cut_left, Cut_right = [1995, 2001] 
        return str(Cut_left), str(Cut_right)
    
    def metrics(self, Metrics, n_wells):
        metrics_result = Metrics.sum(axis = 0)
        normalized_metrics = metrics_result/n_wells
        with open(self.figures_root + 'Error_Aquifer.txt', "w") as outfile:
            print('Raw Model Metrics:  \n' + str(metrics_result), file=outfile)
            print('\n Normalized Model Metrics Per Well:  \n' + str(normalized_metrics), file=outfile)
        np.savetxt(self.figures_root + 'Error_Wells.txt', Metrics.values, fmt='%d')
        return metrics_result, normalized_metrics

    def Model_Training_Metrics_plot(self, Data, name):
        pd.DataFrame(Data).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(-0.5, 5)
        plt.savefig(self.figures_root + '/' + name + '_Training_History')
        plt.show()
    
    def Q_Q_plot(self, Prediction, Observation, name): #y_test_hat, y_test):
        #Plotting Prediction Correlation
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        plt.scatter(Prediction, Observation)
        plt.ylabel('Observation')
        plt.xlabel('Prediction')
        plt.legend(['Prediction', 'Observation'])
        plt.title('Prediction Correlation: ' + name)
        limit_low = 0
        limit_high = 1
        cor_line_x = np.linspace(limit_low, limit_high, 9)
        cor_line_y = cor_line_x
        plt.xlim(limit_low, limit_high)
        plt.ylim(limit_low, limit_high)
        plt.plot(cor_line_x, cor_line_y, color='r')
        ax1.set_aspect('equal', adjustable='box')
        plt.savefig(self.figures_root + '/' + name + '_01_Q_Q')
        plt.show() 

    def observeation_vs_prediction_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name):
        plt.figure(figsize=(12, 8))
        plt.plot(Prediction_X, Prediction_Y, "darkblue")
        plt.plot(Observation_X, Observation_Y, label= 'Observations', color='darkorange')
        plt.ylabel('Groundwater Surface Elevation')
        plt.xlabel('Date')
        plt.legend(['Prediction', 'Observation'])
        plt.title('Observation Vs Prediction: ' + name)
        plt.savefig(self.figures_root  + '/' + name + '_02_Observation')
        plt.show()

    def observeation_vs_imputation_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name):
        plt.figure(figsize=(12, 8))
        plt.plot(Prediction_X, Prediction_Y, "darkblue")
        plt.plot(Observation_X, Observation_Y, label= 'Observations', color='darkorange')
        plt.ylabel('Groundwater Surface Elevation')
        plt.xlabel('Date')
        plt.legend(['Imputed Values', 'Smoothed Observations'])
        plt.title('Observation Vs Imputation: ' + name)
        plt.savefig(self.figures_root  + '/' + name + '_03_Imputation')
        plt.show()

    def raw_observation_vs_prediction(self, Prediction, Raw, name, Aquifer):
        plt.figure(figsize=(12, 8))
        plt.plot(Prediction.index, Prediction, 'darkblue', label='Prediction', linewidth=1.0)
        plt.scatter(Raw.index, Raw, color='darkorange', marker = '*', s=10, label= 'Observations')
        plt.title(Aquifer + ': ' + 'Well: ' + name + ' Raw vs Prediction')
        plt.legend(fontsize = 'x-small')
        plt.tight_layout(True)
        plt.savefig(self.figures_root  + '/' + name + '_04_Prediction_vs_Raw')
        plt.show()
    
    def raw_observation_vs_imputation(self, Prediction, Raw, name, Aquifer):
        plt.figure(figsize=(12, 8))
        plt.plot(Prediction.index, Prediction, 'darkblue', label='Model', linewidth=1.0)
        plt.scatter(Raw.index, Raw, color='darkorange', marker = '*', s=10, label= 'Observations')
        plt.title(Aquifer + ': ' + 'Well: ' + name + ' Raw vs Model')
        plt.legend(fontsize = 'x-small')
        plt.tight_layout(True)
        plt.savefig(self.figures_root  + '/' + name + '_05_Imputation_vs_Raw')
        plt.show()

    def observeation_vs_prediction_scatter_plot(self, Prediction, Y_train, Y_val, name):
        plt.figure(figsize=(12, 8))
        plt.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        plt.scatter(Y_train.index, Y_train, color='darkorange', marker='*', s=10)
        plt.scatter(Y_val.index, Y_val, color='lightgreen', s=10)  
        plt.ylabel('Groundwater Surface Elevation')
        plt.xlabel('Date')
        plt.legend(['Prediction', 'Training Data', 'Validation Data'])
        plt.title('Observation Vs Prediction: ' + name)
        plt.savefig(self.figures_root  + '/' + name + '_06_Observation')
        plt.show()
    
    def prediction_vs_test(self, Prediction, Well_set_original, y_test, name):
        plt.figure(figsize=(12, 8))
        plt.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        plt.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
        plt.scatter(y_test.index, y_test, color='lightgreen', s=10)
        plt.axvline(dt.datetime(int(self.Cut_left), 1, 1), linewidth=0.25)
        plt.axvline(dt.datetime(int(self.Cut_right), 1, 1), linewidth=0.25)
        plt.ylabel('Groundwater Surface Elevation')
        plt.xlabel('Date')
        plt.legend(['Prediction', 'Training Data', 'Test Data'])
        plt.title('Observation Vs Prediction: ' + name)
        plt.savefig(self.figures_root  + '/' + name + '_07_Test')
        plt.show()
        
    def Feature_Importance_box_plot(self, importance_df):
        #All Data       
        importance_df.boxplot(figsize=(20,10))
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Complete')
        plt.show()
    
        #Calc Mean and sort
        importance_mean_df = importance_df.mean()
        importance_mean_df = pd.DataFrame(importance_mean_df.sort_values(axis=0, ascending=False)).T
        importance_mean = importance_df.transpose().reindex(list(importance_mean_df.columns)).transpose()
        importance_mean.iloc[:,:10].boxplot(figsize=(5,5))
        plt.xticks(rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.title('Most Prevalent Features:')
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Uppper')
        plt.show()
        
        #Lower
        importance_mean.iloc[:,importance_mean.shape[1]-10:].boxplot(figsize=(5,5))
        plt.xticks(rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.title('Least Prevalent Features:')
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Lower')
        plt.show()

    def Aquifer_Plot(self, imputed_df):
        plt.figure(figsize=(12, 8))
        plt.plot(imputed_df)
        plt.title('Measured and Interpolated data for all wells')
        plt.savefig(self.figures_root  + '/' + 'Aquifer_Plot')
        plt.show()
           