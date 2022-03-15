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
from scipy import interpolate
from sklearn.metrics import mean_squared_error
import psutil
import gc

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
            
    def log_errors(self, errors, name:str, path:str):
        if len(errors) == 0: pass
        else:
            with open(path + '/' + name + '.txt', 'w') as output:
                for row in errors:
                    output.write(str(row) + '\n')
    
    def Data_Split(self, Data, well_name_temp, Shuffle=False):
        if Shuffle:
            # The frac keyword argument specifies the fraction of rows
            # to return in the random sample
             Data = Data.sample(frac=1)
        Y = Data[well_name_temp].to_frame()
        X = Data.drop(well_name_temp, axis=1)
        return Y, X
    
    def Data_Join(self, pd1, pd2, method='outer', axis=1):
        return pd.concat([pd1, pd2], join = method, axis=1)
    
    def test_range_split(self, df, f_index, name, min_points = 1, 
                         cut_left= None, gap_year=None, random=True, 
                          seed_start = 42, max_tries = 15, max_gap = 5):
        self.gap_year = gap_year
        self.cut_left = cut_left
        attempt = 0
        Y_Test = []
        Y_Train = []
        while attempt <= max_tries and len(Y_Test) <= min_points and len(Y_Train) <= 3:
            if self.gap_year == None: 
                gap_year = np.random.randint(1, max_gap+1)
            if self.cut_left == None: 
                cut_left = None
            cut_left, cut_right = self.define_gap(df, 
                                                  f_index, 
                                                  cut_left, 
                                                  gap_year, 
                                                  seed = seed_start + attempt, 
                                                  random = random)
            cut_left_index = dt.datetime(cut_left, 1, 1)
            cut_right_index = dt.datetime(cut_right, 1, 1)
            Y_Test = df[(df.index >= cut_left_index) & 
                    (df.index < cut_right_index)].dropna() 
            Y_Train = df[(df.index < cut_left_index) |
                    (df.index >= cut_right_index)].dropna() 
            attempt += 1
        if attempt == max_tries:
            self.cut_left, self.cut_right= [f_index[0], f_index[0]]
            error = f'{name} could not be split into train/test sets'
            return df.dropna(), pd.DataFrame(), error
        self.cut_left, self.cut_right= [cut_left, cut_right]
        return Y_Train, Y_Test, None
    
    def define_gap(self, df, f_index, cut_left, gap_year, seed, random=True):
        np.random.seed(seed)
        df = df.dropna()
        if random == True and cut_left == None:
            if df.index[0].year < f_index[0].year: date_min = f_index[0].year
            else: date_min = df.index[0].year
            if df.index[-1].year > f_index[-1].year: cut_range_end = f_index[-1].year
            else: cut_range_end = df.index[-1].year
            
            df_index = df[(df.index >= dt.datetime(date_min, 1, 1)) &
                          (df.index <= dt.datetime(cut_range_end - gap_year, 1, 1))]
            cut_left = df.index[np.random.randint(0, len(df_index))].year
            cut_right = cut_left + gap_year
        elif random == False and cut_left != None:
            cut_right = cut_left + gap_year  
        else: cut_left, cut_right = [1995, 2001] 
        return cut_left, cut_right
    
    def interpolate(self, feature_index, y, name, shift = 60):
        self.shift_rw_max = shift
        index_check = y.dropna()
        startdt = pd.to_datetime(index_check.index[0])
        enddt   = pd.to_datetime(feature_index[0])
        lag = len(pd.date_range(start=startdt,end=enddt,freq='M'))
        if lag > 0: y = index_check
        int_y, x_int_index = self._interpolate_integers(lag, feature_index, y)
        
        p_class = interpolate.pchip(int_y.index, int_y[name], extrapolate=False)
        pchip_index = feature_index.shift(shift-1, freq='MS').union(feature_index.shift(-shift, freq='MS'))
        pchip_int   = np.arange(-shift, len(pchip_index)-shift, 1).astype(int)
        pchip_int   = pd.DataFrame(pchip_int, index = pchip_index, columns = ['x'])
        interp  = p_class(pchip_index[:].astype('int'))
        pchip   = pd.DataFrame(interp, index=pchip_index[:], columns=['pchip'])
        x_int_index = self.Data_Join(x_int_index, pchip)
        x_int_index = x_int_index.drop('pchip', axis=1)
        x_int_index = x_int_index['x'].fillna(pchip_int['x'])
        return pchip, x_int_index, pchip_int
    
    def _interpolate_integers(self, lag, feature_index, data):
        data = data.dropna()
        start = min(feature_index[0], data.index[0])
        end = max(feature_index[-1], data.index[-1])
        data_range = pd.date_range(start= start, end = end, freq='MS')
        int_values = np.arange(-lag, len(data_range)-lag, 1).astype(int)
        x_index    = pd.Series(int_values, index = data_range, name = 'x')
        int_y      = pd.concat([x_index, data], axis=1).dropna()
        return int_y, x_index
    
    def linear_regression(self, f_index, y):
        df = self.Data_Join(f_index, y).dropna()
        x = df['x'].values
        N = len(x)
        y = df['pchip'].values
        one = np.ones((N))
        if len(x) == len(y) and len(x) >= 3:
            A      = np.vstack([x,one]).T
            B0, B1 = np.linalg.lstsq(A, y)[0]
        reg_line = 'y = {}X + {}'.format(B0, B1)
        return (B0, B1, reg_line)
    
    def linear_extrapolation(self, f_index, y):
        df = self.Data_Join(f_index, y).dropna()
        lin_class  = interpolate.InterpolatedUnivariateSpline(df['x'], df['pchip'], k=1)
        lin_extrap = lin_class(f_index)
        linear  = pd.DataFrame(lin_extrap, index = f_index.index, columns = ['Linear NP'])
        return linear
    
    def linear_correction(self, data, f_index, p_index, reg_slope, linear, weight = 1.5):
        left_m  = linear.loc[linear.index <= data.dropna().index[0]]
        right_m = linear.loc[linear.index >= data.dropna().index[-1]]
        left_cor, slopes_L  = self._linear_correction(left_m, f_index, weight, reg_slope, direction = 'left')
        right_cor, slopes_R = self._linear_correction(right_m, f_index, weight, reg_slope, direction = 'right')
        linear.loc[left_m.index] = left_cor
        linear.loc[right_m.index] = right_cor
        linear.columns = ['linear']
        linear = linear.loc[p_index.index]
        return linear, slopes_L, slopes_R
       
    def _linear_correction(self, extrapolated, f_index, weight, reg_slope, direction = 'left'):
        df = self.Data_Join(f_index, extrapolated).dropna()
        slopes = pd.DataFrame(index=['Linear Ext', 'Linear Adj'], columns=['Slope', 'Intercept'])
        diff   = extrapolated.diff()
        mean   = diff.mean().values
        if mean == 0: mean = np.finfo(float).eps
        mag_lim = abs(weight * reg_slope)
        mag_cor = min(mag_lim, abs(mean))

        if direction == 'left': 
            mean = mean
            sign = (mean/ abs(mean))
            x_int = np.arange(-len(df['x'])+1, 0+1, 1).astype(int)
            df['x'] = x_int
            int_index  = df['x']
            intercept = extrapolated.loc[extrapolated.index[-1]].values

        elif direction == 'right': 
            sign = (mean/ abs(mean))
            x_int = np.arange(0, len(df['x']), 1).astype(int)
            df['x'] = x_int
            int_index  = df['x']            
            intercept = extrapolated.loc[extrapolated.index[0]].values
            
        slopes.iloc[0, 0] = mean    
        dif_adj = np.mean(np.hstack((mag_cor * sign, reg_slope)))
        correction = int_index * dif_adj + intercept
        correction.name = 'Correction'
        correction = correction.to_frame()
        
        slopes.iloc[1, 0] = dif_adj
        slopes.iloc[0, 1] = intercept
        slopes.iloc[1, 1] = intercept
        
        meta = {'Slopes': slopes, 'Index': int_index}
        return correction, meta
    
    def rolling_windows(self, df, windows = [3, 6]):
        rw_dict = dict()
        for i, months in enumerate(windows):
            key = str(months) + 'm_rw'
            rw = df.rolling(months, center=True).mean()
            rw_dict[key] = rw
        rw = pd.DataFrame.from_dict(rw_dict)
        return rw
        
    def scaler_pipline(self, x, scaler, pca, table_dumbies, os_names, pca_col_names, train=False):
        if train == True:
            os_names = x.columns[-len(os_names):]
            x_temp = scaler.fit_transform(x)
            x_temp_rw = x_temp[:,-len(os_names):]
            x_temp_rw = pd.DataFrame(x_temp_rw, index = x.index, columns = os_names)
            x_temp = x_temp[:,:-len(os_names)]
            x_temp = pca.fit_transform(x_temp)
            x_temp = pd.DataFrame(x_temp, index = x.index, columns = pca_col_names)
            X = self.Data_Join(x_temp, x_temp_rw)
            X = self.Data_Join(X, table_dumbies, method='inner')
            variance = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
            #pca_full = pd.DataFrame(pca.components_, columns = x.columns[:-len(os_names):], index=pca_col_names)
            #test = pca_full.abs().sum(axis=0)
            #pca_index = 1
            return [X, scaler, pca, variance]
            
        else:
            os_names = x.columns[-len(os_names):]
            x_temp = scaler.transform(x)
            x_temp_rw = x_temp[:,-len(os_names):]
            x_temp_rw = pd.DataFrame(x_temp_rw, index = x.index, columns = os_names)
            x_temp = x_temp[:,:-len(os_names)]
            x_temp = pca.transform(x_temp)
            x_temp = pd.DataFrame(x_temp, index = x.index, columns = pca_col_names)
            x_temp = self.Data_Join(x_temp, x_temp_rw)
            X = self.Data_Join(x_temp, table_dumbies, method='inner')
            return X
    
    def feature_correlation(self, df, Feature_Data, raw, r_score):
        data_clean = Feature_Data.dropna()
        fi = len(Feature_Data)
        wi = len(data_clean)
        data_zc = data_clean - data_clean.mean()
        index = Feature_Data.columns[0]
        df.loc[index, 'FI'] = fi
        df.loc[index, 'WI'] = wi
        names = Feature_Data.columns[1:].tolist()
        r_score2 = r_score.tolist()
        s_list = list(zip(names, r_score2))
        s_list.sort(reverse=True, key = lambda x: x[1])
        for i, obj in enumerate(s_list):
            name, r = obj
            if raw.index[-1] > Feature_Data.index[-1]: index_mask = Feature_Data.index
            else: index_mask = pd.date_range(Feature_Data.index[0], raw.index[-1], freq='MS')
            fip = raw[name].loc[index_mask].count()/fi
            wip = raw[name].loc[Feature_Data.dropna().index].count()/wi
            rmse  = mean_squared_error(data_zc[index].values, data_zc[name].values, squared=False)
            output = [name, r, fip, wip, rmse]
            col_names = [f'F{i}', f'F{i} r2', f'F{i} fip', f'F{i} wip', f'F{i} nRMSE']
            if col_names[0] not in df.columns:
                df[col_names] = np.nan
            df.loc[index, col_names] = output
        return df
    
    def metrics(self, Metrics, n_wells):
        metrics_result = Metrics.sum(axis = 0)
        normalized_metrics = metrics_result/n_wells
        with open(self.figures_root + 'Error_Aquifer.txt', "w") as outfile:
            print('Raw Model Metrics:  \n' + str(metrics_result), file=outfile)
            print('\n Normalized Model Metrics Per Well:  \n' + str(normalized_metrics), file=outfile)
        np.savetxt(self.figures_root + 'Error_Wells.txt', Metrics.values, fmt='%d')
        return metrics_result, normalized_metrics

    def Model_Training_Metrics_plot(self, Data, name, show=False):
        fig = plt.figure()
        for key in Data: plt.plot(Data[key])
        plt.grid(True)
        plt.gca().set_ylim(-0.5, 5)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(Data.keys())
        plt.savefig(self.figures_root + '/' + name + '_Training_History')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
    
    def trend_plot(self, Raw, pchip, x_int_index, slope, y_int, slopes_L, slopes_R, weight, name, extension = '.png', long_trend = False, show=False):
        pchip = pchip.dropna()
        y_reg = slope * x_int_index + y_int
        
        x_L   = slopes_L['Index']
        y_L_adj = x_L * slopes_L['Slopes'].iloc[1,0] + slopes_L['Slopes'].iloc[1,1]
        
        x_R   = slopes_R['Index']
        y_R_adj = x_R * slopes_R['Slopes'].iloc[1,0] + slopes_R['Slopes'].iloc[1,1]
        
        limits_L_u = x_L*weight*abs(slope) + slopes_L['Slopes'].iloc[1,1]
        limits_L_d = x_L*weight*-abs(slope) + slopes_L['Slopes'].iloc[1,1]

        limits_R_u = x_R*weight*abs(slope) + slopes_R['Slopes'].iloc[0,1]
        limits_R_d = x_R*weight*-abs(slope) + slopes_R['Slopes'].iloc[0,1]   
        
        fig = plt.figure()
        plt.plot(pchip.index, pchip, color = "black")
        plt.scatter(Raw.index, Raw, s= 3, c= 'red')
        plt.plot(x_int_index.index, y_reg)

        plt.fill_between(x_L.index, limits_L_u, limits_L_d)
        plt.fill_between(x_R.index, limits_R_u, limits_R_d)
        
        plt.plot(x_L.index, y_L_adj)
        plt.plot(x_R.index, y_R_adj)
        plt.title('Trends')
        plt.legend(['Pchip', 'Regression', 'Slope Adj L', 'Slope Adj R', 'Data', 'Slope Basis Left', 'Slope Basis Right'])
        plt.savefig(self.figures_root + '/' + name + '_00_Trend' + extension)
        if show: plt.show(fig)
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def rw_plot(self, rw, name, save = False, extension = '.png', show=False):
        fig = plt.figure()
        plt.plot(rw)
        if save: plt.savefig(self.figures_root + '/' + name + '_00_RW' + extension)
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
    
    def Q_Q_plot(self, Prediction, Observation, name, limit_low = 0, limit_high = 1, extension = '.png', show=False):
        #Plotting Prediction Correlation
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.scatter(Prediction, Observation)
        plt.ylabel('Observation')
        plt.xlabel('Prediction')
        plt.legend(['Prediction', 'Observation'])
        plt.title('Prediction Correlation: ' + name)
        cor_line_x = np.linspace(limit_low, limit_high, 9)
        cor_line_y = cor_line_x
        plt.xlim(limit_low, limit_high)
        plt.ylim(limit_low, limit_high)
        plt.plot(cor_line_x, cor_line_y, color='r')
        ax1.set_aspect('equal', adjustable='box')
        plt.savefig(self.figures_root + '/' + name + '_01_Q_Q' + extension)
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def observeation_vs_prediction_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction_X, Prediction_Y, "darkblue")
        # Potential bug # pandas version 1.4 pandas.errors.InvalidIndexError: (slice(None, None, None), None)
        ax.plot(Observation_X, Observation_Y, label= 'Observations', color='darkorange')
        ax.set_ylabel('Groundwater Surface Elevation')
        ax.legend(['Prediction', 'Observation'])
        ax.set_title('Observation Vs Prediction: ' + name)
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          fig.savefig(self.figures_root  + '/' + name + '_02_Observation', bbox_inches=extent.expanded(1.1, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_02_Observation')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
        
    def residual_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name, show=False):
        data = self.Data_Join(Prediction_Y, Observation_Y).dropna()
        data.columns = ['Prediction_Y', 'Observation_Y']
        fig = plt.figure(figsize=(12, 8))
        plt.plot(data.index, data['Prediction_Y'] - data['Observation_Y'], marker = 'o', linestyle='None', color = "black")
        plt.ylabel('Prediction Residual Error')
        plt.title('Residual Error: ' + name)
        plt.plot(data.index, np.zeros(shape = (len(data.index), 1)), color = 'royalblue', linewidth= 4.0)
        plt.savefig(self.figures_root  + '/' + name + '_03_Residual_Plot')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)  
        gc.collect()

    def observeation_vs_imputation_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name, show=False):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(Prediction_X, Prediction_Y, "darkblue")
        plt.plot(Observation_X, Observation_Y, label= 'Observations', color='darkorange')
        plt.ylabel('Groundwater Surface Elevation')
        plt.xlabel('Date')
        plt.legend(['Imputed Values', 'Smoothed Observations'])
        plt.title('Observation Vs Imputation: ' + name)
        plt.savefig(self.figures_root  + '/' + name + '_04_Imputation')
        if show: plt.show()
        else: 
            fig.clf()
            plt.close(fig)
        gc.collect()

    def raw_observation_vs_prediction(self, Prediction, Raw, name, Aquifer, metrics=None, error_on = False, test=False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, 'darkblue', label='Prediction', linewidth=1.0)
        ax.scatter(Raw.index, Raw, color='darkorange', marker = '*', s=10, label= 'Observations')
        ax.set_title(Aquifer + ': ' + 'Well: ' + name + ' Raw vs Prediction')
        ax.legend(fontsize = 'x-small')
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          if test:
               ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          fig.savefig(self.figures_root  + '/' + name + '_05_Prediction_vs_Raw', bbox_inches=extent.expanded(1.1, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_05_Prediction_vs_Raw')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
    
    def raw_observation_vs_imputation(self, Prediction, Raw, name, Aquifer, metrics=None, show=False):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(Prediction.index, Prediction, 'darkblue', label='Model', linewidth=1.0)
        plt.scatter(Raw.index, Raw, color='darkorange', marker = '*', s=10, label= 'Observations')
        plt.title(Aquifer + ': ' + 'Well: ' + name + ' Raw vs Model')
        plt.legend(fontsize = 'x-small')
        plt.savefig(self.figures_root  + '/' + name + '_06_Imputation_vs_Raw')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def observeation_vs_prediction_scatter_plot(self, Prediction, Y_train, Y_val, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        ax.scatter(Y_train.index, Y_train, color='darkorange', marker='*', s=10)
        ax.scatter(Y_val.index, Y_val, color='lightgreen', s=10)  
        ax.legend(['Prediction', 'Training Data', 'Validation Data'])
        ax.set_ylabel('Groundwater Surface Elevation')
        ax.set_title('Observation Vs Prediction: ' + name)
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          fig.savefig(self.figures_root  + '/' + name + '_07_Observation', bbox_inches=extent.expanded(1.2, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_07_Observation')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
    
    def prediction_vs_test(self, Prediction, Well_set_original, y_test, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        ax.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
        ax.scatter(y_test.index, y_test, color='lightgreen', s=10)
        ax.set_ylabel('Groundwater Surface Elevation')
        ax.legend(['Prediction', 'Training Data', 'Test Data'])
        ax.set_title('Observation Vs Prediction: ' + name)
        ax.axvline(dt.datetime(int(self.cut_left), 1, 1), linewidth=0.25)
        ax.axvline(dt.datetime(int(self.cut_right), 1, 1), linewidth=0.25)
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)  
          ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          fig.savefig(self.figures_root  + '/' + name + '_08_Test', bbox_inches=extent.expanded(1.2, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_08_Test')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
        
    def prediction_kfold(self, Prediction, Well_set_original, y_test, name, metrics=None, error_on = False, show=False, plot=False):
        if plot == True:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
            ax.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
            ax.scatter(y_test.index, y_test, color='lightgreen', s=10)
            ax.set_ylabel('Groundwater Surface Elevation')
            ax.legend(['Prediction', 'Training Data', 'Test Data'])
            ax.set_title('Observation Vs Prediction: ' + name)
            ax.axvline(y_test.index[0], linewidth=0.25)
            ax.axvline(y_test.index[-1], linewidth=0.25)
            if error_on:
              ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                      fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
              ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                      fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)  
              ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                      fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
              extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
              fig.savefig(self.figures_root  + '/' + name + '_08_kfold', bbox_inches=extent.expanded(1.2, 1.6))
            else: fig.savefig(self.figures_root  + '/' + name + '_08_kfold')
            if show: plt.show()
            else:
                fig.clf()
                plt.close(fig)
            gc.collect()
        
    def prediction_vs_test_kfold(self, Prediction, Well_set_original, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        ax.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
        ax.set_ylabel('Groundwater Surface Elevation')
        ax.legend(['Prediction', 'Training Data', 'Test Data'])
        ax.set_title('Observation Vs Prediction: ' + name)
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)  
          ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          fig.savefig(self.figures_root  + '/' + name + '_08_Test_kfold', bbox_inches=extent.expanded(1.2, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_08_Test_kfold')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)    
        gc.collect()
        
    def Feature_Importance_box_plot(self, importance_df, show=False):
        #All Data       
        importance_df.boxplot(figsize=(20,10))
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Complete')
        if show: plt.show()
        else: plt.close()
    
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
        if show: plt.show()
        else: plt.close()
        
        #Lower
        importance_mean.iloc[:,importance_mean.shape[1]-10:].boxplot(figsize=(5,5))
        plt.xticks(rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.title('Least Prevalent Features:')
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Lower')
        if show: plt.show()
        else: plt.close()
        gc.collect()

    def feature_plot(self, Feature_Data, raw, name, show=False):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(Feature_Data)
        for i, n in enumerate(Feature_Data.columns[1:]):
            ax.scatter(raw.index, raw[n], color='black', marker='*', s=3)
        legend = Feature_Data.columns.tolist() + ['Observed Measurements']
        ax.legend(legend, loc="lower left", bbox_to_anchor=(0.02, -0.35),
          ncol=2, fancybox=True, shadow=True)
        ax.set_ylabel('Groundwater Surface Elevation')
        ax.set_title('Well Feature Correlation: ' + name)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(self.figures_root  + '/' + name + '_09_Features', bbox_inches=extent.expanded(1.3, 1.8))
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def Aquifer_Plot(self, imputed_df, show=False):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(imputed_df)
        plt.title('Measured and Interpolated data for all wells')
        plt.savefig(self.figures_root  + '/' + 'Aquifer_Plot')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
           