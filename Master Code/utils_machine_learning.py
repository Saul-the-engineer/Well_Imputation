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
        
    def feature_correlation(self, df, Feature_Data, raw, r_score):
        fi = len(Feature_Data)
        wi = len(Feature_Data.dropna())
        index = Feature_Data.columns[0]
        df.loc[index, 'FI'] = fi
        df.loc[index, 'WI'] = wi
        names = Feature_Data.columns[1:].tolist()
        r_score2 = r_score.tolist()
        s_list = list(zip(names, r_score2))
        s_list.sort(reverse=True, key = lambda x: x[1])
        for i, obj in enumerate(s_list):
            name, r = obj
            fip = raw[name].loc[Feature_Data.index].count()/fi
            wip = raw[name].loc[Feature_Data.dropna().index].count()/wi
            output = [name, r, fip, wip]
            col_names = [f'F{i}', f'F{i} r2', f'F{i} fip', f'F{i} wip']
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

    def Model_Training_Metrics_plot(self, Data, name):
        pd.DataFrame(Data).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(-0.5, 5)
        plt.savefig(self.figures_root + '/' + name + '_Training_History')
        plt.show() 
    
    def trend_plot(self, Raw, pchip, x_int_index, slope, y_int, slopes_L, slopes_R, weight, name):
        pchip = pchip.dropna()
        y_reg = slope * x_int_index + y_int
        
        x_L   = slopes_L['Index']
        y_L_ex  = x_L * slopes_L['Slopes'].iloc[0,0] + slopes_L['Slopes'].iloc[1,1]
        y_L_adj = x_L * slopes_L['Slopes'].iloc[1,0] + slopes_L['Slopes'].iloc[1,1]
        
        x_R   = slopes_R['Index']
        y_R_ex  = x_R * slopes_R['Slopes'].iloc[0,0] + slopes_R['Slopes'].iloc[0,1]
        y_R_adj = x_R * slopes_R['Slopes'].iloc[1,0] + slopes_R['Slopes'].iloc[1,1]
        
        limits_L_u = x_L*weight*abs(slope) + slopes_L['Slopes'].iloc[1,1]
        limits_L_d = x_L*weight*-abs(slope) + slopes_L['Slopes'].iloc[1,1]

        limits_R_u = x_R*weight*abs(slope) + slopes_R['Slopes'].iloc[0,1]
        limits_R_d = x_R*weight*-abs(slope) + slopes_R['Slopes'].iloc[0,1]   
        
        
        plt.plot(pchip.index, pchip, color = "black")
        plt.scatter(Raw.index, Raw, s= 3, c= 'red')
        plt.plot(x_int_index.index, y_reg)

        plt.fill_between(x_L.index, limits_L_u, limits_L_d)
        plt.fill_between(x_R.index, limits_R_u, limits_R_d)
        
        plt.plot(x_L.index, y_L_adj)
        plt.plot(x_R.index, y_R_adj)
        plt.title('Trends')
        plt.legend(['Pchip', 'Regression', 'Slope Adj L', 'Slope Adj R', 'Data', 'Slope Basis Left', 'Slope Basis Right'])
        plt.savefig(self.figures_root + '/' + name + '_00_Trend')
        plt.show()
        
        '''
        plt.plot(pchip.index, pchip, color = "black")
        plt.scatter(Raw.index, Raw, s= 3, c= 'red')
        plt.plot(x_int_index.index, y_reg)
        plt.fill_between(x_L.index, limits_L_u, limits_L_d)
        plt.fill_between(x_R.index, limits_R_u, limits_R_d)
        plt.plot(x_L.index, y_L_adj)
        plt.plot(x_R.index, y_R_adj)
        plt.plot(x_L.index, y_L_ex)
        plt.plot(x_R.index, y_R_ex)
        plt.title('Trends')
        plt.legend(['Pchip', 'Regression', 'Slope Adj L', 'Slope Adj R', 'Slope Ex L', 'Slope Ex R', 'Data', 'Slope Basis Left', 'Slope Basis Right'])
        plt.show()
        '''
    def rw_plot(self, rw, name):
        rw.plot()
        #plt.savefig(self.figures_root + '/' + name + '_00_RW')
        plt.show()
    
    def Q_Q_plot(self, Prediction, Observation, name, limit_low = 0, limit_high = 1):
        #Plotting Prediction Correlation
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
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
        plt.savefig(self.figures_root + '/' + name + '_01_Q_Q')
        plt.show() 

    def observeation_vs_prediction_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name, metrics=None, error_on = False):
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
        plt.show()
        
    def residual_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name):
        data = self.Data_Join(Prediction_Y, Observation_Y).dropna()
        data.columns = ['Prediction_Y', 'Observation_Y']
        plt.figure(figsize=(12, 8))
        plt.plot(data.index, data['Prediction_Y'] - data['Observation_Y'], marker = 'o', linestyle='None', color = "black")
        plt.ylabel('Prediction Residual Error')
        plt.title('Residual Error: ' + name)
        plt.plot(data.index, np.zeros(shape = (len(data.index), 1)), color = 'royalblue', linewidth= 4.0)
        plt.savefig(self.figures_root  + '/' + name + '_03_Residual_Plot')
        plt.show()

    def observeation_vs_imputation_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name):
        plt.figure(figsize=(12, 8))
        plt.plot(Prediction_X, Prediction_Y, "darkblue")
        plt.plot(Observation_X, Observation_Y, label= 'Observations', color='darkorange')
        plt.ylabel('Groundwater Surface Elevation')
        plt.xlabel('Date')
        plt.legend(['Imputed Values', 'Smoothed Observations'])
        plt.title('Observation Vs Imputation: ' + name)
        plt.savefig(self.figures_root  + '/' + name + '_04_Imputation')
        plt.show()

    def raw_observation_vs_prediction(self, Prediction, Raw, name, Aquifer, metrics=None, error_on = False):
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
          fig.savefig(self.figures_root  + '/' + name + '_05_Prediction_vs_Raw', bbox_inches=extent.expanded(1.1, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_05_Prediction_vs_Raw')
        plt.show()
    
    def raw_observation_vs_imputation(self, Prediction, Raw, name, Aquifer):
        plt.figure(figsize=(12, 8))
        plt.plot(Prediction.index, Prediction, 'darkblue', label='Model', linewidth=1.0)
        plt.scatter(Raw.index, Raw, color='darkorange', marker = '*', s=10, label= 'Observations')
        plt.title(Aquifer + ': ' + 'Well: ' + name + ' Raw vs Model')
        plt.legend(fontsize = 'x-small')
        plt.savefig(self.figures_root  + '/' + name + '_06_Imputation_vs_Raw')
        plt.show()

    def observeation_vs_prediction_scatter_plot(self, Prediction, Y_train, Y_val, name, metrics=None, error_on = False):
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
          fig.savefig(self.figures_root  + '/' + name + '_07_Observation', bbox_inches=extent.expanded(1.1, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_07_Observation')
        plt.show()
    
    def prediction_vs_test(self, Prediction, Well_set_original, y_test, name, metrics=None, error_on = False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        ax.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
        ax.scatter(y_test.index, y_test, color='lightgreen', s=10)
        ax.set_ylabel('Groundwater Surface Elevation')
        ax.legend(['Prediction', 'Training Data', 'Test Data'])
        ax.set_title('Observation Vs Prediction: ' + name)
        ax.axvline(dt.datetime(int(self.Cut_left), 1, 1), linewidth=0.25)
        ax.axvline(dt.datetime(int(self.Cut_right), 1, 1), linewidth=0.25)
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)  
          ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          fig.savefig(self.figures_root  + '/' + name + '_08_Test', bbox_inches=extent.expanded(1.1, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_08_Test')
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

    def feature_plot(self, Feature_Data, raw, name):
        plt.plot(Feature_Data)
        for i, n in enumerate(Feature_Data.columns[1:]):
            plt.scatter(raw.index, raw[n], color='black', marker='*', s=3)
        legend = Feature_Data.columns.tolist() + ['Observed Measurements']
        plt.legend(legend)
        plt.ylabel('Groundwater Surface Elevation')
        plt.title('Well Feature Correlation: ' + name)
        plt.savefig(self.figures_root  + '/' + name + '_09_Features')
        plt.show()

    def Aquifer_Plot(self, imputed_df):
        plt.figure(figsize=(12, 8))
        plt.plot(imputed_df)
        plt.title('Measured and Interpolated data for all wells')
        plt.savefig(self.figures_root  + '/' + 'Aquifer_Plot')
        plt.show()
           