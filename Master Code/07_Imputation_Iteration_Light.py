# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:32:26 2020

@author: saulg
"""

import pandas as pd
import numpy as np
import utils_machine_learning
import warnings
from scipy.stats import pearsonr
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.random import set_seed
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import RootMeanSquaredError

warnings.simplefilter(action='ignore')

np.random.seed(42)
set_seed(seed=42)

#Data Settings
aquifer_name = 'Escalante-Beryl, UT'
data_root =    './Datasets/'
val_split = 0.30

errors = []

iterations = 2
for iteration in range(0, iterations):
    figures_root = f'./Wells Imputed_iteration_{iteration+1}'
    
    # Model Setup
    imp = utils_machine_learning.imputation(data_root, figures_root)


    # Measured Well Data
    Original_Raw_Points = pd.read_hdf(data_root + '03_Original_Points.h5')
    Well_Data = imp.read_pickle('Well_Data', data_root)
    if iteration == 0: Well_Data_Pretrained = imp.read_pickle('Well_Data_Imputed', data_root)
    else: Well_Data_Pretrained = imp.read_pickle(f'Well_Data_Imputed_iteration_{iteration-1}', data_root)
    
    # Getting Well Dates
    Feature_Index = Well_Data_Pretrained['Data'].index
    
    # Importing Metrics and Creating Error DataFrame
    columns = ['Train ME',     'Train RMSE',      'Train MAE',      'Train Points',      'Train r2',
               'Validation ME','Validation RMSE', 'Validation MAE', 'Validation Points', 'Validation r2',
               'Test ME',      'Test RMSE',       'Test MAE',       'Test Points',       'Test r2',
               'Comp R2']
    Summary_Metrics = pd.DataFrame(columns = columns)
    
    # Feature importance Tracker
    Feature_Importance = pd.DataFrame()
    
    # Creating Empty Imputed DataFrame
    Imputed_Data = pd.DataFrame(index=Feature_Index)
    Feature_Correlation = pd.DataFrame(index=Well_Data['Data'].columns, columns = ['FI', 'WI'])
    
    loop = tqdm(total = len(Well_Data['Data'].columns), position = 0, leave = False)
    for i, well in enumerate(Well_Data['Data']):
        try:
            # Get Well raw readings for single well
            y_raw = Original_Raw_Points[well].fillna(limit=2, method='ffill')
            
            # Get Well readings for single well
            y_well = pd.DataFrame(Well_Data['Data'][well], index = Feature_Index[:])

            # Add Dumbies
            table_dumbies = pd.get_dummies(Feature_Index.month_name())
            table_dumbies.index = Feature_Index
            table_dumbies['Months'] = (Feature_Index - Feature_Index[0]).astype(int)
            table_dumbies['Months'] = table_dumbies['Months']/table_dumbies['Months'][-1]
            
            # Create Well Trend
            windows = [12,36,60]
            shift = int(max(windows)/2)
            weight = 1.5
            pchip, x_int_index, pchip_int_index  = imp.interpolate(Feature_Index, y_raw, well, shift = shift)
            slope, y_int, reg_line = imp.linear_regression(x_int_index, pchip.dropna())
            linear_extrap = imp.linear_extrapolation(x_int_index, pchip)
            trend, slopes_L, slopes_R  = imp.linear_correction(pchip, x_int_index, pchip_int_index, slope, linear_extrap, weight = weight)
            imp.trend_plot(y_raw, pchip, x_int_index, slope, y_int, slopes_L, slopes_R, weight, well)
            pchip = pchip['pchip'].fillna(trend['linear'])
            rw = imp.rolling_windows(pchip, windows = windows)
            rw = rw[rw[rw.columns[-1]].notna()]
            imp.rw_plot(rw, well)
            table_rw = pd.DataFrame(rw, index=rw.index, columns = rw.columns)

            # Load Pretrained Data Drop current Column
            Feature_Data = Well_Data_Pretrained['Data'].drop(well, axis=1)
            
            # Selecting Best Features
            fs = SelectKBest(score_func=r_regression, k=min(3, Feature_Data.shape[1]))
            fs_data = imp.Data_Join(Feature_Data, y_well).dropna()
            fs.fit(fs_data.drop(well, axis=1), fs_data[well])
            cols = fs.get_support(indices=True)
            Feature_Data = Feature_Data.iloc[:,cols]
            
            feature_r = fs.scores_[cols]
            feature_temp = pd.concat([y_well, Feature_Data], axis=1, join='outer')
            imp.feature_plot(feature_temp, Well_Data['Data'], well)
            Feature_Correlation = imp.feature_correlation(Feature_Correlation, feature_temp, Well_Data['Data'], feature_r)
            
            # Join Best features with Rolling Windows
            Feature_Data = imp.Data_Join(Feature_Data, table_rw).dropna()
            
            # Joining Features to Well Data
            Well_set = y_well.join(Feature_Data, how='outer')
            Well_set = Well_set[Well_set[Well_set.columns[1]].notnull()]
            Well_set_clean = Well_set.dropna()
            
            # Feature Split
            Y, X = imp.Data_Split(Well_set_clean, well)
            
            # Create Test Set
            y_subset, y_test, e = imp.test_range_split(
                df = Y, f_index = Feature_Index, name = well, min_points = 1, 
                cut_left = None, gap_year = 3, random = True, max_tries = 15, max_gap = 12)
            if e != None:
                errors.append((i, e))
                imp.log_errors(errors, 'errors', data_root)           
            x_test = imp.Data_Join(y_test, X, method='inner').drop(y_test.columns, axis=1)
            x_subset = X.drop(x_test.index, axis=0)
    
            # Create training and validation set
            x_train, x_val, y_train, y_val = train_test_split(x_subset, y_subset, test_size = val_split, random_state = 42)

            # Feature scaling and joining of dumbie variables for training variables
            fs = StandardScaler()
            x_train = pd.DataFrame(fs.fit_transform(x_train), index = x_train.index, columns = x_train.columns)
            x_train = imp.Data_Join(x_train, table_dumbies, method='inner')
                    
            # Feature scaling and joining of dumbie variables for validation, test, prediction
            x_val = pd.DataFrame(fs.transform(x_val), index = x_val.index, columns = x_val.columns)
            x_val = imp.Data_Join(x_val, table_dumbies, method='inner')
                        
            x_test = pd.DataFrame(fs.transform(x_test), index = x_test.index, columns = x_test.columns)
            x_test = imp.Data_Join(x_test, table_dumbies, method='inner')
                        
            X = pd.DataFrame(fs.transform(X), index = X.index, columns = X.columns)
            X = imp.Data_Join(X, table_dumbies, method='inner')
            
            X_pred = pd.DataFrame(fs.transform(Feature_Data), index = Feature_Data.index, columns = Feature_Data.columns)
            X_pred = imp.Data_Join(X_pred, table_dumbies, method='inner')

            # Transform Y values
            ws = StandardScaler()
            y_train = pd.DataFrame(ws.fit_transform(y_train), index = y_train.index, columns = y_train.columns)
            y_val = pd.DataFrame(ws.transform(y_val), index = y_val.index, columns = y_val.columns)
            y_test = pd.DataFrame(ws.transform(y_test), index = y_test.index, columns = y_test.columns)
            Y = pd.DataFrame(ws.transform(Y), index = Y.index, columns = Y.columns)

            # Model Initialization
            hidden_nodes = 50
            opt = Adam(learning_rate=0.001)
            model = Sequential()
            model.add(Dense(hidden_nodes, input_dim = x_train.shape[1], activation = None, use_bias=True,
                kernel_initializer='glorot_uniform', kernel_regularizer= L2(l2=0.01)))
            model.add(Dropout(rate=0.2))
            model.add(Dense(2*hidden_nodes, input_dim = x_train.shape[1], activation = None, use_bias=True,
                kernel_initializer='glorot_uniform'))
            model.add(Dropout(rate=0.2))
            model.add(Dense(1))
            model.compile(optimizer = opt, loss='mse', metrics=[RootMeanSquaredError()])
            
            # Hyper Paramter Adjustments
            early_stopping = callbacks.EarlyStopping(
                                monitor='val_loss', 
                                patience=7, 
                                min_delta=0.0, 
                                restore_best_weights=True)
            adaptive_lr    = callbacks.ReduceLROnPlateau(
                                monitor='val_loss', 
                                factor=0.1, 
                                min_lr=0)
            history        = model.fit(
                                x_train, 
                                y_train, 
                                epochs=700, 
                                validation_data = (x_val, y_val), 
                                verbose= 0, 
                                callbacks=[early_stopping, adaptive_lr])
            
            # Score and Tracking Metrics
            y_train     = pd.DataFrame(ws.inverse_transform(y_train), index=y_train.index, 
                                      columns = ['Y Train']).sort_index(axis=0, ascending=True)
            y_train_hat = pd.DataFrame(ws.inverse_transform(model.predict(x_train)), index=x_train.index, 
                                      columns = ['Y Train Hat']).sort_index(axis=0, ascending=True)
            y_val       = pd.DataFrame(ws.inverse_transform(y_val), index=y_val.index,
                                       columns = ['Y Val']).sort_index(axis=0, ascending=True)
            y_val_hat   = pd.DataFrame(ws.inverse_transform(model.predict(x_val)), index=x_val.index,
                                       columns = ['Y Val Hat']).sort_index(axis=0, ascending=True)
            train_points, val_points = [len(y_train)], [len(y_val)]
            
            train_me    = (sum(y_train.values - y_train_hat.values) / train_points).item()
            train_rmse  = mean_squared_error(y_train.values, y_train_hat.values, squared=False)
            train_mae   = mean_absolute_error(y_train.values, y_train_hat.values)
    
            val_me      = (sum(y_val.values - y_val_hat.values) / val_points).item()
            val_rmse    = mean_squared_error(y_val.values, y_val_hat.values, squared=False)
            val_mae     = mean_absolute_error(y_val.values, y_val_hat.values)
            
            train_e      = [train_me, train_rmse, train_mae]
            val_e        = [val_me, val_rmse, val_mae]
            train_errors = np.array([train_e + val_e]).reshape((1,6))
            errors_col   = ['Train ME','Train RMSE', 'Train MAE',
                            'Validation ME','Validation RMSE', 'Validation MAE']
            df_metrics   = pd.DataFrame(train_errors, index=([str(well)]), columns = errors_col)
            
            df_metrics['Train Points']      = train_points
            df_metrics['Validation Points'] = val_points
            df_metrics['Train r2'], _       = pearsonr(y_train.values.flatten(), y_train_hat.values.flatten())
            df_metrics['Validation r2'], _  = pearsonr(y_val.values.flatten(), y_val_hat.values.flatten())
            Summary_Metrics = pd.concat(objs=[Summary_Metrics, df_metrics])
    
            # Model Prediction
            Prediction = pd.DataFrame(
                            ws.inverse_transform(model.predict(X_pred)), 
                            index=X_pred.index, columns = ['Prediction'])
            
            Comp_R2    = r2_score(
                            ws.inverse_transform(Y.values.reshape(-1,1)), 
                            ws.inverse_transform(model.predict(X)))
            Summary_Metrics.loc[well,'Comp R2'] = Comp_R2
            
            # Test Sets and Plots
            try:
                y_test       = pd.DataFrame(ws.inverse_transform(y_test), index=y_test.index,
                                   columns = ['Y Test']).sort_index(axis=0, ascending=True)
                y_test_hat   = pd.DataFrame(ws.inverse_transform(model.predict(x_test)), index=y_test.index, 
                                   columns = ['Y Test Hat']).sort_index(axis=0, ascending=True)
                test_points  = len(y_test)
                test_me      = (sum(y_test.values - y_test_hat.values) / test_points).item()
                test_rmse    = mean_squared_error(y_test.values, y_test_hat.values, squared=False)
                test_mae     = mean_absolute_error(y_test.values, y_test_hat.values)
                            
                test_errors  = np.array([test_me, test_rmse, test_mae]).reshape((1,3))
                test_cols    = ['Test ME', 'Test RMSE', 'Test MAE']
                test_metrics = pd.DataFrame(test_errors,
                                            index = [str(well)], 
                                            columns = test_cols)
                test_metrics['Test Points'] = test_points
                test_metrics['Test r2'], _  = pearsonr(y_test.values.flatten(), y_test_hat.values.flatten())
                Summary_Metrics.loc[well,test_metrics.columns] = test_metrics.loc[well]
    
                imp.prediction_vs_test(Prediction['Prediction'], 
                                              y_well.drop(y_test.index, axis=0),  
                                              y_test, 
                                              str(well), 
                                              Summary_Metrics.loc[well], 
                                              error_on = True)
            except: 
                Summary_Metrics.loc[well,['Test ME','Test RMSE', 'Test MAE']] = np.NAN
                Summary_Metrics.loc[well, 'Test Points'] = 0 
                Summary_Metrics.loc[well,'Test r2'] = np.NAN
                imp.prediction_vs_test(Prediction['Prediction'], 
                                        y_well.drop(y_test.index, axis=0),
                                        y_test, 
                                        str(well))

            # Data Filling
            Gap_time_series = pd.DataFrame(Well_Data['Data'][well], index = Prediction.index)
            Filled_time_series = Gap_time_series[well].fillna(Prediction['Prediction'])
            if y_raw.dropna().index[-1] > Prediction.index[-1]:
                Filled_time_series = pd.concat([Filled_time_series, y_raw.dropna()], join='outer', axis=1)
                Filled_time_series = Filled_time_series.iloc[:,0]
                Filled_time_series = Filled_time_series.fillna(y_raw)
            Imputed_Data = pd.concat([Imputed_Data, Filled_time_series], join='outer', axis=1)
    
            # Model Plots
            imp.Model_Training_Metrics_plot(history.history, str(well))
            imp.Q_Q_plot(y_val_hat, y_val, str(well), limit_low = y_val.min()[0], limit_high = y_val.max()[0])
            imp.observeation_vs_prediction_plot(Prediction.index, Prediction['Prediction'], y_well.index, y_well, str(well), Summary_Metrics.loc[well], error_on = True)
            imp.residual_plot(Prediction.index, Prediction['Prediction'], y_well.index, y_well, str(well))
            imp.observeation_vs_imputation_plot(Imputed_Data.loc[Prediction.index].index, Imputed_Data.loc[Prediction.index][well], y_well.index, y_well, str(well))
            imp.raw_observation_vs_prediction(Prediction, y_raw, str(well), aquifer_name, Summary_Metrics.loc[well], error_on = True)
            imp.raw_observation_vs_imputation(Filled_time_series, y_raw, str(well), aquifer_name)
            imp.observeation_vs_prediction_scatter_plot(Prediction['Prediction'], y_train, y_val, str(well), Summary_Metrics.loc[well], error_on = True)
            loop.update(1)
        except Exception as e:
            errors.append((i, e))
            imp.log_errors(errors, 'errors', data_root)
    
    loop.close()
    Well_Data['Feature Correlation'] = Feature_Correlation   
    Well_Data['Data'] = Imputed_Data.loc[Prediction.index]
    Well_Data['Metrics'] = Summary_Metrics
    Well_Data['Original'] = Original_Raw_Points
    Summary_Metrics.to_csv(data_root  + '/' + f'06-{iteration}_Metrics.csv', index=True)
    imp.Save_Pickle(Well_Data, f'Well_Data_Imputed_iteration_{iteration}', data_root)
    imp.Save_Pickle(Imputed_Data, f'Well_Data_Imputed_Raw_{iteration}', data_root)
    imp.Aquifer_Plot(Well_Data['Data']) 