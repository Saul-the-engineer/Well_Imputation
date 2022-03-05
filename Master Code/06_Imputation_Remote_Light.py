# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:32:26 2020

@author: saulg
"""
import pandas as pd
import numpy as np
import utils_machine_learning
import warnings
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA

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
figures_root = './Figures Imputed'
val_split = 0.30

errors = []
# Model Setup
imp = utils_machine_learning.imputation(data_root, figures_root)

# Measured Well Data
Original_Raw_Points = pd.read_hdf(data_root + '03_Original_Points.h5')
Well_Data = imp.read_pickle('Well_Data', data_root)
PDSI_Data = imp.read_pickle('PDSI_Data_EEMD', data_root)
GLDAS_Data = imp.read_pickle('GLDAS_EEMD', data_root)


# Getting Well Dates
Feature_Index = GLDAS_Data[list(GLDAS_Data.keys())[0]].index

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

# Starting Learning Loop
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
        windows = [6, 12, 36, 60]
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
            
        # PDSI Selection
        (well_x, well_y) = Well_Data['Location'].loc[well]
        df_temp = PDSI_Data['Location'].dropna(axis=0)
        pdsi_dist = pd.DataFrame(cdist(np.array(([well_x,well_y])).reshape((1,2)), df_temp, metric='euclidean'), columns=df_temp.index).T
        pdsi_key = pdsi_dist[0].idxmin()
        table_pdsi = PDSI_Data[pdsi_key]
                
        # GLDAS Selection
        df_temp = GLDAS_Data['Location'].dropna(axis=0)
        gldas_dist = pd.DataFrame(cdist(np.array(([well_x,well_y])).reshape((1,2)), df_temp, metric='euclidean'), columns=df_temp.index).T
        gldas_key = gldas_dist[0].idxmin()
        table_gldas = GLDAS_Data[gldas_key]
        
        # Calculate surface water
        sw_names = ['SoilMoi0_10cm_inst',
                    'SoilMoi10_40cm_inst',
                    'SoilMoi40_100cm_inst',
                    'SoilMoi100_200cm_inst',
                    'CanopInt_inst',
                    'SWE_inst']
        table_sw  = table_gldas[sw_names].sum(axis=1)
        table_sw.name = 'Surface Water'
        
        # Temporary merging gldas + PDSI before PCA
        Feature_Data = imp.Data_Join(table_pdsi, table_gldas).dropna()
        Feature_Data = imp.Data_Join(Feature_Data, table_rw).dropna()
        Feature_Data = imp.Data_Join(Feature_Data, table_sw).dropna()
        
        # Joining Features to Well Data
        Well_set = y_well.join(Feature_Data, how='outer')
        Well_set = Well_set[Well_set[Well_set.columns[1]].notnull()]
        Well_set_clean = Well_set.dropna()
        
        # Split Data into Training/Validation sets
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
        
        # Run feature scaler and PCA on GLDAS and PDSI Training Data
        pca_components = 25
        pca_col_names = ['PCA '+ str(comp) for comp in range(pca_components)]
        only_scale = windows + [table_sw.name]
        pca = PCA(n_components=pca_components)
        fs = StandardScaler()
        
        temp_out = imp.scaler_pipline(x_train, fs, pca, table_dumbies, 
                                      only_scale, pca_col_names, train=True)
        x_train, fs, pca, variance = temp_out
    
        # Transform validation and test sets
        x_val = imp.scaler_pipline(x_val, fs, pca, table_dumbies, 
                                   only_scale, pca_col_names, train=False)
        x_test = imp.scaler_pipline(x_test, fs, pca, table_dumbies, 
                                   only_scale, pca_col_names, train=False)
        X = imp.scaler_pipline(X, fs, pca, table_dumbies, 
                                   only_scale, pca_col_names, train=False)
        X_pred = imp.scaler_pipline(Feature_Data, fs, pca, table_dumbies, 
                                   only_scale, pca_col_names, train=False)   
        
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
        model.add(Dense(hidden_nodes, input_dim = x_train.shape[1], activation = 'relu', use_bias=True,
            kernel_initializer='glorot_uniform', kernel_regularizer= L2(l2=0.1)))
        model.add(Dropout(rate=0.2))
        model.add(Dense(2*hidden_nodes, input_dim = x_train.shape[1], activation = 'relu', use_bias=True,
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
        
        
        test_cols    = ['Test ME', 'Test RMSE', 'Test MAE']
        
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
Well_Data['Data'] = Imputed_Data.loc[Prediction.index]
Well_Data['Metrics'] = Summary_Metrics
Well_Data['Original'] = Original_Raw_Points
Summary_Metrics.to_csv(data_root  + '/' + '06_Metrics.csv', index=True)
imp.Save_Pickle(Well_Data, 'Well_Data_Imputed', data_root)
imp.Save_Pickle(Imputed_Data, 'Well_Data_Imputed_Raw', data_root)
imp.Aquifer_Plot(Well_Data['Data'])