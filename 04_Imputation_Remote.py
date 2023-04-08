# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:32:26 2020

@author: saulg
"""
import pandas as pd
import numpy as np
import utils_04_machine_learning
import warnings
from scipy.spatial.distance import cdist #Required >1.8.1
from scipy.stats import pearsonr

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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
aquifer_name = 'Central Valley, CA'
data_root =    './Datasets/'
figures_root = './Figures Imputed'
val_split = 0.30
errors = []

# Model Setup
imp = utils_04_machine_learning.imputation(data_root, figures_root)

# Measured Well Data
Well_Data = imp.read_pickle('Well_Data_75', data_root)
PDSI_Data = imp.read_pickle('PDSI_Data', data_root)
GLDAS_Data = imp.read_pickle('GLDAS_Data', data_root)
Original_Obs_Points = Well_Data['Data']


# Getting Well Dates
Feature_Index = GLDAS_Data[list(GLDAS_Data.keys())[0]].index

# Importing Metrics and Creating Error DataFrame
columns = ['Train ME',     'Train RMSE',      'Train MAE',      'Train Points',      'Train r2',
           'Validation ME','Validation RMSE', 'Validation MAE', 'Validation Points', 'Validation r2',
           'Test ME',      'Test RMSE',       'Test MAE',       'Test Points',       'Test r2',
           'Comp R2']
Summary_Metrics = pd.DataFrame(columns = columns)

# Creating Empty Imputed DataFrame
Imputed_Data = pd.DataFrame(index=Feature_Index)
Model_Output = pd.DataFrame(index=Feature_Index)
Well_Data['Runs'] = {}

# Starting Learning Loop
loop = tqdm(total = len(Well_Data['Data'].columns), position = 0, leave = False)
for i, well in enumerate(Well_Data['Data']):
    try:
        # Get Well raw readings for single well
        y_raw = Original_Obs_Points[well].fillna(limit=2, method='ffill')
        
        # Get Well readings for single well
        y_well = pd.DataFrame(Well_Data['Data'][well], index = Feature_Index[:])
                
        # Add Dumbies
        table_dumbies = pd.get_dummies(Feature_Index.month_name())
        table_dumbies.index = Feature_Index
        table_dumbies['Months'] = (Feature_Index - Feature_Index[0]).astype(int)
        table_dumbies['Months'] = table_dumbies['Months']/table_dumbies['Months'][-1]
        
        # Create Prior Based on Well Trends
        windows = [18, 24, 36, 60]
        shift = int(max(windows)/2)
        pchip, x_int_index, pchip_int_index  = imp.interpolate(Feature_Index, y_raw, well, shift = shift)
        prior = imp.linear_extrap(x_int_index, pchip.dropna(), shift, reg_perc = [1.0, 0.5, 0.25, 0.10], 
                                  max_sd = 6, force_left = 'negative', force_right = 'negative')
        linear_extrap, extrap_df, extrap_md = prior
        imp.trend_plot(linear_extrap, extrap_df, extrap_md, y_raw, well)
        rw = imp.rolling_windows(linear_extrap, windows = windows)
        rw = rw[rw[rw.columns[-1]].notna()]
        table_rw = pd.DataFrame(rw, index=rw.index, columns = rw.columns)
        imp.rw_plot(y_raw, rw, well, save = True, extension = '.png', show=False)
            
        # PDSI Selection
        (well_x, well_y) = Well_Data['Location'].loc[well]
        well_loc = np.array(([well_x,well_y])).reshape((1,2))
        df_temp = PDSI_Data['Location'].dropna(axis=0).astype(float)
        pdsi_dist = pd.DataFrame(cdist(well_loc, df_temp, metric='euclidean'), columns=df_temp.index).T
        pdsi_key = pdsi_dist[0].idxmin()
        table_pdsi = PDSI_Data[pdsi_key]
                
        # GLDAS Selection
        df_temp = GLDAS_Data['Location'].dropna(axis=0).astype(float)
        gldas_dist = pd.DataFrame(cdist(well_loc, df_temp, metric='euclidean'), columns=df_temp.index).T
        gldas_key = gldas_dist[0].idxmin()
        table_gldas = GLDAS_Data[gldas_key]
        table_gldas = table_gldas[['Psurf_f_inst', 
                            'Wind_f_inst', 
                            'Qair_f_inst', 
                            'Qh_tavg', 
                            'Qsb_acc', 
                            'PotEvap_tavg', 
                            'Tair_f_inst', 
                            'Rainf_tavg',
                            'SoilMoi0_10cm_inst',
                            'SoilMoi10_40cm_inst',
                            'SoilMoi40_100cm_inst',
                            'SoilMoi100_200cm_inst',
                            'CanopInt_inst',
                            'SWE_inst',
                            'Lwnet_tavg',
                            'Swnet_tavg',
                            ]]
        
        # Calculate surface water
        sw_names = ['SoilMoi0_10cm_inst',
                    'SoilMoi10_40cm_inst',
                    'SoilMoi40_100cm_inst',
                    'SoilMoi100_200cm_inst',
                    'CanopInt_inst',
                    'SWE_inst']
        table_sw  = table_gldas[sw_names].sum(axis=1)
        table_sw.name = 'Surface Water'
        
        # Generate additional groundwater features
        gw_names = ['Qsb_acc',
                    'SWE_inst',
                    'Rainf_tavg']
        table_gwf  = table_gldas[gw_names]
        table_gwf['ln(QSB_acc)'] = np.log(table_gwf['Qsb_acc'])
        table_gwf['ln(RW 4 Rainf_tavg)'] = np.log(table_gwf['Rainf_tavg'].rolling(4, min_periods=1).sum())
        table_gwf['Sum Soil Moist'] = (table_sw - table_gldas['CanopInt_inst'] - table_gldas['SWE_inst']).rolling(3, min_periods=1).sum()
        table_gwf = table_gwf.drop(gw_names, axis=1)
        
        # Temporary merging gldas + PDSI before PCA
        Feature_Data = imp.Data_Join(table_pdsi, table_gldas).dropna()
        Feature_Data = imp.Data_Join(Feature_Data, table_rw).dropna()
        Feature_Data = imp.Data_Join(Feature_Data, table_sw).dropna()
        Feature_Data = imp.Data_Join(Feature_Data, table_gwf).dropna()
        Feature_Data = imp.Data_Join(Feature_Data, table_dumbies).dropna()
        
        # Joining Features to Well Data
        Well_set = y_well.join(Feature_Data, how='outer')
        Well_set = Well_set[Well_set[Well_set.columns[1]].notnull()]
        Well_set_clean = Well_set.dropna()
        
        # Split Data into Training/Validation sets
        Y, X = imp.Data_Split(Well_set_clean, well)
        
        # Run feature scaler on feature data, fs, ws for well scaler
        only_scale = windows + [table_sw.name] + table_gwf.columns.to_list()
        no_scale = table_dumbies.columns.to_list()
        fs = StandardScaler()
        ws = StandardScaler()
        
        # Create number of Folds
        folds = 5
        (Y_kfold, X_kfold) = (Y.to_numpy(), X.to_numpy())
        kfold = KFold(n_splits = folds, shuffle = False)
        temp_metrics = pd.DataFrame(columns = columns)
        j = 1
        n_epochs = []
        model_run_col = [*range(1, folds+2)]
        Model_Runs = pd.DataFrame(index=Feature_Index, columns=model_run_col)
        
        # Train K-folds grab error metrics average results
        for train_index, test_index in kfold.split(Y_kfold, X_kfold):
            x_train, x_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index,:]
            
            # Create validation and training sets
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=42)
            
            x_train, fs = imp.scaler_pipline(x_train, fs, no_scale, train=True)
        
            # Transform validation and test sets
            x_val = imp.scaler_pipline(x_val, fs, no_scale, train=False)
            x_test = imp.scaler_pipline(x_test, fs, no_scale, train=False)
            X_pred_temp = imp.scaler_pipline(Feature_Data, fs, no_scale, train=False)

            
            # Transform Y values
            y_train = pd.DataFrame(ws.fit_transform(y_train), index = y_train.index, columns = y_train.columns)
            y_val = pd.DataFrame(ws.transform(y_val), index = y_val.index, columns = y_val.columns)
            y_test = pd.DataFrame(ws.transform(y_test), index = y_test.index, columns = y_test.columns)
          
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
                                patience=5, 
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
            
            train_me    = (sum(y_train_hat.values - y_train.values) / train_points).item()
            train_rmse  = mean_squared_error(y_train.values, y_train_hat.values, squared=False)
            train_mae   = mean_absolute_error(y_train.values, y_train_hat.values)
    
            val_me      = (sum(y_val_hat.values - y_val.values) / val_points).item()
            val_rmse    = mean_squared_error(y_val.values, y_val_hat.values, squared=False)
            val_mae     = mean_absolute_error(y_val.values, y_val_hat.values)
            
            train_e      = [train_me, train_rmse, train_mae]
            val_e        = [val_me, val_rmse, val_mae]
            
            test_cols    = ['Test ME', 'Test RMSE', 'Test MAE']
            
            train_errors = np.array([train_e + val_e]).reshape((1,6))
            errors_col   = ['Train ME','Train RMSE', 'Train MAE',
                            'Validation ME','Validation RMSE', 'Validation MAE']
            df_metrics   = pd.DataFrame(train_errors, index=([str(j)]), columns = errors_col)
            
            df_metrics['Train Points']      = train_points
            df_metrics['Validation Points'] = val_points
            df_metrics['Train r2'], _       = pearsonr(y_train.values.flatten(), y_train_hat.values.flatten())
            df_metrics['Validation r2'], _  = pearsonr(y_val.values.flatten(), y_val_hat.values.flatten())
            temp_metrics = pd.concat(objs=[temp_metrics, df_metrics])
            
            # Model Prediction
            Prediction_temp = pd.DataFrame(
                            ws.inverse_transform(model.predict(X_pred_temp)), 
                            index=X_pred_temp.index, columns = ['Prediction'])
            
            # append prediction to model runs
            Model_Runs[j] = Prediction_temp.astype(float)
            
            # Test Sets and Plots
            try:
                y_test       = pd.DataFrame(ws.inverse_transform(y_test), index=y_test.index,
                                   columns = ['Y Test']).sort_index(axis=0, ascending=True)
                y_test_hat   = pd.DataFrame(ws.inverse_transform(model.predict(x_test)), index=y_test.index, 
                                   columns = ['Y Test Hat']).sort_index(axis=0, ascending=True)
                test_points  = len(y_test)
                test_me      = (sum(y_test_hat.values - y_test.values) / test_points).item()
                test_rmse    = mean_squared_error(y_test.values, y_test_hat.values, squared=False)
                test_mae     = mean_absolute_error(y_test.values, y_test_hat.values)
                            
                test_errors  = np.array([test_me, test_rmse, test_mae]).reshape((1,3))
                test_cols    = ['Test ME', 'Test RMSE', 'Test MAE']
                test_metrics = pd.DataFrame(test_errors,
                                            index = [str(j)], 
                                            columns = test_cols)
                test_metrics['Test Points'] = test_points
                test_metrics['Test r2'], _  = pearsonr(y_test.values.flatten(), y_test_hat.values.flatten())
                temp_metrics.loc[str(j), test_metrics.columns] = test_metrics.loc[str(j)]
                plot_kfolds = True
                imp.prediction_kfold(Prediction_temp['Prediction'], 
                                              y_well.drop(y_test.index, axis=0),  
                                              y_test, 
                                              str(well) +"_kfold_" + str(j), 
                                              temp_metrics.loc[str(j)], 
                                              error_on = True,
                                              plot = plot_kfolds)

            except: 
                temp_metrics.loc[str(j), ['Test ME','Test RMSE', 'Test MAE']] = np.NAN
                temp_metrics.loc[str(j), 'Test Points'] = 0 
                temp_metrics.loc[str(j),'Test r2'] = np.NAN
                imp.prediction_kfold(Prediction_temp['Prediction'], 
                                        y_well.drop(y_test.index, axis=0),
                                        y_test, 
                                        str(well) +"_kfold_" + str(j),
                                        plot = plot_kfolds)
            j += 1
            n_epochs.append(len(history.history['loss']))
            
        epochs = int(sum(n_epochs)/folds)
        
        # Reset feature scalers
        X, fs  = imp.scaler_pipline(X, fs, no_scale, train=True)
        X_pred = imp.scaler_pipline(Feature_Data, fs, no_scale, train=False)
        Y = pd.DataFrame(ws.fit_transform(Y), index = Y.index, columns = Y.columns)
        
        # Retrain Model with number of epochs
        history = model.fit(X, Y, epochs = epochs, verbose = 0)
        metrics_avg = pd.DataFrame(temp_metrics.mean(), columns=[well]).transpose()
        Summary_Metrics = pd.concat(objs=[Summary_Metrics, metrics_avg])
        
        # Model Prediction
        Prediction = pd.DataFrame(
                        ws.inverse_transform(model.predict(X_pred)), 
                        index=X_pred.index, columns = [well])
        Model_Runs[folds+1] = Prediction.astype(float)
        Well_Data['Runs'][well] = Model_Runs
        spread = pd.DataFrame(index = Prediction.index, columns = ['mean', 'std'])
        spread['mean'] = Model_Runs.mean(axis=1)
        spread['std'] = Model_Runs.std(axis=1)
        Comp_R2    = r2_score(
                        ws.inverse_transform(Y.values.reshape(-1,1)), 
                        ws.inverse_transform(model.predict(X)))
        Summary_Metrics.loc[well,'Comp R2'] = Comp_R2

        # Data Filling
        Gap_time_series = pd.DataFrame(Well_Data['Data'][well], index = Prediction.index)
        Filled_time_series = Gap_time_series[well].fillna(Prediction[well])
        if y_raw.dropna().index[-1] > Prediction.index[-1]:
            Filled_time_series = pd.concat([Filled_time_series, y_raw.dropna()], join='outer', axis=1)
            Filled_time_series = Filled_time_series.iloc[:,0]
            Filled_time_series = Filled_time_series.fillna(y_raw)
        Imputed_Data = pd.concat([Imputed_Data, Filled_time_series], join='outer', axis=1)
        Model_Output = pd.concat([Model_Output, Prediction], join='outer', axis=1)
                
        # Model Plots
        imp.prediction_vs_test_kfold(Prediction[well], y_well, str(well), Summary_Metrics.loc[well], error_on = True)
        imp.raw_observation_vs_prediction(Prediction[well], y_raw, str(well), aquifer_name, Summary_Metrics.loc[well], error_on = True, test=True)
        imp.raw_observation_vs_filled(Filled_time_series, y_raw, str(well) + '_Confidence Interval', aquifer_name, 
                spread, ci = 3, conf_interval = True, metrics = Summary_Metrics.loc[well], error_on = True, test=True) 
        imp.raw_observation_vs_filled(Filled_time_series, y_raw, str(well), aquifer_name, 
                metrics = Summary_Metrics.loc[well], error_on = True, test=True) 
        imp.residual_plot(Prediction.index, Prediction[well], y_well.index, y_well, well)
        imp.Model_Training_Metrics_plot(history.history, str(well))
        loop.update(1)
    except Exception as e:
        errors.append((i, e))
        imp.log_errors(errors, 'errors', data_root)

loop.close()
Well_Data['Data_Smooth'] = imp.smooth(Imputed_Data.loc[Prediction.index], Well_Data['Data'], window = 18)
Well_Data['Data'] = Imputed_Data.loc[Prediction.index]
Well_Data['Raw_Output'] = Model_Output.loc[Prediction.index]
Well_Data['Metrics'] = Summary_Metrics
Summary_Metrics.to_csv(data_root  + '/' + '06_Metrics.csv', index=True)
imp.Save_Pickle(Well_Data, 'Well_Data_Imputed', data_root)
imp.Save_Pickle(Imputed_Data, 'Well_Data_Imputed_Raw', data_root)
imp.Aquifer_Plot(Well_Data['Data'])