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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import RootMeanSquaredError, mean_absolute_error

warnings.simplefilter(action='ignore')

#Data Settings
aquifer_name = 'Escalante-Beryl, UT'
data_root =    './Datasets/'
figures_root = './Figures Imputed'

###### Model Setup
imputation = utils_machine_learning.imputation(data_root, figures_root)

###### Measured Well Data
Original_Raw_Points = pd.read_hdf(data_root + '03_Original_Points.h5')
Well_Data = imputation.read_pickle('Well_Data', data_root)
PDSI_Data = imputation.read_pickle('PDSI_Data_EEMD', data_root)
GLDAS_Data = imputation.read_pickle('GLDAS_Data_Augmented', data_root)

###### Getting Well Dates
Feature_Index = GLDAS_Data[list(GLDAS_Data.keys())[0]].index

###### Importing Metrics and Creating Error DataFrame
Summary_Metrics = pd.DataFrame(columns=['Train MSE','Train RMSE', 'Train MAE',
                                        'Validation MSE','Validation RMSE', 'Validation MAE',
                                        'Test MSE','Test RMSE', 'Test MAE'])
###### Feature importance Tracker
Feature_Importance = pd.DataFrame()
###### Creating Empty Imputed DataFrame
Imputed_Data = pd.DataFrame(index=Feature_Index)


for i, well in enumerate(Well_Data['Data'].columns):
    try:
        Raw = Original_Raw_Points[well].fillna(limit=2, method='ffill')
        
        ###### Get Well readings for single well
        Well_set_original = pd.DataFrame(Well_Data['Data'][well], index = Feature_Index[:])
        well_scaler = MinMaxScaler()
        well_scaler.fit(Well_set_original)
        Well_set_temp = pd.DataFrame(well_scaler.transform(Well_set_original), index = Well_set_original.index, columns=([well]))
        
        
        ###### PDSI Selection
        (well_x, well_y) = Well_Data['Location'].loc[well]
        df_temp = pd.DataFrame(index=PDSI_Data['Location'].index, columns =(['Longitude', 'Latitude']))
        for j, cell in enumerate(PDSI_Data['Location'].index):
            df_temp.loc[cell] = PDSI_Data['Location'].loc[cell]
        pdsi_dist = pd.DataFrame(cdist(np.array(([well_x,well_y])).reshape((1,2)), df_temp, metric='euclidean'), columns=df_temp.index).T
        pdsi_key = pdsi_dist[0].idxmin()
        Feature_Data = PDSI_Data[pdsi_key]
        
        
        ###### GLDAS Selection
        df_temp = pd.DataFrame(index=PDSI_Data['Location'].index, columns =(['Longitude', 'Latitude']))
        for j, cell in enumerate(GLDAS_Data['Location'].index):
            df_temp.loc[cell] = GLDAS_Data['Location'].loc[cell]
        gldas_dist = pd.DataFrame(cdist(np.array(([well_x,well_y])).reshape((1,2)), df_temp, metric='euclidean'), columns=df_temp.index).T
        gldas_key = gldas_dist[0].idxmin()
        
        ###### Feature Join
        Feature_Data = imputation.Data_Join(Feature_Data, GLDAS_Data[gldas_key]).dropna()

        ###### Feature Scaling
        feature_scaler = StandardScaler() #StandardScaler() #MinMaxScaler()
        feature_scaler.fit(Feature_Data)
        Feature_Data = pd.DataFrame(feature_scaler.transform(Feature_Data), index = Feature_Data.index, columns=Feature_Data.columns)


        ###### Joining Features to Well Data
        Well_set = Well_set_temp.join(Feature_Data, how='outer')
        Well_set = Well_set[Well_set[Well_set.columns[1]].notnull()]
        Well_set_clean = Well_set.dropna()
        Y, X = imputation.Data_Split(Well_set_clean, well)
        (Y_kfold, X_kfold) = (Y.to_numpy(), X.to_numpy())
        kfold = KFold(n_splits = 5, shuffle = True)
        temp_metrics = pd.DataFrame(columns=[Summary_Metrics.columns])
        j = 1
        for train_index, test_index in kfold.split(Y_kfold, X_kfold):
            x_train, x_test = X_kfold[train_index], X_kfold[test_index]
            y_train, y_test = Y_kfold[train_index], Y_kfold[test_index]
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.30, random_state=42)

        ###### Model Initialization
            hidden_nodes = 300
            opt = Adam(learning_rate=0.001)
            model = Sequential()
            model.add(Dense(hidden_nodes, input_dim = X.shape[1], activation = 'relu', use_bias=True,
                kernel_initializer='glorot_uniform', kernel_regularizer= L2(l2=0.01))) #he_normal
            model.add(Dropout(rate=0.2))
            model.add(Dense(1))
            model.compile(optimizer = opt, loss='mse', metrics=[RootMeanSquaredError(), mean_absolute_error])
        
        ###### Hyper Paramter Adjustments
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0, restore_best_weights=True)
            history = model.fit(x_train, y_train, epochs=500, validation_data = (x_val, y_val), verbose= 3, callbacks=[early_stopping])
            train_mse = model.evaluate(x_train, y_train)
            validation_mse = model.evaluate(x_val, y_val)
            test_mse = model.evaluate(x_test, y_test)
            
            df_metrics = pd.DataFrame(np.array([train_mse + validation_mse + test_mse]).reshape((1,9)), 
                                      index=([str(j)]), columns=([Summary_Metrics.columns]))
            temp_metrics = pd.concat(objs=[temp_metrics, df_metrics])
            print(j)
            j += 1
        
        ###### Permutation Feature Importance
        print('Working on Permutation Feature Importance...')
        results = permutation_importance(model, x_test, y_test, n_repeats=5, random_state=42, scoring='neg_mean_squared_error')
        importance_df = pd.DataFrame(results.importances_mean, index = Feature_Data.columns, columns=([well])).sort_index(ascending=True).transpose()
        Feature_Importance = Feature_Importance.append(importance_df)
        
        ###### Score and Tracking Metrics
        Summary_Metrics.loc[cell] = temp_metrics.mean()
        y_test_hat = model.predict(x_test)
        
        ###### Model Prediction
        Prediction = pd.DataFrame(well_scaler.inverse_transform(model.predict(Feature_Data)), index=Feature_Data.index, columns = ['Prediction'])
        Gap_time_series = pd.DataFrame(Well_Data['Data'][well], index = Prediction.index)
        Filled_time_series = Gap_time_series[well].fillna(Prediction['Prediction'])
        Imputed_Data = pd.concat([Imputed_Data, Filled_time_series], join='inner', axis=1)

        ###### Model Plots
        imputation.Model_Training_Metrics_plot(history.history, str(well))
        imputation.Q_Q_plot(y_test_hat, y_test, str(well))
        imputation.observeation_vs_prediction_plot(Prediction.index, Prediction['Prediction'], Well_set_original.index, Well_set_original, str(well))
        imputation.observeation_vs_imputation_plot(Imputed_Data.index, Imputed_Data[well], Well_set_original.index, Well_set_original, str(well))
        imputation.raw_observation_vs_prediction(Prediction, Raw, str(well), aquifer_name)
        imputation.raw_observation_vs_imputation(Filled_time_series, Raw, str(well))
        print('Next Well')

    except Exception as e:
        print(e)


Summary_Metrics.to_hdf(data_root  + '/' + '06_Metrics.h5', key='metrics', mode='w')
imputation.Feature_Importance_box_plot(Feature_Importance)
Feature_Importance.to_pickle(data_root  + '/' + 'Feature_Importance.pickle') 
Well_Data_Imputed = Well_Data
Well_Data_Imputed['Data'] = Imputed_Data
imputation.Save_Pickle(Well_Data_Imputed, 'Well_Data_Imputed', data_root)
imputation.Aquifer_Plot(Well_Data_Imputed['Data']) 