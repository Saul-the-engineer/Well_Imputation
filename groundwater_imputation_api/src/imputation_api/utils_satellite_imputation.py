import sys
import utils
import utils_ml
import logging
import traceback
import os
import pandas as pd
import numpy as np


from tqdm import tqdm
from typing import List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from keras import callbacks
from keras.regularizers import L2


def satellite_imputation(
    aquifer_name: str = "aquifer",
    pdsi_pickle: str = "pdsi.pkl",
    gldas_pickle: str = "gldas.pkl",
    well_data_pickle: str = "well_data.pkl",
    output_file: str = "output.pkl",
    timeseries_name: str = "timeseries_processed",
    locations_name: str = "locations_processed",
    validation_split: float = 0.3,
    folds: int = 5,
    regression_intercept_percentage: float = 0.10,
    regression_percentages: List[float] = [0.10, 0.15, 0.25, 0.5, 1.0],
    windows: List[int] = [18, 24, 36, 60],
):
    sys.path.append("..")

    # set project arguments
    project_args = utils_ml.ProjectSettings(
        aquifer_name=aquifer_name,
        iteration_current=0,
        iteration_target=3,
        artifacts_dir=None,
    )

    # create metrics class
    metrics_class = utils_ml.Metrics(
        validation_split=validation_split,
        folds=folds,
    )

    # Configure logging
    logging.basicConfig(
        filename=os.path.join(project_args.this_dir, "error.log"), level=logging.ERROR
    )

    # Load preprocessed data
    data_dict_pdsi = utils.load_pickle(
        file_name=pdsi_pickle,
        directory=project_args.dataset_outputs_dir,
    )
    data_dict_gldas = utils.load_pickle(
        file_name=gldas_pickle,
        directory=project_args.dataset_outputs_dir,
    )
    data_dict_well = utils.load_pickle(
        file_name=well_data_pickle,
        directory=project_args.dataset_outputs_dir,
    )

    # create dictionary for model outputs
    data_dict_well["runs"] = {}

    # create list of well ids and imputation range for dataframe creation
    imputation_range = utils.make_interpolation_index()
    well_ids = list(map(str, data_dict_well[timeseries_name].columns))

    # create summary metrics dataframe and imputation dataframe
    imputation_df = pd.DataFrame(
        index=imputation_range,
    )
    location_df = pd.DataFrame(
        columns=["Longitude", "Latitude"],
    )
    metrics_df = pd.DataFrame(
        index=well_ids,
        columns=metrics_class.metrics,
    )
    prediction_df = pd.DataFrame(
        index=imputation_range,
    )

    # start imputation loop
    for i, well_id in tqdm(
        enumerate(well_ids),
        total=len(well_ids),
        position=0,
        leave=False,
    ):
        logging.info(f"Starting imputation for well: {well_id}")

        try:
            # setup well object to establish indecies
            well_class = utils_ml.WellIndecies(
                well_id=well_id,
                timeseries=data_dict_well[timeseries_name][well_id],
                location=pd.DataFrame(data_dict_well[locations_name].loc[well_id]).T,
                imputation_range=imputation_range,
            )

            # parse raw and data series with corresponding indecies
            y_raw = well_class.raw_series
            y_data = well_class.data

            # create one-hot encodings for observation month
            table_dumbies = utils_ml.get_dummy_variables(y_data.index)

            # create prior features
            prior, prior_features = utils_ml.generate_priors(
                y_data=y_data,
                indecies=well_class,
                regression_percentages=regression_percentages,
                regression_intercept_percentage=regression_intercept_percentage,
                windows=windows,
            )

            # pdsi data cell selection
            table_pdsi = utils_ml.remote_sensing_data_selection(
                data_dict=data_dict_pdsi,
                location_key=locations_name,
                location_query=well_class.location,
            )

            # gldas data cell selection
            table_gldas = utils_ml.remote_sensing_data_selection(
                data_dict=data_dict_gldas,
                location_key=locations_name,
                location_query=well_class.location,
            )

            # subset GLDAS data
            table_gldas = table_gldas[
                [
                    "Psurf_f_inst",
                    "Wind_f_inst",
                    "Qair_f_inst",
                    "Qh_tavg",
                    "Qsb_acc",
                    "PotEvap_tavg",
                    "Tair_f_inst",
                    "Rainf_tavg",
                    "SoilMoi0_10cm_inst",
                    "SoilMoi10_40cm_inst",
                    "SoilMoi40_100cm_inst",
                    "SoilMoi100_200cm_inst",
                    "CanopInt_inst",
                    "SWE_inst",
                    "Lwnet_tavg",
                    "Swnet_tavg",
                ]
            ]

            # calculate surface water feature from GLDAS
            sw_names = [
                "SoilMoi0_10cm_inst",
                "SoilMoi10_40cm_inst",
                "SoilMoi40_100cm_inst",
                "SoilMoi100_200cm_inst",
                "CanopInt_inst",
                "SWE_inst",
            ]
            table_sw = pd.DataFrame(
                table_gldas[sw_names].sum(axis=1).rename("Surface Water")
            )

            # generate additional groundwater features
            gw_names = ["Qsb_acc", "SWE_inst", "Rainf_tavg"]
            table_gwf = (
                table_gldas[gw_names]
                .assign(
                    **{
                        "ln(QSB_acc)": np.log(table_gldas["Qsb_acc"]),
                        "ln(RW 4 Rainf_tavg)": np.log(
                            table_gldas["Rainf_tavg"].rolling(4, min_periods=1).sum()
                        ),
                        "Sum Soil Moist": (
                            table_sw.squeeze()
                            - table_gldas["CanopInt_inst"]
                            - table_gldas["SWE_inst"]
                        )
                        .rolling(3, min_periods=1)
                        .sum(),
                    }
                )
                .drop(columns=gw_names)
            )

            # collect feature dataframes into list: pdsi, gldas, prior features, sw, gwf, and dummies
            tables_merge = [
                table_pdsi,
                table_gldas,
                prior_features,
                table_sw,
                table_gwf,
                table_dumbies,
            ]

            # iteratively merge predictors dataframes and drop rows with missing values
            merged_df = utils_ml.reduce(
                lambda left, right: pd.merge(
                    left=left,
                    right=right,
                    left_index=True,
                    right_index=True,
                    how="left",
                ),
                tables_merge,
            ).dropna()

            # merge features with labels
            well_set = y_data.to_frame(name=well_id).join(merged_df, how="outer")

            # match timeseries index by droping missing rows not in the label set
            well_set_clean = well_set.dropna()

            # split dataframe into features and labels
            y, x = utils_ml.dataframe_split(well_set_clean, well_id)

            # specify features that will be passed through without scaling
            features_to_pass = table_dumbies.columns.to_list()

            # create scaler object
            scaler_features = utils_ml.StandardScaler()
            scaler_labels = utils_ml.StandardScaler()

            # Create folds to split data into train, validation, and test sets
            n_epochs = []
            n_folds = folds
            current_fold = 1

            (y_kfold, x_kfold) = (y.to_numpy(), x.to_numpy())
            kfold = utils_ml.KFold(n_splits=n_folds, shuffle=False)
            temp_metrics = pd.DataFrame(columns=metrics_class.metrics)
            model_runs = pd.DataFrame(index=imputation_range)

            # train k-folds grab error metrics average results
            logging.info(f"Starting k-fold cross validation for well: {well_id}")
            for train_index, test_index in kfold.split(y_kfold, x_kfold):
                x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index, :]

                # Create validation and training sets
                x_train, x_val, y_train, y_val = utils_ml.train_test_split(
                    x_train, y_train, test_size=0.30, random_state=42
                )

                # Create scaler for features and labels
                x_train, scaler_features = utils_ml.scaler_pipeline(
                    x=x_train,
                    scaler_object=scaler_features,
                    features_to_pass=features_to_pass,
                    train=True,
                )

                x_val = utils_ml.scaler_pipeline(
                    x=x_val,
                    scaler_object=scaler_features,
                    features_to_pass=features_to_pass,
                    train=False,
                )
                x_test = utils_ml.scaler_pipeline(
                    x=x_test,
                    scaler_object=scaler_features,
                    features_to_pass=features_to_pass,
                    train=False,
                )

                # create predictors for full time series
                x_pred_temp = utils_ml.scaler_pipeline(
                    x=merged_df,
                    scaler_object=scaler_features,
                    features_to_pass=features_to_pass,
                    train=False,
                )

                # Transform Y values
                y_train = pd.DataFrame(
                    scaler_labels.fit_transform(y_train),
                    index=y_train.index,
                    columns=y_train.columns,
                )
                y_val = pd.DataFrame(
                    scaler_labels.transform(y_val),
                    index=y_val.index,
                    columns=y_val.columns,
                )
                y_test = pd.DataFrame(
                    scaler_labels.transform(y_test),
                    index=y_test.index,
                    columns=y_test.columns,
                )

                # Model Initialization
                hidden_nodes = 50
                opt = Adam(learning_rate=0.001)
                model = Sequential()
                model.add(
                    Dense(
                        hidden_nodes,
                        input_dim=x_train.shape[1],
                        activation="relu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        kernel_regularizer=L2(l2=0.1),
                    )
                )
                model.add(Dropout(rate=0.2))
                model.add(
                    Dense(
                        2 * hidden_nodes,
                        input_dim=x_train.shape[1],
                        activation="relu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                    )
                )
                model.add(Dropout(rate=0.2))
                model.add(Dense(1))
                model.compile(
                    optimizer=opt,
                    loss="mse",
                    metrics=[RootMeanSquaredError()],
                )

                # Hyper Paramter Adjustments
                early_stopping = callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=0.0,
                    restore_best_weights=True,
                )

                adaptive_lr = callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.1, min_lr=0
                )

                history = model.fit(
                    x_train,
                    y_train,
                    epochs=700,
                    validation_data=(x_val, y_val),
                    verbose=0,
                    callbacks=[early_stopping, adaptive_lr],
                )

                # Score and Tracking Metrics
                y_train = pd.DataFrame(
                    scaler_labels.inverse_transform(y_train),
                    index=y_train.index,
                    columns=["Y Train"],
                ).sort_index(axis=0, ascending=True)

                y_train_hat = pd.DataFrame(
                    scaler_labels.inverse_transform(model.predict(x_train)),
                    index=x_train.index,
                    columns=["Y Train Hat"],
                ).sort_index(axis=0, ascending=True)

                y_val = pd.DataFrame(
                    scaler_labels.inverse_transform(y_val),
                    index=y_val.index,
                    columns=["Y Val"],
                ).sort_index(axis=0, ascending=True)

                y_val_hat = pd.DataFrame(
                    scaler_labels.inverse_transform(model.predict(x_val)),
                    index=x_val.index,
                    columns=["Y Val Hat"],
                ).sort_index(axis=0, ascending=True)

                train_points, val_points = [len(y_train)], [len(y_val)]
                train_me = (
                    sum(y_train_hat.values - y_train.values) / train_points
                ).item()
                train_rmse = mean_squared_error(
                    y_train.values, y_train_hat.values, squared=False
                )
                train_mae = mean_absolute_error(y_train.values, y_train_hat.values)

                val_me = (sum(y_val_hat.values - y_val.values) / val_points).item()
                val_rmse = mean_squared_error(
                    y_val.values, y_val_hat.values, squared=False
                )
                val_mae = mean_absolute_error(y_val.values, y_val_hat.values)

                train_e = [train_me, train_rmse, train_mae]
                val_e = [val_me, val_rmse, val_mae]

                test_cols = ["Test ME", "Test RMSE", "Test MAE"]

                train_errors = np.array([train_e + val_e]).reshape((1, 6))
                errors_col = [
                    "Train ME",
                    "Train RMSE",
                    "Train MAE",
                    "Validation ME",
                    "Validation RMSE",
                    "Validation MAE",
                ]
                df_metrics = pd.DataFrame(
                    train_errors, index=([str(current_fold)]), columns=errors_col
                )

                df_metrics["Train Points"] = train_points
                df_metrics["Validation Points"] = val_points
                df_metrics["Train r2"], _ = pearsonr(
                    y_train.values.flatten(),
                    y_train_hat.values.flatten(),
                )
                df_metrics["Validation r2"], _ = pearsonr(
                    y_val.values.flatten(),
                    y_val_hat.values.flatten(),
                )
                temp_metrics = pd.concat(
                    objs=[
                        temp_metrics,
                        df_metrics,
                    ]
                )

                # Model Prediction
                prediction_temp = pd.DataFrame(
                    scaler_labels.inverse_transform(model.predict(x_pred_temp)),
                    index=x_pred_temp.index,
                    columns=[current_fold],
                )

                # append prediction to model runs
                model_runs = model_runs.join(prediction_temp, how="outer")

                # Test Sets and Plots
                try:
                    y_test = pd.DataFrame(
                        scaler_labels.inverse_transform(y_test),
                        index=y_test.index,
                        columns=["Y Test"],
                    ).sort_index(axis=0, ascending=True)
                    y_test_hat = pd.DataFrame(
                        scaler_labels.inverse_transform(model.predict(x_test)),
                        index=y_test.index,
                        columns=["Y Test Hat"],
                    ).sort_index(axis=0, ascending=True)
                    test_points = len(y_test)
                    test_me = (
                        sum(y_test_hat.values - y_test.values) / test_points
                    ).item()
                    test_rmse = mean_squared_error(
                        y_test.values, y_test_hat.values, squared=False
                    )
                    test_mae = mean_absolute_error(y_test.values, y_test_hat.values)

                    test_errors = np.array([test_me, test_rmse, test_mae]).reshape(
                        (1, 3)
                    )
                    test_cols = ["Test ME", "Test RMSE", "Test MAE"]
                    test_metrics = pd.DataFrame(
                        test_errors, index=[str(current_fold)], columns=test_cols
                    )
                    test_metrics["Test Points"] = test_points
                    test_metrics["Test r2"], _ = pearsonr(
                        y_test.values.flatten(), y_test_hat.values.flatten()
                    )
                    temp_metrics.loc[
                        str(current_fold), test_metrics.columns
                    ] = test_metrics.loc[str(current_fold)]

                except:
                    temp_metrics.loc[
                        str(current_fold), ["Test ME", "Test RMSE", "Test MAE"]
                    ] = np.NAN
                    temp_metrics.loc[str(current_fold), "Test Points"] = 0
                    temp_metrics.loc[str(current_fold), "Test r2"] = np.NAN

                current_fold += 1
                n_epochs.append(len(history.history["loss"]))

            # Log metrics
            logging.info(f"Finished k-fold cross validation for well: {well_id}")
            logging.info(f"Starting model training for well: {well_id}")
            epochs = int(sum(n_epochs) / n_folds)

            # Reset feature scalers
            x, scaler_features = utils_ml.scaler_pipeline(
                x,
                scaler_features,
                features_to_pass,
                train=True,
            )
            x_pred = utils_ml.scaler_pipeline(
                merged_df,
                scaler_features,
                features_to_pass,
                train=False,
            )
            y = pd.DataFrame(
                scaler_labels.fit_transform(y),
                index=y.index,
                columns=y.columns,
            )

            # Retrain Model with number of epochs
            history = model.fit(x, y, epochs=epochs, verbose=0)
            metrics_avg = pd.DataFrame(
                temp_metrics.mean(), columns=[well_id]
            ).transpose()
            metrics_df = pd.concat(objs=[metrics_df, metrics_avg])

            # Model Prediction
            prediction = pd.DataFrame(
                scaler_labels.inverse_transform(model.predict(x_pred)).astype(float),
                index=x_pred.index,
                columns=[well_id],
            )

            # append location to location dataframe
            location_df = pd.concat(
                [location_df, well_class.location], axis=0, ignore_index=False
            )

            # append prediction to prediction dataframe
            prediction_df = prediction_df.join(prediction, how="outer")

            # append prediction to model runs
            model_runs = model_runs.join(prediction, how="outer")
            data_dict_well["runs"][well_id] = model_runs

            # calculate spread and comp r2
            spread = pd.DataFrame(index=prediction.index, columns=["mean", "std"])
            spread["mean"] = model_runs.mean(axis=1)
            spread["std"] = model_runs.std(axis=1)
            comp_r2 = r2_score(
                scaler_labels.inverse_transform(y.values.reshape(-1, 1)),
                scaler_labels.inverse_transform(model.predict(x)),
            )
            metrics_df.loc[well_id, "Comp R2"] = comp_r2

            # Data Filling
            gap_time_series = pd.DataFrame(
                data_dict_well["timeseries"][well_id], index=prediction.index
            )
            filled_time_series = gap_time_series[well_id].fillna(prediction[well_id])
            if y_raw.dropna().index[-1] > prediction.index[-1]:
                filled_time_series = pd.concat(
                    [filled_time_series, y_raw.dropna()], join="outer", axis=1
                )
                filled_time_series = filled_time_series.iloc[:, 0]
                filled_time_series = filled_time_series.fillna(y_raw)
            imputation_df = imputation_df.join(filled_time_series, how="outer")
            logging.info(f"Finished model training for well: {well_id}")

        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            FileNotFoundError,
            PermissionError,
            ConnectionError,
            Exception,
        ) as error:
            logging.error(f"An exception ocurred on well: {well_id}\n{error}")
            logging.error(traceback.format_exc())
            continue

    # build output dictionary
    data_dict_well["Data"] = imputation_df.loc[imputation_range]
    data_dict_well["Predictions"] = prediction_df.loc[imputation_range]
    data_dict_well["Locations"] = location_df
    data_dict_well["Metrics"] = metrics_df

    utils.save_pickle(
        data_dict_well,
        output_file,
        project_args.dataset_outputs_dir,
    )
    logging.info(f"Finished imputation for {aquifer_name} aquifer")
    logging.info(
        "Added the following data to the data dictionary: Data, Predictions, Locations, Metrics"
    )
    logging.info(f"Saved data dictionary to {output_file}")
