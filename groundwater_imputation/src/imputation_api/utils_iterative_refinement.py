import os
import math
import pandas as pd
import numpy as np
import sys
import utils
import utils_ml
import logging
import traceback

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


def iterative_refinement(
    aquifer_name: str = "aquifer",
    imputed_data_pickle: str = "source.pkl",
    output_file: str = "output.pkl",
    timeseries_name: str = "timeseries",
    locations_name: str = "Locations",
    imputed_name: str = "Data",
    weight_correlation: float = 0.70,
    num_features: int = 5,
    feature_threshold: float = 0.60,
    n_iterations: int = 2,
    validation_split: float = 0.3,
    folds: int = 5,
    batch_size: int = 32,
    regression_intercept_percentage: float = 0.10,
    regression_percentages: List[float] = [0.10, 0.25, 0.5, 1.0],
    windows: List[int] = [24],
):
    sys.path.append("..")

    # set weight of the distance correlation
    weight_distance = 1 - weight_correlation
    output_filename = output_file.split(".")[0]
    output_filetype = output_file.split(".")[1]

    # set project arguments
    project_args = utils_ml.ProjectSettings(
        aquifer_name=aquifer_name,
        iteration_current=1,
        iteration_target=n_iterations,
        artifacts_dir="artifacts",
    )

    # Configure logging
    logging.basicConfig(
        filename=os.path.join(
            project_args.dataset_outputs_dir, "iterative_refinement_error.log"
        ),
        level=logging.ERROR,
    )

    # Begin iterative refinement loop
    for iteration in range(n_iterations):
        logging.info(
            f"Starting iteration {project_args.iteration_current} of {n_iterations}"
        )

        # create metrics class
        metrics_class = utils_ml.Metrics(
            validation_split=validation_split,
            folds=folds,
        )

        # Load preprocessed data
        data_dict = utils.load_pickle(
            file_name=imputed_data_pickle,
            directory=project_args.dataset_outputs_dir,
        )
        data_dict_well = data_dict[timeseries_name]
        data_dict_locations = data_dict[locations_name]
        data_dict_features = data_dict[imputed_name]

        # create dictionary for model outputs
        data_dict["runs"] = {}

        # create list of well ids and imputation range for dataframe creation
        imputation_range = utils.make_interpolation_index()
        well_ids = list(map(str, data_dict_well.columns))

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
        correlation_df = pd.DataFrame(
            index=well_ids,
            columns=["pearson correlation", "distance correlation", "combined_score"],
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
                    timeseries=data_dict_well[well_id],
                    location=pd.DataFrame(data_dict_locations.loc[well_id]).T,
                    imputation_range=imputation_range,
                )

                # parse raw and data series with corresponding indecies
                y_raw = well_class.raw_series
                y_data = well_class.data

                # create one-hot encodings for observation month
                table_dumbies = utils_ml.get_dummy_variables(y_data.index)

                # create prior features, uses less windows for less leakage
                prior, prior_features = utils_ml.generate_priors(
                    y_data=y_data,
                    indecies=well_class,
                    regression_percentages=regression_percentages,
                    regression_intercept_percentage=regression_intercept_percentage,
                    windows=windows,
                )

                # drop existing well from pretrained data
                feature_data = data_dict_features.drop(columns=[well_id], axis=1)
                location_data = data_dict_locations.drop(well_id, axis=0)

                # calculate feature correlations to target
                feature_importance = utils_ml.calculate_feature_correlations(
                    source_series=y_data,
                    target_dataframe=feature_data,
                )

                distance_importance = utils_ml.calculate_distance_correlations(
                    source_series=well_class.location,
                    target_dataframe=location_data,
                )

                # calculate combined correlation
                correlation_importance = pd.concat(
                    [feature_importance, distance_importance],
                    axis=1,
                )

                correlation_importance["combined_score"] = (
                    correlation_importance["pearson correlation"] * weight_correlation
                    + correlation_importance["distance correlation"] * weight_distance
                )

                correlation_importance.sort_values(
                    by=["combined_score"], axis=0, ascending=False, inplace=True
                )

                # add correlation importance to correlation dataframe for tracking
                correlation_df.loc[well_id] = correlation_importance.head(
                    num_features
                ).mean()

                # select top features and determine if additional features are needed
                sample_score = correlation_importance.head(num_features)[
                    "combined_score"
                ].mean()
                if feature_threshold > sample_score:
                    # difference is using the floor so the number is multiplied by 10, floored, and divided by 10
                    difference = feature_threshold * 10 - math.floor(sample_score * 10)
                    additional_features = int(difference)
                    num_features_total = num_features + additional_features
                else:
                    num_features_total = num_features

                # collect feature dataframes into list: pdsi, gldas, prior features, sw, gwf, and dummies
                tables_merge = [
                    prior_features,
                    table_dumbies,
                    feature_data[correlation_importance.head(num_features_total).index],
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
                    hidden_nodes = 64
                    optimizer = Adam(learning_rate=0.001)
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
                            activation="relu",
                            use_bias=True,
                            kernel_initializer="glorot_uniform",
                        )
                    )
                    model.add(Dropout(rate=0.2))
                    model.add(Dense(1))
                    
                    model.compile(
                        optimizer=optimizer,
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
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=0,
                        callbacks=[early_stopping, adaptive_lr],
                    )

                    # Predictions
                    y_train_hat = model.predict(x_train)
                    y_val_hat = model.predict(x_val)
                    y_test_hat = model.predict(x_test)
                    y_pred_hat = model.predict(x_pred_temp)

                    # Score and Tracking Metrics
                    # Model Prediction
                    y_train = pd.DataFrame(
                        scaler_labels.inverse_transform(y_train),
                        index=y_train.index,
                        columns=["Y Train"],
                    ).sort_index(axis=0, ascending=True)

                    y_train_hat = pd.DataFrame(
                        scaler_labels.inverse_transform(y_train_hat),
                        index=x_train.index,
                        columns=["Y Train Hat"],
                    ).sort_index(axis=0, ascending=True)

                    y_val = pd.DataFrame(
                        scaler_labels.inverse_transform(y_val),
                        index=y_val.index,
                        columns=["Y Val"],
                    ).sort_index(axis=0, ascending=True)

                    y_val_hat = pd.DataFrame(
                        scaler_labels.inverse_transform(y_val_hat),
                        index=x_val.index,
                        columns=["Y Val Hat"],
                    ).sort_index(axis=0, ascending=True)

                    # Error Metrics
                    train_points, val_points = [len(y_train)], [len(y_val)]
                    train_me = (
                        sum(y_train_hat.values - y_train.values) / train_points
                    ).item()
                    val_me = (sum(y_val_hat.values - y_val.values) / val_points).item()

                    train_rmse = mean_squared_error(
                        y_train.values, y_train_hat.values, squared=False
                    )
                    val_rmse = mean_squared_error(
                        y_val.values, y_val_hat.values, squared=False
                    )
                    train_mae = mean_absolute_error(y_train.values, y_train_hat.values)
                    val_mae = mean_absolute_error(y_val.values, y_val_hat.values)

                    # compile errors
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
                        scaler_labels.inverse_transform(y_pred_hat),
                        index=x_pred_temp.index,
                        columns=[current_fold],
                    )

                    # append prediction to model runs
                    model_runs = model_runs.join(prediction_temp, how="outer")

                    # Test Sets and Plots
                    try:
                        # Model Prediction
                        y_test = pd.DataFrame(
                            scaler_labels.inverse_transform(y_test),
                            index=y_test.index,
                            columns=["Y Test"],
                        ).sort_index(axis=0, ascending=True)
                        y_test_hat = pd.DataFrame(
                            scaler_labels.inverse_transform(y_test_hat),
                            index=y_test.index,
                            columns=["Y Test Hat"],
                        ).sort_index(axis=0, ascending=True)

                        # Test Metrics
                        test_points = len(y_test)
                        test_me = (
                            sum(y_test_hat.values - y_test.values) / test_points
                        ).item()
                        test_rmse = mean_squared_error(
                            y_test.values, y_test_hat.values, squared=False
                        )
                        test_mae = mean_absolute_error(y_test.values, y_test_hat.values)

                        # concatenate test errors
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
                x_pred_full_hat = model.predict(x_pred)
                metrics_avg = pd.DataFrame(
                    temp_metrics.mean(), columns=[well_id]
                ).transpose()
                metrics_df = pd.concat(objs=[metrics_df, metrics_avg])

                # Model Prediction
                prediction = pd.DataFrame(
                    scaler_labels.inverse_transform(x_pred_full_hat).astype(float),
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
                data_dict["runs"][well_id] = model_runs

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
                    data_dict_well[well_id], index=prediction.index
                )
                filled_time_series = gap_time_series[well_id].fillna(
                    prediction[well_id]
                )
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
        data_dict["Data"] = imputation_df.loc[imputation_range]
        data_dict["Predictions"] = prediction_df.loc[imputation_range]
        data_dict["Locations"] = location_df
        data_dict["Metrics"] = metrics_df
        data_dict["Correlations"] = correlation_df

        utils.save_pickle(
            data_dict,
            file_name=f"{output_filename}_{project_args.iteration_current}.{output_filetype}",
            directory=project_args.dataset_outputs_dir,
        )
        logging.info(f"Finished imputation for {aquifer_name} aquifer")
        logging.info(
            "Added the following data to the data dictionary: Data, Predictions, Locations, Metrics, Correlations"
        )
        logging.info(
            f"Saved data dictionary to {project_args.dataset_outputs_dir}/{output_filename}_{project_args.iteration_current}.{output_filetype}"
        )

        # update project arguments
        project_args.iteration_current += 1
