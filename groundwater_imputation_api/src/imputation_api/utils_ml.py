import utils
import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
from tqdm import tqdm
from functools import reduce
from typing import List, Dict, Tuple, Any, Union
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy.stats import theilslopes
from scipy.spatial.distance import cdist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from keras import callbacks
from keras.regularizers import L2


class ProjectSettings:
    def __init__(
        self,
        aquifer_name: str = "aquifer",
        iteration_current: int = 1,
        iteration_target: int = 3,
        artifacts_dir: Path = None,
    ):
        self.aquifer_name = aquifer_name
        self.iteration_current = iteration_current
        self.iteration_target = iteration_target
        if artifacts_dir is None:
            THIS_DIR: str = Path(__file__).parent.absolute()
            self.this_dir = THIS_DIR
            self.artifacts_dir = THIS_DIR / "artifacts"
        else:
            self.artifacts_dir = artifacts_dir
        self.aquifer_data_dir = self.artifacts_dir / "aquifer_data"
        self.aquifer_figure_dir = self.artifacts_dir / "aquifer_figures"
        self.aquifer_shape_dir = self.artifacts_dir / "aquifer_shapes"
        self.dataset_outputs_dir = self.artifacts_dir / "dataset_outputs"

        directories = [
            self.aquifer_data_dir,
            self.aquifer_shape_dir,
            self.aquifer_figure_dir,
            self.dataset_outputs_dir,
        ]
        for path in directories:
            os.makedirs(path, exist_ok=True)

        assert iteration_target >= 1
        assert iteration_current >= 1
        assert iteration_current <= iteration_target


class Metrics:
    def __init__(self, validation_split: float = 0.30, folds: int = 5):
        self.validation_split = validation_split
        self.folds = folds
        self.errors = []
        self.metrics = [
            "Train ME",
            "Train RMSE",
            "Train MAE",
            "Train Points",
            "Train r2",
            "Validation ME",
            "Validation RMSE",
            "Validation MAE",
            "Validation Points",
            "Validation r2",
            "Test ME",
            "Test RMSE",
            "Test MAE",
            "Test Points",
            "Test r2",
            "Comp R2",
        ]


class WellIndecies:
    def __init__(
        self,
        well_id: str,
        timeseries: pd.Series,
        location: pd.DataFrame,
        imputation_range: pd.DatetimeIndex,
    ) -> None:
        self.well_id = str(well_id)
        self.raw_series = timeseries.astype(float)
        self.location = location
        self.imputation_range = imputation_range
        try:
            self.data = timeseries.loc[imputation_range]
        except:
            temp_df = pd.concat(
                [timeseries, pd.DataFrame(index=imputation_range)], axis=1
            )
            self.data = temp_df.loc[imputation_range].squeeze()
        self.data_available = self.data.dropna()
        self.data_missing = self.data[self.data.isna()]
        self.interpolation_index = pd.date_range(
            start=self.data_available.index[0],
            end=self.data_available.index[-1],
            freq="MS",
        )
        self.get_center_index(self.data)
        self.get_left_index(imputation_range)
        self.get_right_index(imputation_range)

    def get_center_index(self, series: pd.Series):
        self.center_start = series.dropna().index[0]
        self.center_end = series.dropna().index[-1]
        self.center_index = series[
            (series.index >= self.center_start) & (series.index <= self.center_end)
        ].index

    def get_left_index(self, imputation_range: pd.DatetimeIndex):
        left_start = imputation_range[0]
        if left_start == self.center_start:
            self.left_index = None
        else:
            left_end = self.center_start
            self.left_index = imputation_range[
                (imputation_range >= left_start) & (imputation_range < left_end)
            ]

    def get_right_index(self, imputation_range: pd.DatetimeIndex):
        right_start = self.center_end
        if right_start == self.imputation_range[-1]:
            self.right_index = None
        else:
            right_end = self.imputation_range[-1]
            self.right_index = imputation_range[
                (imputation_range > right_start) & (imputation_range <= right_end)
            ]


class Imputation:
    def __init__(
        self,
        project_args: ProjectSettings,
        metrics: Metrics,
        well_indecies: WellIndecies,
        imputation_type: str = "remote_sensing",
    ):
        self.project_args = (project_args,)
        self.metrics = (Metrics,)
        self.well_indecies = (WellIndecies,)
        self.imputation_type = (imputation_type,)
        assert self.imputation_type in ["remote_sensing", "iterative"]


def get_dummy_variables(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Create dummy variables for month and year
    """
    dumbies_df = pd.get_dummies(index.month_name())
    dumbies_df = dumbies_df.set_index(index)
    dumbies_df = dumbies_df.astype(int)
    return dumbies_df


def generate_priors(
    y_data: pd.Series,
    indecies: WellIndecies,
    regression_percentages: list = [0.10, 0.15, 0.25, 0.5, 1.0],
    regression_intercept_percentage: float = 0.10,
    windows: list = [18, 24, 36, 60],
) -> pd.DataFrame:
    center = interpolate_center(
        series=y_data.dropna(),
        imputation_index=indecies.center_index,
    )
    left = extrapolate_left(
        series=y_data,
        left_imputation_index=indecies.left_index,
        regression_percentages=regression_percentages,
        regression_intercept_percentage=regression_intercept_percentage,
    )
    right = extrapolate_right(
        series=y_data,
        right_imputation_index=indecies.right_index,
        regression_percentages=regression_percentages,
        regression_intercept_percentage=regression_intercept_percentage,
    )
    merge_df_list = [
        prior_section
        for prior_section in [left, right, center]
        if prior_section is not None
    ]
    prior = merge_series(series_list=merge_df_list)
    prior_features = extract_priors(prior, windows)
    return prior, prior_features


def interpolate_center(
    series: pd.Series, imputation_index: pd.DatetimeIndex
) -> pd.Series:
    center = utils.interpolate_pchip(series=series, interp_index=imputation_index)
    return center


def extrapolate_left(
    series: pd.Series,
    left_imputation_index: pd.DatetimeIndex,
    regression_percentages: list = [0.10, 0.15, 0.25, 0.5, 1.0],
    regression_intercept_percentage: float = 0.10,
) -> pd.Series:
    # check if there are measurements before the imputation index
    # if there are, we don't need to extrapolate the values will be derived using pchip
    if left_imputation_index is not None:
        series_no_nan = series.dropna()
        slopes = []
        for percentage in regression_percentages:
            series_subset_mask = generate_percentage_idecies(
                index=series_no_nan.index,
                percentage=percentage,
            )
            slope, _, _, _ = interpolate_theil_slope(series_no_nan.loc[series_subset_mask])
            slopes.append(slope)
        slope_average = np.mean(slopes)
        intercept = np.mean(
            series.loc[
                generate_percentage_idecies(
                    index=series_no_nan.index,
                    percentage=regression_intercept_percentage,
                )
            ]
        )
        series = pd.Series(index=left_imputation_index)
        series_extrapolation = linear_extrapolate(
            series=series, 
            slope=slope_average, 
            intercept=intercept,
        )
        series_extrapolation = reverse_series(series_extrapolation)
    else:
        series_extrapolation = None
    return series_extrapolation


def extrapolate_right(
    series: pd.Series,
    right_imputation_index: pd.DatetimeIndex,
    regression_percentages: list = [0.10, 0.15, 0.25, 0.5, 1.0],
    regression_intercept_percentage: float = 0.10,
) -> pd.Series:
    # check if there are measurements after the imputation index
    # if there are, we don't need to extrapolate the values will be derived using pchip
    if right_imputation_index is not None:
        series_no_nan = series.dropna()
        series_no_nan = reverse_series(series_no_nan)
        slopes = []
        for percentage in regression_percentages:
            series_subset_mask = generate_percentage_idecies(
                index=series_no_nan.index,
                percentage=percentage,
            )
            slope, _, _, _ = interpolate_theil_slope(series_no_nan.loc[series_subset_mask])
            slopes.append(slope)
        slope_average = np.mean(slopes)
        intercept = np.mean(
            series.loc[
                generate_percentage_idecies(
                    index=series_no_nan.index,
                    percentage=regression_intercept_percentage,
                )
            ]
        )
        series = pd.Series(index=right_imputation_index)
        series_extrapolation = linear_extrapolate(
            series=series,
            slope=slope_average,
            intercept=intercept,
            )
    else:
        series_extrapolation = None
    return series_extrapolation


def reverse_series(series: pd.Series) -> pd.Series:
    return series[::-1]


def check_date_greater_than(
    datetime_index1: pd.DatetimeIndex, datetime_index2: pd.DatetimeIndex
) -> bool:
    return datetime_index1 > datetime_index2


def interpolate_theil_slope(
    series: pd.Series,
) -> float:
    slope, intercept, low_slope, high_slope = theilslopes(
        series.values, series.index.to_julian_date(), 0.95
    )
    return slope, intercept, low_slope, high_slope


def generate_percentage_idecies(
    index: pd.DatetimeIndex, percentage: float
) -> pd.DatetimeIndex:
    return index[: int(len(index) * percentage)]


def linear_extrapolate(
    series: pd.Series,
    slope: float,
    intercept: float,
) -> pd.Series:
    extrapolation_index_int = series.index.to_julian_date()
    extrapolation = (
        slope * (extrapolation_index_int - extrapolation_index_int[0]) + intercept
    )
    extrapolation = pd.Series(extrapolation, index=series.index)
    return extrapolation


def merge_series(series_list: list) -> pd.Series:
    series_list = [series for series in series_list if series is not None]
    series = pd.concat(series_list, axis=0)
    return series.sort_index()


def concat_dataframes(
    df1: pd.DataFrame, df2: pd.DataFrame, method: str = "outer", axis: int = 1
):
    return pd.concat([df1, df2], join=method, axis=1)


def extract_priors(series: pd.Series, windows: list) -> pd.DataFrame:
    rw_dict = dict()
    for _, window in enumerate(windows):
        key = str(window) + "m_rw"
        rw = series.rolling(window, center=True, min_periods=1).mean()
        rw_dict[key] = rw
    rw = pd.DataFrame.from_dict(rw_dict)
    return rw


def get_nearest_data_index(
    query: pd.DataFrame,
    source: pd.DataFrame,
    k: int = 1,
) -> list:
    query = query.apply(pd.to_numeric, errors="coerce").astype(float)
    source = source.apply(pd.to_numeric, errors="coerce").astype(float)
    # assess that the number of columns in the query and source are the same
    assert query.shape[1] == source.shape[1]
    dist = pd.DataFrame(
        cdist(query, source, metric="euclidean"), columns=source.index
    ).T
    dist = dist.sort_values(by=0)
    nearest_data = source.loc[dist.index[:k]]
    return nearest_data.index.tolist()


def get_data_from_dataframe(
    cells: list,
    data_df: pd.DataFrame,
) -> pd.DataFrame:
    return data_df.loc[cells]


def remote_sensing_data_selection(
    data_dict: dict,
    location_key: str,
    location_query: pd.DataFrame,
    k=1,
) -> pd.DataFrame:
    cell_names = get_nearest_data_index(
        query=location_query, source=data_dict[location_key], k=k
    )
    data = data_dict[cell_names[0]]
    return data


def dataframe_split(
    dataframe: pd.DataFrame,
    split_column: str,
) -> Union[pd.DataFrame, pd.DataFrame]:
    y = dataframe[split_column].to_frame()
    x = dataframe.drop(columns=split_column)
    return y, x


def dataframe_join(
    df1: pd.DataFrame, df2: pd.DataFrame, method: str = "outer", axis: int = 1
) -> pd.DataFrame:
    return pd.concat([df1, df2], join=method, axis=1)


def scaler_pipeline(
    x: pd.DataFrame,
    scaler_object: object,
    features_to_pass: List[str],
    train: bool = False,
) -> Union[pd.DataFrame, object]:
    if train == True:
        x_scale = x.drop(features_to_pass, axis=1)
        x_partial = scaler_object.fit_transform(x_scale)
        x_partial = pd.DataFrame(
            x_partial, index=x_scale.index, columns=x_scale.columns
        )
        x = dataframe_join(x_partial, x[features_to_pass], method="inner")
        return [x, scaler_object]

    else:
        x_scale = x.drop(features_to_pass, axis=1)
        x_temp = scaler_object.transform(x_scale)
        x_temp = pd.DataFrame(x_temp, index=x_scale.index, columns=x_scale.columns)
        x = dataframe_join(x_temp, x[features_to_pass], method="inner")
        return x
