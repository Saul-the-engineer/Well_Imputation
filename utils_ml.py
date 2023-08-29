import numpy as np
import pandas as pd
import os
import logging


class metrics:
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


class well_indecies:
    def __init__(
        self,
        well_id: str,
        series: pd.Series,
        location: pd.DataFrame,
        imputation_range: pd.DatetimeIndex,
    ):
        self.well_id = str(well_id)
        self.raw_series = series.astype(float)
        self.location = location
        self.imputation_range = imputation_range
        self.data = series.loc[imputation_range]
        self.data_available = self.data.dropna()
        self.data_missing = self.data[self.data.isna()]
        self.interpolation_index = pd.date_range(
            start=self.data_available.index[0],
            end=self.data_available.index[-1],
            freq="MS",
        )


def log_error(well_id: str, error: str):
    logging.error(f"Error processing well {well_id}: {str(error)}")
