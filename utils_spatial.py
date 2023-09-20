import utils
import logging
import math
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as dt
from netCDF4 import Variable
from typing import Tuple

logging.basicConfig(
    filename="storage_change.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)


def create_spatial_interpolation_netcdf():
    pass


class StorageChangeCalculator:
    def __init__(
        self,
        units: str = "English",
        storage_coefficient: float = 0.2,
        anisotropic: str = "x",
    ):
        """
        Initialize the StorageChangeCalculator.

        :param raster: The netCDF4 dataset containing raster data.
        :type raster: nc.Dataset
        :param units: The system of measurement for imputed data. Choose between "English" and "Metric".
        :type units: str, optional
        :param storage_coefficient: The storage coefficient of the aquifer.
        :type storage_coefficient: float, optional
        """

        self.units = units
        self.storage_coefficient = storage_coefficient
        self.anisotropic = anisotropic
        self.degrees_to_radians = math.pi / 180

        if units == "English":
            self.unit_coeff = 1
            self.area_coeff = 43560  # 1 acre = 43560 ft^2
            self.vol_coeff = 1  # 1 acre-ft = 1 acre * 1 ft
            self.conversion_factor = 5280  # 1 mile = 5280 ft
            self.conversion_factor2 = 10**6  # 1 million = 1e6
            self.radius_earth = (
                3958.8 * self.conversion_factor
            )  # radius of earth is 3958.8 miles
            self.calculation_unit = "feet"
            self.area_unit = "million acres"
            self.volume_unit = "million acre-feet"

        # TODO: Fix metric units
        # elif units == "Metric":
        #     self.unit_coeff = 0.3048  # 1 ft = 0.3048 m
        #     self.area_coeff = 1  # 1 km^2 = 1 km^2
        #     self.vol_coeff = 1000  # 1 km^3 = 1000 m^3
        #     self.conversion_factor = 1000  # 1 km = 1000 m
        #     self.conversion_factor2 = 1  # 1 = 1 (no conversion)
        #     self.radius_earth = (
        #         6378.1 * self.conversion_factor
        #     )  # radius of earth is 6371 km
        #     self.calculation_unit = "meters"
        #     self.area_unit = "km^2"
        #     self.volume_unit = "km^3"

        else:
            logging.error(
                "Invalid units. Please choose between 'English' and 'Metric'."
            )
            raise ValueError(
                "Invalid units. Please choose between 'English' and 'Metric'."
            )

        # Check anisotropy that the cells have the same resolution in one direction
        assert self.anisotropic in [
            "x",
            "y",
        ], "Invalid anisotropy. Please choose between 'x' and 'y'."
        logging.info(f"Cells have same resolution in self.{anisotropic} direction.")

        # Check storage coefficient
        assert (
            self.storage_coefficient >= 0 and self.storage_coefficient <= 1
        ), "The storage coefficient must be between 0 and 1."
        logging.info(
            "The storage coefficient is set to %.2f.", self.storage_coefficient
        )

        return

    def calulate_storage_curve(
        self,
        raster: nc.Dataset,
        date_range_filter: Tuple[str, str] = None,
    ) -> pd.Series:
        """
        Calculate and plot the storage change curve. Over specified time period.

        This function calculates the storage change curve for a given aquifer based on
        imputed raster data. It computes the area of the aquifer, the drawdown at each
        time step, and plots the storage depletion curve. Part of this function is
        based on the code from the NOAA, where we need to convert from degrees to
        radians and calculate the area of the aquifer.

        :param raster: The netCDF4 dataset containing raster data.
        :type raster: nc.Dataset
        :param date_start: The start date of the time period.
        :type date_start: str
        :param date_end: The end date of the time period.
        :type date_end: str

        :return: None
        :rtype: None
        """
        logging.info("Calculating raster area...")
        area = self.calculate_wgs84_area(raster)
        logging.info("Area calculated.")
        logging.info(
            f"The area of the aquifer is: {round(area / self.conversion_factor2 / self.area_coeff, 2)} {self.area_unit}"
        )

        logging.info("Calculating monthly deltas...")
        delta_h = self.calculate_monthly_deltas(raster)
        logging.info("Monthly deltas calculated.")

        logging.info("Calculating storage change...")
        storage_change = self.calculate_volume(delta_h, area, self.storage_coefficient)
        logging.info("Storage change calculated.")

        if date_range_filter:
            logging.info("Filtering storage change curve...")
            storage_change = self.filter_timeseries(storage_change, date_range_filter)
            logging.info("Storage change curve filtered.")

        storage_change = self.unit_conversion(storage_change)
        logging.info(
            f"Final drawdown calculated: {round(storage_change[-1], 2)} {self.volume_unit}"
        )

        return storage_change

    def calculate_monthly_deltas(
        self,
        raster: nc.Dataset,
    ) -> pd.Series:
        """
        Calculate a storage change curve metric over a specified area and time range.

        This function calculates the storage change curve metric by comparing
        the water table elevation (WTE) at each time step with the WTE at the
        initial time step and averaging the drawdown across the entire aquifer
        for the specified date range.

        :param raster: The netCDF4 dataset containing raster data.
        :type raster: nc.Dataset
        :param area: The area of the aquifer.
        :type area: float
        :param date_start: The start date of the time period.
        :type date_start: str
        :param date_end: The end date of the time period.
        :type date_end: str
        :return: The storage change curve metric.
        :rtype: pd.Series
        """

        # Check that the raster has a time dimension
        self.check_time_dimension(raster)

        # Extract time index and validate date range
        datetime_index = self.extract_and_validate_time_index(
            raster,
        )

        # Extract raster values and calculate differences
        raster_difference = self.calculate_raster_differences(raster)

        # Create a pandas series of the raster differences
        delta_h = pd.Series(raster_difference, index=datetime_index)

        return delta_h

    def check_time_dimension(self, raster: nc.Dataset):
        if "time" not in raster.dimensions:
            logging.error("No time dimension found.")
            raise ValueError("No time dimension found.")

    def extract_and_validate_time_index(
        self,
        raster: nc.Dataset,
    ) -> pd.DatetimeIndex:
        """
        Extract the time index from a netCDF4 time variable.

        :param time_variable: A netCDF4 time variable.
        :type time_variable: Variable
        :param datetime_units: The units of the time variable. Must be in the format "%Y-%m-%d".
        :type datetime_units: str
        :return: A pandas DatetimeIndex. The time index of the netCDF4 time variable.
        :rtype: pd.DatetimeIndex
        """
        datetime_units = raster["time"].units
        cft_timestamps = nc.num2date(raster["time"][:], datetime_units)
        datetime_objects = [
            pd.Timestamp(dt.strftime("%Y-%m-%d %H:%M:%S")) for dt in cft_timestamps
        ]
        datetime_index = pd.DatetimeIndex(datetime_objects)
        return datetime_index

    def calculate_raster_differences(self, raster: nc.Dataset):
        raster_values = raster["tsvalue"][:]
        raster_difference = raster_values[:] - raster_values[0]
        raster_difference = np.nanmean(raster_difference, axis=(1, 2))
        return raster_difference

    def filter_timeseries(
        self, timeseries: pd.Series, date_range_filter: Tuple[str, str]
    ):
        filter_start = pd.Timestamp(date_range_filter[0])
        filter_end = pd.Timestamp(date_range_filter[1])

        assert filter_start <= filter_end, "The start date must be before the end date."
        assert (
            filter_start >= timeseries.index[0]
        ), "The start date is not within the time range of the raster."
        assert (
            filter_end <= timeseries.index[-1]
        ), "The end date is not within the time range of the raster."

        # Filter the timeseries
        filtered_data = timeseries[
            (timeseries.index >= filter_start) & (timeseries.index <= filter_end)
        ]

        # Reset the index to start at 0
        filtered_data = filtered_data - filtered_data[0]
        return filtered_data

    def calculate_wgs84_area(
        self,
        raster: nc.Dataset,
    ):
        """
        Calculate the area of the aquifer.

        Loop through every cell in the raster and calculate the area of the aquifer
        based on the cell size and the latitude of the cell. This function is based
        on the code from the NOAA, where we need to convert from degrees to radians
        and calculate the area of the aquifer.

        Area calculated based on the equation found here:
        https://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2004/msg00023.html
            A = 2 * pi * R * h
            h = R * (1 - sin(lat))
            A = 2 * pi * R^2 * (1 - sin(lat))
        Therefore the area between two lines of latitude is:
        A = 2 * pi * R^2 * |(lat1)-(lat2)|
        Converting from degrees to radians turns the equation into:
        A = 2*pi*R^2 * |sin(lat1)-sin(lat2)| * |lon1-lon2|/360
        A = (pi/180) *L * R^2 * |sin(lat1)-sin(lat2)| * |lon1-lon2|
        A = arc_length * R^2 * |sin(lat1)-sin(lat2)| * |lon1-lon2|

        :param raster: The netCDF4 dataset containing raster data.
        :type raster: nc.Dataset
        :param y_resolution: The resolution of the raster in the y-direction.
        :type y_resolution: float
        :param x_resolution: The resolution of the raster in the x-direction.
        :type x_resolution: float

        :return: The area of the aquifer.
        :rtype: float
        """

        # TODO: Works for x, but not y. Need to fix.

        # this assumes all cells will be the same size in one dimension currently x
        x_resolution = abs(round(raster["lon"][0] - raster["lon"][1], 7))  # degrees
        y_resolution = abs(round(raster["lat"][0] - raster["lat"][1], 7))  # degrees

        area = 0

        # Loop through each y row
        for y in range(raster.dimensions["lat"].size):
            # Define the upper and lower bounds of the row
            mylatmax = math.radians(raster["lat"][y] + (y_resolution / 2))
            mylatmin = math.radians(raster["lat"][y] - (y_resolution / 2))

            # Count how many cells in each row are in aquifer (i.e. and, therefore, not nan)
            num_cells = (~np.isnan(raster["tsvalue"][0, y, :])).sum()

            # Calculate the area of the row
            area += (
                (self.degrees_to_radians * x_resolution * num_cells)
                * (self.radius_earth) ** 2
                * abs((math.sin(mylatmin) - math.sin(mylatmax)))
            )

        return area

    def calculate_volume(
        self, timeseries: pd.Series, area: float, storage_coefficient: float
    ):
        """
        Calculate the storage change curve.

        :param timeseries: The timeseries of the storage change curve.
        :type timeseries: pd.Series
        :param area: The area of the aquifer.
        :type area: float
        :param storage_coefficient: The storage coefficient of the aquifer.
        :type storage_coefficient: float
        :return: The storage change curve.
        :rtype: pd.Series
        """

        storage_change = timeseries * area * storage_coefficient
        return storage_change

    def unit_conversion(self, storage_change: pd.Series):
        """
        Convert the units of the storage change curve.

        :param storage_change: The storage change curve.
        :type storage_change: pd.Series
        :return: The storage change curve with converted units.
        :rtype: pd.Series
        """

        storage_change = (
            storage_change / self.conversion_factor2 / self.area_coeff / self.vol_coeff
        )

        return storage_change
